"""memory/task_store.py — ACID SQLite 存储（JSON-first 永久稳定模式）。

设计原则
--------
- schema 永远不 ALTER TABLE：所有可扩展字段存入 `data TEXT` (JSON 列)
- 旧列式 schema 一次性迁移：读取旧行 → 重建表 → 重新写入
- WAL 模式：并发读不阻塞写
- Task / Failure 是轻量 dataclass，不依赖 ORM
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)

# ── 永久稳定 DDL ────────────────────────────────────────────────────────────
_CREATE_TASKS = """
CREATE TABLE IF NOT EXISTS tasks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT    NOT NULL DEFAULT '',
    status     TEXT    NOT NULL DEFAULT 'pending',
    priority   TEXT    NOT NULL DEFAULT 'normal',
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    data       TEXT    NOT NULL DEFAULT '{}'
);
"""

_CREATE_FAILURES = """
CREATE TABLE IF NOT EXISTS failures (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT    NOT NULL,
    dismissed  INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    data       TEXT    NOT NULL DEFAULT '{}'
);
"""

_CREATE_FACTS = """
CREATE TABLE IF NOT EXISTS facts (
    key        TEXT PRIMARY KEY,
    value      TEXT    NOT NULL DEFAULT '',
    scope      TEXT    NOT NULL DEFAULT 'general',
    updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    run_at      TEXT    NOT NULL,            -- ISO8601 UTC，string compare 可排序
    repeat_secs INTEGER NOT NULL DEFAULT 0, -- 0 = 一次性；>0 = 重复间隔秒数
    status      TEXT    NOT NULL DEFAULT 'pending',  -- pending | done | cancelled
    payload     TEXT    NOT NULL DEFAULT '{}',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_signals_pending
    ON signals(run_at) WHERE status='pending';
"""

_CREATE_CHAT = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT    NOT NULL,            -- 'user' | 'assistant'
    content    TEXT    NOT NULL,
    session_id TEXT    NOT NULL DEFAULT '',
    status     TEXT    NOT NULL DEFAULT 'pending',  -- pending | processed
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chat_pending
    ON chat_messages(status, id) WHERE role='user';
"""

# ── 性能索引（幂等，IF NOT EXISTS，对存量 DB 同样有效）────────────────────────
# 分析依据：
#   tasks.get_active()   → 每 tick 执行一次 WHERE status IN (...) ORDER BY priority → 无索引=全表扫
#   tasks.list_tasks()   → WHERE status=? 无索引=全表扫
#   tasks.enqueue_if_absent() → WHERE title=? AND status NOT IN (...) 无索引=全表扫
#   failures.list_failures()  → WHERE dismissed=0 ORDER BY id DESC 无索引=全表扫
#   failures.count_failures_by_kind() → WHERE kind=? AND dismissed=0 无索引=全表扫
_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_tasks_status
    ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_title
    ON tasks(title);
CREATE INDEX IF NOT EXISTS idx_failures_active
    ON failures(dismissed, id DESC);
CREATE INDEX IF NOT EXISTS idx_failures_kind
    ON failures(kind, dismissed);
"""


# ── 数据对象 ────────────────────────────────────────────────────────────────

@dataclass
class Task:
    id: int
    title: str
    status: str
    priority: str
    created_at: str
    # 核心 data 字段（data JSON 的常用键）
    goal: str = ""
    source: str = "external"
    next_step: str = ""
    # 其余 data 键，动态扩展无需代码变动
    extras: dict[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def from_row(cls, row: Any) -> "Task":
        """row = (id, title, status, priority, created_at, data_json)"""
        rid, title, status, priority, created_at, data_raw = row
        try:
            data: dict[str, Any] = json.loads(data_raw or "{}")
        except Exception:
            data = {}
        goal = data.pop("goal", "")
        source = data.pop("source", "external")
        next_step = data.pop("next_step", "")
        return cls(
            id=rid,
            title=title,
            status=status,
            priority=priority,
            created_at=created_at,
            goal=goal,
            source=source,
            next_step=next_step,
            extras=data,
        )

    def to_data_json(self) -> str:
        d = {"goal": self.goal, "source": self.source, "next_step": self.next_step}
        d.update(self.extras)
        return json.dumps(d, ensure_ascii=False)


@dataclass
class Failure:
    id: int
    kind: str
    dismissed: bool
    created_at: str
    # 核心 data 字段
    summary: str = ""
    context: str = ""
    task_id: str = ""
    extras: dict[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def from_row(cls, row: Any) -> "Failure":
        """row = (id, kind, dismissed, created_at, data_json)"""
        rid, kind, dismissed, created_at, data_raw = row
        try:
            data: dict[str, Any] = json.loads(data_raw or "{}")
        except Exception:
            data = {}
        summary = data.pop("summary", "")
        context = data.pop("context", "")
        task_id = data.pop("task_id", "")
        return cls(
            id=rid,
            kind=kind,
            dismissed=bool(dismissed),
            created_at=created_at,
            summary=summary,
            context=context,
            task_id=task_id,
            extras=data,
        )


# ── 存储层 ──────────────────────────────────────────────────────────────────

class TaskStore:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._db_opt: Optional[aiosqlite.Connection] = None

    @property
    def _db(self) -> aiosqlite.Connection:
        assert self._db_opt is not None, "TaskStore not open — call open() first"
        return self._db_opt

    async def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db_opt = await aiosqlite.connect(str(self._path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        # 检测旧 schema 并迁移
        await self._migrate()
        # 建表（幂等）+ 补充性能索引（IF NOT EXISTS，对存量 DB 同样生效）
        await self._db.executescript(
            _CREATE_TASKS + _CREATE_FAILURES + _CREATE_FACTS + _CREATE_SIGNALS + _CREATE_CHAT + _CREATE_INDEXES
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db_opt:
            await self._db_opt.close()
            self._db_opt = None

    # ── 一次性迁移（旧列式 → JSON-first）────────────────────────────────

    async def _migrate(self) -> None:
        """检测旧列式 schema，迁移至 JSON-first。幂等：已含 data 列则跳过。"""
        db = self._db_opt
        assert db is not None

        # 检测 tasks 表是否存在 data 列
        async with db.execute(
            "SELECT COUNT(*) FROM pragma_table_info('tasks') WHERE name='data'"
        ) as cur:
            row = await cur.fetchone()
            if row and row[0] > 0:
                # tasks 已是 JSON-first；再确认 failures 有 dismissed 列
                async with db.execute(
                    "SELECT COUNT(*) FROM pragma_table_info('failures') WHERE name='dismissed'"
                ) as cur2:
                    row2 = await cur2.fetchone()
                    if not (row2 and row2[0] > 0):
                        # 过渡态：补充 dismissed 列（一次性，纯追加）
                        await db.execute(
                            "ALTER TABLE failures ADD COLUMN dismissed INTEGER NOT NULL DEFAULT 0"
                        )
                        await db.commit()
                return  # 已是 JSON-first，无需迁移

        # 旧 tasks 表存在？
        async with db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tasks'"
        ) as cur:
            row = await cur.fetchone()
            if not (row and row[0] > 0):
                return  # 全新数据库，让 open() 正常建表

        logger.info("[task_store] 检测到旧列式 schema，开始一次性迁移 → JSON-first")

        # 读取旧数据
        old_tasks: list[dict[str, Any]] = []
        try:
            async with db.execute(
                "SELECT id, title, goal, priority, status, source, next_step, created_at FROM tasks"
            ) as cur:
                async for r in cur:
                    old_tasks.append({
                        "id": r[0], "title": r[1], "goal": r[2] or "",
                        "priority": r[3] or "normal", "status": r[4] or "pending",
                        "source": r[5] or "external", "next_step": r[6] or "",
                        "created_at": r[7] or "",
                    })
        except Exception:
            pass

        old_failures: list[dict[str, Any]] = []
        try:
            async with db.execute(
                "SELECT id, kind, summary, context, task_id, created_at FROM failures"
            ) as cur:
                async for r in cur:
                    old_failures.append({
                        "id": r[0], "kind": r[1], "summary": r[2] or "",
                        "context": r[3] or "", "task_id": r[4] or "",
                        "created_at": r[5] or "",
                    })
        except Exception:
            pass

        # 删除旧表，重建新表
        await db.executescript("""
            DROP TABLE IF EXISTS tasks;
            DROP TABLE IF EXISTS failures;
        """)
        await db.executescript(_CREATE_TASKS + _CREATE_FAILURES)

        # 回填旧数据（保持原 id，使用 INSERT OR REPLACE）
        for t in old_tasks:
            data = json.dumps({
                "goal": t["goal"], "source": t["source"], "next_step": t["next_step"]
            }, ensure_ascii=False)
            await db.execute(
                "INSERT OR REPLACE INTO tasks (id, title, status, priority, created_at, data) "
                "VALUES (?,?,?,?,?,?)",
                (t["id"], t["title"], t["status"], t["priority"], t["created_at"], data),
            )

        for f in old_failures:
            data = json.dumps({
                "summary": f["summary"], "context": f["context"], "task_id": f["task_id"]
            }, ensure_ascii=False)
            await db.execute(
                "INSERT OR REPLACE INTO failures (id, kind, dismissed, created_at, data) "
                "VALUES (?,?,0,?,?)",
                (f["id"], f["kind"], f["created_at"], data),
            )

        await db.commit()
        logger.info("[task_store] 迁移完成：%d 任务, %d 失败记录", len(old_tasks), len(old_failures))

    # ── 任务操作 ─────────────────────────────────────────────────────────

    async def add_task(
        self,
        title: str,
        goal: str = "",
        priority: str = "normal",
        source: str = "external",
    ) -> int:
        data = json.dumps({"goal": goal, "source": source, "next_step": ""}, ensure_ascii=False)
        async with self._db.execute(
            "INSERT INTO tasks (title, priority, data) VALUES (?,?,?)",
            (title.strip(), priority, data),
        ) as cur:
            task_id: int = cur.lastrowid or 0
        await self._db.commit()
        return task_id

    async def get_task_by_id(self, task_id: int) -> Optional[Task]:
        async with self._db.execute(
            "SELECT id, title, status, priority, created_at, data FROM tasks WHERE id=?",
            (task_id,),
        ) as cur:
            row = await cur.fetchone()
        return Task.from_row(row) if row else None

    async def get_active(self) -> Optional[Task]:
        """返回优先级最高的待处理/进行中任务（pending → in_progress，priority 高优先）。"""
        async with self._db.execute(
            """SELECT id, title, status, priority, created_at, data
               FROM tasks
               WHERE status IN ('pending','in_progress')
               ORDER BY
                 CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                               WHEN 'normal' THEN 2 ELSE 3 END,
                 id
               LIMIT 1""",
        ) as cur:
            row = await cur.fetchone()
        return Task.from_row(row) if row else None

    async def list_tasks(
        self, status: Optional[str] = None, limit: int = 50
    ) -> list[Task]:
        if status:
            sql = ("SELECT id, title, status, priority, created_at, data "
                   "FROM tasks WHERE status=? ORDER BY id LIMIT ?")
            args = (status, limit)
        else:
            sql = ("SELECT id, title, status, priority, created_at, data "
                   "FROM tasks ORDER BY id LIMIT ?")
            args = (limit,)
        async with self._db.execute(sql, args) as cur:
            rows = await cur.fetchall()
        return [Task.from_row(r) for r in rows]

    async def update_status(
        self, task_id: int, status: str, next_step: str = ""
    ) -> None:
        """更新 status；同时将 next_step 写入 data JSON（最小化写入，不覆盖其他 data 字段）。"""
        task = await self.get_task_by_id(task_id)
        if not task:
            return
        task.status = status
        if next_step:
            task.next_step = next_step
        await self._db.execute(
            "UPDATE tasks SET status=?, data=? WHERE id=?",
            (status, task.to_data_json(), task_id),
        )
        await self._db.commit()

    async def update_task_data(self, task_id: int, extra_dict: dict[str, Any]) -> None:
        """将 extra_dict 合并进 data JSON（不覆盖 goal/source/next_step）。"""
        task = await self.get_task_by_id(task_id)
        if not task:
            return
        task.extras.update(extra_dict)
        await self._db.execute(
            "UPDATE tasks SET data=? WHERE id=?",
            (task.to_data_json(), task_id),
        )
        await self._db.commit()

    async def enqueue_if_absent(
        self,
        title: str,
        goal: str = "",
        priority: str = "normal",
        source: str = "internal",
    ) -> bool:
        """如果标题相同的未完成任务不存在，则创建。返回是否新建。"""
        title = title.strip()
        async with self._db.execute(
            "SELECT id FROM tasks WHERE title=? AND status NOT IN ('done','failed') LIMIT 1",
            (title,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            return False
        await self.add_task(title, goal=goal, priority=priority, source=source)
        return True

    # ── 失败记录 ─────────────────────────────────────────────────────────

    async def record_failure(
        self,
        kind: str,
        summary: str,
        context: str = "",
        task_id: str = "",
    ) -> None:
        data = json.dumps(
            {"summary": summary, "context": context, "task_id": task_id},
            ensure_ascii=False,
        )
        await self._db.execute(
            "INSERT INTO failures (kind, data) VALUES (?,?)", (kind, data)
        )
        await self._db.commit()

    async def list_failures(self, limit: int = 20) -> list[Failure]:
        async with self._db.execute(
            "SELECT id, kind, dismissed, created_at, data FROM failures "
            "WHERE dismissed=0 ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [Failure.from_row(r) for r in rows]

    async def list_failures_for_task(self, task_id: str, limit: int = 20) -> list[Failure]:
        async with self._db.execute(
            "SELECT id, kind, dismissed, created_at, data FROM failures "
            "WHERE (json_extract(data,'$.task_id')=? OR json_extract(data,'$.task_id')='') AND dismissed=0 "
            "ORDER BY id LIMIT ?",
            (task_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [Failure.from_row(r) for r in rows]

    async def count_failures_by_kind(self, kind: str) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) FROM failures WHERE kind=? AND dismissed=0", (kind,)
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else 0

    async def dismiss_failure(self, failure_id: int) -> None:
        await self._db.execute(
            "UPDATE failures SET dismissed=1 WHERE id=?", (failure_id,)
        )
        await self._db.commit()

    # ── Facts KV ─────────────────────────────────────────────────────────

    async def set_fact(self, key: str, value: str, scope: str = "general") -> None:
        await self._db.execute(
            "INSERT INTO facts (key, value, scope, updated_at) VALUES (?,?,?,datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "scope=excluded.scope, updated_at=excluded.updated_at",
            (key, value, scope),
        )
        await self._db.commit()

    async def get_fact(self, key: str) -> tuple[str, bool]:
        """返回 (value, found)。"""
        async with self._db.execute(
            "SELECT value FROM facts WHERE key=?", (key,)
        ) as cur:
            row = await cur.fetchone()
        if row:
            return row[0], True
        return "", False

    # ── 调度信号（cron 机制）──────────────────────────────────────────────

    async def add_signal(
        self,
        title: str,
        run_at: str,
        repeat_secs: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """添加一条调度信号。run_at 为 ISO8601 UTC 字符串，返回新记录 id。"""
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        async with self._db.execute(
            "INSERT INTO signals (title, run_at, repeat_secs, payload) VALUES (?,?,?,?)",
            (title, run_at, repeat_secs, payload_json),
        ) as cur:
            new_id = cur.lastrowid
        await self._db.commit()
        return new_id  # type: ignore[return-value]

    async def due_signals(self) -> list[dict[str, Any]]:
        """返回所有 run_at <= 当前 UTC 时间 且 status='pending' 的信号。"""
        rows: list[dict[str, Any]] = []
        async with self._db.execute(
            "SELECT id, title, run_at, repeat_secs, payload "
            "FROM signals WHERE status='pending' AND run_at <= datetime('now') "
            "ORDER BY run_at"
        ) as cur:
            async for row in cur:
                try:
                    payload = json.loads(row[4] or "{}")
                except Exception:
                    payload = {}
                rows.append({
                    "id": row[0],
                    "title": row[1],
                    "run_at": row[2],
                    "repeat_secs": row[3],
                    "payload": payload,
                })
        return rows

    async def ack_signal(self, signal_id: int) -> None:
        """确认信号已处理。一次性信号标记为 done；重复信号更新 run_at 到下次触发时间。"""
        async with self._db.execute(
            "SELECT repeat_secs, run_at FROM signals WHERE id=?", (signal_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return
        repeat_secs: Any = row[0]
        if repeat_secs and repeat_secs > 0:
            # 更新到下次触发时间（从当前 run_at + interval，防止漂移）
            await self._db.execute(
                "UPDATE signals SET run_at=datetime(run_at, ?||' seconds') WHERE id=?",
                (str(repeat_secs), signal_id),
            )
        else:
            await self._db.execute(
                "UPDATE signals SET status='done' WHERE id=?", (signal_id,)
            )
        await self._db.commit()

    async def list_signals(self, limit: int = 30, include_done: bool = False) -> list[dict[str, Any]]:
        """列出调度信号（默认只列 pending；include_done=True 则包含 done）。"""
        where = "" if include_done else "WHERE status='pending'"
        rows: list[dict[str, Any]] = []
        async with self._db.execute(
            f"SELECT id, title, run_at, repeat_secs, status, payload "
            f"FROM signals {where} ORDER BY run_at LIMIT ?",
            (limit,),
        ) as cur:
            async for row in cur:
                try:
                    payload = json.loads(row[5] or "{}")
                except Exception:
                    payload = {}
                rows.append({
                    "id": row[0],
                    "title": row[1],
                    "run_at": row[2],
                    "repeat_secs": row[3],
                    "status": row[4],
                    "payload": payload,
                })
        return rows

    async def cancel_signal(self, signal_id: int) -> None:
        """取消一条调度信号。"""
        await self._db.execute(
            "UPDATE signals SET status='cancelled' WHERE id=?", (signal_id,)
        )
        await self._db.commit()

    # ── 对话消息（chat IPC）────────────────────────────────────────────────

    async def add_chat_message(self, role: str, content: str, session_id: str = "") -> int:
        """写入一条对话消息（role='user'|'assistant'）。"""
        async with self._db.execute(
            "INSERT INTO chat_messages(role, content, session_id) VALUES (?,?,?)",
            (role, content, session_id),
        ) as cur:
            row_id: int = cur.lastrowid or 0
        await self._db.commit()
        return row_id

    async def has_pending_chat_message(self) -> bool:
        """非破坏性检查：是否有待处理的 user 消息（仅用于早唤醒轮询，不消费）。"""
        async with self._db.execute(
            "SELECT 1 FROM chat_messages WHERE role='user' AND status='pending' LIMIT 1"
        ) as cur:
            return await cur.fetchone() is not None

    async def pop_pending_chat_message(self) -> Optional[dict[str, Any]]:
        """原子获取并标记最早一条待处理 user 消息（无则返回 None）。"""
        async with self._db.execute(
            "SELECT id, content, session_id FROM chat_messages "
            "WHERE role='user' AND status='pending' ORDER BY id LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        mid, content, session_id = row
        await self._db.execute(
            "UPDATE chat_messages SET status='processed' WHERE id=?", (mid,)
        )
        await self._db.commit()
        return {"id": mid, "content": content, "session_id": session_id}

    async def get_chat_messages_since(self, since_id: int = 0, session_id: str = "") -> list[dict[str, Any]]:
        """返回 id > since_id 的所有消息（可选按 session_id 过滤）。"""
        if session_id:
            sql = (
                "SELECT id, role, content, created_at FROM chat_messages "
                "WHERE id > ? AND session_id = ? ORDER BY id"
            )
            params: tuple[Any, ...] = (since_id, session_id)
        else:
            sql = (
                "SELECT id, role, content, created_at FROM chat_messages "
                "WHERE id > ? ORDER BY id"
            )
            params = (since_id,)
        async with self._db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [{"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]} for r in rows]
