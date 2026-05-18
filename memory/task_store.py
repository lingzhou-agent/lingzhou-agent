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

from store.memory import (
    ChatMessageStore,
    FactStore,
    FailureStore,
    MetaReflectionStore,
    RunStore,
    SignalStore,
    TaskStateStore,
)

logger = logging.getLogger(__name__)

_TASK_CORE_DATA_KEYS = frozenset({
    "goal",
    "source",
    "next_step",
    "chain_id",
    "parent_task_id",
    "current_step",
    "wait_kind",
    "wait_key",
    "state_json",
    "wait_json",
    "result_json",
    "async_job_id",
    "model_tier",
})

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

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      INTEGER NOT NULL DEFAULT 0,
    run_type     TEXT    NOT NULL DEFAULT 'tool_chain',
    worker_type  TEXT    NOT NULL DEFAULT 'tool-chain-worker',
    status       TEXT    NOT NULL DEFAULT 'pending',
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    started_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT    NOT NULL DEFAULT '',
    data         TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_runs_task_status
    ON runs(task_id, status, id DESC);
CREATE INDEX IF NOT EXISTS idx_runs_status
    ON runs(status, id DESC);
"""

_CREATE_META_REFLECTIONS = """
CREATE TABLE IF NOT EXISTS meta_reflections (
    id                TEXT PRIMARY KEY,
    target_kind       TEXT NOT NULL,
    trigger           TEXT NOT NULL,
    loop_level        TEXT NOT NULL,
    diagnosis         TEXT NOT NULL,
    proposal          TEXT NOT NULL,
    verification_plan TEXT NOT NULL DEFAULT '',
    decision          TEXT NOT NULL DEFAULT 'defer',
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    data              TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_meta_reflections_loop
    ON meta_reflections(loop_level, created_at DESC);
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
    chain_id: str = ""
    parent_task_id: str = ""
    current_step: str = ""
    wait_kind: str = ""
    wait_key: str = ""
    state_json: dict[str, Any] = field(default_factory=dict[str, Any])
    wait_json: dict[str, Any] = field(default_factory=dict[str, Any])
    result_json: dict[str, Any] = field(default_factory=dict[str, Any])
    async_job_id: str = ""
    model_tier: str = ""
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
        chain_id = data.pop("chain_id", "")
        parent_task_id = data.pop("parent_task_id", "")
        current_step = data.pop("current_step", "")
        wait_kind = data.pop("wait_kind", "")
        wait_key = data.pop("wait_key", "")
        state_json = data.pop("state_json", {}) or {}
        wait_json = data.pop("wait_json", {}) or {}
        result_json = data.pop("result_json", {}) or {}
        async_job_id = data.pop("async_job_id", "")
        model_tier = data.pop("model_tier", "")
        return cls(
            id=rid,
            title=title,
            status=status,
            priority=priority,
            created_at=created_at,
            goal=goal,
            source=source,
            next_step=next_step,
            chain_id=chain_id,
            parent_task_id=parent_task_id,
            current_step=current_step,
            wait_kind=wait_kind,
            wait_key=wait_key,
            state_json=state_json,
            wait_json=wait_json,
            result_json=result_json,
            async_job_id=async_job_id,
            model_tier=model_tier,
            extras=data,
        )

    def to_data_json(self) -> str:
        d = {
            "goal": self.goal,
            "source": self.source,
            "next_step": self.next_step,
            "chain_id": self.chain_id,
            "parent_task_id": self.parent_task_id,
            "current_step": self.current_step,
            "wait_kind": self.wait_kind,
            "wait_key": self.wait_key,
            "state_json": self.state_json,
            "wait_json": self.wait_json,
            "result_json": self.result_json,
            "async_job_id": self.async_job_id,
            "model_tier": self.model_tier,
        }
        d.update({k: v for k, v in self.extras.items() if k not in _TASK_CORE_DATA_KEYS})
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


@dataclass
class Run:
    id: int
    task_id: int
    run_type: str
    worker_type: str
    status: str
    created_at: str
    started_at: str = ""
    completed_at: str = ""
    input_json: dict[str, Any] = field(default_factory=dict[str, Any])
    output_json: dict[str, Any] = field(default_factory=dict[str, Any])
    log_text: str = ""
    error_text: str = ""
    tool_name: str = ""
    session_id: str = ""
    model_tier: str = ""
    progress: str = ""
    extras: dict[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def from_row(cls, row: Any) -> "Run":
        rid, task_id, run_type, worker_type, status, created_at, started_at, completed_at, data_raw = row
        try:
            data: dict[str, Any] = json.loads(data_raw or "{}")
        except Exception:
            data = {}
        input_json = data.pop("input_json", {}) or {}
        output_json = data.pop("output_json", {}) or {}
        log_text = data.pop("log_text", "")
        error_text = data.pop("error_text", "")
        tool_name = data.pop("tool_name", "")
        session_id = data.pop("session_id", "")
        model_tier = data.pop("model_tier", "")
        progress = data.pop("progress", "")
        return cls(
            id=rid,
            task_id=task_id,
            run_type=run_type,
            worker_type=worker_type,
            status=status,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            input_json=input_json,
            output_json=output_json,
            log_text=log_text,
            error_text=error_text,
            tool_name=tool_name,
            session_id=session_id,
            model_tier=model_tier,
            progress=progress,
            extras=data,
        )

    def to_data_json(self) -> str:
        data = {
            "input_json": self.input_json,
            "output_json": self.output_json,
            "log_text": self.log_text,
            "error_text": self.error_text,
            "tool_name": self.tool_name,
            "session_id": self.session_id,
            "model_tier": self.model_tier,
            "progress": self.progress,
        }
        data.update(self.extras)
        return json.dumps(data, ensure_ascii=False)


@dataclass
class MetaReflection:
    id: str
    target_kind: str
    trigger: str
    loop_level: str
    diagnosis: str
    proposal: str
    verification_plan: str
    decision: str
    created_at: str
    task_id: int = 0
    run_id: int = 0
    tool_name: str = ""
    extras: dict[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def from_row(cls, row: Any) -> "MetaReflection":
        rid, target_kind, trigger, loop_level, diagnosis, proposal, verification_plan, decision, created_at, data_raw = row
        try:
            data: dict[str, Any] = json.loads(data_raw or "{}")
        except Exception:
            data = {}
        task_id = int(data.pop("task_id", 0) or 0)
        run_id = int(data.pop("run_id", 0) or 0)
        tool_name = str(data.pop("tool_name", "") or "")
        return cls(
            id=str(rid),
            target_kind=target_kind,
            trigger=trigger,
            loop_level=loop_level,
            diagnosis=diagnosis,
            proposal=proposal,
            verification_plan=verification_plan,
            decision=decision,
            created_at=created_at,
            task_id=task_id,
            run_id=run_id,
            tool_name=tool_name,
            extras=data,
        )

    def to_data_json(self) -> str:
        data = {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "tool_name": self.tool_name,
        }
        data.update(self.extras)
        return json.dumps(data, ensure_ascii=False)


# ── 存储层 ──────────────────────────────────────────────────────────────────

class TaskStore:
    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path) if isinstance(db_path, str) else db_path
        self._db_opt: Optional[aiosqlite.Connection] = None
        self._chat = ChatMessageStore(lambda: self._db)
        self._facts = FactStore(lambda: self._db)
        self._failures = FailureStore(lambda: self._db)
        self._signals = SignalStore(lambda: self._db)
        self._tasks = TaskStateStore(lambda: self._db)
        self._runs = RunStore(lambda: self._db)
        self._meta_reflections = MetaReflectionStore(lambda: self._db)

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
            _CREATE_TASKS + _CREATE_FAILURES + _CREATE_FACTS + _CREATE_SIGNALS + _CREATE_CHAT + _CREATE_RUNS + _CREATE_META_REFLECTIONS + _CREATE_INDEXES
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
        *,
        status: str = "pending",
        next_step: str = "",
        chain_id: str = "",
        parent_task_id: str = "",
        current_step: str = "",
        wait_kind: str = "",
        wait_key: str = "",
        state_json: dict[str, Any] | None = None,
        wait_json: dict[str, Any] | None = None,
        result_json: dict[str, Any] | None = None,
        async_job_id: str = "",
        model_tier: str = "",
        extras: dict[str, Any] | None = None,
    ) -> int:
        return await self._tasks.add_task(
            title,
            goal,
            priority,
            source,
            status=status,
            next_step=next_step,
            chain_id=chain_id,
            parent_task_id=parent_task_id,
            current_step=current_step,
            wait_kind=wait_kind,
            wait_key=wait_key,
            state_json=state_json,
            wait_json=wait_json,
            result_json=result_json,
            async_job_id=async_job_id,
            model_tier=model_tier,
            extras=extras,
        )

    async def get_task_by_id(self, task_id: int) -> Optional[Task]:
        return await self._tasks.get_task_by_id(task_id)

    async def list_runnable_tasks(self, limit: int = 20) -> list[Task]:
        return await self._tasks.list_runnable_tasks(limit)

    async def get_active(self) -> Optional[Task]:
        return await self._tasks.get_active()

    async def list_tasks(
        self, status: Optional[str] = None, limit: int = 50
    ) -> list[Task]:
        return await self._tasks.list_tasks(status=status, limit=limit)

    async def update_status(
        self, task_id: int, status: str, next_step: str | None = None
    ) -> None:
        await self._tasks.update_status(task_id, status, next_step)

    async def mark_waiting(
        self,
        task_id: int,
        *,
        wait_kind: str,
        wait_key: str = "",
        wait_json: dict[str, Any] | None = None,
        current_step: str | None = None,
        next_step: str | None = None,
    ) -> None:
        await self._tasks.mark_waiting(
            task_id,
            wait_kind=wait_kind,
            wait_key=wait_key,
            wait_json=wait_json,
            current_step=current_step,
            next_step=next_step,
        )

    async def resume_task(
        self,
        task_id: int,
        *,
        status: str = "resumed",
        current_step: str | None = None,
        next_step: str | None = None,
        result_json: dict[str, Any] | None = None,
    ) -> None:
        await self._tasks.resume_task(
            task_id,
            status=status,
            current_step=current_step,
            next_step=next_step,
            result_json=result_json,
        )

    async def update_task_data(self, task_id: int, extra_dict: dict[str, Any]) -> None:
        await self._tasks.update_task_data(task_id, extra_dict)

    async def update_task_result(self, task_id: int, result_json: dict[str, Any]) -> None:
        await self._tasks.update_task_result(task_id, result_json)

    async def sync_task_progress(
        self,
        task_id: int,
        *,
        current_step: str | None = None,
        next_step: str | None = None,
    ) -> None:
        await self._tasks.sync_task_progress(
            task_id,
            current_step=current_step,
            next_step=next_step,
        )

    async def add_run(
        self,
        *,
        task_id: int = 0,
        run_type: str = "tool_chain",
        worker_type: str = "tool-chain-worker",
        status: str = "running",
        input_json: dict[str, Any] | None = None,
        output_json: dict[str, Any] | None = None,
        log_text: str = "",
        error_text: str = "",
        tool_name: str = "",
        session_id: str = "",
        model_tier: str = "",
        progress: str = "",
        extras: dict[str, Any] | None = None,
    ) -> int:
        return await self._runs.add_run(
            task_id=task_id,
            run_type=run_type,
            worker_type=worker_type,
            status=status,
            input_json=input_json,
            output_json=output_json,
            log_text=log_text,
            error_text=error_text,
            tool_name=tool_name,
            session_id=session_id,
            model_tier=model_tier,
            progress=progress,
            extras=extras,
        )

    async def get_run_by_id(self, run_id: int) -> Optional[Run]:
        return await self._runs.get_run_by_id(run_id)

    async def list_runs(
        self,
        *,
        task_id: int | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Run]:
        return await self._runs.list_runs(task_id=task_id, status=status, limit=limit)

    async def update_run(
        self,
        run_id: int,
        *,
        status: str | None = None,
        output_json: dict[str, Any] | None = None,
        log_text: str | None = None,
        error_text: str | None = None,
        session_id: str | None = None,
        model_tier: str | None = None,
        progress: str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        await self._runs.update_run(
            run_id,
            status=status,
            output_json=output_json,
            log_text=log_text,
            error_text=error_text,
            session_id=session_id,
            model_tier=model_tier,
            progress=progress,
            extras=extras,
        )

    async def add_meta_reflection(
        self,
        *,
        reflection_id: str,
        target_kind: str,
        trigger: str,
        loop_level: str,
        diagnosis: str,
        proposal: str,
        verification_plan: str = "",
        decision: str = "defer",
        task_id: int = 0,
        run_id: int = 0,
        tool_name: str = "",
        extras: dict[str, Any] | None = None,
    ) -> None:
        await self._meta_reflections.add_meta_reflection(
            reflection_id=reflection_id,
            target_kind=target_kind,
            trigger=trigger,
            loop_level=loop_level,
            diagnosis=diagnosis,
            proposal=proposal,
            verification_plan=verification_plan,
            decision=decision,
            task_id=task_id,
            run_id=run_id,
            tool_name=tool_name,
            extras=extras,
        )

    async def list_meta_reflections(self, limit: int = 20, loop_level: str | None = None) -> list[MetaReflection]:
        return await self._meta_reflections.list_meta_reflections(limit=limit, loop_level=loop_level)

    async def enqueue_if_absent(
        self,
        title: str,
        goal: str = "",
        priority: str = "normal",
        source: str = "internal",
    ) -> bool:
        return await self._tasks.enqueue_if_absent(title, goal=goal, priority=priority, source=source)

    # ── 失败记录 ─────────────────────────────────────────────────────────

    async def record_failure(
        self,
        kind: str,
        summary: str,
        context: str = "",
        task_id: str = "",
    ) -> None:
        await self._failures.record_failure(kind, summary, context, task_id)

    async def list_failures(self, limit: int = 20) -> list[Failure]:
        return await self._failures.list_failures(limit)

    async def list_failures_for_task(self, task_id: str, limit: int = 20) -> list[Failure]:
        return await self._failures.list_failures_for_task(task_id, limit)

    async def count_failures_by_kind(self, kind: str) -> int:
        return await self._failures.count_failures_by_kind(kind)

    async def dismiss_failure(self, failure_id: int) -> None:
        await self._failures.dismiss_failure(failure_id)

    # ── Facts KV ─────────────────────────────────────────────────────────

    async def set_fact(self, key: str, value: str, scope: str = "general") -> None:
        await self._facts.set_fact(key, value, scope)

    async def get_fact(self, key: str) -> tuple[str, bool]:
        """返回 (value, found)。"""
        return await self._facts.get_fact(key)

    async def list_facts(self, prefix: str = "", limit: int = 100) -> list[tuple[str, str]]:
        return await self._facts.list_facts(prefix, limit)

    async def delete_fact(self, key: str) -> None:
        await self._facts.delete_fact(key)

    # ── 调度信号（cron 机制）──────────────────────────────────────────────

    async def add_signal(
        self,
        title: str,
        run_at: str,
        repeat_secs: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """添加一条调度信号。run_at 为 ISO8601 UTC 字符串，返回新记录 id。"""
        return await self._signals.add_signal(title, run_at, repeat_secs, payload)

    async def due_signals(self) -> list[dict[str, Any]]:
        """返回所有 run_at <= 当前 UTC 时间 且 status='pending' 的信号。"""
        return await self._signals.due_signals()

    async def ack_signal(self, signal_id: int) -> None:
        """确认信号已处理。一次性信号标记为 done；重复信号更新 run_at 到下次触发时间。"""
        await self._signals.ack_signal(signal_id)

    async def list_signals(self, limit: int = 30, include_done: bool = False) -> list[dict[str, Any]]:
        """列出调度信号（默认只列 pending；include_done=True 则包含 done）。"""
        return await self._signals.list_signals(limit, include_done)

    async def get_signal(self, signal_id: int) -> dict[str, Any] | None:
        """按 id 查询单条调度信号；不存在或已删除时返回 None。"""
        return await self._signals.get_signal(signal_id)

    async def cancel_signal(self, signal_id: int) -> None:
        """取消一条调度信号。"""
        await self._signals.cancel_signal(signal_id)

    # ── 对话消息（chat IPC）────────────────────────────────────────────────

    async def add_chat_message(
        self,
        role: str,
        content: str,
        chat_id: str = "",
    ) -> int:
        """写入一条对话消息（role='user'|'assistant'）。"""
        return await self._chat.add_message(
            role,
            content,
            chat_id=chat_id,
        )

    async def has_pending_chat_message(self) -> bool:
        """非破坏性检查：是否有待处理的 user 消息（仅用于早唤醒轮询，不消费）。"""
        return await self._chat.has_pending_message()

    async def pop_pending_chat_message(self) -> Optional[dict[str, Any]]:
        """原子获取并标记最早一条待处理 user 消息（无则返回 None）。"""
        return await self._chat.pop_pending_message()

    async def drain_pending_for_chat(
        self, chat_id: str, after_id: int
    ) -> list[dict[str, Any]]:
        """原子获取并标记同 chat_id 中 id > after_id 的所有 pending 用户消息。
        用于合并图片等紧随文本消息之后到达的附件消息。
        """
        return await self._chat.drain_pending_for_chat(chat_id, after_id)

    async def get_chat_messages_since(
        self,
        since_id: int = 0,
        chat_id: str = "",
    ) -> list[dict[str, Any]]:
        """返回 id > since_id 的所有消息（可选按 chat_id 过滤）。"""
        return await self._chat.get_messages_since(
            since_id,
            chat_id=chat_id,
        )

    async def reset_in_progress_tasks(self) -> int:
        """重启时将所有 in_progress 任务重置为 pending。返回重置数量。"""
        result = await self._db.execute(
            "UPDATE tasks SET status='pending' WHERE status='in_progress'"
        )
        await self._db.commit()
        return result.rowcount if result else 0
