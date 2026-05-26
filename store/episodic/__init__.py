"""memory/episodic.py — 情节记忆（EpisodicMemory）。

双层存储（解决 O(n) P0 问题）：
    1. episodic/task-{id}.md / episodic/global.md  — 人类可读叙事（直接截取末尾注入 LLM context）
  2. episodic.db               — 结构化事件 + 叙事 FTS5 索引
     - events 表：替代 O(n) events.jsonl 扫描，O(log n) 索引查询
     - narrative 表 + narrative_fts：支持跨任务叙事全文检索

向后兼容：
  - 公开接口签名不变，调用方零改动
  - 启动时一次性将历史 events.jsonl 导入 DB（幂等）
  - DB 损坏时自动重建（从 .md 文件恢复叙事 FTS5，从 jsonl 恢复 events）

设计依据：
  - Tulving (1983): WHAT+WHEN+CONTEXT+AFFECT 四元素绑定
  - Johnson & Raye (1981): 来源监控（source_type）
  - Ricoeur (1984): 叙事连续性（task_id 组织，跨 chat 持久）
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Any

_log = logging.getLogger("lingzhou.episodic")

# source_type 常量（Johnson & Raye 1981 来源监控）
SRC_HUMAN      = "human"
SRC_INFERENCE  = "inference"
SRC_EXECUTION  = "execution"
SRC_SYSTEM     = "system"
SRC_REFLECTION = "reflection"


def _source_from_role(role: str) -> str:
    """从 role 自动派生 source_type（默认推断）。"""
    _map = {
        "user":           SRC_HUMAN,
        "assistant":      SRC_INFERENCE,
        "assistant_reply": SRC_INFERENCE,
        "consolidation":  SRC_SYSTEM,
        "reflection":     SRC_REFLECTION,
        "tool":           SRC_EXECUTION,
        "system":         SRC_SYSTEM,
    }
    return _map.get(role, SRC_INFERENCE)


# ── SQLite DDL ──────────────────────────────────────────────────────────────
_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- 结构化事件表（替代 events.jsonl O(n) 扫描）
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    ts         TEXT NOT NULL,
    data       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_type_id ON events(event_type, id DESC);

-- 叙事记录表（.md 文件的结构化镜像，用于 FTS5 检索）
CREATE TABLE IF NOT EXISTS narrative (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT,
    chat_id     TEXT,
    interlocutor_id TEXT,
    role        TEXT NOT NULL,
    source_type TEXT NOT NULL,
    content     TEXT NOT NULL,
    affect      TEXT,
    ts          TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS narrative_fts USING fts5(
    id UNINDEXED,
    task_id,
    chat_id,
    interlocutor_id,
    role,
    content,
    tokenize='unicode61'
);
-- 查询索引（幂等，DDL 统一管理）
CREATE INDEX IF NOT EXISTS idx_narrative_task_id ON narrative(task_id);
CREATE INDEX IF NOT EXISTS idx_narrative_chat_id ON narrative(chat_id);
CREATE INDEX IF NOT EXISTS idx_narrative_interlocutor_id ON narrative(interlocutor_id);
CREATE INDEX IF NOT EXISTS idx_narrative_ts ON narrative(ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


class EpisodicMemory:
    def __init__(self, memory_dir: Path, max_events: int = 0) -> None:
        self._dir = memory_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._narrative_dir = self._dir / "episodic"
        self._narrative_dir.mkdir(parents=True, exist_ok=True)
        self._chat_dir = self._narrative_dir / "chat"
        self._chat_dir.mkdir(parents=True, exist_ok=True)
        self._interlocutor_dir = self._narrative_dir / "interlocutor"
        self._interlocutor_dir.mkdir(parents=True, exist_ok=True)
        self._daily_dir = self._narrative_dir / "daily"
        self._daily_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_narrative_files()
        self._max_events = max_events
        self._db_path = memory_dir / "episodic.db"
        self._conn = None
        self._session_depth = 0
        with self._db_session():
            self._migrate_from_jsonl()  # 一次性历史数据导入

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self, "_conn_ref", None)
        if conn is None:
            raise RuntimeError("episodic db session is not open")
        return conn

    @_conn.setter
    def _conn(self, value: sqlite3.Connection | None) -> None:
        self._conn_ref = value

    @contextmanager
    def _db_session(self):
        if self._conn_ref is not None:
            self._session_depth += 1
            try:
                yield self._conn_ref
            finally:
                self._session_depth -= 1
            return

        conn = self._open_db()
        self._conn = conn
        self._session_depth = 1
        try:
            yield conn
        finally:
            self._session_depth -= 1
            if self._session_depth == 0:
                self.close()

    def close(self) -> None:
        conn = getattr(self, "_conn_ref", None)
        self._conn_ref = None
        self._session_depth = 0
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    @property
    def max_events(self) -> int:
        """Maximum retained events per type; 0 means unlimited."""
        return self._max_events

    # ── DB 初始化 & 迁移 ─────────────────────────────────────────────────────

    def _open_db(self) -> sqlite3.Connection:
        """打开 DB；损坏时自动删除并重建（_migrate_from_jsonl 重新导入历史）。"""
        try:
            conn = self._connect()
            conn.executescript(_DDL)
            self._ensure_schema_compat(conn)
            conn.commit()
            return conn
        except sqlite3.DatabaseError:
            self._db_path.unlink(missing_ok=True)
            conn = self._connect()
            conn.executescript(_DDL)
            self._ensure_schema_compat(conn)
            conn.commit()
            return conn

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema_compat(self, conn: sqlite3.Connection) -> None:
        narrative_columns = set(self._table_columns(conn, "narrative"))
        if "chat_id" not in narrative_columns:
            conn.execute("ALTER TABLE narrative ADD COLUMN chat_id TEXT")
        if "interlocutor_id" not in narrative_columns:
            conn.execute("ALTER TABLE narrative ADD COLUMN interlocutor_id TEXT")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_narrative_chat_id ON narrative(chat_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_narrative_interlocutor_id ON narrative(interlocutor_id)")

        fts_columns = set(self._table_columns(conn, "narrative_fts"))
        if "chat_id" not in fts_columns or "interlocutor_id" not in fts_columns:
            self._rebuild_narrative_fts(conn)

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
        try:
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        except Exception:
            return []
        return [str(row[1]) for row in rows]

    def _rebuild_narrative_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE IF EXISTS narrative_fts")
        conn.execute(
            "CREATE VIRTUAL TABLE narrative_fts USING fts5("
            " id UNINDEXED,"
            " task_id,"
            " chat_id,"
            " interlocutor_id,"
            " role,"
            " content,"
            " tokenize='unicode61'"
            ")"
        )
        conn.execute(
            "INSERT INTO narrative_fts(id, task_id, chat_id, interlocutor_id, role, content) "
            "SELECT id, COALESCE(task_id, ''), COALESCE(chat_id, ''), COALESCE(interlocutor_id, ''), role, content FROM narrative"
        )

    def _migrate_from_jsonl(self) -> None:
        """一次性：将历史 events.jsonl 导入 SQLite events 表（幂等，count>0 时跳过）。"""
        path = self._dir / "events.jsonl"
        if not path.exists():
            return
        try:
            count = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            if count > 0:
                return  # 已导入，跳过
            for raw in path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                    et = d.pop("t", "unknown")
                    ts = d.pop("ts", datetime.now(UTC).isoformat())
                    self._conn.execute(
                        "INSERT INTO events(event_type, ts, data) VALUES (?, ?, ?)",
                        (et, ts, json.dumps(d, ensure_ascii=False)),
                    )
                except Exception:
                    pass
            self._conn.commit()
        except Exception:
            pass

    # ── 叙事层（.md + narrative DB）─────────────────────────────────────────

    @staticmethod
    def _narrative_filename(task_id: str | None) -> str:
        return f"task-{task_id}.md" if task_id else "global.md"

    @staticmethod
    def _chat_filename(chat_id: str) -> str:
        normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(chat_id or "")).strip("._-")
        slug = normalized[:48] or "chat"
        digest = hashlib.md5(str(chat_id).encode("utf-8")).hexdigest()[:10]
        return f"chat-{slug}-{digest}.md"

    @staticmethod
    def _interlocutor_filename(interlocutor_id: str) -> str:
        normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(interlocutor_id or "")).strip("._-")
        slug = normalized[:48] or "interlocutor"
        digest = hashlib.md5(str(interlocutor_id).encode("utf-8")).hexdigest()[:10]
        return f"interlocutor-{slug}-{digest}.md"

    @staticmethod
    def _daily_filename(day_stamp: str) -> str:
        return f"{day_stamp}.md"

    @classmethod
    def narrative_path_for_dir(cls, memory_dir: Path, task_id: str | None) -> Path:
        return Path(memory_dir) / "episodic" / cls._narrative_filename(task_id)

    @classmethod
    def daily_path_for_dir(cls, memory_dir: Path, day_stamp: str) -> Path:
        return Path(memory_dir) / "episodic" / "daily" / cls._daily_filename(day_stamp)

    @classmethod
    def chat_path_for_dir(cls, memory_dir: Path, chat_id: str) -> Path:
        return Path(memory_dir) / "episodic" / "chat" / cls._chat_filename(chat_id)

    @classmethod
    def interlocutor_path_for_dir(cls, memory_dir: Path, interlocutor_id: str) -> Path:
        return Path(memory_dir) / "episodic" / "interlocutor" / cls._interlocutor_filename(interlocutor_id)

    @classmethod
    def legacy_narrative_path_for_dir(cls, memory_dir: Path, task_id: str | None) -> Path:
        return Path(memory_dir) / cls._narrative_filename(task_id)

    def _task_path(self, task_id: str | None) -> Path:
        return self.narrative_path_for_dir(self._dir, task_id)

    def _daily_path(self, day_stamp: str) -> Path:
        return self.daily_path_for_dir(self._dir, day_stamp)

    def _chat_path(self, chat_id: str) -> Path:
        return self.chat_path_for_dir(self._dir, chat_id)

    def _interlocutor_path(self, interlocutor_id: str) -> Path:
        return self.interlocutor_path_for_dir(self._dir, interlocutor_id)

    def _legacy_task_path(self, task_id: str | None) -> Path:
        return self.legacy_narrative_path_for_dir(self._dir, task_id)

    def _resolve_task_path(self, task_id: str | None) -> Path:
        path = self._task_path(task_id)
        if path.exists():
            return path
        legacy = self._legacy_task_path(task_id)
        return legacy if legacy.exists() else path

    def _iter_legacy_narrative_files(self) -> list[Path]:
        paths: list[Path] = []
        global_path = self._legacy_task_path(None)
        if global_path.exists():
            paths.append(global_path)
        paths.extend(sorted(self._dir.glob("task-*.md")))
        return paths

    def _iter_narrative_files(self) -> list[Path]:
        files: dict[str, Path] = {}
        global_path = self._task_path(None)
        if global_path.exists():
            files[global_path.name] = global_path
        for md_path in sorted(self._narrative_dir.glob("task-*.md")):
            files.setdefault(md_path.name, md_path)
        for legacy_path in self._iter_legacy_narrative_files():
            files.setdefault(legacy_path.name, legacy_path)
        return [files[name] for name in sorted(files)]

    def _migrate_legacy_narrative_files(self) -> None:
        """将旧版根目录 narrative 文件迁移到 episodic/ 子目录（幂等）。"""
        for legacy_path in self._iter_legacy_narrative_files():
            target = self._narrative_dir / legacy_path.name
            if target.exists():
                continue
            try:
                legacy_path.rename(target)
            except OSError as exc:
                _log.warning("[episodic] 迁移 narrative 文件失败: %s -> %s (%s)", legacy_path, target, exc)

    @staticmethod
    def _append_markdown_block(path: Path, block: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(block)

    @staticmethod
    def _day_stamp_from_ts(ts: str) -> str:
        return (ts or "").strip()[:10]

    def _insert_narrative_row(
        self,
        *,
        task_id: str | None,
        chat_id: str | None,
        interlocutor_id: str | None,
        role: str,
        source_type: str,
        content: str,
        affect_json: str | None,
        ts: str,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO narrative(task_id, chat_id, interlocutor_id, role, source_type, content, affect, ts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, chat_id, interlocutor_id, role, source_type, content, affect_json, ts),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def _sync_narrative_fts(
        self,
        *,
        row_id: int,
        task_id: str | None,
        chat_id: str | None,
        interlocutor_id: str | None,
        role: str,
        content: str,
    ) -> None:
        self._conn.execute(
            "INSERT INTO narrative_fts(id, task_id, chat_id, interlocutor_id, role, content) VALUES (?, ?, ?, ?, ?, ?)",
            (row_id, task_id or "", chat_id or "", interlocutor_id or "", role, content),
        )
        self._conn.commit()

    def record(
        self,
        role: str,
        content: str,
        task_id: str | None = None,
        source_type: str = "",
        affect: dict[str, Any] | None = None,
        *,
        chat_id: str | None = None,
        interlocutor_id: str | None = None,
    ) -> None:
        """追加一条情节记录（Tulving 1983 四元素绑定）。"""
        ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        src = source_type or _source_from_role(role)

        meta_parts = [f"role={role}", f"src={src}"]
        if chat_id:
            meta_parts.append(f"chat={chat_id}")
        if interlocutor_id:
            meta_parts.append(f"interlocutor={interlocutor_id}")
        if affect:
            v = affect.get("valence")
            a = affect.get("arousal")
            if v is not None and a is not None:
                meta_parts.append(f"affect=({float(v):.2f},{float(a):.2f})")

        meta = " | ".join(meta_parts)
        block = f"\n---\n**[{ts}]** `{meta}`\n\n{content.strip()}\n"

        # 1. 写 .md（人类可读叙事，LLM context 注入源）
        self._append_markdown_block(self._task_path(task_id), block)
        if chat_id and role in {"user", "assistant_reply"}:
            self._append_markdown_block(self._chat_path(chat_id), block)
        if interlocutor_id and role in {"user", "assistant_reply"}:
            self._append_markdown_block(self._interlocutor_path(interlocutor_id), block)
        self._append_markdown_block(self._daily_path(self._day_stamp_from_ts(ts)), block)

        affect_json = json.dumps(affect, ensure_ascii=False) if affect else None

        with self._db_session():
            # 2. 先稳定写 narrative 表（供 recent-turns / recent-narrative 使用）
            try:
                row_id = self._insert_narrative_row(
                    task_id=task_id,
                    chat_id=chat_id,
                    interlocutor_id=interlocutor_id,
                    role=role,
                    source_type=src,
                    content=content,
                    affect_json=affect_json,
                    ts=ts,
                )
            except Exception as _narrative_err:
                _log.warning("[episodic] narrative 写入失败（.md 已保留）: %s", _narrative_err)
                return

            # 3. 再同步 FTS5（失败时仅影响全文检索，不影响主叙事层）
            try:
                self._sync_narrative_fts(
                    row_id=row_id,
                    task_id=task_id,
                    chat_id=chat_id,
                    interlocutor_id=interlocutor_id,
                    role=role,
                    content=content,
                )
            except Exception as _fts_err:
                _log.warning("[episodic] FTS5 写入失败（narrative 已提交，search 将退回 .md 扫描）: %s", _fts_err)

    @staticmethod
    def _load_markdown_context(path: Path, max_chars: int = 4000) -> str:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8")
        if len(text) <= max_chars:
            return text
        head_chars = max_chars // 4
        tail_chars = max_chars - head_chars - 60
        head = text[:head_chars]
        tail = text[-tail_chars:]
        omitted = len(text) - head_chars - tail_chars
        sep = f"\n\n… （省略约 {omitted} 字符的中间部分）…\n\n"
        return head + sep + tail

    def load_for_context(self, task_id: str | None, max_chars: int = 4000) -> str:
        """读取情节记忆，注入 LLM context。

        分段策略（"头部摘要 + 尾部完整"）：
        - 全文 <= max_chars：直接返回全文
        - 全文 > max_chars：取头部 head_chars 字符（包含任务目标/关键决策）
          + 省略提示 + 尾部 tail_chars 字符（最近行动/当前状态）
          head:tail = 1:3，尾部权重更高（近期上下文更重要）。
        比纯末尾截断保留了任务起点信息，避免 LLM 对长任务"失忆"。
        """
        return self._load_markdown_context(self._resolve_task_path(task_id), max_chars)

    def load_for_chat_context(self, chat_id: str | None, max_chars: int = 4000) -> str:
        """读取 chat 维度的情节连续性，跨 task 保留同一 chat 的完整对话线索。"""
        if not chat_id:
            return ""
        return self._load_markdown_context(self._chat_path(chat_id), max_chars)

    def load_for_interlocutor_context(self, interlocutor_id: str | None, max_chars: int = 4000) -> str:
        """读取当前交互对象维度的情节连续性，跨 chat 保留同一对象的互动片段。"""
        if not interlocutor_id:
            return ""
        return self._load_markdown_context(self._interlocutor_path(interlocutor_id), max_chars)

    def load_for_task_narrative(self, task_id: str | None, max_chars: int = 4000) -> str:
        """任务叙事模式（Ricoeur 1984）：跨 chat 读取该任务的完整情节流。"""
        return self.load_for_context(task_id, max_chars)

    def load_recent_daily_context(self, days: int = 2, max_chars: int = 1200) -> str:
        """读取最近若干天的 daily 叙事，用于跨任务的短程连续性。"""
        days = max(1, days)
        if max_chars <= 0:
            return ""

        stamps = [
            (datetime.now(UTC) - timedelta(days=offset)).strftime("%Y-%m-%d")
            for offset in range(days)
        ]
        per_day_limit = max(120, max_chars // max(1, len(stamps)))
        sections: list[str] = []
        total_chars = 0

        for stamp in stamps:
            path = self._daily_path(stamp)
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if not text:
                continue

            snippet = text
            if len(snippet) > per_day_limit:
                omitted = len(snippet) - per_day_limit
                snippet = f"… （省略约 {omitted} 字符的更早内容）…\n\n{snippet[-per_day_limit:]}"
            block = f"[{stamp}]\n{snippet}"
            if total_chars + len(block) > max_chars:
                remaining = max_chars - total_chars
                if remaining <= 0:
                    break
                block = block[:remaining]
            sections.append(block)
            total_chars += len(block)
            if total_chars >= max_chars:
                break

        return "\n\n---\n\n".join(sections)

    def search_recent_daily(self, query: str, days: int = 2, max_chars: int = 1200) -> str:
        """在最近若干天的 daily 中按 query 检索相关片段。

        用于长期记忆命中不足时的短期补短，避免每轮固定注入整段 recent daily。
        """
        query = (query or "").strip()
        if not query:
            return ""
        days = max(1, days)
        if max_chars <= 0:
            return ""

        safe = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE)
        strict = [t.lower() for t in safe.split() if len(t) >= 2 and not (t.isascii() and len(t) < 5)]
        relaxed = [t.lower() for t in safe.split() if len(t) > 1]
        term_sets = [strict if strict else relaxed]
        if strict and relaxed != strict:
            term_sets.append(relaxed)
        if not term_sets[0]:
            return ""

        scored_hits: list[tuple[int, int, str]] = []
        for terms in term_sets:
            scored_hits = []
            for offset in range(days):
                stamp = (datetime.now(UTC) - timedelta(days=offset)).strftime("%Y-%m-%d")
                path = self._daily_path(stamp)
                if not path.exists():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                for block in reversed(text.split("---")):
                    block = block.strip()
                    if not block:
                        continue
                    body = "\n".join(
                        line for line in block.splitlines()
                        if line.strip() and not line.startswith("**[")
                    ).strip()
                    if not body:
                        continue
                    lower_body = body.lower()
                    match_count = sum(1 for term in terms if term in lower_body)
                    if match_count <= 0:
                        continue
                    snippet = f"[{stamp}]\n{block[:400]}"
                    # 优先保留更多 query 词命中的片段；同分时优先更新近的记录。
                    scored_hits.append((match_count, -offset, snippet))
            if scored_hits:
                break

        if not scored_hits:
            return ""

        scored_hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
        hits: list[str] = []
        total = 0
        for _, _, snippet in scored_hits:
            hits.append(snippet)
            total += len(snippet)
            if total >= max_chars:
                return "\n\n---\n\n".join(hits)
        return "\n\n---\n\n".join(hits)

    def get_recent_turns(
        self,
        task_id: str | None = None,
        limit: int = 3,
        *,
        chat_id: str | None = None,
        interlocutor_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """从 narrative 表返回最近 limit 条对话轮次（用户消息 + 智能体回复）。

        这是 STM 对话缓冲的正确来源——基于情节记忆而非原始 chat_messages 表，
        保留了 Tulving (1983) 四元素绑定中的时间标签和情感状态。

        返回列表按时间升序（最旧→最新），字段:
            role: "user" | "assistant_reply"
            content: str
            ts: str (UTC)
            affect: dict | None  {"valence": float, "arousal": float}
        """
        with self._db_session():
            try:
                if chat_id and interlocutor_id:
                    sql = (
                        "SELECT role, content, ts, affect FROM narrative "
                        "WHERE chat_id = ? AND interlocutor_id = ? AND role IN ('user', 'assistant_reply') "
                        "ORDER BY id DESC LIMIT ?"
                    )
                    rows = self._conn.execute(sql, (chat_id, interlocutor_id, limit)).fetchall()
                elif chat_id:
                    sql = (
                        "SELECT role, content, ts, affect FROM narrative "
                        "WHERE chat_id = ? AND role IN ('user', 'assistant_reply') "
                        "ORDER BY id DESC LIMIT ?"
                    )
                    rows = self._conn.execute(sql, (chat_id, limit)).fetchall()
                elif interlocutor_id:
                    sql = (
                        "SELECT role, content, ts, affect FROM narrative "
                        "WHERE interlocutor_id = ? AND role IN ('user', 'assistant_reply') "
                        "ORDER BY id DESC LIMIT ?"
                    )
                    rows = self._conn.execute(sql, (interlocutor_id, limit)).fetchall()
                elif task_id:
                    sql = (
                        "SELECT role, content, ts, affect FROM narrative "
                        "WHERE task_id = ? AND role IN ('user', 'assistant_reply') "
                        "ORDER BY id DESC LIMIT ?"
                    )
                    rows = self._conn.execute(sql, (task_id, limit)).fetchall()
                else:
                    sql = (
                        "SELECT role, content, ts, affect FROM narrative "
                        "WHERE role IN ('user', 'assistant_reply') "
                        "ORDER BY id DESC LIMIT ?"
                    )
                    rows = self._conn.execute(sql, (limit,)).fetchall()
            except Exception as e:
                _log.warning("[episodic] get_recent_turns 失败: %s", e)
                return []
        result: list[dict[str, Any]] = []
        for r in reversed(rows):
            affect: dict[str, Any] | None = None
            if r["affect"]:
                try:
                    affect = json.loads(r["affect"])
                except Exception:
                    pass
            result.append({
                "role": r["role"],
                "content": r["content"] or "",
                "ts": r["ts"] or "",
                "affect": affect,
            })
        return result

    def list_tasks(self) -> list[str]:
        """返回已有情节记忆的任务 ID 列表。"""
        return [p.stem.removeprefix("task-") for p in self._iter_narrative_files() if p.name.startswith("task-")]

    def list_recent_narrative(self, limit: int = 10) -> list[dict[str, Any]]:
        """返回最新若干条叙事记录（不解释用户时间词，仅供 recent 预热）。"""
        with self._db_session():
            try:
                rows = self._conn.execute(
                    "SELECT task_id, chat_id, role, content, ts FROM narrative ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
            except Exception:
                return []

    def query_recent_narrative(self, hours: int = 24, limit: int = 10) -> list[dict[str, Any]]:
        """时间窗叙事查询：返回最近 hours 小时内的叙事记录（供实体共指消解使用）。

        字段：task_id, role, content, ts
        基于 idx_narrative_ts 索引，O(log n)。
        """
        since_dt = datetime.now(UTC) - timedelta(hours=max(1, hours))
        since_str = since_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._db_session():
            try:
                rows = self._conn.execute(
                    "SELECT task_id, chat_id, role, content, ts FROM narrative"
                    " WHERE ts >= ? ORDER BY id DESC LIMIT ?",
                    (since_str, limit),
                ).fetchall()
                return [dict(r) for r in rows]
            except Exception:
                return []

    # ── 结构化事件日志（O(log n) SQLite；JSONL 降级兜底）────────────────────

    def record_event(self, event_type: str, data: dict[str, Any]) -> None:
        """追加一条结构化事件（perception / emotion 快照）。"""
        ts = datetime.now(UTC).isoformat()
        try:
            with self._db_session():
                self._conn.execute(
                    "INSERT INTO events(event_type, ts, data) VALUES (?, ?, ?)",
                    (event_type, ts, json.dumps(data, ensure_ascii=False)),
                )
                self._conn.commit()
                if self._max_events > 0:
                    self._rotate_events_db(event_type)
        except Exception:
            # DB 不可用时降级写 JSONL
            path = self._dir / "events.jsonl"
            entry: dict[str, Any] = {"t": event_type, "ts": ts, **data}
            try:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def _rotate_events_db(self, event_type: str) -> None:
        """保留该类型最新 max_events 条，删除超出的旧记录。"""
        try:
            self._conn.execute(
                """DELETE FROM events WHERE event_type = ? AND id NOT IN (
                    SELECT id FROM events WHERE event_type = ? ORDER BY id DESC LIMIT ?
                )""",
                (event_type, event_type, self._max_events),
            )
            self._conn.commit()
        except Exception:
            pass

    def list_events(self, event_type: str, limit: int = 10) -> list[dict[str, Any]]:
        """返回最近 limit 条指定类型事件（时间升序）。O(log n) 索引扫描。"""
        with self._db_session():
            try:
                rows = self._conn.execute(
                    "SELECT ts, data FROM events WHERE event_type = ? ORDER BY id DESC LIMIT ?",
                    (event_type, limit),
                ).fetchall()
                result: list[dict[str, Any]] = []
                for row in reversed(rows):
                    try:
                        d_data = json.loads(row["data"])
                        d_data["t"] = event_type
                        d_data["ts"] = row["ts"]
                        result.append(d_data)
                    except Exception:
                        pass
                return result
            except Exception:
                return self._fallback_list_events(event_type, limit)

    def _fallback_list_events(self, event_type: str, limit: int) -> list[dict[str, Any]]:
        """DB 不可用时回退到 JSONL 逆序扫描。"""
        path = self._dir / "events.jsonl"
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        matched: list[dict[str, Any]] = []
        for raw in reversed(lines):
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
                if d.get("t") == event_type:
                    matched.append(d)
                    if len(matched) >= limit:
                        break
            except Exception:
                pass
        matched.reverse()
        return matched

    def list_events_multi(
        self, event_types: list[str], limit: int = 10
    ) -> dict[str, list[dict[str, Any]]]:
        """一次查询，按类型分桶返回最近 limit 条（时间升序）。O(log n) 索引扫描。"""
        result: dict[str, list[dict[str, Any]]] = {t: [] for t in event_types}
        if not event_types:
            return result
        with self._db_session():
            try:
                placeholders = ",".join("?" * len(event_types))
                rows = self._conn.execute(
                    f"SELECT event_type, ts, data FROM events"
                    f" WHERE event_type IN ({placeholders}) ORDER BY id DESC",
                    event_types,
                ).fetchall()
                for row in rows:
                    et = row["event_type"]
                    if et in result and len(result[et]) < limit:
                        try:
                            d = json.loads(row["data"])
                            d["t"] = et
                            d["ts"] = row["ts"]
                            result[et].append(d)
                        except Exception:
                            pass
                for v in result.values():
                    v.reverse()  # 恢复时间升序
                return result
            except Exception:
                # DB 不可用：降级逐类型 JSONL 扫描
                path = self._dir / "events.jsonl"
                if not path.exists():
                    return result
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    return result
                for raw in reversed(lines):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        d = json.loads(raw)
                        t = d.get("t")
                        if t in result and len(result[t]) < limit:
                            result[t].append(d)
                    except Exception:
                        pass
                    if all(len(v) >= limit for v in result.values()):
                        break
                for v in result.values():
                    v.reverse()
                return result

    def search(
        self,
        query: str,
        max_chars: int = 2000,
        exclude_task_id: str | None = None,
    ) -> str:
        """全文检索情节记忆：narrative FTS5（O(log n)）+ .md 文件降级扫描。

        exclude_task_id: 跳过该 task 自身的 narrative 行（避免把当前任务目标作为
        跨任务命中反复注入）。
        """
        if not query.strip():
            return ""
        hits: list[str] = []
        total = 0
        # 用于检测「content 本质上就是查询文本本身」（旧任务目标被 FTS5 回显）：
        # 条件：query 是 content 的子串，且 content 长度 < query 的 1.3 倍（即 content ≈ query，非扩展内容）
        _q = query.strip()

        with self._db_session():
            # 1. FTS5 narrative 检索
            safe = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE)
            # ASCII 词 ≥5 字符（过滤 "core" "loop" "task" 等常见路径/词，防止 OR 泛命中）；
            # 非 ASCII（中文等）词 ≥2 字符。若严格过滤后为空则回退原行为。
            _strict = [t for t in safe.split() if len(t) >= 2 and not (t.isascii() and len(t) < 5)]
            terms = _strict if _strict else [t for t in safe.split() if len(t) > 1]
            if terms:
                fts_query = " OR ".join(terms)
                try:
                    rows = self._conn.execute(
                        "SELECT task_id, chat_id, role, content FROM narrative_fts"
                        " WHERE narrative_fts MATCH ? LIMIT 50",  # 扩大候选集，补偿 Python 层 exclude_task_id 过滤导致的有效命中减少
                        (fts_query,),
                    ).fetchall()
                    for row in rows:
                        # 跳过当前任务自身的条目（避免将自己的历史作为跨任务命中）
                        if exclude_task_id and row['task_id'] == exclude_task_id:
                            continue
                        # 跳过 content 本质上就是查询文本本身（旧任务目标被 FTS5 回显）
                        if _q and row['content'].strip() == _q:
                            continue
                        origin = f"task={row['task_id'] or 'global'}"
                        if row['chat_id']:
                            origin += f" chat={row['chat_id']}"
                        snippet = f"[{origin} role={row['role']}] {row['content'][:300]}"
                        hits.append(snippet)
                        total += len(snippet)
                        if total >= max_chars:
                            return "\n\n---\n\n".join(hits)
                except Exception:
                    pass  # FTS5 失败降级到文件扫描

        # 2. 降级：.md 文件关键词扫描
        if total < max_chars:
            keywords = [kw.lower() for kw in query.split() if kw]
            for md_path in self._iter_narrative_files():
                # 跳过当前任务自身的 .md
                if exclude_task_id and md_path.name == f"task-{exclude_task_id}.md":
                    continue
                try:
                    text = md_path.read_text(encoding="utf-8")
                except Exception:
                    continue
                for block in text.split("---"):
                    block = block.strip()
                    if not block:
                        continue
                    # 跳过块内容本质上就是查询文本本身（剖除 markdown 元数据行后比较）
                    _block_body = '\n'.join(
                        l for l in block.splitlines() if l.strip() and not l.startswith('**[')
                    ).strip()
                    if _q and _block_body == _q:
                        continue
                    lower = block.lower()
                    if all(kw in lower for kw in keywords):
                        snippet = f"[{md_path.name}]\n{block[:400]}"
                        hits.append(snippet)
                        total += len(snippet)
                        if total >= max_chars:
                            return "\n\n---\n\n".join(hits)

        return "\n\n---\n\n".join(hits) if hits else ""
