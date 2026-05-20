"""memory/episodic.py — 情节记忆（EpisodicMemory）。

双层存储（解决 O(n) P0 问题）：
  1. task-{id}.md / global.md  — 人类可读叙事（保持原行为；直接截取末尾注入 LLM context）
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
    role        TEXT NOT NULL,
    source_type TEXT NOT NULL,
    content     TEXT NOT NULL,
    affect      TEXT,
    ts          TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS narrative_fts USING fts5(
    id UNINDEXED,
    task_id,
    role,
    content,
    tokenize='unicode61'
);
"""


class EpisodicMemory:
    def __init__(self, memory_dir: Path, max_events: int = 0) -> None:
        self._dir = memory_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_events = max_events
        self._db_path = memory_dir / "episodic.db"
        self._conn = self._open_db()
        self._migrate_schema()       # 幂等索引补丁
        self._migrate_from_jsonl()  # 一次性历史数据导入

    @property
    def max_events(self) -> int:
        """Maximum retained events per type; 0 means unlimited."""
        return self._max_events

    # ── DB 初始化 & 迁移 ─────────────────────────────────────────────────────

    def _migrate_schema(self) -> None:
        """幂等索引补全。

        新增索引：
          idx_narrative_task_id — load_for_context 按 task_id 过滤
          idx_narrative_ts      — query_recent_narrative 时间窗查询
          idx_events_ts         — 按时间范围查询事件
        """
        ddl_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_narrative_task_id ON narrative(task_id)",
            "CREATE INDEX IF NOT EXISTS idx_narrative_ts ON narrative(ts)",
            "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)",
        ]
        for ddl in ddl_indexes:
            try:
                self._conn.execute(ddl)
            except Exception:
                pass
        try:
            self._conn.commit()
        except Exception:
            pass

    def _open_db(self) -> sqlite3.Connection:
        """打开 DB；损坏时自动删除并重建（_migrate_from_jsonl 重新导入历史）。"""
        try:
            conn = self._connect()
            conn.executescript(_DDL)
            conn.commit()
            return conn
        except sqlite3.DatabaseError:
            self._db_path.unlink(missing_ok=True)
            conn = self._connect()
            conn.executescript(_DDL)
            conn.commit()
            return conn

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

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

    def _task_path(self, task_id: str | None) -> Path:
        name = f"task-{task_id}.md" if task_id else "global.md"
        return self._dir / name

    def record(
        self,
        role: str,
        content: str,
        task_id: str | None = None,
        source_type: str = "",
        affect: dict[str, Any] | None = None,
    ) -> None:
        """追加一条情节记录（Tulving 1983 四元素绑定）。"""
        ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        src = source_type or _source_from_role(role)

        meta_parts = [f"role={role}", f"src={src}"]
        if affect:
            v = affect.get("valence")
            a = affect.get("arousal")
            if v is not None and a is not None:
                meta_parts.append(f"affect=({float(v):.2f},{float(a):.2f})")

        meta = " | ".join(meta_parts)
        block = f"\n---\n**[{ts}]** `{meta}`\n\n{content.strip()}\n"

        # 1. 写 .md（人类可读叙事，LLM context 注入源）
        path = self._task_path(task_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(block)

        # 2. 写 narrative DB + FTS5（用于 search() 全文检索）
        try:
            cur = self._conn.execute(
                "INSERT INTO narrative(task_id, role, source_type, content, affect, ts)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    task_id, role, src, content,
                    json.dumps(affect, ensure_ascii=False) if affect else None,
                    ts,
                ),
            )
            row_id = cur.lastrowid
            self._conn.execute(
                "INSERT INTO narrative_fts(id, task_id, role, content) VALUES (?, ?, ?, ?)",
                (row_id, task_id or "", role, content),
            )
            self._conn.commit()
        except Exception as _fts_err:
            _log.warning("[episodic] FTS5 写入失败（不影响 .md 主流程）: %s", _fts_err)

    def load_for_context(self, task_id: str | None, max_chars: int = 4000) -> str:
        """读取情节记忆，注入 LLM context。

        分段策略（"头部摘要 + 尾部完整"）：
        - 全文 <= max_chars：直接返回全文
        - 全文 > max_chars：取头部 head_chars 字符（包含任务目标/关键决策）
          + 省略提示 + 尾部 tail_chars 字符（最近行动/当前状态）
          head:tail = 1:3，尾部权重更高（近期上下文更重要）。
        比纯末尾截断保留了任务起点信息，避免 LLM 对长任务"失忆"。
        """
        path = self._task_path(task_id)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8")
        if len(text) <= max_chars:
            return text
        head_chars = max_chars // 4
        tail_chars = max_chars - head_chars - 60  # 60 for separator
        head = text[:head_chars]
        tail = text[-tail_chars:]
        omitted = len(text) - head_chars - tail_chars
        sep = f"\n\n… （省略约 {omitted} 字符的中间部分）…\n\n"
        return head + sep + tail

    def load_for_task_narrative(self, task_id: str | None, max_chars: int = 4000) -> str:
        """任务叙事模式（Ricoeur 1984）：跨 chat 读取该任务的完整情节流。"""
        return self.load_for_context(task_id, max_chars)

    def get_recent_turns(
        self,
        task_id: str | None,
        limit: int = 3,
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
        try:
            if task_id:
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
        return [p.stem.removeprefix("task-") for p in self._dir.glob("task-*.md")]

    def query_recent_narrative(self, hours: int = 24, limit: int = 10) -> list[dict[str, Any]]:
        """时间窗叙事查询：返回最近 hours 小时内的叙事记录（供实体共指消解使用）。

        字段：task_id, role, content, ts
        基于 idx_narrative_ts 索引，O(log n)。
        """
        since_dt = datetime.now(UTC) - timedelta(hours=max(1, hours))
        since_str = since_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        try:
            rows = self._conn.execute(
                "SELECT task_id, role, content, ts FROM narrative"
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

    def search(self, query: str, max_chars: int = 2000) -> str:
        """全文检索情节记忆：narrative FTS5（O(log n)）+ .md 文件降级扫描。"""
        if not query.strip():
            return ""
        hits: list[str] = []
        total = 0

        # 1. FTS5 narrative 检索
        safe = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE)
        terms = [t for t in safe.split() if len(t) > 1]
        if terms:
            fts_query = " OR ".join(terms)
            try:
                rows = self._conn.execute(
                    "SELECT task_id, role, content FROM narrative_fts"
                    " WHERE narrative_fts MATCH ? LIMIT 20",
                    (fts_query,),
                ).fetchall()
                for row in rows:
                    snippet = f"[task={row['task_id'] or 'global'} role={row['role']}] {row['content'][:300]}"
                    hits.append(snippet)
                    total += len(snippet)
                    if total >= max_chars:
                        return "\n\n---\n\n".join(hits)
            except Exception:
                pass  # FTS5 失败降级到文件扫描

        # 2. 降级：.md 文件关键词扫描
        if total < max_chars:
            keywords = [kw.lower() for kw in query.split() if kw]
            for md_path in sorted(self._dir.glob("*.md")):
                try:
                    text = md_path.read_text(encoding="utf-8")
                except Exception:
                    continue
                for block in text.split("---"):
                    block = block.strip()
                    if not block:
                        continue
                    lower = block.lower()
                    if all(kw in lower for kw in keywords):
                        snippet = f"[{md_path.name}]\n{block[:400]}"
                        hits.append(snippet)
                        total += len(snippet)
                        if total >= max_chars:
                            return "\n\n---\n\n".join(hits)

        return "\n\n---\n\n".join(hits) if hits else ""
