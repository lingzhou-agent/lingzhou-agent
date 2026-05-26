"""store/task/schema.py — DDL 常量与任务层工具函数。"""
from __future__ import annotations

from typing import Any

from store.task.models import Task

# ── 公开常量 ────────────────────────────────────────────────────────────────

OPEN_TASK_STATUSES = ("pending", "ready", "in_progress", "resumed", "waiting")
RUNNABLE_TASK_STATUSES = ("pending", "ready", "in_progress", "resumed")
TASK_DUPLICATE_REUSE_SCORE = 0.66
TASK_SIMILARITY_CONTEXT_SCORE = 0.45
_TASK_SIMILARITY_SCAN_LIMIT = 200

_TASK_STATUS_RANK = {
    "in_progress": 0,
    "resumed": 1,
    "ready": 2,
    "pending": 3,
    "waiting": 4,
}
_TASK_PRIORITY_RANK = {
    "critical": 0,
    "high": 1,
    "normal": 2,
    "low": 3,
}

# ── 工具函数 ────────────────────────────────────────────────────────────────

def build_task_similarity_query(*parts: Any) -> str:
    seen: set[str] = set()
    cleaned: list[str] = []
    for part in parts:
        text = " ".join(str(part or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return " ".join(cleaned)


def _normalize_task_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _task_search_text(task: Task) -> str:
    return build_task_similarity_query(task.title, task.goal, task.next_step)


def _task_similarity_score(query_text: str, task: Task) -> float:
    from memory.quality_checker import calculate_relevance

    query = _normalize_task_text(query_text)
    candidate = _normalize_task_text(_task_search_text(task))
    if not query or not candidate:
        return 0.0
    if query == candidate:
        return 1.0

    title = _normalize_task_text(task.title)
    goal = _normalize_task_text(task.goal)
    next_step = _normalize_task_text(task.next_step)
    score = max(
        calculate_relevance(query, candidate),
        calculate_relevance(query, title) if title else 0.0,
        calculate_relevance(query, goal) if goal else 0.0,
        calculate_relevance(query, next_step) if next_step else 0.0,
    )

    if title and len(title) >= 4 and (query in title or title in query):
        score = max(score, 0.9)
    if goal and len(goal) >= 6 and (query in goal or goal in query):
        score = max(score, 0.85)
    if next_step and len(next_step) >= 6 and (query in next_step or next_step in query):
        score = max(score, 0.8)

    return min(score, 1.0)


def build_task_run_result_patch(
    *,
    run_id: int,
    status: str,
    worker_type: str,
    tool_name: str,
    session_id: str,
    summary: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "last_run_id": int(run_id),
        "last_run_status": str(status or ""),
        "worker_type": str(worker_type or ""),
        "tool_name": str(tool_name or ""),
        "session_id": str(session_id or ""),
        "summary": str(summary or ""),
        "error": error,
    }


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
    status     TEXT    NOT NULL DEFAULT 'pending',  -- user: pending | processing | processed ; assistant: pending | processed | delivered
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

FULL_DDL = (
    _CREATE_TASKS
    + _CREATE_FAILURES
    + _CREATE_FACTS
    + _CREATE_SIGNALS
    + _CREATE_CHAT
    + _CREATE_RUNS
    + _CREATE_META_REFLECTIONS
    + _CREATE_INDEXES
)
