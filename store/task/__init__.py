"""store/task/__init__.py — 统一任务存储入口（TaskStore + 所有子存储）。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from store.task.models import Failure, MetaReflection, Run, Task
from store.task.schema import (
    OPEN_TASK_STATUSES,
    RUNNABLE_TASK_STATUSES,
    TASK_DUPLICATE_REUSE_SCORE,
    TASK_SIMILARITY_CONTEXT_SCORE,
    _CREATE_CHAT,
    _CREATE_FACTS,
    _CREATE_FAILURES,
    _CREATE_INDEXES,
    _CREATE_META_REFLECTIONS,
    _CREATE_RUNS,
    _CREATE_SIGNALS,
    _CREATE_TASKS,
    _TASK_PRIORITY_RANK,
    _TASK_SIMILARITY_SCAN_LIMIT,
    _TASK_STATUS_RANK,
    build_task_run_result_patch,
    build_task_similarity_query,
)
from store.task.chat import ChatMessageStore, sanitize_chat_content
from store.task.fact import FactStore, build_fact_upsert
from store.task.failure import FailureStore
from store.task.ingress import IngressStore
from store.task.reflection import MetaReflectionStore
from store.task.run import RunStore
from store.task.signal import SignalStore
from store.task.state import TaskStateStore, build_task_data, build_task_insert

logger = logging.getLogger(__name__)


def _task_similarity_score(query_text: str, task: Task) -> float:
    from store.task.schema import _task_similarity_score as _score
    return _score(query_text, task)


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
        await self._migrate()
        await self._db.executescript(
            _CREATE_TASKS + _CREATE_FAILURES + _CREATE_FACTS + _CREATE_SIGNALS
            + _CREATE_CHAT + _CREATE_RUNS + _CREATE_META_REFLECTIONS + _CREATE_INDEXES
        )
        await self._db.execute(
            "UPDATE chat_messages SET status='pending' WHERE role='user' AND status='processing'"
        )
        await self._db.commit()
        await self._migrate_interlocutor_facts()

    async def close(self) -> None:
        if self._db_opt:
            await self._db_opt.close()
            self._db_opt = None

    async def wal_checkpoint(self) -> None:
        """触发 WAL checkpoint（TRUNCATE 模式）。"""
        await self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        await self._db.commit()

    # ── 一次性迁移（旧列式 → JSON-first）────────────────────────────────

    async def _migrate(self) -> None:
        db = self._db_opt
        assert db is not None

        async with db.execute(
            "SELECT COUNT(*) FROM pragma_table_info('tasks') WHERE name='data'"
        ) as cur:
            row = await cur.fetchone()
            if row and row[0] > 0:
                async with db.execute(
                    "SELECT COUNT(*) FROM pragma_table_info('failures') WHERE name='dismissed'"
                ) as cur2:
                    row2 = await cur2.fetchone()
                    if not (row2 and row2[0] > 0):
                        await db.execute(
                            "ALTER TABLE failures ADD COLUMN dismissed INTEGER NOT NULL DEFAULT 0"
                        )
                        await db.commit()
                return

        async with db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tasks'"
        ) as cur:
            row = await cur.fetchone()
            if not (row and row[0] > 0):
                return

        logger.info("[task_store] 检测到旧列式 schema，开始一次性迁移 → JSON-first")

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

        await db.executescript("""
            DROP TABLE IF EXISTS tasks;
            DROP TABLE IF EXISTS failures;
        """)
        await db.executescript(_CREATE_TASKS + _CREATE_FAILURES)

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

    async def _migrate_interlocutor_facts(self) -> None:
        db = self._db_opt
        assert db is not None

        async with db.execute(
            "SELECT key, value, scope, updated_at FROM facts ORDER BY updated_at ASC"
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return

        profile_ids = {
            str(value or "").strip()
            for key, value, _, _ in rows
            if str(key or "").endswith(":person_profile_id") and str(value or "").strip()
        }

        migrations: list[tuple[str, str, str, str, str]] = []
        for key, value, scope, updated_at in rows:
            normalized_key = str(key or "").strip()
            new_key = ""
            if normalized_key.endswith(":person_profile_id"):
                new_key = normalized_key[: -len(":person_profile_id")] + ":interlocutor_profile_id"
            elif normalized_key.startswith("user:"):
                parts = normalized_key.split(":")
                if len(parts) >= 3 and parts[1] in profile_ids:
                    new_key = "interlocutor:" + ":".join(parts[1:])
            if new_key and new_key != normalized_key:
                migrations.append((normalized_key, new_key, str(value or ""), str(scope or "general"), str(updated_at or "")))

        if not migrations:
            return

        for old_key, new_key, value, scope, updated_at in migrations:
            await db.execute(
                """
                INSERT INTO facts (key, value, scope, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = CASE WHEN excluded.updated_at >= facts.updated_at THEN excluded.value ELSE facts.value END,
                    scope = CASE WHEN excluded.updated_at >= facts.updated_at THEN excluded.scope ELSE facts.scope END,
                    updated_at = CASE WHEN excluded.updated_at >= facts.updated_at THEN excluded.updated_at ELSE facts.updated_at END
                """,
                (new_key, value, scope or "general", updated_at or ""),
            )
            await db.execute("DELETE FROM facts WHERE key=?", (old_key,))
        await db.commit()
        logger.info("[task_store] 已迁移 %d 条旧 person_profile facts 到 interlocutor", len(migrations))

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
            title, goal, priority, source,
            status=status, next_step=next_step, chain_id=chain_id,
            parent_task_id=parent_task_id, current_step=current_step,
            wait_kind=wait_kind, wait_key=wait_key, state_json=state_json,
            wait_json=wait_json, result_json=result_json,
            async_job_id=async_job_id, model_tier=model_tier, extras=extras,
        )

    async def get_task_by_id(self, task_id: int) -> Optional[Task]:
        return await self._tasks.get_task_by_id(task_id)

    async def list_runnable_tasks(self, limit: int = 20) -> list[Task]:
        return await self._tasks.list_runnable_tasks(limit)

    async def list_open_tasks(
        self,
        limit: int = 50,
        *,
        statuses: tuple[str, ...] | list[str] | None = None,
    ) -> list[Task]:
        normalized_statuses = tuple(
            str(status or "").strip()
            for status in (statuses or OPEN_TASK_STATUSES)
            if str(status or "").strip()
        )
        if limit <= 0 or not normalized_statuses:
            return []
        placeholders = ",".join("?" for _ in normalized_statuses)
        sql = (
            "SELECT id, title, status, priority, created_at, data "
            f"FROM tasks WHERE status IN ({placeholders}) "
            "ORDER BY "
            "CASE status "
            "    WHEN 'in_progress' THEN 0 "
            "    WHEN 'resumed' THEN 1 "
            "    WHEN 'ready' THEN 2 "
            "    WHEN 'pending' THEN 3 "
            "    WHEN 'waiting' THEN 4 "
            "    ELSE 5 "
            "END, "
            "CASE priority "
            "    WHEN 'critical' THEN 0 "
            "    WHEN 'high' THEN 1 "
            "    WHEN 'normal' THEN 2 "
            "    ELSE 3 "
            "END, "
            "id LIMIT ?"
        )
        args: tuple[Any, ...] = (*normalized_statuses, int(limit))
        async with self._db.execute(sql, args) as cur:
            rows = await cur.fetchall()
        return [Task.from_row(row) for row in rows]

    async def find_similar_open_tasks(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = TASK_SIMILARITY_CONTEXT_SCORE,
        exclude_task_ids: list[int] | tuple[int, ...] | set[int] | None = None,
        statuses: tuple[str, ...] | list[str] | None = None,
    ) -> list[tuple[Task, float]]:
        from store.task.schema import _task_similarity_score as score_fn

        query_text = build_task_similarity_query(query)
        if limit <= 0 or not query_text:
            return []

        exclude_ids = {
            int(task_id)
            for task_id in (exclude_task_ids or [])
            if str(task_id).strip()
        }
        scan_limit = min(
            _TASK_SIMILARITY_SCAN_LIMIT,
            max(int(limit) * 6, 24),
        )
        candidates = await self.list_open_tasks(limit=scan_limit, statuses=statuses)
        scored: list[tuple[Task, float]] = []
        for task in candidates:
            if task.id in exclude_ids:
                continue
            score = score_fn(query_text, task)
            if score < float(min_score):
                continue
            scored.append((task, score))

        scored.sort(
            key=lambda item: (
                -item[1],
                _TASK_STATUS_RANK.get(item[0].status, 99),
                _TASK_PRIORITY_RANK.get(item[0].priority, 99),
                item[0].id,
            )
        )
        return scored[: int(limit)]

    async def get_active(self) -> Optional[Task]:
        return await self._tasks.get_active()

    async def list_tasks(self, status: Optional[str] = None, limit: int = 50) -> list[Task]:
        return await self._tasks.list_tasks(status=status, limit=limit)

    async def update_status(
        self,
        task_id: int,
        status: str,
        next_step: str | None = None,
        *,
        current_step: str | None = None,
        model_tier: str | None = None,
        result_json: dict[str, Any] | None = None,
    ) -> None:
        await self._tasks.update_status(
            task_id, status, next_step,
            current_step=current_step, model_tier=model_tier, result_json=result_json,
        )

    async def mark_waiting(
        self,
        task_id: int,
        *,
        wait_kind: str,
        wait_key: str = "",
        wait_json: dict[str, Any] | None = None,
        current_step: str | None = None,
        next_step: str | None = None,
        result_json: dict[str, Any] | None = None,
    ) -> None:
        await self._tasks.mark_waiting(
            task_id, wait_kind=wait_kind, wait_key=wait_key, wait_json=wait_json,
            current_step=current_step, next_step=next_step, result_json=result_json,
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
            task_id, status=status, current_step=current_step,
            next_step=next_step, result_json=result_json,
        )

    async def update_task_data(self, task_id: int, extra_dict: dict[str, Any]) -> None:
        await self._tasks.update_task_data(task_id, extra_dict)

    async def pop_task_inbox(self, task_id: int) -> list[str]:
        return await self._tasks.pop_task_inbox(task_id)

    async def update_task_result(self, task_id: int, result_json: dict[str, Any]) -> None:
        await self._tasks.update_task_result(task_id, result_json)

    async def sync_task_progress(
        self,
        task_id: int,
        *,
        current_step: str | None = None,
        next_step: str | None = None,
    ) -> None:
        await self._tasks.sync_task_progress(task_id, current_step=current_step, next_step=next_step)

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
            task_id=task_id, run_type=run_type, worker_type=worker_type, status=status,
            input_json=input_json, output_json=output_json, log_text=log_text,
            error_text=error_text, tool_name=tool_name, session_id=session_id,
            model_tier=model_tier, progress=progress, extras=extras,
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
        task_id: int | None = None,
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
            run_id, task_id=task_id, status=status, output_json=output_json,
            log_text=log_text, error_text=error_text, session_id=session_id,
            model_tier=model_tier, progress=progress, extras=extras,
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
            reflection_id=reflection_id, target_kind=target_kind, trigger=trigger,
            loop_level=loop_level, diagnosis=diagnosis, proposal=proposal,
            verification_plan=verification_plan, decision=decision,
            task_id=task_id, run_id=run_id, tool_name=tool_name, extras=extras,
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

    async def record_failure(self, kind: str, summary: str, context: str = "", task_id: str = "") -> None:
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
        return await self._facts.get_fact(key)

    async def list_facts(self, prefix: str = "", limit: int = 100) -> list[tuple[str, str]]:
        return await self._facts.list_facts(prefix, limit)

    async def delete_fact(self, key: str) -> None:
        await self._facts.delete_fact(key)

    # ── 调度信号（cron 机制）──────────────────────────────────────────────

    async def add_signal(self, title: str, run_at: str, repeat_secs: int = 0, payload: dict[str, Any] | None = None) -> int:
        return await self._signals.add_signal(title, run_at, repeat_secs, payload)

    async def due_signals(self) -> list[dict[str, Any]]:
        return await self._signals.due_signals()

    async def ack_signal(self, signal_id: int) -> None:
        await self._signals.ack_signal(signal_id)

    async def list_signals(self, limit: int = 30, include_done: bool = False) -> list[dict[str, Any]]:
        return await self._signals.list_signals(limit, include_done)

    async def get_signal(self, signal_id: int) -> dict[str, Any] | None:
        return await self._signals.get_signal(signal_id)

    async def cancel_signal(self, signal_id: int) -> None:
        await self._signals.cancel_signal(signal_id)

    # ── 对话消息（chat IPC）────────────────────────────────────────────────

    async def add_chat_message(self, role: str, content: str, chat_id: str = "") -> int:
        return await self._chat.add_message(role, content, chat_id=chat_id)

    async def has_pending_chat_message(self) -> bool:
        return await self._chat.has_pending_message()

    async def pop_pending_chat_message(self) -> Optional[dict[str, Any]]:
        return await self._chat.pop_pending_message()

    async def drain_pending_for_chat(self, chat_id: str, after_id: int) -> list[dict[str, Any]]:
        return await self._chat.drain_pending_for_chat(chat_id, after_id)

    async def mark_chat_messages_processed(self, message_ids: list[int] | tuple[int, ...]) -> None:
        await self._chat.mark_messages_processed(message_ids)

    async def release_chat_messages(self, message_ids: list[int] | tuple[int, ...]) -> None:
        await self._chat.release_messages(message_ids)

    async def get_chat_messages_since(self, since_id: int = 0, chat_id: str = "") -> list[dict[str, Any]]:
        return await self._chat.get_messages_since(since_id, chat_id=chat_id)

    async def get_recent_chat_messages(self, limit: int = 6, chat_id: str = "") -> list[dict[str, Any]]:
        return await self._chat.get_recent_messages(limit, chat_id=chat_id)

    async def reset_in_progress_tasks(self) -> int:
        result = await self._db.execute(
            "UPDATE tasks SET status='pending' WHERE status='in_progress'"
        )
        await self._db.commit()
        return result.rowcount if result else 0


__all__ = [
    "TaskStore",
    "Task",
    "Failure",
    "Run",
    "MetaReflection",
    "ChatMessageStore",
    "FactStore",
    "FailureStore",
    "IngressStore",
    "MetaReflectionStore",
    "RunStore",
    "SignalStore",
    "TaskStateStore",
    "sanitize_chat_content",
    "build_task_data",
    "build_task_insert",
    "build_fact_upsert",
    "build_task_run_result_patch",
    "build_task_similarity_query",
    "OPEN_TASK_STATUSES",
    "RUNNABLE_TASK_STATUSES",
    "TASK_DUPLICATE_REUSE_SCORE",
    "TASK_SIMILARITY_CONTEXT_SCORE",
]
