from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

import aiosqlite

from ._base import BaseAsyncStore
from .models import Task

logger = logging.getLogger(__name__)

_UNSET = object()


def build_task_data(
    *,
    goal: str = "",
    source: str = "external",
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
) -> dict[str, Any]:
    data = {
        "goal": goal,
        "source": source,
        "next_step": next_step,
        "chain_id": chain_id,
        "parent_task_id": parent_task_id,
        "current_step": current_step,
        "wait_kind": wait_kind,
        "wait_key": wait_key,
        "state_json": state_json or {},
        "wait_json": wait_json or {},
        "result_json": result_json or {},
        "async_job_id": async_job_id,
        "model_tier": model_tier,
    }
    if extras:
        data.update(extras)
    return data


def build_task_insert(
    title: str,
    *,
    status: str,
    priority: str,
    data: dict[str, Any],
) -> tuple[str, str, str, str]:
    return (
        title.strip(),
        str(status or "pending"),
        str(priority or "normal"),
        json.dumps(data, ensure_ascii=False),
    )


class TaskStateStore(BaseAsyncStore):

    async def _save_task(self, task: Task) -> None:
        await self._db.execute(
            "UPDATE tasks SET status=?, data=? WHERE id=?",
            (task.status, task.to_data_json(), task.id),
        )
        await self._db.commit()

    async def _patch_task(
        self,
        task_id: int,
        *,
        status: Any = _UNSET,
        next_step: Any = _UNSET,
        current_step: Any = _UNSET,
        model_tier: Any = _UNSET,
        wait_kind: Any = _UNSET,
        wait_key: Any = _UNSET,
        wait_json: Any = _UNSET,
        result_json: Any = _UNSET,
        merge_result_json: bool = False,
        extras: dict[str, Any] | None = None,
    ) -> bool:
        task = await self.get_task_by_id(task_id)
        if not task:
            return False
        if status is not _UNSET:
            task.status = str(status or "")
        if next_step is not _UNSET:
            task.next_step = str(next_step or "")
        if current_step is not _UNSET:
            task.current_step = str(current_step or "")
        if model_tier is not _UNSET:
            task.model_tier = str(model_tier or "")
        if wait_kind is not _UNSET:
            task.wait_kind = str(wait_kind or "")
        if wait_key is not _UNSET:
            task.wait_key = str(wait_key or "")
        if wait_json is not _UNSET:
            task.wait_json = dict(wait_json or {})
        if result_json is not _UNSET:
            incoming = dict(result_json or {})
            if merge_result_json:
                merged = dict(task.result_json or {})
                merged.update(incoming)
                task.result_json = merged
            else:
                task.result_json = incoming
        if extras:
            task.extras.update(extras)
        await self._save_task(task)
        return True

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
        data = build_task_data(
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
            extras=extras,
        )
        insert_args = build_task_insert(
            title,
            status=status,
            priority=priority,
            data=data,
        )
        async with self._db.execute(
            "INSERT INTO tasks (title, status, priority, data) VALUES (?,?,?,?)",
            insert_args,
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

    async def list_runnable_tasks(self, limit: int = 20) -> list[Task]:
        async with self._db.execute(
            """SELECT id, title, status, priority, created_at, data
               FROM tasks
               WHERE status IN ('pending','ready','in_progress','resumed')
               ORDER BY
                CASE status
                        WHEN 'in_progress' THEN 0
                        WHEN 'resumed' THEN 1
                        WHEN 'ready' THEN 2
                        WHEN 'pending' THEN 3
                        ELSE 4
                END,
                CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                            WHEN 'normal' THEN 2 ELSE 3 END,
                id
               LIMIT ?""",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [Task.from_row(row) for row in rows]

    async def get_active(self) -> Optional[Task]:
        runnable = await self.list_runnable_tasks(limit=1)
        return runnable[0] if runnable else None

    async def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[Task]:
        if status:
            sql = (
                "SELECT id, title, status, priority, created_at, data "
                "FROM tasks WHERE status=? ORDER BY id LIMIT ?"
            )
            args: tuple[Any, ...] = (status, limit)
        else:
            sql = (
                "SELECT id, title, status, priority, created_at, data "
                "FROM tasks ORDER BY id LIMIT ?"
            )
            args = (limit,)
        async with self._db.execute(sql, args) as cur:
            rows = await cur.fetchall()
        return [Task.from_row(row) for row in rows]

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
        await self._patch_task(
            task_id,
            status=status,
            next_step=next_step if next_step is not None else _UNSET,
            current_step=current_step if current_step is not None else _UNSET,
            model_tier=model_tier if model_tier is not None else _UNSET,
            result_json=result_json if result_json is not None else _UNSET,
            merge_result_json=True,
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
        await self._patch_task(
            task_id,
            status="waiting",
            wait_kind=wait_kind,
            wait_key=wait_key,
            wait_json=wait_json or {},
            current_step=current_step if current_step is not None else _UNSET,
            next_step=next_step if next_step is not None else _UNSET,
            result_json=result_json if result_json is not None else _UNSET,
            merge_result_json=True,
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
        await self._patch_task(
            task_id,
            status=status,
            wait_kind="",
            wait_key="",
            wait_json={},
            current_step=current_step if current_step is not None else _UNSET,
            next_step=next_step if next_step is not None else _UNSET,
            result_json=result_json if result_json is not None else _UNSET,
            merge_result_json=True,
        )

    async def update_task_data(self, task_id: int, extra_dict: dict[str, Any]) -> None:
        protected_keys = {"goal", "source", "next_step", "result_json"}
        ignored = [key for key in extra_dict.keys() if key in protected_keys]
        if ignored:
            logger.debug("update_task_data ignored protected task fields: %s", ",".join(sorted(ignored)))

        extras = {
            key: value
            for key, value in extra_dict.items()
            if key not in protected_keys and key not in {"current_step", "model_tier"}
        }
        await self._patch_task(
            task_id,
            current_step=(str(extra_dict.get("current_step") or "") if "current_step" in extra_dict else _UNSET),
            model_tier=(str(extra_dict.get("model_tier") or "") if "model_tier" in extra_dict else _UNSET),
            extras=extras,
        )

    async def pop_task_inbox(self, task_id: int) -> list[str]:
        task = await self.get_task_by_id(task_id)
        if not task:
            return []

        raw_messages = task.extras.get("inbox_messages")
        if not isinstance(raw_messages, list):
            return []

        messages = [str(item).strip() for item in raw_messages if str(item).strip()]
        if not messages:
            if raw_messages != []:
                await self._patch_task(task_id, extras={"inbox_messages": []})
            return []

        await self._patch_task(task_id, extras={"inbox_messages": []})
        return messages

    async def update_task_result(self, task_id: int, result_json: dict[str, Any]) -> None:
        await self._patch_task(task_id, result_json=result_json, merge_result_json=True)

    async def sync_task_progress(
        self,
        task_id: int,
        *,
        current_step: str | None = None,
        next_step: str | None = None,
    ) -> None:
        await self._patch_task(
            task_id,
            current_step=current_step if current_step is not None else _UNSET,
            next_step=next_step if next_step is not None else _UNSET,
        )

    async def enqueue_if_absent(
        self,
        title: str,
        goal: str = "",
        priority: str = "normal",
        source: str = "internal",
    ) -> bool:
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
