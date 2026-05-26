from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

import aiosqlite

if TYPE_CHECKING:
    from store.task.models import MetaReflection


class MetaReflectionStore:
    def __init__(self, db_getter: Callable[[], aiosqlite.Connection]) -> None:
        self._db_getter = db_getter

    @property
    def _db(self) -> aiosqlite.Connection:
        return self._db_getter()

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
        data = {
            "task_id": task_id,
            "run_id": run_id,
            "tool_name": tool_name,
        }
        if extras:
            data.update(extras)
        await self._db.execute(
            "INSERT OR REPLACE INTO meta_reflections (id, target_kind, trigger, loop_level, diagnosis, proposal, verification_plan, decision, data) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                reflection_id,
                target_kind,
                trigger,
                loop_level,
                diagnosis,
                proposal,
                verification_plan,
                decision,
                json.dumps(data, ensure_ascii=False),
            ),
        )
        await self._db.commit()

    async def list_meta_reflections(
        self,
        limit: int = 20,
        loop_level: str | None = None,
    ) -> list["MetaReflection"]:
        from store.task.models import MetaReflection

        if loop_level:
            async with self._db.execute(
                "SELECT id, target_kind, trigger, loop_level, diagnosis, proposal, verification_plan, decision, created_at, data FROM meta_reflections WHERE loop_level=? ORDER BY created_at ASC, id ASC LIMIT ?",
                (loop_level, limit),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with self._db.execute(
                "SELECT id, target_kind, trigger, loop_level, diagnosis, proposal, verification_plan, decision, created_at, data FROM meta_reflections ORDER BY created_at ASC, id ASC LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        return [MetaReflection.from_row(row) for row in rows]
