from __future__ import annotations

import json
from typing import Callable

import aiosqlite

from ._base import BaseAsyncStore
from .models import Failure


class FailureStore(BaseAsyncStore):

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
        return [Failure.from_row(row) for row in rows]

    async def list_failures_for_task(self, task_id: str, limit: int = 20) -> list[Failure]:
        async with self._db.execute(
            "SELECT id, kind, dismissed, created_at, data FROM failures "
            "WHERE (json_extract(data,'$.task_id')=? OR json_extract(data,'$.task_id')='') AND dismissed=0 "
            "ORDER BY id DESC LIMIT ?",
            (task_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [Failure.from_row(row) for row in rows]

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
