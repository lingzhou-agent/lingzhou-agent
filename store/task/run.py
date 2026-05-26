from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Callable, Optional

import aiosqlite

from ._base import BaseAsyncStore
from .models import Run


class RunStore(BaseAsyncStore):

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
        data = {
            "input_json": input_json or {},
            "output_json": output_json or {},
            "log_text": log_text,
            "error_text": error_text,
            "tool_name": tool_name,
            "session_id": session_id,
            "model_tier": model_tier,
            "progress": progress,
        }
        if extras:
            data.update(extras)
        now = datetime.now(UTC).isoformat()
        async with self._db.execute(
            "INSERT INTO runs (task_id, run_type, worker_type, status, created_at, started_at, data) VALUES (?,?,?,?,?,?,?)",
            (task_id, run_type, worker_type, status, now, now, json.dumps(data, ensure_ascii=False)),
        ) as cur:
            run_id: int = cur.lastrowid or 0
        await self._db.commit()
        return run_id

    async def get_run_by_id(self, run_id: int) -> Optional[Run]:
        async with self._db.execute(
            "SELECT id, task_id, run_type, worker_type, status, created_at, started_at, completed_at, data FROM runs WHERE id=?",
            (run_id,),
        ) as cur:
            row = await cur.fetchone()
        return Run.from_row(row) if row else None

    async def list_runs(
        self,
        *,
        task_id: int | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Run]:
        clauses: list[str] = []
        args: list[Any] = []
        if task_id is not None:
            clauses.append("task_id=?")
            args.append(task_id)
        if status:
            clauses.append("status=?")
            args.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        args.append(limit)
        async with self._db.execute(
            f"SELECT id, task_id, run_type, worker_type, status, created_at, started_at, completed_at, data FROM runs {where} ORDER BY id DESC LIMIT ?",
            tuple(args),
        ) as cur:
            rows = await cur.fetchall()
        return [Run.from_row(row) for row in rows]

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
        run = await self.get_run_by_id(run_id)
        if not run:
            return
        if task_id is not None:
            run.task_id = task_id
        if status:
            run.status = status
        if output_json is not None:
            run.output_json = output_json
        if log_text is not None:
            run.log_text = log_text
        if error_text is not None:
            run.error_text = error_text
        if session_id is not None:
            run.session_id = session_id
        if model_tier is not None:
            run.model_tier = model_tier
        if progress is not None:
            run.progress = progress
        if extras:
            run.extras.update(extras)
        if run.status in {"succeeded", "failed", "cancelled"} and not run.completed_at:
            run.completed_at = datetime.now(UTC).isoformat()
        await self._db.execute(
            "UPDATE runs SET task_id=?, status=?, completed_at=?, data=? WHERE id=?",
            (run.task_id, run.status, run.completed_at, run.to_data_json(), run_id),
        )
        await self._db.commit()
