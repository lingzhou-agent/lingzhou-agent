from __future__ import annotations

import json
from typing import Any, Callable

import aiosqlite

from ._base import BaseAsyncStore


class SignalStore(BaseAsyncStore):

    async def add_signal(
        self,
        title: str,
        run_at: str,
        repeat_secs: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> int:
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        async with self._db.execute(
            "INSERT INTO signals (title, run_at, repeat_secs, payload) VALUES (?,?,?,?)",
            (title, run_at, repeat_secs, payload_json),
        ) as cur:
            new_id = cur.lastrowid
        await self._db.commit()
        return int(new_id or 0)

    async def due_signals(self) -> list[dict[str, Any]]:
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
        async with self._db.execute(
            "SELECT repeat_secs, run_at FROM signals WHERE id=?", (signal_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return
        repeat_secs: Any = row[0]
        if repeat_secs and repeat_secs > 0:
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
        where = "" if include_done else "WHERE status='pending'"
        rows: list[dict[str, Any]] = []
        async with self._db.execute(
            f"SELECT id, title, run_at, repeat_secs, status, payload FROM signals {where} ORDER BY run_at LIMIT ?",
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

    async def get_signal(self, signal_id: int) -> dict[str, Any] | None:
        async with self._db.execute(
            "SELECT id, title, run_at, repeat_secs, status, payload FROM signals WHERE id=?",
            (signal_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[5] or "{}")
        except Exception:
            payload = {}
        return {
            "id": row[0],
            "title": row[1],
            "run_at": row[2],
            "repeat_secs": row[3],
            "status": row[4],
            "payload": payload,
        }

    async def cancel_signal(self, signal_id: int) -> None:
        await self._db.execute(
            "UPDATE signals SET status='cancelled' WHERE id=?", (signal_id,)
        )
        await self._db.commit()
