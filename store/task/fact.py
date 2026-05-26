from __future__ import annotations

from typing import Callable

import aiosqlite


FACT_UPSERT_SQL = (
    "INSERT INTO facts (key, value, scope, updated_at) VALUES (?,?,?,datetime('now')) "
    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
    "scope=excluded.scope, updated_at=excluded.updated_at"
)


def build_fact_upsert(key: str, value: str, *, scope: str = "general") -> tuple[str, tuple[str, str, str]]:
    return FACT_UPSERT_SQL, (str(key), str(value), str(scope or "general"))


class FactStore:
    def __init__(self, db_getter: Callable[[], aiosqlite.Connection]) -> None:
        self._db_getter = db_getter

    @property
    def _db(self) -> aiosqlite.Connection:
        return self._db_getter()

    async def set_fact(self, key: str, value: str, scope: str = "general") -> None:
        sql, params = build_fact_upsert(key, value, scope=scope)
        await self._db.execute(sql, params)
        await self._db.commit()

    async def get_fact(self, key: str) -> tuple[str, bool]:
        async with self._db.execute(
            "SELECT value FROM facts WHERE key=?", (key,)
        ) as cur:
            row = await cur.fetchone()
        if row:
            return row[0], True
        return "", False

    async def list_facts(self, prefix: str = "", limit: int = 100) -> list[tuple[str, str]]:
        if prefix:
            async with self._db.execute(
                "SELECT key, value FROM facts WHERE key LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (f"{prefix}%", limit),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with self._db.execute(
                "SELECT key, value FROM facts ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        return [(str(key), str(value)) for key, value in rows]

    async def delete_fact(self, key: str) -> None:
        await self._db.execute("DELETE FROM facts WHERE key=?", (key,))
        await self._db.commit()
