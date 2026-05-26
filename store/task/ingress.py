from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .chat import build_chat_message_insert
from .fact import build_fact_upsert
from .state import build_task_data, build_task_insert


class IngressStore:
    """同步入口仓储：给 channel/webhook 这类线程侧入口统一写入 runtime DB。"""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path).expanduser()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def add_chat_message(
        self,
        role: str,
        content: str,
        *,
        chat_id: str = "",
        status: str = "pending",
    ) -> int:
        insert_args = build_chat_message_insert(
            role,
            content,
            chat_id=chat_id,
            status=status,
        )
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO chat_messages(role, content, session_id, status) VALUES (?,?,?,?)",
                insert_args,
            )
            return int(cur.lastrowid or 0)

    def set_fact(self, key: str, value: str, *, scope: str = "general") -> None:
        sql, params = build_fact_upsert(key, value, scope=scope)
        with self._connect() as conn:
            conn.execute(sql, params)

    def get_fact(self, key: str) -> tuple[str, bool]:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        if row is None:
            return "", False
        return str(row[0] or ""), True

    def list_tables(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def ingest_user_message(
        self,
        content: str,
        *,
        chat_id: str,
        facts: dict[str, str | tuple[str, str]] | None = None,
    ) -> int:
        insert_args = build_chat_message_insert(
            "user",
            content,
            chat_id=chat_id,
            status="pending",
        )
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO chat_messages(role, content, session_id, status) VALUES (?,?,?,?)",
                insert_args,
            )
            message_id = int(cur.lastrowid or 0)
            for key, raw_value in (facts or {}).items():
                if isinstance(raw_value, tuple):
                    value, scope = raw_value
                else:
                    value, scope = str(raw_value), "general"
                sql, params = build_fact_upsert(key, value, scope=scope)
                conn.execute(sql, params)
            return message_id

    def list_pending_assistant_messages(
        self,
        *,
        chat_prefix: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        params: tuple[Any, ...]
        if chat_prefix:
            sql = (
                "SELECT id, content, session_id AS chat_id, created_at FROM chat_messages "
                "WHERE role='assistant' AND session_id LIKE ? "
                "AND status IN ('pending','processed') "
                "ORDER BY id ASC LIMIT ?"
            )
            params = (f"{chat_prefix}%", limit)
        else:
            sql = (
                "SELECT id, content, session_id AS chat_id, created_at FROM chat_messages "
                "WHERE role='assistant' AND status IN ('pending','processed') "
                "ORDER BY id ASC LIMIT ?"
            )
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": int(row["id"]),
                "content": str(row["content"] or ""),
                "chat_id": str(row["chat_id"] or ""),
                "created_at": str(row["created_at"] or ""),
            }
            for row in rows
        ]

    def mark_chat_message_delivered(self, message_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_messages SET status='delivered' WHERE id=?",
                (int(message_id),),
            )

    def add_task(
        self,
        title: str,
        *,
        goal: str = "",
        priority: str = "normal",
        source: str = "external",
        status: str = "pending",
        next_step: str = "",
        extras: dict[str, Any] | None = None,
    ) -> int:
        data = build_task_data(
            goal=goal,
            source=source,
            next_step=next_step,
            extras=extras,
        )
        insert_args = build_task_insert(
            title,
            status=status,
            priority=priority,
            data=data,
        )
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO tasks (title, status, priority, data) VALUES (?,?,?,?)",
                insert_args,
            )
            return int(cur.lastrowid or 0)
