from __future__ import annotations

import re
from typing import Any, Callable, Optional

import aiosqlite

from ._base import BaseAsyncStore

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CHAT_ZERO_WIDTH_CHARS = {"\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"}
_CJK_NEIGHBOR_RE = re.compile(
    r"(?<=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef])"
    r"[ \t\u00a0\u3000]+"
    r"(?=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef])"
)


def sanitize_chat_content(content: str) -> str:
    text = str(content or "")
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_chars: list[str] = []
    for ch in text:
        if ch in _CHAT_ZERO_WIDTH_CHARS or ch == "\ufffd":
            continue
        if ord(ch) < 32 and ch not in {"\n", "\t"}:
            continue
        cleaned_chars.append(ch)
    text = "".join(cleaned_chars)
    text = _CJK_NEIGHBOR_RE.sub("", text)
    return text.strip()


def build_chat_message_insert(
    role: str,
    content: str,
    *,
    chat_id: str = "",
    status: str = "pending",
) -> tuple[str, str, str, str]:
    cleaned = sanitize_chat_content(content)
    resolved_chat_id = str(chat_id or "")
    return str(role), cleaned, resolved_chat_id, str(status or "pending")


class ChatMessageStore(BaseAsyncStore):

    async def add_message(
        self,
        role: str,
        content: str,
        chat_id: str = "",
    ) -> int:
        resolved_role, cleaned, resolved_chat_id, _ = build_chat_message_insert(
            role,
            content,
            chat_id=chat_id,
        )
        async with self._db.execute(
            "INSERT INTO chat_messages(role, content, session_id) VALUES (?,?,?)",
            (resolved_role, cleaned, resolved_chat_id),
        ) as cur:
            row_id: int = cur.lastrowid or 0
        await self._db.commit()
        return row_id

    async def has_pending_message(self) -> bool:
        async with self._db.execute(
            "SELECT 1 FROM chat_messages WHERE role='user' AND status='pending' LIMIT 1"
        ) as cur:
            return await cur.fetchone() is not None

    async def pop_pending_message(self) -> Optional[dict[str, Any]]:
        async with self._db.execute(
            "SELECT id, content, session_id FROM chat_messages "
            "WHERE role='user' AND status='pending' ORDER BY id LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        mid, content, chat_id = row
        async with self._db.execute(
            "UPDATE chat_messages SET status='processing' WHERE id=? AND status='pending'",
            (mid,),
        ) as cur:
            updated = cur.rowcount or 0
        if updated <= 0:
            await self._db.rollback()
            return None
        await self._db.commit()
        return {"id": mid, "content": content, "chat_id": chat_id}

    async def drain_pending_for_chat(
        self,
        chat_id: str,
        after_id: int,
    ) -> list[dict[str, Any]]:
        async with self._db.execute(
            "SELECT id, content FROM chat_messages "
            "WHERE role='user' AND status='pending' AND session_id=? AND id>? ORDER BY id",
            (chat_id, after_id),
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        placeholders = ",".join("?" * len(ids))
        async with self._db.execute(
            f"UPDATE chat_messages SET status='processing' WHERE status='pending' AND id IN ({placeholders})",
            ids,
        ) as cur:
            updated = cur.rowcount or 0
        if updated <= 0:
            await self._db.rollback()
            return []
        await self._db.commit()
        return [{"id": r[0], "content": r[1]} for r in rows]

    async def mark_messages_processed(self, message_ids: list[int] | tuple[int, ...]) -> None:
        ids = [int(mid) for mid in message_ids if int(mid) > 0]
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        await self._db.execute(
            f"UPDATE chat_messages SET status='processed' WHERE status='processing' AND id IN ({placeholders})",
            ids,
        )
        await self._db.commit()

    async def release_messages(self, message_ids: list[int] | tuple[int, ...]) -> None:
        ids = [int(mid) for mid in message_ids if int(mid) > 0]
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        await self._db.execute(
            f"UPDATE chat_messages SET status='pending' WHERE status='processing' AND id IN ({placeholders})",
            ids,
        )
        await self._db.commit()

    async def get_messages_since(
        self,
        since_id: int = 0,
        chat_id: str = "",
    ) -> list[dict[str, Any]]:
        resolved_chat_id = str(chat_id or "")
        if resolved_chat_id:
            sql = (
                "SELECT id, role, content, created_at FROM chat_messages "
                "WHERE id > ? AND session_id = ? ORDER BY id"
            )
            params: tuple[Any, ...] = (since_id, resolved_chat_id)
        else:
            sql = (
                "SELECT id, role, content, created_at FROM chat_messages "
                "WHERE id > ? ORDER BY id"
            )
            params = (since_id,)
        async with self._db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [{"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]} for r in rows]

    async def get_recent_messages(
        self,
        limit: int = 6,
        chat_id: str = "",
    ) -> list[dict[str, Any]]:
        resolved_chat_id = str(chat_id or "")
        if resolved_chat_id:
            sql = (
                "SELECT id, role, content, created_at FROM chat_messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?"
            )
            params: tuple[Any, ...] = (resolved_chat_id, limit)
        else:
            sql = (
                "SELECT id, role, content, created_at FROM chat_messages "
                "ORDER BY id DESC LIMIT ?"
            )
            params = (limit,)
        async with self._db.execute(sql, params) as cur:
            rows = list(await cur.fetchall())
        rows = rows[::-1]
        return [{"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]} for r in rows]
