"""core/loop/chat.py - loop 的 chat 绑定、回复落库与交互入口。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from memory.task_store import Task
from memory.working import WMItem

from .logging import _strip_memory_context

_log = logging.getLogger("lingzhou.loop")


async def _bind_chat_id(
    loop: Any,
    active_task: Task | None,
    chat_id: str | None,
) -> None:
    resolved_chat_id = (chat_id or "").strip()
    if not resolved_chat_id:
        return
    await loop._task_store.set_fact("chat:last_chat_id", resolved_chat_id, scope="system")
    if active_task is not None:
        await loop._task_store.set_fact(
            f"task:{active_task.id}:chat_id",
            resolved_chat_id,
            scope="task",
        )


async def _resolve_reply_chat_id(
    loop: Any,
    active_task: Task | None,
    chat_id: str | None,
) -> str | None:
    resolved_chat_id = (chat_id or "").strip()
    if resolved_chat_id:
        return resolved_chat_id
    if active_task is None:
        return None

    task_source = str(getattr(active_task, "source", "") or "").strip()
    if task_source.startswith(("wechat:", "chat:")):
        return task_source

    task_chat_id, task_chat_found = await loop._task_store.get_fact(f"task:{active_task.id}:chat_id")
    if task_chat_found and task_chat_id.strip():
        return task_chat_id.strip()

    legacy_chat_id, legacy_chat_found = await loop._task_store.get_fact(
        f"task:{active_task.id}:chat_session_id"
    )
    if legacy_chat_found and legacy_chat_id.strip():
        return legacy_chat_id.strip()
    return None


async def _process_pending_chat_turn(loop: Any, cycle: int) -> tuple[int, bool]:
    chat_message = await loop._task_store.pop_pending_chat_message()
    if not chat_message:
        return cycle, False

    cycle += 1
    user_message = str(chat_message.get("content") or "")
    chat_id = str(chat_message.get("chat_id") or "")
    msg_id: int = chat_message.get("id") or 0

    # 短暂等待，让同一会话紧随而来的附件消息（如图片下载完成后插入的消息）有时间落库，
    # 然后原子 drain — 合并为同一个 LLM 上下文轮次，避免"看不到图片"的竞态。
    if chat_id:
        await asyncio.sleep(0.4)
        follow_ups = await loop._task_store.drain_pending_for_chat(chat_id, after_id=msg_id)
        if follow_ups:
            extra = "\n".join(m["content"] for m in follow_ups)
            user_message = f"{user_message}\n{extra}".strip()
            _log.debug("[chat] merged %d follow-up message(s) into turn (ids=%s)",
                       len(follow_ups), [m["id"] for m in follow_ups])

    _log.info("[chat] user › %s", user_message[:200])
    reply = await loop._tick(
        cycle,
        user_message=user_message,
        chat_id=chat_id,
    )
    if reply:
        reply = _strip_memory_context(reply)
    _log.info("[chat] assistant › %s", (reply or "")[:200])
    if not reply:
        await loop._task_store.add_chat_message(
            "assistant",
            "(请求已处理,任务正在后台继续)",
            chat_id=chat_id,
        )
    return cycle, True


async def _tick_interact_impl(loop: Any, cycle: int, user_message: str) -> str:
    """interact 命令的单次入口:完整内环 + 返回 reply_to_user。"""
    if loop._conv_history:
        hist_text = "\n".join(
            f"[用户] {user}\n[灵舟] {assistant}" for user, assistant in loop._conv_history
        )
        loop._wm.add(WMItem(
            kind="conversation_history",
            content=f"[近期对话记录]\n{hist_text}",
            priority=loop._cfg.thresholds.wm_pri_history,
        ))
    reply = await loop._tick(cycle, user_message=user_message)
    if reply:
        reply = _strip_memory_context(reply)
        loop._conv_history.append((user_message, reply))
    return reply
