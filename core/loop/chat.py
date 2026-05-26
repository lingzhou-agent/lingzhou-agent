"""core/loop/chat.py - loop 的 chat 绑定、回复落库与交互入口。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from store.task import Task
from memory.working import WMItem
from .dispatcher import TickJob

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
    if chat_id is not None:
        resolved_chat_id = str(chat_id or "").strip()
        return resolved_chat_id

    if active_task is not None:
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

    last_chat_id, last_chat_found = await loop._task_store.get_fact("chat:last_chat_id")
    if last_chat_found and last_chat_id.strip():
        return last_chat_id.strip()
    return None


async def _process_pending_chat_turn(loop: Any, cycle: int) -> tuple[int, bool]:
    original_cycle = cycle
    dispatcher = getattr(loop, "_tick_dispatcher", None)
    if dispatcher is not None and dispatcher.enabled:
        can_accept = getattr(dispatcher, "can_accept", None)
        if callable(can_accept) and not can_accept():
            _log.debug("[chat] tick queue saturated, defer pending chat pickup")
            return cycle, False

    chat_message = await loop._task_store.pop_pending_chat_message()
    if not chat_message:
        return cycle, False

    user_message = str(chat_message.get("content") or "")
    chat_id = str(chat_message.get("chat_id") or "")
    msg_id: int = chat_message.get("id") or 0
    reserved_message_ids: list[int] = [msg_id] if msg_id else []

    # drain 同一会话紧随而来的消息，合并为同一 LLM 上下文轮次
    if chat_id:
        # wechat 通道：用户经常紧接文字发图片，各为独立 iLink 消息。
        # 图片消息需要下载解密才能写入 DB，可能晚于文字消息被 asyncio 消费。
        # 短暂等待让同批次图片消息有机会写入 DB 再 drain。
        if chat_id.startswith("wechat:"):
            delay = loop._cfg.loop.wechat_coalesce_delay
            if delay > 0:
                await asyncio.sleep(delay)
        follow_ups = await loop._task_store.drain_pending_for_chat(chat_id, after_id=msg_id)
        if follow_ups:
            reserved_message_ids.extend(int(m.get("id") or 0) for m in follow_ups if int(m.get("id") or 0) > 0)
            extra = "\n".join(m["content"] for m in follow_ups)
            user_message = f"{user_message}\n{extra}".strip()
            _log.debug("[chat] merged %d follow-up message(s) into turn (ids=%s)",
                       len(follow_ups), [m["id"] for m in follow_ups])

    if dispatcher is not None and dispatcher.enabled:
        active_task = await loop._task_store.get_active()
        dispatch_cycle = await loop._next_dispatch_cycle()
        chain_key = loop._resolve_tick_chain_key(active_task=active_task, chat_id=chat_id, source="chat")
        job = TickJob(
            cycle=dispatch_cycle,
            chain_key=chain_key,
            user_message=user_message,
            chat_id=chat_id,
            chat_message_ids=tuple(reserved_message_ids),
            source="chat",
        )
        accepted = await loop._tick_dispatcher.enqueue(job)
        if not accepted:
            await loop._task_store.release_chat_messages(reserved_message_ids)
            _log.debug("[chat] tick queue full, defer chat retry chat_id=%s", chat_id)
            return original_cycle, True
        _log.info("[chat] user › %s", user_message)
        return dispatch_cycle, True

    cycle += 1
    _log.info("[chat] user › %s", user_message)
    try:
        reply = await loop._tick(
            cycle,
            user_message=user_message,
            chat_id=chat_id,
        )
    except Exception:
        await loop._task_store.release_chat_messages(reserved_message_ids)
        raise

    await loop._task_store.mark_chat_messages_processed(reserved_message_ids)
    if reply:
        reply = _strip_memory_context(reply)
    _log.info("[chat] assistant › %s", reply or "")
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
