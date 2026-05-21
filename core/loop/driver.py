"""core/loop/driver.py - loop 生命周期调度与事件驱动等待。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

_log = logging.getLogger("lingzhou.loop")


async def _run_cycle_impl(loop: Any, cycle: int) -> int:
    cycle, handled_chat = await loop._process_pending_chat_turn(cycle)
    if not handled_chat:
        cycle += 1
        await loop._tick(cycle)
    return cycle


async def _wait_after_cycle_impl(loop: Any) -> None:
    cfg = loop._cfg
    after_task = await loop._task_store.get_active()
    if loop._last_decision == "act" and after_task is not None:
        min_wait = cfg.loop.idle_with_task_bounds[0] / 1000.0 if cfg.loop.idle_with_task_bounds else cfg.loop.min_act_gap / 1000.0
        act_gap = max(float(min_wait), float(cfg.loop.min_act_gap) / 1000.0)
        await _wait_for_event_impl(loop, act_gap, after_task)
    else:
        # 等待间隔决策树：
        #   ① LLM 上轮主动要求 next_idle_gap_secs → 优先尊重 LLM 意图
        #   ② 有活跃 task（上轮 decision≠act，或 act 后 task 仍在）→ active_idle_gap（较短，保持响应）
        #   ③ bootstrap 未完成 → 同 ② 缩短间隔（避免引导阶段空等 max_idle_gap）
        #   ④ 真正空闲（无 task、无 bootstrap）→ max_idle_gap（节省 CPU/计费）
        if loop._pending_idle_gap is not None:
            gap = loop._pending_idle_gap                   # ① LLM 主动调度
        elif after_task is not None:
            gap = cfg.loop.active_idle_gap / 1000.0        # ② 有活跃任务
        elif getattr(loop, '_bootstrap_mode', 'none') == "full":
            # bootstrap 未完成时，等同于有隐式未完成工作：缩短轮询间隔提升响应度
            gap = cfg.loop.active_idle_gap / 1000.0        # ③ bootstrap 阶段
        else:
            gap = cfg.loop.max_idle_gap / 1000.0           # ④ 完全空闲
        await _wait_for_event_impl(loop, gap, after_task)
    await loop._maybe_hot_reload_provider()


async def _wait_for_event_impl(loop: Any, max_wait: float, before_task: Any) -> None:
    """事件驱动等待: chat 消息、task 状态变化、超时任一发生即唤醒。"""
    cfg = loop._cfg
    poll = max(cfg.loop.wake_poll_interval / 1000.0, 0.05)  # 最小 50ms，防止 wake_poll_interval=0 导致紧密轮询
    before_sig = (
        before_task.id if before_task else None,
        before_task.status if before_task else None,
    )
    event_loop = asyncio.get_running_loop()
    deadline = event_loop.time() + max_wait
    while True:
        remaining = deadline - event_loop.time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(poll, remaining))
        if await loop._task_store.has_pending_chat_message():
            _log.debug("[wake] chat 消息到达,提前唤醒")
            break
        if cfg.loop.wake_on_task_change:
            now = await loop._task_store.get_active()
            now_sig = (now.id if now else None, now.status if now else None)
            if now_sig != before_sig:
                _log.info("[wake] task 状态变化 %s → %s", before_sig, now_sig)
                break
