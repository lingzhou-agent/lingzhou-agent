"""core/loop/continue_phase.py — tick 内续判工具循环。

从 tick.py 分离，避免 tick.py 超长。
由 tick() 在初始动作执行后、满足续判条件时调用。
"""
from __future__ import annotations

import logging
from typing import Any

from core.judgment import JudgmentOutput
from memory.working import WMItem

from .common import (
    _maybe_reconcile_bootstrap,
    _preferred_continue_tier,
    _resolve_thinking_override,
    _should_continue_within_tick,
    _tool_history_entry,
)
from .progress import action_key_param

_log = logging.getLogger("lingzhou.loop")


async def _run_continue_phase(
    *,
    loop: Any,
    ctx: Any,
    user_message: str,
    active_task: Any,
    cognitive_signals: Any,
    action: JudgmentOutput,
    result: Any,
    tool_history: list[dict[str, Any]],
) -> tuple[JudgmentOutput, Any]:
    """执行 continue 阶段的工具续判循环。

    返回 (final_action, final_result)，调用方用这两个值替换初始阶段的 action/result。
    tool_history 通过引用追加（in-place mutation）。
    """
    cfg = loop._cfg
    affect = {"valence": loop._emotion.valence, "arousal": loop._emotion.arousal}
    _continue_plan_streak = 0  # task.plan 连续调用计数（continue phase 专属防死锁）
    _wm_delta: list[dict] = []  # 本 tick continue 阶段新增的 WM 条目（不在 tool_history 里）

    for inner in range(cfg.loop.max_tool_rounds - 1):
        if await loop._task_store.has_pending_chat_message():
            _log.debug("[continue] chat 消息到达，中断工具循环 inner=%d", inner)
            break

        next_tier = _preferred_continue_tier(action, user_message=user_message) or ""
        continue_thinking = _resolve_thinking_override(
            cfg,
            user_message=user_message,
            model_strategy=action.model_strategy,
        )
        cont = await loop._judgment.decide_continue(
            tool_history,
            user_message=user_message,
            active_task=active_task,
            prefer_tier=next_tier or None,
            thinking_override=continue_thinking,
            routing_overrides=loop._pending_routing_overrides,
            wm_delta=_wm_delta or None,
        )

        if cont.decision == "act":
            tool_name = cont.chosen_action_id or ""
            key_param = action_key_param(cont.params)
            # task.plan 连续调用防死锁：连续 ≥2 次强制跳出 continue loop
            if tool_name == "task.plan":
                _continue_plan_streak += 1
                # 本 tick 内已出现过 task.plan（无论是否 skip）→ continue 阶段不再允许重复 plan
                already_planned = any(item.get("tool") == "task.plan" for item in tool_history)
                if already_planned or _continue_plan_streak >= 2:
                    _log.warning(
                        "[continue] task.plan 连续 %d 次，强制中断 continue 循环",
                        _continue_plan_streak,
                    )
                    _forced_break = WMItem(
                        kind="self_awareness",
                        content=(
                            f"[强制中断] continue 阶段连续 {_continue_plan_streak} 次 task.plan，"
                            "下一 tick 必须直接执行计划中的工具，禁止再次 plan"
                        ),
                        priority=loop._cfg.thresholds.wm_pri_critical,
                    )
                    loop._wm.add(_forced_break)
                    _wm_delta.append(_forced_break.to_dict())
                    break
            else:
                _continue_plan_streak = 0
            for behavior_item in loop._behavior.on_act(
                tool_name,
                key_param,
                str(active_task.id) if active_task else None,
                cont.params,
            ):
                loop._wm.add(behavior_item)
                _wm_delta.append(behavior_item.to_dict())
            loop._behavior.apply_cognitive_probe(cognitive_signals)

        cont_result = await loop._execution.dispatch(cont, ctx)

        if cont_result.summary and (not cont_result.skipped or cont_result.error):
            tool_name = cont.chosen_action_id or ""
            key_param = action_key_param(cont.params)
            prefix = f"[{tool_name}{'  ' + key_param if key_param else ''}] "
            _result_item = WMItem(
                kind=tool_name or cont_result.kind,
                content=prefix + cont_result.summary,
                priority=cont_result.priority,
            )
            loop._wm.add(_result_item)
            # 不论成功还是 error-skipped，均追加到 wm_delta，让后续 continue 轮可感知
            _wm_delta.append(_result_item.to_dict())
        if cont.reflection and cont.reflection.strip():
            loop._wm.add(WMItem(
                kind="synthesis",
                content=f"[合成] {cont.reflection.strip()}",
                priority=loop._cfg.thresholds.wm_pri_insight,
            ))
        if cont.rationale:
            loop._episodic.record(
                role="assistant",
                content=f"[inner-{inner + 1}] {cont.rationale}",
                task_id=str(active_task.id) if active_task else None,
                affect=affect,
            )

        if cont.decision == "act":
            if cont_result.error and "oldtextnotfound" in (cont_result.error or "").lower():
                for behavior_item in loop._behavior.on_edit_failure(cont_result.error or ""):
                    loop._wm.add(behavior_item)
            loop._behavior.on_act_result(cont.chosen_action_id or "", cont_result.summary or "")
            tool_history.append(_tool_history_entry(cont, cont_result))

        action = cont
        result = cont_result
        if action.reply_to_user or not _should_continue_within_tick(action):
            break
        # PlanUnchanged：计划结构没变，继续循环只会死锁；WM 中已有"请直接执行"提示，
        # 直接跳出 continue 阶段，让下一 tick 感知 WM 后执行具体工具。
        if cont_result.error == "PlanUnchanged":
            _log.debug("[continue] PlanUnchanged — 跳出 continue 循环，下一 tick 直接执行")
            break

    # ② continue 循环结束后同步检测（兜底 inner 轮删除 BOOTSTRAP.md 的场景）
    await _maybe_reconcile_bootstrap(loop)

    return action, result
