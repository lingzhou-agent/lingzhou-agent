"""core/loop/tick.py - tick 编排与收尾后处理实现。"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from rich.console import Console

from core.judgment import JudgmentOutput, tool_tier
from core.perception import (
    build_emotion_replay,
    build_perception_replay,
    compute_judgment_signals,
    derive_ethos_state,
)
from core.run_refresh import refresh_running_runs
from core.task_runtime import (
    VALID_MODEL_TIERS,
    _consume_task_runtime_hints,
    _ingest_actionable_meta_reflections,
    _sync_task_progress_state,
)
from memory.semantic import MemoryNode
from memory.task_store import Task
from memory.working import WMItem
from tools.registry import ToolResult

from .chat import _bind_chat_id, _resolve_reply_chat_id
from .common import (
    _EVENT_APPEND_CHARS,
    _EVENT_BODY_MAX_CHARS,
    _EVENT_NEW_BODY_CHARS,
    _EVENT_TITLE_CHARS,
    _SEM_TAG_TASK_CHARS,
    _SEM_TITLE_CHARS,
    _infer_valence_from_text,
    _next_thinking_override,
    _perception_replay_fallback,
    _preferred_continue_tier,
    _prefer_tier_for_task,
    _resolve_thinking_override,
    _should_continue_within_tick,
    _task_model_tier,
    _thinking_floor,
)
from .logging import (
    _clip_reply_for_log,
    _clip_signal_text,
    _fallback_reply_for_user,
    _format_action_feedback_line,
    _strip_memory_context,
    _summarize_state_delta,
)
from .postprocess import (
    _SUCCESS_STALL_TRACK_TOOLS,
    _write_success_stall_meta_reflection,
)
from .progress import (
    action_key_param,
    _action_made_progress,
    _result_fingerprint,
)

console = Console()
_log = logging.getLogger("lingzhou.loop")


def _tool_history_entry(action: JudgmentOutput, result: ToolResult) -> dict[str, Any]:
    summary = str(result.summary or "")
    error = str(result.error or "")
    status = "error" if error else ("skipped" if result.skipped else "ok")
    error_category = ""
    if error:
        err_lower = error.lower()
        error_category = (
            "transient"
            if any(marker in err_lower for marker in ("timeout", "connect", "reset", "unavailable", "rate", "429", "503"))
            else "fatal"
        )
    return {
        "tool": action.chosen_action_id or "",
        "params": action.params or {},
        "result": f"ERROR[{error_category}]: {summary}" if error else summary,
        "summary": summary,
        "error": error,
        "error_category": error_category,
        "skipped": bool(result.skipped),
        "status": status,
        "state_delta": dict(result.state_delta or {}) if isinstance(result.state_delta, dict) else {},
    }


async def _maybe_reconcile_bootstrap(loop: Any) -> None:
    """如果 BOOTSTRAP.md 已被本 tick 删除，写入 setupCompletedAt 并切换到正常模式。"""
    if loop._bootstrap_mode != "full":
        return
    bootstrap_path = loop._cfg.workspace_dir / "BOOTSTRAP.md"
    if bootstrap_path.exists():
        return
    from core.workspace.state import reconcile_bootstrap_completion
    reconcile_bootstrap_completion(loop._cfg.workspace_dir)
    await loop._soul.refresh_identity(loop._judgment)
    loop._bootstrap_mode = "none"
    _log.info("[bootstrap] BOOTSTRAP.md 已删除，切换到正常运行模式")


def _maybe_inject_bootstrap_signal(loop: Any, active_task: Any) -> None:
    """bootstrap_mode=full 时，向 WM 注入引导待完成感知信号（无论是否有活跃任务）。

    BOOTSTRAP.md 以静态 identity 前缀注入系统提示词，LLM 倾向于将其视为"背景说明"
    而非"当前待办工作"。此函数在动态感知层（WM）补充一条高优先级条目，
    将引导任务拉入 LLM 每轮的主动注意焦点——不是命令，是感知。
    LLM 依然可以基于整体判断决定此刻是否行动。

    注意：不再以 active_task 为过滤条件——有任务时同样注入，
    确保 LLM 始终感知到"bootstrap 尚未关闭"这一事实。
    """
    if loop._bootstrap_mode != "full":
        return
    if active_task is not None:
        content = (
            "[初始化未完成] BOOTSTRAP.md 仍然存在，初始化步骤尚未全部完成并确认。"
            "当前有活跃任务，可在任务完成后处理初始化，"
            "或在本轮穿插完成初始化步骤（逐项确认 IDENTITY/SOUL/USER/TOOLS 内容是否落实），"
            "完成后用 file.delete 删除 BOOTSTRAP.md 以结束引导阶段。"
        )
    else:
        content = (
            "[初始化待完成] BOOTSTRAP.md 仍然存在，说明初始化检查项尚未全部完成并确认。"
            "当前无活跃任务，这是推进初始化的自然时机："
            "逐项确认 IDENTITY / SOUL / USER / TOOLS 的内容是否已具体落实，"
            "完成后用 file.delete 删除 BOOTSTRAP.md 以结束引导阶段。"
        )
    loop._wm.add(WMItem(
        kind="bootstrap",
        content=content,
        priority=0.90,
    ))


async def _tick_impl(loop: Any, cycle: int, user_message: str = "", chat_id: str | None = None) -> str:
    """执行一轮完整认知 tick,返回 reply_to_user(interact 模式时非空)。"""
    cfg = loop._cfg
    ctx = loop._make_ctx()
    reply = ""

    if user_message:
        loop._wm.add(WMItem(
            kind="user_message",
            content=f"[用户消息] {user_message[:200]}",
            priority=0.95,
        ))

    loop._maybe_inject_budget_warning()

    running_updates = await refresh_running_runs(loop._task_store, episodic=loop._episodic, semantic=loop._semantic)
    active_task = await loop._task_store.get_active()
    await _ingest_actionable_meta_reflections(loop._task_store, loop._wm)
    active_task = await _consume_task_runtime_hints(loop._task_store, active_task, loop._wm)
    await _bind_chat_id(loop, active_task, chat_id)

    if not user_message:
        loop._maybe_inject_self_drive()
        _maybe_inject_bootstrap_signal(loop, active_task)
    loop._wm.clear(kinds={"run_monitor"})
    if running_updates:
        running_count = sum(1 for item in running_updates if item.get("status") == "running")
        finished_count = sum(1 for item in running_updates if item.get("status") in {"succeeded", "failed", "cancelled"})
        loop._wm.add(WMItem(
            kind="run_monitor",
            content=f"[Run 监控] running={running_count} finished={finished_count}",
            priority=0.58,
        ))
        for item in running_updates:
            crystal = str(item.get("crystal") or "").strip()
            if crystal:
                loop._wm.add(WMItem(
                    kind="progress_crystal",
                    content=f"[运行中结晶 run#{item.get('run_id')}] {crystal[:280]}",
                    priority=0.72,
                ))
                loop._episodic.record_event("run_progress", {
                    "run_id": item.get("run_id"),
                    "task_id": item.get("task_id"),
                    "session_id": item.get("session_id"),
                    "excerpt": crystal[:800],
                })

    for sig in await loop._task_store.due_signals():
        payload = sig.get("payload") or {}
        note = (payload.get("note") or "").strip()
        repeat_desc = f"每 {sig['repeat_secs']}s 重复" if sig.get("repeat_secs") else "一次性"
        parts = [
            (
                f"[调度触发 #{sig['id']}] {sig['title']}"
                f"({repeat_desc},已送达本轮上下文;是否响应由你决定。"
                "delivery 后该 signal 会由 runtime 自动推进/完成,通常无需再调用 schedule.ack)"
            ),
        ]
        if note:
            parts.append(f"任务内容:{note}")
        loop._wm.add(WMItem(
            kind="scheduler",
            content="\n".join(parts),
            priority=loop._cfg.thresholds.wm_pri_signal,
        ))
        await loop._task_store.ack_signal(sig["id"])
        _log.info("[scheduler] signal fired: #%s %s", sig["id"], sig["title"])

    now = time.monotonic()
    if now - loop._last_heartbeat_at >= loop._cfg.loop.heartbeat_interval:
        heartbeat_path = loop._cfg.workspace_dir / "HEARTBEAT.md"
        if heartbeat_path.exists():
            try:
                heartbeat_md = heartbeat_path.read_text(encoding="utf-8").strip()
                if heartbeat_md:
                    loop._wm.add(WMItem(
                        kind="heartbeat",
                        content=f"[心跳自检]\n{heartbeat_md}",
                        priority=loop._cfg.thresholds.wm_pri_signal,
                    ))
                    _log.info("[heartbeat] 注入 WM,间隔 %ds", loop._cfg.loop.heartbeat_interval)
            except Exception:
                pass
        loop._last_heartbeat_at = now

    next_step_fulfilled: bool | None = None
    if loop._last_next_step:
        next_step_fulfilled = loop._last_act_progressful
    percept = await loop._perception.sense(
        loop._wm,
        active_task,
        last_next_step=loop._last_next_step,
        last_decision=loop._last_decision,
    )

    loop._episodic.record_event("perception", {
        "prediction_error": round(percept.prediction_error, 4),
        "workspace_dirty": percept.workspace_dirty,
        "wm_pressure": round(loop._wm.pressure, 4),
    })

    events_batch = loop._episodic.list_events_multi(["perception", "emotion"], limit=8)
    perception_events = events_batch["perception"]
    try:
        perception_replay = build_perception_replay(
            perception_events,
            high_error_threshold=cfg.thresholds.prediction_error_task,
        )
    except Exception:
        perception_replay = _perception_replay_fallback()

    if active_task is None:
        loop._idle_cycles += 1
    else:
        loop._idle_cycles = 0
        loop._last_curiosity_signal_idle_cycle = 0

    cognitive_signals = loop._perception.derive_cognitive_signals(
        percept,
        loop._wm,
        loop._emotion,
        cfg,
        has_active_task=active_task is not None,
        idle_cycles=loop._idle_cycles,
        next_step_fulfilled=next_step_fulfilled,
    )
    loop._behavior.apply_cognitive_probe(cognitive_signals)
    cognitive_signals.last_action_tool = loop._last_action_tool
    cognitive_signals.last_action_key = loop._last_action_key
    cognitive_signals.last_action_status = loop._last_action_status
    cognitive_signals.last_action_summary = loop._last_action_summary
    cognitive_signals.last_action_error = loop._last_action_error
    cognitive_signals.last_action_state_delta = loop._last_action_state_delta
    cognitive_signals.last_action_progressful = loop._last_act_progressful if loop._last_action_status else None
    cognitive_signals.last_action_progress_reason = loop._last_act_progress_reason if loop._last_action_status else ""
    cognitive_signals.recent_action_history = list(loop._recent_action_feedback)

    failures_recent = await loop._task_store.list_failures(limit=5)
    loop._emotion.derive_from_signals(
        failure_count=len(failures_recent),
        prediction_error=percept.prediction_error,
        wm_pressure=loop._wm.pressure,
        workspace_dirty=percept.workspace_dirty,
        alpha=cfg.emotion.ema_alpha,
        high_error_streak=perception_replay.high_error_streak,
        replay_trend=perception_replay.trend,
        has_active_task=active_task is not None,
        has_next_step=bool(active_task and active_task.next_step),
        task_status=active_task.status if active_task else "",
    )

    loop._episodic.record_event("emotion", {
        "valence": round(loop._emotion.valence, 4),
        "arousal": round(loop._emotion.arousal, 4),
        "dominance": round(loop._emotion.dominance, 4),
        "dominant": loop._emotion.dominant,
        "regulation_strategy": loop._emotion.regulation.strategy,
        "regulation_reason": loop._emotion.regulation.reason,
    })

    emotion_replay = build_emotion_replay(events_batch["emotion"])

    ethos_baseline_json, _ = await loop._task_store.get_fact("soul:ethos_baseline")
    ethos_baseline = json.loads(ethos_baseline_json) if ethos_baseline_json else None
    ethos_state = derive_ethos_state(
        failure_count=len(failures_recent),
        high_error_streak=perception_replay.high_error_streak,
        has_active_task=active_task is not None,
        has_next_step=bool(active_task and active_task.next_step),
        perception_trend=perception_replay.trend,
        emotion_down_regulate_streak=emotion_replay.down_regulate_streak,
        baseline=ethos_baseline,
        ema_alpha=cfg.soul.ethos_ema_alpha,
        floor_truth=cfg.soul.ethos_floor_truth,
        floor_caution=cfg.soul.ethos_floor_caution,
    )

    await loop._task_store.set_fact("soul:ethos_baseline", json.dumps({
        "truth": ethos_state.values.truth,
        "caution": ethos_state.values.caution,
        "continuity": ethos_state.values.continuity,
        "curiosity": ethos_state.values.curiosity,
        "care": ethos_state.values.care,
    }))

    signals = compute_judgment_signals(
        failure_count=len(failures_recent),
        high_error_streak=perception_replay.high_error_streak,
        perception_trend=perception_replay.trend,
        emotion_state=loop._emotion,
    )
    axioms_json, _ = await loop._task_store.get_fact("soul:hard_axioms")
    hard_boundaries: list[str] = json.loads(axioms_json) if axioms_json else []

    if active_task is None:
        await loop._maybe_curiosity_task(ethos_state)

    has_external_signal = any(item.get("kind") in ("heartbeat", "scheduler") for item in loop._wm.get_top(20))
    skip_llm = (
        cfg.loop.judge_every > 1
        and not user_message
        and active_task is None
        and not has_external_signal
        and loop._ticks_since_judge < cfg.loop.judge_every - 1
    )
    if skip_llm:
        loop._ticks_since_judge += 1
        action = JudgmentOutput.wait(
            reason=f"[按请求聚合] 空闲跳过 LLM({loop._ticks_since_judge}/{cfg.loop.judge_every})"
        )
        _log.debug(
            "[loop] tick=%d 跳过 LLM 判断(聚合 %d/%d)",
            cycle,
            loop._ticks_since_judge,
            cfg.loop.judge_every,
        )
    else:
        pending_initial_thinking = loop._pending_thinking_override
        if user_message:
            chat_floor = cfg.loop.chat_thinking if cfg.loop.chat_thinking != cfg.thinking else None
            pending_initial_thinking = _thinking_floor(pending_initial_thinking, chat_floor)
        thinking_override = _resolve_thinking_override(
            cfg,
            user_message=user_message,
            pending_override=pending_initial_thinking,
        )
        action = await loop._judgment.decide(
            percept,
            loop._wm,
            loop._task_store,
            loop._episodic,
            loop._semantic,
            loop._emotion,
            user_message=user_message,
            ethos_state=ethos_state,
            judgment_signals=signals,
            hard_boundaries=hard_boundaries,
            perception_replay=perception_replay,
            cognitive_signals=cognitive_signals,
            thinking_override=thinking_override,
            phase="initial",
            prefer_tier=_prefer_tier_for_task(loop._pending_tier, active_task),
            routing_overrides=loop._pending_routing_overrides,
        )
        loop._pending_tier = None
        loop._pending_thinking_override = None
        loop._ticks_since_judge = 0

    loop._judgment.self_model.record_tick()
    loop._judgment.self_model.record_api_call()
    call_meta = loop._judgment.last_call_meta
    actual_model = call_meta.get("model_ref") or cfg.model
    actual_thinking = call_meta.get("thinking") or cfg.thinking
    actual_tier = call_meta.get("tier") or "default"
    actual_phase = call_meta.get("phase") or "initial"
    actual_skills = call_meta.get("skills") or "none"
    model_tag = (
        f" model={actual_model} tier={actual_tier} phase={actual_phase} thinking={actual_thinking} skills={actual_skills}"
        if actual_thinking != "off"
        else f" model={actual_model} tier={actual_tier} phase={actual_phase} skills={actual_skills}"
    )
    console.print(
        f"[bold cyan][loop][/bold cyan] tick={cycle} "
        f"decision={action.decision} tool={action.chosen_action_id}"
        f"[dim]{model_tag}[/dim]"
    )
    _log.info(
        "[loop] tick=%d decision=%s tool=%s model=%s tier=%s phase=%s thinking=%s skills=%s rationale=%s",
        cycle,
        action.decision,
        action.chosen_action_id,
        actual_model,
        actual_tier,
        actual_phase,
        actual_thinking,
        actual_skills,
        action.rationale or "",
    )

    if action.decision == "act":
        tool_id = action.chosen_action_id or ""
        key_param = action_key_param(action.params)
        current_task_id = str(active_task.id) if active_task else None
        for item in loop._behavior.on_act(tool_id, key_param, current_task_id):
            loop._wm.add(item)
    else:
        for item in loop._behavior.on_wait(action.decision, active_task is not None):
            loop._wm.add(item)

    action = loop._behavior.apply_execution_gate(action, cognitive_signals)
    result = await loop._execution.dispatch(action, ctx)
    tool_history: list[dict[str, Any]] = []
    if action.decision == "act":
        tool_history.append(_tool_history_entry(action, result))

    if action.decision == "act" and not result.error:
        tool = action.chosen_action_id or ""
        path = (action.params or {}).get("path") or ""
        if tool == "file.read":
            max_chars = int((action.params or {}).get("max_chars") or 4000)
            start = int((action.params or {}).get("start") or 0)
            end = int((action.params or {}).get("end") or 0)
            for item in loop._behavior.on_read(path, max_chars, result.summary, start=start, end=end):
                loop._wm.add(item)
        elif tool == "file.list":
            for item in loop._behavior.on_list(path, result.summary):
                loop._wm.add(item)
        elif tool == "file.edit" and result.error and "OldTextNotFound" in (result.error or ""):
            for item in loop._behavior.on_edit_failure(result.error or ""):
                loop._wm.add(item)

    # ① in-session bootstrap 完成检测（主工具执行后）
    await _maybe_reconcile_bootstrap(loop)

    if _should_continue_within_tick(
        action,
        user_message=user_message,
        has_active_task=active_task is not None,
    ):
        affect = {"valence": loop._emotion.valence, "arousal": loop._emotion.arousal}
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
            )

            if cont.decision == "act":
                tool_name = cont.chosen_action_id or ""
                key_param = action_key_param(cont.params)
                for behavior_item in loop._behavior.on_act(tool_name, key_param, str(active_task.id) if active_task else None):
                    loop._wm.add(behavior_item)
                loop._behavior.apply_cognitive_probe(cognitive_signals)
            cont = loop._behavior.apply_execution_gate(cont, cognitive_signals)
            cont_result = await loop._execution.dispatch(cont, ctx)

            if cont_result.summary and not cont_result.skipped:
                tool_name = cont.chosen_action_id or ""
                key_param = action_key_param(cont.params)
                prefix = f"[{tool_name}{'  ' + key_param if key_param else ''}] "
                loop._wm.add(WMItem(kind=tool_name or cont_result.kind, content=prefix + cont_result.summary, priority=cont_result.priority))
            if cont.reflection and cont.reflection.strip():
                loop._wm.add(WMItem(kind="synthesis", content=f"[合成] {cont.reflection.strip()}", priority=0.88))
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
                tool_history.append(_tool_history_entry(cont, cont_result))

            action = cont
            result = cont_result
            if action.reply_to_user or not _should_continue_within_tick(action):
                break

        # ② continue 循环结束后同步检测（兜底 inner 轮删除 BOOTSTRAP.md 的场景）
        await _maybe_reconcile_bootstrap(loop)

    if user_message and not action.reply_to_user:
        reply_only = await loop._judgment.decide_continue(
            tool_history,
            user_message=user_message,
            active_task=active_task,
            prefer_tier="reasoner",
            thinking_override=_thinking_floor(
                _resolve_thinking_override(
                    cfg,
                    user_message=user_message,
                    model_strategy=action.model_strategy,
                ),
                "low",
            ),
            routing_overrides=loop._pending_routing_overrides,
            reply_only=True,
        )
        if reply_only.reply_to_user:
            action.reply_to_user = reply_only.reply_to_user
            if reply_only.rationale:
                action.rationale = reply_only.rationale
            if reply_only.reflection and not action.reflection:
                action.reflection = reply_only.reflection
            if reply_only.next_step and not action.next_step:
                action.next_step = reply_only.next_step

    if user_message and not action.reply_to_user:
        action.reply_to_user = _fallback_reply_for_user(action, result, active_task)

    if action.reply_to_user:
        action.reply_to_user = _strip_memory_context(action.reply_to_user)
        _log.info(
            "[task-reply] task=%s decision=%s reply=%s",
            active_task.id if active_task else 0,
            action.decision,
            _clip_reply_for_log(action.reply_to_user),
        )
        outbound_chat_id = await _resolve_reply_chat_id(loop, active_task, chat_id)
        if outbound_chat_id is not None:
            await loop._task_store.add_chat_message(
                "assistant",
                action.reply_to_user,
                chat_id=outbound_chat_id,
            )

    reply = await _tick_finalize_impl(
        loop,
        action,
        result,
        active_task,
        cycle,
        user_message,
        cognitive_signals,
        reply,
        chat_id,
        perception_replay,
    )
    return reply


async def _tick_finalize_impl(
    loop: Any,
    action: JudgmentOutput,
    result: ToolResult | Any,
    active_task: Task | None,
    cycle: int,
    user_message: str,
    cognitive_signals: Any,
    reply: str,
    chat_id: str | None = None,
    perception_replay: Any = None,
) -> str:
    cfg = loop._cfg

    await loop._post_tick_memory(action, result, active_task, cycle, user_message)
    await loop._save_self_model()

    if cycle % cfg.loop.consolidate_every == 0:
        if loop._wm.pressure >= loop._cfg.thresholds.wm_pressure_task:
            await loop._consolidate(active_task)
        # 感知 global.md 膨胀 → 注入信号让 LLM 自主决定是否压缩
        try:
            _gm = loop._cfg.memory_dir / "global.md"
            if _gm.exists():
                _sz = _gm.stat().st_size
                _lc = len(_gm.read_text().splitlines())
                if _sz > 80000 or _lc > 600:
                    from memory.working import WMItem
                    loop._wm.add(WMItem(
                        kind="self_awareness",
                        content=(
                            f"[记忆压力] global.md 当前 {_lc} 行 / {_sz} 字节，已经较大。"
                            "建议在空闲时压缩旧记录：保留最近 100 条详情，旧内容用 LLM 提炼为关键要点摘要。"
                            "压缩方法：file.read global.md 的旧段落 → 提炼为简短摘要 → file.edit 替换。"
                        ),
                        priority=0.75,
                    ))
        except Exception:
            pass
        await loop._soul.sync_md()
        # 定期 WAL checkpoint 防止 DB 膨胀
        try:
            await loop._task_store._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass

    should_evolve = False
    if perception_replay is not None:
        should_evolve = (
            cfg.evolution.enabled and (
                perception_replay.high_error_streak >= cfg.evolution.error_streak_evolve
                or cycle % cfg.loop.evolve_every == 0
            )
        )
    if should_evolve:
        ctx = loop._make_ctx()
        results = await loop._evolution.run(ctx)
        for evolve_result in results:
            if evolve_result.success:
                console.print(f"[green][evolution] {evolve_result.target} 已进化[/green]")
                if evolve_result.target.startswith("prompt:"):
                    prompt_key = evolve_result.target.split(":", 1)[1]
                    loop._judgment.reload_prompt(prompt_key)
        await loop._soul.refresh_identity(loop._judgment)

    previous_task_next_step = (active_task.next_step or "") if active_task else ""
    prev_sig = loop._last_action_sig
    prev_fp = loop._last_result_fp
    cur_sig = f"{action.chosen_action_id or ''}|{action_key_param(action.params)}" if action.decision == "act" else ""
    cur_fp = _result_fingerprint(result.summary) if action.decision == "act" and not result.error and not result.skipped else ""
    loop._last_next_step = action.next_step or ""
    loop._last_decision = action.decision
    loop._last_act_error = bool(action.decision == "act" and result.error)
    loop._last_act_progressful, loop._last_act_progress_reason = _action_made_progress(action, result, prev_sig=prev_sig, prev_fp=prev_fp)
    loop._last_action_tool = action.chosen_action_id or ""
    loop._last_action_key = action_key_param(action.params) if action.decision == "act" else ""
    loop._last_action_summary = _clip_signal_text(result.summary or "") if action.decision == "act" else ""
    loop._last_action_error = _clip_signal_text(result.error or "", 100) if action.decision == "act" else ""
    loop._last_action_state_delta = _summarize_state_delta(result.state_delta) if action.decision == "act" else ""
    if action.decision == "act":
        if result.error:
            loop._last_action_status = "error"
        elif result.skipped:
            loop._last_action_status = "skipped"
        else:
            loop._last_action_status = "ok"
    else:
        loop._last_action_status = action.decision
    loop._recent_action_feedback.append(
        _format_action_feedback_line(
            action,
            result,
            progressful=loop._last_act_progressful,
        )
    )
    loop._last_action_sig = cur_sig
    loop._last_result_fp = cur_fp
    active_task = await _sync_task_progress_state(
        loop._task_store,
        active_task,
        previous_next_step=previous_task_next_step,
        action=action,
        progressful=loop._last_act_progressful,
        state_delta=result.state_delta,
    )
    await _bind_chat_id(loop, active_task, chat_id)
    await _maybe_record_success_stall_reflection_impl(loop, active_task, action, result, cycle)

    next_tier = str((action.model_strategy or {}).get("next_phase_tier", "") or "")
    task_tier = _task_model_tier(active_task)
    actual_tier = (loop._judgment.last_call_meta or {}).get("tier") or "default"
    persist_tier = next_tier if next_tier in VALID_MODEL_TIERS else (task_tier or (actual_tier if actual_tier in VALID_MODEL_TIERS else ""))
    if active_task and persist_tier and persist_tier != task_tier:
        await loop._task_store.update_task_data(active_task.id, {"model_tier": persist_tier})
        active_task.model_tier = persist_tier
    if next_tier in {"reader", "reasoner", "repair"}:
        loop._pending_tier = next_tier
    else:
        tool_id = action.chosen_action_id or ""
        if action.decision == "act" and tool_tier(tool_id, loop._registry) == "reader":
            loop._pending_tier = "reader"
        else:
            loop._pending_tier = None

    raw_gap = (action.model_strategy or {}).get("next_idle_gap_secs")
    if raw_gap is not None:
        try:
            gap_f = float(raw_gap)
            has_task = (await loop._task_store.get_active()) is not None
            if has_task:
                bounds = cfg.loop.idle_with_task_bounds
                lo, hi = (float(bounds[0]), float(bounds[1])) if len(bounds) >= 2 else (2.0, 30.0)
            else:
                bounds = cfg.loop.idle_no_task_bounds
                lo, hi = (float(bounds[0]), float(bounds[1])) if len(bounds) >= 2 else (5.0, 300.0)
            loop._pending_idle_gap = max(lo, min(hi, gap_f))
        except (TypeError, ValueError):
            loop._pending_idle_gap = None
    else:
        loop._pending_idle_gap = None

    raw_overrides = (action.model_strategy or {}).get("routing_overrides")
    if isinstance(raw_overrides, dict):
        if not raw_overrides:
            loop._pending_routing_overrides = None
            await loop._task_store.set_fact("pref:routing_overrides", "", scope="system")
        else:
            valid = {
                key: value for key, value in raw_overrides.items()
                if key in {"reader", "reasoner", "repair"} and isinstance(value, str) and value
            }
            if valid:
                loop._pending_routing_overrides = valid
                await loop._task_store.set_fact("pref:routing_overrides", json.dumps(valid), scope="system")

    loop._pending_thinking_override = _next_thinking_override(action.model_strategy)

    await loop._task_store.set_fact("soul:emotion_state", json.dumps({
        "valence": round(loop._emotion.valence, 4),
        "arousal": round(loop._emotion.arousal, 4),
        "dominance": round(loop._emotion.dominance, 4),
    }))

    return action.reply_to_user


async def _maybe_record_success_stall_reflection_impl(
    loop: Any,
    active_task: Task | None,
    action: JudgmentOutput,
    result: ToolResult,
    cycle: int,
) -> None:
    tool_name = action.chosen_action_id or ""
    qualifies = (
        active_task is not None
        and action.decision == "act"
        and not result.error
        and not result.skipped
        and not loop._last_act_progressful
        and tool_name in _SUCCESS_STALL_TRACK_TOOLS
    )
    if not qualifies:
        loop._success_stall_task_id = str(active_task.id) if active_task else None
        loop._success_stall_streak = 0
        return

    assert active_task is not None
    task_id = str(active_task.id)
    if loop._success_stall_task_id != task_id:
        loop._success_stall_task_id = task_id
        loop._success_stall_streak = 0

    loop._success_stall_streak += 1
    if loop._success_stall_streak != 2:
        return

    await _write_success_stall_meta_reflection(
        loop._task_store,
        active_task,
        action,
        result,
        streak=loop._success_stall_streak,
        cycle=cycle,
    )


async def _post_tick_memory_impl(
    loop: Any,
    action: JudgmentOutput,
    result: Any,
    active_task: Any,
    cycle: int,
    user_message: str,
) -> None:
    if active_task and active_task.status not in ("done", "failed"):
        refreshed = await loop._task_store.get_task_by_id(active_task.id)
        if refreshed and refreshed.status in ("done", "failed"):
            marker = f"crystallized:{refreshed.id}"
            _, already = await loop._task_store.get_fact(marker)
            if not already:
                narrative = loop._episodic.load_for_context(str(refreshed.id), max_chars=40000)
                if narrative.strip():
                    node_id = f"task_summary_{refreshed.id}"
                    loop._semantic.upsert(MemoryNode(
                        id=node_id,
                        kind="task_summary",
                        title=f"[{refreshed.status}] {refreshed.title[:60]}",
                        body=narrative,
                        activation=0.9 if refreshed.status == "done" else 0.7,
                        valence=loop._emotion.valence,
                        tags=["task_summary", refreshed.status, f"task_{refreshed.id}"],
                    ))
                await loop._task_store.set_fact(marker, "1", scope="system")

    if result.summary and not result.skipped:
        tool_id = action.chosen_action_id or ""
        key_param = action_key_param(action.params)
        wm_prefix = f"[{tool_id}{'  ' + key_param if key_param else ''}] "
        loop._wm.add(WMItem(
            kind=tool_id or result.kind,
            content=wm_prefix + result.summary,
            priority=result.priority,
        ))

    if action.reflection and action.reflection.strip():
        loop._wm.add(WMItem(
            kind="synthesis",
            content=f"[合成] {action.reflection.strip()}",
            priority=0.88,
        ))

    affect = {"valence": loop._emotion.valence, "arousal": loop._emotion.arousal}
    if action.rationale:
        clean_rationale = _strip_memory_context(action.rationale)
        loop._episodic.record(
            role="assistant",
            content=f"[cycle={cycle}] {clean_rationale}",
            task_id=str(active_task.id) if active_task else None,
            affect=affect,
        )

    if action.reflection:
        clean_reflection = _strip_memory_context(action.reflection)
        node_id = f"insight_{hashlib.md5(clean_reflection.encode()).hexdigest()[:10]}"
        loop._semantic.upsert(MemoryNode(
            id=node_id,
            kind="learned_insight",
            title=clean_reflection[:_SEM_TITLE_CHARS],
            body=clean_reflection,
            activation=0.9,
            valence=loop._emotion.valence,
            tags=["reflection", active_task.title[:_SEM_TAG_TASK_CHARS] if active_task else "free"],
        ))
        ref_valence = _infer_valence_from_text(clean_reflection, loop._emotion.valence)
        delta = ref_valence - loop._emotion.valence
        if abs(delta) > 0.01:
            loop._emotion.valence = round(
                loop._emotion.valence + min(max(delta, -0.05), 0.05),
                4,
            )

        if active_task:
            turns_key = f"chat:{active_task.id}:turns"
            turns_val, _ = await loop._task_store.get_fact(turns_key)
            turns = int(turns_val or "0") + 1
            await loop._task_store.set_fact(turns_key, str(turns), scope="system")
            crystallize_every = loop._cfg.memory.chat_crystallize_every
            if turns % crystallize_every == 0:
                ts_label = datetime.now(UTC).strftime("%Y-%m-%d")
                evt_id = f"event-task{active_task.id}-{ts_label}"
                existing = loop._semantic.get(evt_id)
                if existing:
                    existing.body = (existing.body + f"\n- {clean_reflection[:_EVENT_APPEND_CHARS]}")[-_EVENT_BODY_MAX_CHARS:]
                    existing.activation = min(1.0, existing.activation + 0.05)
                    loop._semantic.upsert(existing)
                else:
                    source = getattr(active_task, "source", "") or ""
                    chat_id = source[5:] if source.startswith("chat:") else source
                    tags = ["event", ts_label]
                    if chat_id:
                        tags.append(chat_id)
                    loop._semantic.upsert(MemoryNode(
                        id=evt_id,
                        kind="event",
                        title=f"[{ts_label}] {active_task.title[:_EVENT_TITLE_CHARS]}",
                        body=clean_reflection[:_EVENT_NEW_BODY_CHARS],
                        activation=0.85,
                        valence=loop._emotion.valence,
                        tags=tags,
                    ))

    if user_message:
        loop._episodic.record(
            role="user",
            content=user_message,
            task_id=str(active_task.id) if active_task else None,
            source_type="human",
        )
        if action.reply_to_user:
            loop._episodic.record(
                role="assistant_reply",
                content=_strip_memory_context(action.reply_to_user),
                task_id=str(active_task.id) if active_task else None,
                affect=affect,
            )
