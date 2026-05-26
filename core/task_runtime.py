from __future__ import annotations

import json
import logging
from datetime import datetime, UTC
from typing import Any

from core.judgment import JudgmentOutput
from store.task import TaskStore, Task
from memory.working import WorkingMemory, WMItem

_log = logging.getLogger("lingzhou.loop")

VALID_MODEL_TIERS = frozenset({"reader", "reasoner", "repair"})


def _suggest_tier_from_text(text: str) -> str | None:
    lowered = (text or "").lower()
    for tier in ("repair", "reasoner", "reader"):
        if tier in lowered:
            return tier
    return None


def _meta_reflection_set_fact_instruction(key: str, value: Any, *, scope: str = "system") -> str:
    serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return (
        "该建议尚未自动生效。若认可，请调用 memory.set_fact，"
        f"key={key}，scope={scope}，value={serialized}。"
    )


async def _ingest_actionable_meta_reflections(task_store: TaskStore, wm: WorkingMemory) -> list[str]:
    injected: list[str] = []
    for reflection in await task_store.list_meta_reflections(limit=10):
        if reflection.decision not in {"apply", "rollback"}:
            continue
        fact_key = f"meta_reflection:ingested:{reflection.id}"
        _, found = await task_store.get_fact(fact_key)
        if found:
            continue
        applied_change = "recorded"
        followup = ""
        if reflection.target_kind == "threshold":
            raw_policy, policy_found = await task_store.get_fact("control:durable_failure_policy")
            policy = {"threshold": 3, "ttl_sec": 7200}
            if policy_found and raw_policy.strip():
                try:
                    loaded = json.loads(raw_policy)
                    if isinstance(loaded, dict):
                        policy["threshold"] = int(loaded.get("threshold") or policy["threshold"])
                        policy["ttl_sec"] = int(loaded.get("ttl_sec") or policy["ttl_sec"])
                except Exception:
                    pass
            if reflection.decision == "rollback":
                policy = {"threshold": 3, "ttl_sec": 7200}
                applied_change = "queued durable failure policy rollback hint"
            else:
                policy["threshold"] = max(1, policy["threshold"] + 1)
                policy["ttl_sec"] = max(900, policy["ttl_sec"] // 2)
                applied_change = (
                    f"queued durable failure policy hint threshold={policy['threshold']} ttl={policy['ttl_sec']}"
                )
            await task_store.set_fact(
                "control:meta_reflection_hint:threshold",
                json.dumps(
                    {
                        "reflection_id": reflection.id,
                        "decision": reflection.decision,
                        "proposal": reflection.proposal,
                        "verification_plan": reflection.verification_plan,
                        "suggested_policy": policy,
                    },
                    ensure_ascii=False,
                ),
                scope="system",
            )
            followup = _meta_reflection_set_fact_instruction(
                "control:durable_failure_policy",
                policy,
                scope="system",
            )
        elif reflection.target_kind == "task_split" and reflection.task_id:
            await task_store.set_fact(
                f"task:{reflection.task_id}:needs_replan",
                json.dumps(
                    {
                        "reflection_id": reflection.id,
                        "decision": reflection.decision,
                        "proposal": reflection.proposal,
                        "verification_plan": reflection.verification_plan,
                    },
                    ensure_ascii=False,
                ),
                scope="task",
            )
            applied_change = "set task replan hint"
            followup = "该建议尚未自动写回任务。若认可，请调用 task.update 修改 next_step。"
        elif reflection.target_kind == "routing":
            preferred_tier = _suggest_tier_from_text(reflection.proposal)
            if reflection.task_id:
                await task_store.set_fact(
                    f"task:{reflection.task_id}:routing_guard",
                    json.dumps(
                        {
                            "reflection_id": reflection.id,
                            "decision": reflection.decision,
                            "tool_name": reflection.tool_name,
                            "proposal": reflection.proposal,
                            "preferred_tier": preferred_tier,
                        },
                        ensure_ascii=False,
                    ),
                    scope="task",
                )
            await task_store.set_fact(
                "control:meta_reflection_hint:routing",
                json.dumps(
                    {
                        "reflection_id": reflection.id,
                        "decision": reflection.decision,
                        "tool_name": reflection.tool_name,
                        "proposal": reflection.proposal,
                        "verification_plan": reflection.verification_plan,
                        "preferred_tier": preferred_tier,
                        "task_id": reflection.task_id,
                    },
                    ensure_ascii=False,
                ),
                scope="system",
            )
            if reflection.task_id:
                applied_change = "queued task routing guard"
                followup = "该建议尚未自动改写 task.model_tier。若认可，请调用 task.update 修改 model_tier。"
            elif reflection.decision == "rollback":
                applied_change = "queued routing rollback hint"
                followup = _meta_reflection_set_fact_instruction("pref:routing_overrides", "", scope="system")
            else:
                applied_change = "queued routing control hint"
                followup = "该建议尚未自动改写全局路由。若认可，请调用 memory.set_fact 更新 pref:routing_overrides。"
        else:
            await task_store.set_fact(
                f"control:meta_reflection_hint:{reflection.target_kind}",
                json.dumps(
                    {
                        "reflection_id": reflection.id,
                        "decision": reflection.decision,
                        "proposal": reflection.proposal,
                        "verification_plan": reflection.verification_plan,
                    },
                    ensure_ascii=False,
                ),
                scope="system",
            )
            applied_change = f"queued {reflection.target_kind} control hint"
            followup = "该建议尚未自动生效。若认可，请调用 memory.set_fact 写入相应 control/pref 事实。"
        wm.add(WMItem(
            kind="meta_reflection",
            content=(
                f"[双环反思 {reflection.decision}] target={reflection.target_kind} tool={reflection.tool_name or 'unknown'}\n"
                f"已处理：{applied_change}\n"
                f"诊断：{reflection.diagnosis}\n"
                f"建议：{reflection.proposal}\n"
                f"验证：{reflection.verification_plan}"
                + (f"\n处理建议：{followup}" if followup else "")
            )[:1200],
            priority=0.76 if reflection.decision == "rollback" else 0.72,
        ))
        if reflection.task_id:
            await task_store.set_fact(
                f"task:{reflection.task_id}:meta_reflection",
                json.dumps(
                    {
                        "reflection_id": reflection.id,
                        "decision": reflection.decision,
                        "target_kind": reflection.target_kind,
                        "proposal": reflection.proposal,
                        "verification_plan": reflection.verification_plan,
                    },
                    ensure_ascii=False,
                ),
                scope="task",
            )
        await task_store.set_fact(fact_key, datetime.now(UTC).isoformat(), scope="system")
        _log.info("[meta-reflection] surfaced reflection=%s target=%s change=%s", reflection.id, reflection.target_kind, applied_change)
        injected.append(reflection.id)
    return injected


async def _consume_task_runtime_hints(
    task_store: TaskStore,
    task: Task | None,
    wm: WorkingMemory,
) -> Task | None:
    if task is None:
        return None

    last_replan_id = str(task.extras.get("last_replan_reflection_id") or "")
    raw_replan, replan_found = await task_store.get_fact(f"task:{task.id}:needs_replan")
    if replan_found and raw_replan.strip():
        try:
            replan = json.loads(raw_replan)
        except Exception:
            replan = {}
        reflection_id = str(replan.get("reflection_id") or "")
        if reflection_id and reflection_id != last_replan_id:
            proposal = str(replan.get("proposal") or "").strip()
            verification = str(replan.get("verification_plan") or "").strip()
            replan_step = proposal or verification or "先重拆任务，再继续执行。"
            await task_store.update_task_data(task.id, {"last_replan_reflection_id": reflection_id})
            task.extras["last_replan_reflection_id"] = reflection_id
            wm.add(WMItem(
                kind="task_replan",
                content=(
                    f"[任务重规划建议] task#{task.id}\n"
                    f"建议 next_step: {replan_step[:240]}\n"
                    f"验证: {verification[:180] or '（无）'}\n"
                    "该建议尚未自动写回任务。若认可，请调用 task.update 修改 next_step。"
                ),
                priority=0.84,
            ))
            _log.info("[runtime-hint] task=%s surfaced replan hint=%s", task.id, replan_step)

    last_meta_id = str(task.extras.get("last_task_meta_reflection_id") or "")
    raw_meta, meta_found = await task_store.get_fact(f"task:{task.id}:meta_reflection")
    if meta_found and raw_meta.strip():
        try:
            meta_payload = json.loads(raw_meta)
        except Exception:
            meta_payload = {}
        reflection_id = str(meta_payload.get("reflection_id") or "")
        if reflection_id and reflection_id != last_meta_id:
            target_kind = str(meta_payload.get("target_kind") or "reflection")
            decision = str(meta_payload.get("decision") or "defer")
            proposal = str(meta_payload.get("proposal") or "").strip()
            verification = str(meta_payload.get("verification_plan") or "").strip()
            wm.add(WMItem(
                kind="task_reflection",
                content=(
                    f"[任务级反思 {decision}] target={target_kind}\n"
                    f"建议：{proposal or '（无）'}\n"
                    f"验证：{verification or '（无）'}"
                )[:320],
                priority=0.78,
            ))
            await task_store.update_task_data(task.id, {"last_task_meta_reflection_id": reflection_id})
            task.extras["last_task_meta_reflection_id"] = reflection_id
            _log.info("[runtime-hint] task=%s surface task meta reflection=%s", task.id, reflection_id)

    last_routing_id = str(task.extras.get("last_routing_reflection_id") or "")
    raw_guard, guard_found = await task_store.get_fact(f"task:{task.id}:routing_guard")
    if guard_found and raw_guard.strip():
        try:
            guard = json.loads(raw_guard)
        except Exception:
            guard = {}
        reflection_id = str(guard.get("reflection_id") or "")
        if reflection_id and reflection_id != last_routing_id:
            tool_name = str(guard.get("tool_name") or "unknown")
            proposal = str(guard.get("proposal") or "").strip()
            preferred_tier = str(guard.get("preferred_tier") or "").strip()
            tier = preferred_tier if preferred_tier in VALID_MODEL_TIERS else "repair"
            await task_store.update_task_data(task.id, {"last_routing_reflection_id": reflection_id})
            task.extras["last_routing_reflection_id"] = reflection_id
            wm.add(WMItem(
                kind="routing_guard",
                content=(
                    f"[路由护栏建议] task#{task.id} tool={tool_name}\n"
                    f"建议 tier: {tier}\n"
                    f"理由: {proposal[:220] or f'切换到 {tier} tier 复核动作选择。'}\n"
                    "该建议尚未自动应用。若认可，请调用 task.update 设置 model_tier。"
                ),
                priority=0.82,
            ))
            _log.info("[runtime-hint] task=%s surfaced routing guard tier=%s tool=%s", task.id, tier, tool_name)

    return task


async def _sync_task_progress_state(
    task_store: TaskStore,
    task: Task | None,
    *,
    previous_next_step: str,
    action: JudgmentOutput,
    progressful: bool,
    state_delta: dict[str, Any] | None = None,
) -> Task | None:
    if task is None:
        return None

    latest = await task_store.get_task_by_id(task.id) or task
    planned_next = str(action.next_step or "").strip()
    explicit_current_step = None
    if state_delta is not None and "current_step" in state_delta:
        explicit_current_step = str(state_delta.get("current_step") or "").strip()
    current_step = latest.current_step
    next_step = latest.next_step
    updated = False

    if explicit_current_step is not None and current_step != explicit_current_step:
        current_step = explicit_current_step
        updated = True

    if progressful and previous_next_step:
        if explicit_current_step is None and current_step != previous_next_step:
            current_step = previous_next_step
            updated = True
        if planned_next:
            if not next_step or next_step == previous_next_step:
                next_step = planned_next
                updated = True
        elif next_step == previous_next_step:
            next_step = ""
            updated = True
    elif planned_next and not next_step:
        next_step = planned_next
        updated = True

    if not updated:
        return latest

    await task_store.sync_task_progress(
        latest.id,
        current_step=current_step,
        next_step=next_step,
    )
    _log.info(
        "[task-progress] task=%s current_step=%s next_step=%s progressful=%s",
        latest.id,
        current_step[:120],
        next_step[:120],
        progressful,
    )
    refreshed = await task_store.get_task_by_id(latest.id)
    return refreshed or latest