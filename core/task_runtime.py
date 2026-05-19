from __future__ import annotations

import json
import logging
import re
from datetime import datetime, UTC
from typing import Any

from core.judgment import JudgmentOutput
from memory.task_store import TaskStore, Task
from memory.working import WorkingMemory, WMItem

_log = logging.getLogger("lingzhou.loop")

VALID_MODEL_TIERS = frozenset({"reader", "reasoner", "repair"})


def _suggest_tier_from_text(text: str) -> str | None:
    lowered = (text or "").lower()
    for tier in ("repair", "reasoner", "reader"):
        if tier in lowered:
            return tier
    return None


def _extract_reflection_policy(text: str) -> dict[str, int]:
    policy: dict[str, int] = {}
    threshold_match = re.search(r"(?:threshold|阈值)\s*[:=：]?\s*(\d+)", text, re.IGNORECASE)
    ttl_match = re.search(r"(?:ttl(?:_sec)?|静默(?:窗口|时长)?)\s*[:=：]?\s*(\d+)", text, re.IGNORECASE)
    if threshold_match:
        policy["threshold"] = int(threshold_match.group(1))
    if ttl_match:
        policy["ttl_sec"] = int(ttl_match.group(1))
    return policy


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
            suggested = _extract_reflection_policy(
                "\n".join([reflection.diagnosis, reflection.proposal, reflection.verification_plan])
            )
            if reflection.decision == "rollback":
                policy = {"threshold": 3, "ttl_sec": 7200}
                applied_change = "reset durable failure policy"
            else:
                policy["threshold"] = max(1, int(suggested.get("threshold") or (policy["threshold"] + 1)))
                policy["ttl_sec"] = max(900, int(suggested.get("ttl_sec") or (policy["ttl_sec"] // 2)))
                applied_change = f"set durable failure threshold={policy['threshold']} ttl={policy['ttl_sec']}"
            await task_store.set_fact("control:durable_failure_policy", json.dumps(policy, ensure_ascii=False), scope="system")
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
        elif reflection.target_kind == "routing":
            if reflection.task_id:
                await task_store.set_fact(
                    f"task:{reflection.task_id}:routing_guard",
                    json.dumps(
                        {
                            "reflection_id": reflection.id,
                            "decision": reflection.decision,
                            "tool_name": reflection.tool_name,
                            "proposal": reflection.proposal,
                            "preferred_tier": _suggest_tier_from_text(reflection.proposal),
                        },
                        ensure_ascii=False,
                    ),
                    scope="task",
                )
            if reflection.decision == "rollback":
                await task_store.set_fact("pref:routing_overrides", "", scope="system")
                applied_change = "cleared routing overrides"
            else:
                applied_change = "set routing guard"
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
        wm.add(WMItem(
            kind="meta_reflection",
            content=(
                f"[双环反思 {reflection.decision}] target={reflection.target_kind} tool={reflection.tool_name or 'unknown'}\n"
                f"执行：{applied_change}\n"
                f"诊断：{reflection.diagnosis}\n"
                f"建议：{reflection.proposal}\n"
                f"验证：{reflection.verification_plan}"
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
        _log.info("[meta-reflection] applied reflection=%s target=%s change=%s", reflection.id, reflection.target_kind, applied_change)
        injected.append(reflection.id)
    return injected


async def _consume_task_runtime_hints(
    task_store: TaskStore,
    task: Task | None,
    wm: WorkingMemory,
) -> Task | None:
    if task is None:
        return None

    updated = False
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
            if task.next_step != replan_step:
                await task_store.update_status(task.id, task.status, next_step=replan_step)
                task.next_step = replan_step
                updated = True
                _log.info("[runtime-hint] task=%s apply replan next_step=%s", task.id, replan_step)
            await task_store.update_task_data(task.id, {"last_replan_reflection_id": reflection_id})
            task.extras["last_replan_reflection_id"] = reflection_id
            wm.add(WMItem(
                kind="task_replan",
                content=f"[任务重规划] task#{task.id} {replan_step[:240]}",
                priority=0.84,
            ))

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
            if task.model_tier != tier:
                await task_store.update_task_data(task.id, {"model_tier": tier})
                task.model_tier = tier
                updated = True
                _log.info("[runtime-hint] task=%s apply routing guard via %s tier for tool=%s", task.id, tier, tool_name)
            await task_store.update_task_data(task.id, {"last_routing_reflection_id": reflection_id})
            task.extras["last_routing_reflection_id"] = reflection_id
            wm.add(WMItem(
                kind="routing_guard",
                content=f"[路由护栏] task#{task.id} tool={tool_name} {proposal[:220] or f'切换到 {tier} tier 复核动作选择。'}",
                priority=0.82,
            ))

    if updated:
        refreshed = await task_store.get_task_by_id(task.id)
        return refreshed or task
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