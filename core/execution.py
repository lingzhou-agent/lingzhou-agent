"""core/execution.py — 执行层。

职责：
- 接收 JudgmentOutput，dispatch 到具体工具
- 处理 act / pause / wait 三种决策
- 失败时写入 failures 表（绑定当前任务 ID，P2-B 原则）
- 对稳定重复失败的确定性动作做持久降噪（durable failure sensing）
- 返回 ToolResult 给 loop 层整合
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from core.config import run_result_memory_affect
from core.worker import WorkerLayer
from store.task import build_task_run_result_patch
from tools.registry import ToolResult, ToolContext, tool_has_capability

_log = logging.getLogger("lingzhou.execution")

if TYPE_CHECKING:
    from core.config import Config
    from core.judgment import JudgmentOutput
    from memory.working import WorkingMemory, WMItem
    from store.task import TaskStore
    from tools.registry import ToolRegistry


_DURABLE_FAILURE_TTL_SEC = 7200
_DURABLE_FAILURE_THRESHOLD = 3
_LOG_TEXT_CHARS = 240


def _default_durable_failure_policy() -> dict[str, int]:
    return {
        "threshold": _DURABLE_FAILURE_THRESHOLD,
        "ttl_sec": _DURABLE_FAILURE_TTL_SEC,
    }


async def _load_durable_failure_policy(task_store: "TaskStore | None") -> dict[str, int]:
    policy = _default_durable_failure_policy()
    if task_store is None:
        return policy
    raw, found = await task_store.get_fact("control:durable_failure_policy")
    if not found or not raw.strip():
        return policy
    try:
        data = json.loads(raw)
    except Exception:
        return policy
    threshold = int(data.get("threshold") or policy["threshold"])
    ttl_sec = int(data.get("ttl_sec") or policy["ttl_sec"])
    if threshold > 0:
        policy["threshold"] = threshold
    if ttl_sec > 0:
        policy["ttl_sec"] = ttl_sec
    return policy


def action_key_param(params: dict[str, Any] | None) -> str:
    p = params or {}
    return (
        p.get("path")
        or p.get("name")
        or p.get("title")
        or p.get("key")
        or str(p.get("id") or "")
        or p.get("command")
        or p.get("query")
        or ""
    )


def _failure_fact_key(action: "JudgmentOutput") -> str:
    sig = f"{action.chosen_action_id or ''}|{action_key_param(action.params)}"
    digest = hashlib.md5(sig.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"durable_failure:{digest}"


def _classify_durable_failure(result: ToolResult) -> str | None:
    text = "\n".join(x for x in [result.summary, result.error or "", result.evidence] if x).lower()
    patterns = {
        "missing_path": [
            "no such file or directory", "路径不存在", "文件不存在", "未找到", "找不到脚本",
        ],
        "not_a_directory": ["not a directory", "不是目录"],
        "not_a_file": ["not a file", "不是文件"],
        "empty_path": ["path 不能为空", "emptypath"],
        "command_not_found": ["command not found", "工具不存在"],
    }
    for code, needles in patterns.items():
        if any(n in text for n in needles):
            return code
    return None


def _clip_log_text(value: Any, limit: int = _LOG_TEXT_CHARS) -> str:
    text = str(value or "").replace("\n", "\\n").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _tool_result_log_fields(result: ToolResult) -> tuple[str, str, str]:
    log_summary = ""
    if isinstance(result.metadata, dict):
        log_summary = str(result.metadata.get("log_summary") or "").strip()
    summary = _clip_log_text(log_summary or result.summary)
    error = _clip_log_text(result.error or "")
    state = ""
    if isinstance(result.state_delta, dict) and result.state_delta:
        try:
            state = _clip_log_text(json.dumps(result.state_delta, ensure_ascii=False, sort_keys=True))
        except Exception:
            state = _clip_log_text(result.state_delta)
    return summary, error, state


def _worker_log_fields(result: ToolResult) -> str:
    meta = result.metadata if isinstance(result.metadata, dict) else {}
    parts: list[str] = []
    path = str(meta.get("worker_path") or "").strip()
    if path:
        parts.append(f"path={path}")
    mode = str(meta.get("execution_mode") or "").strip()
    if mode:
        parts.append(f"mode={mode}")
    for key, label in (
        ("worker_limit", "limit"),
        ("worker_wait_ms", "wait_ms"),
        ("worker_inflight", "inflight"),
        ("worker_waiting", "queue"),
        ("worker_peak_inflight", "peak"),
        ("dispatch_ms", "dispatch_ms"),
    ):
        value = meta.get(key)
        if value in (None, ""):
            continue
        parts.append(f"{label}={value}")
    monitor = meta.get("run_monitor")
    if isinstance(monitor, dict):
        kind = str(monitor.get("kind") or "").strip()
        if kind:
            parts.append(f"monitor={kind}")
    return " ".join(parts) or "-"


def _worker_limit_for_type(cfg: "Config", worker_type: str) -> int:
    loop_cfg = getattr(cfg, "loop", None)
    attr_name = {
        "tool-chain-worker": "max_tool_chain_workers",
        "exec-worker": "max_exec_workers",
        "multimodal-worker": "max_multimodal_workers",
        "llm-worker": "max_llm_workers",
    }.get(worker_type, "max_tool_chain_workers")
    try:
        return max(1, int(getattr(loop_cfg, attr_name, 1) or 1))
    except (TypeError, ValueError):
        return 1


_TARGET_TASK_TOOLS = frozenset({
    "task.advance",
    "task.complete",
    "task.fail",
    "task.resume",
    "task.steer",
    "task.update",
    "task.wait",
})


def _coerce_task_id(value: Any) -> int:
    try:
        task_id = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return task_id if task_id > 0 else 0


def _planned_run_task_id(action: "JudgmentOutput", active_task_id: int) -> int:
    tool_name = action.chosen_action_id or ""
    if tool_name in _TARGET_TASK_TOOLS:
        return _coerce_task_id((action.params or {}).get("task_id")) or active_task_id
    return active_task_id


def _resolved_run_task_id(result: ToolResult, active_task_id: int) -> int:
    tool_name = ""
    if isinstance(result.metadata, dict):
        tool_name = str(result.metadata.get("tool_name") or "")
    if tool_name in _TARGET_TASK_TOOLS and isinstance(result.metadata, dict):
        return _coerce_task_id(result.metadata.get("task_id")) or active_task_id
    return active_task_id


def _infer_run_profile(
    tool_name: str,
    params: dict[str, Any] | None = None,
    *,
    registry: "ToolRegistry | None" = None,
) -> tuple[str, str]:
    p = params or {}
    if p.get("monitor_fact_key") or p.get("status_fact_key"):
        _log.debug("[run-profile] tool=%s classified as llm-worker via fact monitor", tool_name)
        return "llm", "llm-worker"
    if tool_has_capability(registry, tool_name, "run_spawn"):
        return "exec", "exec-worker"
    if tool_has_capability(registry, tool_name, "multimodal"):
        return "multimodal", "multimodal-worker"
    return "tool_chain", "tool-chain-worker"


def _active_plan_step(task: Any | None) -> str:
    if task is None:
        return ""
    extras = getattr(task, "extras", None)
    if not isinstance(extras, dict):
        return ""
    raw_plan = extras.get("plan")
    if not isinstance(raw_plan, list):
        return ""
    for item in raw_plan:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").strip() != "in_progress":
            continue
        step = str(item.get("step") or "").strip()
        if step:
            return step
    return ""


def _run_status_from_result(result: ToolResult) -> str:
    if (
        isinstance(result.state_delta, dict)
        and result.metadata.get("session_id")
        and result.state_delta.get("process") == "started"
        and result.state_delta.get("background")
    ):
        return "running"
    if result.error and not result.skipped:
        return "failed"
    if result.skipped:
        return "cancelled"
    return "succeeded"


def _run_progress_text(result: ToolResult) -> str:
    if isinstance(result.state_delta, dict):
        progress = str(result.state_delta.get("progress") or "").strip()
        if progress:
            return progress[:2000]
    progress = str(result.metadata.get("progress") or "").strip()
    if progress:
        return progress[:2000]
    return (result.summary or "").strip()[:2000]


def _meta_reflection_decision(target_kind: str, loop_level: str, text: str) -> str:
    lowered = (text or "").lower()
    if any(token in lowered for token in ("rollback", "回滚", "regressed", "regression")):
        return "rollback"
    if loop_level == "double" or target_kind in {"threshold", "routing", "task_split"}:
        return "apply"
    return "defer"


def _record_run_started(
    ctx: ToolContext,
    *,
    run_id: int,
    task_id: int,
    tool_name: str,
    run_type: str,
    worker_type: str,
    model_tier: str,
) -> None:
    if ctx.episodic is None:
        return
    ctx.episodic.record_event(
        "run_started",
        {
            "run_id": run_id,
            "task_id": task_id,
            "tool_name": tool_name,
            "run_type": run_type,
            "worker_type": worker_type,
            "model_tier": model_tier,
        },
    )


def record_run_outcome_memory(
    episodic: Any | None,
    semantic: Any | None,
    *,
    memory_cfg: Any | None,
    run_id: int,
    task_id: int,
    tool_name: str,
    worker_type: str,
    status: str,
    progress: str,
    summary: str,
    error: str,
    evidence: str = "",
) -> None:
    is_failure = bool(error) or status == "failed"
    if episodic is not None:
        event_type = "run_failed" if is_failure or status == "failed" else "run_completed"
        episodic.record_event(
            event_type,
            {
                "run_id": run_id,
                "task_id": task_id,
                "tool_name": tool_name,
                "worker_type": worker_type,
                "status": status,
                "summary": summary[:800],
                "error": error[:400],
            },
        )
    if semantic is None:
        return
    from store.semantic import MemoryNode

    tags = [status]
    if tool_name:
        tags.append(tool_name)
    if worker_type:
        tags.append(worker_type)
    if task_id:
        tags.append(f"task:{task_id}")
    if is_failure and "failed" not in tags:
        tags.append("failed")
    body_parts = [
        f"status={status}",
        f"tool={tool_name or 'unknown'}",
    ]
    if progress:
        body_parts.append(f"progress={progress}")
    if summary:
        body_parts.append(f"summary={summary}")
    if error:
        body_parts.append(f"error={error}")
    if evidence:
        body_parts.append(f"evidence={evidence[:1200]}")
    activation, valence = run_result_memory_affect(
        memory_cfg,
        is_failure=is_failure,
    )
    semantic.upsert(MemoryNode(
        id=f"run-result-{run_id}",
        kind="run_result",
        title=f"[run#{run_id}] {tool_name or 'unknown'} {status}",
        body="\n".join(body_parts)[:4000],
        activation=activation,
        valence=valence,
        tags=tags,
    ))


def record_meta_reflection_memory(
    episodic: Any | None,
    semantic: Any | None,
    meta: dict[str, str | int],
) -> None:
    reflection_id = str(meta.get("reflection_id") or "")
    target_kind = str(meta.get("target_kind") or "")
    loop_level = str(meta.get("loop_level") or "")
    decision = str(meta.get("decision") or "defer")
    task_id = int(meta.get("task_id") or 0)
    run_id = int(meta.get("run_id") or 0)
    tool_name = str(meta.get("tool_name") or "")
    diagnosis = str(meta.get("diagnosis") or "")
    proposal = str(meta.get("proposal") or "")
    verification_plan = str(meta.get("verification_plan") or "")

    if episodic is not None and loop_level == "double":
        episodic.record_event(
            "double_loop_reflection",
            {
                "reflection_id": reflection_id,
                "run_id": run_id,
                "task_id": task_id,
                "tool_name": tool_name,
                "target_kind": target_kind,
                "decision": decision,
            },
        )
    if semantic is None:
        return
    from store.semantic import MemoryNode

    tags = ["meta_reflection", target_kind, loop_level, decision]
    if tool_name:
        tags.append(tool_name)
    if task_id:
        tags.append(f"task:{task_id}")
    semantic.upsert(MemoryNode(
        id=f"meta-reflection-{reflection_id}",
        kind="meta_reflection",
        title=f"[{decision}] {target_kind or 'reflection'} run#{run_id}",
        body=(
            f"diagnosis={diagnosis}\n"
            f"proposal={proposal}\n"
            f"verification_plan={verification_plan}"
        )[:4000],
        activation=0.8 if decision != "defer" else 0.7,
        valence=0.42 if decision == "rollback" else 0.58,
        tags=tags,
    ))
    if decision in {"apply", "rollback"}:
        rule_target = target_kind or "rule"
        rule_tool = tool_name or "unknown-tool"
        semantic.upsert(MemoryNode(
            id=f"rule-revision-{reflection_id}",
            kind="rule_revision",
            title=f"[{decision}] {rule_target} via {rule_tool} run#{run_id}",
            body=(
                f"target_kind={target_kind}\n"
                f"tool_name={tool_name}\n"
                f"proposal={proposal}\n"
                f"verification_plan={verification_plan}"
            )[:4000],
            activation=0.83,
            valence=0.46 if decision == "rollback" else 0.62,
            tags=[target_kind, decision, tool_name or ""] if tool_name else [target_kind, decision],
        ))


def _should_record_run_outcome(status: str) -> bool:
    return status in {"succeeded", "failed", "cancelled"}


def build_meta_reflection(
    *,
    run_id: int,
    task_id: int,
    tool_name: str,
    result: ToolResult,
) -> dict[str, str | int] | None:
    if not (result.error or result.skipped):
        return None
    text = "\n".join(x for x in [tool_name, result.error or "", result.summary, result.evidence] if x).lower()
    target_kind = "tool"
    trigger = "failure_pattern"
    loop_level = "single"
    diagnosis = f"动作 {tool_name or 'unknown'} 在 run#{run_id} 结束为 {_run_status_from_result(result)}，需要复盘失败来源。"
    proposal = "优先检查工具实现、输入参数或外部资源，然后重试同一动作。"
    verification_plan = "在相同 task 上用同一输入重跑一次，确认错误消失或 summary 改善。"

    if "knownstablefailure" in text:
        target_kind = "threshold"
        diagnosis = f"动作 {tool_name or 'unknown'} 被稳定失败降噪机制拦截，说明当前静默阈值或外部状态需要复查。"
        proposal = "确认外部状态是否恢复；若频繁误杀，则调整 durable failure 阈值或静默策略。"
        verification_plan = "等待静默窗口结束后重跑，并比较是否仍被直接跳过。"
    elif "emptypath" in text or "path 不能为空" in text:
        target_kind = "task_split"
        loop_level = "double"
        diagnosis = f"动作 {tool_name or 'unknown'} 缺少必要资源定位，问题更像任务拆分不完整，而不只是工具报错。"
        proposal = "在创建读取/写入类 run 之前，先增加资源发现或路径确认步骤，再执行目标动作。"
        verification_plan = "先补一条定位资源的子步骤，再重跑原动作，确认不再出现空路径错误。"
    elif "toolnotfound" in text or "工具不存在" in text:
        target_kind = "routing"
        loop_level = "double"
        diagnosis = f"动作 {tool_name or 'unknown'} 未注册，说明判断层的动作选择或工具清单存在漂移。"
        proposal = "校正 action 选择规则或工具清单注入，避免继续选择不存在的动作。"
        verification_plan = "重新做一次 judgment，确认 chosen_action_id 落在已注册工具集合内。"
    decision = _meta_reflection_decision(target_kind, loop_level, text)

    return {
        "reflection_id": f"mr-{uuid.uuid4().hex[:12]}",
        "target_kind": target_kind,
        "trigger": trigger,
        "loop_level": loop_level,
        "diagnosis": diagnosis,
        "proposal": proposal,
        "verification_plan": verification_plan,
        "decision": decision,
        "task_id": task_id,
        "run_id": run_id,
        "tool_name": tool_name,
    }


class ExecutionLayer:
    def __init__(self, registry: "ToolRegistry", cfg: "Config") -> None:
        self._registry = registry
        self._cfg = cfg
        self._workers = WorkerLayer(cfg)

    async def dispatch(self, action: "JudgmentOutput", ctx: ToolContext) -> ToolResult:
        """根据 decision 类型分发执行。"""
        match action.decision:
            case "wait":
                return ToolResult(
                    summary=f"wait: {action.rationale[:200]}",
                    skipped=True,
                    kind="wait",
                    priority=0.3,
                )
            case "pause":
                from memory.working import WMItem
                ctx.wm.add(WMItem(
                    kind="caution",
                    content=f"pause: {action.rationale[:300]}",
                    priority=0.9,
                ))
                return ToolResult(
                    summary=f"pause: {action.rationale[:200]}",
                    skipped=True,
                    kind="pause",
                    priority=0.9,
                )
            case "act":
                if action.parallel_actions:
                    return await self._dispatch_parallel(action, ctx)
                return await self._dispatch_act(action, ctx)
            case _:
                return ToolResult(
                    summary=f"未知决策类型: {action.decision!r}",
                    skipped=True,
                    kind="error",
                )

    async def _dispatch_parallel(self, action: "JudgmentOutput", ctx: ToolContext) -> ToolResult:
        """gather 并行执行 parallel_actions 列表中的多个工具，合并结果返回。"""
        import asyncio
        from core.judgment import JudgmentOutput as _JO

        sub_actions = [
            _JO(
                decision="act",
                chosen_action_id=item["action_id"],
                params=dict(item.get("params") or {}),
                rationale=action.rationale,
            )
            for item in action.parallel_actions
            if isinstance(item, dict) and isinstance(item.get("action_id"), str) and item["action_id"]
        ]
        if not sub_actions:
            return ToolResult(summary="parallel_actions 为空，退化为 wait", skipped=True, kind="wait")

        _log.info(
            "[exec.parallel] launching %d tools: %s",
            len(sub_actions), [a.chosen_action_id for a in sub_actions],
        )
        results: list[ToolResult] = list(await asyncio.gather(
            *[self._dispatch_act(a, ctx) for a in sub_actions]
        ))
        merged_summary = "\n".join(
            f"[{a.chosen_action_id}] {r.summary}"
            for a, r in zip(sub_actions, results)
            if r.summary
        )
        errors = [r.error for r in results if r.error]
        # 合并所有错误信息（不只暴露首个），让 behavior_tracker 和 failure 记录看到全部失败
        combined_error = "; ".join(errors) if errors else None
        return ToolResult(
            summary=merged_summary,
            error=combined_error,
            kind="execute_result",
            priority=max((r.priority for r in results), default=0.9),
            metadata={"parallel_count": len(sub_actions), "errors": errors},
        )

    async def _dispatch_act(self, action: "JudgmentOutput", ctx: ToolContext) -> ToolResult:
        run_id: int | None = None
        run_type = "tool_chain"
        worker_type = "tool-chain-worker"
        active_task = await ctx.task_store.get_active() if ctx.task_store is not None else None
        active_task_id = active_task.id if active_task else 0
        run_task_id = _planned_run_task_id(action, active_task_id)
        task_tier = (active_task.model_tier or "").strip() if active_task is not None else ""
        durable_policy = await _load_durable_failure_policy(ctx.task_store)
        durable_threshold = int(durable_policy.get("threshold") or _DURABLE_FAILURE_THRESHOLD)
        durable_ttl_sec = int(durable_policy.get("ttl_sec") or _DURABLE_FAILURE_TTL_SEC)
        if ctx.task_store is not None:
            run_type, worker_type = _infer_run_profile(
                action.chosen_action_id or "",
                action.params,
                registry=self._registry,
            )
            run_id = await ctx.task_store.add_run(
                task_id=run_task_id,
                run_type=run_type,
                worker_type=worker_type,
                status="running",
                input_json={
                    "decision": action.decision,
                    "tool": action.chosen_action_id or "",
                    "params": action.params or {},
                },
                tool_name=action.chosen_action_id or "",
                model_tier=task_tier,
            )
            if run_id is not None:
                _record_run_started(
                    ctx,
                    run_id=run_id,
                    task_id=run_task_id,
                    tool_name=action.chosen_action_id or "",
                    run_type=run_type,
                    worker_type=worker_type,
                    model_tier=task_tier,
                )
                _log.info(
                    "[run-start] run=%s task=%s tool=%s worker=%s limit=%s tier=%s",
                    run_id,
                    run_task_id,
                    action.chosen_action_id or "-",
                    worker_type,
                    _worker_limit_for_type(self._cfg, worker_type),
                    task_tier or "-",
                )

        def _stamp_result_metadata(tool_result: ToolResult) -> ToolResult:
            tool_result.metadata.setdefault("tool_name", action.chosen_action_id or "")
            tool_result.metadata.setdefault("worker_type", worker_type)
            if run_id is not None:
                tool_result.metadata.setdefault("run_id", run_id)
            return tool_result

        entry = self._registry.get(action.chosen_action_id)
        if not entry:
            _log.warning(
                "[exec-miss] run=%s task=%s tool=%s not registered",
                run_id or 0,
                run_task_id,
                action.chosen_action_id or "-",
            )
            result = _stamp_result_metadata(ToolResult(
                summary=f"工具不存在: {action.chosen_action_id!r}",
                error="ToolNotFound",
                skipped=True,
                kind="error",
            ))
            await self._finalize_run(run_id, result, ctx)
            return result

        if self._cfg.loop.debug:
            _log.debug("[exec] %s params=%s", action.chosen_action_id, action.params)
        _log.info("[exec] %s", action.chosen_action_id)

        failure_key = _failure_fact_key(action)

        # Pre-dispatch: 检查该动作是否处于 durable mute 期
        if ctx.task_store is not None:
            raw_mute, mute_found = await ctx.task_store.get_fact(failure_key)
            if mute_found and raw_mute:
                try:
                    mute_data = json.loads(raw_mute)
                    muted_until = float(mute_data.get("muted_until") or 0)
                    if muted_until > time.time():
                        _log.info(
                            "[exec-mute] run=%s task=%s tool=%s reason=%s count=%s muted_until=%s",
                            run_id or 0,
                            run_task_id,
                            action.chosen_action_id or "-",
                            mute_data.get("reason") or "-",
                            mute_data.get("count") or 0,
                            int(muted_until),
                        )
                        result = _stamp_result_metadata(ToolResult(
                            summary=(
                                f"跳过已知稳定失败动作 {action.chosen_action_id!r}："
                                f" {mute_data.get('last_summary', '')[:200]}"
                            ),
                            error="KnownStableFailure",
                            skipped=True,
                            kind="error",
                        ))
                        await self._finalize_run(run_id, result, ctx)
                        return result
                except Exception:
                    pass

        # Pre-dispatch: plan gate — 有未对齐的 in_progress 步骤时阻止变更类操作
        action_id = action.chosen_action_id or ""
        _is_mutation = not action_id.startswith("task.") and action_id not in {
            "memory.get_fact", "file.read", "file.list", "file.search",
        }
        if _is_mutation and active_task is not None and ctx.task_store is not None:
            _plan = active_task.extras.get("plan") if active_task.extras else None
            if _plan:
                _in_progress_steps = [s.get("step", "") for s in _plan if s.get("status") == "in_progress"]
                _cur_step = (active_task.current_step or "").strip()
                if _in_progress_steps and not _cur_step:
                    _step_name = _in_progress_steps[0]
                    _log.info(
                        "[exec-gate] run=%s task=%s tool=%s blocked step=%s",
                        run_id or 0,
                        run_task_id,
                        action.chosen_action_id or "-",
                        _step_name,
                    )
                    result = _stamp_result_metadata(ToolResult(
                        summary=f"当前步骤未对齐，请先完成「{_step_name}」再执行变更操作",
                        error="PlanStepMismatch",
                        skipped=True,
                        kind="error",
                    ))
                    await self._finalize_run(run_id, result, ctx)
                    return result

        dispatch_started = time.monotonic()
        try:
            result = await self._workers.dispatch(worker_type, entry, action, ctx)
        except Exception as exc:
            _log.exception(
                "[exec-error] run=%s task=%s tool=%s worker=%s dispatch raised",
                run_id or 0,
                run_task_id,
                action.chosen_action_id or "-",
                worker_type,
            )
            result = ToolResult(
                summary=f"工具执行异常: {exc}",
                evidence=str(exc),
                error=str(exc),
                kind="execute_result",
            )
        result = _stamp_result_metadata(result)
        result.metadata.setdefault("dispatch_ms", int((time.monotonic() - dispatch_started) * 1000))

        _summary_log, _error_log, _state_log = _tool_result_log_fields(result)
        _worker_log = _worker_log_fields(result)
        _log.info(
            "[tool-result] tool=%s worker=%s worker_meta=%s skipped=%s error=%s summary=%s state=%s",
            action.chosen_action_id,
            worker_type,
            _worker_log,
            result.skipped,
            _error_log or "-",
            _summary_log or "-",
            _state_log or "-",
        )

        # 失败时写入 failures 表，绑定当前任务（P2-B 任务边界原则）
        if result.error and not result.skipped and ctx.task_store is not None:
            task_id = str(_resolved_run_task_id(result, run_task_id) or "")
            await ctx.task_store.record_failure(
                kind=action.chosen_action_id,
                summary=result.summary[:300],
                context=result.evidence[:200],
                task_id=task_id,
            )

        # 更新 durable failure 状态（对所有“可识别的确定性失败”生效）
        if ctx.task_store is not None:
            reason = _classify_durable_failure(result)
            if result.error and reason:
                raw, found = await ctx.task_store.get_fact(failure_key)
                prev: dict[str, Any] = {}
                if found:
                    try:
                        prev = json.loads(raw)
                    except Exception:
                        prev = {}
                count = int(prev.get("count") or 0) + 1 if prev.get("reason") == reason else 1
                payload = {
                    "tool": action.chosen_action_id,
                    "key": action_key_param(action.params),
                    "reason": reason,
                    "count": count,
                    "last_summary": result.summary[:200],
                    "last_seen": time.time(),
                    "muted_until": time.time() + durable_ttl_sec if count >= durable_threshold else 0,
                    "policy_threshold": durable_threshold,
                    "policy_ttl_sec": durable_ttl_sec,
                }
                await ctx.task_store.set_fact(failure_key, json.dumps(payload, ensure_ascii=False), scope="system")
            elif not result.error:
                await ctx.task_store.set_fact(
                    failure_key,
                    json.dumps({
                        "tool": action.chosen_action_id,
                        "key": action_key_param(action.params),
                        "reason": "",
                        "count": 0,
                        "last_summary": result.summary[:200],
                        "last_seen": time.time(),
                        "muted_until": 0,
                    }, ensure_ascii=False),
                    scope="system",
                )

        await self._finalize_run(run_id, result, ctx, active_task_id=run_task_id or None)
        return result

    async def _finalize_run(
        self,
        run_id: int | None,
        result: ToolResult,
        ctx: ToolContext,
        *,
        active_task_id: int | None = None,
    ) -> None:
        if run_id is None or ctx.task_store is None:
            return
        result.metadata.setdefault("run_id", run_id)
        if isinstance(result.state_delta, dict):
            result.state_delta.setdefault("run_id", run_id)
        resolved_task_id = _resolved_run_task_id(result, active_task_id or 0)
        status = _run_status_from_result(result)
        progress = _run_progress_text(result)
        await ctx.task_store.update_run(
            run_id,
            task_id=resolved_task_id,
            status=status,
            output_json=result.to_dict(),
            log_text=result.summary[:4000],
            error_text=result.error or "",
            session_id=str(result.metadata.get("session_id") or ""),
            progress=progress,
        )
        _summary_log, _error_log, _state_log = _tool_result_log_fields(result)
        _worker_log = _worker_log_fields(result)
        _log.info(
            "[run-finalize] run=%s task=%s status=%s tool=%s worker=%s worker_meta=%s progress=%s error=%s state=%s",
            run_id,
            resolved_task_id,
            status,
            str(result.metadata.get("tool_name") or "-"),
            str(result.metadata.get("worker_type") or "-"),
            _worker_log,
            _clip_log_text(progress or "") or "-",
            _error_log or "-",
            _state_log or "-",
        )
        if _should_record_run_outcome(status):
            record_run_outcome_memory(
                ctx.episodic,
                ctx.semantic,
                memory_cfg=getattr(ctx.config, "memory", None),
                run_id=run_id,
                task_id=resolved_task_id,
                tool_name=str(result.metadata.get("tool_name") or ""),
                worker_type=str(result.metadata.get("worker_type") or ""),
                status=status,
                progress=progress,
                summary=result.summary,
                error=result.error or "",
                evidence=result.evidence,
            )
        if resolved_task_id:
            await ctx.task_store.update_task_result(
                resolved_task_id,
                build_task_run_result_patch(
                    run_id=run_id,
                    status=status,
                    worker_type=str(result.metadata.get("worker_type") or ""),
                    tool_name=str(result.metadata.get("tool_name") or ""),
                    session_id=str(result.metadata.get("session_id") or ""),
                    summary=result.summary,
                    error=result.error,
                ),
            )
        meta = build_meta_reflection(
            run_id=run_id,
            task_id=resolved_task_id,
            tool_name=str(result.metadata.get("tool_name") or ""),
            result=result,
        )
        if meta:
            await ctx.task_store.add_meta_reflection(
                reflection_id=str(meta["reflection_id"]),
                target_kind=str(meta["target_kind"]),
                trigger=str(meta["trigger"]),
                loop_level=str(meta["loop_level"]),
                diagnosis=str(meta["diagnosis"]),
                proposal=str(meta["proposal"]),
                verification_plan=str(meta["verification_plan"]),
                decision=str(meta["decision"]),
                task_id=int(meta["task_id"]),
                run_id=int(meta["run_id"]),
                tool_name=str(meta["tool_name"]),
            )
            record_meta_reflection_memory(ctx.episodic, ctx.semantic, meta)
