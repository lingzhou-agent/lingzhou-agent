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

from core.worker import WorkerLayer
from tools.registry import ToolResult, ToolContext

_log = logging.getLogger("lingzhou.execution")

if TYPE_CHECKING:
    from core.config import Config
    from core.judgment import JudgmentOutput
    from memory.working import WorkingMemory, WMItem
    from memory.task_store import TaskStore
    from tools.registry import ToolRegistry


_DURABLE_FAILURE_TTL_SEC = 7200
_DURABLE_FAILURE_THRESHOLD = 3

_EXEC_RUN_TOOLS = frozenset({"exec", "process.write", "process.poll", "process.log", "process.kill", "process.list"})
_MULTIMODAL_RUN_TOOLS = frozenset({"image.analyze"})


def _action_key_param(params: dict[str, Any] | None) -> str:
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
    sig = f"{action.chosen_action_id or ''}|{_action_key_param(action.params)}"
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


def _infer_run_profile(tool_name: str) -> tuple[str, str]:
    if tool_name in _EXEC_RUN_TOOLS:
        return "exec", "exec-worker"
    if tool_name in _MULTIMODAL_RUN_TOOLS:
        return "multimodal", "multimodal-worker"
    return "tool_chain", "tool-chain-worker"


def _run_status_from_result(result: ToolResult) -> str:
    if (
        isinstance(result.state_delta, dict)
        and result.state_delta.get("process") == "started"
        and result.state_delta.get("background")
        and result.metadata.get("session_id")
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


def _record_run_outcome(
    ctx: ToolContext,
    *,
    run_id: int,
    task_id: int,
    tool_name: str,
    worker_type: str,
    status: str,
    progress: str,
    result: ToolResult,
) -> None:
    is_failure = bool(result.error)
    if ctx.episodic is not None:
        event_type = "run_failed" if is_failure or status == "failed" else "run_completed"
        ctx.episodic.record_event(
            event_type,
            {
                "run_id": run_id,
                "task_id": task_id,
                "tool_name": tool_name,
                "worker_type": worker_type,
                "status": status,
                "summary": result.summary[:800],
                "error": (result.error or "")[:400],
            },
        )
    if ctx.semantic is None:
        return
    from memory.semantic import MemoryNode

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
    if result.summary:
        body_parts.append(f"summary={result.summary}")
    if result.error:
        body_parts.append(f"error={result.error}")
    if result.evidence:
        body_parts.append(f"evidence={result.evidence[:1200]}")
    ctx.semantic.upsert(MemoryNode(
        id=f"run-result-{run_id}",
        kind="run_result",
        title=f"[run#{run_id}] {tool_name or 'unknown'} {status}",
        body="\n".join(body_parts)[:4000],
        activation=0.82 if is_failure or status == "failed" else 0.72,
        valence=0.35 if is_failure or status == "failed" else 0.65,
        tags=tags,
    ))


def _record_meta_reflection(ctx: ToolContext, meta: dict[str, str | int]) -> None:
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

    if ctx.episodic is not None and loop_level == "double":
        ctx.episodic.record_event(
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
    if ctx.semantic is None:
        return
    from memory.semantic import MemoryNode

    tags = ["meta_reflection", target_kind, loop_level, decision]
    if tool_name:
        tags.append(tool_name)
    if task_id:
        tags.append(f"task:{task_id}")
    ctx.semantic.upsert(MemoryNode(
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
        ctx.semantic.upsert(MemoryNode(
            id=f"rule-revision-{reflection_id}",
            kind="rule_revision",
            title=f"[{decision}] {target_kind or 'rule'}",
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


def _build_meta_reflection(
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
        self._workers = WorkerLayer()

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
                return await self._dispatch_act(action, ctx)
            case _:
                return ToolResult(
                    summary=f"未知决策类型: {action.decision!r}",
                    skipped=True,
                    kind="error",
                )

    async def _dispatch_act(self, action: "JudgmentOutput", ctx: ToolContext) -> ToolResult:
        run_id: int | None = None
        run_type = "tool_chain"
        worker_type = "tool-chain-worker"
        active_task = await ctx.task_store.get_active() if ctx.task_store is not None else None
        task_tier = (active_task.model_tier or "").strip() if active_task is not None else ""
        if ctx.task_store is not None:
            run_type, worker_type = _infer_run_profile(action.chosen_action_id or "")
            run_id = await ctx.task_store.add_run(
                task_id=active_task.id if active_task else 0,
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
                    task_id=active_task.id if active_task else 0,
                    tool_name=action.chosen_action_id or "",
                    run_type=run_type,
                    worker_type=worker_type,
                    model_tier=task_tier,
                )

        entry = self._registry.get(action.chosen_action_id)
        if not entry:
            result = ToolResult(
                summary=f"工具不存在: {action.chosen_action_id!r}",
                error="ToolNotFound",
                skipped=True,
                kind="error",
            )
            await self._finalize_run(run_id, result, ctx)
            return result

        if self._cfg.loop.debug:
            _log.debug("[exec] %s params=%s", action.chosen_action_id, action.params)
        _log.info("[exec] %s", action.chosen_action_id)

        # durable failure sensing：对稳定重复失败的确定性动作做短期持久降噪
        failure_key = _failure_fact_key(action)
        if ctx.task_store is not None:
            raw, found = await ctx.task_store.get_fact(failure_key)
            if found:
                try:
                    info = json.loads(raw)
                except Exception:
                    info = {}
                muted_until = float(info.get("muted_until") or 0)
                count = int(info.get("count") or 0)
                reason = str(info.get("reason") or "stable_failure")
                if count >= _DURABLE_FAILURE_THRESHOLD and muted_until > time.time():
                    result = ToolResult(
                        summary=(
                            f"跳过已知稳定失败动作：{action.chosen_action_id} {_action_key_param(action.params)}\n"
                            f"原因: {reason}；最近已连续失败 {count} 次。"
                            " 若外部状态已修复，请等待静默窗口结束后重试，或更换动作/参数。"
                        ),
                        evidence=raw,
                        error="KnownStableFailure",
                        skipped=True,
                        kind="execute_result",
                        priority=0.4,
                    )
                    await self._finalize_run(run_id, result, ctx)
                    return result

        try:
            result = await self._workers.dispatch(worker_type, entry, action, ctx)
        except Exception as exc:
            result = ToolResult(
                summary=f"工具执行异常: {exc}",
                evidence=str(exc),
                error=str(exc),
                kind="execute_result",
            )

        # 失败时写入 failures 表，绑定当前任务（P2-B 任务边界原则）
        if result.error and not result.skipped and ctx.task_store is not None:
            task_id = str(active_task.id) if active_task else ""
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
                    "key": _action_key_param(action.params),
                    "reason": reason,
                    "count": count,
                    "last_summary": result.summary[:200],
                    "last_seen": time.time(),
                    "muted_until": time.time() + _DURABLE_FAILURE_TTL_SEC if count >= _DURABLE_FAILURE_THRESHOLD else 0,
                }
                await ctx.task_store.set_fact(failure_key, json.dumps(payload, ensure_ascii=False), scope="system")
            elif not result.error:
                await ctx.task_store.set_fact(
                    failure_key,
                    json.dumps({
                        "tool": action.chosen_action_id,
                        "key": _action_key_param(action.params),
                        "reason": "",
                        "count": 0,
                        "last_summary": result.summary[:200],
                        "last_seen": time.time(),
                        "muted_until": 0,
                    }, ensure_ascii=False),
                    scope="system",
                )

        await self._finalize_run(run_id, result, ctx, active_task_id=active_task.id if active_task else None)
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
        status = _run_status_from_result(result)
        progress = _run_progress_text(result)
        await ctx.task_store.update_run(
            run_id,
            status=status,
            output_json=result.to_dict(),
            log_text=result.summary[:4000],
            error_text=result.error or "",
            session_id=str(result.metadata.get("session_id") or ""),
            progress=progress,
        )
        if _should_record_run_outcome(status):
            _record_run_outcome(
                ctx,
                run_id=run_id,
                task_id=active_task_id or 0,
                tool_name=str(result.metadata.get("tool_name") or ""),
                worker_type=str(result.metadata.get("worker_type") or ""),
                status=status,
                progress=progress,
                result=result,
            )
        if active_task_id:
            await ctx.task_store.update_task_result(
                active_task_id,
                {
                    "last_run_id": run_id,
                    "last_run_status": status,
                    "worker_type": str(result.metadata.get("worker_type") or ""),
                    "tool_name": str(result.metadata.get("tool_name") or ""),
                    "session_id": str(result.metadata.get("session_id") or ""),
                    "summary": result.summary,
                    "error": result.error,
                },
            )
        meta = _build_meta_reflection(
            run_id=run_id,
            task_id=active_task_id or 0,
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
            _record_meta_reflection(ctx, meta)
