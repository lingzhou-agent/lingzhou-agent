"""tools/task_ops.py — 任务管理工具（供 LLM 通过判断层调用）。"""
from __future__ import annotations

import uuid
from typing import Any


def _resolve_task(task_id: Any, ctx: ToolContext):
    if task_id is None:
        return ctx.task_store.get_active()
    try:
        return ctx.task_store.get_task_by_id(int(task_id))
    except Exception:
        return ctx.task_store.get_active()

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool
from memory.semantic import MemoryNode


_INFO_ONLY_COMPLETION_TOOLS = frozenset({
    "file.read", "file.list",
    "memory.search", "memory.get_fact",
    "task.list", "schedule.list",
    "skill.list", "skill.search",
    "shell.capabilities",
})

# 会修改文件系统/代码/DB 的工具 — 需要后续验证才算有效完成
_MUTATION_TOOLS = frozenset({
    "file.edit", "file.write", "file.delete",
    # 注意：shell.run 不在此列表。虽然 shell 可执行破坏性命令，
    # 但它同时是 _VERIFY_TOOLS 的唯一成员，放入会导致死循环：
    # shell.run(验证) → 被视为新 mutation → 要求再次验证 → 无限循环。
    # 文件级 mutation 的验证需求由 file.edit/write/delete 触发已足够。
})

# 验证型动作 — 能确认修改是否正确的工具/模式
_VERIFY_TOOLS = frozenset({
    "shell.run",  # 通常用于跑测试/验证
})


@tool(ToolManifest(
    name="task.advance",
    description="将活跃任务推进到 in_progress 状态并更新 next_step（首次取任务时调用）",
    progress_category="mutation",
        params=[
        ToolParam("task_id", "number", "可选：显式指定要推进的任务 id；不传则使用当前 active task", required=False),
        ToolParam("next_step", "string", "计划的下一步描述", required=False),
    ],
))
async def task_advance(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="无活跃任务可推进", skipped=True)
    if task.status == "in_progress":
        return ToolResult(summary=f"任务 [{task.id}] 已在进行中", skipped=True)
    next_step = (params.get("next_step") or "").strip() or task.next_step
    await ctx.task_store.update_status(task.id, "in_progress", next_step)
    return ToolResult(
        summary=f"任务 [{task.id}] 已推进至 in_progress: {task.title[:60]}",
        evidence=f"task_id={task.id} next_step={next_step[:80]}",
        resource_key=str(task.id),
        state_delta={"task_status": "in_progress", "next_step": next_step},
        metadata={"task_id": task.id, "next_step": next_step, "chain_id": task.chain_id},
    )


@tool(ToolManifest(
    name="task.add",
    description="创建一个新任务",
    progress_category="mutation",
        params=[
        ToolParam("title", "string", "任务标题（简洁）", required=True),
        ToolParam("goal", "string", "任务目标（详细）", required=False),
        ToolParam("priority", "string", "优先级: low/normal/high/critical", required=False),
        ToolParam("model_tier", "string", "可选：任务默认模型层级 reader/reasoner/repair", required=False),
        ToolParam("chain_id", "string", "可选：任务链 id；不传则自动继承或创建", required=False),
        ToolParam("parent_task_id", "number", "可选：父任务 id，用于形成任务链", required=False),
        ToolParam("current_step", "string", "可选：当前步骤名", required=False),
        ToolParam("next_step", "string", "可选：下一步说明", required=False),
    ],
))
async def task_add(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    title = (params.get("title") or "").strip()
    if not title:
        return ToolResult(summary="任务标题不能为空", skipped=True)
    goal = params.get("goal") or ""
    priority = params.get("priority") or "normal"
    parent_task_id = params.get("parent_task_id")
    parent = await _resolve_task(parent_task_id, ctx) if parent_task_id is not None else await ctx.task_store.get_active()
    chain_id = (params.get("chain_id") or (parent.chain_id if parent and parent.chain_id else f"chain-{uuid.uuid4().hex[:8]}"))
    current_step = (params.get("current_step") or "").strip()
    next_step = (params.get("next_step") or "").strip()
    model_tier = (params.get("model_tier") or "").strip()
    task_id = await ctx.task_store.add_task(
        title,
        goal,
        priority,
        source="external",
        chain_id=chain_id,
        parent_task_id=str(parent.id) if parent else (str(parent_task_id or "") if parent_task_id else ""),
        current_step=current_step,
        next_step=next_step,
        model_tier=model_tier,
    )
    return ToolResult(
        summary=f"任务已创建: [{task_id}] {title}",
        evidence=f"task_id={task_id}",
        resource_key=str(task_id),
        state_delta={"task_status": "pending", "chain_id": chain_id},
        metadata={"task_id": task_id, "chain_id": chain_id, "parent_task_id": str(parent.id) if parent else ""},
    )


@tool(ToolManifest(
    name="task.complete",
    description="将当前活跃任务标记为完成，并将任务叙事编译进语义记忆",
    progress_category="mutation",
        params=[
        ToolParam("task_id", "number", "可选：显式指定要完成的任务 id；不传则使用当前 active task", required=False),
        ToolParam("force", "boolean", "可选：强制完成，跳过轻量证据门槛", required=False),
    ],
))
async def task_complete(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="无活跃任务可完成", skipped=True)

    force = bool(params.get("force") or False)
    if not force and ctx.task_store is not None:
        recent_runs = await ctx.task_store.list_runs(task_id=task.id, limit=12)
        recent_tools = [
            r.tool_name for r in recent_runs
            if r.status == "succeeded" and r.tool_name and not r.tool_name.startswith("task.")
        ]
        only_info_browsing = bool(recent_tools) and all(t in _INFO_ONLY_COMPLETION_TOOLS for t in recent_tools)
        explicit_progress = bool((task.current_step or "").strip() or str(task.result_json.get("summary") or "").strip())
        if only_info_browsing and not explicit_progress:
            return ToolResult(
                summary=(
                    f"任务 [{task.id}] 暂不允许完成：最近仅有信息浏览类动作，"
                    "且缺少 current_step / result summary 等明确结论。"
                    "请先更新当前步骤或产出结论，再完成任务。"
                ),
                error="InsufficientEvidence",
                skipped=True,
                metadata={"task_id": task.id, "recent_tools": recent_tools},
            )

        # 新门槛：如果最近有 mutation（file.edit/write/delete），
        # 要求其后至少有一个成功的验证动作（如跑测试）。
        # 注意：list_runs 返回的是 ORDER BY id DESC（最新在前），
        # 因此用 min(idx) 找最新 mutation，[:idx] 取比它更新的工具。
        has_mutation = any(t in _MUTATION_TOOLS for t in recent_tools)
        if has_mutation:
            # 最新 mutation 的索引（DESC 顺序下 min = 最近）
            latest_mutation_idx = min(i for i, t in enumerate(recent_tools) if t in _MUTATION_TOOLS)
            # 比最新 mutation 更近的工具（索引更小）
            post_mutation_tools = recent_tools[:latest_mutation_idx]
            verified_after = any(t in _VERIFY_TOOLS for t in post_mutation_tools)
            if not verified_after:
                return ToolResult(
                    summary=(
                        f"任务 [{task.id}] 暂不允许完成：最近有修改动作"
                        f"（{recent_tools[latest_mutation_idx]}），"
                        "但缺少后续验证（如 shell.run 跑测试）。"
                        "请先验证修改正确性，再完成任务。"
                    ),
                    error="MutationWithoutVerification",
                    skipped=True,
                    metadata={
                        "task_id": task.id,
                        "last_mutation": recent_tools[latest_mutation_idx],
                        "post_mutation_tools": post_mutation_tools,
                    },
                )

    await ctx.task_store.update_status(task.id, "done", "completed via agent")

    # 自动 dismiss 该任务关联的未消除 failure（任务既然完成，旧失败已不再阻塞）
    task_failures = await ctx.task_store.list_failures_for_task(str(task.id), limit=30)
    dismissed_count = 0
    for f in task_failures:
        if not f.dismissed:
            await ctx.task_store.dismiss_failure(f.id)
            dismissed_count += 1

    # 程序性记忆编译：任务叙事 → 语义记忆节点（Anderson 1983 ACT-R）
    task_id_str = str(task.id)
    narrative = ctx.episodic.load_for_context(task_id_str, max_chars=1200)
    if narrative.strip():
        node = MemoryNode(
            id=f"skill-{uuid.uuid4().hex[:12]}",
            kind="learned_skill",
            title=f"完成: {task.title[:80]}",
            body=narrative[:1200],
            activation=0.8,
            valence=0.5,
        )
        ctx.semantic.upsert(node)
        return ToolResult(
            summary=f"任务 [{task.id}] 已完成，叙事已编译进语义记忆",
            evidence=f"task_id={task.id} skill_node={node.id}",
            resource_key=str(task.id),
            state_delta={"task_status": "done", "compiled_skill": node.id},
            metadata={"task_id": task.id, "skill_node": node.id, "chain_id": task.chain_id},
        )

    return ToolResult(
        summary=f"任务 [{task.id}] 已完成",
        evidence=f"task_id={task.id}",
        resource_key=str(task.id),
        state_delta={"task_status": "done"},
        metadata={"task_id": task.id, "chain_id": task.chain_id},
    )


@tool(ToolManifest(
    name="task.list",
    description="列出任务列表",
    progress_category="info",
        params=[
        ToolParam("status", "string", "过滤状态: pending/in_progress/ready/resumed/waiting/done/all", required=False),
        ToolParam("limit", "number", "最多返回条数，默认 10", required=False),
    ],
))
async def task_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    status = params.get("status") or None
    if status == "all":
        status = None
    limit = int(params.get("limit") or 10)
    tasks = await ctx.task_store.list_tasks(status=status, limit=limit)
    if not tasks:
        return ToolResult(summary="没有匹配的任务", skipped=True)
    lines = []
    for t in tasks:
        chain = f" chain={t.chain_id}" if t.chain_id else ""
        wait = f" wait={t.wait_kind}:{t.wait_key}" if t.wait_kind else ""
        step = f" step={t.current_step}" if t.current_step else ""
        lines.append(f"[{t.id}] [{t.status}] [{t.priority}] {t.title}{chain}{step}{wait}")
    return ToolResult(summary="\n".join(lines), evidence=f"count={len(tasks)}", metadata={"count": len(tasks)})


@tool(ToolManifest(
    name="task.update",
    description="更新当前活跃任务的 next_step 或状态",
    progress_category="mutation",
        params=[
        ToolParam("task_id", "number", "可选：显式指定要更新的任务 id；不传则使用当前 active task", required=False),
        ToolParam("next_step", "string", "下一步计划", required=False),
        ToolParam("status", "string", "新状态: pending/ready/in_progress/resumed/waiting/blocked/failed", required=False),
        ToolParam("current_step", "string", "当前步骤名", required=False),
        ToolParam("model_tier", "string", "可选：任务默认模型层级 reader/reasoner/repair", required=False),
    ],
))
async def task_update(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="无活跃任务", skipped=True)
    status = params.get("status") or task.status
    next_step = str(params.get("next_step") or "").strip() if "next_step" in params else task.next_step
    current_step = str(params.get("current_step") or "").strip() if "current_step" in params else task.current_step
    model_tier = str(params.get("model_tier") or "").strip() if "model_tier" in params else task.model_tier
    if "current_step" in params:
        await ctx.task_store.update_task_data(task.id, {"current_step": current_step})
    if "model_tier" in params:
        await ctx.task_store.update_task_data(task.id, {"model_tier": model_tier})
    await ctx.task_store.update_status(task.id, status, next_step)
    return ToolResult(
        summary=f"任务 [{task.id}] 已更新: status={status}",
        evidence=f"task_id={task.id} next_step={next_step[:80]}",
        resource_key=str(task.id),
        state_delta={"task_status": status, "next_step": next_step, "current_step": current_step, "model_tier": model_tier},
        metadata={"task_id": task.id, "chain_id": task.chain_id},
    )


@tool(ToolManifest(
    name="task.fail",
    description="将当前活跃任务标记为失败，记录失败原因并写入失败日志（触发进化反馈）",
    progress_category="mutation",
        params=[
        ToolParam("task_id", "number", "可选：显式指定要失败的任务 id；不传则使用当前 active task", required=False),
        ToolParam("reason", "string", "失败原因摘要", required=True),
    ],
))
async def task_fail(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="无活跃任务可标记失败", skipped=True)
    reason = (params.get("reason") or "未知原因").strip()
    await ctx.task_store.update_status(task.id, "failed", reason)
    # 自动 dismiss 旧 failure，task 本身的 failed 状态就是最终记录
    task_failures = await ctx.task_store.list_failures_for_task(str(task.id), limit=30)
    for f in task_failures:
        if not f.dismissed:
            await ctx.task_store.dismiss_failure(f.id)
    await ctx.task_store.record_failure(
        kind="task_failure",
        summary=reason,
        context=f"task_id={task.id} title={task.title[:80]}",
        task_id=str(task.id),
    )
    return ToolResult(
        summary=f"任务 [{task.id}] 已标记失败: {reason[:80]}",
        evidence=f"task_id={task.id} reason={reason[:120]}",
        resource_key=str(task.id),
        state_delta={"task_status": "failed", "reason": reason},
        metadata={"task_id": task.id, "chain_id": task.chain_id},
    )


@tool(ToolManifest(
    name="task.wait",
    description="把任务切到 waiting，并记录等待条件（外部结果 / 定时器 / 子任务等）",
    params=[
        ToolParam("task_id", "number", "可选：显式指定任务 id；不传则使用当前 active task", required=False),
        ToolParam("wait_kind", "string", "等待类型，如 process/task/signal/time/external", required=True),
        ToolParam("wait_key", "string", "等待对象键，如 session_id / child_task_id / signal_key", required=False),
        ToolParam("current_step", "string", "当前步骤名", required=False),
        ToolParam("next_step", "string", "恢复后下一步", required=False),
    ],
))
async def task_wait(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="无活跃任务可等待", skipped=True)
    wait_kind = (params.get("wait_kind") or "").strip().lower()
    if not wait_kind:
        return ToolResult(summary="wait_kind 不能为空", skipped=True)
    wait_key = (params.get("wait_key") or "").strip()
    valid_wait_kinds = {"process", "task", "signal", "time", "external"}
    if wait_kind not in valid_wait_kinds:
        return ToolResult(summary=f"不支持的 wait_kind: {wait_kind}", skipped=True)
    current_step = str(params.get("current_step") or "").strip() if "current_step" in params else None
    next_step = str(params.get("next_step") or "").strip() if "next_step" in params else task.next_step
    await ctx.task_store.mark_waiting(
        task.id,
        wait_kind=wait_kind,
        wait_key=wait_key,
        wait_json={"wait_kind": wait_kind, "wait_key": wait_key},
        current_step=current_step,
        next_step=next_step,
    )
    return ToolResult(
        summary=f"任务 [{task.id}] 已进入 waiting: {wait_kind}{'/' + wait_key if wait_key else ''}",
        resource_key=str(task.id),
        state_delta={"task_status": "waiting", "wait_kind": wait_kind, "wait_key": wait_key},
        metadata={"task_id": task.id, "chain_id": task.chain_id, "wait_kind": wait_kind, "wait_key": wait_key},
    )


@tool(ToolManifest(
    name="task.resume",
    description="把 waiting/blocked 的任务恢复到 resumed/ready，并附带恢复结果",
    params=[
        ToolParam("task_id", "number", "要恢复的任务 id", required=True),
        ToolParam("status", "string", "恢复后的状态，默认 resumed，也可设为 ready/in_progress", required=False),
        ToolParam("current_step", "string", "恢复后当前步骤名", required=False),
        ToolParam("next_step", "string", "恢复后下一步", required=False),
    ],
))
async def task_resume(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="找不到要恢复的任务", skipped=True)
    status = (params.get("status") or "resumed").strip()
    current_step = str(params.get("current_step") or "").strip() if "current_step" in params else None
    next_step = str(params.get("next_step") or "").strip() if "next_step" in params else task.next_step
    await ctx.task_store.resume_task(
        task.id,
        status=status,
        current_step=current_step,
        next_step=next_step,
        result_json={"resumed_via": "task.resume"},
    )
    return ToolResult(
        summary=f"任务 [{task.id}] 已恢复: status={status}",
        resource_key=str(task.id),
        state_delta={"task_status": status, "current_step": current_step if current_step is not None else task.current_step, "next_step": next_step if next_step is not None else task.next_step},
        metadata={"task_id": task.id, "chain_id": task.chain_id, "status": status},
    )
