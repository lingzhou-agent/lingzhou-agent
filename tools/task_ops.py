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


@tool(ToolManifest(
    name="task.advance",
    description="将活跃任务推进到 in_progress 状态并更新 next_step（首次取任务时调用）",
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
    params=[
        ToolParam("title", "string", "任务标题（简洁）", required=True),
        ToolParam("goal", "string", "任务目标（详细）", required=False),
        ToolParam("priority", "string", "优先级: low/normal/high/critical", required=False),
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
    task_id = await ctx.task_store.add_task(
        title,
        goal,
        priority,
        source="external",
        chain_id=chain_id,
        parent_task_id=str(parent.id) if parent else (str(parent_task_id or "") if parent_task_id else ""),
        current_step=current_step,
        next_step=next_step,
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
    params=[
        ToolParam("task_id", "number", "可选：显式指定要完成的任务 id；不传则使用当前 active task", required=False),
    ],
))
async def task_complete(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="无活跃任务可完成", skipped=True)

    await ctx.task_store.update_status(task.id, "done", "completed via agent")

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
    params=[
        ToolParam("task_id", "number", "可选：显式指定要更新的任务 id；不传则使用当前 active task", required=False),
        ToolParam("next_step", "string", "下一步计划", required=False),
        ToolParam("status", "string", "新状态: pending/ready/in_progress/resumed/waiting/blocked/failed", required=False),
        ToolParam("current_step", "string", "当前步骤名", required=False),
    ],
))
async def task_update(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="无活跃任务", skipped=True)
    status = params.get("status") or task.status
    next_step = params.get("next_step") or task.next_step
    current_step = (params.get("current_step") or "").strip()
    if current_step:
        await ctx.task_store.update_task_data(task.id, {"current_step": current_step})
    await ctx.task_store.update_status(task.id, status, next_step)
    return ToolResult(
        summary=f"任务 [{task.id}] 已更新: status={status}",
        evidence=f"task_id={task.id} next_step={next_step[:80]}",
        resource_key=str(task.id),
        state_delta={"task_status": status, "next_step": next_step, "current_step": current_step or task.current_step},
        metadata={"task_id": task.id, "chain_id": task.chain_id},
    )


@tool(ToolManifest(
    name="task.fail",
    description="将当前活跃任务标记为失败，记录失败原因并写入失败日志（触发进化反馈）",
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
    wait_kind = (params.get("wait_kind") or "").strip()
    if not wait_kind:
        return ToolResult(summary="wait_kind 不能为空", skipped=True)
    wait_key = (params.get("wait_key") or "").strip()
    current_step = (params.get("current_step") or "").strip()
    next_step = (params.get("next_step") or task.next_step or "").strip()
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
    current_step = (params.get("current_step") or "").strip()
    next_step = (params.get("next_step") or task.next_step or "").strip()
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
        state_delta={"task_status": status, "current_step": current_step, "next_step": next_step},
        metadata={"task_id": task.id, "chain_id": task.chain_id, "status": status},
    )
