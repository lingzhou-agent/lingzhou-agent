"""tools/task_ops.py — 任务管理工具（供 LLM 通过判断层调用）。"""
from __future__ import annotations

import uuid
from typing import Any

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool
from memory.semantic import MemoryNode


@tool(ToolManifest(
    name="task.advance",
    description="将活跃任务推进到 in_progress 状态并更新 next_step（首次取任务时调用）",
    params=[
        ToolParam("next_step", "string", "计划的下一步描述", required=False),
    ],
))
async def task_advance(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await ctx.task_store.get_active()
    if not task:
        return ToolResult(summary="无活跃任务可推进", skipped=True)
    if task.status == "in_progress":
        return ToolResult(summary=f"任务 [{task.id}] 已在进行中", skipped=True)
    next_step = (params.get("next_step") or "").strip() or task.next_step
    await ctx.task_store.update_status(task.id, "in_progress", next_step)
    return ToolResult(
        summary=f"任务 [{task.id}] 已推进至 in_progress: {task.title[:60]}",
        evidence=f"task_id={task.id} next_step={next_step[:80]}",
    )


@tool(ToolManifest(
    name="task.add",
    description="创建一个新任务",
    params=[
        ToolParam("title", "string", "任务标题（简洁）", required=True),
        ToolParam("goal", "string", "任务目标（详细）", required=False),
        ToolParam("priority", "string", "优先级: low/normal/high/critical", required=False),
    ],
))
async def task_add(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    title = (params.get("title") or "").strip()
    if not title:
        return ToolResult(summary="任务标题不能为空", skipped=True)
    goal = params.get("goal") or ""
    priority = params.get("priority") or "normal"
    task_id = await ctx.task_store.add_task(title, goal, priority, source="external")
    return ToolResult(
        summary=f"任务已创建: [{task_id}] {title}",
        evidence=f"task_id={task_id}",
    )


@tool(ToolManifest(
    name="task.complete",
    description="将当前活跃任务标记为完成，并将任务叙事编译进语义记忆",
    params=[],
))
async def task_complete(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await ctx.task_store.get_active()
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
        )

    return ToolResult(
        summary=f"任务 [{task.id}] 已完成",
        evidence=f"task_id={task.id}",
    )


@tool(ToolManifest(
    name="task.list",
    description="列出任务列表",
    params=[
        ToolParam("status", "string", "过滤状态: pending/in_progress/done/all", required=False),
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
    lines = [f"[{t.id}] [{t.status}] [{t.priority}] {t.title}" for t in tasks]
    return ToolResult(summary="\n".join(lines), evidence=f"count={len(tasks)}")


@tool(ToolManifest(
    name="task.update",
    description="更新当前活跃任务的 next_step 或状态",
    params=[
        ToolParam("next_step", "string", "下一步计划", required=False),
        ToolParam("status", "string", "新状态: in_progress/failed/pending", required=False),
    ],
))
async def task_update(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await ctx.task_store.get_active()
    if not task:
        return ToolResult(summary="无活跃任务", skipped=True)
    status = params.get("status") or task.status
    next_step = params.get("next_step") or task.next_step
    await ctx.task_store.update_status(task.id, status, next_step)
    return ToolResult(
        summary=f"任务 [{task.id}] 已更新: status={status}",
        evidence=f"task_id={task.id} next_step={next_step[:80]}",
    )


@tool(ToolManifest(
    name="task.fail",
    description="将当前活跃任务标记为失败，记录失败原因并写入失败日志（触发进化反馈）",
    params=[
        ToolParam("reason", "string", "失败原因摘要", required=True),
    ],
))
async def task_fail(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await ctx.task_store.get_active()
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
    )
