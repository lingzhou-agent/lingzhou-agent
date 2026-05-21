"""tools/plan.py — task.plan 工具。

让 LLM 为当前任务维护结构化执行计划。
每步三种状态：pending / in_progress / completed，至多一步 in_progress。
"""

from __future__ import annotations

import json
from typing import Any

from tools.registry import tool, ToolManifest, ToolResult, ToolParam, ToolContext


@tool(ToolManifest(
    name="task.plan",
    description=(
        "为当前任务维护结构化执行计划。\n"
        "plan 是一个步骤列表，每步有 step（描述）和 status（pending/in_progress/completed）。\n"
        "至多一个步骤标记为 in_progress。适合非平凡的、多步骤工作。"
    ),
    progress_category="info",
    capabilities=("plan_bootstrap_exempt", "plan_alignment_exempt"),
    params=[
        ToolParam("plan", "object", "步骤列表: [{\"step\":\"...\", \"status\":\"pending|in_progress|completed\"}]", required=True),
        ToolParam("task_id", "number", "任务 ID（可选，默认当前活跃任务）", required=False),
    ],
))
async def task_plan(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    plan_raw = params.get("plan")
    if not plan_raw:
        return ToolResult(summary="plan 不能为空", error="EmptyPlan", skipped=True)

    # 解析 plan
    if isinstance(plan_raw, str):
        try:
            plan_raw = json.loads(plan_raw)
        except json.JSONDecodeError:
            return ToolResult(summary="plan 必须是 JSON 数组", error="InvalidPlan", skipped=True)

    if not isinstance(plan_raw, list) or len(plan_raw) == 0:
        return ToolResult(summary="plan 必须是非空数组", error="InvalidPlan", skipped=True)

    # 验证步骤
    valid_statuses = {"pending", "in_progress", "completed"}
    in_progress_count = 0
    clean_plan = []
    for i, s in enumerate(plan_raw):
        if not isinstance(s, dict):
            return ToolResult(summary=f"plan[{i}] 必须是对象", error="InvalidStep", skipped=True)
        step_text = str(s.get("step", "")).strip()
        if not step_text:
            return ToolResult(summary=f"plan[{i}].step 不能为空", error="InvalidStep", skipped=True)
        status = str(s.get("status", "pending")).strip()
        if status not in valid_statuses:
            return ToolResult(summary=f"plan[{i}].status 无效: {status}，可选 {valid_statuses}", error="InvalidStatus", skipped=True)
        if status == "in_progress":
            in_progress_count += 1
        clean_plan.append({"step": step_text, "status": status})

    if in_progress_count > 1:
        return ToolResult(
            summary=f"至多一个步骤为 in_progress，当前有 {in_progress_count} 个。",
            error="TooManyInProgress",
            skipped=True,
        )

    # 找到当前任务
    task_id = params.get("task_id")
    task = None
    if ctx.task_store:
        if task_id:
            task = await ctx.task_store.get_task_by_id(int(task_id))
        else:
            task = await ctx.task_store.get_active()

    if not task:
        return ToolResult(
            summary="未找到任务。请先创建任务，或指定 task_id。",
            error="NoTask",
            skipped=True,
        )

    # 提前计算统计量（幂等检查和摘要均依赖）
    done = sum(1 for s in clean_plan if s["status"] == "completed")
    pending = sum(1 for s in clean_plan if s["status"] == "pending")

    # 幂等检查：已有 in_progress 步骤且新 plan 无进度提升 → 拒绝，引导 LLM 直接执行。
    # 不比较步骤文本——LLM 会微变描述绕过精确匹配；只看进度是否推进即可。
    existing_plan = (task.extras or {}).get("plan") or []
    if existing_plan and isinstance(existing_plan, list):
        existing_done = sum(1 for s in existing_plan if isinstance(s, dict) and s.get("status") == "completed")
        existing_in_progress = next(
            (s.get("step", "") for s in existing_plan if isinstance(s, dict) and s.get("status") == "in_progress"),
            None,
        )
        if (
            existing_in_progress is not None
            and len(clean_plan) <= len(existing_plan)
            and done <= existing_done
        ):
            hint = f"，请立即执行当前步骤：{existing_in_progress}" if existing_in_progress else "，请直接执行计划中的下一步骤"
            return ToolResult(
                summary=f"计划已有进行中步骤，无需重新规划（{len(existing_plan)} 步，{existing_done}✅）{hint}",
                error="PlanUnchanged",
                skipped=True,
            )

    # 持久化 plan 到任务 extras
    await ctx.task_store.update_task_data(task.id, {"plan": clean_plan})

    # 生成摘要
    active = next((s["step"] for s in clean_plan if s["status"] == "in_progress"), None)
    lines = [f"任务 #{task.id} 计划已更新: {len(clean_plan)} 步 ({done}✅ {in_progress_count}🔄 {pending}⏳)"]
    for i, s in enumerate(clean_plan, 1):
        icon = "✅" if s["status"] == "completed" else "🔄" if s["status"] == "in_progress" else "⏳"
        lines.append(f"  [{i}] {icon} {s['step']}")

    return ToolResult(
        summary="\n".join(lines),
        evidence=json.dumps(clean_plan, ensure_ascii=False),
        metadata={"task_id": task.id, "steps": len(clean_plan), "done": done, "pending": pending},
        state_delta={"plan_steps": len(clean_plan), "plan_done": done},
    )
