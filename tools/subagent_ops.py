"""tools/subagent_ops.py — 子灵工具。

提供两个工具：
  subagent.run    — 派生子灵执行子任务（Tier-0~Tier-2）
  subagent.absorb — 将子灵语义记忆合并入父灵语义记忆（Tier-3）
"""
from __future__ import annotations

import json
import logging
from typing import Any

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool

_log = logging.getLogger("lingzhou.subagent_ops")


def _normalized_subagent_tags(raw_tags: Any, sub_id: str) -> list[str]:
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for item in raw_tags:
            text = str(item or "").strip()
            if text and text not in tags:
                tags.append(text)
    elif isinstance(raw_tags, str):
        text = raw_tags.strip()
        if text:
            tags.append(text)

    for item in ["subagent", f"subagent:{sub_id}"]:
        if item not in tags:
            tags.append(item)
    return tags

# ── subagent.run ────────────────────────────────────────────────────────────────

_MANIFEST_RUN = ToolManifest(
    name="subagent.run",
    description=(
        "派生一个子灵执行专项子任务。子灵继承父灵的记忆与配置，"
        "拥有独立工作记忆。默认只开放读型/信息型工具，"
        "不会把 run、failure、fact、task/schedule 变更回写到父灵状态；"
        "高权限与变更类工具默认受限。"
        "isolated_memory=true 时使用独立存储命名空间（Tier-1）；"
        "inherit_ethos=true 时继承父灵价值观基线（Tier-2）。"
        "子灵执行完毕后，关键观察注入父灵工作记忆。"
    ),
    progress_category="mutation",
    capabilities=(),
    params=[
        ToolParam("goal", "string", "子灵要完成的具体目标描述", required=True),
        ToolParam("max_ticks", "number", "子灵最多执行的 tick 数，默认 6", required=False),
        ToolParam("allowed_tools", "string", "允许子灵调用的工具名列表，逗号分隔；空=除黑名单外全部可用", required=False),
        ToolParam("isolated_memory", "boolean", "是否使用独立记忆命名空间（Tier-1），默认 false", required=False),
        ToolParam("inherit_ethos", "boolean", "是否继承父灵价值观基线（Tier-2），默认 true", required=False),
        ToolParam("label", "string", "子灵标签，用于竞争进化时标识候选版本", required=False),
    ],
)


@tool(_MANIFEST_RUN)
async def subagent_run(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """派生子灵并等待其执行完毕，结果注入父灵 WM。"""
    goal = (params.get("goal") or "").strip()
    if not goal:
        return ToolResult(summary="子灵目标为空，跳过", skipped=True)

    judgment = ctx.judgment
    execution = ctx.execution
    registry = ctx.registry

    if judgment is None or execution is None or registry is None:
        return ToolResult(
            summary="子灵无法启动：父灵上下文未注入 judgment/execution/registry",
            error="missing_parent_ctx",
        )

    max_ticks = int(params.get("max_ticks") or 6)
    max_ticks = max(1, min(max_ticks, 20))

    allowed_raw = (params.get("allowed_tools") or "").strip()
    allowed_tools: list[str] | None = (
        [t.strip() for t in allowed_raw.split(",") if t.strip()]
        if allowed_raw else None
    )

    # 默认值处理
    isolated_memory = bool(params.get("isolated_memory") or False)
    inherit_ethos_raw = params.get("inherit_ethos")
    inherit_ethos = True if inherit_ethos_raw is None else bool(inherit_ethos_raw)
    label = (params.get("label") or "").strip()

    from core.subagent import SubagentConfig, make_subagent_runner

    sub_cfg = SubagentConfig(
        goal=goal,
        max_ticks=max_ticks,
        allowed_tools=allowed_tools,
        isolated_memory=isolated_memory,
        inherit_ethos=inherit_ethos,
        label=label,
    )

    runner = make_subagent_runner(sub_cfg, ctx, judgment, execution, registry)

    try:
        result = await runner.run()
    except Exception as exc:
        _log.exception("[subagent_ops] 子灵运行异常: %s", exc)
        return ToolResult(
            summary=f"子灵执行异常: {exc}",
            error=str(exc),
        )

    # 结果注入父灵 WM
    try:
        from memory.working import WMItem
        ctx.wm.add(WMItem(
            kind="subagent_result",
            content=result.to_wm_content(),
            priority=0.88,
        ))
    except Exception:
        pass

    status_label = "完成" if result.completed else ("错误" if result.error else "未完成")
    short_summary = (result.last_summary[:120] if result.last_summary else "")
    summary = f"子灵[{result.subagent_id}] {status_label} ticks={result.ticks_run}"
    if short_summary:
        summary += f" | {short_summary}"

    return ToolResult(
        summary=summary,
        metadata={
            "subagent_id": result.subagent_id,
            "goal": result.goal,
            "ticks_run": result.ticks_run,
            "completed": result.completed,
            "error": result.error,
            "observations": result.observations,
            "label": result.label,
            "memory_dir": result.memory_dir,
            "absorbed_memories_count": len(result.absorbed_memories),
            # 序列化待合并节点（供 subagent.absorb 读取）
            "absorbed_memories": result.absorbed_memories,
        },
        state_delta={
            "subagent_completed": result.completed,
            "subagent_ticks": result.ticks_run,
        },
    )


# ── subagent.absorb ─────────────────────────────────────────────────────────────

_MANIFEST_ABSORB = ToolManifest(
    name="subagent.absorb",
    description=(
        "将子灵的语义记忆节点合并入父灵语义记忆（Tier-3 结果合并）。"
        "需传入 subagent.run 返回结果中的 subagent_id 和 absorbed_memories。"
        "父灵可选择性地吸收子灵的学习成果，实现知识传承。"
    ),
    progress_category="mutation",
    capabilities=(),
    params=[
        ToolParam("subagent_id", "string", "子灵 ID（来自 subagent.run 的返回值）", required=True),
        ToolParam("memories_json", "string", "待吸收的节点列表 JSON（来自 subagent.run metadata.absorbed_memories）", required=True),
        ToolParam("max_absorb", "number", "最多吸收的节点数，默认 5", required=False),
    ],
)


@tool(_MANIFEST_ABSORB)
async def subagent_absorb(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """将子灵语义记忆节点合并入父灵语义记忆。"""
    sub_id = (params.get("subagent_id") or "").strip()
    memories_raw = (params.get("memories_json") or "").strip()

    if not sub_id or not memories_raw:
        return ToolResult(summary="缺少 subagent_id 或 memories_json", skipped=True)
    if ctx.semantic is None:
        return ToolResult(summary="父灵语义记忆未注入，无法吸收子灵结果", error="missing_semantic")

    try:
        nodes: list[dict] = json.loads(memories_raw)
    except Exception as exc:
        return ToolResult(summary=f"memories_json 解析失败: {exc}", error=str(exc))

    if not isinstance(nodes, list):
        return ToolResult(summary="memories_json 格式错误：应为列表", error="bad_format")

    requested_total = len(nodes)
    max_absorb = int(params.get("max_absorb") or 5)
    max_absorb = max(0, max_absorb)
    nodes = nodes[:max_absorb]
    truncated = max(0, requested_total - len(nodes))

    absorbed = 0
    invalid = 0
    errors: list[str] = []

    for idx, node_dict in enumerate(nodes, start=1):
        try:
            from store.semantic import MemoryNode
            if not isinstance(node_dict, dict):
                invalid += 1
                continue

            raw_id = str(node_dict.get("id") or "").strip() or f"node-{idx}"
            title = str(node_dict.get("title") or "").strip()
            body = str(node_dict.get("body") or "")
            if not title or not body.strip():
                invalid += 1
                continue

            node_kwargs: dict[str, Any] = {
                "id": f"absorbed-{sub_id}-{raw_id}",
                "kind": str(node_dict.get("kind") or "subagent_learn"),
                "title": title,
                "body": body,
                "activation": float(node_dict.get("activation", 0.4) or 0.4),
                "valence": float(node_dict.get("valence", 0.5) or 0.5),
                "importance": float(node_dict.get("importance", 0.0) or 0.0),
                "tags": _normalized_subagent_tags(node_dict.get("tags"), sub_id),
                "source": f"subagent:{sub_id}",
            }
            created_at = str(node_dict.get("created_at") or "").strip()
            if created_at:
                node_kwargs["created_at"] = created_at

            # 构造 MemoryNode（跳过缺失必须字段的节点）
            node = MemoryNode(**node_kwargs)
            ctx.semantic.upsert(node)
            absorbed += 1
        except Exception as exc:
            errors.append(str(exc)[:80])

    summary = f"子灵[{sub_id}] 已吸收 {absorbed}/{len(nodes)} 条语义记忆"
    if truncated:
        summary += f"（另有 {truncated} 条因 max_absorb 未吸收）"
    if invalid:
        summary += f"（{invalid} 条缺少标题或正文已跳过）"
    if errors:
        summary += f"（{len(errors)} 条失败）"

    return ToolResult(
        summary=summary,
        metadata={
            "subagent_id": sub_id,
            "absorbed": absorbed,
            "selected_total": len(nodes),
            "requested_total": requested_total,
            "truncated": truncated,
            "invalid": invalid,
            "errors": errors,
        },
        state_delta={"absorbed_memories": absorbed},
    )

