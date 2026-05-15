"""tools/memory_ops.py — 记忆操作工具。"""
from __future__ import annotations

import json
import uuid
from typing import Any

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool
from memory.working import WMItem
from memory.semantic import MemoryNode

_PRIORITY_ALIASES = {"high": 0.9, "medium": 0.6, "mid": 0.6, "low": 0.3, "critical": 1.0}


def _coerce_optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else str(value).strip()


def _coerce_fact_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple, bool, int, float)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()

def _parse_float(val: Any, default: float) -> float:
    """把 '0.8' / 'high' / 0.8 / None 都安全转成 float。"""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    if s in _PRIORITY_ALIASES:
        return _PRIORITY_ALIASES[s]
    try:
        return float(s)
    except ValueError:
        return default


@tool(ToolManifest(
    name="memory.add_wm",
    description="向工作记忆添加一条观察或结论",
    params=[
        ToolParam("content", "string", "要记录的内容", required=True),
        ToolParam("kind", "string", "类型标签，如 observation/conclusion/caution", required=False),
        ToolParam("priority", "number", "优先级 0-1，默认 0.8", required=False),
    ],
))
async def memory_add_wm(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    content = (params.get("content") or "").strip()
    if not content:
        return ToolResult(summary="内容不能为空", skipped=True)
    kind = params.get("kind") or "observation"
    priority = _parse_float(params.get("priority"), 0.8)
    ctx.wm.add(WMItem(kind=kind, content=content, priority=priority))
    return ToolResult(summary=f"已写入工作记忆: {content[:80]}", evidence=f"kind={kind}")


@tool(ToolManifest(
    name="memory.add_semantic",
    description="将知识或技能固化到语义（长期）记忆",
    params=[
        ToolParam("title", "string", "节点标题", required=True),
        ToolParam("body", "string", "节点内容", required=True),
        ToolParam("kind", "string", "节点类型: learned_skill/fact/observation", required=False),
        ToolParam("activation", "number", "初始激活值 0-1，默认 0.7", required=False),
    ],
))
async def memory_add_semantic(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    title = (params.get("title") or "").strip()
    body = (params.get("body") or "").strip()
    if not title or not body:
        return ToolResult(summary="title 和 body 不能为空", skipped=True)
    node = MemoryNode(
        id=f"node-{uuid.uuid4().hex[:12]}",
        kind=params.get("kind") or "observation",
        title=title,
        body=body,
        activation=_parse_float(params.get("activation"), 0.7),
    )
    ctx.semantic.upsert(node)
    return ToolResult(
        summary=f"已写入语义记忆: {title}",
        evidence=f"node_id={node.id}",
    )


@tool(ToolManifest(
    name="memory.set_fact",
    description="设置一个持久化 key-value 事实",
    params=[
        ToolParam("key", "string", "事实 key", required=True),
        ToolParam("value", "string", "事实 value", required=True),
        ToolParam("scope", "string", "作用域，默认 general", required=False),
    ],
))
async def memory_set_fact(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    key = (params.get("key") or "").strip()
    value = _coerce_fact_value(params.get("value"))
    if not key:
        return ToolResult(summary="key 不能为空", skipped=True)
    await ctx.task_store.set_fact(key, value, scope=params.get("scope") or "general")
    return ToolResult(summary=f"已设置事实: {key}={value[:80]}", evidence=f"key={key}")


@tool(ToolManifest(
    name="memory.search",
    description="搜索语义记忆节点。当你需要先回忆再行动时使用。",
    params=[
        ToolParam("query", "string", "搜索查询", required=True),
        ToolParam("top_k", "number", "返回条数，默认 5", required=False),
        ToolParam("kind", "string", "仅返回指定 kind 的节点", required=False),
        ToolParam("tag", "string", "仅返回包含指定 tag 的节点", required=False),
        ToolParam("task_id", "string", "仅返回包含 task:{id} tag 的节点", required=False),
        ToolParam("path_prefix", "string", "仅返回标题/正文/tag 中包含该路径前缀的节点", required=False),
        ToolParam("id_prefix", "string", "仅返回 id 以该前缀开头的节点", required=False),
    ],
))
async def memory_search(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = _coerce_optional_text(params.get("query"))
    if not query:
        return ToolResult(summary="query 不能为空", skipped=True)
    top_k = int(params.get("top_k") or 5)
    hits = ctx.semantic.retrieve(
        query,
        top_k=top_k,
        kind=_coerce_optional_text(params.get("kind")) or None,
        tag=_coerce_optional_text(params.get("tag")) or None,
        task_id=_coerce_optional_text(params.get("task_id")) or None,
        path_prefix=_coerce_optional_text(params.get("path_prefix")) or None,
        id_prefix=_coerce_optional_text(params.get("id_prefix")) or None,
    )
    if not hits:
        return ToolResult(summary=f"没有找到与 {query!r} 相关的语义记忆", skipped=True)
    lines = []
    for i, hit in enumerate(hits, 1):
        title = str(hit.get("title") or "")
        body = str(hit.get("body") or "")[:180]
        score = hit.get("score")
        score_part = ""
        if isinstance(score, (int, float)):
            score_part = f" (score={float(score):.3f})"
        lines.append(f"[{i}] {title}{score_part}\n{body}")
    return ToolResult(summary="\n\n".join(lines))


@tool(ToolManifest(
    name="memory.get_fact",
    description="读取一个持久化 key-value 事实",
    params=[
        ToolParam("key", "string", "事实 key", required=True),
    ],
))
async def memory_get_fact(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    key = (params.get("key") or "").strip()
    if not key:
        return ToolResult(summary="key 不能为空", skipped=True)
    value, found = await ctx.task_store.get_fact(key)
    if not found:
        return ToolResult(summary=f"事实不存在: {key}", skipped=True)
    return ToolResult(summary=f"{key} = {value}", evidence=f"key={key}")


@tool(ToolManifest(
    name="failure.dismiss",
    description="豁免指定失败记录，同 kind 的失败以后不再重复记录",
    params=[
        ToolParam("failure_id", "number", "失败记录 ID", required=True),
    ],
))
async def failure_dismiss(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    fid = int(params.get("failure_id") or 0)
    if not fid:
        return ToolResult(summary="failure_id 不能为空", skipped=True)
    await ctx.task_store.dismiss_failure(fid)
    return ToolResult(summary=f"已豁免失败记录 #{fid}", evidence=f"failure_id={fid}")


@tool(ToolManifest(
    name="reflect.structural",
    description="将当前工作记忆的高优先级内容合成为一条结构性洞察，写入语义记忆",
    params=[
        ToolParam("insight", "string", "洞察摘要（1-3句）", required=True),
        ToolParam("title", "string", "洞察标题（简短）", required=False),
    ],
))
async def reflect_structural(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    insight = (params.get("insight") or "").strip()
    if not insight:
        return ToolResult(summary="洞察内容不能为空", skipped=True)

    title = (params.get("title") or "").strip() or insight[:50]
    wm_summary = "\n".join(
        f"  [{i['kind']}] {i['content'][:80]}"
        for i in ctx.wm.get_top(8)
    )
    body = f"{insight}\n\n来源（工作记忆摘要）:\n{wm_summary}" if wm_summary else insight

    node = MemoryNode(
        id=f"reflect-{uuid.uuid4().hex[:12]}",
        kind="structural",
        title=title,
        body=body,
        activation=0.85,
    )
    ctx.semantic.upsert(node)

    # 同时写入情节记忆，保留推理轨迹
    task = await ctx.task_store.get_active()
    ctx.episodic.record(
        role="reflection",
        content=f"**{title}**\n\n{insight}",
        task_id=str(task.id) if task else None,
    )
    return ToolResult(
        summary=f"结构性洞察已写入语义记忆: {title}",
        evidence=f"node_id={node.id}",
        priority=0.5,  # 洞察记录本身不需要长时间留在 WM
    )


@tool(ToolManifest(
    name="memory.snapshot",
    description="【由 runtime 自动调用，禁止手动调用】快照当前工作记忆与运行时状态摘要，写入情节记忆供复盘，然后清空工作记忆（释放 WM 压力）",
    params=[],
))
async def memory_snapshot(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    wm_items = ctx.wm.get_top(20)
    failures = await ctx.task_store.list_failures(limit=5)
    task = await ctx.task_store.get_active()

    pressure_before = ctx.wm.pressure

    lines = [
        f"WM 条目: {len(wm_items)}  压力: {pressure_before:.0%}",
        f"近期失败: {len(failures)} 条",
        f"情绪: valence={ctx.emotion.valence:.2f} arousal={ctx.emotion.arousal:.2f}",
        f"活跃任务: {task.title[:60] if task else '无'}",
        "",
        "工作记忆前 5 条:",
    ]
    for item in wm_items[:5]:
        lines.append(f"  [{item['kind']}] {item['content'][:80]}")

    snapshot_text = "\n".join(lines)
    ctx.episodic.record(
        role="snapshot",
        content=snapshot_text,
        task_id=str(task.id) if task else None,
    )

    # 快照后清空 WM，保留身份镀樔（快照的语义：持久化草稿 → 清空草稿）
    ctx.wm.clear(preserve_kinds={"bootstrap_identity"})

    return ToolResult(
        summary=f"运行时快照已记录并清空 WM（{pressure_before:.0%} → 0%）\n{snapshot_text[:200]}",
        evidence=f"wm_before={len(wm_items)} failures={len(failures)}",
        priority=0.4,  # snapshot 结果本身是低价值记录，不应积压 WM
    )
