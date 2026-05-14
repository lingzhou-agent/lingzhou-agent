"""tools/skill_ops.py — skills 查询工具。

给 LLM 提供自我感知能力：
- skill.list   列出当前 active skills
- skill.search 按关键词搜索 skills

目的：补足“只知道当前命中的 skills，不知道自己总体有哪些 skills”的缺口。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool


def _load_registry(ctx: ToolContext):
    from core.skill import SkillRegistry

    workspace_dir = Path(ctx.config.loop.workspace_dir)
    skills_dir = workspace_dir / "skills"
    return SkillRegistry(skills_dir=skills_dir)


def _format_skill_line(skill) -> str:
    origin = "builtin" if not getattr(skill, "source_path", "") else "workspace"
    triggers = getattr(skill, "triggers", []) or []
    trig = f" | triggers: {', '.join(triggers[:5])}" if triggers else ""
    return f"- {skill.name} [{origin}] — {skill.description}{trig}"


@tool(ToolManifest(
    name="skill.list",
    description="列出当前可用的 active skills（builtin + workspace）。当你不确定自己有哪些 skills 可用时调用。",
    params=[
        ToolParam("scope", "string", "all|custom|builtin，默认 all", required=False),
        ToolParam("limit", "number", "最多返回多少条，默认 50", required=False),
    ],
))
async def skill_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    reg = _load_registry(ctx)
    skills = reg.all_skills()
    scope = str(params.get("scope") or "all").lower()
    limit = int(params.get("limit") or 50)

    if scope == "custom":
        skills = [s for s in skills if getattr(s, "source_path", "")]
    elif scope == "builtin":
        skills = [s for s in skills if not getattr(s, "source_path", "")]

    skills = sorted(skills, key=lambda s: s.name)[:limit]
    if not skills:
        return ToolResult(summary="（没有匹配的 skills）")
    lines = [_format_skill_line(s) for s in skills]
    return ToolResult(summary=f"当前可用 skills ({len(skills)} 个):\n" + "\n".join(lines))


@tool(ToolManifest(
    name="skill.search",
    description="按关键词搜索当前可用 skills。当你怀疑有某类 skill 但当前没被激活时调用。",
    params=[
        ToolParam("query", "string", "搜索关键词，如 bug/refactor/提醒/交互/学习", required=True),
        ToolParam("limit", "number", "最多返回多少条，默认 20", required=False),
    ],
))
async def skill_search(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = str(params.get("query") or "").strip().lower()
    if not query:
        return ToolResult(summary="query 不能为空", error="EmptyQuery")

    reg = _load_registry(ctx)
    limit = int(params.get("limit") or 20)
    hits = []
    for s in reg.all_skills():
        hay = " ".join([
            s.name,
            s.description or "",
            " ".join(getattr(s, "tags", []) or []),
            " ".join(getattr(s, "triggers", []) or []),
        ]).lower()
        if query in hay:
            hits.append(s)

    hits = sorted(hits, key=lambda s: s.name)[:limit]
    if not hits:
        return ToolResult(summary=f"没有找到与 {query!r} 匹配的 skills")
    lines = [_format_skill_line(s) for s in hits]
    return ToolResult(summary=f"skill.search 命中 {len(hits)} 个:\n" + "\n".join(lines))
