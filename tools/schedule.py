"""tools/schedule.py — 调度信号工具（cron 机制）。

数字生命的时间感知层：让灵舟能设置备忘录、定期反思触发器、自动提醒。

信号触发后通过 WM 注入本轮认知上下文，优先级=0.9（高于普通工作记忆）。
重复信号在 ack 后自动推进 run_at（无漂移，基于上次计划时间递增）。
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool


def _parse_run_at(run_at_str: str | None) -> str:
    """将用户输入的时间字符串规范化为 ISO8601 UTC。

    支持格式：
    - ISO8601（含 T，带/不带时区）：原样保留或转换为 UTC
    - 纯日期 'YYYY-MM-DD'：当天 00:00 UTC
    - 偏移量 '+3600'（秒）或 '+1h' / '+30m' / '+1d'：相对当前时间
    """
    if not run_at_str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    s = run_at_str.strip()

    # 相对偏移：+1h / +30m / +1d / +3600
    if s.startswith("+"):
        offset_str = s[1:]
        seconds = 0
        if offset_str.endswith("d"):
            seconds = int(offset_str[:-1]) * 86400
        elif offset_str.endswith("h"):
            seconds = int(offset_str[:-1]) * 3600
        elif offset_str.endswith("m"):
            seconds = int(offset_str[:-1]) * 60
        else:
            seconds = int(offset_str)
        dt = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ISO8601 with timezone
    if "T" in s:
        s_norm = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s_norm)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass

    # 纯日期
    try:
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        pass

    raise ValueError(f"无法解析时间：{run_at_str!r}，支持格式：ISO8601 / +1h / +30m / +1d / +3600")


@tool(ToolManifest(
    name="schedule.add",
    description=(
        "添加一条调度信号（备忘录/提醒/定期触发器）。"
        "到期时自动注入工作记忆，触发当轮认知响应。"
        "run_at 支持：ISO8601、+1h、+30m、+1d、+3600 秒偏移。"
        "repeat_secs>0 时为重复信号（如 86400=每天）。"
    ),
    params=[
        ToolParam("title", "string", "信号标题，将出现在 WM 提醒中", required=True),
        ToolParam("run_at", "string",
                  "触发时间：ISO8601 UTC / +1h / +30m / +1d / +3600 等", required=True),
        ToolParam("repeat_secs", "number",
                  "重复间隔秒数（0=一次性，86400=每天，3600=每小时）", required=False),
        ToolParam("note", "string", "附加说明，存入 payload", required=False),
    ],
))
async def schedule_add(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    title = (params.get("title") or "").strip()
    if not title:
        return ToolResult(summary="title 不能为空", skipped=True)

    run_at_raw = params.get("run_at") or ""
    try:
        run_at = _parse_run_at(run_at_raw)
    except ValueError as exc:
        return ToolResult(summary=str(exc), skipped=True)

    repeat_secs = int(params.get("repeat_secs") or 0)
    note = (params.get("note") or "").strip()
    payload = {"note": note} if note else {}

    sig_id = await ctx.task_store.add_signal(title, run_at, repeat_secs, payload)
    repeat_desc = f"，每 {repeat_secs}s 重复" if repeat_secs else "，一次性"
    return ToolResult(
        summary=f"已添加调度信号 #{sig_id}：{title}，触发时间={run_at}{repeat_desc}",
        evidence=f"id={sig_id}",
    )


@tool(ToolManifest(
    name="schedule.list",
    description=(
        "列出调度信号。注意：重复信号由 loop 自动触发并自动 ack，status='pending' 表示'活跃计划'而非'待手动处理'。"
        "⚡ 表示当前已到期（需处理），⏰ 表示未来触发（无需处理）。"
        "仅在需要查看计划列表或管理信号时使用，不要用此工具确认信号是否已处理——信号触发时 WM 中已有完整提醒。"
    ),
    params=[
        ToolParam("include_done", "boolean", "是否包含已完成信号，默认 false", required=False),
        ToolParam("limit", "number", "最多返回条数，默认 20", required=False),
    ],
))
async def schedule_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from datetime import datetime, timezone
    include_done = bool(params.get("include_done", False))
    limit = int(params.get("limit") or 20)
    sigs = await ctx.task_store.list_signals(limit=limit, include_done=include_done)
    if not sigs:
        return ToolResult(summary="暂无调度信号", evidence="")
    now_utc = datetime.now(timezone.utc)
    lines = []
    for s in sigs:
        repeat = f" 重复{s['repeat_secs']}s" if s["repeat_secs"] else ""
        try:
            run_at_str = s["run_at"].replace("Z", "+00:00")
            run_at_dt = datetime.fromisoformat(run_at_str)
            if run_at_dt.tzinfo is None:
                run_at_dt = run_at_dt.replace(tzinfo=timezone.utc)
            due_tag = "⚡到期" if run_at_dt <= now_utc else f"⏰{s['run_at']}"
        except Exception:
            due_tag = s["run_at"]
        _payload = s.get("payload") or {}
        _note = (_payload.get("note") or "").strip()
        note_part = f" | 内容：{_note}" if _note else ""
        lines.append(f"#{s['id']} [{due_tag}]{repeat} — {s['title']}{note_part}")
    detail = "\n".join(lines)
    return ToolResult(summary=f"共 {len(sigs)} 条信号:\n{detail}", evidence=detail)


@tool(ToolManifest(
    name="schedule.ack",
    description=(
        "确认一条调度信号已处理完毕。"
        "一次性信号标记为 done；重复信号自动推进到下次触发时间。"
        "处理完 schedule.list 返回的信号后必须调用此工具，否则信号将持续出现在 pending 列表。"
    ),
    params=[
        ToolParam("id", "number", "信号 id（由 schedule.list 查询）", required=True),
    ],
))
async def schedule_ack(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    sig_id = params.get("id")
    if sig_id is None:
        return ToolResult(summary="id 不能为空", skipped=True)
    await ctx.task_store.ack_signal(int(sig_id))
    return ToolResult(summary=f"已确认信号 #{sig_id} 处理完毕", evidence=f"id={sig_id} acked")


@tool(ToolManifest(
    name="schedule.cancel",
    description="取消一条调度信号（按 id）",
    params=[
        ToolParam("id", "number", "信号 id（由 schedule.list 查询）", required=True),
    ],
))
async def schedule_cancel(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    sig_id = params.get("id")
    if sig_id is None:
        return ToolResult(summary="id 不能为空", skipped=True)
    sig = await ctx.task_store.get_signal(int(sig_id))
    if sig is None:
        return ToolResult(summary=f"信号 #{sig_id} 不存在", skipped=True)
    await ctx.task_store.cancel_signal(int(sig_id))
    return ToolResult(
        summary=f"已取消信号 #{sig_id}: {sig['title']}",
        evidence=f"id={sig_id} title={sig['title'][:60]} run_at={sig['run_at']} status=cancelled",
    )
