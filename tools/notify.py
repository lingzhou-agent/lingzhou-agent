"""tools/notify.py — 主动通知工具（支持微信通道）。

用法：lingzhou 可主动调用 wechat.send 向用户发送微信消息，
无需等待用户先发消息。需要先通过 lingzhou gateway setup --channel wechat 完成配置。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool

_GW_PATH = Path("~/.lingzhou/gateway/wechat.json").expanduser()


@tool(ToolManifest(
    name="wechat.send",
    description=(
        "主动通过微信给用户发消息（无需用户先发消息触发）。\n"
        "to_user 省略时自动发给最近联系的用户（wechat:last_user）。\n"
        "需要先运行 lingzhou gateway setup --channel wechat 完成微信通道配置。"
    ),
    prefer_tier="reasoner",
    params=[
        ToolParam("text", "string", "要发送的消息内容", required=True),
        ToolParam("to_user", "string", "目标微信用户ID（省略则发给 wechat:last_user）", required=False),
    ],
))
async def wechat_send(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    text = (params.get("text") or "").strip()
    if not text:
        return ToolResult(summary="text 不能为空", error="EmptyText", skipped=True)

    # ── 读取网关配置 ────────────────────────────────────────────────────────────
    if not _GW_PATH.exists():
        return ToolResult(
            summary="微信通道未配置，请先运行: lingzhou gateway setup --channel wechat",
            error="NotConfigured",
            skipped=True,
        )
    try:
        gw_conf: dict[str, Any] = json.loads(_GW_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return ToolResult(summary=f"读取微信配置失败: {exc}", error="ConfigReadError", skipped=True)

    base_url: str = gw_conf.get("base_url", "https://ilinkai.weixin.qq.com")
    token: str = gw_conf.get("token", "")
    if not token:
        return ToolResult(summary="微信 token 未配置", error="NoToken", skipped=True)

    # ── 确定目标用户 ────────────────────────────────────────────────────────────
    to_user: str = (params.get("to_user") or "").strip()
    if not to_user:
        val, found = await ctx.task_store.get_fact("wechat:last_user")
        if not found or not val:
            return ToolResult(
                summary=(
                    "未找到最近联系用户，请通过 to_user 参数指定目标微信用户ID，"
                    "或先收到一条来自用户的微信消息以自动记录。"
                ),
                error="NoTargetUser",
                skipped=True,
            )
        to_user = val.strip()

    # ── 获取 iLink context_token（可选，提高会话连贯性）─────────────────────────
    ctx_token_val, ctx_found = await ctx.task_store.get_fact(f"wechat:ctx:{to_user}")
    ctx_token: str | None = ctx_token_val.strip() if ctx_found and ctx_token_val else None

    # ── 发送 ───────────────────────────────────────────────────────────────────
    try:
        from channels.wechat import send_text  # 动态导入避免循环
        send_text(base_url, token, to_user, text, ctx_token)
    except Exception as exc:
        return ToolResult(summary=f"微信发送失败: {exc}", error="SendError")

    preview = text[:60] + ("…" if len(text) > 60 else "")
    return ToolResult(
        summary=f"✅ 已通过微信发送给 {to_user[:20]}: {preview}",
        evidence=f"to={to_user}, len={len(text)}",
    )
