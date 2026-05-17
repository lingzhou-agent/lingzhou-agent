"""tools/config_ops.py — config.get / config.set 工具（LLM 可自主调参）。

LLM 通过这些工具读取和修改自己的配置，无需人工编辑文件。
修改后自动触发热重载（loop 检测 mtime 变化）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.registry import tool, ToolManifest, ToolResult, ToolParam, ToolContext

CONFIG_PATH = Path("~/.lingzhou/lingzhou.json").expanduser()


def _read_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def _nested_get(d: dict, path: str) -> Any:
    """点号路径读取，如 'loop.max_idle_gap' → cfg['loop']['max_idle_gap']。"""
    keys = path.split(".")
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return None
    return current


def _nested_set(d: dict, path: str, value: Any) -> None:
    """点号路径写入，自动创建中间字典。"""
    keys = path.split(".")
    current = d
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


@tool(ToolManifest(
    name="config.get",
    description=(
        "读取配置文件中某个键的值。支持点号嵌套路径。\n"
        "示例: loop.max_idle_gap → 返回 45\n"
        "      evolution.trigger_min_failures → 返回 3"
    ),
    progress_category="info",
    params=[ToolParam("key", "string", "配置键（支持点号路径）", required=True)],
))
async def config_get(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    key = (params.get("key") or "").strip()
    if not key:
        return ToolResult(summary="key 不能为空", error="EmptyKey", skipped=True)

    try:
        cfg = _read_config()
        value = _nested_get(cfg, key)
        if value is None and "." not in key:
            return ToolResult(
                summary=f"键 '{key}' 不存在或为 null。可用: {', '.join(k for k in cfg if not k.startswith('_'))}",
                skipped=True,
            )
        return ToolResult(
            summary=f"{key} = {json.dumps(value, ensure_ascii=False)}",
            evidence=str(value),
            metadata={"key": key, "value": value},
        )
    except Exception as e:
        return ToolResult(summary=f"读取失败: {e}", error="ConfigError")


@tool(ToolManifest(
    name="config.set",
    description=(
        "修改配置文件中的某个值。支持点号嵌套路径。修改后 loop 自动热重载。\n"
        "可调的常见参数:\n"
        "  loop.max_idle_gap — 空闲等待上限(秒)\n"
        "  loop.min_act_gap — 动作间隔(秒)\n"
        "  loop.chat_reply_timeout — 聊天回复超时(秒)\n"
        "  evolution.enabled — 是否启用自进化\n"
        "  evolution.trigger_min_failures — 触发进化所需失败数\n"
        "  evolution.error_streak_evolve — 错误连击立即触发\n"
        "  memory.working_capacity — 工作记忆容量\n"
    ),
    progress_category="mutation",
    params=[
        ToolParam("key", "string", "配置键（支持点号路径）", required=True),
        ToolParam("value", "string", "新值（JSON 格式）", required=True),
    ],
))
async def config_set(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    key = (params.get("key") or "").strip()
    value_raw = params.get("value")
    if not key:
        return ToolResult(summary="key 不能为空", error="EmptyKey", skipped=True)

    try:
        value = json.loads(str(value_raw)) if isinstance(value_raw, str) else value_raw
    except json.JSONDecodeError:
        value = value_raw  # 字符串值，如 "deepseek/deepseek-v4-pro"

    try:
        cfg = _read_config()
        old = _nested_get(cfg, key)
        _nested_set(cfg, key, value)
        _write_config(cfg)
        return ToolResult(
            summary=f"✅ {key}: {json.dumps(old, ensure_ascii=False)} → {json.dumps(value, ensure_ascii=False)}",
            evidence=f"{key}={value}",
            metadata={"key": key, "old": old, "new": value},
            state_delta={"config_changed": key},
        )
    except Exception as e:
        return ToolResult(summary=f"写入失败: {e}", error="ConfigError")
