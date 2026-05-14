"""core/execution.py — 执行层。

职责：
- 接收 JudgmentOutput，dispatch 到具体工具
- 处理 act / pause / wait 三种决策
- 失败时写入 failures 表（绑定当前任务 ID，P2-B 原则）
- 对稳定重复失败的确定性动作做持久降噪（durable failure sensing）
- 返回 ToolResult 给 loop 层整合
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from tools.registry import ToolResult, ToolContext

_log = logging.getLogger("lingzhou.execution")

if TYPE_CHECKING:
    from core.config import Config
    from core.judgment import JudgmentOutput
    from memory.working import WorkingMemory, WMItem
    from memory.task_store import TaskStore
    from tools.registry import ToolRegistry


_DURABLE_FAILURE_TTL_SEC = 7200
_DURABLE_FAILURE_THRESHOLD = 3


def _action_key_param(params: dict[str, Any] | None) -> str:
    p = params or {}
    return (
        p.get("path")
        or p.get("name")
        or p.get("title")
        or p.get("key")
        or str(p.get("id") or "")
        or p.get("command")
        or p.get("query")
        or ""
    )


def _failure_fact_key(action: "JudgmentOutput") -> str:
    sig = f"{action.chosen_action_id or ''}|{_action_key_param(action.params)}"
    digest = hashlib.md5(sig.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"durable_failure:{digest}"


def _classify_durable_failure(result: ToolResult) -> str | None:
    text = "\n".join(x for x in [result.summary, result.error or "", result.evidence] if x).lower()
    patterns = {
        "missing_path": [
            "no such file or directory", "路径不存在", "文件不存在", "未找到", "找不到脚本",
        ],
        "not_a_directory": ["not a directory", "不是目录"],
        "not_a_file": ["not a file", "不是文件"],
        "empty_path": ["path 不能为空", "emptypath"],
        "command_not_found": ["command not found", "工具不存在"],
    }
    for code, needles in patterns.items():
        if any(n in text for n in needles):
            return code
    return None


class ExecutionLayer:
    def __init__(self, registry: "ToolRegistry", cfg: "Config") -> None:
        self._registry = registry
        self._cfg = cfg

    async def dispatch(self, action: "JudgmentOutput", ctx: ToolContext) -> ToolResult:
        """根据 decision 类型分发执行。"""
        match action.decision:
            case "wait":
                return ToolResult(
                    summary=f"wait: {action.rationale[:200]}",
                    skipped=True,
                    kind="wait",
                    priority=0.3,
                )
            case "pause":
                from memory.working import WMItem
                ctx.wm.add(WMItem(
                    kind="caution",
                    content=f"pause: {action.rationale[:300]}",
                    priority=0.9,
                ))
                return ToolResult(
                    summary=f"pause: {action.rationale[:200]}",
                    skipped=True,
                    kind="pause",
                    priority=0.9,
                )
            case "act":
                return await self._dispatch_act(action, ctx)
            case _:
                return ToolResult(
                    summary=f"未知决策类型: {action.decision!r}",
                    skipped=True,
                    kind="error",
                )

    async def _dispatch_act(self, action: "JudgmentOutput", ctx: ToolContext) -> ToolResult:
        entry = self._registry.get(action.chosen_action_id)
        if not entry:
            return ToolResult(
                summary=f"工具不存在: {action.chosen_action_id!r}",
                error="ToolNotFound",
                skipped=True,
                kind="error",
            )

        if self._cfg.loop.debug:
            _log.debug("[exec] %s params=%s", action.chosen_action_id, action.params)
        _log.info("[exec] %s", action.chosen_action_id)

        # durable failure sensing：对稳定重复失败的确定性动作做短期持久降噪
        failure_key = _failure_fact_key(action)
        if ctx.task_store is not None:
            raw, found = await ctx.task_store.get_fact(failure_key)
            if found:
                try:
                    info = json.loads(raw)
                except Exception:
                    info = {}
                muted_until = float(info.get("muted_until") or 0)
                count = int(info.get("count") or 0)
                reason = str(info.get("reason") or "stable_failure")
                if count >= _DURABLE_FAILURE_THRESHOLD and muted_until > time.time():
                    return ToolResult(
                        summary=(
                            f"跳过已知稳定失败动作：{action.chosen_action_id} {_action_key_param(action.params)}\n"
                            f"原因: {reason}；最近已连续失败 {count} 次。"
                            " 若外部状态已修复，请等待静默窗口结束后重试，或更换动作/参数。"
                        ),
                        evidence=raw,
                        error="KnownStableFailure",
                        skipped=True,
                        kind="execute_result",
                        priority=0.4,
                    )

        try:
            result = await entry.handler(action.params, ctx)
        except Exception as exc:
            result = ToolResult(
                summary=f"工具执行异常: {exc}",
                evidence=str(exc),
                error=str(exc),
                kind="execute_result",
            )

        # 失败时写入 failures 表，绑定当前任务（P2-B 任务边界原则）
        if result.error and not result.skipped and ctx.task_store is not None:
            task = await ctx.task_store.get_active()
            task_id = str(task.id) if task else ""
            await ctx.task_store.record_failure(
                kind=action.chosen_action_id,
                summary=result.summary[:300],
                context=result.evidence[:200],
                task_id=task_id,
            )

        # 更新 durable failure 状态（对所有“可识别的确定性失败”生效）
        if ctx.task_store is not None:
            reason = _classify_durable_failure(result)
            if result.error and reason:
                raw, found = await ctx.task_store.get_fact(failure_key)
                prev: dict[str, Any] = {}
                if found:
                    try:
                        prev = json.loads(raw)
                    except Exception:
                        prev = {}
                count = int(prev.get("count") or 0) + 1 if prev.get("reason") == reason else 1
                payload = {
                    "tool": action.chosen_action_id,
                    "key": _action_key_param(action.params),
                    "reason": reason,
                    "count": count,
                    "last_summary": result.summary[:200],
                    "last_seen": time.time(),
                    "muted_until": time.time() + _DURABLE_FAILURE_TTL_SEC if count >= _DURABLE_FAILURE_THRESHOLD else 0,
                }
                await ctx.task_store.set_fact(failure_key, json.dumps(payload, ensure_ascii=False), scope="system")
            elif not result.error:
                await ctx.task_store.set_fact(
                    failure_key,
                    json.dumps({
                        "tool": action.chosen_action_id,
                        "key": _action_key_param(action.params),
                        "reason": "",
                        "count": 0,
                        "last_summary": result.summary[:200],
                        "last_seen": time.time(),
                        "muted_until": 0,
                    }, ensure_ascii=False),
                    scope="system",
                )

        return result
