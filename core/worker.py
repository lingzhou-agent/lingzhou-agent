"""core/worker.py — Run worker 抽象。

这一层现在承担真正的 worker 语义边界：
- tool-chain-worker：普通工具链调用
- exec-worker：后台/前台进程执行与监控元数据规范化
- multimodal-worker：多模态输入归一化
- llm-worker：LLM 驱动工具的监控协议归一化
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable

from tools.registry import ToolContext, ToolResult

if TYPE_CHECKING:
    from core.judgment import JudgmentOutput
    from tools.registry import ToolEntry


WorkerHandler = Callable[["ToolEntry", "JudgmentOutput", ToolContext], Awaitable[ToolResult]]


class WorkerLayer:
    def __init__(self) -> None:
        self._handlers: dict[str, WorkerHandler] = {
            "tool-chain-worker": self._execute_tool_chain,
            "exec-worker": self._execute_exec,
            "multimodal-worker": self._execute_multimodal,
            "llm-worker": self._execute_llm,
        }

    async def dispatch(
        self,
        worker_type: str,
        entry: "ToolEntry",
        action: "JudgmentOutput",
        ctx: ToolContext,
    ) -> ToolResult:
        handler = self._handlers.get(worker_type, self._execute_tool_chain)
        result = await handler(entry, action, ctx)
        result.metadata.setdefault("worker_type", worker_type)
        result.metadata.setdefault("tool_name", action.chosen_action_id or "")
        return result

    async def _execute_tool_chain(
        self,
        entry: "ToolEntry",
        action: "JudgmentOutput",
        ctx: ToolContext,
    ) -> ToolResult:
        result = await entry.handler(action.params, ctx)
        result.metadata.setdefault("worker_path", "tool-chain")
        return result

    async def _execute_exec(
        self,
        entry: "ToolEntry",
        action: "JudgmentOutput",
        ctx: ToolContext,
    ) -> ToolResult:
        result = await entry.handler(action.params, ctx)
        result.metadata.setdefault("worker_path", "exec")
        background = bool(isinstance(result.state_delta, dict) and result.state_delta.get("background"))
        result.metadata.setdefault("execution_mode", "background" if background else "foreground")
        if background and not result.metadata.get("session_id") and result.resource_key:
            result.metadata["session_id"] = result.resource_key
        if background and result.metadata.get("session_id"):
            result.metadata.setdefault(
                "run_monitor",
                {"kind": "process", "session_id": str(result.metadata.get("session_id") or "")},
            )
        return result

    async def _execute_multimodal(
        self,
        entry: "ToolEntry",
        action: "JudgmentOutput",
        ctx: ToolContext,
    ) -> ToolResult:
        result = await entry.handler(action.params, ctx)
        result.metadata.setdefault("worker_path", "multimodal")
        image_count = 0
        for key in ("path", "paths", "image", "images"):
            value = action.params.get(key)
            if not value:
                continue
            if isinstance(value, list):
                image_count += len(value)
            else:
                image_count += 1
        result.metadata.setdefault("modality", "image")
        result.metadata.setdefault("input_count", max(1, image_count) if image_count else 1)
        return result

    async def _execute_llm(
        self,
        entry: "ToolEntry",
        action: "JudgmentOutput",
        ctx: ToolContext,
    ) -> ToolResult:
        result = await entry.handler(action.params, ctx)
        result.metadata.setdefault("worker_path", "llm")
        result.metadata.setdefault("reasoning_mode", "tool-mediated-llm")
        monitor_key = str(
            action.params.get("monitor_fact_key")
            or action.params.get("status_fact_key")
            or ""
        ).strip()
        if monitor_key:
            result.metadata.setdefault(
                "run_monitor",
                {
                    "kind": "fact",
                    "key": monitor_key,
                    "status_field": str(action.params.get("monitor_status_field") or "status"),
                    "progress_field": str(action.params.get("monitor_progress_field") or "progress"),
                },
            )
        return result