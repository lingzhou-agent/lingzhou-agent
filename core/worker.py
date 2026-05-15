"""core/worker.py — Run worker 抽象。

当前阶段保持最小实现：
- 根据 worker_type 选择执行器
- 默认仍复用现有工具 handler
- 为后续异步/并行 worker 扩展预留稳定边界
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
            "tool-chain-worker": self._execute_via_tool,
            "exec-worker": self._execute_via_tool,
            "multimodal-worker": self._execute_via_tool,
            "llm-worker": self._execute_via_tool,
        }

    async def dispatch(
        self,
        worker_type: str,
        entry: "ToolEntry",
        action: "JudgmentOutput",
        ctx: ToolContext,
    ) -> ToolResult:
        handler = self._handlers.get(worker_type, self._execute_via_tool)
        result = await handler(entry, action, ctx)
        result.metadata.setdefault("worker_type", worker_type)
        result.metadata.setdefault("tool_name", action.chosen_action_id or "")
        return result

    async def _execute_via_tool(
        self,
        entry: "ToolEntry",
        action: "JudgmentOutput",
        ctx: ToolContext,
    ) -> ToolResult:
        return await entry.handler(action.params, ctx)