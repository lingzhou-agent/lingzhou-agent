"""tools/ask.py — task.ask 工具。

LLM 可通过此工具向用户提问，获取澄清或额外信息。
在微信/webchat 通道中表现为文本提问，在 CLI 中表现为终端提示。
"""

from __future__ import annotations

from typing import Any

from tools.registry import tool, ToolManifest, ToolResult, ToolParam, ToolContext


@tool(ToolManifest(
    name="task.ask",
    description=(
        "登记一次需要用户补充信息的澄清请求。\n"
        "适合场景：任务信息不足、需要用户确认、遇到歧义需要澄清。\n"
        "真正发送给用户的话应写入 reply_to_user；task.ask 只负责把这次外部依赖记入执行轨迹。\n"
        "choices 可选：最多 4 个预定义选项，用户可从中选择或自由回答。"
    ),
    progress_category="info",
    params=[
        ToolParam("question", "string", "要问的问题", required=True),
        ToolParam("choices", "object", "可选项列表（JSON 数组，最多4个）", required=False),
    ],
))
async def task_ask(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    question = (params.get("question") or "").strip()
    if not question:
        return ToolResult(summary="question 不能为空", error="EmptyQuestion", skipped=True)

    choices_raw = params.get("choices")
    choices: list[str] = []
    if choices_raw:
        if isinstance(choices_raw, str):
            import json
            try:
                choices_raw = json.loads(choices_raw)
            except json.JSONDecodeError:
                pass
        if isinstance(choices_raw, list):
            choices = [str(c).strip() for c in choices_raw[:4] if str(c).strip()]

    lines = [f"已登记用户澄清请求: {question}"]
    if choices:
        for i, c in enumerate(choices, 1):
            lines.append(f"  [{i}] {c}")
        lines.append(f"  [5] 其他（请直接回复）")

    summary = "\n".join(lines)
    return ToolResult(
        summary=summary,
        evidence=question,
        metadata={"question": question, "choices": choices},
        state_delta={"asked": question},
    )
