"""core/loop/logging.py - loop 的日志与用户可见文本 helper。"""

from __future__ import annotations

from typing import Any

from core.judgment import JudgmentOutput
from core.execution import action_key_param
from memory.task_store import Task
from tools.registry import ToolResult

DEFAULT_LOG_REPLY_CHARS = 240


def _strip_memory_context(text: str) -> str:
    """剥离 LLM 输出中意外泄露的 <memory-context>...</memory-context> 内容。"""
    import re as _re

    cleaned = _re.sub(r"<memory-context>.*?</memory-context>", "", text, flags=_re.DOTALL)
    return cleaned.strip() or text.strip()


def _clip_reply_for_log(text: str, limit: int = DEFAULT_LOG_REPLY_CHARS) -> str:
    cleaned = _strip_memory_context(text).replace("\n", "\\n").strip()
    return cleaned


def _clip_signal_text(text: str, limit: int = 160) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _summarize_state_delta(state_delta: dict[str, Any] | None, limit: int = 120) -> str:
    if not state_delta:
        return ""
    parts = [f"{key}={state_delta[key]}" for key in sorted(state_delta)]
    return _clip_signal_text("; ".join(parts), limit)


def _format_action_feedback_line(
    action: JudgmentOutput,
    result: ToolResult,
    *,
    progressful: bool,
) -> str:
    tool = action.chosen_action_id or action.decision or "-"
    key = action_key_param(action.params) if action.decision == "act" else ""
    status = "error" if result.error else ("skipped" if result.skipped else ("ok" if action.decision == "act" else action.decision))
    parts = [f"tool={tool}"]
    if key:
        parts.append(f"key={key}")
    parts.append(f"status={status}")
    parts.append(f"progressful={progressful}")
    if result.error:
        parts.append(f"error={_clip_signal_text(result.error, 80)}")
    if result.state_delta:
        parts.append(f"state_delta={_summarize_state_delta(result.state_delta, 90)}")
    if result.summary:
        parts.append(f"summary={_clip_signal_text(result.summary, 100)}")
    return " | ".join(parts)


def _fallback_reply_for_user(action: JudgmentOutput, result: ToolResult, active_task: Task | None) -> str:
    def _brief(text: str, limit: int = 80) -> str:
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 3)] + "..."

    def _fact_line(prefix: str, value: str) -> str:
        value = value.strip()
        return f"{prefix}: {value}" if value else ""

    next_step = str(action.next_step or (active_task.next_step if active_task else "") or "").strip()
    if result.error:
        lines = [
            _fact_line("状态", "error"),
            _fact_line("detail", _brief(result.summary or result.error, 100)),
            _fact_line("next", _brief(next_step, 60)) if next_step else "",
        ]
        return ";".join(line for line in lines if line)

    if action.decision in {"wait", "pause"}:
        raw_basis = action.rationale or ""
        if any(tech in raw_basis for tech in ("缺少 chosen_action_id", "LLM 输出解析失败", "无效 decision", "list index out of range", "not defined")):
            raw_basis = ""
        basis = _brief(raw_basis or "需要更多信息后再继续。", 100)
        lines = [
            _fact_line("状态", action.decision),
            _fact_line("basis", basis),
            _fact_line("next", _brief(next_step, 60)) if next_step else "",
        ]
        return ";".join(line for line in lines if line)

    task_status = str((result.state_delta or {}).get("task_status") or "").strip()
    if task_status == "waiting":
        wait_kind = str((result.state_delta or {}).get("wait_kind") or "external").strip()
        wait_key = str((result.state_delta or {}).get("wait_key") or "").strip()
        wait_desc = wait_kind + (f"/{wait_key}" if wait_key else "")
        lines = [
            _fact_line("状态", "waiting"),
            _fact_line("wait", wait_desc),
            _fact_line("next", _brief(next_step, 60)) if next_step else "",
        ]
        return ";".join(line for line in lines if line)

    lines = [
        _fact_line("状态", "progressed"),
        _fact_line("basis", _brief(action.rationale or "已完成本轮处理，正在整理基于证据的答复。", 100)),
        _fact_line("next", _brief(next_step, 60)) if next_step else "",
    ]
    return ";".join(line for line in lines if line)