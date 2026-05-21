"""core/loop/common.py - loop 包内共享常量与纯 helper。"""

from __future__ import annotations

import logging
from typing import Any

from core.config import Config
from core.judgment import JudgmentOutput, READER_TOOLS
from core.perception import PerceptionReplaySummary
from core.task_runtime import VALID_MODEL_TIERS
from memory.task_store import Task
from tools.registry import ToolResult

# 上下文截断具名常量(语义记忆 & 日志截断阈值;调整后重启即生效,不影响已存数据)
_LOG_RATIONALE_CHARS = 120
_SEM_TITLE_CHARS = 60
_SEM_TAG_TASK_CHARS = 20
_EVENT_TITLE_CHARS = 40
_EVENT_APPEND_CHARS = 8000
_EVENT_BODY_MAX_CHARS = 40000
_EVENT_NEW_BODY_CHARS = 16000

_VALENCE_POS = frozenset(["完成", "成功", "理解", "学到", "进步", "有效", "清晰", "好", "正确", "解决", "突破"])
_VALENCE_NEG = frozenset(["失败", "错误", "困惑", "卡住", "无法", "问题", "不对", "不清", "循环", "重复", "卡顿"])


def _infer_valence_from_text(text: str, current: float) -> float:
    """从 reflection 文本推断情绪效价倾向。"""
    pos = sum(1 for word in _VALENCE_POS if word in text)
    neg = sum(1 for word in _VALENCE_NEG if word in text)
    if pos + neg == 0:
        return current
    ratio = pos / (pos + neg)
    target = 0.3 + ratio * 0.7
    return current * 0.8 + target * 0.2


def _next_thinking_override(model_strategy: dict[str, Any] | None) -> str | None:
    raw = (model_strategy or {}).get("thinking_override")
    valid = {"off", "minimal", "low", "medium", "high"}
    if isinstance(raw, str) and raw in valid:
        return raw
    return None


def _thinking_floor(value: str | None, floor: str | None) -> str | None:
    order = {"off": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4}
    if floor is None:
        return value
    if value is None:
        return floor
    return value if order.get(value, -1) >= order.get(floor, -1) else floor


def _resolve_thinking_override(
    cfg: Config,
    *,
    user_message: str,
    pending_override: str | None = None,
    model_strategy: dict[str, Any] | None = None,
) -> str | None:
    if pending_override is not None:
        return pending_override
    next_override = _next_thinking_override(model_strategy)
    if next_override is not None:
        return next_override
    if user_message:
        return cfg.loop.chat_thinking if cfg.loop.chat_thinking != cfg.thinking else None
    return cfg.loop.autonomous_thinking if cfg.loop.autonomous_thinking != cfg.thinking else None


def _should_continue_within_tick(
    action: JudgmentOutput,
    *,
    user_message: str = "",
    has_active_task: bool = False,
) -> bool:
    """task.complete/fail 后不续计；mutation+用户消息+有任务时暂停让用户确认。"""
    if action.decision != "act":
        return False
    if (action.chosen_action_id or "") in {"task.complete", "task.fail"}:
        return False
    # mutation tool in a user-prompted tick with active task: don't auto-continue
    if user_message and has_active_task and (action.chosen_action_id or "") not in READER_TOOLS:
        return False
    return True


def _preferred_continue_tier(action: JudgmentOutput, *, user_message: str = "") -> str | None:
    next_tier = str((action.model_strategy or {}).get("next_phase_tier", "") or "")
    if next_tier in VALID_MODEL_TIERS:
        return next_tier
    # 只要刚执行的工具属于读取类，续判就保持 reader tier（无论是否有 user_message）
    if (action.chosen_action_id or "") in READER_TOOLS:
        return "reader"
    return None


def _task_model_tier(task: Task | None) -> str | None:
    if not task:
        return None
    tier = (task.model_tier or "").strip()
    return tier if tier in VALID_MODEL_TIERS else None


def _next_initial_tier_hint(action: JudgmentOutput) -> str | None:
    next_tier = str((action.model_strategy or {}).get("next_phase_tier", "") or "")
    return next_tier if next_tier in VALID_MODEL_TIERS else None


def _prefer_tier_for_task(pending_tier: str | None, task: Task | None) -> str | None:
    if pending_tier in VALID_MODEL_TIERS:
        return pending_tier
    task_tier = _task_model_tier(task)
    return task_tier if task_tier in {"reasoner", "repair"} else None


def _perception_replay_fallback() -> PerceptionReplaySummary:
    """感知回放的兜底默认值，防止 build_perception_replay 异常导致 NameError。"""
    return PerceptionReplaySummary()

_log = logging.getLogger("lingzhou.loop")


def _tool_history_entry(action: JudgmentOutput, result: ToolResult) -> dict[str, Any]:
    summary = str(result.summary or "")
    error = str(result.error or "")
    status = "ok" if not error and not result.skipped else ("skipped" if result.skipped else "error")
    if error:
        err_lower = error.lower()
        error_category = (
            "transient"
            if any(marker in err_lower for marker in ("timeout", "connect", "reset", "unavailable", "rate", "429", "503"))
            else "fatal"
        )
    else:
        error_category = ""
    return {
        "tool": action.chosen_action_id or "",
        "params": action.params or {},
        "result": f"ERROR[{error_category}]: {summary}" if error else summary,
        "summary": summary,
        "error": error,
        "error_category": error_category,
        "skipped": bool(result.skipped),
        "status": status,
        "state_delta": dict(result.state_delta or {}) if isinstance(result.state_delta, dict) else {},
    }


async def _maybe_reconcile_bootstrap(loop: Any) -> None:
    """如果 BOOTSTRAP.md 已被本 tick 删除，写入 setupCompletedAt 并切换到正常模式。"""
    if loop._bootstrap_mode != "full":
        return
    bootstrap_path = loop._cfg.workspace_dir / "BOOTSTRAP.md"
    if bootstrap_path.exists():
        return
    from core.workspace.state import reconcile_bootstrap_completion
    reconcile_bootstrap_completion(loop._cfg.workspace_dir)
    await loop._soul.refresh_identity(loop._judgment)
    loop._bootstrap_mode = "none"
    _log.info("[bootstrap] BOOTSTRAP.md 已删除，切换到正常运行模式")