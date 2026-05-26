"""core/loop/common.py - loop 包内共享常量与纯 helper。"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.config import Config
from core.judgment import JudgmentOutput, tool_tier
from core.perception import PerceptionReplaySummary
from core.task_runtime import VALID_MODEL_TIERS
from store.task import Task
from tools.registry import ToolResult

# 上下文截断具名常量(语义记忆 & 日志截断阈值;调整后重启即生效,不影响已存数据)
_LOG_RATIONALE_CHARS = 120
_SEM_TITLE_CHARS = 60
_SEM_TAG_TASK_CHARS = 20
_EVENT_TITLE_CHARS = 40
_EVENT_APPEND_CHARS = 8000
_EVENT_BODY_MAX_CHARS = 40000
_EVENT_NEW_BODY_CHARS = 16000

_VALENCE_HINT_RE = re.compile(r"(?:valence|情绪效价)\s*[:=：]\s*(0(?:\.\d+)?|1(?:\.0+)?)", re.IGNORECASE)


def _explicit_valence_hint(text: str) -> float | None:
    match = _VALENCE_HINT_RE.search(text or "")
    if not match:
        return None
    try:
        value = float(match.group(1))
    except Exception:
        return None
    if 0.0 <= value <= 1.0:
        return value
    return None


def _infer_valence_from_text(text: str, current: float, emotion_cfg: Any) -> float:
    """从 reflection 文本里的显式 valence hint 推断效价。

    不再依赖 Python 侧正负关键词词表，避免用硬编码词义去塑形情绪轨迹。
    若 reflection 没有给出结构化 hint，则保持当前值不变。
    """
    hinted = _explicit_valence_hint(text)
    if hinted is None:
        return current
    history_weight = max(0.0, float(getattr(emotion_cfg, "reflection_valence_history_weight", 0.0)))
    hint_weight = max(0.0, float(getattr(emotion_cfg, "reflection_valence_hint_weight", 0.0)))
    total_weight = history_weight + hint_weight
    if total_weight <= 0:
        return current
    return current * (history_weight / total_weight) + hinted * (hint_weight / total_weight)


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
    registry: Any | None = None,
) -> bool:
    """task.complete/fail 后不续计；mutation+用户消息+有任务时暂停让用户确认。"""
    if action.decision != "act":
        return False
    if (action.chosen_action_id or "") in {"task.complete", "task.fail"}:
        return False
    # mutation tool in a user-prompted tick with active task: don't auto-continue
    if user_message and has_active_task and tool_tier(action.chosen_action_id or "", registry) != "reader":
        return False
    return True


def _preferred_continue_tier(
    action: JudgmentOutput,
    *,
    user_message: str = "",
    registry: Any | None = None,
) -> str | None:
    next_tier = str((action.model_strategy or {}).get("next_phase_tier", "") or "")
    if next_tier in VALID_MODEL_TIERS:
        return next_tier
    return None


def _task_model_tier(task: Task | None) -> str | None:
    if not task:
        return None
    tier = (task.model_tier or "").strip()
    return tier if tier in VALID_MODEL_TIERS else None


def _next_initial_tier_hint(action: JudgmentOutput) -> str | None:
    next_tier = str((action.model_strategy or {}).get("next_phase_tier", "") or "")
    return next_tier if next_tier in VALID_MODEL_TIERS else None


def _prefer_tier_for_task(
    pending_tier: str | None,
    task: Task | None,
    *,
    has_user_message: bool = False,
) -> str | None:
    if has_user_message:
        return "reasoner"
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