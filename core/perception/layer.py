"""core/perception/layer.py — 感知层入口：Percept + PerceptionLayer。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.config import Config
    from memory.working import WorkingMemory
    from store.task import Task
    from core.perception.emotion import EmotionState
    from core.perception.signals import CognitiveSignals


@dataclass
class Percept:
    """一个认知 tick 的感知结果。"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    prediction_error: float = 0.0      # 与上一轮预期的偏差 [0, 1]
    workspace_dirty: bool = False       # 工作目录是否有未追踪变更
    workspace_fingerprint: str = ""     # 用于检测变化的哈希
    summary: str = ""                   # 给语义检索用的查询词

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "prediction_error": round(self.prediction_error, 3),
            "workspace_dirty": self.workspace_dirty,
        }


class PerceptionLayer:
    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg
        self._last_fingerprint: str = ""
        self._last_wm_size: int = 0

    async def sense(
        self,
        wm: "WorkingMemory",
        active_task: "Task | None" = None,
        *,
        last_next_step: str = "",
        last_decision: str = "wait",
    ) -> Percept:
        """生成本轮感知快照。"""
        fingerprint = self._workspace_fingerprint()
        workspace_dirty = (fingerprint != self._last_fingerprint and self._last_fingerprint != "")
        prediction_error = self._compute_prediction_error(
            wm, workspace_dirty,
            last_next_step=last_next_step,
            last_decision=last_decision,
        )

        self._last_fingerprint = fingerprint
        self._last_wm_size = len(wm)

        summary = active_task.goal if active_task else "当前状态"

        return Percept(
            prediction_error=prediction_error,
            workspace_dirty=workspace_dirty,
            workspace_fingerprint=fingerprint,
            summary=summary,
        )

    def derive_cognitive_signals(
        self,
        percept: Percept,
        wm: "WorkingMemory",
        emotion: "EmotionState",
        cfg: "Config",
        *,
        has_active_task: bool = False,
        idle_cycles: int = 0,
        next_step_fulfilled: bool | None = None,
    ) -> "CognitiveSignals":
        """将感知信号转化为认知状态报告，注入 LLM 判断上下文。

        设计原则：此方法只计算信号强度，不产生任何决策或任务文字。
        """
        from core.perception.signals import CognitiveSignals
        t = cfg.thresholds
        return CognitiveSignals(
            emotion_activation=emotion.activation,
            emotion_alert=emotion.activation > t.emotion_activation_task,
            wm_pressure=wm.pressure,
            wm_pressure_alert=wm.pressure > t.wm_pressure_task,
            prediction_error=percept.prediction_error,
            prediction_error_alert=percept.prediction_error > t.prediction_error_task,
            has_active_task=has_active_task,
            idle_cycles=idle_cycles,
            next_step_fulfilled=next_step_fulfilled,
        )

    def _workspace_fingerprint(self) -> str:
        """对工作目录浅层文件做轻量哈希，检测是否有变更。"""
        try:
            # 使用 cfg.workspace_dir（配置明确的工作目录），而非进程 cwd（两者可能不同）
            watch_dir = self._cfg.workspace_dir if self._cfg.workspace_dir.exists() else Path.cwd()
            entries = sorted(
                (p.name, p.stat().st_mtime)
                for p in watch_dir.iterdir()
                if not p.name.startswith(".")
            )
            raw = str(entries).encode()
            return hashlib.md5(raw).hexdigest()[:16]
        except Exception:
            return ""

    def _compute_prediction_error(
        self,
        wm: "WorkingMemory",
        workspace_dirty: bool,
        *,
        last_next_step: str = "",
        last_decision: str = "wait",
    ) -> float:
        """预测误差：WM 大小变化 + 工作区变更 + 上轮计划未执行。

        next_step_miss：上轮 LLM 声明了 next_step 却选择 wait/pause（计划漂移信号）。
        """
        if self._last_wm_size == 0 or len(wm) == 0:
            wm_signal = 0.0
        else:
            wm_delta = abs(len(wm) - self._last_wm_size) / self._last_wm_size
            wm_signal = min(wm_delta, 1.0) * 0.4
        env_signal = 0.5 if workspace_dirty else 0.0
        next_step_miss = 0.25 if (last_next_step and last_decision in ("wait", "pause")) else 0.0
        return round(min(wm_signal + env_signal + next_step_miss, 1.0), 3)

    def reset_wm_baseline(self, new_size: int = 0) -> None:
        """在 WM 被主动清空后同步感知基准，避免下一轮产生假预测误差。"""
        self._last_wm_size = new_size
