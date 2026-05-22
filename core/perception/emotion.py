"""core/perception/emotion.py — 情绪状态：OCC 评价模型 + Russell Core Affect。

参考：Ortony, Clore, Collins (1988)；Russell (2003)；Gross (1998)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.config import Config


# ── 辅助 ──────────────────────────────────────────────────────────────────────

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _clamp_signed(x: float) -> float:
    """clamp 到 [-1, 1]。"""
    return max(-1.0, min(1.0, x))


# ── OCC 评价维度 ───────────────────────────────────────────────────────────────

@dataclass
class Appraisal:
    """OCC 评价维度（Scherer 2001 多级评价视角）。"""
    novelty: float = 0.0          # 新奇性：未预期 / 变化
    goal_congruence: float = 0.0  # 目标相关性：正 = 达成感，负 = 受阻感
    control: float = 0.0          # 控制感：能影响当前处境的程度
    certainty: float = 0.0        # 确定性：对状态可预测程度的感知


@dataclass
class Feeling:
    """离散情感（OCC categories 简化集）。"""
    name: str
    intensity: float
    cause: str = ""


@dataclass
class Regulation:
    """Gross (1998) 情绪调节策略。"""
    strategy: str = "maintain"   # maintain | down-regulate | up-regulate
    reason: str = ""


@dataclass
class EmotionState:
    """完整情绪状态。valence/arousal 保持直接属性以兼容现有调用。"""
    # Russell (2003) Core Affect
    valence: float = 0.6      # 效价：负面(0) ↔ 正面(1)
    arousal: float = 0.5      # 唤醒：平静(0) ↔ 激活(1)
    dominance: float = 0.5    # 支配感：无力(0) ↔ 主导(1)
    # OCC 评价与离散情感
    appraisal: Appraisal = field(default_factory=Appraisal)
    feelings: list[Feeling] = field(default_factory=list[Feeling])
    dominant: str = ""         # 强度最高的离散情感名称
    regulation: Regulation = field(default_factory=Regulation)

    @classmethod
    def from_config(cls, cfg: "Config") -> "EmotionState":
        return cls(
            valence=cfg.emotion.baseline_valence,
            arousal=cfg.emotion.baseline_arousal,
        )

    def derive_from_signals(
        self,
        failure_count: int,
        prediction_error: float,
        wm_pressure: float,
        workspace_dirty: bool,
        alpha: float,
        *,
        high_error_streak: int = 0,
        replay_trend: str = "stable",
        task_status: str = "",
        has_next_step: bool = False,
        has_active_task: bool = False,
    ) -> None:
        """从感知信号确定性推导情绪状态（OCC 1988 评价理论）。

        设计原则：感知信号 → 评价维度 → core affect → 离散情感 → 调节策略。
        LLM 不参与情绪计算（LLM 自报告自身情绪属自引用错误）。
        """
        prediction = clamp01(prediction_error)
        wm_trust = clamp01(1.0 - wm_pressure)
        failures = clamp01(failure_count / 3.0)
        blocked = 1.0 if task_status == "blocked" else 0.0
        next_s = 1.0 if has_next_step else 0.0
        has_task = 1.0 if has_active_task else 0.0
        recovering = 1.0 if replay_trend == "recovering" else 0.0
        high_err = clamp01(high_error_streak / 3.0)

        # ── OCC 评价维度 ────────────────────────────────────────────────────────
        app = Appraisal(
            novelty        = clamp01(0.25 + 0.55 * prediction + 0.20 * (1.0 - wm_trust)),
            goal_congruence= _clamp_signed(
                0.35 * wm_trust + 0.15 * next_s + 0.12 * recovering
                - 0.55 * failures - 0.25 * blocked
            ),
            control        = clamp01(
                0.20 + 0.45 * wm_trust + 0.15 * next_s + 0.10 * has_task
                - 0.20 * prediction - 0.10 * high_err
            ),
            certainty      = clamp01(
                0.20 + 0.60 * wm_trust + 0.10 * next_s - 0.40 * prediction
            ),
        )

        # ── 离散情感（强度 < 0.15 过滤）──────────────────────────────────────
        raw_feelings: list[tuple[str, float, str]] = [
            ("distress",   clamp01(max(0, -app.goal_congruence) * 0.8 + 0.2 * failures), "goal_failure"),
            ("frustration",clamp01(0.6 * blocked + 0.4 * prediction),                    "blocked_or_error"),
            ("fear",       clamp01(0.7 * prediction + 0.2 * (1.0 - app.certainty)),       "uncertainty"),
            ("hope",       clamp01(0.45 * next_s + 0.25 * wm_trust + 0.20 * recovering), "recoverable_path"),
            ("confidence", clamp01(0.55 * app.control + 0.25 * wm_trust),                 "available_control"),
            ("relief",     clamp01(0.45 * recovering + 0.20 * wm_trust - 0.20 * prediction), "improving_trend"),
            ("joy",        clamp01(max(0, app.goal_congruence) * 0.55),                   "goal_progress"),
        ]
        feelings = sorted(
            [Feeling(n, i, c) for n, i, c in raw_feelings if i >= 0.15],
            key=lambda f: -f.intensity,
        )[:6]

        # ── Core Affect（Russell 2003）────────────────────────────────────────
        pos_avg = sum(f.intensity for f in feelings if f.name in {"hope", "confidence", "relief", "joy"})
        neg_avg = sum(f.intensity for f in feelings if f.name in {"distress", "frustration", "fear"})
        target_v = clamp01(0.35 + app.goal_congruence * 0.45 + (pos_avg - neg_avg) * 0.20)
        target_a = clamp01(0.20 + app.novelty * 0.45 + (1.0 - app.control) * 0.35)
        target_d = clamp01(0.30 + app.control * 0.45 + app.certainty * 0.25)

        # EMA 平滑（性格不因单次经历骤变）
        self.valence   = alpha * target_v + (1 - alpha) * self.valence
        self.arousal   = alpha * target_a + (1 - alpha) * self.arousal
        self.dominance = alpha * target_d + (1 - alpha) * self.dominance

        self.appraisal = app
        self.feelings  = feelings
        self.dominant  = feelings[0].name if feelings else ""

        # ── 调节策略（Gross 1998）────────────────────────────────────────────
        if self.arousal > 0.75 or self.valence < 0.30 or (replay_trend == "worsening" and self.valence < 0.45):
            reason = "高唤醒/低效价或趋势恶化" if self.arousal > 0.75 or self.valence < 0.30 else "趋势恶化预警"
            self.regulation = Regulation("down-regulate", reason)
        elif (replay_trend == "recovering" and self.valence < 0.55) or (recovering and self.valence < 0.60):
            self.regulation = Regulation("up-regulate", "趋势改善，保持恢复势头")
        elif high_error_streak >= 2:
            self.regulation = Regulation("down-regulate", "连续高预测误差，需切换策略")
        else:
            self.regulation = Regulation("maintain", "")

    @property
    def activation(self) -> float:
        """合成激活值，用于情绪驱动任务阈值比较。"""
        return self.valence * 0.4 + self.arousal * 0.6


# ── 感知 / 情绪重放摘要 ────────────────────────────────────────────────────────

@dataclass
class PerceptionReplaySummary:
    """最近 N 次感知事件的趋势摘要。"""
    samples: int = 0
    avg_prediction_error: float = 0.0
    high_error_streak: int = 0      # 从最新往前，连续 prediction_error > high_error_threshold 的次数
    trend: str = "insufficient_data"  # stable | worsening | recovering
    hints: list[str] = field(default_factory=list[str])


@dataclass
class EmotionReplaySummary:
    """最近 N 次情绪事件的趋势摘要。"""
    samples: int = 0
    down_regulate_streak: int = 0   # 从最新往前，连续 down-regulate 的次数
    trend: str = "insufficient_data"


def build_perception_replay(
    events: list[dict[str, Any]],
    high_error_threshold: float = 0.7,
) -> PerceptionReplaySummary:
    """从持久化的 perception_events 构建重放摘要。"""
    r = PerceptionReplaySummary(samples=len(events))
    if not events:
        r.hints = ["尚无感知历史，趋势判断暂不可用"]
        return r
    errors = [e["prediction_error"] for e in events]
    r.avg_prediction_error = sum(errors) / len(errors)
    for err in reversed(errors):
        if err < high_error_threshold:
            break
        r.high_error_streak += 1
    if len(events) >= 2:
        delta = errors[-1] - errors[0]
        if delta >= 0.15:
            r.trend = "worsening"
        elif delta <= -0.15:
            r.trend = "recovering"
        else:
            r.trend = "stable"
    if r.high_error_streak >= 2:
        r.hints.append("预测误差持续偏高，应切换策略或补充证据再重试")
    if r.trend == "recovering":
        r.hints.append("感知趋势改善中，保持较窄恢复路径并持续验证")
    if not r.hints:
        r.hints.append("感知历史相对稳定，可支持下一步判断")
    return r


def build_emotion_replay(events: list[dict[str, Any]]) -> EmotionReplaySummary:
    """从持久化的 emotion_events 构建重放摘要。"""
    r = EmotionReplaySummary(samples=len(events))
    if not events:
        return r
    for ev in reversed(events):
        if ev.get("regulation_strategy") != "down-regulate":
            break
        r.down_regulate_streak += 1
    if len(events) >= 2:
        delta = events[-1]["valence"] - events[0]["valence"]
        r.trend = "recovering" if delta >= 0.10 else ("worsening" if delta <= -0.10 else "stable")
    return r
