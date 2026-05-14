"""core/perception.py — 感知层 + 情绪状态 + 内部任务生成。

关键设计：
1. EmotionState：EMA 更新，基线来自 config，不硬编码
2. Percept：一次感知快照，包含环境信号 + 内部信号
3. derive_internal_tasks：阈值全部来自 config.thresholds
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.config import Config
    from memory.working import WorkingMemory
    from memory.task_store import Task


# ── 辅助 ──────────────────────────────────────────────────────────────────────

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _clamp_signed(x: float) -> float:
    """clamp 到 [-1, 1]。"""
    return max(-1.0, min(1.0, x))


# ── 情绪状态（OCC 评价模型 + Russell Core Affect）─────────────────────────────
# 参考：Ortony, Clore, Collins (1988)；Russell (2003)；Gross (1998)

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
        # 中间信号归一化
        prediction = _clamp01(prediction_error)
        wm_trust = _clamp01(1.0 - wm_pressure)      # WM 压力越高 → 可信度越低
        failures = _clamp01(failure_count / 3.0)
        blocked = 1.0 if task_status == "blocked" else 0.0
        next_s = 1.0 if has_next_step else 0.0
        has_task = 1.0 if has_active_task else 0.0
        recovering = 1.0 if replay_trend == "recovering" else 0.0
        high_err = _clamp01(high_error_streak / 3.0)

        # ── OCC 评价维度 ────────────────────────────────────────────────────────
        app = Appraisal(
            novelty        = _clamp01(0.25 + 0.55 * prediction + 0.20 * (1.0 - wm_trust)),
            goal_congruence= _clamp_signed(
                0.35 * wm_trust + 0.15 * next_s + 0.12 * recovering
                - 0.55 * failures - 0.25 * blocked
            ),
            control        = _clamp01(
                0.20 + 0.45 * wm_trust + 0.15 * next_s + 0.10 * has_task
                - 0.20 * prediction - 0.10 * high_err
            ),
            certainty      = _clamp01(
                0.20 + 0.60 * wm_trust + 0.10 * next_s - 0.40 * prediction
            ),
        )

        # ── 离散情感（强度 < 0.15 过滤）──────────────────────────────────────
        raw_feelings: list[tuple[str, float, str]] = [
            ("distress",   _clamp01(max(0, -app.goal_congruence) * 0.8 + 0.2 * failures), "goal_failure"),
            ("frustration",_clamp01(0.6 * blocked + 0.4 * prediction),                    "blocked_or_error"),
            ("fear",       _clamp01(0.7 * prediction + 0.2 * (1.0 - app.certainty)),       "uncertainty"),
            ("hope",       _clamp01(0.45 * next_s + 0.25 * wm_trust + 0.20 * recovering), "recoverable_path"),
            ("confidence", _clamp01(0.55 * app.control + 0.25 * wm_trust),                 "available_control"),
            ("relief",     _clamp01(0.45 * recovering + 0.20 * wm_trust - 0.20 * prediction), "improving_trend"),
            ("joy",        _clamp01(max(0, app.goal_congruence) * 0.55),                   "goal_progress"),
        ]
        feelings = sorted(
            [Feeling(n, i, c) for n, i, c in raw_feelings if i >= 0.15],
            key=lambda f: -f.intensity,
        )[:6]

        # ── Core Affect（Russell 2003）────────────────────────────────────────
        pos_avg = sum(f.intensity for f in feelings if f.name in {"hope", "confidence", "relief", "joy"})
        neg_avg = sum(f.intensity for f in feelings if f.name in {"distress", "frustration", "fear"})
        target_v = _clamp01(0.35 + app.goal_congruence * 0.45 + (pos_avg - neg_avg) * 0.20)
        target_a = _clamp01(0.20 + app.novelty * 0.45 + (1.0 - app.control) * 0.35)
        target_d = _clamp01(0.30 + app.control * 0.45 + app.certainty * 0.25)

        # EMA 平滑（性格不因单次经历骤变）
        self.valence   = alpha * target_v + (1 - alpha) * self.valence
        self.arousal   = alpha * target_a + (1 - alpha) * self.arousal
        self.dominance = alpha * target_d + (1 - alpha) * self.dominance

        # 结构字段直接赋值（每 tick 重新推导）
        self.appraisal = app
        self.feelings  = feelings
        self.dominant  = feelings[0].name if feelings else ""

        # ── 调节策略（Gross 1998）────────────────────────────────────────────
        if self.arousal > 0.75 or self.valence < 0.30:
            reason = "高唤醒或低效价" if self.arousal > 0.75 else "持续低效价"
            self.regulation = Regulation("down-regulate", reason)
        elif recovering and self.valence < 0.55:
            self.regulation = Regulation("up-regulate", "感知趋势改善中，保持恢复势头")
        else:
            self.regulation = Regulation("maintain", "")

    @property
    def activation(self) -> float:
        """合成激活值，用于情绪驱动任务阈值比较。"""
        return self.valence * 0.4 + self.arousal * 0.6


# ── 感知 / 情绪重放摘要 ────────────────────────────────────────────────────────

@dataclass
class PerceptionReplaySummary:
    """最近 N 次感知事件的趋势摘要（Hermes BuildPerceptionReplay 移植）。"""
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


# ── Ethos 价值层（Hermes ethos.DeriveWithBaseline 移植）────────────────────────
# 参考：Kohlberg (1969) 道德发展内化原则；McCloskey & Glucksberg (1978) 概念渐变

@dataclass
class EthosValues:
    truth: float = 0.65         # 诚实优先
    caution: float = 0.60       # 行动前先确认
    continuity: float = 0.60    # 维持任务连续性
    curiosity: float = 0.45     # 主动感知，不被动等待
    care: float = 0.55          # 对用户数据和状态负责


@dataclass
class EthosBias:
    """当前 tick 的行为倾向，用于候选动作预排名（Hermes rank.go 移植）。"""
    prefer_verification: bool = False   # 优先验证类动作
    prefer_narrow_scope: bool = False   # 优先收窄范围
    preserve_continuity: bool = False   # 优先维持任务连续
    avoid_overclaiming: bool = False    # 避免过度承诺
    reasons: list[str] = field(default_factory=list[str])


@dataclass
class EthosState:
    values: EthosValues = field(default_factory=EthosValues)
    bias: EthosBias = field(default_factory=EthosBias)


def derive_ethos_state(
    failure_count: int,
    high_error_streak: int,
    has_active_task: bool,
    has_next_step: bool,
    perception_trend: str,
    emotion_down_regulate_streak: int,
    baseline: dict[str, float] | None = None,
    ema_alpha: float = 0.9,
    floor_truth: float = 0.50,
    floor_caution: float = 0.45,
) -> EthosState:
    """每 tick 从信号确定性推导 EthosState（含 EMA 基线混合）。

    注：以下信号→价值映射系数是初始默认值，可通过 evolution 机制进化。
    EMA 基线（soul:ethos_baseline）随每次经历缓慢漂移，是真正的"性格记忆"。
    如果某维度长期被同一信号强制拉高/拉低，可在 reflection 中质疑这些映射是否合理。

    ema_alpha / floor_truth / floor_caution 均从 cfg.soul.* 传入，不再硬编码。
    """
    v = EthosValues()
    if failure_count > 0:
        v.truth   = _clamp01(v.truth   + 0.10)
        v.caution = _clamp01(v.caution + 0.10)
        v.curiosity = _clamp01(v.curiosity - 0.08)
    if high_error_streak >= 2:
        v.truth   = _clamp01(v.truth   + 0.10)
        v.caution = _clamp01(v.caution + 0.12)
        v.care    = _clamp01(v.care    - 0.08)
    if has_active_task:
        v.continuity = _clamp01(v.continuity + 0.12)
    if has_next_step:
        v.continuity = _clamp01(v.continuity + 0.08)
        v.care       = _clamp01(v.care       + 0.06)
    if perception_trend == "recovering":
        v.curiosity = _clamp01(v.curiosity + 0.08)
        v.care      = _clamp01(v.care      + 0.04)
    # EMA 混合历史基线（演化速率由 ema_alpha 控制，从 cfg.soul.ethos_ema_alpha 传入）
    if baseline:
        a = ema_alpha
        v.truth      = _clamp01(a * baseline.get("truth",      v.truth)      + (1-a) * v.truth)
        v.caution    = _clamp01(a * baseline.get("caution",    v.caution)    + (1-a) * v.caution)
        v.continuity = _clamp01(a * baseline.get("continuity", v.continuity) + (1-a) * v.continuity)
        v.curiosity  = _clamp01(a * baseline.get("curiosity",  v.curiosity)  + (1-a) * v.curiosity)
        v.care       = _clamp01(a * baseline.get("care",       v.care)       + (1-a) * v.care)
    # 运行时下限（floor_truth / floor_caution 从 cfg.soul.* 传入，防止极端场景下完全崩溃）
    v.truth   = max(v.truth,   floor_truth)
    v.caution = max(v.caution, floor_caution)

    # ChoiceBias：从价值推导行为倾向
    bias = EthosBias()
    reasons: list[str] = []
    if v.caution > 0.70 or failure_count >= 2:
        bias.prefer_verification = True
        reasons.append("谨慎度高，优先验证")
    if failure_count >= 2 or high_error_streak >= 2:
        bias.prefer_narrow_scope = True
        reasons.append("多次失败，收窄操作范围")
    if v.continuity > 0.70 and has_active_task:
        bias.preserve_continuity = True
        reasons.append("任务连续性优先")
    if emotion_down_regulate_streak >= 2:
        bias.avoid_overclaiming = True
        reasons.append("情绪持续下调，避免过度承诺")
    bias.reasons = reasons
    return EthosState(values=v, bias=bias)


# ── 判断信号（预计算，供 rank + LLM 消费）─────────────────────────────────────
# 参考：Hermes judgment/simulate.go simulateDecision

@dataclass
class JudgmentSignals:
    """在 LLM 调用前确定性推导的判断信号（姿态）。"""
    require_more_evidence: bool = False
    prefer_narrow_scope: bool = False
    posture: str = "act"    # act | pause | narrow


def compute_judgment_signals(
    failure_count: int,
    high_error_streak: int,
    perception_trend: str,
    emotion_state: EmotionState,
) -> JudgmentSignals:
    """LLM 前的确定性预判：减少"冷启动"时 LLM 从零估算的不确定性。"""
    sig = JudgmentSignals()
    if high_error_streak >= 2 or (perception_trend == "worsening" and failure_count > 0):
        sig.require_more_evidence = True
    if failure_count >= 2 or high_error_streak >= 2:
        sig.prefer_narrow_scope = True
    if failure_count >= 3 or (
        failure_count >= 1 and emotion_state.regulation.strategy == "down-regulate"
    ):
        sig.posture = "narrow"
    elif high_error_streak >= 2 or (perception_trend == "worsening" and failure_count >= 2):
        sig.posture = "pause"
    return sig


# ── 认知信号（LLM 判断输入） ──────────────────────────────────────────────────────
# 设计原则：此类只报告信号强度，不产生任何决策或任务文字。
# 是否 task.add、如何命名任务、如何响应异常，全部由 LLM 在 judgment 层决定。

@dataclass
class CognitiveSignals:
    """感知层推导的认知状态信号，注入 LLM 判断上下文。"""
    emotion_activation: float = 0.0
    emotion_alert: bool = False             # 激活超阈值
    wm_pressure: float = 0.0
    wm_pressure_alert: bool = False         # WM 压力超阈值
    prediction_error: float = 0.0
    prediction_error_alert: bool = False    # 预测误差超阈值
    has_active_task: bool = False
    idle_cycles: int = 0                    # 无活跃任务持续轮次
    next_step_fulfilled: bool | None = None  # 上轮 next_step 是否被执行（None=首轮）
    # 循环探针（由 loop 注入）：给 LLM 的结构化反循环信号
    repeat_action_count: int = 0
    repeat_action_tool: str = ""
    repeat_action_key: str = ""
    repeat_read_count: int = 0
    repeat_read_path: str = ""
    loop_probe_version: int = 0

    def to_text(self) -> str:
        """格式化为 LLM 可读文本，注入 judgment bundle。"""
        lines: list[str] = []
        lines.append(
            "loop_probe="
            f"{{version={self.loop_probe_version}, "
            f"repeat_action_count={self.repeat_action_count}, "
            f"repeat_action_tool='{self.repeat_action_tool}', "
            f"repeat_action_key='{self.repeat_action_key}', "
            f"repeat_read_count={self.repeat_read_count}, "
            f"repeat_read_path='{self.repeat_read_path}'}}"
        )
        if self.emotion_alert:
            lines.append(
                f"⚠️ 情绪激活偏高（{self.emotion_activation:.2f}）："
                "可能处于压力或亢奋状态，建议自检或放缓节奏"
            )
        if self.wm_pressure_alert:
            lines.append(
                f"⚠️ 工作记忆压力偏高（{self.wm_pressure:.0%}）"
                "  请先调用 memory.snapshot（快照并清空 WM），"
                "再视情况用 reflect.structural 提炼洞察"
            )
        if self.prediction_error_alert:
            lines.append(
                f"⚠️ 预测误差偏高（{self.prediction_error:.2f}）："
                "环境或任务状态超出预期，建议重新评估"
            )
        if not self.has_active_task:
            lines.append(f"ℹ️ 当前无活跃任务（已空转 {self.idle_cycles} 轮）")
            if self.idle_cycles >= 2:
                lines.append(
                    '→ 建议：使用 task.add 创建一个自驱任务，'
                    '例如"建立环境认知地图"或"初始化自我状态检查"'
                )
        if self.next_step_fulfilled is False:
            lines.append(
                "⚠️ 上一轮计划的 next_step 未真正推进（可能是 wait/pause，或 act 但没有产生新结果），"
                "注意避免计划漂移"
            )
        if not lines:
            lines.append("✓ 认知状态正常，无异常信号")
        return "\n".join(lines)


# ── 感知快照 ───────────────────────────────────────────────────────────────────

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


# ── 感知层 ─────────────────────────────────────────────────────────────────────

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
        emotion: EmotionState,
        cfg: "Config",
        *,
        has_active_task: bool = False,
        idle_cycles: int = 0,
        next_step_fulfilled: bool | None = None,
    ) -> CognitiveSignals:
        """将感知信号转化为认知状态报告，注入 LLM 判断上下文。

        设计原则：此方法只计算信号强度，不产生任何决策或任务文字。
        是否 task.add、如何命名任务、如何响应异常，全部由 LLM 在 judgment 层决定。
        """
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
            cwd = Path.cwd()
            entries = sorted(
                (p.name, p.stat().st_mtime)
                for p in cwd.iterdir()
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
        # WM 为空（首轮/清空后）或已被主动清空时，跳过大小变化信号，避免产生假预测误差
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
