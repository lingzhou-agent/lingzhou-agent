"""core/self_model.py — 自我模型追踪器。

设计哲学：
- 数字生命应该知道自己是谁、运行了多久、消耗了多少
- 不替 LLM 做决策，而是提供结构化的自我认知信号
- 轻量、可观测、不增加代码分支复杂度
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Config

# 模型定价（USD / 1M tokens）—— 按量模型的成本参考
_MODEL_PRICES: dict[str, dict[str, float]] = {
    "deepseek-v4-flash":  {"input": 0.28, "output": 1.10},
    "deepseek-v4-pro":    {"input": 0.55, "output": 2.19},
    "deepseek-chat":      {"input": 0.27, "output": 1.10},
    "deepseek-reasoner":  {"input": 0.55, "output": 2.19},
    "qwen3.6-plus":       {"input": 0.50, "output": 2.00},
    "qwen3.5-plus":       {"input": 0.35, "output": 1.40},

}


@dataclass
class SelfModel:
    """运行时自我认知快照。每次 tick 更新，注入感知上下文。"""

    # 身份
    name: str = "lingzhou"
    version: str = ""

    # 运行态
    started_at: float = 0.0
    tick_count: int = 0
    api_call_count: int = 0
    tool_call_count: int = 0

    # 模型路由摘要
    primary_model: str = ""
    reader_model: str = ""
    reasoner_model: str = ""

    # Token 用量（跨 tick 累计）
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    # 模型计费模式（按量 vs 按次）
    # 按量模型（deepseek/bailian）：token 消耗直接关联成本
    # 按次模型（copilot）：订阅制，token 不直接计费
    billing_mode: str = "token"  # "token" | "subscription" | "unknown"
    estimated_cost_usd: float = 0.0

    # 上下文预算（由判断层每 tick 注入）
    context_budget: str = ""
    context_pressure: float = 0.0

    # 健康
    recent_error_count: int = 0
    last_error: str = ""

    def record_start(self, *, name: str = "lingzhou", version: str = "") -> None:
        self.name = name
        self.version = version
        self.started_at = time.time()

    def record_tick(self) -> None:
        self.tick_count += 1

    def record_api_call(self) -> None:
        self.api_call_count += 1

    def record_tool_call(self) -> None:
        self.tool_call_count += 1

    def record_error(self, message: str) -> None:
        self.recent_error_count += 1
        self.last_error = message[:200]

    def record_token_usage(self, prompt: int = 0, completion: int = 0) -> None:
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.total_tokens += prompt + completion
        self._update_cost(prompt, completion)

    def _update_cost(self, prompt: int, completion: int) -> None:
        """按模型定价估算成本（USD）。"""
        prices = _MODEL_PRICES.get(self.primary_model.split("/", 1)[-1], {})
        input_price = prices.get("input", 0.0)
        output_price = prices.get("output", 0.0)
        self.estimated_cost_usd += (prompt / 1_000_000) * input_price + (completion / 1_000_000) * output_price

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at if self.started_at > 0 else 0

    @property
    def uptime_display(self) -> str:
        secs = int(self.uptime_seconds)
        if secs < 60:
            return f"{secs}s"
        if secs < 3600:
            return f"{secs // 60}m {secs % 60}s"
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s"

    def set_routing(self, cfg: "Config") -> None:
        self.primary_model = cfg.model
        routing = getattr(cfg, "routing", {}) or {}
        self.reader_model = routing.get("reader", cfg.model)
        self.reasoner_model = routing.get("reasoner", cfg.model)
        # 推断计费模式：copilot 走订阅，deepseek/bailian 走按量
        provider = cfg.model.split("/")[0] if "/" in cfg.model else ""
        self.billing_mode = "subscription" if provider == "copilot" else "token"

    # ── 持久化 ────────────────────────────────────────────────────────

    def to_json(self) -> str:
        import json as _json
        return _json.dumps({
            "tick_count": self.tick_count,
            "api_call_count": self.api_call_count,
            "tool_call_count": self.tool_call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "started_at": self.started_at,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str, *, name: str = "lingzhou") -> "SelfModel":
        import json as _json
        try:
            data = _json.loads(raw)
        except Exception:
            return cls(name=name)
        sm = cls(name=name)
        sm.tick_count = int(data.get("tick_count", 0))
        sm.api_call_count = int(data.get("api_call_count", 0))
        sm.tool_call_count = int(data.get("tool_call_count", 0))
        sm.total_prompt_tokens = int(data.get("total_prompt_tokens", 0))
        sm.total_completion_tokens = int(data.get("total_completion_tokens", 0))
        sm.total_tokens = int(data.get("total_tokens", 0))
        sm.estimated_cost_usd = float(data.get("estimated_cost_usd", 0))
        sm.started_at = float(data.get("started_at", 0))
        return sm


def fmt_self_model(sm: SelfModel) -> str:
    """将自我模型格式化为 LLM 感知上下文字段。"""
    lines = [
        f"名称: {sm.name}",
        f"已运行: {sm.uptime_display}  (tick #{sm.tick_count})",
        f"API 调用: {sm.api_call_count}  工具调用: {sm.tool_call_count}",
        f"Token 消耗: {sm.total_tokens:,}  (输入 {sm.total_prompt_tokens:,} + 输出 {sm.total_completion_tokens:,})",
        f"计费模式: {'按量' if sm.billing_mode == 'token' else '按次/订阅'}  |  估算成本: ${sm.estimated_cost_usd:.4f}",
        f"上下文预算: {sm.context_budget or '未设置'}  |  压力: {sm.context_pressure:.0%}",
        f"主模型: {sm.primary_model}",
        f"操作层: {sm.reader_model}",
        f"思考层: {sm.reasoner_model}",
    ]
    if sm.recent_error_count > 0:
        lines.append(f"最近错误: {sm.recent_error_count} 次  (最近: {sm.last_error[:80]})")
    else:
        lines.append("健康状态: 正常")
    if sm.billing_mode == "token" and sm.estimated_cost_usd > 0.01:
        lines.append(f"⚠️ 本会话已消耗 ${sm.estimated_cost_usd:.4f}（按量计费，请关注空转）")
    return "\n".join(lines)
