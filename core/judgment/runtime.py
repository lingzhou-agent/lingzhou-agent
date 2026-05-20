"""core/judgment/runtime.py — 判断层。

职责：
1. 组装 bundle（运行时状态 → 结构化 context）
2. 填入 prompts/judgment.md 模板（{{variable}} 语法）
3. 调用 LLM provider
4. 解析 JSON 输出 → JudgmentOutput

解耦原则：此模块不知道工具如何执行，只负责"决定做什么"。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from provider.catalog import lookup_model
from core.self_model import SelfModel, fmt_self_model
from tools.registry import tool_has_capability
from .context import (
    _emotion_label,
    _fill_template,
    _fmt_chat_history,
    _fmt_cognitive_signals,
    _fmt_context_facts,
    _fmt_current_time,
    _fmt_durable_failures,
    _fmt_ethos,
    _fmt_failures,
    _fmt_hard_boundaries,
    _fmt_judgment_signals,
    _fmt_memories,
    _fmt_memory_system,
    _fmt_perception_replay,
    _fmt_percept,
    _fmt_probe_sensors,
    _fmt_blind_spots,
    _fmt_recent_runs,
    _fmt_shell_capabilities,
    _fmt_skill_catalog,
    _fmt_skills,
    _fmt_soul,
    _fmt_task,
    _fmt_tools,
    _fmt_waiting_tasks,
    _fmt_wm,
    _load_context_facts_snapshot,
    _load_durable_failure_snapshot,
    _validate_context_schema,
    apply_context_budget,
)

_log = logging.getLogger("lingzhou.judgment")

# 低成本读取/枚举类工具 → reader tier（也被 loop.py 用于自动 tier 推断）
_READER_TOOLS = frozenset({
    "file.list", "file.read",
    "memory.get_fact", "memory.search",
    "schedule.list", "schedule.ack", "schedule.cancel",
    "shell.capabilities",
    "task.list",
    "failure.dismiss",
    # 探针：只读列举，不触发执行
    "probe.list",
})
READER_TOOLS = _READER_TOOLS  # 公开别名，供外部模块引用
# 写入/推理/高风险工具 → reasoner tier
_REASONER_TOOLS = frozenset({
    "shell.run",
    "file.write",
    "task.add", "task.update", "task.advance", "task.complete", "task.fail",
    "memory.add_wm", "memory.add_semantic", "memory.set_fact", "memory.snapshot",
    "reflect.structural",
    "schedule.add",
    # 探针：安装/拆除/执行/启停均有副作用
    "probe.install", "probe.remove", "probe.run", "probe.enable", "probe.disable",
})

_TOOL_TIER_MAPPING = {
    "reader": sorted(_READER_TOOLS),
    "reasoner": sorted(_REASONER_TOOLS),
    "repair": [],
}

def _structured_tool_history_window(tool_history: list[dict[str, Any]]) -> tuple[str, str]:
    history_parts: list[str] = []
    structured_window: list[dict[str, Any]] = []
    start_index = max(0, len(tool_history) - 6)
    for index, item in enumerate(tool_history[start_index:], start=start_index + 1):
        raw_params = item.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        status = str(item.get("status") or "").strip() or (
            "error" if str(item.get("error") or "").strip() else ("skipped" if item.get("skipped") else "ok")
        )
        summary = str(item.get("summary") or item.get("result") or "").strip()
        error = str(item.get("error") or "").strip()
        state_delta = item.get("state_delta") if isinstance(item.get("state_delta"), dict) else {}
        key = (
            params.get("path")
            or params.get("name")
            or params.get("title")
            or params.get("key")
            or str(params.get("id") or "")
            or params.get("command")
            or params.get("query")
            or ""
        )
        structured_window.append({
            "index": index,
            "tool": str(item.get("tool") or ""),
            "status": status,
            "key": str(key),
            "summary": summary[:400],
            "error": error[:240],
            "error_category": str(item.get("error_category") or ""),
            "state_delta": state_delta,
        })
        parts = [f"[{index}] tool={item.get('tool', '')} status={status}"]
        if key:
            parts.append(f"key={key}")
        if summary:
            parts.append(f"summary={summary[:240]}")
        if error:
            parts.append(f"error={error[:180]}")
        if state_delta:
            try:
                state_text = json.dumps(state_delta, ensure_ascii=False, sort_keys=True)
            except Exception:
                state_text = str(state_delta)
            parts.append(f"state_delta={state_text[:200]}")
        history_parts.append(" | ".join(parts))
    return (
        json.dumps(structured_window, ensure_ascii=False, indent=2),
        "\n".join(history_parts),
    )


def _build_team_view_from_cfg(cfg) -> str:
    """从配置构建思考模型的团队视图（不依赖运行时 model list）。"""
    routing = getattr(cfg, "routing", {}) or {}
    lines = ["## 团队架构（你作为思考模型，统筹以下资源）"]
    for tier in ("reader", "reasoner", "repair"):
        model = routing.get(tier, cfg.model)
        role = {
            "reader": "操作层 — 快速/低成本，适合 file.read/task.list/memory.search 等轻量查询",
            "reasoner": "思考层 — 你本人，深度推理，适合复杂决策/代码修改/用户交互",
            "repair": "修复层 — 格式修复/小修小补",
        }.get(tier, tier)
        lines.append(f"- {tier}: {model}")
        lines.append(f"  {role}")
    lines.append("")
    lines.append("调度规则:")
    lines.append("- 你是 reasoner（思考层），负责复杂决策。简单操作应委派给 reader")
    lines.append("- 委派方式: 在 model_strategy 中设置 next_phase_tier='reader'")
    lines.append("- reader 的操作层模型成本低速度快，不要在 reasoner 上做纯查询")
    lines.append("- 关键决策、代码修改、用户交互必须由你亲自处理")
    return "\n".join(lines)


def tool_tier(tool_id: str, registry = None) -> str:
    """判断工具应该用哪个 tier。

    优先级：manifest.prefer_tier → 硬编码集合 → 默认 reasoner。
    数据驱动的工具可以声明 prefer_tier，无需改此处。
    """
    # 从注册表读取 manifest
    if registry is not None and tool_id:
        entry = registry.get(tool_id)
        if entry and entry.manifest.prefer_tier:
            return entry.manifest.prefer_tier
    # 回退到硬编码集合
    if tool_id in _READER_TOOLS:
        return "reader"
    if tool_id in _REASONER_TOOLS:
        return "reasoner"
    return "reasoner"


@dataclass
class ModelSelection:
    phase: str
    tier: str
    model_ref: str
    thinking: str


@dataclass
class ModelHealth:
    cooldown_until: float = 0.0
    failure_streak: int = 0
    last_error: str = ""
    last_code: str = ""

if TYPE_CHECKING:
    from core.config import Config
    from core.perception import (
        Percept, EmotionState, EthosState, JudgmentSignals, PerceptionReplaySummary,
        CognitiveSignals,
    )
    from core.skill import Skill
    from memory.working import WorkingMemory
    from memory.task_store import TaskStore
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from tools.registry import ToolRegistry
    from provider.base import Provider


# ── 判断输出 ───────────────────────────────────────────────────────────────────

@dataclass
class JudgmentOutput:
    decision: str = "wait"              # act | pause | wait
    chosen_action_id: str = ""          # 工具名称
    params: dict[str, Any] = field(default_factory=lambda: {})  # type: ignore[assignment]
    rationale: str = ""                 # 内部推理过程（内部独白）
    reflection: str = ""                # 对最近经历的后验反思（写入语义记忆）
    reply_to_user: str = ""             # 对人类的外部回复（与 rationale 明确分离）
    next_step: str = ""
    model_strategy: dict[str, Any] = field(default_factory=dict)
    applied_skills: list[str] = field(default_factory=list)  # LLM 实际应用的技能名单

    @classmethod
    def wait(cls, reason: str = "") -> "JudgmentOutput":
        return cls(decision="wait", rationale=reason, reply_to_user="")

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @classmethod
    def from_llm(cls, text: str) -> "JudgmentOutput":
        """从 LLM 输出文本解析 JudgmentOutput，容错处理。"""
        original = text.strip()
        text = original
        # 防御：剥离 <think>...</think> 块（provider 层已处理，此处兜底）
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        # 防御：剥离 LLM 混入的非 JSON 内容（markdown 树形图等）
        # 模式：以路径/目录树开头的行（如 "provider/\n├── __init__.py"）
        if not text.startswith("{") and not text.startswith("```"):
            stripped = re.sub(
                r"(?:^|\n)(?:[\w./_-]+/\s*\n)?(?:[│├└──+|\\]+[ \t]+[\w.+/_-]+[ \t]*(?:#.*)?\n)+",
                "\n",
                text,
            )
            if stripped != text:
                text = stripped.strip()
        # 如果清理后文本为空或不含 JSON，提前返回避免无谓的 parse 尝试
        if not text or ("{" not in text and "decision" not in text):
            return cls(decision="pause", rationale=f"LLM 输出解析失败（非JSON）: {original[:120]}")
        # 裸代码检测：LLM 直接输出 bash/python 脚本时提前标记
        _CODE_PREFIXES = ("#!/", "```bash", "```python", "```sh", "```shell", "# -*-")
        _is_raw_code = any(text.lstrip().startswith(p) for p in _CODE_PREFIXES)
        # 提取 JSON 块（支持 ```json ... ``` 或裸 JSON）
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if match:
            text = match.group(1).strip()
        else:
            # 尝试找第一个 { ... }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback 1: 尝试修复常见 JSON 格式错误（单引号→双引号、移除尾随逗号）后重试
            fixed = text.replace("'", '"')
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            try:
                data = json.loads(fixed)
            except json.JSONDecodeError:
                # 裸代码兜底：将代码内容封装到 reply_to_user，不丢失产出
                if _is_raw_code:
                    return cls(
                        decision="pause",
                        chosen_action_id="",
                        params={},
                        rationale="[auto-wrap] LLM 输出了裸代码，已封装为 reply_to_user",
                        reflection="格式错误：代码应放入 reply_to_user 或 params 字段，不能直接输出",
                        reply_to_user=original,
                        next_step="",
                        model_strategy={},
                    )
                return cls(decision="pause", rationale=f"LLM 输出解析失败: {text}")

        return cls(
            decision=cls._coerce_text(data.get("decision", "wait")).lower(),
            chosen_action_id=cls._coerce_text(data.get("chosen_action_id", "")),
            params=dict(data.get("params") or {}),
            rationale=cls._coerce_text(data.get("rationale", "")),
            reflection=cls._coerce_text(data.get("reflection", "")),
            reply_to_user=cls._coerce_text(data.get("reply_to_user", "")),
            next_step=cls._coerce_text(data.get("next_step", "")),
            model_strategy=dict(data.get("model_strategy") or {}),
            applied_skills=[str(s) for s in (data.get("applied_skills") or []) if s],
        )


# ── 判断层 ─────────────────────────────────────────────────────────────────────

# Note (P2): JudgmentLayer is 1684 lines. Future refactoring could extract:
# - _build_routing_prompt → separate routing module
# - _parse_action → action parser module
# - _select_tool → tool selector module
class JudgmentLayer:
    def __init__(
        self,
        provider: "Provider",
        registry: "ToolRegistry",
        cfg: "Config",
    ) -> None:
        from core.skill import SkillRegistry
        from core.reference import ReferenceResolver
        self._provider = provider
        self._registry = registry
        self._cfg = cfg
        self._system_prompt = cfg.load_prompt("system")
        self._identity_prefix: str = ""   # bootstrap 注入的永久身份前缀（不随 WM 驱逐）
        self._judgment_template = cfg.load_prompt("judgment")
        _skills_dir = Path(cfg.loop.workspace_dir).expanduser() / "skills"
        self._skills = SkillRegistry(skills_dir=_skills_dir)
        self._ref_resolver = ReferenceResolver(provider=provider)
        # 自我模型追踪：持久化运行态与任务连续性
        self.self_model = SelfModel()
        # 分层路由 providers：{"simple": <provider>, "complex": <provider>}
        # 由 loop.open() 在 bootstrap 后注入，未配置时为空字典
        self._routing_providers: dict[str, "Provider"] = {}
        # 内层工具循环用：缓存上一次 decide() 组装的完整上下文，由 decide_continue() 复用
        self._last_context_text: str = ""
        # 探针系统引用：由 CognitionLoop.__init__ 在创建 ProbeManager 后注入
        self._probe_manager: Any = None
        # 最近一次真实 LLM 调用元数据（供 loop 日志输出实际 model/tier/thinking）
        self._last_call_meta: dict[str, str] = {
            "phase": "",
            "tier": "default",
            "model_ref": cfg.model,
            "thinking": cfg.thinking,
            "skills": "",
        }
        self._last_selected_skills: list[Skill] = []
        # 每个模型最近一次调用错误（用于注入 model routing truth）
        self._provider_errors: dict[str, str] = {}
        # 每个模型的健康状态（429/400/timeout 触发冷却窗口，避免短时间重复打爆同一 provider）
        self._model_health: dict[str, ModelHealth] = {}
        # 运行时临时 provider 缓存：routing_overrides 指定的临时 model 按需创建并缓存
        self._override_providers: dict[str, "Provider"] = {}

    def _track_token_usage(self, provider: "Provider") -> None:
        """从 provider 读取 last_usage 并累积到 self_model。"""
        usage = getattr(provider, "last_usage", None)
        if isinstance(usage, dict):
            self.self_model.record_token_usage(
                prompt=usage.get("prompt_tokens", 0),
                completion=usage.get("completion_tokens", 0),
            )

    def set_identity_prefix(self, prefix: str) -> None:
        """由 SoulManager.bootstrap() 调用，将 BOOTSTRAP.md/IDENTITY.md 永久注入 system prompt。"""
        self._identity_prefix = prefix
        _log.debug("[judgment] identity_prefix 已设置（%d 字符）", len(prefix))

    def reload_prompt(self, key: str) -> None:
        """evolution 进化提示词后调用，热重载模板。"""
        if key == "judgment":
            self._judgment_template = self._cfg.load_prompt("judgment")
        elif key == "system":
            self._system_prompt = self._cfg.load_prompt("system")
    def set_routing_providers(self, providers: dict[str, "Provider"]) -> None:
        """注入分层路由 providers（由 CognitionLoop.open() 调用）。
        key: 'simple'（空闲/后台 tick）或 'complex'（有用户消息 / 高优先任务）
        """
        self._routing_providers = providers
        if providers:
            tiers = list(providers.keys())
            _log.info("[judgment] 路由 providers 已设置: %s", tiers)

    @property
    def last_call_meta(self) -> dict[str, str]:
        return dict(self._last_call_meta)

    @staticmethod
    def _skills_for_log(skills: list["Skill"]) -> str:
        if not skills:
            return "none"
        return ",".join(skill.name for skill in skills[:3])

    def _routing_aliases(self, tier: str) -> tuple[str, ...]:
        return {
            "reader": ("reader", "simple"),
            "reasoner": ("reasoner", "complex"),
            "repair": ("repair", "reader", "simple"),
        }.get(tier, (tier,))

    def _resolve_tier_model(self, tier: str) -> tuple[str, str]:
        for alias in self._routing_aliases(tier):
            model_ref = self._cfg.routing.get(alias)
            if model_ref:
                return alias, model_ref
        return "default", self._cfg.model

    def _tier_fallback_models(self, tier: str) -> list[str]:
        """返回某个 tier 的显式回退模型链（按配置顺序）。"""
        out: list[str] = []
        for key in (tier, *self._routing_aliases(tier)):
            for m in self._cfg.model_fallbacks.get(key, []):
                if m and m not in out:
                    out.append(m)
        return out

    def _tier_model_candidates(
        self,
        tier: str,
        routing_overrides: dict[str, str] | None = None,
    ) -> list[str]:
        """按优先级构建 tier 的候选模型：override -> routing 主模型 -> 显式 fallback -> 顶层 model。"""
        candidates: list[str] = []

        override_model = (routing_overrides or {}).get(tier)
        if override_model:
            candidates.append(override_model)

        _, primary = self._resolve_tier_model(tier)
        if primary and primary not in candidates:
            candidates.append(primary)

        for m in self._tier_fallback_models(tier):
            if m not in candidates:
                candidates.append(m)

        if self._cfg.model not in candidates:
            candidates.append(self._cfg.model)

        return candidates

    def _get_health(self, model_ref: str) -> ModelHealth:
        h = self._model_health.get(model_ref)
        if h is None:
            h = ModelHealth()
            self._model_health[model_ref] = h
        return h

    def _classify_error_code(self, err_text: str) -> str:
        text = (err_text or "").lower()
        if " 429 " in f" {text} " or "too many requests" in text:
            return "429"
        if " 401 " in f" {text} " or "unauthorized" in text:
            return "401"
        if " 403 " in f" {text} " or "forbidden" in text:
            return "403"
        if " 400 " in f" {text} " or "bad request" in text:
            return "400"
        if "readtimeout" in text or "timeout" in text:
            return "timeout"
        return "other"

    def _cooldown_seconds(self, code: str, failure_streak: int) -> float:
        streak = max(1, failure_streak)
        if code == "429":
            return min(180.0, 30.0 * streak)
        if code in {"401", "403"}:
            return min(300.0, 120.0 + 30.0 * (streak - 1))
        if code == "400":
            return min(180.0, 45.0 * streak)
        if code == "timeout":
            return min(120.0, 20.0 * streak)
        return min(90.0, 15.0 * streak)

    def _mark_model_failure(self, model_ref: str, err_text: str) -> None:
        code = self._classify_error_code(err_text)
        health = self._get_health(model_ref)
        health.failure_streak += 1
        health.last_error = err_text  # 保留完整错误信息，不截断
        health.last_code = code
        health.cooldown_until = time.time() + self._cooldown_seconds(code, health.failure_streak)
        self._provider_errors[model_ref] = health.last_error

    def _mark_model_success(self, model_ref: str) -> None:
        health = self._get_health(model_ref)
        health.failure_streak = 0
        health.last_error = ""
        health.last_code = ""
        health.cooldown_until = 0.0
        self._provider_errors.pop(model_ref, None)

    def _is_model_available(self, model_ref: str) -> bool:
        return self._get_health(model_ref).cooldown_until <= time.time()

    def _find_or_create_provider(self, model_ref: str) -> "Provider":
        """按 model_ref 找到或创建 provider（用于 routing_overrides 临时覆盖）。"""
        if model_ref == self._cfg.model:
            return self._provider
        for p in self._routing_providers.values():
            if getattr(p, "_model", None) == model_ref:
                return p
        if model_ref not in self._override_providers:
            from provider import create_provider_with_model
            self._override_providers[model_ref] = create_provider_with_model(self._cfg, model_ref)
        return self._override_providers[model_ref]

    def _fallback_tiers(self, tier: str) -> tuple[str, ...]:
        if tier == "reasoner":
            return ("reader", "repair")
        if tier == "reader":
            return ("reasoner", "repair")
        if tier == "repair":
            return ("reader", "reasoner")
        return ("reader", "reasoner", "repair")

    def _tool_history_has_error(self, tool_history: list[dict[str, Any]] | None) -> bool:
        if not tool_history:
            return False
        return any(str(item.get("result", "")).startswith("ERROR:") for item in tool_history)

    def _select_tier(
        self,
        *,
        phase: str,
        user_message: str,
        current_action: str = "",
        tool_history: list[dict[str, Any]] | None = None,
        prefer_tier: str | None = None,
    ) -> str:
        if phase == "repair":
            return "repair"
        if prefer_tier in {"reader", "reasoner", "repair"}:
            return prefer_tier
        if phase == "continue":
            if current_action in _REASONER_TOOLS:
                return "reasoner"
            if current_action in _READER_TOOLS and not self._tool_history_has_error(tool_history):
                return "reader"
            if user_message or self._tool_history_has_error(tool_history):
                return "reasoner"
            if tool_history and len(tool_history) >= self._cfg.loop.continue_reasoner_after_n_tools:
                return "reasoner"
            return "reader"
        if phase in {"reply", "final"}:
            return "reasoner"
        return "reasoner"

    def _select_provider(
        self,
        *,
        phase: str,
        user_message: str,
        current_action: str = "",
        tool_history: list[dict[str, Any]] | None = None,
        prefer_tier: str | None = None,
        thinking_override: str | None = None,
        routing_overrides: dict[str, str] | None = None,
    ) -> tuple["Provider", ModelSelection]:
        tier = self._select_tier(
            phase=phase,
            user_message=user_message,
            current_action=current_action,
            tool_history=tool_history,
            prefer_tier=prefer_tier,
        )
        chosen_tier = tier
        chosen_model = self._cfg.model
        provider: "Provider" = self._provider
        selected = False

        # 先试当前 tier，再按 tier fallback 试其他 tier。
        # 每个 tier 内按：override -> routing 主模型 -> model_fallbacks -> 顶层 model。
        for cand_tier in (tier, *self._fallback_tiers(tier)):
            for model_ref in self._tier_model_candidates(cand_tier, routing_overrides=routing_overrides):
                if not self._is_model_available(model_ref):
                    continue
                try:
                    provider = self._find_or_create_provider(model_ref)
                    chosen_tier = cand_tier
                    chosen_model = model_ref
                    selected = True
                    break
                except Exception as e:
                    _log.warning("[routing] tier=%s model=%s provider 构建失败，跳过: %s", cand_tier, model_ref, e)
                    continue
            if selected:
                break

        thinking = thinking_override if thinking_override is not None else self._cfg.thinking
        return provider, ModelSelection(phase=phase, tier=chosen_tier, model_ref=chosen_model, thinking=thinking)

    def _cost_level_for_model(self, model_ref: str, reasoning: bool) -> str:
        _name = model_ref.lower()
        if "gpt-5" in _name or "o3" in _name or "qwen3-max" in _name:
            return "high"
        if reasoning or "mini" in _name or "qwen3.5" in _name:
            return "medium"
        return "low"

    def _latency_level_for_model(self, model_ref: str, reasoning: bool) -> str:
        _name = model_ref.lower()
        if "gpt-5" in _name or "o3" in _name:
            return "high"
        if reasoning or "max" in _name:
            return "medium"
        return "low"

    def _build_model_routing_section(
        self,
        *,
        phase: str,
        user_message: str,
        current_action: str,
        tool_history: list[dict[str, Any]] | None,
        effective_thinking: str,
        routing_overrides: dict[str, str] | None = None,
    ) -> str:
        route_tiers: list[str] = ["reader", "reasoner", "repair"]
        available_models: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for tier in route_tiers:
            _, model_ref = self._resolve_tier_model(tier)
            key = (tier, model_ref)
            if key in seen:
                continue
            seen.add(key)
            model_id = model_ref.split("/", 1)[1] if "/" in model_ref else model_ref
            spec = lookup_model(model_id) or {}
            reasoning = bool(spec.get("reasoning"))
            last_error = self._provider_errors.get(model_ref)
            health = self._get_health(model_ref)
            # 检查该 tier 是否被临时覆盖
            override_model = (routing_overrides or {}).get(tier)
            available_models.append({
                "tier": tier,
                "model": model_ref,
                "available": self._is_model_available(model_ref),
                "reasoning": reasoning,
                "cost_level": self._cost_level_for_model(model_ref, reasoning),
                "latency_level": self._latency_level_for_model(model_ref, reasoning),
                "context_window": spec.get("context_window") or self._cfg.context_window_tokens,
                "current_thinking": effective_thinking or self._cfg.thinking,
                "last_error": last_error,
                "last_error_code": health.last_code or None,
                "cooldown_remaining_sec": max(0, int(health.cooldown_until - time.time())),
                "overridden_by": override_model if override_model and override_model != model_ref else None,
            })

        task_explore_count = 0
        repeat_action_count = 0
        repeat_read_count = 0
        if tool_history:
            task_explore_count = sum(1 for item in tool_history if item.get("tool") in {"file.list", "file.read", "shell.run"})
            if len(tool_history) >= 2:
                _last_tool = str(tool_history[-1].get("tool", ""))
                repeat_action_count = sum(1 for item in reversed(tool_history) if str(item.get("tool", "")) == _last_tool)
                if _last_tool == "file.read":
                    _last_path = json.dumps(tool_history[-1].get("params", {}), ensure_ascii=False)
                    repeat_read_count = sum(
                        1 for item in reversed(tool_history)
                        if str(item.get("tool", "")) == "file.read"
                        and json.dumps(item.get("params", {}), ensure_ascii=False) == _last_path
                    )

        ask_evidence_hits = sum(
            1 for item in (tool_history or [])
            if tool_has_capability(self._registry, str(item.get("tool") or ""), "ask_evidence")
            and str(item.get("result") or "").strip()
            and not str(item.get("result") or "").startswith("ERROR[")
        )

        posture = "respond" if user_message else ("converge" if task_explore_count >= 4 else "conserve")
        implicit_next_phase_default = None
        if current_action in _READER_TOOLS:
            implicit_next_phase_default = {
                "tier": "reader",
                "trigger": f"last_action={current_action}",
                "condition": "仅在本轮未显式设置 next_phase_tier 时生效",
            }
        capability_mapping: dict[str, list[str]] = {}
        current_action_caps: list[str] = []
        for manifest in self._registry.list_manifests():
            for cap in manifest.capabilities:
                capability_mapping.setdefault(cap, []).append(manifest.name)
            if manifest.name == current_action:
                current_action_caps = sorted(list(manifest.capabilities))
        payload = {
            "active_overrides": routing_overrides or {},
            "tool_tier_mapping": _TOOL_TIER_MAPPING,
            "tool_capability_mapping": {k: sorted(v) for k, v in capability_mapping.items()},
            "current_action_capabilities": current_action_caps,
            "implicit_next_phase_default": implicit_next_phase_default,
            "tier_descriptions": {
                "reader": "轻量感知层：适合常规状态查询、读文件、检查计划、无复杂推理的心跳 tick",
                "reasoner": "深度推理层：适合用户交互、要求判断、处理复杂状态、制定或调整计划",
                "repair": "修复层：专用于解析失败、格式错误、小修小补",
            },
            "delegation_guide": (
                "你是当前层的决策者，可以通过 model_strategy 中的以下字段调控下一轮行为：\n"
                "• next_phase_tier：分配下轮的推理层级。reader=轻量感知，reasoner=深度推理，repair=修复。"
                "示例：本轮已完成复杂判断并写入任务，下轮只需追踪状态 → next_phase_tier=reader；\n"
                "• tool_tier_mapping：runtime 当前对工具族的默认分层真相；若你觉得某次具体动作应临时跨层处理，可通过 next_phase_tier 或 routing_overrides 调整，但不要假装这份映射不存在。\n"
                "• tool_capability_mapping：runtime 注入的工具能力真相（如 ask_evidence / plan_bootstrap_exempt / completion_verify）。"
                "优先按能力标签推理，不要仅凭工具名字猜类别。\n"
                "• implicit_next_phase_default：runtime 的隐式下轮 tier 默认行为。若该字段非空，表示你本轮若不显式设置 next_phase_tier，loop 可能按这里的规则自动选择下一轮 tier。\n"
                "• next_idle_gap_secs / next_idle_gap_ms：【必须设置其中之一！】你的生命节奏控制器。"
                "next_idle_gap_secs 单位秒（小数可用，如 0.5 = 500ms），next_idle_gap_ms 单位毫秒（整数，如 500 = 500ms）；两者同时设置时 ms 优先。"
                "范围由 idle_with_task_bounds / idle_no_task_bounds 决定（默认有任务时 100ms-30s，无任务时 5s-300s）。"
                "你必须根据当前上下文主动选择一个合理值，不要依赖默认："
                "已发起shell预计30s出结果 → next_idle_gap_secs=35；刚回复完用户等下一步 → next_idle_gap_secs=120；"
                "任务推进中需快速追踪 → next_idle_gap_ms=500；实时等待短命令结束 → next_idle_gap_ms=200。"
                "不设置此字段则用兜底值 60 秒。控制权在你手里。\n"
                "• routing_overrides：临时覆盖 tier→model 映射，格式 {\"reader\": \"bailian/qwen3.6-plus\"}。"
                "可选 tier: reader / reasoner / repair。从 catalog_models 中选择可用模型。"
                "设为 {} 可清除覆盖。覆盖持久到显式修改，无需每轮重复设置。\n"
                "• thinking_override：覆盖下一轮的 thinking 等级，可选局 off / minimal / low / medium / high。"
                "当前等级见 available_models[].current_thinking。"
                "示例：下轮需要深度推理 → thinking_override=\"high\"；下轮只需快速响应 → thinking_override=\"low\"。"
                "设为 null 或不填则恢复全局配置。仅对支持 thinking 的模型有效（reasoning=true）。\n"
                "没有明确偏好时用 default，进化机制将决定。"
            ),
            "budget_state": {
                "task_explore_count": task_explore_count,
                "repeat_action_count": repeat_action_count,
                "repeat_read_count": repeat_read_count,
                "ask_evidence_hits": ask_evidence_hits,
                "global_cost_posture": posture,
            },
            "routing_hint": {
                "phase": phase,
                "current_action": current_action,
                "user_message_present": bool(user_message),
            },
        }
        # 全量 catalog 模型列表（所有 provider 所有模型），让 LLM 能看到可用选项
        from provider import catalog as _cat
        catalog_entries: list[dict[str, Any]] = []
        for _pname in _cat.list_providers():
            for _m in _cat.list_provider_models(_pname):
                catalog_entries.append({
                    "model": f"{_pname}/{_m.get('id', '')}",
                    "provider": _pname,
                    "reasoning": bool(_m.get("reasoning")),
                    "context_window": _m.get("context_window"),
                })
        payload["catalog_models"] = catalog_entries
        return json.dumps(payload, ensure_ascii=False, indent=2)

    async def decide(
        self,
        percept: "Percept",
        wm: "WorkingMemory",
        task_store: "TaskStore",
        episodic: "EpisodicMemory",
        semantic: "SemanticMemory",
        emotion: "EmotionState",
        user_message: str = "",
        ethos_state: "EthosState | None" = None,
        judgment_signals: "JudgmentSignals | None" = None,
        hard_boundaries: "list[str] | None" = None,
        perception_replay: "PerceptionReplaySummary | None" = None,
        cognitive_signals: "CognitiveSignals | None" = None,
        thinking_override: "str | None" = None,
        prefer_tier: "str | None" = None,
        routing_overrides: "dict[str, str] | None" = None,
        phase: str = "initial",
    ) -> JudgmentOutput:
        """组装上下文，调用 LLM，返回决策。
        
        thinking_override: 覆盖 cfg.thinking（如 chat 模式用 "low" 加速首轮判断）。
        routing_overrides: 临时覆盖 tier→model 映射（由 loop.py 从 model_strategy 读取）。
        """
        from provider.base import Message

        context_text = await self._assemble_context(
            percept, wm, task_store, episodic, semantic, emotion,
            user_message=user_message,
            ethos_state=ethos_state,
            judgment_signals=judgment_signals,
            hard_boundaries=hard_boundaries,
            perception_replay=perception_replay,
            cognitive_signals=cognitive_signals,
            phase=phase,
            current_action="",
            tool_history=None,
            effective_thinking=thinking_override or self._cfg.thinking,
            routing_overrides=routing_overrides,
        )
        # 缓存给内层工具循环的续判请求用
        self._last_context_text = context_text

        _sys = (
            self._identity_prefix + "\n\n" + self._system_prompt
            if self._identity_prefix
            else self._system_prompt
        )
        messages = [
            Message(role="system", content=_sys),
            Message(role="user", content=context_text),
        ]

        selected_provider, selection = self._select_provider(
            phase=phase,
            user_message=user_message,
            prefer_tier=prefer_tier,
            thinking_override=thinking_override,
            routing_overrides=routing_overrides,
        )
        raw: str | None = None
        for _attempt in range(2):
            _primary = self._last_selected_skills[0] if self._last_selected_skills else None
            self._last_call_meta = {
                "phase": selection.phase,
                "tier": selection.tier,
                "model_ref": selection.model_ref,
                "thinking": selection.thinking,
                "skills": self._skills_for_log(self._last_selected_skills),
                "primary_skill": _primary.name if _primary else None,
                "primary_skill_guidance": bool(_primary and getattr(_primary, "guidance", None)),
            }
            try:
                raw = await selected_provider.chat(messages, thinking_override=thinking_override)
                self._mark_model_success(selection.model_ref)
                self._track_token_usage(selected_provider)
                break
            except Exception as exc:
                _err = str(exc) or repr(exc)
                self._mark_model_failure(selection.model_ref, _err)
                if _attempt == 0:
                    _fallback_tier = self._fallback_tiers(selection.tier)[0]
                    fb_provider, fb_selection = self._select_provider(
                        phase=phase,
                        user_message=user_message,
                        current_action="",
                        tool_history=None,
                        prefer_tier=_fallback_tier,
                        thinking_override=thinking_override,
                        routing_overrides=routing_overrides,
                    )
                    if fb_selection.model_ref != selection.model_ref:
                        _log.warning(
                            "[judgment] LLM 调用失败，切换模型重试: from=%s(%s) to=%s(%s) err=%s",
                            selection.model_ref,
                            selection.tier,
                            fb_selection.model_ref,
                            fb_selection.tier,
                            _err,
                        )
                        selected_provider, selection = fb_provider, fb_selection
                        continue
                    _log.warning("[judgment] LLM 调用失败，1s 后重试: %s", _err)
                    await asyncio.sleep(1.0)
                else:
                    _log.warning("[judgment] LLM 调用失败: %s", _err)
                    return self._simulate_safe_output(
                        failure_count=0,
                        signals=judgment_signals,
                        hard_boundaries=hard_boundaries or [],
                        reason=_err,
                    )
        assert raw is not None  # 两次都失败时上面已 return

        output = JudgmentOutput.from_llm(raw)

        # 解析失败时尝试一次修复，避免因为截断/格式噪声直接进入空转
        if output.rationale.startswith("LLM 输出解析失败"):
            repaired = await self._repair_output(context_text, raw)
            if repaired is not None:
                output = repaired
            else:
                await task_store.record_failure("judgment_parse", output.rationale)  # 保留完整信息，不截断

        if output.decision not in ("act", "pause", "wait"):
            output = JudgmentOutput.wait(reason=f"无效 decision: {output.decision!r}")
        if output.decision == "act" and not output.chosen_action_id:
            repaired = await self._repair_output(context_text, raw)
            if repaired is not None and repaired.decision == "act" and repaired.chosen_action_id:
                output = repaired
            else:
                output = JudgmentOutput.wait(reason="act 决策缺少 chosen_action_id")

        _applied = ",".join(output.applied_skills) if output.applied_skills else "none"
        _log.info(
            "[judgment] phase=%s tier=%s model=%s thinking=%s applied_skills=%s decision=%s action=%s rationale=%s",
            selection.phase, selection.tier, selection.model_ref, selection.thinking,
            _applied,
            output.decision, output.chosen_action_id, output.rationale or "",
        )

        return output

    async def decide_continue(
        self,
        tool_history: list[dict],
        user_message: str = "",
        active_task: Any | None = None,
        prefer_tier: str | None = None,
        thinking_override: str | None = None,
        routing_overrides: "dict[str, str] | None" = None,
        reply_only: bool = False,
    ) -> JudgmentOutput:
        """内层工具循环的续判请求。

        不重践 perception 链路，直接在上次 decide() 缓存的全量上下文后面追加工具历史续判。
        每次 HTTP 请求与普通请求相同，但输入 token 显著减少（不重发全量感知层）。

        Args:
            tool_history: [{"tool": str, "params": dict, "result": str}, ...]
            user_message:  原始用户消息（不再次向 LLM 重复，仅用于选择 provider tier）
        """
        from provider.base import Message

        if not self._last_context_text:
            return JudgmentOutput.wait(reason="[inner-loop] no cached context for continuation")

        history_json_block, history_block = _structured_tool_history_window(tool_history)
        if reply_only:
            continuation_context = (
                f"{self._last_context_text}\n\n"
                "---\n"
                "## 结构化最近工具结果(JSON)\n"
                f"{history_json_block}\n\n"
                "## 本轮已执行工具历史\n"
                f"{history_block}\n\n"
                "你现在处于最终回复阶段。禁止再调用任何工具。"
                "请只基于已有证据生成对用户的最终 reply_to_user。"
                "decision 只能是 pause 或 wait，chosen_action_id 必须留空。"
            )
        else:
            hint = (
                "用户正在等待回复，尽快在本轮设置 reply_to_user 字段。"
                if user_message else ""
            )
            continuation_context = (
                f"{self._last_context_text}\n\n"
                "---\n"
                "## 结构化最近工具结果(JSON)\n"
                f"{history_json_block}\n\n"
                "## 本轮已执行工具历史\n"
                f"{history_block}\n\n"
                "优先依据结构化结果判断当前状态，不要只凭模糊回忆续写。\n\n"
                f"请根据以上结果继续执行下一个必要工具，或生成最终回复（reply_to_user 非空）。{hint}"
            )

        _sys = (
            self._identity_prefix + "\n\n" + self._system_prompt
            if self._identity_prefix
            else self._system_prompt
        )
        messages = [
            Message(role="system", content=_sys),
            Message(role="user", content=continuation_context),
        ]

        current_action = "" if reply_only else str(tool_history[-1].get("tool", "")) if tool_history else ""
        phase = "reply" if reply_only else "continue"
        forced_prefer_tier = "reasoner" if reply_only else prefer_tier
        selected_provider, selection = self._select_provider(
            phase=phase,
            user_message=user_message,
            current_action=current_action,
            tool_history=tool_history,
            prefer_tier=forced_prefer_tier,
            thinking_override=thinking_override,
            routing_overrides=routing_overrides,
        )
        resolved_thinking = thinking_override
        if resolved_thinking is None and selection.tier == "reasoner" and user_message:
            resolved_thinking = "low"
        self._last_call_meta = {
            "phase": selection.phase,
            "tier": selection.tier,
            "model_ref": selection.model_ref,
            "thinking": resolved_thinking or selection.thinking,
            "skills": self._last_call_meta.get("skills") or "none",
        }
        raw: str | None = None
        for _attempt in range(2):
            try:
                raw = await selected_provider.chat(messages, thinking_override=resolved_thinking)
                self._mark_model_success(selection.model_ref)
                self._track_token_usage(selected_provider)
                break
            except Exception as exc:
                _err = str(exc) or repr(exc)
                self._mark_model_failure(selection.model_ref, _err)
                if _attempt == 0:
                    _fallback_tier = self._fallback_tiers(selection.tier)[0]
                    fb_provider, fb_selection = self._select_provider(
                        phase=phase,
                        user_message=user_message,
                        current_action=current_action,
                        tool_history=tool_history,
                        prefer_tier="reasoner" if reply_only else _fallback_tier,
                        thinking_override=resolved_thinking,
                    )
                    if fb_selection.model_ref != selection.model_ref:
                        _log.warning(
                            "[judgment.continue] LLM 调用失败，切换模型重试: from=%s(%s) to=%s(%s) err=%s",
                            selection.model_ref,
                            selection.tier,
                            fb_selection.model_ref,
                            fb_selection.tier,
                            _err,
                        )
                        selected_provider, selection = fb_provider, fb_selection
                        continue
                    _log.warning("[judgment.continue] LLM 调用失败，1s 后重试: %s", _err)
                    await asyncio.sleep(1.0)
                else:
                    _log.warning("[judgment.continue] LLM 调用失败: %s", _err)
                    return JudgmentOutput.wait(reason=f"[inner-loop] LLM 不可用: {exc!r}")

        if raw is None:
            return JudgmentOutput.wait(reason="[inner-loop] LLM returned None")

        output = JudgmentOutput.from_llm(raw)
        if output.decision not in ("act", "pause", "wait"):
            output = JudgmentOutput.wait(reason=f"无效 decision: {output.decision!r}")
        if output.decision == "act" and not output.chosen_action_id:
            output = JudgmentOutput.wait(reason="act 决策缺少 chosen_action_id")
        if reply_only:
            if not output.reply_to_user.strip():
                output = JudgmentOutput.wait(reason="[reply-only] reply_to_user 不能为空")
            else:
                output = JudgmentOutput(
                    decision=output.decision if output.decision in {"pause", "wait"} else "pause",
                    chosen_action_id="",
                    params={},
                    rationale=output.rationale,
                    reflection=output.reflection,
                    reply_to_user=output.reply_to_user,
                    next_step=output.next_step,
                    model_strategy=dict(output.model_strategy or {}),
                )

        _applied = ",".join(output.applied_skills) if output.applied_skills else "none"
        _log.info(
            "[judgment.continue] round=%d phase=%s tier=%s model=%s thinking=%s applied_skills=%s decision=%s action=%s",
            len(tool_history), selection.phase, selection.tier, selection.model_ref,
            self._last_call_meta["thinking"], _applied,
            output.decision, output.chosen_action_id,
        )
        return output

    async def _repair_output(self, context_text: str, raw: str) -> "JudgmentOutput | None":
        """对被截断或损坏的 JSON 做一次二次修复。"""
        from provider.base import Message

        repair_messages = [
            Message(
                role="system",
                content=(
                    "你是一个严格的 JSON 修复器。"
                    "只输出合法 JSON，不要解释，不要使用 markdown 代码块。"
                    "必须遵循这个 schema: {decision, chosen_action_id, params, rationale, reflection, reply_to_user, next_step, model_strategy}."
                    "如果原输出被截断，请根据上下文重新生成一个完整、简短的 JSON。"
                    "如果 broken_output 是裸代码（bash/python 脚本等），将代码原文放入 reply_to_user 字段，decision 设为 pause，rationale 说明代码已封装。"
                ),
            ),
            Message(
                role="user",
                content=(
                    "下面是原始判断上下文和一段损坏/截断的模型输出，请修复为合法 JSON。\n\n"
                    f"[context]\n{context_text}\n\n"
                    f"[broken_output]\n{raw[:4000]}\n\n"
                    "只返回 JSON，不要用 markdown 代码块包裹。"
                ),
            ),
        ]

        try:
            repaired_raw = await self._provider.chat(
                repair_messages,
                temperature=0.0,
            )
        except Exception as exc:
            _log.warning("[judgment] repair request failed: %s", exc)
            return None

        repaired = JudgmentOutput.from_llm(repaired_raw)
        if repaired.rationale.startswith("LLM 输出解析失败"):
            _log.warning("[judgment] repair failed: %s", repaired.rationale)
            return None

        _log.info("[judgment] malformed JSON repaired via second pass")
        return repaired

    def _simulate_safe_output(
        self,
        failure_count: int,
        signals: "JudgmentSignals | None",
        hard_boundaries: list[str],
        reason: str = "",
    ) -> JudgmentOutput:
        """LLM 不可用时的确定性回退。
        行为原则：posture > wait。"""
        if signals:
            if signals.posture in ("pause", "narrow"):
                return JudgmentOutput.wait(reason=f"[fallback] posture={signals.posture}, LLM 不可用: {reason}")
        return JudgmentOutput.wait(reason=f"[fallback] LLM 不可用: {reason}")

    async def _assemble_context(
        self,
        percept: "Percept",
        wm: "WorkingMemory",
        task_store: "TaskStore",
        episodic: "EpisodicMemory",
        semantic: "SemanticMemory",
        emotion: "EmotionState",
        user_message: str = "",
        ethos_state: "EthosState | None" = None,
        judgment_signals: "JudgmentSignals | None" = None,
        hard_boundaries: "list[str] | None" = None,
        perception_replay: "PerceptionReplaySummary | None" = None,
        cognitive_signals: "CognitiveSignals | None" = None,
        phase: str = "initial",
        current_action: str = "",
        tool_history: list[dict[str, Any]] | None = None,
        effective_thinking: str | None = None,
        routing_overrides: "dict[str, str] | None" = None,
    ) -> str:
        """将运行时状态填入 judgment 模板。"""
        task = await task_store.get_active()

        if task:
            failures = await task_store.list_failures_for_task(
                str(task.id), self._cfg.memory.failure_limit
            )
        else:
            failures = await task_store.list_failures(self._cfg.memory.failure_limit)

        task_id_str = str(task.id) if task else None
        _el = asyncio.get_running_loop()
        # episodic/semantic 使用同步 sqlite3，需经 executor 層驱动，避免阻塞事件循环
        episodic_text = await _el.run_in_executor(
            None, episodic.load_for_context, task_id_str, self._cfg.memory.episodic_max_chars
        )
        recent_runs = await task_store.list_runs(task_id=task.id, limit=6) if task else []
        waiting_tasks = await task_store.list_tasks(status="waiting", limit=5)
        durable_failure_snapshot = await _load_durable_failure_snapshot(task_store)
        context_facts = await _load_context_facts_snapshot(task_store, task)
        probes = await self._probe_manager.list_probes() if self._probe_manager else []

        search_query = (task.goal or task.title) if task else user_message
        episodic_search = (
            await _el.run_in_executor(None, episodic.search, search_query, 16000)
            if search_query else ""
        )
        if episodic_search and episodic_search not in episodic_text:
            episodic_text = episodic_text + "\n\n[跨任务检索命中]\n" + episodic_search
        _log.info("[context] episodic search=%r cross_task_hit=%s",
                  (search_query or "")[:50], bool(episodic_search))

        resolved_entities = await self._ref_resolver.resolve(user_message, semantic, episodic) if user_message else []
        entity_section = self._ref_resolver.format_section(resolved_entities)

        anchors: list[str] = []
        if task:
            anchors.append(task.goal or task.title)
            # 来源身份锚：wechat/chat 用户 ID 作为额外检索锚，让已存储的用户身份节点可被命中
            task_source = str(getattr(task, "source", "") or "")
            if task_source and task_source not in anchors:
                anchors.append(task_source)
        if user_message and user_message not in anchors:
            anchors.append(user_message[:100])
        if failures:
            anchors.append(failures[0].kind)
        emotion_label = _emotion_label(emotion, self._cfg)
        anchors.append(emotion_label)
        memories = await _el.run_in_executor(
            None, semantic.retrieve_multi_anchor, anchors, self._cfg.memory.semantic_top_k
        )
        _log.info("[context] semantic hits=%d anchors=%r",
                  len(memories), [a[:40] for a in anchors[:3]])

        axioms_val, _ = await task_store.get_fact("soul:hard_axioms")
        ethos_val, _ = await task_store.get_fact("soul:ethos_baseline")
        soul_section = _fmt_soul(axioms_val, ethos_val)

        _wm_items = wm.get_top(15)
        all_skills = self._skills.all_skills()
        skills = self._skills.match_for_context()
        self._last_selected_skills = list(skills)
        if skills:
            _log.debug("[skill] 本轮注入 %d 个技能", len(skills))
        else:
            _log.info("[skill] 本轮无可用技能")

        ctx = {
            "task_section": _fmt_task(task),
            "task_facts_section": _fmt_context_facts(context_facts),
            "waiting_tasks_section": _fmt_waiting_tasks(waiting_tasks),
            "recent_runs_section": _fmt_recent_runs(recent_runs),
            "emotion_valence": f"{emotion.valence:.2f}",
            "emotion_arousal": f"{emotion.arousal:.2f}",
            "emotion_dominant": emotion.dominant or "（未确定）",
            "emotion_regulation": f"{emotion.regulation.strategy}（{emotion.regulation.reason}）" if emotion.regulation.reason else emotion.regulation.strategy,
            "wm_section": _fmt_wm(_wm_items, wm_count=len(wm), wm_capacity=wm._capacity,
                                   wm_tokens=wm.total_tokens, wm_token_budget=wm._token_budget),
            "failures_section": _fmt_failures(failures),
            "durable_failure_section": _fmt_durable_failures(durable_failure_snapshot),
            "episodic_section": episodic_text or "（暂无情节记忆）",
            "entity_section": entity_section,
            "memories_section": _fmt_memories(memories),
            "memory_system_section": _fmt_memory_system(
                runtime_db=str(self._cfg.db_path),
                memory_dir=str(self._cfg.memory_dir),
                workspace_dir=str(self._cfg.workspace_dir),
                semantic=semantic,
            ),
            "soul_section": soul_section,
            "tools_section": _fmt_tools(self._registry.list_manifests()),
            "shell_capabilities_section": _fmt_shell_capabilities(),
            "perception_section": _fmt_percept(percept),
            "ethos_section": _fmt_ethos(ethos_state),
            "signals_section": _fmt_judgment_signals(judgment_signals),
            "hard_boundaries_section": _fmt_hard_boundaries(hard_boundaries),
            "perception_replay_section": _fmt_perception_replay(perception_replay),
            "skills_catalog_section": _fmt_skill_catalog(all_skills),
            "primary_skill_section": "",
            "skills_section": _fmt_skills(skills),
            "cognitive_signals_section": _fmt_cognitive_signals(cognitive_signals),
            "probe_sensors_section": _fmt_probe_sensors(probes),
            "blind_spot_section": _fmt_blind_spots(probes, self.self_model.total_tokens),
            "self_model_section": fmt_self_model(self.self_model),
            "team_view": _build_team_view_from_cfg(self._cfg),
            "model_routing_section": self._build_model_routing_section(
                phase=phase,
                user_message=user_message,
                current_action=current_action,
                tool_history=tool_history,
                effective_thinking=effective_thinking or self._cfg.thinking,
                routing_overrides=routing_overrides,
            ),
            "current_time_section": _fmt_current_time(),
            "user_message": user_message or "",
        }
        # STM 对话缓冲：源自情节记忆（narrative 表 role=user/assistant_reply）
        # 不走原始 chat_messages 表，记忆系统本身就是正确的历史源。
        recent_turns = await _el.run_in_executor(None, episodic.get_recent_turns, task_id_str, 3)
        ctx["chat_history_section"] = _fmt_chat_history(recent_turns)
        _validate_context_schema(ctx)
        ctx = apply_context_budget(
            ctx,
            self._cfg.judgment_input_token_budget(),
            skill_min_tokens=self._cfg.thresholds.skill_min_budget_tokens,
        )
        # 注入上下文预算信息到自我模型，供后续判断感知上下文压力
        budget = self._cfg.judgment_input_token_budget()
        if budget:
            used = sum(len(v) for v in ctx.values())
            self.self_model.context_budget = f"{budget // 1000}K" if budget >= 1000 else str(budget)
            self.self_model.context_pressure = min(1.0, used / max(budget, 1))
        return _fill_template(self._judgment_template, ctx)
