"""core/judgment/runtime.py — 判断层（JudgmentLayer 核心类）。

职责：
1. 组装 bundle（运行时状态 → 结构化 context）
2. 填入 prompts/judgment.md 模板（{{variable}} 语法）
3. 调用 LLM provider
4. 解析 JSON 输出 → JudgmentOutput

数据模型 / 工具常量 / 前置改写函数 → output.py
解耦原则：此模块不知道工具如何执行，只负责"决定做什么"。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from provider.catalog import lookup_model
from core.execution import action_key_param
from core.self_model import SelfModel, fmt_self_model
from tools.registry import tool_has_capability
from .output import (
    JudgmentOutput,
    ModelHealth,
    ModelSelection,
    _rewrite_task_ask_to_evidence,
    _rewrite_complex_act_to_task_plan,
    _structured_tool_history_window,
    _build_team_view_from_cfg,
    tool_tier,
    tool_tier_mapping,
)
from .context import (
    _clear_context_cache,
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
    _fmt_config_snapshot,
    _fmt_primary_skill,
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


# ── 认知基底（传入 decide/assemble_context 的感知+记忆层快照） ────────────────

@dataclass(slots=True)
class CognitionFrame:
    """6 个认知基底字段的轻量容器，兼容旧调用点。"""

    percept: "Percept"
    wm: "WorkingMemory"
    task_store: "TaskStore"
    episodic: "EpisodicMemory"
    semantic: "SemanticMemory"
    emotion: "EmotionState"


# ── 判断层 ─────────────────────────────────────────────────────────────────────

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
        self._ref_resolver = ReferenceResolver(
            provider=provider,
            thresholds=cfg.thresholds,
            reason_temperature=cfg.temperature,
        )
        # 自我模型追踪：持久化运行态与任务连续性
        self.self_model = SelfModel()
        # 分层路由 providers：{"simple": <provider>, "complex": <provider>}
        # 由 loop.open() 在 bootstrap 后注入，未配置时为空字典
        self._routing_providers: dict[str, "Provider"] = {}
        # 内层工具循环用：缓存上一次 decide() 组装的完整上下文，由 decide_continue() 复用
        self._last_context_text: str = ""
        # 上下文缓存：key=(section_name, tick)，value=计算好的文本片段
        self._context_cache: dict[str, str] = {}
        # 探针系统引用：由 CognitionLoop.__init__ 在创建 ProbeManager 后注入
        self._probe_manager: Any = None
        # 最近一次真实 LLM 调用元数据（供 loop 日志输出实际 model/tier/thinking）
        self._last_call_meta: dict[str, Any] = {
            "phase": "",
            "tier": "default",
            "model_ref": cfg.model,
            "thinking": cfg.thinking,
            "skills": "",
        }
        self._last_selected_skills: list[Skill] = []
        # 上轮 LLM 实际应用的技能名（用于下轮 match_for_context 优先注入）
        self._last_applied_skill_names: list[str] = []
        # 每个模型最近一次调用错误（用于注入 model routing truth）
        self._provider_errors: dict[str, str] = {}
        # 每个模型的健康状态（429/400/timeout 触发冷却窗口，避免短时间重复打爆同一 provider）
        self._model_health: dict[str, ModelHealth] = {}
        # 运行时临时 provider 缓存：routing_overrides 指定的临时 model 按需创建并缓存
        self._override_providers: dict[str, "Provider"] = {}

    def reload_skills(self) -> None:
        from core.skill import SkillRegistry

        skills_dir = self._cfg.workspace_dir / "skills"
        self._skills = SkillRegistry(skills_dir=skills_dir)
        _log.info("[judgment] 已从 %s 重新加载 skills", skills_dir)

    def _track_token_usage(self, provider: "Provider") -> None:
        """从 provider 读取 last_usage 并累积到 self_model。"""
        usage = getattr(provider, "last_usage", None)
        if isinstance(usage, dict):
            self.self_model.record_token_usage(
                prompt=usage.get("prompt_tokens", 0),
                completion=usage.get("completion_tokens", 0),
            )

    def _coerce_frame_args(
        self,
        frame_or_percept: "CognitionFrame | Percept",
        wm: "WorkingMemory | None",
        task_store: "TaskStore | None",
        episodic: "EpisodicMemory | None",
        semantic: "SemanticMemory | None",
        emotion: "EmotionState | None",
    ) -> tuple[
        "Percept",
        "WorkingMemory",
        "TaskStore",
        "EpisodicMemory",
        "SemanticMemory",
        "EmotionState",
    ]:
        if isinstance(frame_or_percept, CognitionFrame):
            return (
                frame_or_percept.percept,
                frame_or_percept.wm,
                frame_or_percept.task_store,
                frame_or_percept.episodic,
                frame_or_percept.semantic,
                frame_or_percept.emotion,
            )
        if None in (wm, task_store, episodic, semantic, emotion):
            raise TypeError("decide/_assemble_context 缺少认知基底参数")
        return (
            frame_or_percept,
            cast("WorkingMemory", wm),
            cast("TaskStore", task_store),
            cast("EpisodicMemory", episodic),
            cast("SemanticMemory", semantic),
            cast("EmotionState", emotion),
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
    def last_call_meta(self) -> dict[str, Any]:
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
        if " 402 " in f" {text} " or "payment required" in text or "insufficient balance" in text:
            return "402"
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
        if code == "402":
            # 余额耗尽 — 不会自动恢复，本次会话屏蔽 24h
            return 86400.0
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
        # _routing_providers 按 tier 存储，用完整 model_ref 匹配（不能用 p._model 短 ID，会永远不等）
        for p in self._routing_providers.values():
            p_ref = (
                getattr(p, "model_ref", None)
                or getattr(p, "_model_ref", None)
                or getattr(p, "_model", None)
            )
            if p_ref == model_ref:
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
            return "reasoner"
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

    def _set_last_call_meta(
        self,
        selection: ModelSelection,
        *,
        thinking_override: str | None,
        skills: str,
        primary_skill_name: str | None = None,
        primary_skill_guidance: bool | None = None,
    ) -> None:
        meta: dict[str, Any] = {
            "phase": selection.phase,
            "tier": selection.tier,
            "model_ref": selection.model_ref,
            "thinking": thinking_override or selection.thinking,
            "skills": skills,
        }
        if primary_skill_name is not None or primary_skill_guidance is not None:
            meta["primary_skill"] = primary_skill_name
            meta["primary_skill_guidance"] = bool(primary_skill_guidance)
        self._last_call_meta = meta

    async def _chat_with_retry(
        self,
        *,
        selected_provider: "Provider",
        selection: ModelSelection,
        messages: list[Any],
        phase: str,
        user_message: str,
        thinking_override: str | None,
        routing_overrides: dict[str, str] | None,
        log_prefix: str,
        current_action: str = "",
        tool_history: list[dict[str, Any]] | None = None,
        fallback_prefer_tier: str | None = None,
        skills: str = "none",
        primary_skill_name: str | None = None,
        primary_skill_guidance: bool | None = None,
    ) -> tuple[str | None, ModelSelection, Exception | None]:
        raw: str | None = None
        last_error: Exception | None = None
        for _attempt in range(2):
            self._set_last_call_meta(
                selection,
                thinking_override=thinking_override,
                skills=skills,
                primary_skill_name=primary_skill_name,
                primary_skill_guidance=primary_skill_guidance,
            )
            try:
                raw = await selected_provider.chat(messages, thinking_override=thinking_override)
                self._mark_model_success(selection.model_ref)
                self._track_token_usage(selected_provider)
                return raw, selection, None
            except Exception as exc:
                last_error = exc
                _err = str(exc) or repr(exc)
                self._mark_model_failure(selection.model_ref, _err)
                if _attempt == 0:
                    _fallback_tier = fallback_prefer_tier or self._fallback_tiers(selection.tier)[0]
                    fb_provider, fb_selection = self._select_provider(
                        phase=phase,
                        user_message=user_message,
                        current_action=current_action,
                        tool_history=tool_history,
                        prefer_tier=_fallback_tier,
                        thinking_override=thinking_override,
                        routing_overrides=routing_overrides,
                    )
                    if fb_selection.model_ref != selection.model_ref:
                        _log.warning(
                            "%s LLM 调用失败，切换模型重试: from=%s(%s) to=%s(%s) err=%s",
                            log_prefix,
                            selection.model_ref,
                            selection.tier,
                            fb_selection.model_ref,
                            fb_selection.tier,
                            _err,
                        )
                        selected_provider, selection = fb_provider, fb_selection
                        continue
                    _log.warning("%s LLM 调用失败，1s 后重试: %s", log_prefix, _err)
                    await asyncio.sleep(1.0)
                    continue
                _log.warning("%s LLM 调用失败: %s", log_prefix, _err)
        return raw, selection, last_error

    def _record_applied_skills(self, output: JudgmentOutput) -> str:
        applied = ",".join(output.applied_skills) if output.applied_skills else "none"
        if output.applied_skills:
            self._last_applied_skill_names = list(output.applied_skills)
        return applied

    def _effective_registry(self, registry: "Any | None" = None) -> Any:
        return registry or self._registry

    def _counts_as_exploration_budget(self, tool_name: str, registry: "Any | None" = None) -> bool:
        effective_registry = self._effective_registry(registry)
        return any(
            tool_has_capability(effective_registry, tool_name, capability)
            for capability in ("ask_evidence", "completion_info_only", "completion_verify")
        )

    def _build_messages(self, user_content: str) -> list[Any]:
        from provider.base import Message

        system_content = (
            self._identity_prefix + "\n\n" + self._system_prompt
            if self._identity_prefix
            else self._system_prompt
        )
        return [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]

    def _build_continue_context(
        self,
        tool_history: list[dict[str, Any]],
        *,
        user_message: str,
        reply_only: bool,
        wm_delta: list[dict[str, Any]] | None,
    ) -> str:
        history_json_block, history_block = _structured_tool_history_window(tool_history)
        wm_delta_block = ""
        if wm_delta:
            delta_lines = [
                f"- [{item.get('kind', '')}|p={item.get('priority', 0):.2f}] {item.get('content', '')}"
                for item in wm_delta
            ]
            wm_delta_block = "## 本轮新增工作记忆（WM 更新，初始上下文之后）\n" + "\n".join(delta_lines) + "\n\n"
        if reply_only:
            return (
                f"{self._last_context_text}\n\n"
                "---\n"
                f"{wm_delta_block}"
                "## 结构化最近工具结果(JSON)\n"
                f"{history_json_block}\n\n"
                "## 本轮已执行工具历史\n"
                f"{history_block}\n\n"
                "你现在处于最终回复阶段。禁止再调用任何工具。"
                "请只基于已有证据生成对用户的最终 reply_to_user。"
                "decision 只能是 pause 或 wait，chosen_action_id 必须留空。"
            )

        hint = "用户正在等待回复，尽快在本轮设置 reply_to_user 字段。" if user_message else ""
        return (
            f"{self._last_context_text}\n\n"
            "---\n"
            f"{wm_delta_block}"
            "## 结构化最近工具结果(JSON)\n"
            f"{history_json_block}\n\n"
            "## 本轮已执行工具历史\n"
            f"{history_block}\n\n"
            "优先依据结构化结果判断当前状态，不要只凭模糊回忆续写。\n\n"
            f"请根据以上结果继续执行下一个必要工具，或生成最终回复（reply_to_user 非空）。{hint}"
        )

    def _coerce_reply_only_output(self, output: JudgmentOutput) -> JudgmentOutput:
        if not output.reply_to_user.strip():
            return JudgmentOutput.wait(reason="[reply-only] reply_to_user 不能为空")
        return JudgmentOutput(
            decision=output.decision if output.decision in {"pause", "wait"} else "pause",
            chosen_action_id="",
            params={},
            rationale=output.rationale,
            reflection=output.reflection,
            reply_to_user=output.reply_to_user,
            next_step=output.next_step,
            model_strategy=dict(output.model_strategy or {}),
        )

    def _finalize_continue_output(
        self,
        output: JudgmentOutput,
        *,
        reply_only: bool,
        user_message: str,
        active_task: Any | None,
        tool_history: list[dict[str, Any]],
        selection: ModelSelection,
    ) -> JudgmentOutput:
        if reply_only:
            output = self._coerce_reply_only_output(output)

        applied = self._record_applied_skills(output)

        _log.info(
            "[judgment.continue] round=%d phase=%s tier=%s model=%s thinking=%s applied_skills=%s decision=%s action=%s",
            len(tool_history), selection.phase, selection.tier, selection.model_ref,
            self._last_call_meta["thinking"], applied,
            output.decision, output.action_label(),
        )
        return output

    def _build_model_routing_section(
        self,
        *,
        phase: str,
        user_message: str,
        current_action: str,
        tool_history: list[dict[str, Any]] | None,
        effective_thinking: str,
        routing_overrides: dict[str, str] | None = None,
        registry: "Any | None" = None,
    ) -> str:
        effective_registry = self._effective_registry(registry)
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
            def _trailing_repeat_count(matcher: Any) -> int:
                count = 0
                for item in reversed(tool_history):
                    if not matcher(item):
                        break
                    count += 1
                return count

            task_explore_count = sum(
                1
                for item in tool_history
                if self._counts_as_exploration_budget(str(item.get("tool") or ""), effective_registry)
            )
            if len(tool_history) >= 2:
                _last_tool = str(tool_history[-1].get("tool", ""))
                _last_action_sig = f"{_last_tool}|{action_key_param(tool_history[-1].get('params') or {})}"
                repeat_action_count = _trailing_repeat_count(
                    lambda item: (
                        f"{str(item.get('tool', ''))}|{action_key_param(item.get('params') or {})}"
                        == _last_action_sig
                    )
                )
                if _last_tool == "file.read":
                    _last_path = json.dumps(tool_history[-1].get("params", {}), ensure_ascii=False)
                    repeat_read_count = _trailing_repeat_count(
                        lambda item: str(item.get("tool", "")) == "file.read"
                        and json.dumps(item.get("params", {}), ensure_ascii=False) == _last_path
                    )

        ask_evidence_hits = sum(
            1 for item in (tool_history or [])
            if tool_has_capability(effective_registry, str(item.get("tool") or ""), "ask_evidence")
            and str(item.get("result") or "").strip()
            and not str(item.get("result") or "").startswith("ERROR[")
        )

        posture = (
            "respond"
            if user_message
            else (
                "converge"
                if task_explore_count >= self._cfg.thresholds.task_explore_converge_after
                else "conserve"
            )
        )
        implicit_next_phase_default = None

        def _fmt_duration_ms(value: float) -> str:
            ms = float(value)
            if ms >= 1000:
                return f"{ms / 1000.0:g}s"
            return f"{ms:g}ms"

        with_task_bounds = self._cfg.loop.idle_with_task_bounds
        no_task_bounds = self._cfg.loop.idle_no_task_bounds
        with_task_bounds_text = f"{_fmt_duration_ms(with_task_bounds[0])}-{_fmt_duration_ms(with_task_bounds[1])}"
        no_task_bounds_text = f"{_fmt_duration_ms(no_task_bounds[0])}-{_fmt_duration_ms(no_task_bounds[1])}"
        default_gap_text = (
            f"有任务 {_fmt_duration_ms(self._cfg.loop.active_idle_gap)}，"
            f"无任务 {_fmt_duration_ms(self._cfg.loop.max_idle_gap)}"
        )

        capability_mapping: dict[str, list[str]] = {}
        current_action_caps: list[str] = []
        task_plan_calls_this_tick = sum(
            1 for item in (tool_history or []) if str(item.get("tool") or "") == "task.plan"
        )
        continue_task_plan_max = self._cfg.thresholds.continue_task_plan_max_per_tick
        tool_history_compact_threshold = self._cfg.thresholds.continue_tool_history_compact_threshold
        tool_history_keep_last = self._cfg.thresholds.continue_tool_history_keep_last
        tool_history_count = len(tool_history or [])
        for manifest in effective_registry.list_manifests():
            for cap in manifest.capabilities:
                capability_mapping.setdefault(cap, []).append(manifest.name)
            if manifest.name == current_action:
                current_action_caps = sorted(list(manifest.capabilities))
        payload = {
            "active_overrides": routing_overrides or {},
            "tool_tier_mapping": tool_tier_mapping(effective_registry),
            "tool_capability_mapping": {k: sorted(v) for k, v in capability_mapping.items()},
            "current_action_capabilities": current_action_caps,
            "implicit_next_phase_default": implicit_next_phase_default,
            "continue_phase_policy": {
                "task_plan_calls_this_tick": task_plan_calls_this_tick,
                "task_plan_max_per_tick": continue_task_plan_max,
                "task_plan_blocked_next": task_plan_calls_this_tick >= continue_task_plan_max,
                "tool_history_count": tool_history_count,
                "tool_history_compact_threshold": tool_history_compact_threshold,
                "tool_history_keep_last": tool_history_keep_last,
                "tool_history_will_compact_next": (
                    tool_history_count >= tool_history_compact_threshold
                    and tool_history_count > tool_history_keep_last
                ),
            },
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
                "• implicit_next_phase_default：兼容字段。当前 runtime 不再根据上个工具自动套用下轮 tier；"
                "若你希望下轮走 reader / repair，必须由你显式设置 next_phase_tier。该字段通常为 null。\n"
                "• continue_phase_policy：runtime 暴露的 tick 内计划限制真相。若 task_plan_blocked_next=true，"
                "本 tick 再输出 task.plan 会被强制打断；应直接执行计划内工具。"
                "若 tool_history_will_compact_next=true，下一轮会把早期工具记录折叠成 [compacted] 摘要，应尽量在压缩前完成总结或切换到执行。\n"
                "• next_idle_gap_secs / next_idle_gap_ms：【必须设置其中之一！】你的生命节奏控制器。"
                "next_idle_gap_secs 单位秒（小数可用，如 0.5 = 500ms），next_idle_gap_ms 单位毫秒（整数，如 500 = 500ms）；两者同时设置时 ms 优先。"
                f"范围由 idle_with_task_bounds / idle_no_task_bounds 决定（当前有任务时 {with_task_bounds_text}，无任务时 {no_task_bounds_text}）。"
                "你必须根据当前上下文主动选择一个合理值，不要依赖默认："
                "已发起shell预计30s出结果 → next_idle_gap_secs=35；刚回复完用户等下一步 → next_idle_gap_secs=120；"
                "任务推进中需快速追踪 → next_idle_gap_ms=500；实时等待短命令结束 → next_idle_gap_ms=200。"
                f"不设置此字段则用当前 loop 默认备用值（{default_gap_text}）。控制权在你手里。\n"
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
                "ask_evidence_budget": self._cfg.thresholds.ask_evidence_budget,
                "task_explore_converge_after": self._cfg.thresholds.task_explore_converge_after,
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
        # 主 provider 信息
        payload["primary_provider"] = {"model": self._cfg.model}
        # 实体消解模块健康状态
        if hasattr(self, "_ref_resolver") and self._ref_resolver is not None:
            _rr = self._ref_resolver
            payload["reference_resolution"] = {
                "llm_available": _rr.llm_available,
                "last_error": _rr.last_llm_error,
                "last_error_code": _rr.last_llm_error_code,
            }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    async def decide(
        self,
        frame_or_percept: "CognitionFrame | Percept",
        wm: "WorkingMemory | None" = None,
        task_store: "TaskStore | None" = None,
        episodic: "EpisodicMemory | None" = None,
        semantic: "SemanticMemory | None" = None,
        emotion: "EmotionState | None" = None,
        active_task: Any | None = None,
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
        registry_override: "Any | None" = None,
    ) -> JudgmentOutput:
        """组装上下文，调用 LLM，返回决策。
        
        thinking_override: 覆盖 cfg.thinking（如 chat 模式用 "low" 加速首轮判断）。
        routing_overrides: 临时覆盖 tier→model 映射（由 loop.py 从 model_strategy 读取）。
        registry_override: 临时覆盖本轮可见工具集（如子灵受限工具视图）。
        """
        percept, wm, task_store, episodic, semantic, emotion = self._coerce_frame_args(
            frame_or_percept,
            wm,
            task_store,
            episodic,
            semantic,
            emotion,
        )
        try:
            # per-tick 清空静态缓存（静态 section 仅在本 tick 复用）
            self._context_cache.clear()
            _clear_context_cache()
            context_text = await self._assemble_context(
                percept, wm, task_store, episodic, semantic, emotion,
                active_task=active_task,
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
                registry_override=registry_override,
            )
        except Exception as _ctx_exc:
            _log.exception("[judgment] _assemble_context() 异常，返回 wait 兜底: %s", _ctx_exc)
            return self._simulate_safe_output(
                failure_count=0,
                signals=judgment_signals,
                hard_boundaries=hard_boundaries or [],
                reason=f"上下文组装异常: {_ctx_exc}",
            )
        # 缓存给内层工具循环的续判请求用
        self._last_context_text = context_text
        messages = self._build_messages(context_text)

        selected_provider, selection = self._select_provider(
            phase=phase,
            user_message=user_message,
            prefer_tier=prefer_tier,
            thinking_override=thinking_override,
            routing_overrides=routing_overrides,
        )
        _primary = self._last_selected_skills[0] if self._last_selected_skills else None
        raw, selection, llm_error = await self._chat_with_retry(
            selected_provider=selected_provider,
            selection=selection,
            messages=messages,
            phase=phase,
            user_message=user_message,
            thinking_override=thinking_override,
            routing_overrides=routing_overrides,
            log_prefix="[judgment]",
            skills=self._skills_for_log(self._last_selected_skills),
            primary_skill_name=_primary.name if _primary else None,
            primary_skill_guidance=bool(_primary and getattr(_primary, "guidance", None)),
        )
        if raw is None:
            _err = str(llm_error) or repr(llm_error) if llm_error is not None else "unknown error"
            return self._simulate_safe_output(
                failure_count=0,
                signals=judgment_signals,
                hard_boundaries=hard_boundaries or [],
                reason=_err,
            )

        output = JudgmentOutput.from_llm(raw)

        # 解析失败时尝试一次修复，避免因为截断/格式噪声直接进入空转
        output = await self._normalize_output(
            output,
            context_text=context_text,
            raw=raw,
            record_parse_failure=task_store.record_failure,
        )
        _applied = self._record_applied_skills(output)
        _log.info(
            "[judgment] phase=%s tier=%s model=%s thinking=%s applied_skills=%s decision=%s action=%s rationale=%s",
            selection.phase, selection.tier, selection.model_ref, selection.thinking,
            _applied,
            output.decision, output.action_label(), output.rationale or "",
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
        wm_delta: "list[dict[str, Any]] | None" = None,
    ) -> JudgmentOutput:
        """内层工具循环的续判请求。

        不重践 perception 链路，直接在上次 decide() 缓存的全量上下文后面追加工具历史续判。
        每次 HTTP 请求与普通请求相同，但输入 token 显著减少（不重发全量感知层）。

        Args:
            tool_history: [{"tool": str, "params": dict, "result": str}, ...]
            user_message:  原始用户消息（不再次向 LLM 重复，仅用于选择 provider tier）
        """
        if not self._last_context_text:
            return JudgmentOutput.wait(reason="[inner-loop] no cached context for continuation")
        continuation_context = self._build_continue_context(
            tool_history,
            user_message=user_message,
            reply_only=reply_only,
            wm_delta=wm_delta,
        )
        messages = self._build_messages(continuation_context)

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
        raw, selection, llm_error = await self._chat_with_retry(
            selected_provider=selected_provider,
            selection=selection,
            messages=messages,
            phase=phase,
            user_message=user_message,
            current_action=current_action,
            tool_history=tool_history,
            thinking_override=resolved_thinking,
            routing_overrides=routing_overrides,
            fallback_prefer_tier="reasoner" if reply_only else None,
            log_prefix="[judgment.continue]",
            skills=self._last_call_meta.get("skills") or "none",
        )
        if raw is None:
            if llm_error is not None:
                return JudgmentOutput.wait(reason=f"[inner-loop] LLM 不可用: {llm_error!r}")
            return JudgmentOutput.wait(reason="[inner-loop] LLM returned None")

        output = JudgmentOutput.from_llm(raw)
        output = await self._normalize_output(
            output,
            context_text=continuation_context,
            raw=raw,
        )
        return self._finalize_continue_output(
            output,
            reply_only=reply_only,
            user_message=user_message,
            active_task=active_task,
            tool_history=tool_history,
            selection=selection,
        )

    async def _repair_output(self, context_text: str, raw: str) -> "JudgmentOutput | None":
        """对被截断或损坏的 JSON 做一次二次修复。"""
        from provider.base import Message

        repair_messages = [
            Message(
                role="system",
                content=(
                    "你是一个严格的 JSON 修复器。"
                    "只输出合法 JSON，不要解释，不要使用 markdown 代码块。"
                    "必须遵循这个 schema: {decision, chosen_action_id, params, parallel_actions, delegate_tasks, rationale, reflection, reply_to_user, next_step, model_strategy}."  # noqa: E501
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

    async def _normalize_output(
        self,
        output: JudgmentOutput,
        *,
        context_text: str,
        raw: str,
        record_parse_failure: Any | None = None,
    ) -> JudgmentOutput:
        if output.rationale.startswith("LLM 输出解析失败"):
            repaired = await self._repair_output(context_text, raw)
            if repaired is not None:
                output = repaired
            elif record_parse_failure is not None:
                await record_parse_failure("judgment_parse", output.rationale)

        if output.decision not in ("act", "pause", "wait"):
            return JudgmentOutput.wait(reason=f"无效 decision: {output.decision!r}")
        if output.decision == "act" and not output.chosen_action_id \
                and not output.parallel_actions and not output.delegate_tasks:
            return JudgmentOutput.wait(reason="act 决策缺少 chosen_action_id")
        return output

    async def _assemble_context(
        self,
        frame_or_percept: "CognitionFrame | Percept",
        wm: "WorkingMemory | None" = None,
        task_store: "TaskStore | None" = None,
        episodic: "EpisodicMemory | None" = None,
        semantic: "SemanticMemory | None" = None,
        emotion: "EmotionState | None" = None,
        active_task: Any | None = None,
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
        registry_override: "Any | None" = None,
    ) -> str:
        """将运行时状态填入 judgment 模板。"""
        percept, wm, task_store, episodic, semantic, emotion = self._coerce_frame_args(
            frame_or_percept,
            wm,
            task_store,
            episodic,
            semantic,
            emotion,
        )
        effective_registry = self._effective_registry(registry_override)
        task = active_task if active_task is not None else await task_store.get_active()

        task_id_str = str(task.id) if task else None
        _el = asyncio.get_running_loop()
        # episodic/semantic 使用同步 sqlite3，需经 executor 层驱动，避免阻塞事件循环。
        # 显式启动独立任务，既保留并行 IO，又避免把立即值混入 gather。
        episodic_text_future = _el.run_in_executor(
            None,
            episodic.load_for_context,
            task_id_str,
            self._cfg.memory.episodic_max_chars,
        )
        recent_runs_task = (
            asyncio.create_task(task_store.list_runs(task_id=task.id, limit=6))
            if task else None
        )
        waiting_tasks_task = asyncio.create_task(task_store.list_tasks(status="waiting", limit=5))
        durable_failure_task = asyncio.create_task(_load_durable_failure_snapshot(task_store))
        context_facts_task = asyncio.create_task(_load_context_facts_snapshot(
            task_store,
            task,
            exclude_prefixes=self._cfg.thresholds.fact_context_exclude_prefixes,
            task_limit=self._cfg.thresholds.fact_context_task_limit,
            global_limit=self._cfg.thresholds.fact_context_global_limit,
            recent_scan_multiplier=self._cfg.thresholds.fact_context_recent_scan_multiplier,
            recent_scan_min=self._cfg.thresholds.fact_context_recent_scan_min,
        ))
        probes_task = (
            asyncio.create_task(self._probe_manager.list_probes())
            if self._probe_manager else None
        )
        failures_task = asyncio.create_task(
            task_store.list_failures_for_task(str(task.id), self._cfg.memory.failure_limit)
            if task else task_store.list_failures(self._cfg.memory.failure_limit)
        )

        # 统一收口并发上下文抓取，避免前一路异常时后续任务异常/结果无人消费。
        parallel_fetches: list[tuple[str, Any]] = [
            ("episodic_text", episodic_text_future),
            ("waiting_tasks", waiting_tasks_task),
            ("durable_failure_snapshot", durable_failure_task),
            ("context_facts", context_facts_task),
            ("failures", failures_task),
        ]
        if recent_runs_task is not None:
            parallel_fetches.append(("recent_runs", recent_runs_task))
        if probes_task is not None:
            parallel_fetches.append(("probes", probes_task))

        parallel_results = await asyncio.gather(
            *(awaitable for _, awaitable in parallel_fetches),
            return_exceptions=True,
        )
        parallel_data: dict[str, Any] = {}
        parallel_error: BaseException | None = None
        for (name, _), value in zip(parallel_fetches, parallel_results):
            if isinstance(value, BaseException):
                if parallel_error is None:
                    parallel_error = value
                continue
            parallel_data[name] = value
        if parallel_error is not None:
            raise parallel_error

        episodic_text = parallel_data["episodic_text"]
        recent_runs = parallel_data.get("recent_runs", [])
        waiting_tasks = parallel_data["waiting_tasks"]
        durable_failure_snapshot = parallel_data["durable_failure_snapshot"]
        context_facts = parallel_data["context_facts"]
        probes = parallel_data.get("probes", [])
        failures = parallel_data["failures"]

        search_query = user_message or (task.next_step or task.goal or task.title) if task else user_message
        episodic_search = (
            await _el.run_in_executor(None, episodic.search, search_query, 16000, task_id_str)
            if search_query else ""
        )
        if episodic_search and episodic_search not in episodic_text:
            episodic_text = episodic_text + "\n\n[跨任务检索命中]\n" + episodic_search
        _log.info("[context] episodic search=%r cross_task_hit=%s",
                  (search_query or "")[:50], bool(episodic_search))

        resolved_entities = await self._ref_resolver.resolve(user_message, semantic, episodic) if user_message else []
        entity_section = self._ref_resolver.format_section(resolved_entities)

        # 动态构建检索锚点：结合任务、用户原话与近期失败，提升语义记忆命中率
        anchors: list[str] = []
        if task:
            # 优先级：下一步 > 目标 > 标题
            primary_anchor = task.next_step or task.goal or task.title
            if primary_anchor:
                anchors.append(primary_anchor)
            # 身份锚：确保跨会话认人
            task_source = str(getattr(task, "source", "") or "")
            if task_source and task_source not in anchors:
                anchors.append(task_source)
        
        # 用户消息锚：截取关键片段
        if user_message and user_message not in anchors:
            anchors.append(user_message[:100])
        
        # 失败模式锚：若近期有失败，优先检索相关教训
        if failures:
            anchors.append(failures[0].kind)
        
        # 执行语义检索：使用动态锚点集合
        memories = await _el.run_in_executor(
            None, semantic.retrieve_multi_anchor, anchors, self._cfg.memory.semantic_top_k
        )
        _log.info("[context] semantic hits=%d anchors=%r",
                  len(memories), [a[:40] for a in anchors[:3]])

        axioms_fact, ethos_fact = await asyncio.gather(
            task_store.get_fact("soul:hard_axioms"),
            task_store.get_fact("soul:ethos_baseline"),
        )
        axioms_val, _ = axioms_fact
        ethos_val, _ = ethos_fact
        soul_section = _fmt_soul(
            axioms_val,
            ethos_val,
            json.dumps(self._cfg.soul.ethos.baseline, ensure_ascii=False, sort_keys=True),
            json.dumps(self._cfg.soul.hard_axioms, ensure_ascii=False),
        )

        _wm_items = wm.get_top(15)
        all_skills = self._skills.all_skills()
        skills: list[Skill] = []
        self._last_selected_skills = []
        _log.debug("[skill] catalog-only mode: runtime 不预选候选 skill，由模型自行 activation")

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
                max_concurrent_ticks=self._cfg.loop.max_concurrent_ticks,
                max_tick_queue=self._cfg.loop.max_tick_queue,
            ),
            "soul_section": soul_section,
            "tools_section": _fmt_tools(effective_registry.list_manifests()),
            "shell_capabilities_section": _fmt_shell_capabilities(),
            "perception_section": _fmt_percept(percept),
            "ethos_section": _fmt_ethos(ethos_state),
            "signals_section": _fmt_judgment_signals(judgment_signals),
            "hard_boundaries_section": _fmt_hard_boundaries(hard_boundaries),
            "perception_replay_section": _fmt_perception_replay(perception_replay),
            "skills_catalog_section": _fmt_skill_catalog(all_skills),
            "primary_skill_section": _fmt_primary_skill(skills[0] if skills else None),
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
                registry=effective_registry,
            ),
            "current_time_section": _fmt_current_time(),
            "config_section": _fmt_config_snapshot(self._cfg),
            "user_message": user_message or "",
        }
        # STM 对话缓冲：源自情节记忆（narrative 表 role=user/assistant_reply）
        # 不走原始 chat_messages 表，记忆系统本身就是正确的历史源。
        recent_turns = await _el.run_in_executor(
            None,
            episodic.get_recent_turns,
            task_id_str,
            self._cfg.thresholds.chat_history_turn_limit,
        )
        ctx["chat_history_section"] = _fmt_chat_history(
            recent_turns,
            max_chars=self._cfg.thresholds.chat_history_max_chars,
        )
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
