"""core/judgment.py — 判断层。

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
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from provider.catalog import lookup_model

_log = logging.getLogger("lingzhou.judgment")

# 低成本读取/枚举类工具 → reader tier
_READER_TOOLS = frozenset({
    "file.list", "file.read",
    "memory.get_fact",
    "schedule.list", "schedule.ack", "schedule.cancel",
    "shell.capabilities",
    "task.list",
    "failure.dismiss",
})
# 写入/推理/高风险工具 → reasoner tier
_REASONER_TOOLS = frozenset({
    "shell.run",
    "file.write",
    "task.add", "task.update", "task.advance", "task.complete", "task.fail",
    "memory.add_wm", "memory.add_semantic", "memory.set_fact", "memory.snapshot",
    "reflect.structural",
    "schedule.add",
})


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
    from memory.task_store import Task, TaskStore, Failure
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from tools.registry import ToolRegistry, ToolManifest
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

    @classmethod
    def wait(cls, reason: str = "") -> "JudgmentOutput":
        return cls(decision="wait", rationale=reason, reply_to_user="")

    @classmethod
    def from_llm(cls, text: str) -> "JudgmentOutput":
        """从 LLM 输出文本解析 JudgmentOutput，容错处理。"""
        original = text.strip()
        text = original
        # 防御：剥离 <think>...</think> 块（provider 层已处理，此处兜底）
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
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
            return cls.wait(reason=f"LLM 输出解析失败: {text}")

        return cls(
            decision=str(data.get("decision", "wait")).lower(),
            chosen_action_id=str(data.get("chosen_action_id", "")),
            params=dict(data.get("params") or {}),
            rationale=str(data.get("rationale", "")),
            reflection=str(data.get("reflection", "")),
            reply_to_user=str(data.get("reply_to_user", "")),
            next_step=str(data.get("next_step", "")),
            model_strategy=dict(data.get("model_strategy") or {}),
        )


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
        self._ref_resolver = ReferenceResolver(provider=provider)
        # 分层路由 providers：{"simple": <provider>, "complex": <provider>}
        # 由 loop.open() 在 bootstrap 后注入，未配置时为空字典
        self._routing_providers: dict[str, "Provider"] = {}
        # 内层工具循环用：缓存上一次 decide() 组装的完整上下文，由 decide_continue() 复用
        self._last_context_text: str = ""
        # 最近一次真实 LLM 调用元数据（供 loop 日志输出实际 model/tier/thinking）
        self._last_call_meta: dict[str, str] = {
            "phase": "",
            "tier": "default",
            "model_ref": cfg.model,
            "thinking": cfg.thinking,
        }
        # 每个模型最近一次调用错误（用于注入 model routing truth）
        self._provider_errors: dict[str, str] = {}
        # 每个模型的健康状态（429/400/timeout 触发冷却窗口，避免短时间重复打爆同一 provider）
        self._model_health: dict[str, ModelHealth] = {}
        # 运行时临时 provider 缓存：routing_overrides 指定的临时 model 按需创建并缓存
        self._override_providers: dict[str, "Provider"] = {}

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
            if tool_history and len(tool_history) >= 4:
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
                "current_thinking": self._cfg.thinking,
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

        posture = "respond" if user_message else ("converge" if task_explore_count >= 4 else "conserve")
        payload = {
            "available_models": available_models,
            "active_overrides": routing_overrides or {},
            "tier_descriptions": {
                "reader": "轻量感知层：适合常规状态查询、读文件、检查计划、无复杂推理的心跳 tick",
                "reasoner": "深度推理层：适合用户交互、要求判断、处理复杂状态、制定或调整计划",
                "repair": "修复层：专用于解析失败、格式错误、小修小补",
            },
            "delegation_guide": (
                "你是当前层的决策者，可以通过 model_strategy 中的以下字段调控下一轮行为：\n"
                "• next_phase_tier：分配下轮的推理层级。reader=轻量感知，reasoner=深度推理，repair=修复。"
                "示例：本轮已完成复杂判断并写入任务，下轮只需追踪状态 → next_phase_tier=reader；\n"
                "• next_idle_gap_secs：下一轮空闲等待时长（秒，整数，范围 5-600）。默认 60。"
                "示例：已发起 shell 命令，预计 30s 出结果 → next_idle_gap_secs=35；"
                "无任务等待用户下一步 → next_idle_gap_secs=120；任务进行中需快速追踪 → next_idle_gap_secs=10；\n"
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
            self._last_call_meta = {
                "phase": selection.phase,
                "tier": selection.tier,
                "model_ref": selection.model_ref,
                "thinking": selection.thinking,
            }
            try:
                raw = await selected_provider.chat(messages, thinking_override=thinking_override)
                self._mark_model_success(selection.model_ref)
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
            output = JudgmentOutput.wait(reason="act 决策缺少 chosen_action_id")

        _log.info(
            "[judgment] phase=%s tier=%s model=%s thinking=%s decision=%s action=%s rationale=%s",
            selection.phase, selection.tier, selection.model_ref, selection.thinking,
            output.decision, output.chosen_action_id, (output.rationale or "")[:120],
        )

        return output

    async def decide_continue(
        self,
        tool_history: list[dict],
        user_message: str = "",
        prefer_tier: str | None = None,
        routing_overrides: "dict[str, str] | None" = None,
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

        # 构建工具历史摘要（完整保留工具返回内容，现代大模型支持 100k+ context）
        history_parts: list[str] = []
        for i, h in enumerate(tool_history):
            params_str = json.dumps(h.get("params", {}), ensure_ascii=False)
            result_str = str(h.get("result", ""))
            history_parts.append(
                f"[{i + 1}] {h.get('tool', '')}({params_str})\n返回: {result_str}"
            )

        history_block = "\n\n".join(history_parts)
        hint = (
            "用户正在等待回复，尽快在本轮设置 reply_to_user 字段。"
            if user_message else ""
        )

        continuation_context = (
            f"{self._last_context_text}\n\n"
            "---\n"
            "## 本轮已执行工具历史\n"
            f"{history_block}\n\n"
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

        current_action = str(tool_history[-1].get("tool", "")) if tool_history else ""
        selected_provider, selection = self._select_provider(
            phase="continue",
            user_message=user_message,
            current_action=current_action,
            tool_history=tool_history,
            prefer_tier=prefer_tier,
            routing_overrides=routing_overrides,
        )
        # chat 模式 reasoner 阶段用 "low" 加快内层链推理；其余情况跟随配置
        thinking_override = "low" if (selection.tier == "reasoner" and user_message) else "off"
        self._last_call_meta = {
            "phase": selection.phase,
            "tier": selection.tier,
            "model_ref": selection.model_ref,
            "thinking": thinking_override,
        }
        raw: str | None = None
        for _attempt in range(2):
            try:
                raw = await selected_provider.chat(messages, thinking_override=thinking_override)
                self._mark_model_success(selection.model_ref)
                break
            except Exception as exc:
                _err = str(exc) or repr(exc)
                self._mark_model_failure(selection.model_ref, _err)
                if _attempt == 0:
                    _fallback_tier = self._fallback_tiers(selection.tier)[0]
                    fb_provider, fb_selection = self._select_provider(
                        phase="continue",
                        user_message=user_message,
                        current_action=current_action,
                        tool_history=tool_history,
                        prefer_tier=_fallback_tier,
                        thinking_override=thinking_override,
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

        _log.info(
            "[judgment.continue] round=%d phase=%s tier=%s model=%s thinking=%s decision=%s action=%s",
            len(tool_history), selection.phase, selection.tier, selection.model_ref,
            self._last_call_meta["thinking"], output.decision, output.chosen_action_id,
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
        """LLM 不可用时的确定性回退（Hermes simulate.go 移植）。
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
        routing_overrides: "dict[str, str] | None" = None,
    ) -> str:
        """将运行时状态填入 judgment 模板。"""
        task = await task_store.get_active()

        # 任务边界过滤失败记录（P2-B 原则）
        if task:
            failures = await task_store.list_failures_for_task(
                str(task.id), self._cfg.memory.failure_limit
            )
        else:
            failures = await task_store.list_failures(self._cfg.memory.failure_limit)

        # 情节记忆（当前任务叙事）
        task_id_str = str(task.id) if task else None
        episodic_text = episodic.load_for_context(task_id_str, self._cfg.memory.episodic_max_chars)

        # 情节搜索（跨任务全文检索，补充当前任务叙事之外的相关经历）
        search_query = (task.goal or task.title) if task else user_message
        episodic_search = episodic.search(search_query, max_chars=16000) if search_query else ""
        if episodic_search and episodic_search not in episodic_text:
            episodic_text = episodic_text + "\n\n[跨任务检索命中]\n" + episodic_search

        # 实体共指消解（本地候选召回 + LLM 推理判断）
        resolved_entities = await self._ref_resolver.resolve(user_message, semantic, episodic) if user_message else []
        entity_section = self._ref_resolver.format_section(resolved_entities)

        # 语义记忆：多锚点情境召回（goal + user_message + 失败 kind + 情绪）
        anchors: list[str] = []
        if task:
            anchors.append(task.goal or task.title)
        if user_message and user_message not in anchors:
            anchors.append(user_message[:100])
        if failures:
            anchors.append(failures[0].kind)
        emotion_label = _emotion_label(emotion, self._cfg)
        anchors.append(emotion_label)
        memories = semantic.retrieve_multi_anchor(anchors, self._cfg.memory.semantic_top_k)

        # Soul 信息（hard_axioms + ethos_baseline）
        axioms_val, _ = await task_store.get_fact("soul:hard_axioms")
        ethos_val, _ = await task_store.get_fact("soul:ethos_baseline")
        soul_section = _fmt_soul(axioms_val, ethos_val)

        # 按当前情境过滤技能，注入最相关的护栏（阈值及上限从配置传入）
        _wm_items = wm.get_top(15)
        skills = self._skills.match_for_context(
            wm_pressure=wm.pressure,
            has_active_task=task is not None,
            has_next_step=bool(task and task.next_step),
            failure_count=len(failures),
            high_error_streak=perception_replay.high_error_streak if perception_replay else 0,
            failure_threshold=self._cfg.thresholds.skill_failure_threshold,
            wm_pressure_threshold=self._cfg.thresholds.skill_wm_pressure_threshold,
            max_inject=self._cfg.thresholds.skill_max_inject,
        )

        ctx = {
            "task_section": _fmt_task(task),
            "emotion_valence": f"{emotion.valence:.2f}",
            "emotion_arousal": f"{emotion.arousal:.2f}",
            "emotion_dominant": emotion.dominant or "（未确定）",
            "emotion_regulation": f"{emotion.regulation.strategy}（{emotion.regulation.reason}）" if emotion.regulation.reason else emotion.regulation.strategy,
            "wm_section": _fmt_wm(_wm_items, wm_count=len(wm), wm_capacity=wm._capacity),
            "failures_section": _fmt_failures(failures),
            "episodic_section": episodic_text or "（暂无情节记忆）",
            "entity_section": entity_section,
            "memories_section": _fmt_memories(memories),
            "soul_section": soul_section,
            "tools_section": _fmt_tools(self._registry.list_manifests()),
            "shell_capabilities_section": _fmt_shell_capabilities(),
            "perception_section": _fmt_percept(percept),
            "ethos_section": _fmt_ethos(ethos_state),
            "signals_section": _fmt_judgment_signals(judgment_signals),
            "hard_boundaries_section": _fmt_hard_boundaries(hard_boundaries),
            "perception_replay_section": _fmt_perception_replay(perception_replay),
            "skills_section": _fmt_skills(skills),
            "cognitive_signals_section": _fmt_cognitive_signals(cognitive_signals),
            "model_routing_section": self._build_model_routing_section(
                phase=phase,
                user_message=user_message,
                current_action=current_action,
                tool_history=tool_history,
                routing_overrides=routing_overrides,
            ),
            "current_time_section": _fmt_current_time(),
            "user_message": user_message or "",
        }
        ctx = apply_context_budget(
            ctx,
            self._cfg.judgment_input_token_budget(),
            skill_min_tokens=self._cfg.thresholds.skill_min_budget_tokens,
        )
        return _fill_template(self._judgment_template, ctx)


# ── 格式化辅助函数 ─────────────────────────────────────────────────────────────

def _fmt_task(task: "Task | None") -> str:
    if not task:
        return "（无活跃任务，可自主探索或等待）"
    age_str = ""
    if task.created_at:
        try:
            created = datetime.fromisoformat(task.created_at.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            elapsed = datetime.now(timezone.utc) - created
            total_secs = int(elapsed.total_seconds())
            if total_secs < 60:
                age_str = f"（已进行 {total_secs}s）"
            elif total_secs < 3600:
                age_str = f"（已进行 {total_secs // 60}m）"
            elif total_secs < 86400:
                h, m = divmod(total_secs // 60, 60)
                age_str = f"（已进行 {h}h {m}m）"
            else:
                d, rem = divmod(total_secs, 86400)
                age_str = f"（已进行 {d}d {rem // 3600}h）"
        except Exception:
            pass
    return (
        f"ID: {task.id}\n"
        f"标题: {task.title}{age_str}\n"
        f"目标: {task.goal or '（未指定）'}\n"
        f"优先级: {task.priority}\n"
        f"下一步: {task.next_step or '（未指定）'}"
    )


def _fmt_current_time() -> str:
    """生成当前时间行，格式与 OpenClaw current-time.ts 对齐。"""
    now = datetime.now(timezone.utc)
    # 本地 ISO 字符串（服务器时区）
    local_iso = now.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    utc_str = now.strftime("%Y-%m-%d %H:%M UTC")
    return f"当前时间: {local_iso}\n参考 UTC: {utc_str}"


def _fmt_wm(items: list[dict[str, Any]], wm_count: int = 0, wm_capacity: int = 20) -> str:
    header = f"[{wm_count}/{wm_capacity}，{wm_count / wm_capacity:.0%}]"
    if not items:
        return f"{header} （工作记忆为空）"
    # 反循环感知条目（self_awareness / heartbeat 中的循环警告）强制置顶，
    # 避免被正常工具结果条目淹没（Hermes 借鉴：高优先级信号不受顺序影响）
    anti_loop = [i for i in items if i.get("kind") == "self_awareness"]
    rest = [i for i in items if i.get("kind") != "self_awareness"]
    ordered = anti_loop + rest
    lines = [header] + [f"- [{i['kind']}] {i['content']}" for i in ordered]
    return "\n".join(lines)


def _fmt_failures(failures: "list[Failure]") -> str:
    if not failures:
        return "（无近期失败）"
    lines = [f"- [#{f.id}][{f.kind}] {f.summary}" for f in failures]
    return "\n".join(lines)


def _fmt_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "（无相关记忆）"
    lines = [f"- [{m['kind']}] {m['title']}: {m['body']}" for m in memories]
    return "\n".join(lines)


def _fmt_tools(manifests: "list[ToolManifest]") -> str:
    if not manifests:
        return "（无可用工具）"
    lines: list[str] = []
    for m in manifests:
        params_str = ", ".join(
            f"{p.name}({'*' if p.required else '?'})" for p in m.params
        )
        lines.append(f"- `{m.name}`: {m.description}  参数: [{params_str}]")
    return "\n".join(lines)


def _fmt_shell_capabilities() -> str:
    """运行时 shell 能力真相：避免 LLM 将“宿主环境限制”误判为“平台沙盒”。"""
    cmds = (
        "python3", "python", "bash", "sh", "grep", "find", "ls", "cat",
        "sqlite3", "git", "sed", "awk", "jq", "rg",
    )
    available = [c for c in cmds if shutil.which(c)]
    payload = {
        "engine": "asyncio.create_subprocess_shell",
        "execution_model": "one-shot-non-persistent",
        "sandbox": False,
        "network_policy": "inherits-host-environment",
        "default_timeout_sec": 30,
        "default_output_preview_chars": 500,
        "shell": os.environ.get("SHELL") or "/bin/sh",
        "cwd": os.getcwd(),
        "available_commands": available,
        "missing_commands": [c for c in cmds if c not in available],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _fmt_percept(percept: "Percept") -> str:
    return (
        f"预测误差: {percept.prediction_error:.2f}  "
        f"工作区变更: {'是' if percept.workspace_dirty else '否'}"
    )


def _fmt_soul(axioms_val: str, ethos_val: str) -> str:
    parts: list[str] = []
    if axioms_val:
        parts.append(f"绝对禁忌（hard_axioms）: {axioms_val}")
    if ethos_val:
        parts.append(f"价值基线（ethos_baseline）: {ethos_val}")
    return "\n".join(parts) if parts else "（Soul 未初始化，运行 `init` 命令生成）"


def _emotion_label(emotion: "EmotionState", cfg: "Config") -> str:
    """Russell (1980) 环形模型：将 valence/arousal 映射为情绪标签，作为情境销回锐点。
    阈值全部来自 cfg.emotion，不硬编码。"""
    ec = cfg.emotion
    vh, vl = ec.mood_valence_high, ec.mood_valence_low
    ah = ec.mood_arousal_high
    if emotion.valence < vl and emotion.arousal > ah:
        return "焦虑"
    if emotion.valence < vl:
        return "沮丧"
    if emotion.valence > vh and emotion.arousal > ah:
        return "兴奋"
    if emotion.valence > vh:
        return "稳定"
    return "中性"


def _fill_template(template: str, ctx: dict[str, Any]) -> str:
    """替换 {{key}} 占位符，保留其他 { } 不动。"""
    def replace(m: re.Match[str]) -> str:
        key = m.group(1).strip()
        return str(ctx.get(key, f"[未知字段: {key}]"))
    return re.sub(r"\{\{([^}]+)\}\}", replace, template)


def _fmt_ethos(ethos_state: "EthosState | None") -> str:
    if not ethos_state:
        return "（Ethos 未计算）"
    v = ethos_state.values
    b = ethos_state.bias
    lines: list[str] = [
        f"价値图式  truth={v.truth:.2f}  caution={v.caution:.2f}  "
        f"continuity={v.continuity:.2f}  curiosity={v.curiosity:.2f}  care={v.care:.2f}",
    ]
    biases: list[str] = []
    if b.prefer_verification:
        biases.append("prefer_verification")
    if b.prefer_narrow_scope:
        biases.append("prefer_narrow_scope")
    if b.preserve_continuity:
        biases.append("preserve_continuity")
    if b.avoid_overclaiming:
        biases.append("avoid_overclaiming")
    if biases:
        lines.append(f"行为倾向  {', '.join(biases)}")
    if b.reasons:
        lines.append(f"理由      {'; '.join(b.reasons)}")
    return "\n".join(lines)


def apply_context_budget(
    ctx: dict[str, str],
    token_budget: int | None = None,
    max_chars: int | None = None,
    skill_min_tokens: int = 0,
) -> dict[str, str]:
    """按优先级压缩 judgment 输入，优先保留任务、感知、禁忌与 Soul。

    skill_min_tokens: skills_section 下限（小于此就不裁剪），默认 0。
    建议从 cfg.thresholds.skill_min_budget_tokens 传入（默认 80），
    确保压力最大时护栏不是第一个被裁掉的内容。
    """
    if token_budget is None:
        token_budget = max_chars
    if token_budget is None:
        raise TypeError("apply_context_budget() missing required argument: 'token_budget'")
    if token_budget <= 0:
        return ctx

    budgeted = dict(ctx)
    priority = [
        "skills_section",
        "memories_section",
        "episodic_section",
        "wm_section",
        "tools_section",
    ]
    minimum_keep = {
        "skills_section": skill_min_tokens,
        "memories_section": 1,
        "episodic_section": 2,
        "wm_section": 1,
        "tools_section": 2,
    }

    def total_tokens(items: dict[str, str]) -> int:
        return sum(_estimate_tokens(value) for value in items.values())

    current_total = total_tokens(budgeted)
    if current_total <= token_budget:
        return budgeted

    for key in priority:
        if current_total <= token_budget:
            break
        original = budgeted.get(key, "")
        if not original:
            continue

        keep_floor = minimum_keep.get(key, 0)
        original_tokens = _estimate_tokens(original)
        if original_tokens <= keep_floor:
            continue

        reduction = min(original_tokens - keep_floor, current_total - token_budget)
        keep_tokens = max(keep_floor, original_tokens - reduction)
        trimmed = _compress_text_segments(original, keep_tokens)
        budgeted[key] = trimmed
        current_total -= _estimate_tokens(original) - _estimate_tokens(trimmed)

    return budgeted


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数，用于 prompt 预算裁剪。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_chars = sum(1 for ch in text if ord(ch) < 128 and not ch.isspace())
    other = sum(1 for ch in text if ord(ch) >= 128 and not ("\u4e00" <= ch <= "\u9fff"))
    return cjk + max(1, ascii_chars // 4) + max(1, other // 2)


def _compress_text_segments(text: str, keep_tokens: int) -> str:
    if keep_tokens <= 0:
        return ""
    if _estimate_tokens(text) <= keep_tokens:
        return text

    segments = _split_segments(text)
    if not segments:
        return ""

    keep_head: list[str] = []
    keep_tail: list[str] = []
    head_tokens = 0
    tail_tokens = 0

    head_idx = 0
    tail_idx = len(segments) - 1
    turn = 0

    while head_idx <= tail_idx:
        if turn % 2 == 0:
            candidate = segments[head_idx]
            candidate_tokens = _estimate_tokens(candidate)
            if head_tokens + tail_tokens + candidate_tokens <= keep_tokens:
                keep_head.append(candidate)
                head_tokens += candidate_tokens
                head_idx += 1
            elif tail_idx == head_idx and not keep_head and not keep_tail:
                keep_head.append(_compress_single_segment(candidate, keep_tokens))
                break
            else:
                break
        else:
            candidate = segments[tail_idx]
            candidate_tokens = _estimate_tokens(candidate)
            if head_tokens + tail_tokens + candidate_tokens <= keep_tokens:
                keep_tail.append(candidate)
                tail_tokens += candidate_tokens
                tail_idx -= 1
            elif tail_idx == head_idx and not keep_head and not keep_tail:
                keep_tail.append(_compress_single_segment(candidate, keep_tokens))
                break
            else:
                break
        turn += 1

    if not keep_head and not keep_tail:
        return _compress_single_segment(text, keep_tokens)

    body = keep_head + (["\n[...省略...]\n"] if head_idx <= tail_idx else []) + list(reversed(keep_tail))
    return "".join(body)


def _split_segments(text: str) -> list[str]:
    parts = re.split(r"(\n\s*\n)", text)
    segments: list[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"\n\s*\n", part):
            if buffer:
                segments.append(buffer)
                buffer = ""
            segments.append(part)
        else:
            buffer += part
    if buffer:
        segments.append(buffer)
    return segments


def _compress_single_segment(text: str, keep_tokens: int) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) <= 1:
        return text[: max(1, min(len(text), keep_tokens * 4))]

    kept: list[str] = []
    token_count = 0
    for line in lines:
        line_tokens = _estimate_tokens(line)
        if token_count + line_tokens > keep_tokens:
            break
        kept.append(line)
        token_count += line_tokens

    if kept:
        return "".join(kept) + ("\n[...省略...]" if len(kept) < len(lines) else "")
    return text[: max(1, min(len(text), keep_tokens * 4))]


def _fmt_judgment_signals(signals: "JudgmentSignals | None") -> str:
    if not signals:
        return "（JudgmentSignals 未计算）"
    return (
        f"posture={signals.posture}  "
        f"require_more_evidence={signals.require_more_evidence}  "
        f"prefer_narrow_scope={signals.prefer_narrow_scope}"
    )


def _fmt_hard_boundaries(hard_boundaries: "list[str] | None") -> str:
    if not hard_boundaries:
        return "（无 hard_boundary 限制）"
    return "\n".join(f"- {b}" for b in hard_boundaries)


def _fmt_perception_replay(replay: "PerceptionReplaySummary | None") -> str:
    if not replay:
        return "（感知重放不可用）"
    lines = [
        f"样本数={replay.samples}  平均预测误差={replay.avg_prediction_error:.2f}  "
        f"连续高误差={replay.high_error_streak}  趋势={replay.trend}",
    ]
    if replay.hints:
        for hint in replay.hints:
            lines.append(f"提示: {hint}")
    return "\n".join(lines)


def _fmt_skills(skills: "list[Skill]") -> str:
    if not skills:
        return "（暂无认知框架）"
    parts: list[str] = []
    for s in skills:
        parts.append(f"**{s.name}** — {s.description}\n> {s.guidance}")
    return "（以下为全部可选框架，根据实际情境自行判断适用哪些，可全部忽略）\n\n" + "\n\n".join(parts)


def _fmt_cognitive_signals(signals: "CognitiveSignals | None") -> str:
    if signals is None:
        return "（认知信号暂不可用）"
    return signals.to_text()
