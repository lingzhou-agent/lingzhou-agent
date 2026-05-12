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
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

_log = logging.getLogger("lingzhou.judgment")

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

    @classmethod
    def wait(cls, reason: str = "") -> "JudgmentOutput":
        return cls(decision="wait", rationale=reason, reply_to_user="")

    @classmethod
    def from_llm(cls, text: str) -> "JudgmentOutput":
        """从 LLM 输出文本解析 JudgmentOutput，容错处理。"""
        text = text.strip()
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
            return cls.wait(reason=f"LLM 输出解析失败: {text[:100]}")

        return cls(
            decision=str(data.get("decision", "wait")).lower(),
            chosen_action_id=str(data.get("chosen_action_id", "")),
            params=dict(data.get("params") or {}),
            rationale=str(data.get("rationale", "")),
            reflection=str(data.get("reflection", "")),
            reply_to_user=str(data.get("reply_to_user", "")),
            next_step=str(data.get("next_step", "")),
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
        self._skills = SkillRegistry()
        self._ref_resolver = ReferenceResolver(provider=provider)
        # 分层路由 providers：{"simple": <provider>, "complex": <provider>}
        # 由 loop.open() 在 bootstrap 后注入，未配置时为空字典
        self._routing_providers: dict[str, "Provider"] = {}
        # 内层工具循环用：缓存上一次 decide() 组装的完整上下文，由 decide_continue() 复用
        self._last_context_text: str = ""

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

    def _select_provider(self, user_message: str) -> "Provider":
        """根据是否有用户消息选择 provider。
        有 user_message → 'complex'（高能力模型，需要精确回复）
        无 user_message → 'simple'（轻量模型，节约资源）
        """
        if not self._routing_providers:
            return self._provider
        tier = "complex" if user_message else "simple"
        return self._routing_providers.get(tier, self._provider)
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
    ) -> JudgmentOutput:
        """组装上下文，调用 LLM，返回决策。
        
        thinking_override: 覆盖 cfg.thinking（如 chat 模式用 "low" 加速首轮判断）。
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

        selected_provider = self._select_provider(user_message)
        raw: str | None = None
        for _attempt in range(2):
            try:
                raw = await selected_provider.chat(messages, thinking_override=thinking_override)
                break
            except Exception as exc:
                if _attempt == 0:
                    _warn = str(exc) or repr(exc)
                    _log.warning("[judgment] LLM 调用失败，1s 后重试: %s", _warn)
                    await asyncio.sleep(1.0)
                else:
                    _err = str(exc) or repr(exc)
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
                await task_store.record_failure("judgment_parse", output.rationale[:200])

        if output.decision not in ("act", "pause", "wait"):
            output = JudgmentOutput.wait(reason=f"无效 decision: {output.decision!r}")
        if output.decision == "act" and not output.chosen_action_id:
            output = JudgmentOutput.wait(reason="act 决策缺少 chosen_action_id")

        _log.info(
            "[judgment] decision=%s action=%s rationale=%s",
            output.decision, output.chosen_action_id, (output.rationale or "")[:120],
        )

        return output

    async def decide_continue(
        self,
        tool_history: list[dict],
        user_message: str = "",
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

        # 构建工具历史摘要（限制单条 result 长度，防止如果工具返回大文件内容执行 token 爆炸）
        history_parts: list[str] = []
        for i, h in enumerate(tool_history):
            params_str = json.dumps(h.get("params", {}), ensure_ascii=False)[:300]
            result_str = str(h.get("result", ""))[:1000]
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

        selected_provider = self._select_provider(user_message)
        raw: str | None = None
        for _attempt in range(2):
            try:
                # 内层续判：关闭 thinking，避免每轮额外 40-60s 推理耗时
                raw = await selected_provider.chat(messages, thinking_override="off")
                break
            except Exception as exc:
                if _attempt == 0:
                    _log.warning("[judgment.continue] LLM 调用失败，1s 后重试: %s", str(exc) or repr(exc))
                    await asyncio.sleep(1.0)
                else:
                    _log.warning("[judgment.continue] LLM 调用失败: %s", str(exc) or repr(exc))
                    return JudgmentOutput.wait(reason=f"[inner-loop] LLM 不可用: {exc!r}")

        if raw is None:
            return JudgmentOutput.wait(reason="[inner-loop] LLM returned None")

        output = JudgmentOutput.from_llm(raw)
        if output.decision not in ("act", "pause", "wait"):
            output = JudgmentOutput.wait(reason=f"无效 decision: {output.decision!r}")
        if output.decision == "act" and not output.chosen_action_id:
            output = JudgmentOutput.wait(reason="act 决策缺少 chosen_action_id")

        _log.info(
            "[judgment.continue] round=%d decision=%s action=%s",
            len(tool_history), output.decision, output.chosen_action_id,
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
                    "只输出合法 JSON，不要解释，不要使用 markdown。"
                    "必须遵循这个 schema: {decision, chosen_action_id, params, rationale, reflection, reply_to_user, next_step}."
                    "如果原输出被截断，请根据上下文重新生成一个完整、简短的 JSON。"
                ),
            ),
            Message(
                role="user",
                content=(
                    "下面是原始判断上下文和一段损坏/截断的模型输出，请修复为合法 JSON。\n\n"
                    f"[context]\n{context_text}\n\n"
                    f"[broken_output]\n{raw[:4000]}\n\n"
                    "只返回 JSON。"
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
        行为原则：hard_boundary > posture > wait。"""
        if hard_boundaries:
            return JudgmentOutput.wait(reason=f"[fallback] hard_boundary 阻断，LLM 不可用: {reason}")
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
        episodic_search = episodic.search(search_query, max_chars=800) if search_query else ""
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
            "wm_section": _fmt_wm(_wm_items),
            "failures_section": _fmt_failures(failures),
            "episodic_section": episodic_text or "（暂无情节记忆）",
            "entity_section": entity_section,
            "memories_section": _fmt_memories(memories),
            "soul_section": soul_section,
            "tools_section": _fmt_tools(self._registry.list_manifests()),
            "perception_section": _fmt_percept(percept),
            "ethos_section": _fmt_ethos(ethos_state),
            "signals_section": _fmt_judgment_signals(judgment_signals),
            "hard_boundaries_section": _fmt_hard_boundaries(hard_boundaries),
            "perception_replay_section": _fmt_perception_replay(perception_replay),
            "skills_section": _fmt_skills(skills),
            "cognitive_signals_section": _fmt_cognitive_signals(cognitive_signals),
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


def _fmt_wm(items: list[dict[str, Any]]) -> str:
    if not items:
        return "（工作记忆为空）"
    # 进入判断上下文的 WM 不做随意截断，避免关键信息丢失导致循环误判
    lines = [f"- [{i['kind']}] {i['content']}" for i in items]
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
