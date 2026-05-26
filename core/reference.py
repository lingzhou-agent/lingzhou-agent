"""core/reference.py — 跨 chat 实体共指消解（Cross-Chat Entity Coreference Resolution）。

架构理念：
    纯解析式只能"找到"字面匹配——「小张」命中某个交互对象节点。
  但「小张离职了」「上次你推荐的方案有问题」「我想继续之前那个话题」
  这类语义需要真正的推理：理解状态变化、隐式指代、关系性质。
  只有让 LLM 参与思考，才是在构建数字生命，而不是解析器。

两阶段流水线：
    1. 本地预热（快，免费）
        - 轻量提取自我介绍 / 回指提示 / 主题锚点 → ExtractedSignals
    - FTS5 多锚点召回候选集（候选预算来自 config）
  2. LLM 推理（慢，有价值）
     - 将消息 + 候选节点摘要 → 专用小 prompt → LLM 判断实体关联
     - LLM 返回：哪些节点真正被引用、关联性质（引用/状态变化/隐式/自我介绍）
     - Provider 不可用时自动降级为本地评分（保留可用性）

理论依据：
  Wu et al. (2020)  Scalable Zero-shot Entity Linking — mention → KB entity
  Xu et al. (2021)  Beyond Goldfish Memory (MSC) — 跨 session 实体一致性
  Park et al. (2023) Generative Agents — importance 驱动记忆激活
  Anderson (1983)   ACT-R — 多锚点激活叠加
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.config import ThresholdsConfig

if TYPE_CHECKING:
    from store.semantic import SemanticMemory
    from store.episodic import EpisodicMemory
    from store.task import TaskStore
    from provider.base import Provider

_log = logging.getLogger("lingzhou.reference")

_TOPIC_PUNCT_PATTERN = re.compile(r"[，。！？；、,.!?]+")
_SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?；;\n]+")
_SELF_NAME_PATTERNS = (
    re.compile(r"(?:我叫|我的名字是|你可以叫我|请叫我|以后叫我|下次叫我|就叫我)\s*[:：]?\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,24})"),
    re.compile(r"(?:我是)\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,24})"),
)
_INTERLOCUTOR_TYPE_HINTS: dict[str, tuple[str, ...]] = {
    "agent": ("agent", "subagent", "智能体"),
    "bot": ("bot", "robot", "机器人"),
    "assistant": ("assistant", "助手", "助理"),
    "ai": ("ai", "llm", "模型", "gpt", "claude", "qwen", "gemini", "copilot"),
    "webhook": ("webhook",),
    "internal": ("internal",),
    "external": ("external",),
}

# LLM 推理提示（专用小 prompt，与 judgment bundle 完全解耦）
_REASON_SYSTEM = """\
你是灵舟的记忆链接器，负责实体共指消解。

任务：分析用户消息，在候选记忆节点中找出与消息存在真实上下文关联的实体。

判断维度（任一满足即纳入）：
  direct    — 消息直接提及该实体名称或别名
  state     — 消息描述了该实体的状态变化（如"离职""完成""取消"）
  implicit  — 消息通过"上次的""你推荐的"等隐式引用该实体
  self_intro — 消息是自我介绍，与人物节点对应
    temporal  — 消息中的时间感知与候选节点 created_at / 最近叙事上下文吻合

可结合用户消息中的相对时间表达（如"昨天""刚才""上次"）自行判断 temporal 关联，
不要假定外部已经把这些时间词换算成固定小时窗口。

输出格式：JSON 数组，不加 markdown。每项字段：
  node_id          — 候选节点的 id（原样输出，不得修改）
  confidence       — 关联置信度 0.0~1.0（两位小数）
  relationship_note — 一句话说明关联性质（中文，≤20字）

无关候选不输出。无相关实体时输出 []。\
"""

_SPEAKER_REASON_SYSTEM = """\
你是灵舟的当前交互对象识别器。

任务：结合当前用户消息、最近交互记忆、当前 chat 连续性、当前对象的跨 chat 连续性和候选交互对象画像，判断“当前消息来自谁”。

判断原则：
    - chat_id / handle 只能当作线索，不能单独当成身份证明。
    - 必须优先综合：自我介绍、稳定偏好、记忆要求、过往互动连续性、交互对象画像摘要。
    - 如果候选里没有足够匹配的人，可以输出 NEW；如果证据太弱，就输出 UNKNOWN。

输出格式：JSON 对象，不加 markdown。字段：
    node_id            — 命中的交互对象节点 id；新对象填 NEW；无法判断填 UNKNOWN
    confidence         — 0.0~1.0，两位小数
    display_name       — 当前交互对象的称呼；命中旧节点时尽量沿用节点标题
    relationship_note  — 一句话说明为何认成此对象（中文，<=24字）
    evidence           — 最多 3 条证据短句数组
    provisional        — 是否只是临时画像 true/false
\
"""


@dataclass
class ExtractedSignals:
    """轻量预热阶段提取的检索信号。"""
    topic_anchors: list[str] = field(default_factory=list[str])


@dataclass
class ResolvedEntity:
    """经 LLM 推理确认的实体链接结果。"""
    node_id: str
    title: str
    kind: str
    confidence: float
    snippet: str                    # body 前若干字（由 config 控制）
    signal_types: list[str]         # 本地检索路径（调试用）
    relationship_note: str = ""     # LLM 推理给出的关联说明


@dataclass
class ResolvedSpeaker:
    """当前交互对象的画像识别结果。"""
    node_id: str
    title: str
    confidence: float
    snippet: str
    evidence: list[str] = field(default_factory=list)
    relationship_note: str = ""
    signal_types: list[str] = field(default_factory=list)
    provisional: bool = False
    search_anchors: list[str] = field(default_factory=list)
    source_traits: list[str] = field(default_factory=list)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).strip()


def _split_sentences(text: str) -> list[str]:
    return [
        part.strip(" ,，。；;!！?？")
        for part in _SENTENCE_SPLIT_PATTERN.split(_clean_text(text))
        if part and part.strip()
    ]


def _digest_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _speaker_handle_tag(chat_id: str) -> str:
    return f"handle:{chat_id.strip()}"


def _fallback_speaker_title(chat_id: str) -> str:
    if not chat_id:
        return "当前交互对象"
    return f"当前交互对象@{chat_id[-12:]}"


class ReferenceResolver:
    """实体共指消解器：本地候选收窄 + LLM 推理判断。

    JudgmentLayer 持有单例，跨 tick 复用。
    Provider 不可用时自动降级为纯本地评分。
    """

    def __init__(
        self,
        provider: "Provider | None" = None,
        *,
        thresholds: ThresholdsConfig | None = None,
        reason_temperature: float | None = None,
    ) -> None:
        self._provider = provider
        self._last_llm_error: str = ""
        self._last_llm_error_code: str = ""
        self._thresholds = thresholds or ThresholdsConfig()
        self._reason_temperature = reason_temperature

    @property
    def last_llm_error(self) -> str:
        return self._last_llm_error

    @property
    def last_llm_error_code(self) -> str:
        return self._last_llm_error_code

    @property
    def llm_available(self) -> bool:
        return self._provider is not None and not self._last_llm_error

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

    # ── 阶段一：轻量信号提取（< 1ms）────────────────────────────────────────

    def extract_signals(self, message: str) -> ExtractedSignals:
        sigs = ExtractedSignals()
        cleaned = re.sub(r"\s+", " ", message).strip()
        cleaned = _TOPIC_PUNCT_PATTERN.sub(" ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) >= self._thresholds.reference_topic_anchor_min_chars:
            sigs.topic_anchors.append(cleaned[: self._thresholds.reference_anchor_text_chars])
        return sigs

    # ── 阶段二：本地候选召回（FTS5，O(log n)）────────────────────────────────

    def _retrieve_candidates(
        self,
        message: str,
        sigs: ExtractedSignals,
        semantic: "SemanticMemory",
        episodic: "EpisodicMemory",
        source: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """返回 {node_id: node_dict}，最多 reference_candidate_cap 个候选节点。"""
        seen: set[str] = set()
        candidates: dict[str, dict[str, Any]] = {}

        def _add(nodes: list[dict[str, Any]], sig: str) -> None:
            for nd in nodes:
                nid = nd.get("id", "")
                if nid and nid not in seen:
                    seen.add(nid)
                    nd["_sig"] = nd.get("_sig", []) + [sig]
                    candidates[nid] = nd

        # 话题 + 整条消息 → 多锚点召回
        anchors: list[str] = []
        for anchor in [message[: self._thresholds.reference_anchor_text_chars], *sigs.topic_anchors]:
            if anchor and anchor not in anchors:
                anchors.append(anchor)
            if len(anchors) >= self._thresholds.reference_max_anchors:
                break
        _add(semantic.retrieve_multi_anchor(anchors, top_k=self._thresholds.reference_topic_top_k, source=source), "topic")

        # 最近叙事预热 → 只提供最近上下文，不替用户解释时间词
        recent_rows = episodic.list_recent_narrative(limit=self._thresholds.reference_recent_narrative_limit)
        for row in recent_rows:
            content = row.get("content", "")[: self._thresholds.reference_anchor_text_chars]
            if content:
                _add(semantic.retrieve(content, top_k=self._thresholds.reference_recent_semantic_top_k, source=source), "recent")

        return dict(list(candidates.items())[: self._thresholds.reference_candidate_cap])

    def _extract_source_traits(self, message: str, *, chat_id: str = "", source_hint: str = "") -> list[str]:
        traits: list[str] = []

        def _add(trait: str) -> None:
            normalized = trait.strip()
            if normalized and normalized not in traits:
                traits.append(normalized)

        if chat_id:
            channel = chat_id.split(":", 1)[0].strip().lower()
            if channel:
                _add(f"channel={channel}")

        lowered_source = f" {source_hint.lower()} " if source_hint else ""
        for kind, tokens in _INTERLOCUTOR_TYPE_HINTS.items():
            if any(token.isascii() and f" {token} " in lowered_source for token in tokens if token.isascii()):
                _add(f"source_kind={kind}")
            elif any((not token.isascii()) and token in source_hint for token in tokens):
                _add(f"source_kind={kind}")
        if source_hint.strip():
            compact_source = re.sub(r"\s+", " ", source_hint.strip())[:48]
            _add(f"route={compact_source}")

        lowered_message = f" {message.lower()} " if message else ""
        for kind, tokens in _INTERLOCUTOR_TYPE_HINTS.items():
            if any(token.isascii() and f" {token} " in lowered_message for token in tokens if token.isascii()):
                _add(f"counterparty={kind}")
                continue
            if any((not token.isascii()) and token in message for token in tokens):
                _add(f"counterparty={kind}")

        return traits[:5]

    def _extract_identity_cues(self, message: str, *, chat_id: str = "", source_hint: str = "") -> dict[str, list[str]]:
        text = _clean_text(message)
        if not text:
            return {"names": [], "preferences": [], "explicit": [], "source_traits": self._extract_source_traits(message, chat_id=chat_id, source_hint=source_hint)}

        names: list[str] = []
        for pattern in _SELF_NAME_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            name = match.group(1).strip()
            if name and name not in names:
                names.append(name)

        preferences: list[str] = []
        explicit: list[str] = []
        for sentence in _split_sentences(text):
            if any(token in sentence for token in ("我喜欢", "我偏好", "我更喜欢", "请用", "以后用", "先给结论", "直接说结论")):
                if sentence not in preferences:
                    preferences.append(sentence)
            if any(token in sentence for token in ("记住", "别忘了", "请记得")):
                if sentence not in explicit:
                    explicit.append(sentence)

        return {
            "names": names[:2],
            "preferences": preferences[:3],
            "explicit": explicit[:3],
            "source_traits": self._extract_source_traits(message, chat_id=chat_id, source_hint=source_hint),
        }

    def _retrieve_speaker_candidates(
        self,
        message: str,
        semantic: "SemanticMemory",
        *,
        chat_id: str = "",
        recent_turns: list[dict[str, Any]] | None = None,
        chat_continuity: str = "",
        cached_profile_id: str = "",
        source_hint: str = "",
    ) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
        cues = self._extract_identity_cues(message, chat_id=chat_id, source_hint=source_hint)
        seen: set[str] = set()
        candidates: dict[str, dict[str, Any]] = {}

        def _retrieve_profiles(query: str, *, top_k: int, tag: str | None = None) -> list[dict[str, Any]]:
            normalized = query.strip()
            if not normalized:
                return []
            nodes: list[dict[str, Any]] = []
            nodes.extend(semantic.retrieve(normalized, top_k=top_k, kind="interlocutor", tag=tag))
            return nodes

        def _add(nodes: list[dict[str, Any]], signal: str) -> None:
            for raw in nodes:
                node_id = str(raw.get("id") or "").strip()
                if not node_id:
                    continue
                existing = candidates.get(node_id)
                if existing is None:
                    item = dict(raw)
                    item["_sig"] = [signal]
                    candidates[node_id] = item
                    seen.add(node_id)
                else:
                    sigs = list(existing.get("_sig") or [])
                    if signal not in sigs:
                        sigs.append(signal)
                    existing["_sig"] = sigs

        if cached_profile_id:
            cached = semantic.get(cached_profile_id)
            if cached is not None and cached.kind == "interlocutor":
                _add([cached.to_dict()], "cached")

        if message.strip():
            _add(_retrieve_profiles(message[: self._thresholds.reference_anchor_text_chars], top_k=3), "message")

        for name in cues["names"]:
            _add(_retrieve_profiles(name, top_k=3), "self_name")

        if chat_id:
            handle_tag = _speaker_handle_tag(chat_id)
            _add(_retrieve_profiles(chat_id, top_k=2, tag=handle_tag), "handle_tag")
            _add(_retrieve_profiles(chat_id, top_k=2), "handle_text")

        if chat_continuity.strip():
            _add(
                _retrieve_profiles(chat_continuity[: self._thresholds.reference_anchor_text_chars], top_k=2),
                "chat_continuity",
            )

        for trait in cues.get("source_traits", [])[:3]:
            _add(_retrieve_profiles(trait, top_k=1), "source_trait")

        for turn in (recent_turns or [])[-4:]:
            content = _clean_text(str(turn.get("content") or ""))
            if not content:
                continue
            _add(_retrieve_profiles(content[: self._thresholds.reference_anchor_text_chars], top_k=1), "recent_turn")

        return dict(list(candidates.items())[: self._thresholds.reference_candidate_cap]), cues

    # ── 阶段三：LLM 推理（核心思考）────────────────────────────────────────

    async def _llm_reason(
        self,
        message: str,
        candidates: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """让 LLM 从候选集中判断哪些实体真正被引用，以及关联性质。"""
        if self._provider is None:
            self._last_llm_error = ""
            self._last_llm_error_code = ""
            return []
        from provider.base import Message as LLMMessage

        # 构造候选节点摘要（控制 token 数）
        cand_lines: list[str] = []
        for nid, nd in candidates.items():
            body_snippet = nd.get("body", "")[: self._thresholds.reference_candidate_body_chars].replace("\n", " ")
            created_at = str(nd.get("created_at", ""))
            cand_lines.append(
                f'  {{"id":"{nid}","kind":"{nd.get("kind","")}","title":"{nd.get("title","")}","created_at":"{created_at}","body":"{body_snippet}"}}'
            )
        cand_block = "[\n" + ",\n".join(cand_lines) + "\n]"

        user_content = f'用户消息："{message}"\n\n候选节点：\n{cand_block}'

        try:
            raw = await self._provider.chat(
                [
                    LLMMessage(role="system", content=_REASON_SYSTEM),
                    LLMMessage(role="user", content=user_content),
                ],
                temperature=self._reason_temperature,
            )
        except Exception as exc:
            err_text = str(exc) or repr(exc)
            self._last_llm_error = err_text
            self._last_llm_error_code = self._classify_error_code(err_text)
            _log.warning("[reference] LLM 推理失败，降级为本地评分: %s", exc)
            return []
        self._last_llm_error = ""
        self._last_llm_error_code = ""

        # 解析 JSON 数组
        raw = raw.strip()
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            return json.loads(raw[start: end + 1])
        except json.JSONDecodeError:
            return []

    async def _llm_reason_speaker(
        self,
        message: str,
        *,
        candidates: dict[str, dict[str, Any]],
        recent_turns: list[dict[str, Any]] | None = None,
        chat_continuity: str = "",
        interlocutor_continuity: str = "",
        chat_id: str = "",
        source_hint: str = "",
        cues: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        if self._provider is None:
            self._last_llm_error = ""
            self._last_llm_error_code = ""
            return {}
        from provider.base import Message as LLMMessage

        cues = cues or {"names": [], "preferences": [], "explicit": []}
        candidate_lines: list[str] = []
        for node_id, node in candidates.items():
            body_snippet = str(node.get("body") or "")[: self._thresholds.reference_candidate_body_chars].replace("\n", " ")
            candidate_lines.append(
                json.dumps(
                    {
                        "id": node_id,
                        "title": node.get("title", ""),
                        "tags": node.get("tags", []),
                        "created_at": node.get("created_at", ""),
                        "body": body_snippet,
                        "signals": node.get("_sig", []),
                    },
                    ensure_ascii=False,
                )
            )
        turns_block = []
        for turn in (recent_turns or [])[-4:]:
            role = str(turn.get("role") or "?")
            content = _clean_text(str(turn.get("content") or ""))[:120]
            if content:
                turns_block.append(f"- {role}: {content}")

        user_content = "\n".join(
            [
                f'当前用户消息："{message}"',
                f"当前 chat 句柄线索：{chat_id or '（无）'}",
                f"来源路由线索：{source_hint or '（无）'}",
                "从当前消息提取的线索：",
                f"- names: {cues.get('names', [])}",
                f"- preferences: {cues.get('preferences', [])}",
                f"- explicit: {cues.get('explicit', [])}",
                f"- source_traits: {cues.get('source_traits', [])}",
                "最近交互片段：",
                "\n".join(turns_block) if turns_block else "（无）",
                "当前 chat 连续性：",
                chat_continuity[:600] or "（无）",
                "当前对象跨 chat 交互连续性：",
                interlocutor_continuity[:600] or "（无）",
                "候选交互对象画像：",
                "[\n" + ",\n".join(candidate_lines) + "\n]" if candidate_lines else "[]",
            ]
        )

        try:
            raw = await self._provider.chat(
                [
                    LLMMessage(role="system", content=_SPEAKER_REASON_SYSTEM),
                    LLMMessage(role="user", content=user_content),
                ],
                temperature=self._reason_temperature,
            )
        except Exception as exc:
            err_text = str(exc) or repr(exc)
            self._last_llm_error = err_text
            self._last_llm_error_code = self._classify_error_code(err_text)
            _log.warning("[reference] 当前说话人识别失败，降级为本地评分: %s", exc)
            return {}

        self._last_llm_error = ""
        self._last_llm_error_code = ""
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return {}
        try:
            payload = json.loads(raw[start: end + 1])
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _fallback_resolve_speaker(
        self,
        candidates: dict[str, dict[str, Any]],
        *,
        cues: dict[str, list[str]],
        chat_id: str = "",
        cached_profile_id: str = "",
    ) -> ResolvedSpeaker | None:
        best: tuple[float, dict[str, Any]] | None = None
        lowered_names = [name.lower() for name in cues.get("names", [])]
        handle_tag = _speaker_handle_tag(chat_id) if chat_id else ""
        for node_id, node in candidates.items():
            score = float(node.get("score") or 0.0)
            signal_types = list(node.get("_sig") or [])
            score += len(set(signal_types)) * 0.12
            title_body = f"{node.get('title', '')} {node.get('body', '')}".lower()
            tags = {str(tag) for tag in (node.get("tags") or [])}
            if cached_profile_id and node_id == cached_profile_id:
                score += 0.22
            if lowered_names and any(name and name in title_body for name in lowered_names):
                score += 0.35
            if chat_id and (handle_tag in tags or chat_id.lower() in title_body):
                score += 0.26
            if cues.get("source_traits") and any(trait.split("=", 1)[-1] in title_body for trait in cues.get("source_traits", [])):
                score += 0.18
            if best is None or score > best[0]:
                best = (score, node)

        if best is None:
            return None
        confidence = min(best[0], self._thresholds.reference_local_confidence_cap)
        if confidence < self._thresholds.reference_min_confidence:
            return None
        node = best[1]
        return ResolvedSpeaker(
            node_id=str(node.get("id") or ""),
            title=str(node.get("title") or _fallback_speaker_title(chat_id)),
            confidence=round(confidence, 2),
            snippet=str(node.get("body") or "")[: self._thresholds.reference_entity_snippet_chars],
            evidence=[f"本地多线索命中：{', '.join(node.get('_sig', []))}"] if node.get("_sig") else [],
            relationship_note="多线索画像匹配",
            signal_types=list(node.get("_sig") or []),
            provisional=False,
            search_anchors=[str(node.get("title") or "")],
            source_traits=list(cues.get("source_traits") or []),
        )

    def _build_provisional_speaker(
        self,
        *,
        message: str,
        cues: dict[str, list[str]],
        chat_id: str = "",
        hint_title: str = "",
    ) -> ResolvedSpeaker | None:
        display_name = (cues.get("names") or [])[0] if cues.get("names") else hint_title.strip()
        if not display_name:
            display_name = _fallback_speaker_title(chat_id)
        if not display_name and not chat_id:
            return None
        seed = "|".join(part for part in (chat_id, display_name, message[:48]) if part)
        node_id = f"interlocutor-profile-{_digest_text(seed or message or display_name)}"
        evidence: list[str] = []
        if cues.get("names"):
            evidence.append(f"当前消息出现自称：{cues['names'][0]}")
        if cues.get("preferences"):
            evidence.append(f"当前消息出现偏好：{cues['preferences'][0]}")
        if chat_id:
            evidence.append("当前 chat 只能提供延续线索，不能单独定身份")
        snippet_parts = [item for item in [*cues.get("preferences", []), *cues.get("explicit", [])] if item]
        confidence = 0.72 if cues.get("names") else 0.46 if chat_id else 0.38
        return ResolvedSpeaker(
            node_id=node_id,
            title=display_name,
            confidence=round(confidence, 2),
            snippet="；".join(snippet_parts[:2]) or message[: self._thresholds.reference_entity_snippet_chars],
            evidence=evidence[:3],
            relationship_note="当前轮形成临时交互对象画像",
            signal_types=["self_intro" if cues.get("names") else "provisional"],
            provisional=True,
            search_anchors=[display_name, *cues.get("preferences", [])[:1]],
            source_traits=list(cues.get("source_traits") or []),
        )

    async def resolve_current_speaker(
        self,
        message: str,
        semantic: "SemanticMemory",
        *,
        chat_id: str = "",
        recent_turns: list[dict[str, Any]] | None = None,
        chat_continuity: str = "",
        interlocutor_continuity: str = "",
        cached_profile_id: str = "",
        source_hint: str = "",
    ) -> ResolvedSpeaker | None:
        if not _clean_text(message):
            return None

        candidates, cues = self._retrieve_speaker_candidates(
            message,
            semantic,
            chat_id=chat_id,
            recent_turns=recent_turns,
            chat_continuity=chat_continuity,
            source_hint=source_hint,
            cached_profile_id=cached_profile_id,
        )

        llm_result: dict[str, Any] = {}
        if self._provider is not None and (candidates or any(cues.values()) or chat_continuity.strip() or interlocutor_continuity.strip()):
            llm_result = await self._llm_reason_speaker(
                message,
                candidates=candidates,
                recent_turns=recent_turns,
                chat_continuity=chat_continuity,
                interlocutor_continuity=interlocutor_continuity,
                chat_id=chat_id,
                source_hint=source_hint,
                cues=cues,
            )

        if llm_result:
            node_id = str(llm_result.get("node_id") or "").strip()
            confidence = round(float(llm_result.get("confidence") or 0.0), 2)
            evidence = [str(item).strip() for item in (llm_result.get("evidence") or []) if str(item).strip()][:3]
            note = str(llm_result.get("relationship_note") or "").strip()
            if node_id in candidates and confidence >= self._thresholds.reference_min_confidence:
                node = candidates[node_id]
                title = str(llm_result.get("display_name") or node.get("title") or _fallback_speaker_title(chat_id)).strip()
                anchors = [title, *cues.get("names", [])[:1], *cues.get("preferences", [])[:1]]
                return ResolvedSpeaker(
                    node_id=node_id,
                    title=title,
                    confidence=confidence,
                    snippet=str(node.get("body") or "")[: self._thresholds.reference_entity_snippet_chars],
                    evidence=evidence,
                    relationship_note=note,
                    signal_types=list(node.get("_sig") or []),
                    provisional=bool(llm_result.get("provisional")),
                    search_anchors=[anchor for anchor in anchors if anchor],
                    source_traits=list(cues.get("source_traits") or []),
                )
            if node_id == "NEW":
                provisional = self._build_provisional_speaker(
                    message=message,
                    cues=cues,
                    chat_id=chat_id,
                    hint_title=str(llm_result.get("display_name") or "").strip(),
                )
                if provisional is not None:
                    provisional.confidence = max(provisional.confidence, confidence or provisional.confidence)
                    if evidence:
                        provisional.evidence = evidence
                    if note:
                        provisional.relationship_note = note
                return provisional
            if node_id == "UNKNOWN":
                return None

        resolved = self._fallback_resolve_speaker(
            candidates,
            cues=cues,
            chat_id=chat_id,
            cached_profile_id=cached_profile_id,
        )
        if resolved is not None:
            return resolved
        return self._build_provisional_speaker(message=message, cues=cues, chat_id=chat_id)

    async def remember_speaker(
        self,
        speaker: ResolvedSpeaker,
        semantic: "SemanticMemory",
        task_store: "TaskStore | None",
        *,
        message: str,
        chat_id: str = "",
        task_id: str | int | None = None,
        source_hint: str = "",
    ) -> None:
        if not speaker.node_id:
            return

        from datetime import UTC, datetime
        from store.semantic import MemoryNode

        cues = self._extract_identity_cues(message, chat_id=chat_id, source_hint=source_hint)
        existing = semantic.get(speaker.node_id)
        merged_lines: list[str] = []
        if existing is not None and existing.body.strip():
            merged_lines.extend([line.strip() for line in existing.body.splitlines() if line.strip()])
        additions = [
            f"画像摘要: {speaker.snippet}" if speaker.snippet else "",
            f"识别判断: {speaker.relationship_note}" if speaker.relationship_note else "",
            *[f"识别依据: {item}" for item in speaker.evidence[:3]],
            *[f"偏好线索: {item}" for item in cues.get("preferences", [])[:2]],
            *[f"显式记忆要求: {item}" for item in cues.get("explicit", [])[:2]],
            *[f"来源特征: {item}" for item in cues.get("source_traits", [])[:5]],
            *([f"已见 chat 线索: {chat_id}"] if chat_id else []),
        ]
        for line in additions:
            normalized = _clean_text(line)
            if normalized and normalized not in merged_lines:
                merged_lines.append(normalized)

        tags = set(existing.tags if existing is not None else [])
        tags.update({"interlocutor_profile", f"interlocutor:{speaker.node_id}"})
        if chat_id:
            tags.update({f"chat:{chat_id}", _speaker_handle_tag(chat_id)})
        for alias in cues.get("names", [])[:2]:
            tags.add(f"alias:{alias}")
        for trait in cues.get("source_traits", [])[:5]:
            tags.add(trait)

        semantic.upsert(MemoryNode(
            id=speaker.node_id,
            kind="interlocutor",
            title=speaker.title or (existing.title if existing is not None else _fallback_speaker_title(chat_id)),
            body="\n".join(merged_lines[-12:]),
            activation=max(existing.activation if existing is not None else 0.0, max(0.55, speaker.confidence)),
            importance=max(existing.importance if existing is not None else 0.0, 0.58 if not speaker.provisional else 0.45),
            valence=existing.valence if existing is not None else 0.5,
            tags=sorted(tags),
            source=(existing.source if existing is not None and existing.source else "interlocutor_profile"),
            created_at=existing.created_at if existing is not None else datetime.now(UTC).isoformat(),
        ))

        if task_store is None:
            return
        if chat_id:
            await task_store.set_fact(f"chat:{chat_id}:interlocutor_profile_id", speaker.node_id, scope="profile")
        if task_id is not None:
            await task_store.set_fact(f"task:{task_id}:interlocutor_profile_id", speaker.node_id, scope="profile")
        await task_store.set_fact(f"interlocutor:{speaker.node_id}:display_name", speaker.title, scope="profile")
        if chat_id:
            await task_store.set_fact(f"interlocutor:{speaker.node_id}:handle:{_digest_text(chat_id)}", chat_id, scope="profile")
        for pref in cues.get("preferences", [])[:3]:
            await task_store.set_fact(f"interlocutor:{speaker.node_id}:preference:{_digest_text(pref)}", pref, scope="profile")
        for explicit in cues.get("explicit", [])[:3]:
            await task_store.set_fact(f"interlocutor:{speaker.node_id}:explicit:{_digest_text(explicit)}", explicit, scope="profile")
        for trait in cues.get("source_traits", [])[:5]:
            await task_store.set_fact(f"interlocutor:{speaker.node_id}:source_trait:{_digest_text(trait)}", trait, scope="profile")

    # ── 主入口（异步）───────────────────────────────────────────────────────

    async def resolve(
        self,
        message: str,
        semantic: "SemanticMemory",
        episodic: "EpisodicMemory",
    ) -> list["ResolvedEntity"]:
        """两阶段：本地召回候选 → LLM 推理判断 → ResolvedEntity 列表。

        Provider 不可用时：自动降级为本地评分，保持可用性。
        """
        if not message or not message.strip():
            return []

        sigs = self.extract_signals(message)
        candidates = self._retrieve_candidates(message, sigs, semantic, episodic)
        if not candidates:
            return []

        # LLM 推理
        llm_results: list[dict[str, Any]] = []
        if self._provider is not None:
            llm_results = await self._llm_reason(message, candidates)

        # 构造 ResolvedEntity 列表
        entities: list[ResolvedEntity] = []

        if llm_results:
            # LLM 推理路径
            for item in llm_results:
                nid = str(item.get("node_id", ""))
                if nid not in candidates:
                    continue
                confidence = float(item.get("confidence", 0.0))
                if confidence < self._thresholds.reference_min_confidence:
                    continue
                nd = candidates[nid]
                entities.append(ResolvedEntity(
                    node_id=nid,
                    title=nd.get("title", nid),
                    kind=nd.get("kind", "unknown"),
                    confidence=round(confidence, 2),
                    snippet=nd.get("body", "")[: self._thresholds.reference_entity_snippet_chars],
                    signal_types=nd.get("_sig", []),
                    relationship_note=str(item.get("relationship_note", "")),
                ))
        else:
            # 降级路径：本地评分（简单计数信号数）
            for nid, nd in candidates.items():
                sigs_hit = nd.get("_sig", [])
                base = self._thresholds.reference_local_signal_base + len(set(sigs_hit)) * self._thresholds.reference_local_signal_step
                if base < self._thresholds.reference_min_confidence:
                    continue
                entities.append(ResolvedEntity(
                    node_id=nid,
                    title=nd.get("title", nid),
                    kind=nd.get("kind", "unknown"),
                    confidence=round(min(base, self._thresholds.reference_local_confidence_cap), 2),
                    snippet=nd.get("body", "")[: self._thresholds.reference_entity_snippet_chars],
                    signal_types=sigs_hit,
                    relationship_note="（本地评分，LLM 不可用）",
                ))

        entities.sort(key=lambda e: e.confidence, reverse=True)
        return entities[: self._thresholds.reference_entity_section_limit]

    # ── 格式化注入 entity_section ────────────────────────────────────────────

    @staticmethod
    def format_section(entities: list["ResolvedEntity"]) -> str:
        if not entities:
            return "（无可链接的历史实体）"
        lines = ["从记忆中识别到以下相关实体（LLM 推理确认，按置信度排列）："]
        for e in entities:
            note = f" — {e.relationship_note}" if e.relationship_note and "本地评分" not in e.relationship_note else ""
            lines.append(
                f"- [{e.kind}] {e.title}（confidence:{e.confidence:.2f}{note}）"
            )
            if e.snippet:
                lines.append(f"  {e.snippet}")
        return "\n".join(lines)

    @staticmethod
    def format_speaker_section(speaker: "ResolvedSpeaker | None") -> str:
        if speaker is None:
            return "（当前轮尚未稳定识别当前交互对象，先依赖本轮消息与 chat 连续性）"
        status = "临时画像" if speaker.provisional else "稳定画像"
        lines = [f"当前交互对象候选: {speaker.title}（confidence:{speaker.confidence:.2f}，{status}）"]
        if speaker.relationship_note:
            lines.append(f"判断: {speaker.relationship_note}")
        for item in speaker.evidence[:3]:
            lines.append(f"- {item}")
        if speaker.snippet:
            lines.append(f"画像记忆: {speaker.snippet}")
        return "\n".join(lines)
