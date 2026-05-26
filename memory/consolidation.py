"""WM consolidation routing: short-term traces -> episodic, semantic, durable facts.

This module keeps consolidation policy out of the main loop so memory behavior can
evolve as a coherent mechanism instead of scattered one-off writes.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from store.semantic import MemoryNode

_DEFAULT_PROMOTION_SEMANTIC_KINDS = (
    "self_awareness",
    "behavior_sense",
    "task_reflection",
    "meta_reflection",
    "task_replan",
    "routing_guard",
    "task_result",
    "progress_crystal",
    "execute_result",
    "run_monitor",
    "probe_result",
    "subagent_result",
    "skill_activation",
    "skill_evolution",
    "skill_synthesis",
    "self_drive",
    "crash_recovery",
)

_DEFAULT_PROMOTION_FACT_KINDS = ("user_message",)

_KIND_TO_MEMORY_KIND = {
    "self_awareness": "self_model_signal",
    "behavior_sense": "self_model_signal",
    "task_reflection": "consolidated_insight",
    "meta_reflection": "consolidated_insight",
    "task_replan": "plan_revision",
    "routing_guard": "control_rule",
    "task_result": "task_progress",
    "progress_crystal": "task_progress",
    "execute_result": "task_progress",
    "run_monitor": "task_progress",
    "probe_result": "sensor_snapshot",
    "subagent_result": "delegated_result",
    "skill_activation": "learned_skill",
    "skill_evolution": "learned_skill",
    "skill_synthesis": "learned_skill",
    "self_drive": "drive_trace",
    "crash_recovery": "failure_trace",
}

_KIND_IMPORTANCE_BONUS = {
    "self_model_signal": 0.12,
    "consolidated_insight": 0.16,
    "plan_revision": 0.1,
    "control_rule": 0.14,
    "task_progress": 0.08,
    "sensor_snapshot": 0.05,
    "learned_skill": 0.12,
    "failure_trace": 0.1,
}

_TITLE_PREFIX = {
    "self_model_signal": "self-model",
    "consolidated_insight": "insight",
    "plan_revision": "plan",
    "control_rule": "rule",
    "task_progress": "progress",
    "sensor_snapshot": "probe",
    "delegated_result": "subagent",
    "learned_skill": "skill",
    "drive_trace": "drive",
    "failure_trace": "failure",
    "working_trace": "trace",
}


@dataclass(frozen=True)
class ConsolidatedFact:
    key: str
    value: str
    scope: str = "profile"


@dataclass
class ConsolidationPlan:
    episodic_summary: str
    semantic_nodes: list[MemoryNode]
    facts: list[ConsolidatedFact]


def build_consolidation_plan(
    items: list[dict[str, Any]],
    *,
    task_id: str | None,
    task_title: str | None,
    memory_cfg: Any,
    emotion_valence: float,
    now: datetime | None = None,
) -> ConsolidationPlan:
    summary_lines: list[str] = []
    semantic_nodes: list[MemoryNode] = []
    fact_map: dict[str, ConsolidatedFact] = {}
    now = now or datetime.now(UTC)

    semantic_allow = {
        str(v).strip()
        for v in getattr(memory_cfg, "promotion_semantic_kinds", _DEFAULT_PROMOTION_SEMANTIC_KINDS)
        if str(v).strip()
    }
    fact_kinds = {
        str(v).strip()
        for v in getattr(memory_cfg, "promotion_fact_kinds", _DEFAULT_PROMOTION_FACT_KINDS)
        if str(v).strip()
    }
    priority_threshold = float(getattr(memory_cfg, "promotion_priority_threshold", 0.78))
    max_nodes = max(0, int(getattr(memory_cfg, "promotion_max_nodes_per_consolidation", 6)))

    for item in items:
        kind = str(item.get("kind") or "").strip()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        summary_lines.append(f"- [{kind or 'unknown'}] {content}")

        if kind in fact_kinds:
            for fact in _extract_user_facts(content):
                fact_map[fact.key] = fact

        if len(semantic_nodes) >= max_nodes:
            continue
        priority = _coerce_priority(item.get("priority"))
        if priority < priority_threshold and kind not in semantic_allow:
            continue
        node = _build_semantic_node(
            kind=kind,
            content=content,
            priority=priority,
            task_id=task_id,
            task_title=task_title,
            memory_cfg=memory_cfg,
            emotion_valence=emotion_valence,
            now=now,
        )
        if node is not None:
            semantic_nodes.append(node)

    return ConsolidationPlan(
        episodic_summary="\n".join(summary_lines),
        semantic_nodes=semantic_nodes,
        facts=list(fact_map.values()),
    )


def merge_promoted_node(existing: MemoryNode | None, incoming: MemoryNode, *, memory_cfg: Any) -> MemoryNode:
    if existing is None:
        return incoming

    body_max_chars = max(120, int(getattr(memory_cfg, "promotion_body_max_chars", 1200)))
    reinforce_delta = float(getattr(memory_cfg, "promotion_reinforce_delta", 0.05))
    merged_body = existing.body or ""
    incoming_body = (incoming.body or "").strip()
    if incoming_body and incoming_body not in merged_body:
        merged_body = f"{merged_body.rstrip()}\n\n---\n\n{incoming_body}".strip()
    if len(merged_body) > body_max_chars:
        merged_body = merged_body[-body_max_chars:]

    return MemoryNode(
        id=existing.id,
        kind=existing.kind or incoming.kind,
        title=existing.title or incoming.title,
        body=merged_body,
        activation=min(1.0, max(existing.activation, incoming.activation) + reinforce_delta),
        valence=incoming.valence,
        importance=min(1.0, max(existing.importance, incoming.importance)),
        tags=sorted({*(existing.tags or []), *(incoming.tags or [])}),
        source=incoming.source or existing.source,
        created_at=existing.created_at or incoming.created_at,
    )


def current_week_key(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.strftime("%G-W%V")


def build_daily_summary_node(
    recent_daily_text: str,
    *,
    week_key: str | None = None,
    memory_cfg: Any,
    emotion_valence: float,
    existing: MemoryNode | None = None,
    now: datetime | None = None,
) -> MemoryNode | None:
    text = (recent_daily_text or "").strip()
    if not text:
        return None
    now = now or datetime.now(UTC)
    week_key = (week_key or current_week_key(now)).strip()
    max_chars = max(200, int(getattr(memory_cfg, "daily_summary_max_chars", 1800)))
    body = text[-max_chars:]
    return MemoryNode(
        id=f"daily-summary-{week_key}",
        kind="daily_summary",
        title=f"[{week_key}] recent daily summary",
        body=body,
        activation=float(getattr(memory_cfg, "daily_summary_activation", 0.78)),
        valence=max(0.0, min(1.0, float(emotion_valence))),
        importance=float(getattr(memory_cfg, "daily_summary_importance", 0.82)),
        tags=["daily_summary", week_key],
        source="daily_consolidation",
        created_at=(existing.created_at if existing is not None else now.isoformat()),
    )


def _build_semantic_node(
    *,
    kind: str,
    content: str,
    priority: float,
    task_id: str | None,
    task_title: str | None,
    memory_cfg: Any,
    emotion_valence: float,
    now: datetime,
) -> MemoryNode | None:
    if kind == "user_message":
        return None

    cleaned = _clean_content(content)
    min_chars = max(1, int(getattr(memory_cfg, "promotion_min_chars", 24)))
    if _content_length_units(cleaned) < min_chars:
        return None

    memory_kind = _KIND_TO_MEMORY_KIND.get(kind, "working_trace")
    importance_bonus = _KIND_IMPORTANCE_BONUS.get(memory_kind, 0.0)
    importance = min(0.98, max(priority, 0.45) + importance_bonus)
    activation = min(0.96, 0.42 + priority * 0.45 + min(importance_bonus, 0.12))
    body_max_chars = max(120, int(getattr(memory_cfg, "promotion_body_max_chars", 1200)))
    body = cleaned[:body_max_chars]
    digest = hashlib.sha1(f"{memory_kind}|{kind}|{cleaned}".encode("utf-8")).hexdigest()[:16]
    prefix = _TITLE_PREFIX.get(memory_kind, "trace")
    scope_label = f"task#{task_id}" if task_id else "free"
    snippet = cleaned.replace("\n", " ")[:72]
    title = f"[{prefix}] {scope_label} {snippet}".strip()
    tags = ["wm_promoted", memory_kind, kind]
    if task_id:
        tags.append(f"task:{task_id}")
    if task_title:
        tags.append(f"task-title:{_tag_safe(task_title[:32])}")

    return MemoryNode(
        id=f"wm-promoted-{digest}",
        kind=memory_kind,
        title=title,
        body=body,
        activation=activation,
        valence=max(0.0, min(1.0, float(emotion_valence))),
        importance=importance,
        tags=tags,
        source="wm_consolidation",
        created_at=now.isoformat(),
    )


def _extract_user_facts(content: str) -> list[ConsolidatedFact]:
    text = _clean_content(content)
    if not text:
        return []

    facts: dict[str, ConsolidatedFact] = {}
    for pattern in (
        re.compile(r"(?:我叫|我的名字是|你可以叫我|请叫我|以后叫我|下次叫我|就叫我)\s*[:：]?\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,24})"),
    ):
        m = pattern.search(text)
        if m:
            facts["user:name"] = ConsolidatedFact("user:name", m.group(1).strip())
            break

    for sentence in _split_sentences(text):
        normalized = sentence.strip()
        if not normalized:
            continue
        if any(token in normalized for token in ("我喜欢", "我偏好", "我更喜欢", "请用", "以后用")):
            digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
            facts[f"user:preference:{digest}"] = ConsolidatedFact(
                f"user:preference:{digest}",
                normalized,
            )
        if any(token in normalized for token in ("记住", "别忘了", "请记得")):
            digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
            facts[f"user:explicit:{digest}"] = ConsolidatedFact(
                f"user:explicit:{digest}",
                normalized,
            )

    return list(facts.values())


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？!?；;\n]+", text)
    return [part.strip(" ,，。；;!！?？") for part in parts if part and part.strip()]


def _clean_content(content: str) -> str:
    text = re.sub(r"^\[[^\]]+\]\s*", "", content.strip())
    return re.sub(r"\s+", " ", text).strip()


def _content_length_units(text: str) -> int:
    # CJK traces are often semantically dense at shorter character counts.
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
    return len(text) + cjk_chars


def _coerce_priority(raw: Any) -> float:
    try:
        value = float(raw)
    except Exception:
        value = 0.0
    return max(0.0, min(1.0, value))


def _tag_safe(text: str) -> str:
    return re.sub(r"\s+", "-", re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "", text)).strip("-") or "task"