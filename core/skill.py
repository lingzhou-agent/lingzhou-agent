"""core/skill.py — 技能系统（认知护栏）。

技能不是工具：工具是执行能力，技能是注入 LLM 判断前的认知框架。
当前情境匹配的技能以文本形式拼入 judgment bundle，引导而非强制。

设计原则：
- 技能本身可以被 evolution 进化（本文件理论上可热替换）
- 支持两种载体：workspace/skills/*.md 与 workspace/skills/<name>/SKILL.md
- 最多注入 3 个技能，避免 prompt 被护栏淹没
- 自定义技能匹配不仅看内部状态，也看 user_message / task 文本触发
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger("lingzhou.skill")


@dataclass
class Skill:
    name: str
    description: str      # 对人类的一句话说明（日志 / debug 用）
    guidance: str         # 注入 LLM 的引导文本
    tags: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    source_path: str = ""


# ── 五个内置技能 ──────────────────────────────────────────────────────────────

_BUILTIN_SKILLS: list[Skill] = [
    Skill(
        name="runtime.bootstrap",
        description="冷启动阶段：身份内容已注入 WM，无需 file.read",
        guidance=(
            "你正处于冷启动阶段。"
            "SOUL.md、BOOTSTRAP.md、IDENTITY.md 的内容已自动注入工作记忆（kind=bootstrap_identity），"
            "直接从工作记忆中读取，不要再用 file.read 重复读取这些文件。"
            "请根据工作记忆中的身份信息，创建一个有意义的自驱任务。"
        ),
        tags=["bootstrap", "cold_start"],
        triggers=["冷启动", "bootstrap", "启动"],
    ),
    Skill(
        name="provider.integration",
        description="工具调用前确认参数，失败后分析原因再重试",
        guidance=(
            "调用工具前，确认参数名和类型符合工具描述。"
            "工具调用失败时，先分析错误原因再重试——不要盲目重试相同参数。"
            "如果某个文件不存在（FileNotFound），不要反复尝试读取，换一个策略。"
        ),
        tags=["act", "tool_call"],
        triggers=["工具失败", "参数错误", "file not found", "调用失败"],
    ),
    Skill(
        name="task.continuity",
        description="有 next_step 时优先推进而非创建新任务",
        guidance=(
            "当前任务有明确的 next_step，优先推进它，不要分散注意力创建新任务。"
            "每一步完成后立即更新 next_step，保持任务状态连续可追溯。"
        ),
        tags=["continuity", "task"],
        triggers=["next_step", "继续推进", "当前任务"],
    ),
    Skill(
        name="evidence-first-change",
        description="修改前先读取确认，修改后验证结果",
        guidance=(
            "任何写操作（写文件、执行命令）前，先读取当前状态确认前提成立。"
            "操作完成后再次读取验证结果。不确定时，选择范围更小、可逆的操作。"
        ),
        tags=["caution", "verification"],
        triggers=["修改", "写入", "验证", "证据"],
    ),
    Skill(
        name="failure.reflection",
        description="连续失败时停下来分析根因而非重试",
        guidance=(
            "你已经遇到了多次失败。停下来，不要重复相同操作。"
            "分析根因：是参数错误？前提条件不满足？工具本身有问题？"
            "选择不同策略，或向用户报告当前困境请求帮助。"
        ),
        tags=["failure", "reflection"],
        triggers=["失败", "报错", "根因", "重试"],
    ),
]


def _split_frontmatter(content: str) -> tuple[dict[str, str], str]:
    if not content.startswith("---"):
        return {}, content.strip()
    lines = content.splitlines()
    end = -1
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end = i
            break
    if end <= 0:
        return {}, content.strip()

    raw = lines[1:end]
    body = "\n".join(lines[end + 1:]).strip()
    meta: dict[str, str] = {}
    i = 0
    while i < len(raw):
        line = raw[i]
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, value = m.group(1), m.group(2).strip()
        if value in {"|", ">"}:
            i += 1
            block: list[str] = []
            while i < len(raw):
                nxt = raw[i]
                if nxt and not nxt.startswith((" ", "\t")) and re.match(r"^[A-Za-z_][\w-]*:\s*", nxt):
                    break
                block.append(nxt.lstrip())
                i += 1
            meta[key] = "\n".join(block).strip()
            continue
        meta[key] = value.strip().strip('"\'')
        i += 1
    return meta, body


def _parse_listish(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    raw = raw.strip("[]")
    parts = re.split(r"[,，、;；/／\n|]+", raw)
    return [p.strip().strip('"\'') for p in parts if p.strip()]


def _extract_trigger_text(description: str, meta: dict[str, str]) -> list[str]:
    triggers: list[str] = []
    for key in ("trigger", "triggers"):
        if key in meta:
            triggers.extend(_parse_listish(meta[key]))
    m = re.search(r"(?:Triggers?|触发(?:词|器|条件)?)[：:]\s*(.+)$", description, flags=re.IGNORECASE | re.DOTALL)
    if m:
        triggers.extend(_parse_listish(m.group(1)))
    return [t for t in dict.fromkeys(t.strip() for t in triggers if t.strip())]


def _description_without_trigger_tail(description: str) -> str:
    return re.sub(r"(?:Triggers?|触发(?:词|器|条件)?)[：:].*$", "", description, flags=re.IGNORECASE | re.DOTALL).strip()


def _text_terms(text: str, *, expand_ngrams: bool = False) -> set[str]:
    text = (text or "").lower()
    terms: set[str] = set(re.findall(r"[a-z0-9_+-]{3,}", text))
    for seq in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.add(seq)
        if not expand_ngrams:
            continue
        max_n = min(4, len(seq))
        for n in range(2, max_n + 1):
            for i in range(0, len(seq) - n + 1):
                terms.add(seq[i:i + n])
    return terms


def _trim_guidance(text: str, limit: int = 1600) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    trimmed = text[:limit]
    if "\n" in trimmed:
        trimmed = trimmed.rsplit("\n", 1)[0]
    return trimmed.rstrip() + "\n\n[技能内容已截断，保留前段核心 guidance]"


_LOW_SIGNAL_TERMS = {
    '什么', '为什么', '怎么', '如何', '可以', '应该', '需要', '当前', '这个', '那个', '这里',
}


def _custom_context_boost(skill: Skill, hay: str) -> float:
    rules: dict[str, list[str]] = {
        'interaction': ['好奇', '分歧', '你觉得', '对吗', '确认', '提问', '追问', '真正想要', '理解语境'],
        'proactive-work': ['接下来', '下一步', '自己判断', '自主决定', '往前推进', '完成任务后', '等回复'],
        'self-monitoring': ['日志', '异常', '偏了', '漂移', 'edit 失败', '工具执行失败', '文件异常'],
        'error-handling': ['timeout', 'permission', 'denied', '被拒绝', '报错', '错误', 'exec'],
    }
    phrases = rules.get(skill.name, [])
    score = 0.0
    for phrase in phrases:
        if phrase.lower() in hay:
            score += 1.6 if len(phrase) >= 3 else 0.8
    return score


def _context_score(skill: Skill, context_text: str) -> float:
    hay = (context_text or "").lower().strip()
    if not hay:
        return 0.0

    score = 0.0
    hay_terms = _text_terms(hay, expand_ngrams=True)
    seen: set[str] = set()
    phrases = list(skill.triggers) + [t for t in skill.tags if t != "custom"]
    phrases += re.split(r"[-_.]", skill.name)

    for phrase in phrases:
        p = phrase.lower().strip()
        if len(p) < 2 or p in seen:
            continue
        seen.add(p)
        if p in _LOW_SIGNAL_TERMS:
            continue
        if p in hay:
            score += 3.0 if len(p) >= 4 else 1.2
            continue
        p_terms = {t for t in _text_terms(p, expand_ngrams=True) if t not in _LOW_SIGNAL_TERMS}
        shared = len(p_terms & hay_terms)
        if shared:
            score += min(shared, 4) * 0.55

    desc = _description_without_trigger_tail(skill.description).lower()
    desc_terms = {t for t in _text_terms(desc, expand_ngrams=False) if t not in _LOW_SIGNAL_TERMS}
    overlap = len(desc_terms & hay_terms)
    score += min(overlap, 8) * 0.28
    score += _custom_context_boost(skill, hay)
    return score


# ── 技能注册表 ────────────────────────────────────────────────────────────────

class SkillRegistry:
    """技能注册表：内置技能 + workspace 自定义技能。"""

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._skills: list[Skill] = list(_BUILTIN_SKILLS)
        if skills_dir is not None:
            loaded = self._load_from_dir(skills_dir)
            if loaded:
                _log.info("[skill] 从 %s 加载了 %d 个自定义技能", skills_dir, loaded)

    def _iter_skill_files(self, skills_dir: Path) -> list[Path]:
        files: list[Path] = []
        for md in sorted(skills_dir.glob("*.md")):
            if md.name != "SKILL.md":
                files.append(md)
        for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
            files.append(skill_md)
        return files

    def _load_from_dir(self, skills_dir: Path) -> int:
        if not skills_dir.exists():
            return 0
        loaded = 0
        for md_file in self._iter_skill_files(skills_dir):
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                meta, body = _split_frontmatter(content)
                name = meta.get("name") or (md_file.parent.name if md_file.name == "SKILL.md" else md_file.stem)
                description = meta.get("description") or f"自定义技能: {name}"
                tags = _parse_listish(meta.get("tags", "")) or ["custom"]
                triggers = _extract_trigger_text(description, meta)
                guidance = _trim_guidance(body or content)
                if not guidance:
                    continue
                skill = Skill(
                    name=name,
                    description=description,
                    guidance=guidance,
                    tags=tags,
                    triggers=triggers,
                    source_path=str(md_file),
                )
                existing = next((i for i, s in enumerate(self._skills) if s.name == name), -1)
                if existing >= 0:
                    self._skills[existing] = skill
                    _log.debug("[skill] 覆盖内置技能: %s", name)
                else:
                    self._skills.append(skill)
                loaded += 1
            except Exception as exc:
                _log.warning("[skill] 加载 %s 失败: %s", md_file, exc)
        return loaded

    def all_skills(self) -> list[Skill]:
        return list(self._skills)

    def match_for_context(self, **_kwargs: Any) -> list[Skill]:
        """返回全部已注册技能，交由 LLM 自行判断适用哪些。

        Python 层不再打分/排序/截断——机械式评分与误导无异。
        """
        skills = list(self._skills)
        _log.debug("[skill.match] injecting=%d: %s", len(skills), [s.name for s in skills])
        return skills
