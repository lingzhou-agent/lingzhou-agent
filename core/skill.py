"""core/skill.py — 技能系统（认知护栏）。

技能不是工具：工具是执行能力，技能是注入 LLM 判断前的认知框架。
当前情境匹配的技能以文本形式拼入 judgment bundle，引导而非强制。

设计原则：
- 技能本身可以被 evolution 进化（本文件理论上可热替换）
- MatchForContext 是贝叶斯证据累积：多个触发条件 → 最相关的护栏先激活
- 最多注入 3 个技能，避免 prompt 被护栏淹没
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.perception import EthosState


@dataclass
class Skill:
    name: str
    description: str      # 对人类的一句话说明（日志 / debug 用）
    guidance: str         # 注入 LLM 的引导文本（简洁，不超过 3 句）
    tags: list[str] = field(default_factory=list)


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
    ),
    Skill(
        name="task.continuity",
        description="有 next_step 时优先推进而非创建新任务",
        guidance=(
            "当前任务有明确的 next_step，优先推进它，不要分散注意力创建新任务。"
            "每一步完成后立即更新 next_step，保持任务状态连续可追溯。"
        ),
        tags=["continuity", "task"],
    ),
    Skill(
        name="evidence-first-change",
        description="修改前先读取确认，修改后验证结果",
        guidance=(
            "任何写操作（写文件、执行命令）前，先读取当前状态确认前提成立。"
            "操作完成后再次读取验证结果。不确定时，选择范围更小、可逆的操作。"
        ),
        tags=["caution", "verification"],
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
    ),
]


# ── 技能注册表 ────────────────────────────────────────────────────────────────

class SkillRegistry:
    """技能注册表：内置技能 + 从 workspace/skills/*.md 动态加载的自定义技能。

    文件格式（每个 .md 文件是一个技能）：
      - 可选 YAML frontmatter（--- 包裹）：description、tags 字段
      - frontmatter 后的正文即为注入 LLM 的 guidance 文本
      - 文件名（不含扩展名）作为技能 name
      - 自定义技能与内置技能同名时覆盖内置技能
    """

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._skills: list[Skill] = list(_BUILTIN_SKILLS)
        if skills_dir is not None:
            loaded = self._load_from_dir(skills_dir)
            if loaded:
                _log.info("[skill] 从 %s 加载了 %d 个自定义技能", skills_dir, loaded)

    def _load_from_dir(self, skills_dir: Path) -> int:
        """从目录加载 *.md 技能文件，返回加载数量。"""
        if not skills_dir.exists():
            return 0
        loaded = 0
        for md_file in sorted(skills_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                name = md_file.stem
                description = f"自定义技能: {name}"
                tags: list[str] = ["custom"]
                guidance = content
                # 解析可选 frontmatter
                if content.startswith("---"):
                    lines = content.split("\n")
                    end_fm = -1
                    for i, line in enumerate(lines[1:], 1):
                        if line.strip() == "---":
                            end_fm = i
                            break
                    if end_fm > 0:
                        for fm_line in lines[1:end_fm]:
                            if fm_line.startswith("description:"):
                                description = fm_line[len("description:"):].strip()
                            elif fm_line.startswith("tags:"):
                                raw = fm_line[len("tags:"):].strip().strip("[]")
                                tags = [t.strip().strip("\"'") for t in raw.split(",") if t.strip()]
                        guidance = "\n".join(lines[end_fm + 1:]).strip()
                if not guidance:
                    continue
                skill = Skill(name=name, description=description, guidance=guidance, tags=tags)
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
        """返回全部技能，供 LLM 自主判断适用哪些。"""
        return list(self._skills)

    def match_for_context(
        self,
        *,
        wm_pressure: float,
        has_active_task: bool,
        has_next_step: bool,
        failure_count: int,
        high_error_streak: int,
        failure_threshold: int = 3,
        wm_pressure_threshold: float = 0.4,
        max_inject: int = 3,
    ) -> list[Skill]:
        """按当前情境挑选最相关的技能护栏。

        评分函数从离散阈值改为连续强度映射：
        - failure.reflection 得分随失败次数平滑增加，不再是">=3 才激活"
        - evidence-first-change 得分随 WM 压力平滑增加
        阈值参数（failure_threshold / wm_pressure_threshold）是"满分基准点"，
        不是开关门槛——内部状态的连续变化能连续影响护栏权重。
        """
        scored: list[tuple[float, Skill]] = []
        for skill in self._skills:
            score: float = 0.0
            tags = set(skill.tags)

            if skill.name == "runtime.bootstrap" and not has_active_task and failure_count == 0:
                score += 5.0
            if skill.name == "task.continuity" and has_active_task and has_next_step:
                score += 5.0
            # 失败反思：连续评分，failure_count/high_error_streak 越高得分越高（上限 6）
            if skill.name == "failure.reflection":
                _intensity = min((failure_count + high_error_streak) / max(failure_threshold * 2, 1), 1.0)
                score += _intensity * 6.0
            # 证据优先：WM 压力越大，护栏权重越高（上限 2）
            if skill.name == "evidence-first-change":
                score += min(wm_pressure / max(wm_pressure_threshold, 0.01), 1.0) * 2.0
            if skill.name == "provider.integration" and failure_count > 0:
                score += min(failure_count / max(failure_threshold, 1), 1.0) * 2.0

            if not has_active_task and "bootstrap" in tags:
                score += 2.0
            if has_next_step and "continuity" in tags:
                score += 1.0
            if failure_count > 0 and "failure" in tags:
                score += 2.0
            if wm_pressure >= wm_pressure_threshold and "verification" in tags:
                score += 1.0

            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda item: (-item[0], item[1].name))
        selected = [skill for _, skill in scored[:max_inject]]
        if not selected and not has_active_task:
            # 冷启动兜底：至少注入 bootstrap
            bootstrap = next((s for s in self._skills if s.name == "runtime.bootstrap"), None)
            if bootstrap:
                selected = [bootstrap]
        return selected

