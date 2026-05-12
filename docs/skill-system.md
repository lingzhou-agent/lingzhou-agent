# 技能系统

> 技能（Skill）是 lingzhou 的认知防线，不是功能列表。

---

## 1. 核心思想

lingzhou 的技能系统遵循一个核心原则：

**技能不是工具**——工具是执行能力（`file.read`、`shell.run`），  
技能是**认知姿态**，在特定上下文下被激活，影响判断层的推理方向。

lingzhou 在判断前先查询技能注册表，将匹配到的技能注入为 `skills_section`，  
让 LLM 在做决策时知道"当前应该遵守哪条防线"。

---

## 2. 技能 vs 工具

| 维度 | 工具（Tool） | 技能（Skill） |
|---|---|---|
| 本质 | 执行能力 | 认知防线 |
| 注册方式 | `@tool(manifest)` 装饰器 | `SkillRegistry.register()` |
| 触发时机 | 判断层 `chosen_action_id` | 判断层**之前**，上下文匹配 |
| 输出形式 | `result: str` | `guidance: str`（注入 prompt） |
| 持久化 | 无 | 可写入 `nodes/*.json` 技能节点 |

---

## 3. 5 个内置技能

### 3.1 `runtime.bootstrap`
- **激活条件**：冷启动（无活跃任务，无历史失败记录）
- **指导内容**：读取 BOOTSTRAP.md → 识别 workspace 状态 → 提出第一个任务目标
- **防线意义**：防止"无事可做"的空转循环

### 3.2 `provider.integration`
- **激活条件**：`failure_count > 0`（失败次数越多，注入权重越高；连续评分，上限 2.0）
- **指导内容**：调用工具前确认参数名和类型 → 失败后先分析错误原因再重试，不要盲目重试相同参数 → 文件不存在时换策略
- **防线意义**：防止认证失败导致的死循环

### 3.3 `task.continuity`
- **激活条件**：`has_active_task and has_next_step`（两者同时为 True，基础分 5.0 + tag 加分 1.0）
- **指导内容**：优先推进当前 next_step，不要分散注意力创建新任务 → 每步完成后立即更新 next_step
- **防线意义**：防止跨 chat 后"忘记"任务进度

### 3.4 `evidence-first-change`
- **激活条件**：WM 压力越高，注入权重越大（`wm_pressure / wm_pressure_threshold`，上限 2.0）；`wm_pressure_threshold` 默认 0.4
- **指导内容**：写操作前先读取确认前提成立 → 操作后再次读取验证 → 不确定时选范围更小、可逆的操作
- **防线意义**：防止破坏性操作（安全防线）

### 3.5 `failure.reflection`
- **激活条件**：连续评分，`(failure_count + high_error_streak) / (failure_threshold * 2)`，满分 6.0；`failure_threshold` 默认 3，即 6 次失败/高误差时得满分
- **指导内容**：停下来分析根因而非重复操作 → 判断是参数错误 / 前提不满足 / 工具本身有问题 → 选不同策略或向用户报告
- **防线意义**：防止同一错误无限重试

---

## 4. MatchForContext 算法

评分从离散阈值改为**连续强度映射**：每个触发条件对应浮点权重，最终取得分最高的 `max_inject`（默认 3）个技能。

```python
def match_for_context(
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
    scored: list[tuple[float, Skill]] = []

    for skill in registry:
        score = 0.0
        tags = set(skill.tags)

        # ── 技能名称专属规则（高基础分）────────────────────────────────
        if skill.name == "runtime.bootstrap" and not has_active_task and failure_count == 0:
            score += 5.0
        if skill.name == "task.continuity" and has_active_task and has_next_step:
            score += 5.0
        # failure.reflection：连续评分，上限 6.0
        if skill.name == "failure.reflection":
            score += min((failure_count + high_error_streak) / max(failure_threshold * 2, 1), 1.0) * 6.0
        # evidence-first-change：WM 压力越高权重越大，上限 2.0
        if skill.name == "evidence-first-change":
            score += min(wm_pressure / max(wm_pressure_threshold, 0.01), 1.0) * 2.0
        # provider.integration：失败次数越多权重越大，上限 2.0
        if skill.name == "provider.integration" and failure_count > 0:
            score += min(failure_count / max(failure_threshold, 1), 1.0) * 2.0

        # ── 标签通用加分 ────────────────────────────────────────────
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

    scored.sort(key=lambda x: (-x[0], x[1].name))
    selected = [s for _, s in scored[:max_inject]]

    # 冷启动兜底：至少注入 bootstrap
    if not selected and not has_active_task:
        selected = [registry["runtime.bootstrap"]]

    return selected
```

匹配顺序由得分决定，多技能可同时激活，合并为一个 `skills_section` 注入。

---

## 5. 技能注入到判断层

```python
# core/judgment.py _assemble_context()
matched_skills = skill_registry.match_for_context(
    wm_pressure=wm_pressure,
    has_active_task=has_active_task,
    has_next_step=has_next_step,
    failure_count=failure_count,
    high_error_streak=high_error_streak,
)
skills_section = "\n\n".join(s.guidance for s in matched_skills)
ctx["skills_section"] = skills_section or "（无特定技能激活）"
```

在 `prompts/judgment.md` 中：
```
## 当前激活的认知防线
{skills_section}
```

---

## 6. 技能注册接口

```python
# core/skill.py
from dataclasses import dataclass

@dataclass
class Skill:
    name: str          # "runtime.bootstrap"
    description: str   # 人类可读描述
    guidance: str      # 注入 LLM 的指导文本
    trigger: str       # 触发说明（文档用）

class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        self._skills[skill.name] = skill

    def match_for_context(self, ...) -> list[Skill]:
        ...

    def __getitem__(self, name: str) -> Skill:
        return self._skills[name]
```

---

## 7. 未来扩展：workspace 技能扫描

```
~/.lingzhou/workspace/
  skills/
    my-python-style/
      SKILL.md     ← 自动注册为技能节点
    code-review-checklist/
      SKILL.md
```

启动时自动扫描 `skills/*/SKILL.md`，注册为技能，激活条件从 frontmatter 读取：

```markdown
---
name: my-python-style
trigger: "工具调用包含 file.write 且文件是 .py"
---
本项目使用 ruff + mypy，写 Python 时优先...
```

这使得技能可以由用户在 workspace 中定义和演化，而不仅限于内置的 5 条。

---

## 8. 设计规则

1. **技能是防线，不是建议**——匹配到的技能必须出现在判断 prompt 中
2. **技能不直接执行操作**——只影响 LLM 的推理方向
3. **多技能可并存**——顺序合并，不互相覆盖
4. **技能可由 workspace 扩展**——未来 `skills/*/SKILL.md` 自动注册
5. **冷启动防线最重要**——没有 `runtime.bootstrap`，空循环会无限空转
