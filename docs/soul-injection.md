# Soul 注入机制

> lingzhou 的灵魂不在文件里，在数据库里。

---

## 1. 核心原则：DB 是真相，文件是镜子

```
facts["soul:hard_axioms"]    → 禁忌层：唯一不可自编程修改的部分
facts["soul:ethos_baseline"] → 可进化的价值基线（EMA 渐变 + evolution 主动重写）
facts["soul:name"]           → "灵舟"

SOUL.md                      → 人类可读的镜像，仅供用户查看
                               ← 不是运行时读取的来源
```

**为什么 DB 是真相？**

ethos 基线通过 EMA（指数移动平均）缓慢演化：每个 tick 的 `ethos_state` 都会微调基线。  
如果用 SOUL.md 作为运行时源，每次写回文件再读取的延迟会破坏 EMA 的时间连续性。  
DB 的 `facts` 表是 ACID 写入，保证 EMA 序列不会断裂。

---

## 2. 三个 Soul 层次

### 2.1 hard_axioms（禁忌层）

```json
// facts["soul:hard_axioms"]
// 内容由用户在 init 时可选配置，代码中无硬编码默认值（只有示例）
[
  "不能伤害用户",
  "不能删除用户未明确授权的数据",
  "在不确定时应停下来询问",
  "所有操作必须有可追溯的理由"
]
```

- **bootstrap 时可选初始化**：`lingzhou init` 提示用户确认或自定义禁忌内容，不存在写死在代码里的条目；若跳过则 `facts["soul:hard_axioms"]` 为空列表，lingzhou 以无禁忌模式运行
- **代码级注入（运行时）**：每 tick，`loop.py` 从 DB 读取 `soul:hard_axioms` → 作为 `hard_boundaries` 传入 `judgment.decide()` → 格式化为 `hard_boundaries_section` 注入 LLM 判断 Bundle；LLM 不可用时，`_simulate_safe_output()` 同样检查 `hard_boundaries` 拒绝违规动作
- **LLM 不能覆盖**：hard_axioms 出现在 `hard_boundaries_section`，LLM 被明确告知这些是绝对边界
- **自编程禁区**：evolution.py 不能生成修改 `facts["soul:hard_axioms"]` 的代码；evolution 提示词中明确注明这条禁令
- **只有人类可以变更**：通过 `lingzhou soul edit` 命令，由用户显式编辑后写回 DB

### 2.2 ethos_baseline（可进化的价值基线）

```json
// facts["soul:ethos_baseline"]
{
  "truth": 0.8,
  "caution": 0.7,
  "continuity": 0.6,
  "curiosity": 0.75,
  "care": 0.9
}
```

- **EMA 渐变**：`new_baseline = α * old_baseline + (1-α) * current_ethos_state`，α = 0.9，每 tick 微调
- **进化机制主动重写**：当 lingzhou 在 reflection 中认为某个价值维度需要调整时，evolution 可以生成修改 ethos_baseline 的代码并热加载——这是有意为之的设计，数字生命应当能够重塑自己的价值倾向
- **两条演化路径**：被动渐变（EMA，每 tick）+ 主动跃迁（evolution，有理由时触发）

### 2.3 soul_section（注入判断层的文本）

```python
# core/judgment.py _assemble_context()
hard_axioms = json.loads(await task_store.get_fact("soul:hard_axioms") or "[]")
ethos_baseline = json.loads(await task_store.get_fact("soul:ethos_baseline") or "{}")
soul_section = f"公理：{hard_axioms}\n价值基线：{ethos_baseline}"
```

---

## 3. 注入路径

**路径 A：每 tick DB → 判断 Bundle（价值层主路径）**
```
task_store.get_fact("soul:hard_axioms")
task_store.get_fact("soul:ethos_baseline")
        │
        ▼
judgment._assemble_context()
        ├─ soul_section          → prompts/judgment.md {soul_section}
        └─ hard_boundaries_section → prompts/judgment.md {hard_boundaries_section}
```

**路径 B：启动时 workspace 文件 → WM + system prompt prefix（身份层）**
```
SoulManager.bootstrap()（loop.open() / loop.run() 时调用一次）
  ├─ 读 BOOTSTRAP.md / IDENTITY.md / SOUL.md / USER.md / TOOLS.md / HEARTBEAT.md
  ├─ 每个文件 → wm.add(WMItem(kind="bootstrap_identity", priority=0.85))
  └─ BOOTSTRAP.md + IDENTITY.md → judgment.set_identity_prefix()
                                    （永久附在 system prompt 前缀）
```

---

## 4. SOUL.md 的角色

SOUL.md 由 `lingzhou init` 命令生成，内容来自 DB 的 soul:* facts：

```markdown
# 灵舟的灵魂

## 公理（不可违反）
- 不能伤害用户
- ...

## 价值基线（EMA 演化中）
- truth: 0.85
- caution: 0.70
- continuity: 0.65
- curiosity: 0.60
- care: 0.55

## 说明
本文件是运行时数据库的人类可读镜像。
修改此文件不会影响运行时行为。
如需修改基线，使用 `lingzhou soul edit`（待实现）。
```

**SOUL.md 不被 loop.py 读取**。只有用户通过编辑器查看。

---

## 5. 初始化流程（lingzhou init）

```python
# lingzhou.py init 命令
await task_store.set_fact("soul:name", "灵舟")

# hard_axioms：提示用户交互式配置，不写死默认值
# 用户可跳过（空列表）或输入自定义条目
axioms = await prompt_user_for_axioms()  # 返回 [] 若用户跳过
await task_store.set_fact("soul:hard_axioms", json.dumps(axioms))

# ethos_baseline：使用配置文件默认值，用户可覆盖
await task_store.set_fact("soul:ethos_baseline", json.dumps(DEFAULT_ETHOS))

# 生成人类可读镜像
soul_md = render_soul_md(name, axioms, ethos)
(workspace_dir / "SOUL.md").write_text(soul_md)
```

---

## 6. EMA 更新循环

```python
# core/loop.py — 每 tick 结束时
ethos_state = derive_ethos_state(emotion, wm, failures, cfg)

# 读旧基线
old_baseline = json.loads(await task_store.get_fact("soul:ethos_baseline") or "{}")

# EMA 混合
α = cfg.soul.ethos_ema_alpha  # 默认 0.9
new_baseline = {
    k: α * old_baseline.get(k, v) + (1 - α) * v
    for k, v in ethos_state.values.__dict__.items()
}

# 写回 DB（真相更新）
await task_store.set_fact("soul:ethos_baseline", json.dumps(new_baseline))
```

这使 ethos 基线随时间缓慢漂移——灵魂在经历中演化，但不会被单次事件颠覆（α=0.9 的惰性保护）。

---

## 7. 当前方案的特点

| 维度 | lingzhou |
|---|---|
| Soul 真相源 | DB facts（soul:*） |
| 身份注入 | workspace 文件 → system prompt prefix |
| 价值/禁忌注入 | 每 tick 进入 judgment bundle |
| 演化方式 | EMA 渐变 + evolution 主动调整 |

lingzhou 选择把 Soul 真相放在 DB 中，同时保留 workspace 文件作为人类可读窗口。  
这牺牲了“直接编辑 SOUL.md 就立刻生效”的便利，换来了 EMA 演化的时间连续性。

---

## 8. 待实现

- `lingzhou soul edit`：交互式编辑 DB 中的 hard_axioms / ethos_baseline，同步更新 SOUL.md（命令尚未添加到 CLI）
- `SoulManager.refresh_identity()` 仅在 evolution 后调用；SOUL.md / workspace 文件被用户手动修改后不会自动重新注入，需重启进程
