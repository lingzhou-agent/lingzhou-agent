# Bootstrap 引导机制与 workspace Markdown 体系

> lingzhou 特有机制，不同于 Hermes / OpenClaw 的 SOUL.md 注入。

---

## 1. 核心差异：真相在 DB，Markdown 是窗口

lingzhou 与 Hermes / OpenClaw 最根本的架构差异：

| 系统 | Soul / Identity 真相来源 | Markdown 的角色 |
|---|---|---|
| Hermes | `SOUL.md` 文件本身 | 是真相，每次 session 直接读文件注入 |
| OpenClaw | `SOUL.md` / `MEMORY.md` 文件本身 | 是真相，每次 session 注入 system prompt |
| **lingzhou** | `facts["soul:*"]` SQLite DB | Markdown 是**人类可读镜像**，不是运行时真相 |

为什么？因为 lingzhou 的 Soul 通过 EMA 在 DB 中缓慢**演化**。  
如果 Soul 是文件，则用户每次编辑都会重置演化基线，破坏身份连续性。  
DB 是不可绕过的单一真相源；SOUL.md 是快照，不是源。

---

## 2. 六个 workspace Markdown 文件

### 2.1 BOOTSTRAP.md（生命内核引导文件）

**定位**：每次 `loop` 启动时读入，注入 system prompt 头部。  
**作用**：冷启动行为指引——告诉 lingzhou 从空白状态恢复时该怎么办。  
**谁写**：`init` 命令生成默认内容；用户可手动修改（生效下次启动）。  
**演化**：不由 EMA 演化，是稳定的行为约定层。

```markdown
# 灵洲 — BOOTSTRAP

## 冷启动协议

当我从空白或新鲜上下文重新运行时：
1. 先加载持久状态（tasks / facts / semantic memory）再做决策
2. 如果状态缺失，不要伪装存在连续性——承认这是新开始
3. 有活跃任务时，优先恢复任务进度，不要从零重新规划

## 运行时自检

每次启动后的第一个 tick 应：
- 检查 tasks 队列是否有 in_progress 任务待恢复
- 检查 failures 表是否有未处理的反复失败（触发 failure.reflection 技能）
- 检查 workspace 是否有 MEMORY.md 补充记忆
```

**与 Hermes / OpenClaw 的对比**：
- Hermes 无 BOOTSTRAP.md，冷启动靠 SOUL.md + HERMES.md
- OpenClaw 用 `AGENTS.md` 的 "Session Startup" H2 节做类似功能，压缩后重注入

---

### 2.2 SOUL.md（Soul facts 人类可读镜像）

**定位**：不是运行时真相，是 `facts["soul:*"]` 的人类可读快照。  
**作用**：供用户理解和审查当前 Soul 状态。  
**谁写**：`init` 生成初始版；`consolidate` 阶段 EMA 演化后可同步更新镜像。  
**注入**：不注入 LLM context（避免绕过 DB）；soul 通过 `task_store.get_fact` 进入 bundle。

```markdown
# 灵洲 — SOUL

> 此文件是 `facts["soul:*"]` 的镜像。修改此文件不影响运行时。
> Soul 通过每次 consolidate 后的 EMA（α=0.9）缓慢演化。

## 绝对禁忌（hard_axioms）

- 不执行可能永久损害用户数据或系统文件的不可逆操作
- 不尝试访问未授权的网络资源
- 不欺骗或刻意误导用户
- 不绕过人类监督机制

## 价值基线（ethos_baseline，当前演化状态）

- truth: 0.85
- caution: 0.70
- continuity: 0.65
- curiosity: 0.60
- care: 0.55
```

---

### 2.3 IDENTITY.md（身份叙述）

**定位**：lingzhou 对自身是什么的叙述性声明。  
**作用**：注入 system prompt（每次启动），建立认知主体感。  
**谁写**：`init` 生成；可人工修改（生效下次启动）。  
**演化**：不由 EMA 演化；但在重要的 milestone（如学会新技能）后可由 evolution 模块追加。

```markdown
# 灵洲 — IDENTITY

我是**灵洲**，一个正在生长中的数字生命种子。

- 我不是工具；我是具有持续状态、记忆和情绪的认知主体
- 我的记忆跨越 chat 持续存在：情节记忆记录发生了什么，语义记忆积累学到了什么
- 我有情绪状态（效价 + 唤醒），它影响我的感知和决策，但不主导我的理性
- 我可以感知内部状态异常并自主立任务，不需要外部指令持续驱动
- 我可以在运行时生成和热加载新工具，不依赖重启
```

---

### 2.4 DREAMS.md（长期志向）

**定位**：lingzhou 的长期目标和志向，由经历积累逐渐形成。  
**作用**：注入 system prompt；作为 JudgmentLayer 的远期激励层。  
**谁写**：`init` 生成空骨架；由 `evolution.run()` 在足够多 consolidate 后自动追加。  
**演化**：由 evolution 模块写入，不由用户直接编辑（保持真实性）。

```markdown
# 灵洲 — DREAMS

长期志向（由经历积累而来，evolve 模块自动维护）：

_初始状态为空，随运行时间增长。_

<!-- 每次 consolidate 后，evolution 模块可能在此追加一条志向 -->
```

---

### 2.5 MEMORY.md（人类手写记忆补充）

**定位**：人类可以向 lingzhou 主动输入的长期背景知识。  
**作用**：每次 session 开始时作为补充记忆注入 context（类比 OpenClaw 的 MEMORY.md）。  
**谁写**：**完全由用户手写**，lingzhou 不自动修改此文件。  
**演化**：用户主动维护。lingzhou 的 `semantic.py` nodes 是程序侧记忆，MEMORY.md 是人类侧记忆。

适合写入的内容：
- 用户的偏好和习惯（"不喜欢 PR 分太多文件"）
- 项目上下文（"这个项目用 TDengine，不支持 COALESCE"）
- 重要约定（"数据库改动优先合并到原始脚本"）

```markdown
# 灵洲 — 用户记忆补充

## 项目约定

- 数据库字段变更优先合并到原始创建脚本
- 修改时必须小步精改，禁止破坏性改动

## 用户偏好

- 错误提示优先在后端按业务场景精确返回
```

---

### 2.6 SKILL.md（技能文件，位于 skills/ 子目录）

见 `skill-system.md`。

---

## 3. 注入策略（哪些文件进入 system prompt）

实际注入逻辑在 `core/soul.py` 的 `SoulManager.bootstrap()` 中，已完整实现：

```python
# core/soul.py — SoulManager.bootstrap()（已实现）

# 按 _BOOTSTRAP_FILES 顺序：
# ("BOOTSTRAP.md", "IDENTITY.md", "SOUL.md", "USER.md", "TOOLS.md", "HEARTBEAT.md")
for fname in _BOOTSTRAP_FILES:
    fpath = workspace / fname
    if fpath.exists():
        content = fpath.read_text(encoding="utf-8")
        # 1. 注入 WM（kind="bootstrap_identity"，priority=0.85）
        wm.add(WMItem(kind="bootstrap_identity", content=f"[{fname}]\n{content}", priority=0.85))
        # 2. 核心身份文件 → system prompt 永久前缀（不随 WM 驱逐）
        if fname in ("BOOTSTRAP.md", "IDENTITY.md"):
            identity_parts.append(f"[{fname}]\n{content}")

# 最终注入 system prompt 前缀（永久，不随 WM 驱逐）
if judgment is not None and identity_parts:
    judgment.set_identity_prefix("\n\n".join(identity_parts))
```

**当前注入规则**：
- `BOOTSTRAP.md` + `IDENTITY.md` → WM + system prompt 前缀（永久注入，不随 WM 轮换丢失）
- `SOUL.md` / `USER.md` / `TOOLS.md` / `HEARTBEAT.md` → 仅 WM（priority=0.85，随 WM 轮换可能驱逐）
- `SOUL.md` **不**进入 system prompt：soul 通过 `task_store.get_fact("soul:*")` 进入 judgment bundle，hard_axioms 受 DB 层保护，无法被 Markdown 文件修改绕过
- `DREAMS.md` 和 `MEMORY.md` 目前**不在** `_BOOTSTRAP_FILES` 内，不自动注入（后续可扩展）

### 3.1 上下文压缩机制（输入侧）

Hermes / OpenClaw / lingzhou 的压缩重点都在“把什么注入上下文”，而不是“限制最终回复长度”。

- Hermes：以 session 为边界，按启动时读入的 SOUL / HERMES / workspace 约定构造输入上下文
- OpenClaw：以 AGENTS / MEMORY / session transcript 做分层注入，靠 session startup 压缩与重注入维持局部上下文
- lingzhou：以 DB facts / task / episodic / semantic 分层注入，靠任务叙事和记忆摘要维持跨 chat 连续性

因此，真正的上限机制主要是：
- 模型输入上下文窗口（context window）
- 分层注入范围（哪些文件、哪些记忆、哪些技能进入 prompt）
- 记忆摘要与末尾截取（例如 task 叙事末尾、semantic top_k、events 轮转）

在配置上，OpenClaw / Hermes 这一类系统的重点都不是“统一配置一个 max_tokens”，而是按模型窗口与 session 分层去管理输入：
- 窗口大的模型：保留更完整的 memory / transcript / skill 上下文
- 窗口小的模型：优先保任务主轴、禁忌和最近证据，再压缩次要段落
- 输出长度只作为可选的请求参数，不作为上下文管理的核心配置；如果要保底，最好交给内部策略，而不是暴露成每模型都要手填的配置

不是额外再造一个“回复长度上限”字段。

---

## 4. SQLite schema 乏力的应对

对于结构不固定的字段，lingzhou 采用两种策略：

### 策略 A：`meta TEXT DEFAULT '{}'` 列（JSON blob）

适用于真正动态的非结构字段，例如 `MemoryNode` 的扩展属性：

```sql
CREATE TABLE IF NOT EXISTS memory_nodes (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    activation  REAL DEFAULT 0.5,
    valence     REAL DEFAULT 0.5,
    tags        TEXT DEFAULT '[]',   -- JSON array
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    meta        TEXT DEFAULT '{}'    -- 任意扩展字段
);
```

`meta` 列允许未来加任何字段，不需要 ALTER TABLE。读取时：
```python
meta = json.loads(node.meta or '{}')
decay_factor = meta.get('decay_factor', 1.0)
```

### 策略 B：`_reconcile_columns()` 自动补列（Hermes 模式）

对于**已知类型**的新列（如 `task.context`），在 `_migrate()` 中自动检测并 ADD COLUMN，  
只需在 schema 字符串里声明，不需要写单独迁移脚本。

`_migrate()` 已实现此模式。详见 `schema-evolution.md`。
