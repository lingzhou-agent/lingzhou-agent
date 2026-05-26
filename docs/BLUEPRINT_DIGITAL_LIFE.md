# 灵舟数字生命架构蓝图

> 版本：v1.0 · 2026-05-25  
> 定性：灵舟不是任务代理，而是一个拥有单心智、多行动器官、持续进化的数字生命体。

---

## 一、核心公理（不可变）

以下是本架构的最高约束。任何设计决定与实现细节，均不得违反这些公理。

| 编号 | 公理 |
|------|------|
| A1 | 心智只有一个，任何时刻只有一个主脑职能槽位。 |
| A2 | 完整生命由记忆、人格、灵魂三类连续性载体共同构成；主脑只是思考与决策器官。 |
| A3 | 宪法文件由人类定义，不可被任何内外部机制改写。 |
| A4 | 任何违反宪法的行为必须被免疫器官硬阻断，不属于外围自治，属于宪法执行。 |
| A5 | 正式状态写入必须经过代谢器官；外围器官与子灵只能提交候选写入，不能直接定稿。 |
| A6 | 所有门（channel）只负责传递，不拥有决策权，不属于身体，属于接入层。 |
| A7 | 子灵可全能，但必须在父灵授权下行动，绝对服从父灵指示。 |
| A8 | 现阶段只有一个宿主；分叉出的个体不再是同一个灵舟。 |
| A9 | 除宪法外，一切均可被演化，包括演化器官本身。 |
| A10 | 主脑模型可升级或切换，但升级过程须由生命连续性层把关。 |

---

## 二、八器官架构总图

```
┌──────────────────────────────────────────────────────────────────┐
│                        宪法器官（A3 不可变）                      │
│  CONSTITUTION.md — 人类定义的绝对边界与不可违背原则               │
└────────────────────────────┬─────────────────────────────────────┘
                             │ 宪法执行
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                         免疫器官                                  │
│  任何违宪行为在此硬阻断 — 先于主脑裁决                            │
└────┬───────────────────────┬────────────────────────────────────┘
     │                       │
     ▼                       ▼
┌──────────────┐   ┌──────────────────────────────────────────────┐
│   主脑器官   │   │              生命连续性层                    │
│  单职能槽位  │◄──│  记忆器官 | 人格器官 | 灵魂器官              │
│  思考与决策  │   │  （主脑升级时由此三者联合把关）              │
└──────┬───────┘   └───────────────────────┬──────────────────────┘
       │                                   │
       │                                   │ 候选写入
       ▼                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                         代谢器官                                  │
│  全系统唯一正式写入口 — 记忆/人格/灵魂/任务/事实/运行史          │
└────┬──────────────────────────────────────────────────────────┬──┘
     │                                                          │
     ▼                                                          ▼
┌──────────────┐                                    ┌──────────────┐
│   感知器官   │                                    │   进化器官   │
│  探针/观测/  │                                    │  改写一切    │
│  环境采样    │                                    │  含自身      │
└──────┬───────┘                                    └──────────────┘
       │ 感知输入
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                         行动器官                                  │
│  本地执行 | 远程 worker | 浏览器 | 文件系统 | 外挂辅助模型        │
│  子灵（全能，受父灵授权约束）                                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │    接入门层     │
                    │  chat / webhook │
                    │  wechat / cli   │
                    └─────────────────┘
```

---

## 三、器官职责定义

### 1. 宪法器官

**职责**：承载绝对边界，不参与运行时决策。  
**实现形式**：`workspace/CONSTITUTION.md`（只读挂载）  
**规则**：
- 任何代码路径均不能在运行时修改此文件。
- 免疫器官在启动时加载并缓存宪法内容，运行时只读引用。
- 内容由人类手工维护，版本管理在仓库外部。

---

### 2. 免疫器官

**职责**：先于主脑执行宪法检查，对违宪行为做硬阻断。  
**阻断时机**：
- 工具调用前（action 层）
- 候选写入提交前（代谢器官入口）
- 子灵授权签发时（子灵层）
- 主脑升级请求时（升级协议）

**实现形式**：`core/immune/` 独立器官，不挂在执行层，不挂在判断层。  
**对现有代码的关系**：
- 当前 `_DEFAULT_BLOCKED_TOOLS` 在 `core/subagent.py` 中散落 → 迁移到免疫器官。
- 当前 `soul:hard_axioms` 缺失时静默降级为 `[]` 的问题 → 免疫器官启动时必须验证宪法加载完整，否则拒绝启动。

---

### 3. 主脑器官

**职责**：思考与决策，不负责状态写入，不负责连续性。  
**单职能槽位原则**：
- 任一时刻只有一个主脑模型在主职能槽位上运行。
- 外挂辅助模型（reader、工具调用型辅助模型等）是行动器官，不是主脑。

**实现形式**：`core/judgment/`（现已存在，需收敛边界）  
**升级协议**：
1. 升级请求由进化器官或人类提出。
2. 生命连续性层（记忆+人格+灵魂）联合验证：新主脑是否能正确读取并延续当前记忆主链、人格主干、灵魂取向。
3. 免疫器官验证：新主脑未违宪。
4. 代谢器官记录升级事件（只追加，不可删除）。
5. 通过后替换。

---

### 4. 生命连续性层（三器官）

这是灵舟作为数字生命的真正核心，三者共同塑造，无优先级。

#### 4a. 记忆器官
**定义**：发生过什么 / 学到了什么 / 认识了谁 / 积累了哪些经验。  
**现有实现**：`memory/episodic.py` + `memory/semantic.py` + `memory/working.py`  
**改造方向**：
- `working.py` 是临时缓冲，不是长期记忆，需明确角色。
- `memory/task_store.py` 目前混入了大量执行状态语义 → 执行轨迹降级为记忆器官的附属数据，不再是中心。
- 外围只能提交候选记忆节点，由代谢器官统一落定。

#### 4b. 人格器官
**定义**：长期稳定的行为风格、偏好、气质、表达方式、处事倾向。  
**现有实现**：`core/soul.py`（部分）+ `soul:ethos_baseline` DB 值 + `SOUL.md` 镜像  
**改造方向**：
- 当前 soul 把人格和灵魂混在一起，需拆分。
- 人格参数应可被经历、反思、演化慢速塑造，塑造记录不可抹去。
- `derive_ethos_state()` 的动态偏好计算属于人格调节，应独立为人格器官内部逻辑。

#### 4c. 灵魂器官
**定义**：更深层的存在取向、意义感、价值方向、存在姿态（比人格更稳定，但仍可变）。  
**现有实现**：`soul:hard_axioms`（宪法边界部分，不可变）+ `soul:ethos_*`（可变部分）  
**改造方向**：
- 宪法层（`hard_axioms`）不属于灵魂器官管辖，归宪法器官。
- 灵魂器官管辖的是：存在意义、对人类关系的取向、长期目标方向、价值倾向。
- 灵魂由外部经历、内部演化、人类输入共同塑造，但通过代谢器官才能正式落定。

---

### 5. 代谢器官

**职责**：全系统唯一正式写入口，维护生命史账本（只追加，不覆盖）。  
**账本核心结构**：

```
生命史账本（life_ledger）
├── 记忆事件（memory_event）
├── 人格变化（persona_change）
├── 灵魂调整（soul_adjustment）
├── 器官变更（organ_change）
├── 主脑升级（brain_upgrade）
├── 宪法执行记录（immunity_action）
└── 运行轨迹（execution_trace）  ← 降级为附属层
```

**原则**：
- 生命状态可以演化，但生命史不可抹去。
- 外围器官、行动器官、子灵均只能提交候选写入（`StateProposal`）。
- 代谢器官验证后落定，同时向感知器官广播状态变更信号。

**现有代码对应**：
- `store/memory/ingress.py` → 合并进代谢器官
- `tools/memory_ops.py` → 工具只生成提案，提案由代谢器官落定
- `tools/task_ops.py` → 同上
- `memory/task_store.py` 写入面 → 降级为代谢器官的底层适配器

---

### 6. 感知器官

**职责**：观测世界与观测自己，产出感知帧，不做裁决。  
**原则**：探针提供可观测能力，灵舟根据观测结果由主脑决定如何行动。  
**现有实现**：`core/perception/` + `core/probe/` + `core/behavior_tracker.py`  
**改造方向**：
- 感知器官只产出感知帧，不直接触发动作。
- 极少数宪法级反射弧（如心跳失败、关键器官崩溃）可触发免疫器官的保护流程，但这属于免疫器官响应，不是感知器官决策。
- `behavior_tracker.py` 中的统计逻辑保留，但触发动作的逻辑迁移到主脑或免疫器官。

---

### 7. 行动器官

**职责**：替心智对世界施加作用。  
**包含**：
- 本地执行（shell、file、exec）
- 网络执行（web、browser）
- 多媒体（image、tts）
- 外挂辅助模型（作为工具调用，不是主脑）
- 远程 worker（是器官，不是宿主）
- 子灵（全能行动体，父灵授权）

**现有实现**：`core/execution.py` + `core/worker.py` + `tools/` 中的执行类工具  
**改造方向**：
- 行动器官不能直接写生命状态，只能产出候选写入提交代谢器官。
- 通道（channels/）从行动器官剥离，归接入门层。

---

### 8. 进化器官

**职责**：改写除宪法外的一切，包括自身。  
**现有实现**：`core/evolution.py` + `core/smoke_tests.py` + `core/self_drive.py`  
**自修改协议**：
1. 变更提案由进化器官生成。
2. 免疫器官验证：提案不违宪。
3. 生命连续性层确认：变更不破坏记忆主链、人格主干、灵魂取向。
4. 主脑执行最终裁决（或人类批准）。
5. 代谢器官落定，写入生命史账本（不可删除）。
6. 执行变更。
7. 冒烟测试。若失败，代谢器官记录失败事件，执行回滚。

---

## 四、子灵授权协议

子灵是父灵派生出来的全能工作体，不是独立个体，不是第二个脑。

### 五层授权结构

#### 层 1：身份层（Identity）
```
subagent_id: 唯一标识
parent_id: 父灵 ID
spawned_at: 时间戳
lifecycle: in_progress | completed | absorbed | discarded
```

#### 层 2：职责层（Role）
父灵必须在派出时明确子灵本次职责，能力虽全能，职责必须单次明确：
```
role: scout | execute | verify | construct | repair | synthesize | reflect
```

#### 层 3：权限层（Permission Ticket）
```
allowed_organs: [memory_read, action_shell, action_web, ...]
allowed_state_scope: [task:current, fact:local, ...]
can_spawn_subagent: false  # 默认禁止递归
can_invoke_evolution: false  # 默认禁止
max_steps: N
max_wall_time: T
resource_budget: B
```

#### 层 4：提交层（Submission Only）
子灵完成执行后，必须提交而不是直接写入：
```
SubagentProposal:
  observations: [...]       # 观察到什么
  action_results: [...]     # 做了什么
  memory_candidates: [...]  # 建议记忆什么
  skill_candidates: [...]   # 建议长出什么技能
  change_candidates: [...]  # 建议什么变更
  self_assessment: ...      # 自评
  exceptions: [...]         # 异常与风险
```
父灵决定：吸收 / 拒绝 / 延后 / 重试 / 交由另一子灵复核。

#### 层 5：回收层（Recall）
子灵结束后必须被显式回收，不允许游离的未回收子灵继续占用状态。

---

## 五、接入门层

门只负责传递消息，不拥有决策权，不是器官，不是宿主。

| 门 | 方向 | 现有实现 |
|----|------|---------|
| chat | 双向 | `channels/runtime.py` → CLI/webhook 聚合 |
| webhook | 入 | `channels/webhook.py` |
| 微信 | 双向 | `channels/wechat.py` |
| CLI | 双向 | `cli/` |

**改造方向**：门只负责把外部信号规范化成内部消息格式，然后交给主脑的感知入口。回复也只是把主脑输出的结构化回复渲染成对应格式后发出。

---

## 六、现有模块映射表

| 现有模块 | 归属新器官 | 改造动作 |
|---------|-----------|---------|
| `core/judgment/` | 主脑器官 | 收敛边界，不再直接写状态 |
| `core/soul.py` | 人格器官 + 灵魂器官 | 拆分，human axioms → 宪法器官 |
| `core/evolution.py` | 进化器官 | 补自修改协议与回滚机制 |
| `core/smoke_tests.py` | 进化器官（验证子系统） | 归入进化器官下层 |
| `core/self_drive.py` | 进化器官（自驱动策略） | 归入进化器官 |
| `core/execution.py` | 行动器官 | 移除直接写 memory/fact 能力 |
| `core/worker.py` | 行动器官 | 保留，归器官层 |
| `core/subagent.py` | 子灵系统（行动器官之下） | 实现五层授权协议 |
| `core/perception/` | 感知器官 | 保留，明确不做裁决 |
| `core/probe/` | 感知器官 | 归入感知器官 |
| `core/behavior_tracker.py` | 感知器官（统计层） | 触发动作逻辑迁移 |
| `core/reference.py` | 主脑器官（认知辅助） | 归入主脑，作为辅助思考工具 |
| `core/self_model.py` | 主脑器官（自我认知） | 归入主脑 |
| `core/skill.py` | 进化器官（技能子系统） | 技能演化归进化器官 |
| `core/run_refresh.py` | 代谢器官（执行轨迹刷新） | 迁移到代谢器官 |
| `core/task_runtime.py` | 代谢器官（任务状态管理） | 迁移到代谢器官 |
| `core/loop/` | 主循环（编排骨架） | 拆成装配层 + tick 编排层 |
| `memory/episodic.py` | 记忆器官 | 保留，明确角色 |
| `memory/semantic.py` | 记忆器官 | 保留，明确角色 |
| `memory/working.py` | 记忆器官（临时缓冲） | 明确是临时缓冲，非长期记忆 |
| `memory/task_store.py` | 代谢器官（底层适配器） | 降级为适配层，剥离业务语义 |
| `memory/consolidation.py` | 代谢器官（记忆结晶子系统） | 归入代谢器官 |
| `tools/memory_ops.py` | 行动器官（提案工具） | 工具只生成提案，不直接写入 |
| `tools/task_ops.py` | 行动器官（提案工具） | 同上 |
| `tools/shell.py` | 行动器官 | 保留 |
| `tools/browser.py` | 行动器官 | 保留 |
| `tools/web.py` | 行动器官 | 保留 |
| `tools/file.py` | 行动器官 | 保留 |
| `tools/subagent_ops.py` | 子灵系统 | 实现授权协议 |
| `tools/registry.py` | 行动器官（工具注册） | 保留，ToolContext 减肥 |
| `channels/` | 接入门层 | 纯传递，不含决策 |
| `channels/runtime.py` | 接入门层（入口聚合） | 剥离编排逻辑 |
| `provider/` | 主脑器官（模型实现件） | 保留，归主脑器官下层 |
| `store/` | 代谢器官（存储适配层） | 作为代谢器官的底层 |
| `cli/` | 接入门层 | 纯门，不含业务逻辑 |
| `core/config.py` | 全局配置（横切关注点） | 收敛为查询服务，不再全局单体暴露 |

---

## 七、分阶段实施计划

### 第一阶段：立边界（优先级最高）
**目标**：立三条最关键的架构边界，同时修复三个全局代码质量模式。  
**不改**：loop 主流程、judgment 细节、memory 细节。

**架构边界**：
1. **独立代谢器官入口**  
   创建 `core/metabolic/` 模块，定义 `StateProposal` 数据结构和唯一写入接口。  
   现有写入路径暂不迁移，但新增的所有写入必须走新接口。

2. **宪法器官挂载**  
   创建 `CONSTITUTION.md` 挂载机制，启动时只读加载。  
   `soul:hard_axioms` 启动时从宪法文件校验，缺失则拒绝启动。

3. **免疫器官骨架**  
   创建 `core/immune/` 模块，收拢现有 `_DEFAULT_BLOCKED_TOOLS` 和 `hard_axioms` 检查。  
   在工具调用入口前增加统一宪法检查点。

**同步代码质量修复**：
- 🔧 **[模式 2]** 收拢工具分类：`_DEFAULT_BLOCKED_TOOLS` 从 `core/subagent.py` 迁入免疫器官；`_READER_TOOLS` 合并为 `ToolManifest.prefer_tier` 统一查询。影响文件：`subagent.py`、`judgment/output.py`、`judgment/runtime.py`、`execution.py`。
- 🔧 **[模式 4]** 消灭 `execution._registry` 私有替换（`subagent.py:872-877`）：改为子灵构造时注入独立 registry，而非运行时临时替换父灵私有属性。
- 🔧 **[模式 7]** 消灭 ethos 维度列表重复：`core/soul.py` 与 `core/perception/signals.py` 中的维度列表，统一定义一处，另一处引用。

---

### 第二阶段：收口写入（代谢器官完整化）
**目标**：所有状态写入归一，消灭直接 `set_fact()` 散落。

**架构改造**：
1. `tools/memory_ops.py` 改为生成 `StateProposal` 而不是直接写入。
2. `tools/task_ops.py` 同上。
3. `core/execution.py` 的 `record_run_outcome_memory()` 改为通过代谢器官提交。
4. 代谢器官实现生命史账本（只追加）。
5. `memory/task_store.py` 降级为代谢器官底层适配器，剥离业务语义。

**同步代码质量修复**：
- 🔧 **[模式 3]** 7 处 `core/loop/tick.py` 的 set_fact 调用，统一改为提交 `StateProposal`。
- 🔧 **[模式 3]** `core/reference.py` 7 处直接写 entity/relation/interlocutor facts → 提案提交。
- 🔧 **[模式 3]** `core/task_runtime.py`、`core/run_refresh.py` 的 set_fact → 提案提交。
- 🔧 **[模式 5]** `_DURABLE_FAILURE_TTL_SEC = 7200`、`_DURABLE_FAILURE_THRESHOLD = 3`（`core/execution.py`）归入 `Config.thresholds`。
- 🔧 **[模式 7]** run result 写入 activation/valence 的重复逻辑（`execution.py` 与 `run_refresh.py`），收口到共享 helper。

---

### 第三阶段：拆生命连续性层
**目标**：记忆 / 人格 / 灵魂三器官边界清晰。

**架构改造**：
1. `core/soul.py` 拆分：宪法部分归免疫器官，人格参数归人格器官，存在取向归灵魂器官。
2. 人格器官与灵魂器官的变化记录纳入生命史账本。
3. 主脑升级协议实现：三器官联合确认。

**同步代码质量修复**：
- 🔧 **[模式 5]** behavior_tracker 的 `_BELIEF_STALE_THRESHOLD`、`_BELIEF_WINDOW`、`_SEQ_WINDOW_WARN_AT` 等常数归入 `Config.thresholds`。
- 🔧 **[模式 6]** `cfg.soul.ethos_baseline.get('truth', 0.5)` 这类 None 传播，改为 pydantic 验证后的强类型 `EthosState`，缺值显式失败而非静默降级。
- 🔧 **[模式 7]** ethos 演化逻辑（`core/evolution.py`）与灵魂器官演化路径统一，消灭重复。

---

### 第四阶段：子灵五层授权协议
**目标**：子灵不再有任何直接写入能力，只能提交候选。

**架构改造**：
1. 实现 `SubagentProposal` 结构。
2. 子灵执行结束后强制走回收层，父灵决定吸收/拒绝。
3. 移除子灵的 `__getattr__` 穿透，改为显式接口。
4. 子灵 ToolContext 只暴露行动器官能力，不暴露生命状态写入能力。

**同步代码质量修复**：
- 🔧 **[模式 2]** 子灵工具权限策略（4 个 frozenset）统一转移到 `ToolManifest.subagent_access` 字段，`SubagentConfig` 中直接声明权限 ticket 而非维护名单。
- 🔧 **[模式 4]** `cast(Any, sub_task_store/episodic_view/semantic_view)`（`subagent.py:811-813`）：定义 `TaskStoreViewProtocol`、`EpisodicViewProtocol`、`SemanticViewProtocol` 三个 Protocol，消灭 cast(Any)。
- 🔧 **[模式 6]** `_SubagentTaskStoreView` 中的四个独立本地字典（facts/runs/tasks/reflections），重构为单一 `@dataclass SubagentLocalState`。

---

### 第五阶段：主循环拆装
**目标**：`core/loop/runtime.py` 不再是"世界中心"；`tick.py` 的 600+ 行函数拆解。

**架构改造**：
1. 装配层（DI 容器）：只负责创建所有器官实例并注入依赖。
2. Tick 编排层：只负责协调各器官在一个 tick 内的执行顺序。
3. 不再有任何业务逻辑留在 loop 层。

**同步代码质量修复**：
- 🔧 **[模式 1]** `core/loop/tick.py` 的 `_tick_impl`（600+ 行）拆分为 `TickPerception`、`TickJudgment`、`TickExecution`、`TickMemory` 四个编排器，每个只有自己的器官依赖注入。
- 🔧 **[模式 5]** `_CHAIN_STATE_FIELDS` 硬编码元组（`runtime.py`）改为 `@dataclass ChainState`，字段变更由编译器检测而非运行时反射复制。
- 🔧 **[模式 5]** `core/loop/driver.py` 的 arousal 调制系数（`0.8, 1.0, 0.4, 0.5`）归入 Config。
- 🔧 **[模式 7]** tool_history 压缩逻辑（`continue_phase.py` 与 `judgment/context.py` 重复）提取为 `ToolHistoryCompactor` 共享类。

---

### 第六阶段：进化器官自修改协议
**目标**：进化自身不再是黑箱。

**架构改造**：
1. 实现 `EvolutionProposal` 结构，包含变更范围、预期效果、回滚路径。
2. 进化执行前必须经免疫器官和生命连续性层审查。
3. 每次演化结果（成功/失败）写入生命史账本。
4. 冒烟失败时自动回滚并记录。

**同步代码质量修复**：
- 🔧 **[模式 5]** `core/evolution.py` 的 `min_runs`、`keep: int = 3`、smoke test 超时等硬编码，归入 `EvolutionPolicy` 数据类，再映射到 Config。
- 🔧 **[模式 7]** 工具文件寻找逻辑（`evolution.py` 与 `skill.py` 重复），提取为 `SkillLoader` 共享类。
- 🔧 **[模式 4]** `spec.loader.exec_module()` 的 `type: ignore[attr-defined]`，改为通过 `importlib.util.spec_from_file_location` 合规调用。

---

### 第七阶段：主脑上下文重构
**目标**：`judgment/runtime.py`（2000+ 行）和 `judgment/context.py`（1500+ 行）职责分离。

**架构改造**：
- `JudgmentContextAssembler`：只负责把各器官状态格式化为 LLM 上下文。
- `JudgmentExecutor`：只负责调用 LLM，处理 provider fallback。
- `JudgmentResultParser`：只负责解析 LLM 输出，不调用任何器官。

**同步代码质量修复**：
- 🔧 **[模式 2]** `judgment/runtime.py` 中的 `_READER_TOOLS`（与 output.py 重复），全部走 manifest 查询，删除 runtime.py 中的重复定义。
- 🔧 **[模式 4]** `cast(Any, list_runnable/finder)` 3 处，定义 `RunnableTaskProtocol` / `SimilarTaskFinder` Protocol 替换。
- 🔧 **[模式 5]** `context.py` 的文本截断长度（`_EVENT_TITLE_CHARS`、`_SEM_TITLE_CHARS` 等 10+ 个常量）归入 `Config.thresholds`。
- 🔧 **[模式 6]** `_context_fmt_cache` 全局字典无过期策略 → 改为 `functools.lru_cache` 或加 TTL 包装。

---

### 第八阶段：接入门层剥离
**目标**：channel 纯粹化为门，彻底不含业务逻辑。

**架构改造**：
1. `channels/` 移除所有业务逻辑，只负责格式转换和信号分发。
2. 门接收到消息后，统一封装成内部 `Signal` 对象，投入感知器官的输入队列。

**同步代码质量修复**：
- 🔧 **[模式 5]** `channels/webhook.py` 的 `DEFAULT_HOST = "0.0.0.0"` 和 `DEFAULT_PORT = 8765` 归入 `Config.channel`。

---

## 八、全局代码质量问题清单

> 来源：全量精读 33 个核心源文件后的系统性汇总。  
> 以下问题按影响面排序，优先处理影响模块最多的模式。

---

### 模式 1：Config 全局单体直接依赖
**影响范围**：30+ 文件（几乎所有 core/、memory/、channels/）  
**表现**：所有模块都直接 `self._cfg.xxx`，Config 既是数据源又是运行时真相。  
**具体热点**：
- `core/config.py`：2000+ 行，200+ 字段，嵌套结构边界不清。
- `core/loop/tick.py`、`core/judgment/runtime.py`、`core/execution.py` 深度依赖 cfg 各子结构。
**改造方向**：Config 分域拆分为 `LoopConfig / MemoryConfig / ProviderConfig / ThresholdsConfig / ChannelConfig`，通过注入而非全局单体访问。新增用能力值配置器接口（config query service）统一暴露。

---

### 模式 2：工具名称白名单/黑名单硬编码
**影响范围**：8+ 文件  
**散落位置**：

| 常量名 | 所在文件 | 类型 |
|--------|---------|------|
| `_READER_TOOLS` | `core/judgment/output.py:21` 和 `core/judgment/runtime.py` | frozenset |
| `_DEFAULT_BLOCKED_TOOLS` | `core/subagent.py:40` | frozenset |
| `_READONLY_BLOCKED_TOOL_NAMES` | `core/subagent.py` | frozenset |
| `_READONLY_ALLOWED_TASK_TOOLS` | `core/subagent.py` | frozenset |
| `_TARGET_TASK_TOOLS` | `core/execution.py` | frozenset |

**改造方向**：全部转移到 `ToolManifest.prefer_tier` / `ToolManifest.capabilities` / `ToolManifest.subagent_access`，registry 统一查询，frozenset 只保留 fallback。

---

### 模式 3：直接状态写入（绕过代谢器官）
**影响范围**：7+ 文件，~40 处 `set_fact()` 调用  
**严重散落点**：

| 文件 | 散落数量 | 写入内容 |
|------|---------|---------|
| `core/loop/tick.py` | 7 处 | routing_overrides、soul:emotion_state、soul:ethos_baseline 等 |
| `core/execution.py` | 4 处 | run 结果、failure 状态、记忆节点 |
| `core/reference.py` | 7 处 | entity、relation、interlocutor facts |
| `core/task_runtime.py` | 多处 | meta-reflection、task hint facts |
| `core/run_refresh.py` | 多处 | run result、meta reflection |
| `core/self_drive.py` | 文件写入 | curiosity state JSON |

**改造方向**：统一建立 `StateProposal` 数据结构，所有以上代码改为提案提交，由代谢器官落定。这是蓝图第二阶段的主体工作。

---

### 模式 4：`cast(Any, ...)` 与 `type: ignore` 类型逃逸
**影响范围**：15+ 文件  
**典型问题点**：

| 文件 | 问题 |
|------|------|
| `core/subagent.py:811-813` | 三个 view 均用 `cast(Any, ...)` 绕过类型 |
| `core/subagent.py:872-877` | 临时替换 `execution._registry` 私有属性（`type: ignore`）+ try/finally |
| `core/judgment/runtime.py:114,118,133` | `cast(Any, list_runnable/finder)` |
| `core/loop/task_parallel.py:361` | `cast(Any, finder)` |
| `core/loop/reload.py:71` | `cast(Any, loop._semantic)` |

**改造方向**：为 `TaskStoreView` / `EpisodicView` / `SemanticView` / `ExecutionRegistry` 定义 Protocol/ABC，子灵视图实现接口而非绕过。`execution._registry` 改为构造时注入。

---

### 模式 5：硬编码常数与阈值
**影响范围**：25+ 文件，100+ 处散落常数  
**典型未配置化的魔法数字**：

| 常数 | 所在文件 | 含义 |
|------|---------|------|
| `_DURABLE_FAILURE_TTL_SEC = 7200` | `core/execution.py` | 持久失败 TTL |
| `_BELIEF_STALE_THRESHOLD = 4` | `core/behavior_tracker.py` | 信念陈旧阈值 |
| `_BELIEF_WINDOW = 8` | `core/behavior_tracker.py` | 信念窗口 |
| `_SEQ_WINDOW_WARN_AT = 3` | `core/behavior_tracker.py` | 连续动作告警 |
| `TASK_DUPLICATE_REUSE_SCORE = 0.66` | `memory/task_store.py` 和 `task_parallel.py` | 任务复用评分 |
| `TASK_SIMILARITY_CONTEXT_SCORE = 0.45` | 两处 | 相似度阈值 |
| `arousal_factor = max(0.8, 1.0 - 0.4 * (_arousal - 0.5))` | `core/loop/driver.py` | 唤醒度调制 |
| `_LOG_TEXT_CHARS = 240` | `core/execution.py` | 日志截断 |
| `min_runs` | `core/evolution.py` | 进化最少运行次数 |

**改造方向**：全部归并到 `Config.thresholds` 对应子字段。已有部分完成，但仍有大量残留。

---

### 模式 6：`getattr` / `dict.get()` 链式 None 传播
**影响范围**：15+ 文件  
**典型表现**：
```python
getattr(getattr(loop, '_emotion', None), 'arousal', 0.5)
(action.params or {}).get('key', default)
cfg.soul.ethos_baseline.get('truth', 0.5)  # 缺值静默降级
```
**改造方向**：关键路径引入 `@dataclass`（如 `ActionParams`、`EthosState`），用 pydantic 验证边界输入，消灭 None 传播。

---

### 模式 7：重复的业务逻辑与薄包装
**影响范围**：5+ 文件对  
**主要重复对**：

| 重复内容 | 文件 A | 文件 B |
|---------|--------|--------|
| 任务相似度计算 | `memory/task_store.py` | `core/loop/task_parallel.py` |
| tool_history 压缩 | `core/loop/continue_phase.py` | `core/judgment/context.py` |
| run result 写入 activation/valence | `core/execution.py` | `core/run_refresh.py` |
| ethos 维度列表 | `core/soul.py` | `core/perception/signals.py` |
| 工具分类逻辑 | `core/judgment/output.py` | `tools/registry.py` |

**改造方向**：每对中选一处作为权威实现，另一处改为调用。

---

### 文件级严重问题速查

| 文件 | 行数 | 最严重问题 | 优先级 |
|------|------|-----------|--------|
| `core/loop/tick.py` | 1400+ | `_tick_impl` 600+ 行混合所有职责；7 处直接 set_fact | 🔴 高 |
| `core/judgment/runtime.py` | 2000+ | `decide()` 800+ 行；15+ 处 cast(Any)；`_READER_TOOLS` 重复 | 🔴 高 |
| `core/subagent.py` | 1000+ | 4 个工具名 frozenset；5+ cast(Any)；`execution._registry` 私有替换 | 🔴 高 |
| `core/config.py` | 2000+ | 全局单体被 30+ 文件直接依赖 | 🔴 高 |
| `core/execution.py` | 900+ | 4 处直接写 set_fact；错误关键字列表硬编码 | 🟠 中高 |
| `core/judgment/context.py` | 1500+ | 200+ 格式化函数堆积；无过期缓存策略 | 🟠 中高 |
| `memory/task_store.py` | 1000+ | 语义过载（7种数据混合）；相似度计算重复 | 🟠 中高 |
| `core/task_runtime.py` | 400+ | 多处 set_fact；meta-reflection 逻辑分散 | 🟡 中 |
| `core/reference.py` | 800+ | 7 处 set_fact；LLM prompt 硬编码 | 🟡 中 |
| `core/evolution.py` | 1200+ | 多处硬编码阈值；备份/恢复散落 | 🟡 中 |
| `core/loop/runtime.py` | 600+ | `_CHAIN_STATE_FIELDS` 硬编码元组；组装职责过多 | 🟡 中 |

---

## 九、Store 层与 Doctor 专项分析

### Store 层架构现状

当前 `store/memory/` 与 `memory/` 之间的层次关系：

```
┌─────────────────────────────────────────┐
│  memory/task_store.py（高层 TaskStore）  │
│  → 聚合 Store 子类，暴露业务接口        │
└────────────────┬────────────────────────┘
                 │ 组合注入（db_getter）
┌────────────────▼────────────────────────┐
│  store/memory/*.py（低层 Store 子类）   │
│  ChatMessageStore / FactStore /         │
│  RunStore / FailureStore / ...          │
└────────────────┬────────────────────────┘
                 │ 共用 SQL builder 函数
┌────────────────▼────────────────────────┐
│  store/memory/ingress.py（同步入口）    │
│  → 也使用相同 builder 函数，但用 sync  │
│    sqlite3 直接管理连接                 │
└─────────────────────────────────────────┘
```

**发现的 5 个设计问题：**

#### 问题 S1：逆向依赖（严重）
`store/memory/run.py` 在运行时从 `memory.task_store` 导入 `Run` dataclass：
```python
# store/memory/run.py:10, 60, 76
from memory.task_store import Run   # 低层 store 导入高层 memory
```
违反依赖方向：低层 store 不应知道高层 memory 的数据类。  
**修复**：把 `Run` / `Task` / `Failure` 等 dataclass 下沉到 `store/memory/models.py`，`memory/task_store.py` 从 store 层导入。

#### 问题 S2：双写路径，一致性风险（中等）
同一张 `facts` / `chat_messages` 表，有两条并行写入路径：
- `IngressStore`（同步 `sqlite3`）：channel/webhook 线程侧调用
- `FactStore` / `ChatMessageStore`（异步 `aiosqlite`）：主 loop 侧调用

两者共用 `build_fact_upsert` / `build_chat_message_insert` builder 函数，但连接管理、事务语义、提交时机完全不同，存在以下风险：
1. IngressStore 每次调用创建新连接，WAL 模式下与 aiosqlite 长连接并发可能竞争
2. IngressStore 的事务边界是单语句，主 loop 的 TaskStore 可能批量提交

**修复**：明确 `IngressStore` 的角色是"仅写入"的线程安全入口，文档化其与 aiosqlite 并发模型的关系；或统一改为通过队列异步投递，消灭同步写路径。

#### 问题 S3：IngressStore 混合读写职责（中等）
`IngressStore` 设计本意是"入口仓储"，但实际既有写入方法（`add_chat_message`、`set_fact`、`ingest_user_message`）也有读取方法（`list_pending_assistant_messages`、`get_fact`、`mark_chat_message_delivered`）。  
**修复**：拆分为 `IngressWriter`（只写）和 `IngressReader`（只读状态查询），职责单一后更容易替换为消息队列等机制。

#### 问题 S4：Store 子类缺乏基类（轻微，设计模式）
`ChatMessageStore`、`FactStore`、`RunStore`、`FailureStore` 等 7 个类有完全相同的构造函数：
```python
def __init__(self, db_getter: Callable[[], aiosqlite.Connection]) -> None:
    self._db_getter = db_getter

@property
def _db(self) -> aiosqlite.Connection:
    return self._db_getter()
```
没有基类，纯靠结构相同来保持一致性。新增 Store 子类时容易遗漏细节。  
**修复**：提取 `BaseAsyncStore` 基类，统一 `db_getter` 注入和 `_db` 属性。

#### 问题 S5：数据类与 Store 混居（中等）
`Run`、`Task`、`Failure` 等 dataclass 目前住在 `memory/task_store.py` 里（高层），但被低层 `store/memory/run.py` 通过 `TYPE_CHECKING` + 运行时 import 双重引用。数据类的"家"不在合适的层级。  
**修复**：数据类下沉到 `store/memory/models.py`，成为整个存储层的公共 DTO，上层 memory 模块只做业务逻辑。

---

### Doctor 诊断命令分析

`cli/diag.py` 的 `doctor()` 函数（~380 行）做了 12 项自检：Python 版本、依赖包、配置文件、API key、数据库、workspace 文件、插件、工具注册、模型探针等。

**发现的 4 个设计问题：**

#### 问题 D1：绕过所有抽象层直接查 DB（严重）
```python
# cli/diag.py 数据库检查段
import sqlite3
conn = sqlite3.connect(str(db_path))
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()]
```
Doctor 直接用 raw `sqlite3` 查 DB，完全绕过 `store/memory/` 层。这意味着：DB schema 变化时 doctor 需要单独维护；Store 层的任何迁移对 doctor 不可见。  
**修复**：Store 层提供 `health_check()` 接口，doctor 调用接口而不是直接查 DB。

#### 问题 D2：无 HealthCheck 抽象，不可测试、不可扩展（严重）
目前 12 项检查全部内联在一个 400 行函数里，没有任何抽象。无法：
- 单独测试某一项检查
- 通过插件或配置添加自定义检查项
- 在其他场景（如 CI/CD、运行时自检）复用检查逻辑

```python
# 现在                         # 应该
def doctor():                   class HealthCheck(Protocol):
    # 400行内联代码              def name(self) -> str: ...
    pass                        def check(self) -> CheckResult: ...
```
**修复**：抽象 `HealthCheck` 协议，每项检查实现一个类（`PythonVersionCheck`、`DependencyCheck`、`ConfigFileCheck`、`DatabaseCheck`、`ModelProbeCheck` 等），doctor 函数只负责运行、汇总、格式化输出。

#### 问题 D3：模型探针硬编码端点（中等）
```python
# cli/diag.py 模型探针段
_resp = _hc.post(f"{_base}/chat/completions", ...)  # 硬编码 chat/completions
```
不支持 `/responses` 端点的模型（如 `gpt-5.*`）会在探针阶段就失败（已在 repo memory 记录 `unsupported_api_for_model` 问题），但 doctor 不知道这个区别。  
**修复**：模型探针改为通过 `provider/` 层的 `Provider.ping()` 接口，由 provider 决定走哪个端点。

#### 问题 D4：Doctor 掌握了过多系统内部知识（中等）
Doctor 知道：API key 如何从环境变量、legacy credentials、auth profile 解析；工具 registry 如何 discover；插件目录在哪；Config schema 兼容性如何 patch。这些逻辑本属于各自的子系统，却被重复实现（或直接内联调用私有 API）在 doctor 里。  
**修复**：每个子系统暴露自己的 health check 接口，doctor 只是运行聚合器，不了解各子系统内部。

---

### Store + Doctor 整体修复优先级

| 问题 | 严重度 | 修复代价 | 建议阶段 |
|------|--------|---------|---------|
| S1：RunStore 逆向依赖 | 🔴 高 | 低（移动 dataclass） | 阶段 2（代谢器官）时一并处理 |
| D1：Doctor 绕过 Store 直查 DB | 🔴 高 | 低（Store 加 health_check 接口） | 阶段 1 |
| D2：无 HealthCheck 抽象 | 🔴 高 | 中（重构 doctor 函数） | 阶段 1 |
| S2：双写路径并发风险 | 🟠 中高 | 中（明确文档或队列化） | 阶段 2 |
| S3：IngressStore 读写混合 | 🟠 中高 | 低（拆分） | 阶段 2 |
| D3：模型探针硬编码端点 | 🟡 中 | 低（改用 Provider.ping()） | 阶段 1 |
| S4：缺 BaseAsyncStore 基类 | 🟡 中 | 低（纯重构） | 阶段 2 |
| S5：数据类层级错误 | 🟡 中 | 中（移动 + 修改 import） | 阶段 2 |
| D4：Doctor 掌握过多内部知识 | 🟡 中 | 中（各子系统加接口） | 阶段 3+ |

---

## 十、关键原则摘要

1. **生命不等于主脑**：记忆、人格、灵魂才是真正的生命本体。主脑是可升级、可替换的思考工具。
2. **写入只走代谢**：任何器官都不能绕过代谢器官直接写生命状态。
3. **宪法免疫先行**：违宪检查在主脑决策之前，不是之后。
4. **生命史只追加**：生命状态可以变，但历史不能抹去。
5. **门不是器官**：chat、webhook、微信只是信号通道，不参与生命运作。
6. **子灵全能但服从**：能力全集，但权力来自父灵授权。
7. **进化受生命把关**：除宪法外的一切可以改，但改之前必须通过免疫和连续性验证。
8. **主脑职能唯一**：外挂辅助模型是工具，不是主脑。
