# 架构设计

[中文](ARCHITECTURE.md) | [English](ARCHITECTURE.en.md)

## 认知循环

```
         ┌──────────────┐
         │  感知层      │ ← 工作记忆(WM) + 情节记忆(episodic) + 预测误差
         │  Perception  │
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │  好奇心引擎   │ ← Novelty + Learning Progress + Surprise
         │  Self-Drive  │
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │  判断层      │ ← LLM 决策 (act/wait/pause) + 工具选择
         │  Judgment    │
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │  执行层      │ ← 内置工具，内层 continue 循环
         │  Execution   │
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │  反思层      │ ← 情节整合 + 语义编译 + 情绪更新
         │  Reflection  │
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │  进化引擎    │ ← 失败模式检测 → LLM 生成修复 → 热加载
         │  Evolution   │
         └──────────────┘
```

## Tick 调度与并发边界

主循环的并发目标不是“任意 tick 全乱序执行”，而是“无关联 tick 并发、有关联 tick 保序”。

- 同一条任务连续体上的 tick 必须保持 FIFO 顺序。这里的顺序依赖来自 `next_step`、`last_action_*`、`pending_tier`、`ticks_since_judge`、停滞计数等跨 tick 状态。
- 不同任务或无共享连续状态的 tick 可以并发执行，以降低 autonomous LLM 调用阻塞 chat 响应的问题。
- runtime 应通过有界 dispatcher 管理 tick：同 chain 串行、跨 chain 并发，并通过全局并发上限和等待队列上限限制资源占用。
- 设计目标是改善响应性和吞吐，不改变相关工作项的因果顺序；任何带顺序依赖的 tick 都应继续排到对应 chain 的后面。

## 核心模块

### `core/loop/runtime.py` — 主循环 (CognitionLoop)
编排感知→判断→执行→反思全流程。事件驱动等待（chat/task/超时）。包含热配置重载。

在并发 tick 模式下，runtime 还负责：
- 维护全局共享资源（provider / task_store / 记忆系统）
- 通过 dispatcher 将 tick 分配到各自 chain
- 保证同 chain 顺序执行、跨 chain 在上限内并发

`core/loop/__init__.py` 只保留稳定导出。

### `core/judgment/runtime.py` — 判断层 (JudgmentLayer)
LLM 决策引擎：接收 WM + 信号 → 决定 action + tool。支持多模型路由 (reader/reasoner/repair)。内层 continue 循环：多次工具调用不重装上下文。`core/judgment/__init__.py` 只保留稳定导出，context/format helper 已拆到 `core/judgment/context.py`。

**工具元数据**：工具通过 `ToolManifest` 自声明 tier 偏好，减少硬编码：

```python
@tool(ToolManifest(name="file.read", progress_category="info", prefer_tier="reader"))
```

**tier 路由**：`tool_tier(tool_id, registry)` 优先读 manifest.prefer_tier，回退到硬编码集合。`model_strategy` 还支持 `next_phase_tier` / `thinking_override` / `routing_overrides` 在 tick 间调整推理姿态。

**演进方向**：从 tool-level routing 升级到 task-level routing；Judgment 未来需要能选择"创建 Run"而非永远在主循环内串行推进；双环反思应拆到独立 MetaReflection，不在 Judgment 内做。

### `core/perception/` — 感知层 (PerceptionLayer)
从 WM/emotion/episodic 计算预测误差、认知信号。拆分为四个子模块：
- `emotion.py` — OCC 情绪模型（Appraisal / EmotionState / 重放摘要）
- `ethos.py` — 价值层（EthosValues / EthosState / derive_ethos_state）
- `signals.py` — 判断信号与认知信号（JudgmentSignals / CognitiveSignals）
- `layer.py` — 感知层入口（Percept / PerceptionLayer）

`core/perception/__init__.py` 保留所有公开导出，外部调用路径不变。

### `core/self_drive.py` — 自驱力引擎 (SelfDriveEngine)

基于 **Active Inference**（Friston 2013）和 **Intrinsic Motivation**（Oudeyer & Kaplan 2007），综合三种内在驱动力：

| 信号 | 含义 |
|------|------|
| Novelty `C_novelty(t)` | 最近 N tick 中接触的新颖知识比例 |
| Learning Progress `C_progress(t)` | 能力提升速率（完成任务的复杂度趋势） |
| Surprise `C_surprise(t)` | 预测误差均值 |

综合信号 `C(t) = α·C_novelty + β·C_progress + γ·C_surprise`。

**空闲触发逻辑**：loop 无用户消息且无活跃任务时，`C(t)` 超阈值触发自主探索，低于阈值触发自我反思 + 目标生成。LLM 以"内心感知"叙事形式接收驱动信号，自主决定是否响应。

### `core/evolution.py` — 进化引擎 (EvolutionEngine)
检测失败模式 → LLM 生成改进代码 → 语法验证 → 热重载 → 注册验证 → 回滚。后进化验证确保系统可导入。

### `core/behavior_tracker.py` — 行为追踪 (BehaviorTracker)
追踪重复 action/read/list/edit 模式。将探针信号注入 WM 供 LLM 感知，不机械阻塞。

### `core/plugin.py` — 插件系统 (PluginManager)
discover → load → register → start 生命周期。启动时自动加载 plugins/ 目录。

## 记忆系统

### 工作记忆 (WM)
LLM 上下文窗口内的短期记忆。容量和 token 预算可配。

### 情节记忆 (Episodic)
events.jsonl 追加式记录。每次 tick 的 perception/emotion/action 结果。

### 语义记忆 (Semantic)
向量化长期记忆。支持 embedding 混合搜索。任务完成时自动编译叙事。

### 任务存储 (TaskStore)
`memory/task_store.py` 是公共 façade；底层 SQLite 持久化 helper 统一收口在 `store/memory/`。当前持久化主线覆盖 tasks / chat_messages / failures / facts / signals / runs / meta_reflections。

## 工具系统

`tools/` 目录下的所有 Python 文件自动发现。每个工具：
- `@tool(ToolManifest(...))` 装饰器声明
- 异步函数 `async def xxx(params, ctx) -> ToolResult`
- 自动注册到 ToolRegistry

## 通道架构

三个 IO 通道并行运行：
- **local** — 终端交互 (lingzhou chat)
- **wechat** — 微信 iLink 通道
- **webhook** — HTTP 接入

通道 sidecar 在 daemon 线程中运行，与主 asyncio loop 并行。

## 当前实现差距

以下内容已按当前源码重新核对，只保留仍未完全闭环的点；已经由代码和测试覆盖的能力不再列为差距。

### P0：核心闭环仍待补齐

| ID | 蓝图要求 | 当前状态 | 偏差评估 |
|----|----------|----------|----------|
| P0-1 | 自动多模态感知闭环 | `tools/image.py`、`core/worker.py` 与相关测试已证明 `image.analyze`、`multimodal-worker`、视觉模型路由已落地；但 `core/perception/` 仍无直接多模态入口 | 若蓝图目标是“感知层原生消费多模态”，当前仍主要依赖 Judgment 主动调用工具，而不是常驻感知通路 |
| P0-2 | Task-level model routing 自动闭环 | `task.model_tier`、`_prefer_tier_for_task()`、`_apply_tick_model_strategy()` 已存在，task 已能持久化 tier 并参与后续路由 | 已部分实现；但 routing guard / meta reflection 仍只生成提示，默认不会自动改写 `task.model_tier`，且持续偏好目前主要覆盖 `reasoner/repair` |

### P1：结构成熟度

| ID | 蓝图要求 | 当前状态 | 偏差评估 |
|----|----------|----------|----------|
| P1-1 | 引入 Run 抽象（Task/Run 分离）的完整语义 | `Run` 存储、`WorkerLayer`、`refresh_running_runs()`、`build_task_run_result_patch()` 已形成主线 | 主线已落地；若蓝图要求更强的 control-plane / execution-plane 分离、Run ownership 和更丰富的生命周期语义，当前仍偏轻量 |
| P1-2 | MetaReflection 自动执行闭环 | `build_meta_reflection()`、`meta_reflections`、`_ingest_actionable_meta_reflections()` 已实现，并能写入 WM / fact / semantic memory | 双环反思基础设施已存在；剩余差距在于多数建议仍需 LLM 或工具显式确认后写回，不是全自动策略机 |

### P2：质量增强

| ID | 蓝图要求 | 当前状态 | 偏差评估 |
|----|----------|----------|----------|
| P2-1 | 进化效果验证（before-after 对比） | `core/evolution.py` 已有 smoke / rollback 护栏，但未见统一的 before-after 结构化评分闭环 | 部分实现 |

### 小结

已经落地、因此不应再列为 gap 的能力包括：自主内环、Worker executors、Run 状态回流 Task、progress crystal、多 worker / 多任务并发，以及 MetaReflection 基础设施。
当前更准确的未完成点是：自动多模态感知闭环、task-level routing 的自动写回、Run 抽象的进一步成熟，以及结构化进化评估。
