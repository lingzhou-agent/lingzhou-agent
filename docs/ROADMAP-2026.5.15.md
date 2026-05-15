# lingzhou 架构路线图

**创建时间：** 2026-05-15 09:41
**基线版本：** v2026.5.15
**状态：** 设计阶段，待实施

---

## 一、当前状态（v2026.5.15 已完成）

### 已实现能力

| 能力 | 状态 | 代码位置 |
|------|------|---------|
| 认知循环（Perceive→Emotion→Ethos→Judgment→Execute→Memory→Evolve） | ✅ | `core/loop.py` |
| 情绪系统（OCC评价+Core Affect+离散情感+调节策略） | ✅ | `core/perception.py` |
| 价值观系统（5维度EMA+hard_axioms） | ✅ | `core/soul.py` |
| 判断层 + 模型路由（reader/reasoner/repair） | ✅ | `core/judgment.py` |
| 任务管理（8种状态+chain_id+parent_task_id+wait_kind） | ✅ | `tools/task_ops.py` + `memory/task_store.py` |
| 调度系统（一次性/重复信号+自动ack） | ✅ | `tools/schedule.py` |
| 进程管理（exec后台/PTY/超时+process管理） | ✅ | `tools/exec.py` |
| 文件操作（read/write/edit精确替换+list） | ✅ | `tools/file.py` |
| 自进化（importlib.reload热替换） | ✅ | `core/evolution.py` |
| 记忆系统（working/episodic/semantic三层） | ✅ | `memory/` |
| 事件驱动唤醒 | ✅ | `loop.py::_wait_for_event` |
| 跨重启连续性 | ✅ | `loop.py::_restore_state_from_db` |

---

## 二、双环系统体检报告

### 当前三层循环

| 层级 | 代码位置 | 作用 | 论文映射 | 状态 |
|------|---------|------|---------|------|
| 外环（认知环） | `loop.py::_tick()` | Perceive→Execute→Memory | Zimmerman SRL | ✅ 正常 |
| 内环（工具环） | `loop.py::_tick()` inner for | 连续工具调用直到回复 | ReAct | ⚠️ 仅chat模式 |
| 进化环 | `evolution.py::run()` | 失败→修改代码 | Argyris单环 | ⚠️ 只有单环 |

### 欠缺清单

| 欠缺 | 严重度 | 说明 |
|------|--------|------|
| 缺少双环学习（Double-Loop） | 🔴 P0 | 进化环只改代码，不质疑前提假设 |
| 内环仅chat模式 | 🔴 P0 | 自主循环无连续工具调用能力，多步任务效率低 |
| 缺少任务级模型路由 | 🟡 P1 | tier基于单个工具推断，非任务级别锁定 |
| 进化无效果验证 | 🟡 P1 | 改了之后是否变好，没有验证 |
| 进化无回滚机制 | 🟡 P1 | 改得更糟时无法自动恢复 |
| 缺少三环学习（Triple-Loop） | 🟢 P2 | 价值观/身份层面的反思机制 |

---

## 三、目标架构：三层双环 + 异步委派

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 1: 认知循环（Supervisor）                  │
│                                                                  │
│  职责：理解意图 → 制定计划 → 委派执行 → 监控结果 → 反思学习      │
│                                                                  │
│  内环（ReAct）：读→判断→再读→写，连续N轮                          │
│  外环（全认知）：感知→情绪→判断→执行→记忆→进化                    │
│                                                                  │
│  输出：DelegationSpec（委派规格）                                │
└────────────────────┬────────────────────────────────────────────┘
                     │ spawn delegation
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 2: 委派执行层（Worker）                    │
│                                                                  │
│  • exec-worker    : 执行 shell 命令（后台/PTY/超时）              │
│  • llm-worker     : 独立 LLM 调用（可指定不同模型/tier）          │
│  • tool-chain     : 连续工具链（read→write→verify）              │
│  • file-worker    : 文件操作（读/写/edit）                        │
│                                                                  │
│  特性：隔离 + 并行 + 可观测 + 可取消                              │
└────────────────────┬────────────────────────────────────────────┘
                     │ status change / completion event
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 3: 双环学习层（Meta-Learning）             │
│                                                                  │
│  Ring 1 - 单环：怎么做得更好（改代码/改prompt/改阈值）            │
│  Ring 2 - 双环：前提假设对吗（质疑tier分类/judgment/阈值/技能）   │
│  Ring 3 - 三环：学习目标对吗（ethos/身份/价值观冲突）            │
│                                                                  │
│  进化验证 + 自动回滚                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 核心数据流

```
用户消息 / 调度信号 / 心跳
         │
         ▼
┌─────────────────┐
│   认知环判断     │ ← 问：这个任务需要什么？
└────────┬────────┘
         ├─── 简单任务 ──→ 直接在环内执行
         └─── 复杂任务 ──→ 创建 Delegation
                           │
                           ▼
                    ┌─────────────────┐
                    │  DelegationSpec  │
                    └────────┬────────┘
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
         ┌─────────┐  ┌─────────┐  ┌─────────┐
         │ exec    │  │ llm     │  │tool-chain│
         │ worker  │  │ worker  │  │ worker   │
         └────┬────┘  └────┬────┘  └────┬────┘
              └─────────────┼───────────┘
                            │ 事件驱动
                            ▼
                    ┌─────────────────┐
                    │   主循环唤醒     │ ← 问：结果如何？需要调整吗？
                    │   整合结果       │
                    └────────┬────────┘
                             │
                    ┌────────┼────────┐
                    ▼        ▼        ▼
                  成功     失败     进行中
                    │        │        │
                    ▼        ▼        ▼
                结晶     双环分析    继续监控
                记忆     前提质疑    等待事件
```

---

## 四、DelegationSpec 设计

```python
@dataclass
class DelegationSpec:
    """委派规格——主循环创建的"工作订单"。"""
    
    id: str                          # UUID
    task_id: int                     # 所属 task
    worker_type: str                 # exec/llm/tool_chain/file
    description: str                 # 人类可读的描述
    
    # 执行配置
    model_tier: str = "reasoner"     # 使用的模型 tier
    tool_chain: list[str] = field(default_factory=list)  # 工具链
    command: str = ""                # exec 命令
    timeout_seconds: float = 300.0   # 超时
    max_retries: int = 2             # 最大重试
    
    # 上下文
    environment: dict = field(default_factory=dict)
    workdir: str = ""
    input_data: dict = field(default_factory=dict)
    
    # 状态
    status: str = "pending"
    progress: float = 0.0
    result: str = ""
    error: str = ""
    started_at: str = ""
    completed_at: str = ""
    logs: list[str] = field(default_factory=list)
    
    # 双环分析字段
    failure_analysis: str = ""
    root_cause: str = ""
```

---

## 五、双环触发矩阵

```
失败类型                触发环      行动
─────────────────────────────────────────────────
工具执行失败（命令报错）  单环    修复工具代码/修改prompt
策略失败（方向错了）      双环    质疑判断逻辑/tier分类/阈值
任务失败（目标未达成）    双环    质疑任务分解/工具选择
价值观冲突               三环    质疑ethos基线/hard_axioms
```

---

## 六、实施路线图

### Phase 1（当前可做，1-2天）

| 任务 | 优先级 | 预估时间 | 依赖 |
|------|--------|---------|------|
| P1-1: 多模态/视觉（image.analyze 工具） | P0 | 0.5天 | 无 |
| P1-2: 自主循环内环（无用户消息时连续工具调用） | P0 | 1天 | 无 |
| P1-3: 进化效果验证（改了之后对比成功率） | P1 | 1天 | 无 |
| P1-4: 进化回滚机制（改坏自动恢复） | P1 | 0.5天 | P1-3 |

### Phase 2（本周可做）

| 任务 | 优先级 | 预估时间 | 依赖 |
|------|--------|---------|------|
| P2-1: Task-Level Model Routing（task.data增加model_tier） | P1 | 1天 | 无 |
| P2-2: Delegation 概念引入（task增加delegate字段） | P1 | 1天 | 无 |
| P2-3: 主循环委派判断（_should_delegate + _spawn_delegate） | P1 | 1天 | P2-2 |
| P2-4: 双环学习层（Double-Loop judgment质疑） | P1 | 2天 | 无 |

### Phase 3（下周可做）

| 任务 | 优先级 | 预估时间 | 依赖 |
|------|--------|---------|------|
| P3-1: Worker 执行器（subprocess + LLM call） | P1 | 2天 | P2-2, P2-3 |
| P3-2: 事件驱动唤醒（delegate完成唤醒主循环） | P1 | 1天 | P2-3 |
| P3-3: 多delegate并行（asyncio.gather） | P2 | 1天 | P3-1 |
| P3-4: 三环学习（ethos/身份反思） | P2 | 2天 | P2-4 |

---

## 七、权威论文参考

| 论文 | 年份 | 映射 |
|------|------|------|
| Argyris, C. "Single-Loop and Double-Loop Models in Research on Decision Making" | 1976 | Ring 1/Ring 2 学习理论 |
| Wu et al. "Agent Workflow: A Survey" | 2024 | Supervisor-Worker 架构 |
| Meta "Multi-Agent Systems: A Survey" | 2025 | Delegation 模式 |
| Wang et al. "Plan-and-Solve Prompting" | 2023 | 规划→执行→检查闭环 |
| Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning" | 2023 | 双环纠偏原则 |
| Zimmerman, B. J. "Self-Regulated Learning" | 2000 | Forethought→Performance→Self-Reflection |

---

*本文档在 v2026.5.15 tag 基础上创建，随实施进展更新。*
