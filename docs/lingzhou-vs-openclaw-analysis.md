# lingzhou vs OpenClaw/Hermes 能力对比分析

**创建日期：** 2026-05-14
**更新日期：** 2026-05-15（v2026.5.15 基线代码审查）
**前提：** 同等大模型（相同 provider、相同 model），能力差距不在 LLM，而在架构和执行面。

---

## 基线状态（v2026.5.15）

### 已实现能力（代码审查确认）

| 能力 | 状态 | 代码位置 | 行数 |
|------|------|---------|------|
| file.read/write/edit(精确替换) | ✅ | `tools/file.py` | 294 |
| exec/process(后台/PTY/超时/持久化) | ✅ | `tools/exec.py` | 864 |
| shell.run | ✅ | `tools/shell.py` | 168 |
| schedule(一次性/重复/自动ack) | ✅ | `tools/schedule.py` | 181 |
| task.add/advance/complete/fail/wait/resume/update/list | ✅ | `tools/task_ops.py` | 281 |
| memory.get_fact/set_fact/add_wm/add_semantic | ✅ | `tools/memory_ops.py` | 234 |
| 向量检索（text-embedding-v3） | ✅ | `memory/semantic.py` + config | — |
| 认知循环（7步完整） | ✅ | `core/loop.py` | 1222 |
| 情绪系统（OCC+Core Affect+调节） | ✅ | `core/perception.py` | 559 |
| 判断层 + 模型路由（reader/reasoner/repair） | ✅ | `core/judgment.py` | 1380 |
| 自进化（importlib.reload 热替换） | ✅ | `core/evolution.py` | 482 |
| 事件驱动唤醒 | ✅ | `loop.py::_wait_for_event` | — |
| 跨重启连续性 | ✅ | `loop.py::_restore_state_from_db` | — |
| 内层工具环（连续工具调用） | ✅ | `loop.py` inner for | 仅chat模式 |

**工具总数：** 9 个模块，~2300 行工具代码，~5700 行核心代码，~2000 行记忆代码。

---

## 一、核心结论

> **lingzhou 的认知架构（感知→情绪→Ethos→判断→执行→记忆→进化）设计精良，在认知建模深度上超过了 OpenClaw。执行基础设施在 v2026.5.15 已达到可用水平（file.edit + exec/process + schedule + 向量检索均已实现）。**
>
> **当前最大差距不在工具数量，而在：**
> 1. 缺少子代理/委派系统（所有任务串行执行）
> 2. 进化环只有单环学习（不质疑前提假设）
> 3. 内层工具环仅 chat 模式可用（自主循环效率低）
> 4. 缺少任务级模型路由（tier 按工具推断，不按任务锁定）

---

## 二、量化对比（v2026.5.15 基线）

| 维度 | OpenClaw | lingzhou | Hermes |
|------|----------|----------|--------|
| **工具模块数** | 17+ 内置 | 9 模块（27+ 工具） | 36 skills |
| **文件编辑** | read/write/edit | read/write/edit | read/write |
| **进程管理** | exec+process | exec+process | 无 |
| **定时任务** | cron（systemEvent/agentTurn） | schedule（WM注入+自动ack） | cron/ |
| **会话管理** | sessions_spawn/send/list/history | ❌ | ACP（被动被调） |
| **视觉能力** | image 工具 | ❌ | ❌ |
| **计划编排** | update_plan + TaskFlow | task.*（8种状态+chain） | kanban.db |
| **多通道** | 7+ 通道 | CLI | 微信+CLI |
| **插件系统** | 完整 SDK | ❌ | ❌ |
| **Skills 生态** | 21+ 标准技能 | 5 内置防线 | 36 skills |
| **记忆检索** | 向量语义（本地 embedding） | 向量语义（远程 embedding） | state.db |
| **情绪系统** | ❌ | ✅ OCC+Core Affect | ❌ |
| **价值观** | SOUL.md（自然语言） | ✅ 5维度EMA+hard_axioms | 极简 |
| **自进化** | 双环学习+治理审计 | ✅ importlib.reload（单环） | ❌ |
| **模型路由** | 单一+fallback | ✅ reader/reasoner/repair | 单一+fallback |
| **代码沙箱** | 多级别（deny/allowlist/full） | ❌ | ❌ |

---

## 三、lingzhou 的欠缺（按严重度排序）

### 🔴 P0 — 致命欠缺

| 欠缺 | 说明 | 影响 |
|------|------|------|
| **子代理/委派系统** | 所有任务在同一循环内串行执行，不能并行 | 复杂任务效率低，无法隔离上下文 |
| **内环仅 chat 模式** | 自主循环每次 tick 只调用一次工具，然后等 idle gap | 5步任务至少花 4×idle_gap 时间 |
| **只有单环学习** | 进化环只改代码/prompt/阈值，不质疑前提 | 方向性错误不会被纠正 |

### 🟡 P1 — 严重欠缺

| 欠缺 | 说明 | 影响 |
|------|------|------|
| **任务级模型路由** | tier 按单个工具推断，不按任务锁定 | 同一任务频繁切换模型，上下文丢失 |
| **进化效果验证** | 改了之后没有对比验证 | 可能"越改越差" |
| **进化回滚** | 改坏后无法自动恢复 | 进化有风险 |
| **多模态/视觉** | 无 image 工具 | 无法处理图片/截图 |

### 🟢 P2 — 中等欠缺

| 欠缺 | 说明 | 影响 |
|------|------|------|
| **三环学习** | ethos/身份/价值观冲突的反思 | 深层价值观进化缺失 |
| **代码沙箱** | exec 无隔离 | LLM 误操作风险（但单用户环境风险较低） |
| **多通道** | 只有 CLI | 使用场景受限（但优先级不高） |

---

## 四、架构层面的根本差异

### OpenClaw：平台化架构

```
Gateway（进程守护）
  ├── Agent（模型调用 + 工具执行 + 思考链）
  ├── Session（隔离 + 状态保持 + 子代理）
  ├── Tool（17+ 内置工具）
  ├── Skill（SKILL.md 标准）
  ├── Cron（定时任务）
  ├── Channel（7+ 通道）
  └── Plugin SDK（可扩展）
```

**优势：** 执行能力强，通道丰富，插件生态好。
**劣势：** 认知层薄（无情绪模型、无量化价值观）。

### lingzhou：认知原型架构

```
CognitionLoop（单进程循环）
  ├── Perception（感知 + 预测误差）
  ├── Emotion（OCC评价 → Core Affect → 离散情感 → 调节策略）
  ├── Ethos（5维度EMA + hard_axioms）
  ├── Judgment（reader/reasoner/repair 模型路由）
  ├── Execution（工具分发 + 失败追踪）
  ├── Memory（WM → Episodic → Semantic → 结晶）
  └── Evolution（importlib.reload 热替换）
```

**优势：** 认知建模深，情绪系统完整，价值观可演化。
**劣势：** 执行串行化，缺委派系统，进化只有单环。

### Hermes：工具型 Agent

```
Gateway（HTTP + ACP 协议）
  ├── Agent（OpenAI SDK，max_turns=90）
  ├── Skill（36 个）
  ├── Tool（hermes-cli + 内建）
  ├── Memory（memories/ + state.db）
  └── Channel（微信 + CLI + ACP）
```

**优势：** 工具数量多，多模型 fallback。
**劣势：** 人格化弱，gateway 稳定性差。

---

## 五、lingzhou 的亮点（不应丢弃）

1. **OCC 情绪模型**：确定性信号推导情绪，影响决策策略，有研究基础
2. **Ethos EMA 演化**：价值观基线随经历缓慢漂移，身份连续性好设计
3. **四层记忆架构**：WM/Episodic/Semantic/Facts，理论框架优秀
4. **运行时自修改**：`importlib.reload()` 热替换，Python 独有能力
5. **感知驱动循环**：预测误差+WM压力+情绪激活作为行动理由
6. **行为循环检测**：BehaviorTracker 防 LLM 幻觉循环
7. **模型分层路由**：reader/reasoner/repair + 健康状态 + 冷却窗口
8. **事件驱动唤醒**：非固定时钟节拍，由 chat 消息/task 变化/超时唤醒

---

## 六、目标架构：三层双环 + 异步委派

详见 `ROADMAP-2026.5.15.md`。

核心方向：
1. 外环保持完整认知循环
2. 新增委派层（DelegationSpec + Worker），复杂任务异步执行
3. 自主循环也支持内环（不只是 chat 模式）
4. 进化环升级为三层：单环（改代码）+ 双环（质疑前提）+ 三环（反思价值观）
5. 新增进化验证和回滚

---

*本文档在 v2026.5.15 tag 基础上更新，反映代码审查后的真实状态。*
