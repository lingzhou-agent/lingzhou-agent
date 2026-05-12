# lingzhou 系统蓝图

---

## 1. 定位

lingzhou 是一个**数字生命种子**，不是普通聊天壳、工具编排器或 coding shell。

它围绕以下核心设计：
- **身份连续性**：跨 chat 保持记忆、情绪基线与价值观
- **感知驱动**：感知异常（高预测误差、WM 溢出）本身就是行动理由
- **任务优先**：task 是持久推进单元，chat 是对话任务（门 + 房间）
- **自编程自进化**：同进程内热替换工具代码，不依赖重启

---

## 2. 核心架构立场

### 2.1 任务优先的连续性
Chat 是对话入口，也是持久的 chat task。真正承载长期主线的是 task 与 memory。

> **设计原则：chat 是门，也是房间。**

### 2.2 反思判断层居中
中心决策面不是脚本堆叠，而是对结构化证据（任务、感知、情绪、记忆、Ethos）做反思性判断的 LLM 调用。

### 2.3 本地优先
SQLite + 文件，单机即可完整运行。不依赖云服务。

### 2.4 感知与情绪作为运行时模块
不是装饰，是影响谨慎度、优先级和策略切换的真实信号。

### 2.5 运行时自修改优势
运行时生成工具代码 → `compile()` 语法检查 → `importlib.reload` 热替换，无需停止进程重启。

---

## 3. 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│  Bootstrap Layer（生命内核）                                     │
│  init 命令 · Soul 播种 · workspace Markdown 镜像 · 冷启动恢复    │
├─────────────────────────────────────────────────────────────────┤
│  Skill System（技能防线）                                        │
│  SkillRegistry · 5 内置技能 · MatchForContext 上下文激活         │
├─────────────────────────────────────────────────────────────────┤
│  Perception（感知层）                                            │
│  Percept · workspace 指纹 · 预测误差 · 内部任务生成               │
├─────────────────────────────────────────────────────────────────┤
│  Emotion（情绪层）                                               │
│  OCC 评价 → Core Affect（V/A/D）→ 离散情感 → Regulation 策略     │
├─────────────────────────────────────────────────────────────────┤
│  Ethos（价值层）                                                 │
│  EthosValues · EthosBias · EMA 历史基线 · JudgmentSignals         │
├─────────────────────────────────────────────────────────────────┤
│  Judgment（判断层）                                              │
│  Bundle 组装 → judgment.md 模板 → LLM → JudgmentOutput          │
│  确定性回退：hard_boundary > posture > wait                       │
├─────────────────────────────────────────────────────────────────┤
│  Execution（执行层）                                             │
│  act / pause / wait dispatch · 失败记录绑定 task_id              │
├─────────────────────────────────────────────────────────────────┤
│  Evolution（进化层）                                             │
│  synthesize_tool · evolve_tool · 热重载 · 语法检查 · 备份         │
├─────────────────────────────────────────────────────────────────┤
│  Memory Spine（记忆脊柱）                                        │
│  WorkingMemory（有界优先堆）                                      │
│  EpisodicMemory（SQLite events + task-{id}.md 叙事）             │
│  SemanticMemory（SQLite nodes + FTS5 + activation 衰减检索）     │
│  TaskStore（SQLite ACID：tasks / failures / facts）               │
├─────────────────────────────────────────────────────────────────┤
│  BehaviorTracker（行为循环检测 + 执行门控）                       │
│  action streak · read streak · explore budget · apply_gate       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 主循环（一个 tick）

```
感知
  └→ Percept（预测误差 + workspace 指纹）
  └→ record_event("perception") → SQLite events 表
  └→ build_perception_replay()

好奇心驱动任务生成
  └→ _maybe_curiosity_task()（确定性触发：空闲 + curiosity 阈值）→ task_store.add_task(source="curiosity")

情绪更新
  └→ emotion.derive_from_signals() → OCC 评价链
  └→ record_event("emotion") → SQLite events 表
  └→ build_emotion_replay()

Ethos + JudgmentSignals 推导
  └→ derive_ethos_state() → EthosState → EMA 写回 DB
  └→ compute_judgment_signals() → JudgmentSignals

技能上下文激活
  └→ skill_registry.match_for_context() → active_skills

判断
  └→ _assemble_context() → Bundle（含 skill_section / ethos_section / hard_boundaries）
  └→ LLM → JudgmentOutput
     ├→ decision: act | pause | wait
     ├→ chosen_action_id / params
     ├→ rationale（内部独白）
     ├→ reflection（结构化反思洞察 → 写入语义记忆 + 事件结晶）
     └→ next_step

行为门控
  └→ BehaviorTracker.on_act() → 追踪 action/read streak + explore budget
  └→ BehaviorTracker.apply_execution_gate() → 循环时强制 wait

执行
  └→ ExecutionLayer.dispatch()
  └→ 失败 → task_store.record_failure(task_id=...)

记忆整合
  └→ rationale → episodic.record("assistant")
  └→ result → wm.add(WMItem)
  └→ 每 consolidate_every 轮（WM 有压力时）→ episodic 整合 + soul.sync_md()

定期进化（双触发：高错误连续 ≥ 3 OR cycle % evolve_every == 0）
  └→ evolution.run() → evolve_tool() → importlib.reload
  └→ soul.refresh_identity() → 更新 system prompt prefix
```

---

## 5. 运行时文件布局

```
~/.lingzhou/
├── state/
│   └── runtime.db              ← 主 SQLite（ACID 状态机）
├── memory/
│   ├── events (SQLite)         ← 感知 / 情绪时序事件（主路径）
│   ├── events.jsonl            ← 降级备份（DB 不可用时 fallback）
│   ├── global.md               ← 全局情节叙事（无任务绑定时）
│   ├── task-{id}.md            ← 每个任务的情节叙事流
│   └── nodes/
│       └── {id}.json           ← 语义记忆节点（SQLite 主存，json 为同步镜像）
└── workspace/                  ← 人类可读镜像（不是运行时真相）
    ├── BOOTSTRAP.md            ← 冷启动行为指引（注入 system prompt prefix）
    ├── IDENTITY.md             ← 身份叙述（注入 system prompt prefix）
    ├── SOUL.md                 ← soul facts 镜像（init + sync_md 写入）
    ├── USER.md                 ← 用户偏好/关系描述（注入 WM）
    ├── TOOLS.md                ← 工具能力说明（注入 WM）
    ├── HEARTBEAT.md            ← 心跳行为触发说明（注入 WM）
    ├── DREAMS.md               ← 长期志向（evolve 阶段自动更新）
    └── skills/                 ← 本地扩展技能目录
        └── {skill_name}/
            └── SKILL.md
```

---

## 6. 设计理论依据

| 机制 | 理论来源 |
|---|---|
| OCC 情绪评价 | Ortony, Clore, Collins (1988) |
| Core Affect V/A | Russell (2003) |
| Regulation 策略 | Gross (1998) |
| Ethos EMA 演化 | Kruschke (2006)；Schwartz (1992) |
| 情节叙事四元素绑定 | Tulving (1983) |
| 任务叙事连续性 | Ricoeur (1984) |
| 来源监控 | Johnson & Raye (1981) |
| 多锚点收敛激活 | Anderson (1983) ACT-R |
| 工作记忆有界优先队列 | Baddeley (2000) |
| activation 衰减 | Ebbinghaus (1885) |
| TOTE 自适应 sleep | Miller (1960) |
| JudgmentSignals 预判 | Friston (2010) 主动推断 |

---

## 7. 对外实事求是

**已实现**：
- 完整 OCC 情绪链（Appraisal→Feeling→CoreAffect→Regulation）
- Ethos 价值层 + JudgmentSignals 预判 + EMA 写回 DB（每 tick 持久化）
- 四层记忆系统（WM / 情节 / 语义 / SQL）
- 5 内置技能 + MatchForContext 上下文激活（连续评分）
- 自进化引擎（运行时合成 + 热替换）
- SQL schema 自动补列（`_migrate()`）
- `reflection` → 语义记忆 upsert + 事件结晶（loop.py step 7/7b）
- `BOOTSTRAP.md` / `IDENTITY.md` 注入 system prompt（`soul.bootstrap()` + `judgment.set_identity_prefix()`）
- activation 衰减（Ebbinghaus，`semantic.py:effective_activation(decay_lambda)`）
- 事件存储迁移至 SQLite + 按类型轮转（`_rotate_events_db()`，O(log n) 检索）
- FTS5 关键词索引（`semantic.py:_setup_fts5()`，可选启用）

**尚未完整实现**：
- 向量嵌入检索（`embed_fn` 接口预留，`embedding_weight=0.3`，当前无调用方传入）
- task-{id}.md 叙事分段摘要（当前靠 `load_for_context` 末尾截断）
