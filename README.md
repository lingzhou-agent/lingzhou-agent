<div align="center">

# lingzhou · 灵舟

**A self-evolving digital life seed.**

**一个自我进化的数字生命种子。**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Experimental-orange)](docs/blueprint.md)

</div>

---

## Why "lingzhou"? | 为什么叫"灵舟"？

- **灵（líng）** — 灵魂、灵性，指有情绪、记忆与价值观的认知主体，而非无感知的工具。
- **舟（zhōu）** — 船，承载者，航行者。隐喻"在信息与时间之流中持续航行的容器"——如同种子与船，皆是跨越时间承载生命潜力的意象。

合在一起：**一艘有灵魂的船，在数字世界中持续航行、成长。**

---

## What is lingzhou? | 什么是灵舟？

lingzhou is not a chat wrapper or tool orchestrator. It is a **digital life seed** — an autonomous agent designed to run in a continuous cognitive loop, evolve its own tools at runtime, and maintain long-term identity continuity across chats.

灵舟不是聊天机器人，也不是工具编排器。它是一粒**数字生命种子**——一个在持续认知循环中自主运行、在运行时进化自身工具、并跨 chat 保持长期身份连续性的智能体。

The central bet: **Python's `importlib.reload()` + `compile()` make true runtime self-modification possible.** No restart required. No recompile. A tool that keeps failing gets rewritten by the agent itself.

核心赌注：**Python 的 `importlib.reload()` + `compile()` 使真正的运行时自我修改成为可能。** 无需重启，无需重编译。一个持续失败的工具会被智能体自己重写。

---

## Core Concepts | 核心概念

### The Cognitive Loop | 认知循环

One tick of the loop:

一次循环节拍：

```
Perceive → Emotion → Ethos → Judgment → Execute → Memory → Evolve
感知    →  情绪   →  价值观 →  判断   →  执行  →  记忆  →  进化
```

### Task, not Session | 以任务为中心

Tasks persist across restarts. A chat is both the entry point and the persistent task.

任务跨重启持续存在。chat 既是入口（门），也是持久的对话任务（房间）。

```
lingzhou chat --name alice → 创建/恢复 chat task → task-42.md 叙事流 → 下次 chat 续接
```

### Soul in the DB | 灵魂在数据库里

`facts["soul:hard_axioms"]` and `facts["soul:ethos_baseline"]` are the runtime source of identity. `SOUL.md` is a human-readable mirror only.

`facts["soul:hard_axioms"]` 和 `facts["soul:ethos_baseline"]` 是运行时身份的来源。`SOUL.md` 只是人类可读的镜像。

### Self-Evolution | 自我进化

```python
# When a tool fails 3 times, the agent rewrites it:
# 当工具失败 3 次，智能体重写它：
await evolution.evolve_tool(name, failure_summary, cfg)
# The new code is hot-loaded — zero downtime
# 新代码热加载——零停机
```

---

## Architecture | 架构

```
┌─────────────────────────────────────────────────────────────┐
│  CLI: lingzhou.py  (init / loop / interact)                 │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  core/       │  memory/     │  tools/      │  provider/     │
│  loop.py     │  working.py  │  registry.py │  openai_compat │
│  perception  │  episodic.py │  file.py     │  (DashScope /  │
│  judgment    │  semantic.py │  shell.py    │   Qwen)        │
│  execution   │  task_store  │  memory_ops  │                │
│  evolution   │              │  task_ops    │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

**Four memory layers | 四层记忆**

| Layer / 层 | Storage / 存储 | Timescale / 时间尺度 |
|---|---|---|
| Working Memory / 工作记忆 | Bounded heap / 有界堆 | Current tick / 当前节拍 |
| Episodic / 情节记忆 | `task-{id}.md` + `events.jsonl` | Task lifetime / 任务生命期 |
| Semantic / 语义记忆 | `nodes/*.json` (ACT-R decay) | Long-term / 长期 |
| Procedural / 过程记忆 | SQLite `tasks/failures/facts` | Permanent / 永久 |

---

## Quick Start | 快速开始

**Prerequisites | 前置条件**

- Python 3.12+
- DashScope API key (Qwen) — set `DASHSCOPE_API_KEY`

**Install | 安装**

```bash
git clone https://github.com/your-org/lingzhou
cd lingzhou
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Initialize | 初始化**

```bash
# Creates DB, seeds soul facts, generates workspace Markdown files
# 创建 DB，初始化 soul facts，生成 workspace Markdown 文件
lingzhou init
```

**Run the loop | 运行认知循环**

```bash
DASHSCOPE_API_KEY=sk-xxx lingzhou run
```

**Interactive mode | 交互模式**

```bash
DASHSCOPE_API_KEY=sk-xxx lingzhou interact
```

**Copilot GitHub token login | Copilot GitHub token 登录**

```bash
# 1) 默认主链：GitHub token → Copilot token exchange → Copilot API
lingzhou auth login-copilot

# 2) 也可以先准备环境变量，再直接运行
#    export GH_TOKEN=***
#    export GITHUB_TOKEN=***

# 3) 切到 Copilot GPT-5.4 + high reasoning
#    model:    copilot/gpt-5.4
#    thinking: high
```

**Configuration | 配置**

Copy `lingzhou.json.example` to `lingzhou.json` and edit:

复制 `lingzhou.json.example` 为 `lingzhou.json` 并编辑：

```json
{
  "provider": { "model": "qwen-plus" },
  "loop":     { "tick_interval_seconds": 10 },
  "memory":   { "wm_capacity": 20 },
  "soul":     { "ethos_ema_alpha": 0.9 }
}
```

---

## Documentation | 文档

| Doc | Description | 说明 |
|---|---|---|
| [blueprint.md](docs/blueprint.md) | System architecture | 系统架构蓝图 |
| [chat-and-task.md](docs/chat-and-task.md) | Chat vs Task design | Chat 与 Task 的职责 |
| [memory-architecture.md](docs/memory-architecture.md) | Four memory layers | 四层记忆架构 |
| [bootstrap-and-workspace.md](docs/bootstrap-and-workspace.md) | Workspace Markdown system | Workspace Markdown 体系 |
| [soul-injection.md](docs/soul-injection.md) | Soul and EMA evolution | Soul 注入与 EMA 演化 |
| [skill-system.md](docs/skill-system.md) | Cognitive guard rails | 技能防线系统 |
| [judgment-layer.md](docs/judgment-layer.md) | Decision bundle + LLM | 判断层设计 |
| [emotion-module.md](docs/emotion-module.md) | OCC emotion model | OCC 情绪模块 |
| [ethos-module.md](docs/ethos-module.md) | Value system | Ethos 价值层 |
| [schema-evolution.md](docs/schema-evolution.md) | SQLite auto-migration | Schema 自动演化 |
| [python-runtime.md](docs/python-runtime.md) | Why Python | Python 运行时优势 |

---

## Project Status | 项目状态

This is an experimental project in active development. The cognitive loop is functional; several subsystems are planned or partially implemented.

这是一个积极开发中的实验性项目。认知循环已可运行；部分子系统处于规划或实现中。

| Component | Status | 状态 |
|---|---|---|
| Cognitive loop | ✅ Working | 可运行 |
| OCC emotion model | ✅ Working | 可运行 |
| Ethos + JudgmentSignals | ✅ Working | 可运行 |
| SQLite task store | ✅ Working | 可运行 |
| Tool evolution (hot-swap) | ✅ Working | 可运行 |
| Skill system | 🔧 In progress | 进行中 |
| Workspace Markdown injection | 🔧 Planned | 规划中 |
| Activation decay | 🔧 Planned | 规划中 |
| EMA soul write-back | 🔧 Planned | 规划中 |

---

## Contributing | 贡献

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

提交 PR 前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

Key principles for contributors | 贡献者核心原则：

1. **Minimum patch** — change only what is needed / 最小补丁——只改动必要的部分
2. **No hardcoded values** — all thresholds belong in `lingzhou.json` / 无硬编码——所有阈值属于配置文件
3. **DB is truth** — never treat `SOUL.md` as writable runtime state / DB 是真相——不要把 SOUL.md 当可写运行时状态
4. **Idempotent migrations** — `_migrate()` must be safe to run on any version / 幂等迁移——`_migrate()` 必须在任何版本上安全运行

---

## License | 许可证

MIT — see [LICENSE](LICENSE).
