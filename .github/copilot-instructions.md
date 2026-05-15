# Copilot Instructions — lingzhou

This file is AI assistant context for working with the lingzhou codebase.
It contains project conventions, architecture facts, and prior-art research notes.

---

## Project Identity

lingzhou is a **self-evolving digital life seed**, not a chat wrapper or tool orchestrator.

- The agent is named **灵舟 (Língzhōu)**.
- It is designed to run autonomously in a loop, evolving its own tools and behavior.
- Python is chosen because `importlib.reload()` + `compile()` enable true runtime self-modification.
- The primary runtime is `core/loop.py`. One tick = perceive → emotion → ethos → judgment → execute → memory.

---

## Repository Layout

```
lingzhou.py           CLI entry (init / loop / interact)
core/
  config.py           All config from lingzhou.json (no hardcoded values)
  loop.py             Main cognitive loop (tick orchestration)
  perception.py       OCC emotion, Ethos, JudgmentSignals, replay summaries
  judgment.py         Bundle assembly → LLM call → JudgmentOutput parse
  execution.py        Tool dispatch
  evolution.py        LLM code-gen + importlib hot-swap
memory/
  working.py          Bounded priority heap (WMEntry)
  episodic.py         task-{id}.md narrative + events.jsonl structured events
  semantic.py         nodes/*.json — ACT-R multi-anchor retrieval
  task_store.py       SQLite ACID state: tasks / failures / facts
provider/
  openai_compat.py    DashScope/Qwen via OpenAI-compatible API
tools/
  registry.py         @tool decorator + discover() + reload_tool()
  file.py             file.read / file.write / file.list
  shell.py            shell.run
  memory_ops.py       memory.add_wm / add_semantic / set_fact / get_fact
  task_ops.py         task.advance / add / complete / list / fail / update_step
prompts/
  judgment.md         Judgment bundle template (Jinja2-style {placeholders})
  system.md           Static identity text
  evolution.md        Code synthesis prompt
docs/                 Project documentation (bilingual)
```

---

## Key Design Rules

1. **DB is truth** — `facts["soul:*"]` is the runtime source for soul/ethos. `SOUL.md` is a human-readable mirror only.
2. **Task is the primary persistence unit** — a `chat` is both the entry point (door) and the persistent task (room); tasks survive restarts.
3. **`hard_axioms` are the only thing that cannot be self-modified** — they are the ethical floor. Configured by the user at `init` time (optional, no hardcoded defaults); once written to DB, evolution cannot touch them. Everything else (ethos_baseline, skills, tools, judgment prompts) CAN and SHOULD evolve through the evolution mechanism.
4. **All config comes from `lingzhou.json`** — no hardcoded thresholds or paths anywhere in `core/`.
5. **`_migrate()` is idempotent** — `ALTER TABLE ADD COLUMN` only; never drop or rename columns.
6. **`@tool` decorator auto-registers** — `discover(tools_dir)` imports all non-underscore `.py` files.
7. **Skills are cognitive guard rails** — they are injected before LLM judgment, not executed directly.
8. **Minimum patch discipline** — make the smallest change that fixes the problem; do not refactor adjacent code.

---

## Known P0 Issues (历史遗留 — 均已解决)

以下问题已全部修复，保留为历史上下文：

| Issue | 解决位置 | 状态 |
|---|---|---|
| template variables not in judgment.md | `prompts/judgment.md` | ✅ 已修复 |
| `reflection` field missing from `JudgmentOutput` | `core/judgment.py:45` | ✅ 已修复 |
| `skills_section` not wired | `core/judgment.py` | ✅ 已修复 |
| `core/skill.py` (`SkillRegistry`) does not exist | `core/skill.py` | ✅ 已创建 |
| `BOOTSTRAP.md` / `IDENTITY.md` not injected | `core/soul.py:bootstrap()` | ✅ 已修复 |
| Activation decay not implemented | `memory/semantic.py:effective_activation()` | ✅ 已修复 |
| `events.jsonl` unbounded growth, O(n) | `memory/episodic.py:_rotate_events_db()` | ✅ 已修复 |
| EMA write-back not implemented | `core/loop.py:345-356` | ✅ 已修复 |

---

## Provider

- **API**: DashScope (Qwen) via OpenAI-compatible endpoint
- **Model**: `qwen-plus` (default), configurable via `lingzhou.json`
- **Env var**: `DASHSCOPE_API_KEY`
- **Base URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1`

---

## Test Command

```bash
cd /Users/suge/Documents/开发/work/python/lingzhou
DASHSCOPE_API_KEY=<key> .venv/bin/python lingzhou.py run
```

---

## Prior Art Research Notes

These notes are internal AI context. They inform design decisions but are not part of lingzhou's public docs.

### Prior Agent A (session-centric TypeScript agent)

- **Memory model**: SQLite with tables `sessions`, `messages`, `state_meta`, FTS5 virtual table (trigram tokenizer) over messages.
- **Soul injection**: `SOUL.md` file read from `HERMES_HOME` once per session → injected into system prompt as-is.
- **Memory fencing**: `<memory-context>...</memory-context>` XML tags in prompt; `StreamingContextScrubber` strips them from user-visible output.
- **Plugin memory manager**: Separate plugin handles reading/writing memory nodes, decoupled from main agent loop.
- **Schema evolution**: `_reconcile_columns()` pattern — `PRAGMA table_info` diff then `ALTER TABLE ADD COLUMN` for missing columns. lingzhou adopted this pattern in `task_store._migrate()`.
- **Session as truth**: Session is the primary persistence unit. All history lives in the `messages` table. FTS5 search is over message content.
- **Key limitation vs lingzhou**: No task-level persistence across sessions; no EMA-based soul evolution; static TypeScript cannot hot-reload tools.

### Prior Agent B (file-centric TypeScript agent)

- **Workspace files**: `MEMORY.md`, `SOUL.md`, `AGENTS.md`, `USER.md`, `TOOLS.md` — all injected at session start via system prompt concatenation.
- **Memory search**: FTS5 (weight 0.7) + vector embedding (weight 0.3) hybrid search. Retrieval threshold configurable.
- **Storage**: WAL-mode SQLite via `node:sqlite` + Kysely ORM. lingzhou adopted WAL mode.
- **Session startup**: `AGENTS.md` H2 section "## Session Startup" used as cold-start bootstrap — closest analogue to lingzhou's `runtime.bootstrap` skill.
- **Key limitation vs lingzhou**: File-based SOUL.md means soul cannot evolve with EMA continuity; no runtime code generation; no task-level state machine.

### Design Differentiators (lingzhou vs prior art)

| Capability | Agent A | Agent B | lingzhou |
|---|---|---|---|
| Soul evolution (EMA) | ❌ Static file | ❌ Static file | ✅ DB + EMA |
| Runtime tool hot-swap | ❌ | ❌ | ✅ `importlib.reload` |
| Task persistence across restarts | ❌ | Partial | ✅ ACID SQLite |
| OCC emotion model | ❌ | ❌ | ✅ Full appraisal chain |
| Ethos + JudgmentSignals | Partial (ethosValues) | ❌ | ✅ Derived each tick |
| Skill guard rails | ❌ | ❌ | ✅ MatchForContext (continuous scoring) |
| Activation decay (Ebbinghaus) | ❌ | ❌ | ✅ `effective_activation(decay_lambda)` |
| Behavior loop detection | ❌ | ❌ | ✅ `BehaviorTracker` (action/read streak + gate) |
| Memory fencing (context scrub) | ✅ StreamingContextScrubber | ❌ | ✅ `_strip_memory_context()` |
| Cross-restart emotion continuity | ❌ | ❌ | ✅ `soul:emotion_state` persisted in DB |
