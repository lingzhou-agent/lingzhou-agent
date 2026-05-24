# Architecture

[中文](ARCHITECTURE.md) | [English](ARCHITECTURE.en.md)

## Cognition Loop

```text
         ┌──────────────┐
         │ Perception   │ ← working memory + episodic memory + prediction error
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │ Self-Drive   │ ← novelty + learning progress + surprise
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │ Judgment     │ ← LLM decisions (act / wait / pause) + tool choice
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │ Execution    │ ← built-in tools + inner continue loop
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │ Reflection   │ ← episodic consolidation + semantic compilation + emotion update
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │ Evolution    │ ← failure patterns → generated fixes → hot reload
         └──────────────┘
```

## Tick Dispatch and Ordering Boundaries

The goal of concurrent ticks is not "run everything out of order". It is "run unrelated ticks concurrently while preserving order for related work".

- Ticks that belong to the same task continuation chain must stay FIFO. Their ordering depends on cross-tick state such as `next_step`, `last_action_*`, `pending_tier`, `ticks_since_judge`, and stall counters.
- Ticks from unrelated tasks, or ticks that do not share continuation state, may run concurrently to reduce chat starvation while an autonomous LLM call is still in flight.
- The runtime should enforce this through a bounded dispatcher: serialize per chain, run chains concurrently under a global concurrency cap, and keep a bounded pending queue.
- The design goal is better responsiveness and throughput without changing the causal order of related work.

## Core Modules

### `core/loop/runtime.py` — Main loop

Coordinates the full perception → judgment → execution → reflection cycle. The runtime is event-driven and wakes on chat messages, task changes, or timeout boundaries.

With concurrent ticks enabled, the runtime is also responsible for:

- owning global shared resources such as the provider, task store, and memory layers
- dispatching ticks into continuation chains
- preserving FIFO inside one chain while allowing bounded cross-chain concurrency

### `core/judgment/runtime.py` — Judgment layer

This is the LLM decision engine. It receives working memory, runtime signals, task context, and tool manifests, then decides what to do next.

Key features:

- multi-model routing for `reader`, `reasoner`, and `repair`
- continuation loops that reuse cached context instead of rebuilding it every tool round
- manifest-driven tool routing and capability governance

### `core/perception/` — Perception layer

Builds perceptual state from memory, emotion, and recent execution feedback.

Submodules:

- `emotion.py`: OCC-style emotion state and replay summary
- `ethos.py`: value state and ethical baseline
- `signals.py`: judgment and cognitive signals
- `layer.py`: perception entry point

### `core/self_drive.py` — Self-drive engine

Evaluates whether the system should stay quiet, continue exploring, or reorganize itself during idle periods.

### `core/evolution.py` — Evolution engine

Detects failure patterns, asks the model to synthesize a patch, validates the result, hot-reloads the change, and rolls back when validation fails.

### `core/behavior_tracker.py` — Behavior tracker

Tracks repetitive actions and exploration loops, then writes those signals back into working memory for the model to notice.

### `core/plugin.py` — Plugin manager

Implements the lifecycle `discover → load → register → start`, and unload flow on shutdown.

## Memory System

### Working memory

Short-horizon memory that is eligible for direct prompt injection.

### Episodic memory

Append-only event history for turns, actions, and outcomes.

### Semantic memory

Long-term reusable knowledge compiled into searchable nodes.

### Task store

Persistent SQLite-backed storage for tasks, chat messages, failures, facts, signals, runs, and meta reflections.

## Tool System

All Python files under `tools/` are auto-discovered. Every tool is declared with `@tool(ToolManifest(...))`, returns `ToolResult`, and is registered into the shared tool registry.

## Channel Architecture

lingzhou currently supports three IO modes:

- `local` — terminal chat
- `wechat` — WeChat iLink channel
- `webhook` — HTTP integration

These channels run alongside the cognition loop and inject events into the runtime.

## Current Gaps

This section folds the old standalone deviation review into the architecture document so the current design and its remaining gaps live in one place.

The list below was re-checked against the current source tree. It keeps only items that are still not fully closed; capabilities already covered by code and tests are no longer listed as gaps.

### P0: Core Loop Gaps Still Open

| ID | Blueprint Requirement | Current State | Deviation |
|----|-----------------------|---------------|-----------|
| P0-1 | Native multimodal perception loop | `tools/image.py`, `core/worker.py`, and tests show that `image.analyze`, `multimodal-worker`, and vision-model routing are already implemented; however, `core/perception/` still has no direct multimodal entry point | If the blueprint expectation is “multimodal input is consumed by perception itself”, the current system still relies on Judgment to call tools explicitly rather than on a native always-on perception path |
| P0-2 | Automatic task-level model-routing closure | `task.model_tier`, `_prefer_tier_for_task()`, and `_apply_tick_model_strategy()` already exist, so tasks can persist a tier and feed it back into later routing | Partially implemented; routing guard / meta-reflection suggestions still surface as hints by default instead of auto-writing `task.model_tier`, and the persistent preference path is currently focused on `reasoner/repair` |

### P1: Structural Maturity

| ID | Blueprint Requirement | Current State | Deviation |
|----|-----------------------|---------------|-----------|
| P1-1 | Full Run abstraction semantics | `Run` storage, `WorkerLayer`, `refresh_running_runs()`, and `build_task_run_result_patch()` already form a working mainline | The mainline exists; the remaining gap is stronger control-plane / execution-plane separation, clearer Run ownership, and richer lifecycle semantics |
| P1-2 | Fully automatic MetaReflection closure | `build_meta_reflection()`, `meta_reflections`, and `_ingest_actionable_meta_reflections()` are implemented and can write into WM, facts, and semantic memory | The dual-loop reflection substrate exists; the remaining gap is that most proposals still require an explicit LLM or tool-mediated approval before becoming policy |

### P2: Quality Improvements

| ID | Blueprint Requirement | Current State | Deviation |
|----|-----------------------|---------------|-----------|
| P2-1 | Before/after evolution evaluation | `core/evolution.py` already has smoke and rollback guards, but there is still no unified before/after scoring loop | Partial |

### Summary

The following capabilities should no longer be described as open gaps: the autonomous inner loop, worker executors, Run-to-Task state feedback, progress crystallization, multi-worker / multi-task concurrency, and the MetaReflection substrate.
The more accurate remaining gaps are native multimodal perception, automatic task-level routing write-back, further maturation of the Run abstraction, and structured evolution evaluation.