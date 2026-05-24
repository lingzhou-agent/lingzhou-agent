"""认知循环、chat reply、resolve 等集成测试"""
import asyncio
import builtins
import io
import json
import logging
import math
import os
import tempfile
import time
from functools import lru_cache
from datetime import datetime, UTC, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import aiosqlite
import pytest

from conftest import (
    _proj_root,
    _test_config,
    _tool_ctx,
    _execution_layer,
    _tool_registry,
    _judgment_output,
)

def test_bootstrap_wm_injection():
    from memory.working import WorkingMemory, WMItem

    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        (ws / "BOOTSTRAP.md").write_text("# Bootstrap\n你是灵舟。", encoding="utf-8")
        (ws / "SOUL.md").write_text("# Soul\n真实 0.85", encoding="utf-8")

        wm = WorkingMemory(capacity=20)
        for fname in ("BOOTSTRAP.md", "IDENTITY.md", "SOUL.md"):
            fpath = ws / fname
            if fpath.exists():
                content = fpath.read_text(encoding="utf-8")
                # 每个文件用不同 kind，避免 WorkingMemory 按 kind 去重
                wm.add(WMItem(kind=f"bootstrap_identity:{fname}",
                               content=f"[{fname}]\n{content[:400]}", priority=1.0))

        items = wm.get_top(10)
        assert sum(1 for i in items if i["kind"].startswith("bootstrap_identity")) == 2


# ══════════════════════════════════════════════════════════════════════════════
# 完整构造链路（不调 LLM）
# ══════════════════════════════════════════════════════════════════════════════

def test_cognition_loop_init():
    """CognitionLoop.__init__ 不崩溃，关键参数正确传递。"""
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg_path = Path.home() / ".lingzhou" / "lingzhou.json"
    if not cfg_path.exists():
        cfg_path = _proj_root() / "lingzhou.json.example"
    cfg = Config.load(cfg_path)
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        assert loop.semantic.decay_lambda == cfg.memory.semantic_decay_lambda
        assert loop.episodic.max_events == cfg.memory.max_events


def test_curiosity_signal_does_not_auto_create_task():
    asyncio.run(_curiosity_signal_does_not_auto_create_task())


async def _curiosity_signal_does_not_auto_create_task():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg_path = Path.home() / ".lingzhou" / "lingzhou.json"
    if not cfg_path.exists():
        cfg_path = _proj_root() / "lingzhou.json.example"
    cfg = Config.load(cfg_path)
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            loop._idle_cycles = cfg.thresholds.curiosity_idle_min_cycles
            loop._last_curiosity_signal_idle_cycle = 0
            ethos_state = cast(Any, SimpleNamespace(
                values=SimpleNamespace(curiosity=cfg.thresholds.curiosity_idle_task + 0.1)
            ))

            await loop._maybe_curiosity_task(ethos_state)

            tasks = await loop.task_store.list_tasks(limit=20)
            assert tasks == []
            # curiosity 仍然只注入感知信号，不直接创建任务；
            # 验证 _last_curiosity_signal_idle_cycle 已被标记（防止重复触发）
            assert loop._last_curiosity_signal_idle_cycle == loop._idle_cycles
        finally:
            await loop.task_store.close()
            await loop.provider.close()


def test_self_drive_signal_does_not_auto_create_task():
    asyncio.run(_self_drive_signal_does_not_auto_create_task())


def test_crash_recovery_uses_runtime_emotion_fallback():
    from core.loop.startup import _inject_crash_recovery
    from memory.working import WorkingMemory

    with tempfile.TemporaryDirectory() as d:
        state_dir = Path(d)
        (state_dir / "survival.json").write_text(
            json.dumps(
                {
                    "tick": 7,
                    "ts": "2026-05-22T10:00:00Z",
                    "active_task_title": "恢复任务",
                    "last_action": "task.resume",
                    "emotion": {},
                    "exit_type": "crash",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        wm = WorkingMemory(capacity=10)
        loop = SimpleNamespace(
            _cfg=SimpleNamespace(state_dir=state_dir),
            _wm=wm,
            _emotion=SimpleNamespace(valence=0.44, arousal=0.66),
        )

        _inject_crash_recovery(loop)

        items = wm.get_top(5)
        assert len(items) == 1
        assert items[0]["kind"] == "crash_recovery"
        assert "valence=0.44" in items[0]["content"]
        assert "arousal=0.66" in items[0]["content"]


async def _self_drive_signal_does_not_auto_create_task():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg_path = Path.home() / ".lingzhou" / "lingzhou.json"
    if not cfg_path.exists():
        cfg_path = _proj_root() / "lingzhou.json.example"
    cfg = Config.load(cfg_path)
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            loop._behavior._wait_streak = cfg.thresholds.curiosity_idle_min_cycles

            await loop._maybe_inject_self_drive()

            tasks = await loop.task_store.list_tasks(limit=20)
            assert tasks == []

            wm_items = loop._wm.get_top(10)
            self_drive_items = [item for item in wm_items if item["kind"] == "self_drive"]
            assert len(self_drive_items) == 1
            assert "task.add" in self_drive_items[0]["content"]
            assert "source=self_drive" in self_drive_items[0]["content"]
        finally:
            await loop.task_store.close()
            await loop.provider.close()


def test_self_drive_signal_bypasses_idle_judge_aggregation():
    asyncio.run(_self_drive_signal_bypasses_idle_judge_aggregation())


def test_continue_phase_uses_configured_task_plan_max_per_tick():
    asyncio.run(_continue_phase_uses_configured_task_plan_max_per_tick())


def test_continue_phase_rejudges_after_task_plan_limit_block():
    asyncio.run(_continue_phase_rejudges_after_task_plan_limit_block())


async def _continue_phase_uses_configured_task_plan_max_per_tick():
    from core.config import Config
    from core.loop.continue_phase import _run_continue_phase
    from tools.registry import ToolResult
    from memory.working import WorkingMemory

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
        "loop": {
            "max_tool_rounds": 2,
        },
        "thresholds": {
            "continue_task_plan_max_per_tick": 2,
        },
    })

    dispatched: list[str] = []

    class _Store:
        async def has_pending_chat_message(self) -> bool:
            return False

    class _Judgment:
        async def decide_continue(self, *args, **kwargs):
            return _judgment_output(
                decision="act",
                chosen_action_id="task.plan",
                params={"plan": [{"step": "执行真实工具", "status": "in_progress"}]},
            )

    class _Execution:
        async def dispatch(self, action, ctx):
            dispatched.append(action.chosen_action_id or action.decision)
            return ToolResult(summary="计划已更新")

    class _Behavior:
        def on_act(self, *args, **kwargs):
            return []

        def apply_cognitive_probe(self, signals):
            return None

        def on_edit_failure(self, error):
            return []

        def on_act_result(self, tool_id, result_summary):
            return None

    class _Episodic:
        def record(self, **kwargs):
            return None

    loop = SimpleNamespace(
        _cfg=cfg,
        _emotion=SimpleNamespace(valence=0.5, arousal=0.5),
        _task_store=_Store(),
        _judgment=_Judgment(),
        _pending_routing_overrides=None,
        _registry=_tool_registry(),
        _wm=WorkingMemory(capacity=20),
        _behavior=_Behavior(),
        _execution=_Execution(),
        _episodic=_Episodic(),
        _bootstrap_mode="none",
    )

    tool_history = [{
        "tool": "task.plan",
        "params": {"plan": [{"step": "理解范围", "status": "completed"}]},
        "result": "计划已创建",
        "status": "ok",
        "error": "",
        "summary": "计划已创建",
        "state_delta": {},
    }]

    action, result = await _run_continue_phase(
        loop=loop,
        ctx=SimpleNamespace(),
        user_message="",
        active_task=SimpleNamespace(id=1),
        cognitive_signals=SimpleNamespace(),
        action=_judgment_output(decision="act", chosen_action_id="task.plan", params={"plan": []}),
        result=ToolResult(summary="初始计划"),
        tool_history=tool_history,
    )

    assert dispatched == ["task.plan"]
    assert action.chosen_action_id == "task.plan"
    assert result.summary == "计划已更新"
    assert sum(1 for item in tool_history if item.get("tool") == "task.plan") == 2


async def _continue_phase_rejudges_after_task_plan_limit_block():
    from core.config import Config
    from core.loop.continue_phase import _run_continue_phase
    from tools.registry import ToolResult
    from memory.working import WorkingMemory

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
        "loop": {
            "max_tool_rounds": 3,
        },
        "thresholds": {
            "continue_task_plan_max_per_tick": 1,
        },
    })

    dispatched: list[str] = []
    judgment_calls = 0

    class _Store:
        async def has_pending_chat_message(self) -> bool:
            return False

    class _Judgment:
        async def decide_continue(self, *args, **kwargs):
            nonlocal judgment_calls
            judgment_calls += 1
            if judgment_calls == 1:
                return _judgment_output(
                    decision="act",
                    chosen_action_id="task.plan",
                    params={"plan": [{"step": "继续拆解", "status": "in_progress"}]},
                )
            return _judgment_output(
                decision="act",
                chosen_action_id="shell.run",
                params={"command": "pytest -q"},
            )

    class _Execution:
        async def dispatch(self, action, ctx):
            dispatched.append(action.chosen_action_id or action.decision)
            return ToolResult(summary="执行完成")

    class _Behavior:
        def on_act(self, *args, **kwargs):
            return []

        def apply_cognitive_probe(self, signals):
            return None

        def on_edit_failure(self, error):
            return []

        def on_act_result(self, tool_id, result_summary):
            return None

    class _Episodic:
        def record(self, **kwargs):
            return None

    wm = WorkingMemory(capacity=20)
    loop = SimpleNamespace(
        _cfg=cfg,
        _emotion=SimpleNamespace(valence=0.5, arousal=0.5),
        _task_store=_Store(),
        _judgment=_Judgment(),
        _pending_routing_overrides=None,
        _registry=_tool_registry(),
        _wm=wm,
        _behavior=_Behavior(),
        _execution=_Execution(),
        _episodic=_Episodic(),
        _bootstrap_mode="none",
    )

    tool_history = [{
        "tool": "task.plan",
        "params": {"plan": [{"step": "理解范围", "status": "completed"}]},
        "result": "计划已创建",
        "status": "ok",
        "error": "",
        "summary": "计划已创建",
        "state_delta": {},
    }]

    action, result = await _run_continue_phase(
        loop=loop,
        ctx=SimpleNamespace(),
        user_message="",
        active_task=SimpleNamespace(id=1),
        cognitive_signals=SimpleNamespace(),
        action=_judgment_output(decision="act", chosen_action_id="task.plan", params={"plan": []}),
        result=ToolResult(summary="初始计划"),
        tool_history=tool_history,
    )

    assert judgment_calls == 2
    assert dispatched == ["shell.run"]
    assert action.chosen_action_id == "shell.run"
    assert result.summary == "执行完成"
    assert any(item.get("tool") == "task.plan" and item.get("error") == "ContinueTaskPlanLimit" for item in tool_history)
    assert any(item.get("tool") == "shell.run" for item in tool_history)
    assert any(item["kind"] == "self_awareness" and "[计划阻断]" in item["content"] for item in wm.get_top())


def test_continue_phase_uses_configured_tool_history_compaction_threshold():
    asyncio.run(_continue_phase_uses_configured_tool_history_compaction_threshold())


async def _continue_phase_uses_configured_tool_history_compaction_threshold():
    from core.config import Config
    from core.loop.continue_phase import _run_continue_phase
    from tools.registry import ToolResult
    from memory.working import WorkingMemory

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
        "loop": {
            "max_tool_rounds": 2,
        },
        "thresholds": {
            "continue_tool_history_compact_threshold": 2,
            "continue_tool_history_keep_last": 1,
        },
    })

    seen_histories: list[list[dict[str, Any]]] = []

    class _Store:
        async def has_pending_chat_message(self) -> bool:
            return False

    class _Judgment:
        async def decide_continue(self, tool_history, **kwargs):
            seen_histories.append([dict(item) for item in tool_history])
            return _judgment_output(decision="wait", rationale="证据已足够")

    class _Execution:
        async def dispatch(self, action, ctx):
            return ToolResult(summary="等待下一轮")

    loop = SimpleNamespace(
        _cfg=cfg,
        _emotion=SimpleNamespace(valence=0.5, arousal=0.5),
        _task_store=_Store(),
        _judgment=_Judgment(),
        _pending_routing_overrides=None,
        _registry=_tool_registry(),
        _wm=WorkingMemory(capacity=20),
        _execution=_Execution(),
        _episodic=SimpleNamespace(record=lambda **kwargs: None),
        _bootstrap_mode="none",
    )

    tool_history = [
        {
            "tool": "memory.search",
            "params": {"query": "legacy runtime"},
            "result": "命中 1 条",
            "status": "ok",
            "error": "",
            "summary": "命中 1 条",
            "state_delta": {},
        },
        {
            "tool": "task.list",
            "params": {"status": "all"},
            "result": "命中 1 条任务",
            "status": "ok",
            "error": "",
            "summary": "命中 1 条任务",
            "state_delta": {},
        },
    ]

    await _run_continue_phase(
        loop=loop,
        ctx=SimpleNamespace(),
        user_message="",
        active_task=SimpleNamespace(id=1),
        cognitive_signals=SimpleNamespace(),
        action=_judgment_output(decision="act", chosen_action_id="memory.search", params={"query": "legacy runtime"}),
        result=ToolResult(summary="初始结果"),
        tool_history=tool_history,
    )

    assert len(seen_histories) == 1
    assert seen_histories[0][0]["tool"] == "[compacted]"
    assert "早期 1 条工具调用已压缩" in seen_histories[0][0]["result"]
    assert seen_histories[0][1]["tool"] == "task.list"


async def _self_drive_signal_bypasses_idle_judge_aggregation():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.judgment import JudgmentOutput
    from core.loop import CognitionLoop
    from core.loop.tick import _TickJudgmentPrep, _decide_initial_action
    from memory.working import WMItem

    cfg_path = Path.home() / ".lingzhou" / "lingzhou.json"
    if not cfg_path.exists():
        cfg_path = _proj_root() / "lingzhou.json.example"
    cfg = Config.load(cfg_path)
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.loop.judge_every = 3
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        called = {"value": False}

        async def _fake_decide(*args, **kwargs):
            called["value"] = True
            return JudgmentOutput.wait(reason="llm saw self_drive")

        loop._judgment.decide = _fake_decide  # type: ignore[method-assign]
        loop._wm.add(WMItem(kind="self_drive", content="[自驱信号] test", priority=0.9))

        try:
            action = await _decide_initial_action(
                loop,
                cycle=1,
                user_message="",
                active_task=None,
                prep=_TickJudgmentPrep(
                    percept=None,
                    perception_replay=None,
                    cognitive_signals=None,
                    ethos_state=None,
                    signals=None,
                    hard_boundaries=[],
                ),
            )

            assert called["value"] is True
            assert action.rationale == "llm saw self_drive"
            assert loop._ticks_since_judge == 0
        finally:
            await loop.provider.close()


def test_short_continuation_message_is_also_forwarded_to_active_task_inbox():
    from core.loop.tick import _should_steer_active_task_from_user_message
    from memory.task_store import Task

    task = Task(
        id=1,
        title="继续分析 chat 回复",
        status="in_progress",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="继续分析 chat 回复为什么丢失",
        next_step="继续分析这个问题",
    )

    assert _should_steer_active_task_from_user_message(task, "继续分析") is True


def test_task_steer_any_nonempty_user_message_is_forwarded_to_active_task_inbox():
    from core.loop.tick import _should_steer_active_task_from_user_message
    from memory.task_store import Task

    task = Task(
        id=1,
        title="alpha beta",
        status="in_progress",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="alpha beta",
        next_step="alpha beta",
    )

    message = "alpha beta gamma delta"
    assert _should_steer_active_task_from_user_message(task, message) is True
    assert _should_steer_active_task_from_user_message(task, "继续分析") is True
    assert _should_steer_active_task_from_user_message(task, "   ") is False


def test_distinct_user_message_is_queued_into_active_task_inbox():
    asyncio.run(_distinct_user_message_is_queued_into_active_task_inbox())


async def _distinct_user_message_is_queued_into_active_task_inbox():
    from core.loop.tick import _maybe_steer_active_task_from_user_message
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "steer-user-message.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "旧回填任务",
                goal="等待模型加载完成后，再次检查日志确认数据回填进度",
                next_step="继续检查回填进度",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None

            updated = await _maybe_steer_active_task_from_user_message(
                store,
                task,
                "请你使用 puppeteer 去搜索。",
            )

            assert updated is not None
            assert len(updated.extras["inbox_messages"]) == 1
            assert updated.extras["inbox_messages"][0] == "收到新的用户消息：请你使用 puppeteer 去搜索。"

            persisted = await store.get_task_by_id(task_id)
            assert persisted is not None
            assert persisted.extras["inbox_messages"] == updated.extras["inbox_messages"]
        finally:
            await store.close()


def test_invalid_routing_overrides_clear_previous_pending_state():
    asyncio.run(_invalid_routing_overrides_clear_previous_pending_state())


async def _invalid_routing_overrides_clear_previous_pending_state():
    from core.loop.tick import _apply_tick_model_strategy
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "routing-overrides.db")
        await store.open()
        try:
            await store.set_fact("pref:routing_overrides", '{"reader":"demo/model"}', scope="system")
            loop = cast(Any, SimpleNamespace(
                _cfg=SimpleNamespace(
                    loop=SimpleNamespace(
                        idle_with_task_bounds=[100, 30000],
                        idle_no_task_bounds=[5000, 300000],
                    )
                ),
                _task_store=store,
                _pending_routing_overrides={"reader": "demo/model"},
                _pending_tier="reader",
                _pending_idle_gap=1.0,
                _pending_thinking_override="low",
            ))
            action = _judgment_output(
                decision="wait",
                model_strategy={
                    "routing_overrides": {
                        "invalid": "demo/ignored",
                        "reader": "",
                    }
                },
            )

            await _apply_tick_model_strategy(loop, action, None)

            stored_value, found = await store.get_fact("pref:routing_overrides")
            assert loop._pending_routing_overrides is None
            assert found is True
            assert stored_value == ""
        finally:
            await store.close()


def test_run_tick_maintenance_uses_configured_global_md_warn_lines():
    asyncio.run(_run_tick_maintenance_uses_configured_global_md_warn_lines())


def test_run_tick_maintenance_uses_configured_low_pressure_skip_threshold():
    asyncio.run(_run_tick_maintenance_uses_configured_low_pressure_skip_threshold())


async def _run_tick_maintenance_uses_configured_global_md_warn_lines():
    from core.config import Config
    from core.loop.tick import _run_tick_maintenance

    class _WM:
        def __init__(self) -> None:
            self.pressure = 1.0
            self.items: list[Any] = []

        def add(self, item: Any) -> None:
            self.items.append(item)

    class _Soul:
        def __init__(self) -> None:
            self.synced = False

        async def sync_md(self) -> None:
            self.synced = True

    class _DB:
        def __init__(self) -> None:
            self.commands: list[str] = []

        async def execute(self, sql: str) -> None:
            self.commands.append(sql)

    with tempfile.TemporaryDirectory() as d:
        memory_dir = Path(d) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "episodic").mkdir(parents=True, exist_ok=True)
        (memory_dir / "episodic" / "global.md").write_text("a\nb\nc\n", encoding="utf-8")

        cfg = Config.model_validate({
            "providers": {
                "bailian": {
                    "type": "openai_compat",
                    "base_url": "https://example.invalid/v1",
                    "api_key_env": "DASHSCOPE_API_KEY",
                }
            },
            "model": "bailian/qwen3.6-plus",
            "temperature": 0.7,
            "timeout": 60.0,
            "loop": {
                "memory_dir": str(memory_dir),
                "consolidate_every": 1,
            },
            "memory": {
                "global_md_warn_bytes": 999999,
                "global_md_warn_lines": 2,
            },
            "thresholds": {
                "wm_pressure_task": 0.8,
            },
        })

        db = _DB()
        soul = _Soul()
        wm = _WM()
        loop = cast(Any, SimpleNamespace(
            _cfg=cfg,
            _wm=wm,
            _soul=soul,
            _task_store=SimpleNamespace(_db=db),
            consolidated=False,
        ))

        async def _consolidate(active_task: Any) -> None:
            loop.consolidated = True

        loop._consolidate = _consolidate

        await _run_tick_maintenance(loop, active_task=None, cycle=1)

        assert loop.consolidated is True
        assert soul.synced is True
        assert db.commands == ["PRAGMA wal_checkpoint(TRUNCATE)"]
        assert any("global.md 当前 3 行" in item.content for item in wm.items)


async def _run_tick_maintenance_uses_configured_low_pressure_skip_threshold():
    from core.config import Config
    from core.loop.tick import _run_tick_maintenance

    class _WM:
        def __init__(self, pressure: float) -> None:
            self.pressure = pressure

    class _Soul:
        def __init__(self) -> None:
            self.synced = False

        async def sync_md(self) -> None:
            self.synced = True

    class _DB:
        async def execute(self, sql: str) -> None:
            return None

    async def _run_case(skip_threshold: float) -> bool:
        with tempfile.TemporaryDirectory() as d:
            memory_dir = Path(d) / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            cfg = Config.model_validate({
                "providers": {
                    "bailian": {
                        "type": "openai_compat",
                        "base_url": "https://example.invalid/v1",
                        "api_key_env": "DASHSCOPE_API_KEY",
                    }
                },
                "model": "bailian/qwen3.6-plus",
                "temperature": 0.7,
                "timeout": 60.0,
                "loop": {
                    "memory_dir": str(memory_dir),
                    "consolidate_every": 1,
                },
                "memory": {
                    "consolidate_low_pressure_skip_threshold": skip_threshold,
                    "global_md_warn_bytes": 999999,
                    "global_md_warn_lines": 999999,
                },
                "thresholds": {
                    "wm_pressure_task": 0.95,
                },
            })

            loop = cast(Any, SimpleNamespace(
                _cfg=cfg,
                _wm=_WM(pressure=0.86),
                _soul=_Soul(),
                _task_store=SimpleNamespace(_db=_DB()),
                consolidated=False,
            ))

            async def _consolidate(active_task: Any) -> None:
                loop.consolidated = True

            loop._consolidate = _consolidate
            await _run_tick_maintenance(loop, active_task=None, cycle=1)
            return bool(loop.consolidated)

    assert await _run_case(0.90) is False
    assert await _run_case(0.80) is True


def test_post_tick_memory_crystallizes_task_summary_title_with_task_id():
    asyncio.run(_post_tick_memory_crystallizes_task_summary_title_with_task_id())


async def _post_tick_memory_crystallizes_task_summary_title_with_task_id():
    from core.loop.tick import _post_tick_memory_impl
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            task_id = await store.add_task("重复标题任务", goal="crystallize summary", status="in_progress")
            active_task = await store.get_task_by_id(task_id)
            assert active_task is not None
            await store.update_status(task_id, "done", "finished")
            semantic = SemanticMemory(root / "semantic")
            loop = cast(Any, SimpleNamespace(
                _task_store=store,
                _episodic=SimpleNamespace(load_for_context=lambda task_id_str, max_chars=40000: "任务完成叙事"),
                _semantic=semantic,
                _emotion=SimpleNamespace(valence=0.66, arousal=0.44),
                _wm=SimpleNamespace(add=lambda item: None),
                _cfg=SimpleNamespace(
                    thresholds=SimpleNamespace(wm_pri_insight=0.8),
                    memory=SimpleNamespace(chat_crystallize_every=3),
                    emotion=SimpleNamespace(),
                ),
            ))
            action = cast(Any, SimpleNamespace(chosen_action_id="", params={}, reflection="", rationale=""))
            result = SimpleNamespace(summary="", skipped=False, error=None, kind="execute_result", priority=0.5)

            await _post_tick_memory_impl(loop, action, result, active_task, cycle=1, user_message="")

            node = semantic.get(f"task_summary_{task_id}")
            assert node is not None
            assert node.title == f"[done] task#{task_id} 重复标题任务"
        finally:
            await store.close()


def test_post_tick_memory_formats_learned_insight_title_with_hash_suffix():
    asyncio.run(_post_tick_memory_formats_learned_insight_title_with_hash_suffix())


async def _post_tick_memory_formats_learned_insight_title_with_hash_suffix():
    import hashlib

    from core.loop.tick import _post_tick_memory_impl
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore

    prefix = "当检测到行为死循环时，仅改变工具调用参数是不够的，必须在 next_step 中注入具体的、可执行的子目标内容，以强制认"
    reflection_a = prefix + "知路径A"
    reflection_b = prefix + "知路径B"

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            task_id = await store.add_task("反思任务", goal="write insight", status="in_progress")
            active_task = await store.get_task_by_id(task_id)
            assert active_task is not None
            semantic = SemanticMemory(root / "semantic")
            loop = cast(Any, SimpleNamespace(
                _task_store=store,
                _episodic=SimpleNamespace(load_for_context=lambda task_id_str, max_chars=40000: "", record=lambda **kwargs: None),
                _semantic=semantic,
                _emotion=SimpleNamespace(valence=0.66, arousal=0.44),
                _wm=SimpleNamespace(add=lambda item: None),
                _cfg=SimpleNamespace(
                    thresholds=SimpleNamespace(wm_pri_insight=0.8),
                    memory=SimpleNamespace(chat_crystallize_every=99),
                    emotion=SimpleNamespace(),
                ),
            ))
            result = SimpleNamespace(summary="", skipped=False, error=None, kind="execute_result", priority=0.5)

            await _post_tick_memory_impl(
                loop,
                cast(Any, SimpleNamespace(chosen_action_id="", params={}, reflection=reflection_a, rationale="")),
                result,
                active_task,
                cycle=1,
                user_message="",
            )
            await _post_tick_memory_impl(
                loop,
                cast(Any, SimpleNamespace(chosen_action_id="", params={}, reflection=reflection_b, rationale="")),
                result,
                active_task,
                cycle=2,
                user_message="",
            )

            node_a = semantic.get(f"insight_{hashlib.md5(reflection_a.encode()).hexdigest()[:10]}")
            node_b = semantic.get(f"insight_{hashlib.md5(reflection_b.encode()).hexdigest()[:10]}")
            assert node_a is not None
            assert node_b is not None
            assert node_a.title != node_b.title
            assert node_a.title.endswith(f" [{hashlib.md5(reflection_a.encode()).hexdigest()[:6]}]")
            assert node_b.title.endswith(f" [{hashlib.md5(reflection_b.encode()).hexdigest()[:6]}]")
        finally:
            await store.close()


def test_post_tick_memory_crystallizes_event_title_with_task_id():
    asyncio.run(_post_tick_memory_crystallizes_event_title_with_task_id())


async def _post_tick_memory_crystallizes_event_title_with_task_id():
    from core.loop.tick import _post_tick_memory_impl
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            task_id = await store.add_task("事件任务", goal="write event", status="in_progress")
            active_task = await store.get_task_by_id(task_id)
            assert active_task is not None
            semantic = SemanticMemory(root / "semantic")
            loop = cast(Any, SimpleNamespace(
                _task_store=store,
                _episodic=SimpleNamespace(load_for_context=lambda task_id_str, max_chars=40000: "", record=lambda **kwargs: None),
                _semantic=semantic,
                _emotion=SimpleNamespace(valence=0.66, arousal=0.44),
                _wm=SimpleNamespace(add=lambda item: None),
                _cfg=SimpleNamespace(
                    thresholds=SimpleNamespace(wm_pri_insight=0.8),
                    memory=SimpleNamespace(chat_crystallize_every=1),
                    emotion=SimpleNamespace(),
                ),
            ))
            action = cast(Any, SimpleNamespace(chosen_action_id="", params={}, reflection="事件反思", rationale=""))
            result = SimpleNamespace(summary="", skipped=False, error=None, kind="execute_result", priority=0.5)

            await _post_tick_memory_impl(loop, action, result, active_task, cycle=1, user_message="")

            ts_label = datetime.now(UTC).strftime("%Y-%m-%d")
            node = semantic.get(f"event-task{task_id}-{ts_label}")
            assert node is not None
            assert node.title == f"[{ts_label}] task#{task_id} 事件任务"
        finally:
            await store.close()


def test_dev_model_switch_syncs_routing_entries_following_primary_model():
    from cli.dev import _sync_routing_models_on_primary_switch

    cfg_data = {
        "model": "copilot/gpt-5.4",
        "routing": {
            "reader": "bailian/qwen3.6-plus",
            "reasoner": "copilot/gpt-5.4",
            "repair": "copilot/gpt-5.4",
        },
    }

    changed = _sync_routing_models_on_primary_switch(
        cfg_data,
        old_model="copilot/gpt-5.4",
        new_model="copilot/gpt-5.4-mini",
    )

    assert changed == ["reasoner", "repair"]
    assert cfg_data["routing"]["reader"] == "bailian/qwen3.6-plus"
    assert cfg_data["routing"]["reasoner"] == "copilot/gpt-5.4-mini"
    assert cfg_data["routing"]["repair"] == "copilot/gpt-5.4-mini"


def test_dev_model_switch_repairs_stale_same_provider_reasoner_routes_when_reselecting_same_model():
    from cli.dev import _sync_routing_models_on_primary_switch

    cfg_data = {
        "model": "copilot/gpt-5.4-mini",
        "routing": {
            "reader": "bailian/qwen3.6-plus",
            "reasoner": "copilot/gpt-5.4",
            "complex": "copilot/o3",
            "repair": "bailian/qwen3.6-plus",
        },
    }

    changed = _sync_routing_models_on_primary_switch(
        cfg_data,
        old_model="copilot/gpt-5.4-mini",
        new_model="copilot/gpt-5.4-mini",
    )

    assert changed == ["reasoner", "complex"]
    assert cfg_data["routing"]["reader"] == "bailian/qwen3.6-plus"
    assert cfg_data["routing"]["reasoner"] == "copilot/gpt-5.4-mini"
    assert cfg_data["routing"]["complex"] == "copilot/gpt-5.4-mini"
    assert cfg_data["routing"]["repair"] == "bailian/qwen3.6-plus"


def test_dev_model_prefers_current_or_reasoning_model():
    from cli.dev import _preferred_model_index

    models = [
        {"id": "gpt-4.5"},
        {"id": "gpt-5.4-mini", "thinking": True},
        {"id": "o3", "reasoning": True},
    ]

    assert _preferred_model_index(models, current_model_id="o3") == 2
    assert _preferred_model_index(models, current_model_id="") == 1


def test_dev_model_target_selection_updates_reasoner_without_touching_primary_model():
    from cli.dev import _apply_model_target_selection

    cfg_data = {
        "model": "bailian/qwen3.6-plus",
        "routing": {
            "reader": "bailian/qwen-plus",
            "reasoner": "copilot/gpt-5.4",
        },
    }

    result = _apply_model_target_selection(
        cfg_data,
        current_model="bailian/qwen3.6-plus",
        new_model="copilot/o3",
        target="reasoner",
    )

    assert result["target"] == "reasoner"
    assert result["previous"] == "copilot/gpt-5.4"
    assert result["routing_changed"] == ["reasoner"]
    assert result["runtime_override_tier"] == "reasoner"
    assert cfg_data["model"] == "bailian/qwen3.6-plus"
    assert cfg_data["routing"]["reasoner"] == "copilot/o3"


def test_dev_model_target_selection_normalizes_complex_alias_to_reasoner():
    from cli.dev import _apply_model_target_selection

    cfg_data = {
        "model": "bailian/qwen3.6-plus",
        "routing": {},
    }

    result = _apply_model_target_selection(
        cfg_data,
        current_model="bailian/qwen3.6-plus",
        new_model="copilot/gpt-5.4-mini",
        target="complex",
    )

    assert result["target"] == "reasoner"
    assert cfg_data["routing"]["reasoner"] == "copilot/gpt-5.4-mini"


def test_merge_runtime_routing_override_keeps_supported_tiers_only():
    from cli.dev import _merge_runtime_routing_override

    merged = _merge_runtime_routing_override(
        {"reader": "bailian/qwen-plus", "custom": "ignored/model"},
        tier="reasoner",
        model_ref="copilot/gpt-5.4",
    )

    assert merged == {
        "reader": "bailian/qwen-plus",
        "reasoner": "copilot/gpt-5.4",
    }


def test_chat_reply_is_persisted_before_post_tick_cleanup():
    asyncio.run(_chat_reply_is_persisted_before_post_tick_cleanup())


async def _chat_reply_is_persisted_before_post_tick_cleanup():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg_path = Path.home() / ".lingzhou" / "lingzhou.json"
    if not cfg_path.exists():
        cfg_path = _proj_root() / "lingzhou.json.example"
    cfg = Config.load(cfg_path)
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            async def _sense(*args, **kwargs):
                return cast(Any, SimpleNamespace(prediction_error=0.0, workspace_dirty=False))

            loop._perception.sense = _sense
            loop._perception.derive_cognitive_signals = lambda *args, **kwargs: cast(
                Any,
                SimpleNamespace(
                    repeat_action_count=0,
                    repeat_action_tool="",
                    repeat_action_key="",
                    repeat_read_count=0,
                    repeat_read_path="",
                    loop_probe_version=0,
                ),
            )

            async def _decide(*args, **kwargs):
                return _judgment_output(
                    decision="pause",
                    rationale="已经找到根因",
                    reply_to_user="最终答复",
                )

            loop._judgment.decide = _decide
            loop._judgment._last_call_meta = {
                "model_ref": cfg.model,
                "thinking": cfg.thinking,
                "tier": "reasoner",
                "phase": "initial",
            }

            async def _boom(*args, **kwargs):
                raise RuntimeError("post tick cleanup failed")

            loop._post_tick_memory = _boom

            with pytest.raises(RuntimeError, match="post tick cleanup failed"):
                await loop._tick(1, user_message="你好", chat_id="chat-1")

            msgs = await loop.task_store.get_chat_messages_since(0, "chat-1")
            assert len(msgs) == 1
            assert msgs[0]["role"] == "assistant"
            assert msgs[0]["content"] == "最终答复"
        finally:
            await loop.task_store.close()
            await loop.provider.close()


def test_local_chat_reply_is_persisted_for_default_channel():
    asyncio.run(_local_chat_reply_is_persisted_for_default_channel())


async def _local_chat_reply_is_persisted_for_default_channel():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg_path = Path.home() / ".lingzhou" / "lingzhou.json"
    if not cfg_path.exists():
        cfg_path = _proj_root() / "lingzhou.json.example"
    cfg = Config.load(cfg_path)
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            async def _sense(*args, **kwargs):
                return cast(Any, SimpleNamespace(prediction_error=0.0, workspace_dirty=False))

            loop._perception.sense = _sense
            loop._perception.derive_cognitive_signals = lambda *args, **kwargs: cast(
                Any,
                SimpleNamespace(
                    repeat_action_count=0,
                    repeat_action_tool="",
                    repeat_action_key="",
                    repeat_read_count=0,
                    repeat_read_path="",
                    loop_probe_version=0,
                ),
            )

            async def _decide(*args, **kwargs):
                return _judgment_output(
                    decision="pause",
                    rationale="已经得到结论",
                    reply_to_user="这是本地 chat 的回复",
                )

            loop._judgment.decide = _decide
            loop._judgment._last_call_meta = {
                "model_ref": cfg.model,
                "thinking": cfg.thinking,
                "tier": "reasoner",
                "phase": "initial",
            }

            reply = await loop._tick(1, user_message="你好", chat_id="")

            assert reply == "这是本地 chat 的回复"
            msgs = await loop.task_store.get_chat_messages_since(0)
            assert len(msgs) == 1
            assert msgs[0]["role"] == "assistant"
            assert msgs[0]["content"] == "这是本地 chat 的回复"
        finally:
            await loop.task_store.close()
            await loop.provider.close()


def test_resolve_reply_chat_id_falls_back_to_last_chat_fact():
    asyncio.run(_resolve_reply_chat_id_falls_back_to_last_chat_fact())


async def _resolve_reply_chat_id_falls_back_to_last_chat_fact():
    from core.loop.chat import _resolve_reply_chat_id
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "chat-fallback.db")
        await store.open()
        try:
            await store.set_fact("chat:last_chat_id", "wechat:user-9", scope="system")
            loop = SimpleNamespace(_task_store=store)
            chat_id = await _resolve_reply_chat_id(loop, None, None)
            assert chat_id == "wechat:user-9"
            assert await _resolve_reply_chat_id(loop, None, "") == ""
        finally:
            await store.close()


def test_autonomous_followup_reply_uses_bound_chat_session():
    asyncio.run(_autonomous_followup_reply_uses_bound_chat_session())


async def _autonomous_followup_reply_uses_bound_chat_session():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg_path = Path.home() / ".lingzhou" / "lingzhou.json"
    if not cfg_path.exists():
        cfg_path = _proj_root() / "lingzhou.json.example"
    cfg = Config.load(cfg_path)
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            task_id = await loop.task_store.add_task(
                "继续向用户确认",
                goal="等待用户回复后继续",
                source="external",
                next_step="追问用户缺失信息",
            )
            await loop.task_store.update_status(task_id, "in_progress", "追问用户缺失信息")
            await loop.task_store.set_fact(f"task:{task_id}:chat_id", "wechat:user-1", scope="task")

            async def _sense(*args, **kwargs):
                return cast(Any, SimpleNamespace(prediction_error=0.0, workspace_dirty=False))

            loop._perception.sense = _sense
            loop._perception.derive_cognitive_signals = lambda *args, **kwargs: cast(
                Any,
                SimpleNamespace(
                    repeat_action_count=0,
                    repeat_action_tool="",
                    repeat_action_key="",
                    repeat_read_count=0,
                    repeat_read_path="",
                    loop_probe_version=0,
                ),
            )

            async def _decide(*args, **kwargs):
                return _judgment_output(
                    decision="pause",
                    rationale="需要用户补充一个关键参数",
                    reply_to_user="我还缺一个参数，麻烦补充一下。",
                )

            loop._judgment.decide = _decide
            loop._judgment._last_call_meta = {
                "model_ref": cfg.model,
                "thinking": cfg.thinking,
                "tier": "reasoner",
                "phase": "initial",
            }

            reply = await loop._tick(1)

            assert reply == "我还缺一个参数，麻烦补充一下。"
            msgs = await loop.task_store.get_chat_messages_since(0, "wechat:user-1")
            assert len(msgs) == 1
            assert msgs[0]["role"] == "assistant"
            assert msgs[0]["content"] == "我还缺一个参数，麻烦补充一下。"
        finally:
            await loop.task_store.close()
            await loop.provider.close()


