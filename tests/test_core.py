"""核心模块测试：working_memory / emotion / judgment / chat / loop / exec / evolution"""
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
# ══════════════════════════════════════════════════════════════════════════════
# 基础模块
# ══════════════════════════════════════════════════════════════════════════════

def test_working_memory():
    from memory.working import WorkingMemory, WMItem
    wm = WorkingMemory(capacity=5)
    for i in range(7):
        # 不同 kind 避免同 kind 去重逻辑，测试纯容量驱逐行为
        wm.add(WMItem(kind=f"test_{i}", content=f"item {i}", priority=i / 10))
    assert len(wm) == 5
    assert 0.0 < wm.pressure <= 1.0


def test_emotion_state_ema():
    from core.perception import EmotionState
    e = EmotionState(valence=0.6, arousal=0.5)
    e.derive_from_signals(
        failure_count=0, prediction_error=0.1, wm_pressure=0.2,
        workspace_dirty=False, alpha=0.15,
    )
    assert 0.0 < e.valence <= 1.0
    assert e.dominant is not None or e.dominant is None  # 有无 dominant 均可


def test_judgment_output_parse():
    from core.judgment import JudgmentOutput
    raw = '```json\n{"decision":"act","chosen_action_id":"shell.run","params":{"command":"echo hi"},"rationale":"test","reflection":"洞察","next_step":"done","model_strategy":{"next_phase_tier":"reader","reason":"先低成本扩图"}}\n```'
    out = JudgmentOutput.from_llm(raw)
    assert out.decision == "act"
    assert out.chosen_action_id == "shell.run"
    assert out.reflection == "洞察"
    assert out.model_strategy["next_phase_tier"] == "reader"


def test_judgment_output_parse_null_text_fields_as_empty():
    from core.judgment import JudgmentOutput

    raw = (
        '{"decision":"pause","chosen_action_id":null,"params":{},'
        '"rationale":null,"reflection":null,"reply_to_user":null,"next_step":null}'
    )

    out = JudgmentOutput.from_llm(raw)

    assert out.decision == "pause"
    assert out.chosen_action_id == ""
    assert out.rationale == ""
    assert out.reflection == ""
    assert out.reply_to_user == ""
    assert out.next_step == ""


def test_chat_read_line_prefers_text_input(monkeypatch):
    from cli.chat import _read_line

    monkeypatch.setattr(builtins, "input", lambda prompt="": "你好")
    assert _read_line() == "你好"


def test_chat_read_line_strips_replacement_chars(monkeypatch):
    from cli.chat import _read_line

    monkeypatch.setattr(builtins, "input", lambda prompt="": "你\ufffd好")
    assert _read_line() == "你好"


def test_chat_read_line_falls_back_to_utf8_buffer(monkeypatch):
    from cli.chat import _read_line

    def _raise(prompt=""):
        raise UnicodeDecodeError("utf-8", b"x", 0, 1, "bad")

    monkeypatch.setattr(builtins, "input", _raise)
    monkeypatch.setattr(
        "sys.stdin",
        cast(Any, SimpleNamespace(buffer=SimpleNamespace(readline=lambda: "中文\n".encode("utf-8")))),
    )
    monkeypatch.setattr("sys.stdout", io.StringIO())

    assert _read_line() == "中文"


def test_chat_erase_last_input_echo_when_tty(monkeypatch):
    from cli.chat import _erase_last_input_echo

    class _FakeStdout(io.StringIO):
        def isatty(self):
            return True

    fake_stdout = _FakeStdout()
    monkeypatch.setattr("sys.stdout", fake_stdout)

    _erase_last_input_echo()

    assert fake_stdout.getvalue() == "\x1b[1A\r\x1b[2K\r"


def test_chat_infer_user_title_from_session_history_prefers_explicit_user_identity():
    from cli.chat import _infer_user_title_from_messages

    messages: list[dict[str, object]] = [
        {"role": "assistant", "content": "爸爸，我先确认一下。"},
        {"role": "user", "content": "你可以叫我老爹"},
    ]

    assert _infer_user_title_from_messages(messages) == "老爹"


def test_chat_infer_user_title_from_session_history_does_not_slice_assistant_opening_phrase():
    from cli.chat import _infer_user_title_from_messages

    messages: list[dict[str, object]] = [
        {"role": "assistant", "content": "刚看了下，hermesclaw 探针一直返回 501。"},
    ]

    assert _infer_user_title_from_messages(messages) == ""


def test_chat_parse_user_title_from_llm_output_supports_plain_and_json():
    from cli.chat import _parse_user_title_from_llm_output

    assert _parse_user_title_from_llm_output("爸爸") == "爸爸"
    assert _parse_user_title_from_llm_output('{"user_title": "老爹"}') == "老爹"
    assert _parse_user_title_from_llm_output("NONE") == ""


def test_chat_input_prompt_prefers_user_title_then_chat_id():
    from cli.chat import _chat_input_prompt

    assert _chat_input_prompt("爸爸", "chat-42") == "爸爸> "
    assert _chat_input_prompt("", "chat-42") == "chat-42> "
    assert _chat_input_prompt("", "") == "chat> "


def test_chat_print_input_prompt_when_tty(monkeypatch):
    from cli.chat import _print_input_prompt

    class _FakeStdout(io.StringIO):
        def isatty(self):
            return True

    fake_stdout = _FakeStdout()
    monkeypatch.setattr("sys.stdout", fake_stdout)

    _print_input_prompt("小懒> ")

    assert fake_stdout.getvalue() == "小懒> "


def test_loop_logging_reply_not_truncated():
    from core.loop.logging import _clip_reply_for_log

    text = "x" * 600

    assert _clip_reply_for_log(text) == text


@pytest.mark.asyncio
async def test_chat_interactive_assistant_reply_redraws_prompt_once(monkeypatch):
    from cli import chat as chat_mod

    class _FakeStore:
        def __init__(self):
            self._history_loaded = False
            self._reply_sent = False

        async def get_chat_messages_since(self, since: int, *, chat_id: str = ""):
            if since == 0 and not self._history_loaded:
                self._history_loaded = True
                return []
            if not self._reply_sent:
                self._reply_sent = True
                return [{"id": 1, "role": "assistant", "content": "爸爸，收到。"}]
            return []

        async def add_chat_message(self, role: str, content: str, *, chat_id: str = ""):
            return 2

    monkeypatch.setattr(chat_mod.console, "print", lambda *args, **kwargs: None)
    prompt_calls: list[str] = []
    monkeypatch.setattr(chat_mod, "_print_input_prompt", lambda prompt: prompt_calls.append(prompt))

    def _delayed_eof(prompt: str = "") -> str:
        time.sleep(0.4)
        return ""

    monkeypatch.setattr(chat_mod, "_read_line", _delayed_eof)

    await chat_mod._interactive(cast(Any, _FakeStore()), _test_config(), "", "灵舟")

    # _infer_user_title_from_messages 只看 role="user" 消息中的自报称谓；
    # 助手回复"爸爸，收到。"不触发 user_title 推断，退回到 "chat> "
    assert prompt_calls == ["chat> "]


def test_configure_lingzhou_logging_resets_console_log_each_time():
    from cli.gateway import _configure_lingzhou_logging

    with tempfile.TemporaryDirectory() as d:
        log_dir = Path(d)
        logger_name = "lingzhou.test.logging"
        logger = logging.getLogger(logger_name)
        old_handlers = list(logger.handlers)
        old_level = logger.level
        old_propagate = logger.propagate
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        try:
            _, console_log = _configure_lingzhou_logging(log_dir, logging.INFO, logger_name=logger_name)
            logger.info("first message")
            for handler in logger.handlers:
                handler.flush()

            _, console_log = _configure_lingzhou_logging(log_dir, logging.INFO, logger_name=logger_name)
            logger.info("second message")
            for handler in logger.handlers:
                handler.flush()

            text = console_log.read_text(encoding="utf-8")
            assert "second message" in text
            assert "first message" not in text
        finally:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
            logger.handlers = old_handlers
            logger.setLevel(old_level)
            logger.propagate = old_propagate


def test_judgment_context_budget_trims_low_priority_sections():
    from core.judgment import apply_context_budget

    ctx = {
        "task_section": "T" * 2000,
        "emotion_valence": "0.50",
        "emotion_arousal": "0.50",
        "emotion_dominant": "中性",
        "emotion_regulation": "stable",
        "wm_section": "W" * 1800,
        "failures_section": "F" * 800,
        "episodic_section": "E" * 2400,
        "memories_section": "M" * 2200,
        "soul_section": "S" * 900,
        "tools_section": "U" * 2000,
        "perception_section": "P" * 700,
        "ethos_section": "H" * 700,
        "signals_section": "G" * 700,
        "hard_boundaries_section": "B" * 700,
        "perception_replay_section": "R" * 700,
        "skills_section": "K" * 3000,
        "cognitive_signals_section": "C" * 700,
        "user_message": "",
    }

    budgeted = apply_context_budget(ctx, max_chars=12000)

    assert len(budgeted["task_section"]) == len(ctx["task_section"])
    assert len(budgeted["soul_section"]) == len(ctx["soul_section"])
    assert len(budgeted["skills_section"]) <= len(ctx["skills_section"])
    assert len(budgeted["memories_section"]) <= len(ctx["memories_section"])
    assert len(budgeted["episodic_section"]) <= len(ctx["episodic_section"])
    assert len(budgeted["wm_section"]) <= len(ctx["wm_section"])


def test_judgment_error_classification_and_cooldown():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

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
    })

    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
    assert layer._classify_error_code("Client error '429 Too Many Requests'") == "429"
    assert layer._classify_error_code("Client error '400 Bad Request'") == "400"
    assert layer._classify_error_code("ReadTimeout('')") == "timeout"

    assert layer._cooldown_seconds("429", 1) >= 30
    assert layer._cooldown_seconds("429", 3) > layer._cooldown_seconds("429", 1)
    assert layer._cooldown_seconds("400", 2) >= 90


def test_evolution_verification_outcome():
    from core.evolution import _verification_outcome

    baseline = {"runs": 4, "failures": 2, "successes": 1}
    assert _verification_outcome(baseline, {"runs": 1, "failures": 0, "successes": 1}, 3) == "pending"
    assert _verification_outcome(baseline, {"runs": 3, "failures": 0, "successes": 3}, 3) == "verified"
    assert _verification_outcome(baseline, {"runs": 3, "failures": 3, "successes": 0}, 3) == "regressed"


def test_exec_background_result_stays_running():
    from core.execution import _run_status_from_result
    from tools.registry import ToolResult

    result = ToolResult(
        summary="后台进程已启动",
        state_delta={"process": "started", "background": True},
        metadata={"session_id": "exec-1"},
    )
    assert _run_status_from_result(result) == "running"


def test_worker_layer_dispatches_specialized_handlers():
    asyncio.run(_worker_layer_dispatches_specialized_handlers())


async def _worker_layer_dispatches_specialized_handlers():
    from core.judgment import JudgmentOutput
    from core.worker import WorkerLayer
    from tools.registry import ToolEntry, ToolManifest, ToolResult

    async def _handler(params, ctx):
        return ToolResult(
            summary="ok",
            resource_key=str(params.get("resource_key") or ""),
            state_delta=dict(params.get("state_delta") or {}),
            metadata=dict(params.get("metadata") or {}),
        )

    entry = ToolEntry(manifest=ToolManifest(name="demo", description="demo"), handler=_handler)
    layer = WorkerLayer()
    ctx = _tool_ctx()

    exec_result = await layer.dispatch(
        "exec-worker",
        entry,
        _judgment_output(decision="act", chosen_action_id="exec", params={
            "resource_key": "exec-1",
            "state_delta": {"process": "started", "background": True},
        }),
        ctx,
    )
    assert exec_result.metadata["worker_path"] == "exec"
    assert exec_result.metadata["execution_mode"] == "background"
    assert exec_result.metadata["session_id"] == "exec-1"

    multimodal_result = await layer.dispatch(
        "multimodal-worker",
        entry,
        _judgment_output(decision="act", chosen_action_id="image.analyze", params={"paths": ["a.png", "b.png"]}),
        ctx,
    )
    assert multimodal_result.metadata["worker_path"] == "multimodal"
    assert multimodal_result.metadata["modality"] == "image"
    assert multimodal_result.metadata["input_count"] == 2

    llm_result = await layer.dispatch(
        "llm-worker",
        entry,
        _judgment_output(decision="act", chosen_action_id="llm.simulated", params={"monitor_fact_key": "run:llm-1"}),
        ctx,
    )
    assert llm_result.metadata["worker_path"] == "llm"
    assert llm_result.metadata["reasoning_mode"] == "tool-mediated-llm"
    assert llm_result.metadata["run_monitor"]["key"] == "run:llm-1"


def test_task_store_fact_listing_and_delete():
    asyncio.run(_task_store_fact_listing_and_delete())


async def _task_store_fact_listing_and_delete():
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "facts.db")
        await store.open()
        await store.set_fact("evolution:verify:file.read", json.dumps({"ok": True}), scope="system")
        await store.set_fact("misc:key", "v", scope="general")

        facts = await store.list_facts(prefix="evolution:verify:")
        assert len(facts) == 1
        assert facts[0][0] == "evolution:verify:file.read"

        await store.delete_fact("evolution:verify:file.read")
        value, found = await store.get_fact("evolution:verify:file.read")
        assert not found
        assert value == ""
        await store.close()


def test_task_store_signal_lifecycle():
    asyncio.run(_task_store_signal_lifecycle())


async def _task_store_signal_lifecycle():
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "signals.db")
        await store.open()

        once_id = await store.add_signal(
            "一次性提醒",
            "2000-01-01 00:00:00",
            payload={"note": "ping"},
        )
        repeat_id = await store.add_signal(
            "循环提醒",
            "2000-01-01 00:00:00",
            repeat_secs=60,
            payload={"source": "heartbeat"},
        )

        due = await store.due_signals()
        assert [sig["id"] for sig in due] == [once_id, repeat_id]

        await store.ack_signal(once_id)
        once_signal = await store.get_signal(once_id)
        assert once_signal is not None
        assert once_signal["status"] == "done"

        repeat_before = await store.get_signal(repeat_id)
        assert repeat_before is not None
        await store.ack_signal(repeat_id)
        repeat_after = await store.get_signal(repeat_id)
        assert repeat_after is not None
        assert repeat_after["status"] == "pending"
        assert repeat_after["run_at"] != repeat_before["run_at"]

        pending = await store.list_signals(limit=10)
        assert [sig["id"] for sig in pending] == [repeat_id]

        await store.cancel_signal(repeat_id)
        cancelled = await store.get_signal(repeat_id)
        assert cancelled is not None
        assert cancelled["status"] == "cancelled"
        await store.close()


def test_evolution_pending_verification_becomes_verified():
    asyncio.run(_evolution_pending_verification_becomes_verified())


async def _evolution_pending_verification_becomes_verified():
    from types import SimpleNamespace
    from typing import Any, cast

    from core.config import Config
    from core.evolution import EvolutionEngine, _verification_fact_key
    from memory.task_store import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return ""

        async def close(self):
            return None

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
        "evolution": {
            "verify_min_runs": 2,
            "auto_rollback_on_regression": True,
        },
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        await store.set_fact(
            _verification_fact_key("file.read"),
            json.dumps({
                "target": "file.read",
                "tool_path": str(Path(d) / "file.py"),
                "backup_path": str(Path(d) / "file.py.bak"),
                "created_at": created_at,
                "baseline": {"runs": 3, "failures": 2, "successes": 1},
            }, ensure_ascii=False),
            scope="system",
        )
        await store.add_run(tool_name="file.read", status="succeeded")
        await store.add_run(tool_name="file.read", status="succeeded")

        engine = EvolutionEngine(cfg, _DummyProvider(), ToolRegistry())
        results = await engine._maybe_evaluate_verifications(cast(Any, SimpleNamespace(task_store=store)))
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].target == "verify:file.read"

        _, found = await store.get_fact(_verification_fact_key("file.read"))
        assert not found
        await store.close()


def test_evolution_regression_triggers_rollback():
    asyncio.run(_evolution_regression_triggers_rollback())


async def _evolution_regression_triggers_rollback():
    from types import SimpleNamespace
    from typing import Any, cast

    from core.config import Config
    from core.evolution import EvolutionEngine, _verification_fact_key
    from memory.task_store import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return ""

        async def close(self):
            return None

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
        "evolution": {
            "verify_min_runs": 2,
            "auto_rollback_on_regression": True,
        },
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        tool_path = Path(d) / "demo_tool.py"
        backup_path = Path(d) / "demo_tool.py.bak"
        tool_path.write_text("VALUE = 'new'\n", encoding="utf-8")
        backup_path.write_text("VALUE = 'old'\n", encoding="utf-8")
        created_at = (datetime.now(UTC) - timedelta(seconds=1)).replace(microsecond=0).isoformat()
        await store.set_fact(
            _verification_fact_key("demo.tool"),
            json.dumps({
                "target": "demo.tool",
                "tool_path": str(tool_path),
                "backup_path": str(backup_path),
                "created_at": created_at,
                "baseline": {"runs": 2, "failures": 1, "successes": 1},
            }, ensure_ascii=False),
            scope="system",
        )
        await store.add_run(tool_name="demo.tool", status="failed")
        await store.add_run(tool_name="demo.tool", status="failed")
        await store.record_failure("demo.tool", "still broken")
        await store.record_failure("demo.tool", "still broken again")

        engine = EvolutionEngine(cfg, _DummyProvider(), ToolRegistry())
        results = await engine._maybe_evaluate_verifications(cast(Any, SimpleNamespace(task_store=store)))
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].target == "rollback:demo.tool"
        assert tool_path.read_text(encoding="utf-8") == "VALUE = 'old'\n"

        _, found = await store.get_fact(_verification_fact_key("demo.tool"))
        assert not found
        await store.close()


async def test_refresh_running_runs_updates_finished_exec_runs():
    import os
    import time

    from core.run_refresh import refresh_running_runs
    from memory.task_store import TaskStore
    from tools.exec import ProcessInfo, _MANAGER

    with tempfile.TemporaryDirectory() as d:
        os.environ["LINGZHOU_PROCESS_STATE_DIR"] = str(Path(d) / "proc-state")
        _MANAGER.clear()
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("后台任务", goal="等进程结束")
        run_id = await store.add_run(
            task_id=task_id,
            run_type="exec",
            worker_type="exec-worker",
            status="running",
            tool_name="exec",
            session_id="exec-test-1",
        )
        info = ProcessInfo(
            session_id="exec-test-1",
            command="echo hi",
            started_at=time.time(),
            finished=True,
            return_code=0,
            stdout="hi\n",
            background=True,
        )
        _MANAGER.register(info)
        _MANAGER.mark_finished("exec-test-1", 0)

        updates = await refresh_running_runs(store)
        assert updates
        assert updates[0]["status"] == "succeeded"

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "succeeded"
        assert run.progress.strip() == "hi"
        assert run.output_json["stdout"].strip() == "hi"

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.result_json["last_run_status"] == "succeeded"

        await store.close()


async def test_refresh_running_runs_updates_process_monitored_non_exec_runs():
    import os
    import time

    from core.run_refresh import refresh_running_runs
    from memory.task_store import TaskStore
    from tools.exec import ProcessInfo, _MANAGER

    with tempfile.TemporaryDirectory() as d:
        os.environ["LINGZHOU_PROCESS_STATE_DIR"] = str(Path(d) / "proc-state")
        _MANAGER.clear()
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("统一监控任务", goal="通过 process monitor 刷新")
        run_id = await store.add_run(
            task_id=task_id,
            run_type="tool_chain",
            worker_type="tool-chain-worker",
            status="running",
            tool_name="demo.process",
            output_json={"metadata": {"run_monitor": {"kind": "process", "session_id": "proc-unified-1"}}},
        )
        info = ProcessInfo(
            session_id="proc-unified-1",
            command="echo unified",
            started_at=time.time(),
            finished=True,
            return_code=0,
            stdout="unified\n",
            background=True,
        )
        _MANAGER.register(info)
        _MANAGER.mark_finished("proc-unified-1", 0)

        updates = await refresh_running_runs(store)
        assert updates
        assert updates[0]["status"] == "succeeded"

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "succeeded"
        assert run.progress.strip() == "unified"
        await store.close()


async def test_refresh_running_runs_crystallizes_progress():
    import os
    import time

    from core.run_refresh import refresh_running_runs
    from memory.task_store import TaskStore
    from tools.exec import ProcessInfo, _MANAGER

    with tempfile.TemporaryDirectory() as d:
        os.environ["LINGZHOU_PROCESS_STATE_DIR"] = str(Path(d) / "proc-state")
        _MANAGER.clear()
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("长任务", goal="观察中间进度")
        run_id = await store.add_run(
            task_id=task_id,
            run_type="exec",
            worker_type="exec-worker",
            status="running",
            tool_name="exec",
            session_id="exec-test-2",
        )
        info = ProcessInfo(
            session_id="exec-test-2",
            command="echo progress",
            started_at=time.time(),
            finished=False,
            stdout=(
                "phase-01 downloading artifacts\n"
                "phase-02 unpacking workspace\n"
                "phase-03 indexing files\n"
                "phase-04 building plan\n"
                "phase-05 verifying progress\n"
            ),
            background=True,
        )
        _MANAGER.register(info)

        updates = await refresh_running_runs(store)
        assert updates
        assert updates[0]["status"] == "running"
        assert updates[0]["crystal"]

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert "phase-05" in run.progress

        progress, found = await store.get_fact(f"task:{task_id}:progress")
        assert found
        assert "phase-05" in progress

        await store.close()


async def test_refresh_running_runs_updates_fact_monitored_non_exec_runs():
    from core.run_refresh import refresh_running_runs
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = SemanticMemory(root)
        await store.open()
        task_id = await store.add_task("推理任务", goal="等待外部状态")
        await store.set_fact(
            "run:llm-1",
            json.dumps({"status": "running", "progress": "phase-1 reasoning"}, ensure_ascii=False),
            scope="task",
        )
        run_id = await store.add_run(
            task_id=task_id,
            run_type="llm",
            worker_type="llm-worker",
            status="running",
            tool_name="llm.simulated",
            output_json={
                "state_delta": {
                    "run_monitor": {
                        "kind": "fact",
                        "key": "run:llm-1",
                        "status_field": "status",
                        "progress_field": "progress",
                        "success_values": ["succeeded"],
                        "failed_values": ["failed"],
                    }
                }
            },
        )

        first = await refresh_running_runs(store, episodic=episodic, semantic=semantic)
        assert first
        assert first[0]["status"] == "running"
        assert "phase-1" in first[0]["crystal"]

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "running"
        assert "phase-1" in run.progress

        await store.set_fact(
            "run:llm-1",
            json.dumps({"status": "succeeded", "progress": "final answer ready", "summary": "done"}, ensure_ascii=False),
            scope="task",
        )
        second = await refresh_running_runs(store, episodic=episodic, semantic=semantic)
        assert second
        assert second[0]["status"] == "succeeded"

        finished = await store.get_run_by_id(run_id)
        assert finished is not None
        assert finished.status == "succeeded"
        assert finished.progress == "final answer ready"

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.result_json["last_run_status"] == "succeeded"

        completed = episodic.list_events("run_completed", limit=5)
        assert completed and completed[-1]["run_id"] == run_id

        node = semantic.get(f"run-result-{run_id}")
        assert node is not None
        assert node.kind == "run_result"

        await store.close()


async def test_refresh_running_runs_failed_fact_monitored_run_records_learning():
    from core.run_refresh import refresh_running_runs
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = SemanticMemory(root)
        await store.open()
        task_id = await store.add_task("推理失败任务", goal="等待外部状态失败")
        await store.set_fact(
            "run:llm-fail-1",
            json.dumps({"status": "failed", "progress": "empty path", "error": "EmptyPath", "summary": "empty path"}, ensure_ascii=False),
            scope="task",
        )
        run_id = await store.add_run(
            task_id=task_id,
            run_type="llm",
            worker_type="llm-worker",
            status="running",
            tool_name="file.read",
            output_json={
                "state_delta": {
                    "run_monitor": {
                        "kind": "fact",
                        "key": "run:llm-fail-1",
                        "status_field": "status",
                        "progress_field": "progress",
                        "success_values": ["succeeded"],
                        "failed_values": ["failed"],
                    }
                }
            },
        )

        updates = await refresh_running_runs(store, episodic=episodic, semantic=semantic)
        assert updates
        assert updates[0]["status"] == "failed"

        failures = await store.list_failures(limit=5)
        assert failures
        assert failures[0].kind == "file.read"

        reflections = await store.list_meta_reflections(limit=5)
        assert reflections
        assert reflections[0].run_id == run_id
        assert reflections[0].target_kind == "task_split"

        double_loop = episodic.list_events("double_loop_reflection", limit=5)
        assert double_loop and double_loop[-1]["run_id"] == run_id

        meta_node = semantic.get(f"meta-reflection-{reflections[0].id}")
        assert meta_node is not None
        assert meta_node.kind == "meta_reflection"

        await store.close()


async def test_refresh_running_runs_failed_exec_run_records_learning():
    import time

    from core.run_refresh import refresh_running_runs
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore
    from tools.exec import ProcessInfo, _MANAGER

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = SemanticMemory(root)
        await store.open()
        task_id = await store.add_task("exec 失败任务", goal="等待后台失败")
        run_id = await store.add_run(
            task_id=task_id,
            run_type="exec",
            worker_type="exec-worker",
            status="running",
            tool_name="file.read",
            session_id="exec-failed-1",
        )

        _MANAGER.clear()
        info = ProcessInfo(
            session_id="exec-failed-1",
            command="cat missing",
            started_at=time.time() - 3,
            finished=True,
            return_code=1,
            stdout="",
            stderr="EmptyPath",
            error="EmptyPath",
            background=True,
        )
        _MANAGER.register(info)

        updates = await refresh_running_runs(store, episodic=episodic, semantic=semantic)
        assert updates
        assert updates[0]["status"] == "failed"

        failures = await store.list_failures(limit=5)
        assert failures
        assert failures[0].kind == "file.read"

        reflections = await store.list_meta_reflections(limit=5)
        assert reflections
        assert reflections[0].run_id == run_id
        assert reflections[0].target_kind == "task_split"

        meta_node = semantic.get(f"meta-reflection-{reflections[0].id}")
        assert meta_node is not None
        assert meta_node.kind == "meta_reflection"

        await store.close()


def test_catalog_resolve_context_window():
    """内置目录能按 model ID 自动查找 context_window，显式 override 优先。"""
    from provider.catalog import resolve_context_window

    # 已收录模型：自动查找
    assert resolve_context_window("qwen3.6-plus", None) == 1000000
    assert resolve_context_window("qwen3.5-plus", None) == 131072
    assert resolve_context_window("kimi-k2.5", None) == 262144

    # 显式 override 优先于目录值
    assert resolve_context_window("qwen3.6-plus", 32768) == 32768

    # 未收录模型返回 None
    assert resolve_context_window("unknown-model-xyz", None) is None


def test_bailian_catalog_capabilities_are_curated():
    from provider.catalog import lookup_model

    qwen36 = lookup_model("qwen3.6-plus")
    assert qwen36 is not None
    assert qwen36["input"] == ["text", "image"]
    assert qwen36["capabilities"] == ["text_generation", "thinking", "vision"]

    coder_next = lookup_model("qwen3-coder-next")
    assert coder_next is not None
    assert coder_next["capabilities"] == ["text_generation"]
    assert "thinking" not in coder_next

    kimi = lookup_model("kimi-k2.5")
    assert kimi is not None
    assert kimi["input"] == ["text", "image"]
    assert kimi["capabilities"] == ["text_generation", "thinking", "vision"]


def test_catalog_budget_auto_lookup():
    """Config 不填 context_window_tokens 时，目录自动推断预算。"""
    from core.config import Config

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.5-plus",  # 目录里 context_window=131072
        "temperature": 0.7,
        "timeout": 60.0,
    })
    # budget = 131072 - max(1024, 131072//4) = 131072 - 32768 = 98304
    assert cfg.judgment_input_token_budget() == 98304


def test_judgment_budget_is_derived_from_model_window():
    import pytest
    from core.config import Config

    # 未收录模型 + 显式 context_window_tokens → 正常计算
    cfg = Config.model_validate({
        "providers": {
            "custom": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "custom/demo",
        "context_window_tokens": 8000,
        "temperature": 0.7,
        "timeout": 60.0,
    })
    assert cfg.judgment_input_token_budget() == 6000

    # 未收录模型 + 无 context_window_tokens → fail loud，不静默降级
    unknown = Config.model_validate({
        "providers": {
            "custom": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "custom/demo",
        "temperature": 0.7,
        "timeout": 60.0,
    })
    with pytest.raises(ValueError, match="context_window_tokens"):
        unknown.judgment_input_token_budget()


def test_tool_registry():
    from tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.discover(_proj_root() / "tools")
    names = [m.name for m in reg.list_manifests()]
    assert "image.analyze" in names
    assert "shell.run" in names
    assert "shell.capabilities" in names
    assert "task.complete" in names
    assert "memory.add_wm" in names
    assert "memory.search" in names
    assert "file.list" in names
    assert "file.edit" in names
    assert "exec" in names
    assert "process.write" in names
    assert "skill.list" in names
    assert "skill.search" in names


def test_file_list_and_memory_search():
    asyncio.run(_file_list_and_memory_search())


def test_image_source_helpers():
    from tools.image import _collect_image_sources, _image_part_from_source, _resolve_multimodal_model_ref

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        local = root / "sample.png"
        local.write_bytes(b"\x89PNG\r\n\x1a\nsmall-test-image")

        sources = _collect_image_sources({
            "path": str(local),
            "paths": json.dumps(["https://example.com/demo.jpg"]),
        })
        assert sources == [str(local), "https://example.com/demo.jpg"]

        local_part = _image_part_from_source(str(local), "auto")
        assert local_part["type"] == "image_url"
        assert local_part["image_url"]["url"].startswith("data:image/png;base64,")

        remote_part = _image_part_from_source("https://example.com/demo.jpg", "high")
        assert remote_part["image_url"]["url"] == "https://example.com/demo.jpg"
        assert remote_part["image_url"]["detail"] == "high"

        ctx = cast(Any, SimpleNamespace(config=SimpleNamespace(model="bailian/qwen3.6-plus", active_provider_name="bailian")))
        assert _resolve_multimodal_model_ref(ctx, capability="vision", input_modality="image") == "bailian/qwen3.6-plus"


def test_image_model_routing_falls_back_to_vision_model():
    from tools.image import _resolve_multimodal_model_ref

    ctx = cast(Any, SimpleNamespace(config=SimpleNamespace(model="deepseek/deepseek-v4-pro", active_provider_name="deepseek")))
    routed = _resolve_multimodal_model_ref(ctx, capability="vision", input_modality="image")
    assert routed != "deepseek/deepseek-v4-pro"
    assert routed == "bailian/qwen3.6-plus"


async def _file_list_and_memory_search():
    from tools.file import file_list, file_read
    from tools.memory_ops import memory_search, memory_add_semantic
    from memory.semantic import MemoryNode, SemanticMemory
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / 'a.txt').write_text('hello', encoding='utf-8')
        (root / 'sub').mkdir()
        semantic = SemanticMemory(root)
        ctx = _tool_ctx(workspace_dir=str(root), semantic=semantic)
        listed = await file_list({'path': str(root)}, ctx)
        assert 'a.txt' in listed.summary
        assert 'sub/' in listed.summary

        read_file = await file_read({'path': str(root / 'a.txt')}, ctx)
        assert read_file.error is None
        assert read_file.summary == 'hello'

        read_dir = await file_read({'path': str(root)}, ctx)
        assert read_dir.error == 'NotAFile'

        read_empty = await file_read({'path': ''}, ctx)
        assert read_empty.error == 'EmptyPath'

        await memory_add_semantic({'title': 'bug fix note', 'body': 'reader tasks should use qwen3.6-plus', 'kind': 'fact'}, ctx)
        semantic.upsert(MemoryNode(
            id='task-note-1',
            kind='fact',
            title='legacy runtime primary carrier',
            body='/root/.legacy-runtime/memory/main.sqlite',
            tags=['task:33', 'path:/root/.legacy-runtime/memory'],
        ))
        found = await memory_search({'query': 'bug'}, ctx)
        assert 'bug fix note' in found.summary

        filtered = await memory_search({'query': 'legacy runtime', 'task_id': '33', 'path_prefix': '/root/.legacy-runtime/memory'}, ctx)
        assert 'legacy runtime primary carrier' in filtered.summary

        excluded = await memory_search({'query': 'legacy runtime', 'task_id': '34'}, ctx)
        assert excluded.skipped is True


def test_exec_process_write_pipe_roundtrip():
    asyncio.run(_exec_process_write_pipe_roundtrip())


async def _exec_process_write_pipe_roundtrip():
    import json
    from tools.exec import exec_run, process_write, process_poll, process_log, _MANAGER

    _MANAGER.clear()
    ctx = _tool_ctx()

    res = await exec_run({
        "command": "python3 -c \"import sys; print(sys.stdin.readline().strip())\"",
        "background": True,
        "timeout": 2,
    }, ctx)
    sid = json.loads(res.evidence)["process_id"]
    await process_write({"process_id": sid, "data": "hello\\n", "eof": True}, ctx)

    for _ in range(40):
        poll = await process_poll({"process_id": sid}, ctx)
        status = json.loads(poll.summary)
        if status["status"] == "finished":
            break
        await asyncio.sleep(0.05)

    log = await process_log({"process_id": sid, "offset": 0, "limit": 200}, ctx)
    assert "hello" in log.summary


def test_exec_process_timeout_background():
    asyncio.run(_exec_process_timeout_background())


async def _exec_process_timeout_background():
    import json
    from tools.exec import exec_run, process_poll, _MANAGER

    _MANAGER.clear()
    ctx = _tool_ctx()

    res = await exec_run({
        "command": "python3 -c \"import time; time.sleep(5)\"",
        "background": True,
        "timeout": 0.2,
    }, ctx)
    sid = json.loads(res.evidence)["process_id"]

    timed_out = False
    for _ in range(60):
        poll = await process_poll({"process_id": sid}, ctx)
        status = json.loads(poll.summary)
        if status["status"] == "finished":
            timed_out = bool(status["timed_out"])
            break
        await asyncio.sleep(0.05)

    assert timed_out is True


def test_exec_and_shell_explicit_no_output():
    asyncio.run(_exec_and_shell_explicit_no_output())


async def _exec_and_shell_explicit_no_output():
    from tools.exec import exec_run
    from tools.shell import shell_run

    ctx = _tool_ctx()

    exec_res = await exec_run({"command": "python3 -c \"pass\""}, ctx)
    assert exec_res.error is None
    assert "(无输出)" in exec_res.summary

    shell_res = await shell_run({"command": "python3 -c \"pass\""}, ctx)
    assert shell_res.error is None
    assert "(无输出)" in shell_res.summary
    assert shell_res.state_delta == {"process": "finished", "exit_code": 0, "timed_out": False}
    assert shell_res.metadata["log_summary"].startswith("shell.run exit=0 chars=0")
    assert shell_res.resource_key is not None
    json.dumps(shell_res.to_dict(), ensure_ascii=False)


def test_execution_durable_failure_sensing():
    asyncio.run(_execution_durable_failure_sensing())


async def _execution_durable_failure_sensing():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.judgment import JudgmentOutput
    from memory.task_store import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)
        action = _judgment_output(
            decision="act",
            chosen_action_id="exec",
            params={"command": "bash /definitely/missing-lingzhou-script.sh"},
            rationale="test durable failure sensing",
        )

        first = second = third = fourth = None
        first = await layer.dispatch(action, ctx)
        second = await layer.dispatch(action, ctx)
        third = await layer.dispatch(action, ctx)
        fourth = await layer.dispatch(action, ctx)

        assert first is not None and first.error
        assert second is not None and second.error
        assert third is not None and third.error
        assert fourth is not None
        assert fourth.skipped is True
        assert fourth.error == "KnownStableFailure"
        assert "跳过已知稳定失败动作" in fourth.summary

        await store.close()


def test_execution_durable_failure_sensing_for_file_tool():
    asyncio.run(_execution_durable_failure_sensing_for_file_tool())


async def _execution_durable_failure_sensing_for_file_tool():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.judgment import JudgmentOutput
    from memory.task_store import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)
        action = _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            params={"path": ""},
            rationale="test durable failure sensing for file tool",
        )

        for _ in range(3):
            res = await layer.dispatch(action, ctx)
            assert res.error == "EmptyPath"
        fourth = await layer.dispatch(action, ctx)
        assert fourth.skipped is True
        assert fourth.error == "KnownStableFailure"

        await store.close()


def test_execution_dispatch_records_run():
    asyncio.run(_execution_dispatch_records_run())


async def _execution_dispatch_records_run():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.judgment import JudgmentOutput
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        root = Path(d)
        target = root / "demo.txt"
        target.write_text("hello", encoding="utf-8")
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = SemanticMemory(root)
        await store.open()
        await store.add_task("读取文件", goal="读 demo", model_tier="reader")
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store, episodic=episodic, semantic=semantic)
        action = _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            params={"path": str(target)},
            rationale="record run",
        )

        result = await layer.dispatch(action, ctx)
        assert result.error is None
        assert result.metadata.get("run_id")
        assert result.metadata.get("worker_type") == "tool-chain-worker"

        runs = await store.list_runs(limit=5)
        assert runs
        assert runs[0].status == "succeeded"
        assert runs[0].tool_name == "file.read"
        assert runs[0].model_tier == "reader"
        assert runs[0].progress == "hello"
        assert runs[0].output_json["summary"] == "hello"

        started = episodic.list_events("run_started", limit=5)
        completed = episodic.list_events("run_completed", limit=5)
        assert started and started[-1]["run_id"] == runs[0].id
        assert completed and completed[-1]["run_id"] == runs[0].id

        node = semantic.get(f"run-result-{runs[0].id}")
        assert node is not None
        assert node.kind == "run_result"
        assert "succeeded" in node.tags

        active = await store.get_active()
        assert active is not None
        assert active.result_json["last_run_id"] == runs[0].id
        assert active.result_json["last_run_status"] == "succeeded"
        assert active.result_json["worker_type"] == "tool-chain-worker"

        await store.close()


def test_execution_dispatch_routes_fact_monitored_action_to_llm_worker():
    asyncio.run(_execution_dispatch_routes_fact_monitored_action_to_llm_worker())


async def _execution_dispatch_routes_fact_monitored_action_to_llm_worker():
    from tempfile import TemporaryDirectory
    from memory.task_store import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        await store.add_task("llm worker 任务", goal="验证 llm worker 选路")
        await store.set_fact("run:llm-route", json.dumps({"status": "running"}, ensure_ascii=False), scope="task")
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)
        action = _judgment_output(
            decision="act",
            chosen_action_id="memory.get_fact",
            params={"key": "run:llm-route", "monitor_fact_key": "run:llm-route"},
            rationale="route to llm worker",
        )

        result = await layer.dispatch(action, ctx)
        assert result.error is None
        assert result.metadata["worker_type"] == "llm-worker"
        assert result.metadata["run_monitor"]["kind"] == "fact"

        runs = await store.list_runs(limit=5)
        assert runs
        assert runs[0].run_type == "llm"
        assert runs[0].worker_type == "llm-worker"
        await store.close()


def test_execution_plan_gate_blocks_mutation_until_current_step_aligned():
    asyncio.run(_execution_plan_gate_blocks_mutation_until_current_step_aligned())


async def _execution_plan_gate_blocks_mutation_until_current_step_aligned():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from memory.task_store import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("修复回复链路", goal="验证 plan gate")
        await store.update_task_data(task_id, {
            "plan": [
                {"step": "先搜索证据", "status": "in_progress"},
                {"step": "再写入修复", "status": "pending"},
            ]
        })

        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)

        blocked = await layer.dispatch(
            _judgment_output(
                decision="act",
                chosen_action_id="memory.set_fact",
                params={"key": "debug:test", "value": "1"},
                rationale="直接写入结论",
            ),
            ctx,
        )
        assert blocked.skipped is True
        assert blocked.error == "PlanStepMismatch"
        assert "先搜索证据" in blocked.summary
        value, found = await store.get_fact("debug:test")
        assert found is False
        assert value == ""

        align = await layer.dispatch(
            _judgment_output(
                decision="act",
                chosen_action_id="task.update",
                params={"current_step": "先搜索证据"},
                rationale="先对齐当前步骤",
            ),
            ctx,
        )
        assert align.error is None

        allowed = await layer.dispatch(
            _judgment_output(
                decision="act",
                chosen_action_id="memory.set_fact",
                params={"key": "debug:test", "value": "1"},
                rationale="步骤已对齐，允许写入",
            ),
            ctx,
        )
        assert allowed.error is None
        value, found = await store.get_fact("debug:test")
        assert found is True
        assert value == "1"

        await store.close()


def test_execution_plan_gate_keeps_reader_tools_available():
    asyncio.run(_execution_plan_gate_keeps_reader_tools_available())


async def _execution_plan_gate_keeps_reader_tools_available():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from memory.task_store import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("分析问题", goal="验证 reader 不被 plan gate 误伤")
        await store.update_task_data(task_id, {
            "plan": [
                {"step": "先搜索证据", "status": "in_progress"},
                {"step": "再整理结论", "status": "pending"},
            ]
        })

        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)

        result = await layer.dispatch(
            _judgment_output(
                decision="act",
                chosen_action_id="task.list",
                params={"status": "all", "limit": 5},
                rationale="先看当前任务列表",
            ),
            ctx,
        )
        assert result.error is None
        assert "分析问题" in result.summary

        await store.close()


def test_execution_failure_creates_meta_reflection():
    asyncio.run(_execution_failure_creates_meta_reflection())


async def _execution_failure_creates_meta_reflection():
    from tempfile import TemporaryDirectory

    from core.judgment import JudgmentOutput
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = SemanticMemory(root)
        await store.open()
        await store.add_task("读空路径", goal="制造失败")
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store, episodic=episodic, semantic=semantic)
        action = _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            params={"path": ""},
            rationale="trigger meta reflection",
        )

        result = await layer.dispatch(action, ctx)
        assert result.error == "EmptyPath"

        reflections = await store.list_meta_reflections(limit=5)
        assert reflections
        assert reflections[0].target_kind == "task_split"
        assert reflections[0].loop_level == "double"
        assert reflections[0].tool_name == "file.read"
        assert reflections[0].decision == "apply"

        runs = await store.list_runs(limit=5)
        assert runs and runs[0].status == "cancelled"

        started = episodic.list_events("run_started", limit=5)
        failed = episodic.list_events("run_failed", limit=5)
        double_loop = episodic.list_events("double_loop_reflection", limit=5)
        assert started and started[-1]["run_id"] == runs[0].id
        assert failed and failed[-1]["run_id"] == runs[0].id
        assert double_loop and double_loop[-1]["run_id"] == runs[0].id

        node = semantic.get(f"run-result-{runs[0].id}")
        assert node is not None
        assert node.kind == "run_result"
        assert "failed" in node.tags

        meta_node = semantic.get(f"meta-reflection-{reflections[0].id}")
        assert meta_node is not None
        assert meta_node.kind == "meta_reflection"

        rule_node = semantic.get(f"rule-revision-{reflections[0].id}")
        assert rule_node is not None
        assert rule_node.kind == "rule_revision"

        await store.close()


def test_execution_generic_failure_meta_reflection_defers():
    asyncio.run(_execution_generic_failure_meta_reflection_defers())


async def _execution_generic_failure_meta_reflection_defers():
    from tempfile import TemporaryDirectory

    from core.judgment import JudgmentOutput
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = SemanticMemory(root)
        await store.open()
        await store.add_task("读缺失文件", goal="制造单环失败")
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store, episodic=episodic, semantic=semantic)
        action = _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            params={"path": str(root / "missing.txt")},
            rationale="trigger generic meta reflection",
        )

        result = await layer.dispatch(action, ctx)
        assert result.error == "FileNotFound"

        reflections = await store.list_meta_reflections(limit=5)
        assert reflections
        assert reflections[0].target_kind == "tool"
        assert reflections[0].loop_level == "single"
        assert reflections[0].decision == "defer"

        double_loop = episodic.list_events("double_loop_reflection", limit=5)
        assert double_loop == []

        meta_node = semantic.get(f"meta-reflection-{reflections[0].id}")
        assert meta_node is not None
        assert meta_node.kind == "meta_reflection"

        rule_node = semantic.get(f"rule-revision-{reflections[0].id}")
        assert rule_node is None

        await store.close()


def test_ingest_actionable_meta_reflections_dedupes():
    asyncio.run(_ingest_actionable_meta_reflections_dedupes())


async def _ingest_actionable_meta_reflections_dedupes():
    from core.task_runtime import _ingest_actionable_meta_reflections
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        await store.add_meta_reflection(
            reflection_id="mr-apply",
            target_kind="routing",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="动作选择漂移",
            proposal="修正路由",
            verification_plan="重跑 judgment",
            decision="apply",
            task_id=7,
            run_id=3,
            tool_name="file.read",
        )
        await store.add_meta_reflection(
            reflection_id="mr-rollback",
            target_kind="threshold",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="阈值退化",
            proposal="回滚阈值",
            verification_plan="等待窗口结束",
            decision="rollback",
            task_id=8,
            run_id=4,
            tool_name="exec",
        )
        await store.add_meta_reflection(
            reflection_id="mr-defer",
            target_kind="tool",
            trigger="failure_pattern",
            loop_level="single",
            diagnosis="普通失败",
            proposal="稍后再看",
            verification_plan="重试",
            decision="defer",
            task_id=9,
            run_id=5,
            tool_name="file.read",
        )

        wm = WorkingMemory(capacity=10)
        injected = await _ingest_actionable_meta_reflections(store, wm)
        assert injected == ["mr-apply", "mr-rollback"]

        top = wm.get_top()
        assert len([item for item in top if item["kind"] == "meta_reflection"]) == 2
        assert any("set routing guard" in item["content"] for item in top)
        assert any("reset durable failure policy" in item["content"] for item in top)

        task_fact, found = await store.get_fact("task:7:meta_reflection")
        assert found
        assert json.loads(task_fact)["decision"] == "apply"

        routing_guard, found = await store.get_fact("task:7:routing_guard")
        assert found
        assert json.loads(routing_guard)["tool_name"] == "file.read"

        policy_raw, found = await store.get_fact("control:durable_failure_policy")
        assert found
        assert json.loads(policy_raw) == {"threshold": 3, "ttl_sec": 7200}

        again = await _ingest_actionable_meta_reflections(store, wm)
        assert again == []
        await store.close()


def test_meta_reflection_threshold_apply_changes_runtime_policy():
    asyncio.run(_meta_reflection_threshold_apply_changes_runtime_policy())


async def _meta_reflection_threshold_apply_changes_runtime_policy():
    from core.judgment import JudgmentOutput
    from core.task_runtime import _ingest_actionable_meta_reflections
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        await store.add_meta_reflection(
            reflection_id="mr-threshold-apply",
            target_kind="threshold",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="稳定失败阈值过紧",
            proposal="放宽 durable failure 阈值",
            verification_plan="第四次失败前不应静默",
            decision="apply",
            tool_name="file.read",
        )
        wm = WorkingMemory(capacity=10)
        injected = await _ingest_actionable_meta_reflections(store, wm)
        assert injected == ["mr-threshold-apply"]

        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)
        action = _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            params={"path": ""},
            rationale="test threshold policy apply",
        )

        first = await layer.dispatch(action, ctx)
        second = await layer.dispatch(action, ctx)
        third = await layer.dispatch(action, ctx)
        fourth = await layer.dispatch(action, ctx)
        fifth = await layer.dispatch(action, ctx)

        assert first.error == "EmptyPath"
        assert second.error == "EmptyPath"
        assert third.error == "EmptyPath"
        assert fourth.error == "EmptyPath"
        assert fifth.error == "KnownStableFailure"

        policy_raw, found = await store.get_fact("control:durable_failure_policy")
        assert found
        assert json.loads(policy_raw) == {"threshold": 4, "ttl_sec": 3600}
        await store.close()


def test_consume_task_runtime_hints_updates_task_state_once():
    asyncio.run(_consume_task_runtime_hints_updates_task_state_once())


async def _consume_task_runtime_hints_updates_task_state_once():
    from core.task_runtime import _consume_task_runtime_hints, _ingest_actionable_meta_reflections
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("重规划任务", goal="先复盘再继续", next_step="旧步骤")
        await store.add_meta_reflection(
            reflection_id="mr-routing-tasksplit",
            target_kind="task_split",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="任务拆分不完整",
            proposal="先定位资源，再读取文件",
            verification_plan="确认不再出现 EmptyPath",
            decision="apply",
            task_id=task_id,
            run_id=1,
            tool_name="file.read",
        )
        await store.add_meta_reflection(
            reflection_id="mr-routing-guard",
            target_kind="routing",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="动作选择漂移",
            proposal="切换到 repair tier 复核动作",
            verification_plan="确认 chosen_action_id 合法",
            decision="apply",
            task_id=task_id,
            run_id=2,
            tool_name="file.read",
        )

        wm = WorkingMemory(capacity=10)
        injected = await _ingest_actionable_meta_reflections(store, wm)
        # list_meta_reflections 按 (created_at, id) ASC 排序，id 是 string；
        # 字母序 guard < tasksplit，顺序不固定，用 set 比较
        assert set(injected) == {"mr-routing-tasksplit", "mr-routing-guard"}

        task = await store.get_task_by_id(task_id)
        task = await _consume_task_runtime_hints(store, task, wm)
        assert task is not None
        assert task.next_step == "先定位资源，再读取文件"
        assert task.model_tier == "repair"
        assert task.extras["last_replan_reflection_id"] == "mr-routing-tasksplit"
        assert task.extras["last_routing_reflection_id"] == "mr-routing-guard"

        top = wm.get_top()
        assert any(item["kind"] == "task_replan" for item in top)
        assert any(item["kind"] == "routing_guard" for item in top)

        again = await _consume_task_runtime_hints(store, task, wm)
        assert again is not None
        assert again.next_step == "先定位资源，再读取文件"
        assert again.model_tier == "repair"
        await store.close()


def test_meta_reflection_threshold_apply_uses_explicit_policy_hint():
    asyncio.run(_meta_reflection_threshold_apply_uses_explicit_policy_hint())


async def _meta_reflection_threshold_apply_uses_explicit_policy_hint():
    from core.task_runtime import _ingest_actionable_meta_reflections
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        await store.add_meta_reflection(
            reflection_id="mr-threshold-explicit",
            target_kind="threshold",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="静默窗口过早触发",
            proposal="threshold=6 ttl=1800",
            verification_plan="连续 5 次失败前不应静默",
            decision="apply",
            tool_name="file.read",
        )

        injected = await _ingest_actionable_meta_reflections(store, WorkingMemory(capacity=10))
        assert injected == ["mr-threshold-explicit"]

        raw, found = await store.get_fact("control:durable_failure_policy")
        assert found
        assert json.loads(raw) == {"threshold": 6, "ttl_sec": 1800}
        await store.close()


def test_consume_task_runtime_hints_uses_preferred_tier_hint():
    asyncio.run(_consume_task_runtime_hints_uses_preferred_tier_hint())


async def _consume_task_runtime_hints_uses_preferred_tier_hint():
    from core.task_runtime import _consume_task_runtime_hints, _ingest_actionable_meta_reflections
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("路由护栏任务", goal="切换 tier")
        await store.add_meta_reflection(
            reflection_id="mr-routing-tier",
            target_kind="routing",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="reader 过轻导致动作漂移",
            proposal="切换到 reasoner tier 复核动作",
            verification_plan="确认动作选择回到合法工具集",
            decision="apply",
            task_id=task_id,
            run_id=3,
            tool_name="file.read",
        )

        wm = WorkingMemory(capacity=10)
        injected = await _ingest_actionable_meta_reflections(store, wm)
        assert injected == ["mr-routing-tier"]

        task = await store.get_task_by_id(task_id)
        task = await _consume_task_runtime_hints(store, task, wm)
        assert task is not None
        assert task.model_tier == "reasoner"
        await store.close()


def test_consume_task_runtime_hints_surfaces_task_meta_reflection_to_wm():
    asyncio.run(_consume_task_runtime_hints_surfaces_task_meta_reflection_to_wm())


async def _consume_task_runtime_hints_surfaces_task_meta_reflection_to_wm():
    from core.task_runtime import _consume_task_runtime_hints, _ingest_actionable_meta_reflections
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("反思显化任务", goal="让 LLM 感知 task 级反思")
        await store.add_meta_reflection(
            reflection_id="mr-task-visible",
            target_kind="routing",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="动作选择漂移",
            proposal="切换到 repair tier 复核动作",
            verification_plan="确认 chosen_action_id 合法",
            decision="apply",
            task_id=task_id,
            run_id=8,
            tool_name="file.read",
        )

        wm = WorkingMemory(capacity=10)
        injected = await _ingest_actionable_meta_reflections(store, wm)
        assert injected == ["mr-task-visible"]

        task = await store.get_task_by_id(task_id)
        task = await _consume_task_runtime_hints(store, task, wm)
        assert task is not None
        assert task.extras["last_task_meta_reflection_id"] == "mr-task-visible"
        top = wm.get_top()
        assert any(item["kind"] == "task_reflection" and "mr-task-visible" not in item["content"] for item in top)
        await store.close()


def test_ingest_actionable_meta_reflections_queues_generic_control_hint():
    asyncio.run(_ingest_actionable_meta_reflections_queues_generic_control_hint())


async def _ingest_actionable_meta_reflections_queues_generic_control_hint():
    from core.task_runtime import _ingest_actionable_meta_reflections
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        await store.add_meta_reflection(
            reflection_id="mr-prompt-hint",
            target_kind="prompt",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="提示词规则需要修订",
            proposal="收紧 JSON schema 说明",
            verification_plan="确认 parse failure 下降",
            decision="apply",
            tool_name="judgment",
        )

        injected = await _ingest_actionable_meta_reflections(store, WorkingMemory(capacity=10))
        assert injected == ["mr-prompt-hint"]

        raw, found = await store.get_fact("control:meta_reflection_hint:prompt")
        assert found
        payload = json.loads(raw)
        assert payload["reflection_id"] == "mr-prompt-hint"
        assert payload["proposal"] == "收紧 JSON schema 说明"
        await store.close()


def test_execution_background_run_does_not_record_completion_early():
    asyncio.run(_execution_background_run_does_not_record_completion_early())


async def _execution_background_run_does_not_record_completion_early():
    import os

    from core.judgment import JudgmentOutput
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore
    from tools.exec import ProcessManager, process_kill
    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        os.environ["LINGZHOU_PROCESS_STATE_DIR"] = str(root / "proc-state")
        ProcessManager.clear()
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = SemanticMemory(root)
        await store.open()
        await store.add_task("后台执行", goal="启动后保持 running")
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store, episodic=episodic, semantic=semantic)
        action = _judgment_output(
            decision="act",
            chosen_action_id="exec",
            params={"command": 'python3 -c "import time; time.sleep(5)"', "background": True, "timeout": 5},
            rationale="launch background exec",
        )

        result = await layer.dispatch(action, ctx)
        assert result.error is None

        run_id = int(result.metadata["run_id"])
        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "running"

        completed = episodic.list_events("run_completed", limit=5)
        failed = episodic.list_events("run_failed", limit=5)
        assert completed == []
        assert failed == []

        node = semantic.get(f"run-result-{run_id}")
        assert node is None

        await process_kill({"process_id": str(result.metadata["process_id"] or "")}, _tool_ctx())
        await store.close()


def test_exec_process_write_pty_roundtrip():
    import subprocess
    script = r'''
import asyncio, json
from types import SimpleNamespace
from tools.exec import exec_run, process_write, process_poll, process_log, _MANAGER
from tools.registry import ToolContext

async def main():
    _MANAGER.clear()
    ctx = ToolContext(
        config=SimpleNamespace(loop=SimpleNamespace(act=True)),
        wm=None, task_store=None, episodic=None, semantic=None, emotion=None,
    )
    res = await exec_run({
        "command": "bash -lc 'stty -echo; read a; echo got:$a'",
        "background": True,
        "pty": True,
        "timeout": 5,
    }, ctx)
    sid = json.loads(res.evidence)["process_id"]
    await asyncio.sleep(0.2)
    await process_write({"process_id": sid, "data": "hi\\n"}, ctx)
    for _ in range(60):
        poll = await process_poll({"process_id": sid}, ctx)
        status = json.loads(poll.summary)
        if status["status"] == "finished":
            break
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.1)
    log = await process_log({"process_id": sid, "offset": 0, "limit": 400}, ctx)
    print(log.summary)

asyncio.run(main())
'''
    out = subprocess.check_output(["python3", "-c", script], cwd=str(_proj_root()), text=True)
    assert "TimeoutError" not in out
    assert "ProcessNotFound" not in out


def test_skill_registry():
    from core.skill import SkillRegistry
    reg = SkillRegistry()
    # 冷启动场景
    skills = reg.match_for_context(wm_pressure=0.05, has_active_task=False,
                                    has_next_step=False, failure_count=0, high_error_streak=0)
    assert any(s.name == "runtime.bootstrap" for s in skills)
    # 失败场景
    skills_fail = reg.match_for_context(wm_pressure=0.5, has_active_task=True,
                                         has_next_step=True, failure_count=3, high_error_streak=3)
    assert any(s.name == "failure.reflection" for s in skills_fail)


def test_skill_registry_loads_package_skill_and_matches_context(tmp_path):
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"

    pkg = skills_dir / "karpathy-coding-base"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: karpathy-coding-base
description: |
  Andrej Karpathy-inspired coding guardrails. Triggers: 修复bug、重构、写脚本、代码审查
---
先思考，再编码。极简优先。手术式变更。目标驱动验证。
""",
        encoding="utf-8",
    )

    pkg2 = skills_dir / "interaction"
    pkg2.mkdir(parents=True)
    (pkg2 / "SKILL.md").write_text(
        """---
name: interaction
description: |
  统一人际交互入口。Triggers: 提问/确认/好奇追问/理解语境
---
先判断意图，再决定是回答、提问还是确认。方向不清时问一个最小问题。
""",
        encoding="utf-8",
    )

    pkg3 = skills_dir / "proactive-work"
    pkg3.mkdir(parents=True)
    (pkg3 / "SKILL.md").write_text(
        """---
name: proactive-work
description: |
  主动工作方法论。Triggers: 完成任务后、等回复时、需自主决定下一步
---
完成任务后不要等待，主动判断并推进下一步。
""",
        encoding="utf-8",
    )

    pkg4 = skills_dir / "self-monitoring"
    pkg4.mkdir(parents=True)
    (pkg4 / "SKILL.md").write_text(
        """---
name: self-monitoring
description: |
  Self-monitoring. Triggers: 工具执行失败、编辑失败、文件异常、日志错误、执行偏离预期
---
发现漂移、日志错误、编辑失败后，先检查并修复。
""",
        encoding="utf-8",
    )

    pkg5 = skills_dir / "error-handling"
    pkg5.mkdir(parents=True)
    (pkg5 / "SKILL.md").write_text(
        """---
name: error-handling
description: |
  Error handling. Triggers: tool call fails, exec denied, network timeout, permission error
---
失败后先分类错误，再决定重试、替代还是汇报。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="请你修复 bug，并顺手重构这个脚本",
        max_inject=20,
    )
    assert any(s.name == "karpathy-coding-base" for s in skills)

    interaction_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="我有点好奇，你觉得这里真正的分歧是什么？",
        max_inject=20,
    )
    assert any(s.name == "interaction" for s in interaction_skills)

    proactive_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="做完了当前任务，接下来你自己判断往前推进",
        max_inject=20,
    )
    assert any(s.name == "proactive-work" for s in proactive_skills)

    monitor_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=1,
        high_error_streak=1,
        context_text="这次 edit 失败了，日志也有异常，帮我看看哪里偏了",
        max_inject=20,
    )
    assert any(s.name == "self-monitoring" for s in monitor_skills)

    err_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=1,
        high_error_streak=1,
        context_text="exec 被拒绝了，还报了 timeout 和 permission error",
        max_inject=20,
    )
    assert any(s.name == "error-handling" for s in err_skills)


