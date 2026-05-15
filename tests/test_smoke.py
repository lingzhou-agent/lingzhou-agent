"""快速验证测试，不依赖 LLM。"""
import asyncio
import builtins
import io
import json
import logging
import math
import os
import tempfile
import time
from datetime import datetime, UTC, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import aiosqlite
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _proj_root() -> Path:
    return Path(__file__).parent.parent


def _test_config(
    *,
    act: bool = True,
    debug: bool = False,
    workspace_dir: str = "",
    shell_timeout: int = 5,
    shell_max_output_chars: int = 200,
) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            loop=SimpleNamespace(act=act, debug=debug, workspace_dir=workspace_dir),
            thresholds=SimpleNamespace(
                shell_timeout=shell_timeout,
                shell_max_output_chars=shell_max_output_chars,
            ),
        ),
    )


def _tool_ctx(
    *,
    act: bool = True,
    debug: bool = False,
    workspace_dir: str = "",
    shell_timeout: int = 5,
    shell_max_output_chars: int = 200,
    wm: Any = None,
    task_store: Any = None,
    episodic: Any = None,
    semantic: Any = None,
    emotion: Any = None,
):
    from tools.registry import ToolContext

    return cast(Any, ToolContext)(
        config=cast(
            Any,
            _test_config(
                act=act,
                debug=debug,
                workspace_dir=workspace_dir,
                shell_timeout=shell_timeout,
                shell_max_output_chars=shell_max_output_chars,
            ),
        ),
        wm=cast(Any, wm),
        task_store=cast(Any, task_store),
        episodic=cast(Any, episodic),
        semantic=cast(Any, semantic),
        emotion=cast(Any, emotion),
    )


def _execution_layer(reg, *, debug: bool = False):
    from core.execution import ExecutionLayer

    return ExecutionLayer(reg, cast(Any, SimpleNamespace(loop=SimpleNamespace(debug=debug))))


def _judgment_output(**kwargs: Any) -> Any:
    from core.judgment import JudgmentOutput

    return cast(Any, JudgmentOutput)(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 基础模块
# ══════════════════════════════════════════════════════════════════════════════

def test_working_memory():
    from memory.working import WorkingMemory, WMItem
    wm = WorkingMemory(capacity=5)
    for i in range(7):
        wm.add(WMItem(kind="test", content=f"item {i}", priority=i / 10))
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

    from core.loop import _refresh_running_runs
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

        updates = await _refresh_running_runs(store)
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

    from core.loop import _refresh_running_runs
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

        updates = await _refresh_running_runs(store)
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

    from core.loop import _refresh_running_runs
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

        updates = await _refresh_running_runs(store)
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
    from core.loop import _refresh_running_runs
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

        first = await _refresh_running_runs(store, episodic=episodic, semantic=semantic)
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
        second = await _refresh_running_runs(store, episodic=episodic, semantic=semantic)
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
    from core.loop import _refresh_running_runs
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

        updates = await _refresh_running_runs(store, episodic=episodic, semantic=semantic)
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

    from core.loop import _refresh_running_runs
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

        updates = await _refresh_running_runs(store, episodic=episodic, semantic=semantic)
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
    from tools.image import _collect_image_sources, _image_part_from_source

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


async def _file_list_and_memory_search():
    from tools.file import file_list, file_read
    from tools.memory_ops import memory_search, memory_add_semantic
    from memory.semantic import SemanticMemory
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
        found = await memory_search({'query': 'bug'}, ctx)
        assert 'bug fix note' in found.summary


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
    sid = json.loads(res.evidence)["session_id"]
    await process_write({"session_id": sid, "data": "hello\\n", "eof": True}, ctx)

    for _ in range(40):
        poll = await process_poll({"session_id": sid}, ctx)
        status = json.loads(poll.summary)
        if status["status"] == "finished":
            break
        await asyncio.sleep(0.05)

    log = await process_log({"session_id": sid, "offset": 0, "limit": 200}, ctx)
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
    sid = json.loads(res.evidence)["session_id"]

    timed_out = False
    for _ in range(60):
        poll = await process_poll({"session_id": sid}, ctx)
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
    from core.loop import _ingest_actionable_meta_reflections
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
    from core.loop import _ingest_actionable_meta_reflections
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
    from core.loop import _consume_task_runtime_hints, _ingest_actionable_meta_reflections
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
        assert injected == ["mr-routing-tasksplit", "mr-routing-guard"]

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
    from core.loop import _ingest_actionable_meta_reflections
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
    from core.loop import _consume_task_runtime_hints, _ingest_actionable_meta_reflections
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
    from core.loop import _consume_task_runtime_hints, _ingest_actionable_meta_reflections
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
    from core.loop import _ingest_actionable_meta_reflections
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

        await process_kill({"session_id": str(result.metadata["session_id"] or "")}, _tool_ctx())
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
    sid = json.loads(res.evidence)["session_id"]
    await asyncio.sleep(0.2)
    await process_write({"session_id": sid, "data": "hi\\n"}, ctx)
    for _ in range(60):
        poll = await process_poll({"session_id": sid}, ctx)
        status = json.loads(poll.summary)
        if status["status"] == "finished":
            break
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.1)
    log = await process_log({"session_id": sid, "offset": 0, "limit": 400}, ctx)
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
        max_inject=5,
    )
    assert any(s.name == "karpathy-coding-base" for s in skills)

    interaction_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="我有点好奇，你觉得这里真正的分歧是什么？",
        max_inject=3,
    )
    assert any(s.name == "interaction" for s in interaction_skills)

    proactive_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="做完了当前任务，接下来你自己判断往前推进",
        max_inject=3,
    )
    assert any(s.name == "proactive-work" for s in proactive_skills)

    monitor_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=1,
        high_error_streak=1,
        context_text="这次 edit 失败了，日志也有异常，帮我看看哪里偏了",
        max_inject=3,
    )
    assert any(s.name == "self-monitoring" for s in monitor_skills)

    err_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=1,
        high_error_streak=1,
        context_text="exec 被拒绝了，还报了 timeout 和 permission error",
        max_inject=3,
    )
    assert any(s.name == "error-handling" for s in err_skills)


# ══════════════════════════════════════════════════════════════════════════════
# TaskStore — JSON-first
# ══════════════════════════════════════════════════════════════════════════════

def test_task_store_basic():
    asyncio.run(_task_store_basic())

async def _task_store_basic():
    from memory.task_store import TaskStore
    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "test.db")
        await store.open()

        tid = await store.add_task("任务A", goal="目标", priority="high", source="external")
        t = await store.get_task_by_id(tid)
        assert t is not None
        assert t.goal == "目标"
        assert t.source == "external"
        assert t.next_step == ""

        await store.update_status(tid, "in_progress", "步骤1")
        t2 = await store.get_task_by_id(tid)
        assert t2 is not None
        assert t2.status == "in_progress"
        assert t2.next_step == "步骤1"

        # 扩展字段（无需 ALTER TABLE）
        await store.update_task_data(
            tid,
            {"tags": ["ai"], "score": 99, "model_tier": "reader", "current_step": "检查任务状态", "next_step": "不应覆盖"},
        )
        t3 = await store.get_task_by_id(tid)
        assert t3 is not None
        assert t3.extras["score"] == 99
        assert t3.model_tier == "reader"
        assert t3.current_step == "检查任务状态"
        assert t3.next_step == "步骤1"  # 原有字段未被覆盖

        await store.sync_task_progress(tid, current_step="步骤1", next_step="步骤2")
        t4 = await store.get_task_by_id(tid)
        assert t4 is not None
        assert t4.current_step == "步骤1"
        assert t4.next_step == "步骤2"

        await store.sync_task_progress(tid, next_step="")
        t5 = await store.get_task_by_id(tid)
        assert t5 is not None
        assert t5.next_step == ""

        await store.update_task_result(tid, {"summary": "first", "score": 1})
        await store.update_task_result(tid, {"last_run_status": "succeeded"})
        t6 = await store.get_task_by_id(tid)
        assert t6 is not None
        assert t6.result_json["summary"] == "first"
        assert t6.result_json["score"] == 1
        assert t6.result_json["last_run_status"] == "succeeded"

        await store.mark_waiting(tid, wait_kind="process", wait_key="exec-1")
        await store.resume_task(tid, result_json={"resumed_via": "task.resume"})
        t7 = await store.get_task_by_id(tid)
        assert t7 is not None
        assert t7.result_json["summary"] == "first"
        assert t7.result_json["last_run_status"] == "succeeded"
        assert t7.result_json["resumed_via"] == "task.resume"

        # 失败记录
        await store.record_failure("tool_error", "报错", context="ctx", task_id=str(tid))
        await store.record_failure("provider_error", "网络", task_id="")
        failures = await store.list_failures_for_task(str(tid))
        assert len(failures) == 2
        assert failures[0].summary == "报错"

        # count_failures_by_kind
        assert await store.count_failures_by_kind("tool_error") == 1

        # facts
        import json
        await store.set_fact("soul:ethos_baseline", json.dumps({"truth": 0.85}))
        v, found = await store.get_fact("soul:ethos_baseline")
        assert found and json.loads(v)["truth"] == 0.85

        # enqueue_if_absent 去重
        a1 = await store.enqueue_if_absent("dup task")
        a2 = await store.enqueue_if_absent("dup task")
        assert a1 and not a2

        await store.close()


def test_task_store_run_lifecycle():
    asyncio.run(_task_store_run_lifecycle())


async def _task_store_run_lifecycle():
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runs.db")
        await store.open()
        task_id = await store.add_task("任务A", goal="目标")
        run_id = await store.add_run(
            task_id=task_id,
            run_type="tool_chain",
            worker_type="tool-chain-worker",
            input_json={"tool": "file.read"},
            tool_name="file.read",
            model_tier="reader",
            progress="queued",
        )

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "running"
        assert run.tool_name == "file.read"
        assert run.model_tier == "reader"
        assert run.progress == "queued"

        await store.update_run(
            run_id,
            status="succeeded",
            output_json={"summary": "ok"},
            log_text="ok",
            progress="done",
        )
        finished = await store.get_run_by_id(run_id)
        assert finished is not None
        assert finished.status == "succeeded"
        assert finished.output_json["summary"] == "ok"
        assert finished.progress == "done"
        assert finished.completed_at

        runs = await store.list_runs(task_id=task_id)
        assert len(runs) == 1

        await store.close()


def test_task_update_can_clear_runtime_fields():
    asyncio.run(_task_update_can_clear_runtime_fields())


async def _task_update_can_clear_runtime_fields():
    from memory.task_store import TaskStore
    from tools.task_ops import task_update

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "tasks.db")
        await store.open()
        task_id = await store.add_task(
            "清理任务状态",
            goal="验证 task.update 可以清空运行提示字段",
            next_step="旧下一步",
            current_step="旧当前步骤",
            model_tier="reader",
        )

        ctx = _tool_ctx(task_store=store)
        res = await task_update(
            {
                "task_id": task_id,
                "status": "in_progress",
                "next_step": "",
                "current_step": "",
                "model_tier": "",
            },
            ctx,
        )

        assert res.error is None
        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "in_progress"
        assert task.next_step == ""
        assert task.current_step == ""
        assert task.model_tier == ""

        await store.close()


def test_task_wait_resume_can_clear_runtime_fields():
    asyncio.run(_task_wait_resume_can_clear_runtime_fields())


async def _task_wait_resume_can_clear_runtime_fields():
    from memory.task_store import TaskStore
    from tools.task_ops import task_resume, task_wait

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "wait-resume.db")
        await store.open()
        task_id = await store.add_task(
            "等待恢复任务",
            goal="验证 waiting/resume 可以显式清空步骤字段",
            next_step="旧下一步",
            current_step="旧当前步骤",
        )

        ctx = _tool_ctx(task_store=store)
        wait_res = await task_wait(
            {
                "task_id": task_id,
                "wait_kind": "process",
                "wait_key": "exec-1",
                "current_step": "",
                "next_step": "",
            },
            ctx,
        )
        assert wait_res.error is None

        waited = await store.get_task_by_id(task_id)
        assert waited is not None
        assert waited.status == "waiting"
        assert waited.current_step == ""
        assert waited.next_step == ""

        resume_res = await task_resume(
            {
                "task_id": task_id,
                "status": "ready",
                "current_step": "",
                "next_step": "",
            },
            ctx,
        )
        assert resume_res.error is None

        resumed = await store.get_task_by_id(task_id)
        assert resumed is not None
        assert resumed.status == "ready"
        assert resumed.current_step == ""
        assert resumed.next_step == ""

        await store.close()


def test_chat_messages_are_sanitized_on_write():
    asyncio.run(_chat_messages_are_sanitized_on_write())


def test_task_wait_allows_external_wait_without_wait_key():
    asyncio.run(_task_wait_allows_external_wait_without_wait_key())


async def _task_wait_allows_external_wait_without_wait_key():
    from memory.task_store import TaskStore
    from tools.task_ops import task_wait

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "wait-guard.db")
        await store.open()
        task_id = await store.add_task("等待外部路径", goal="验证 task.wait 不会强制要求 wait_key")

        ctx = _tool_ctx(task_store=store)
        wait_res = await task_wait(
            {
                "task_id": task_id,
                "wait_kind": "external",
            },
            ctx,
        )

        assert wait_res.skipped is False
        assert wait_res.error is None

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "waiting"
        assert task.wait_kind == "external"
        assert task.wait_key == ""

        await store.close()


def test_task_wait_rejects_unknown_wait_kind():
    asyncio.run(_task_wait_rejects_unknown_wait_kind())


async def _task_wait_rejects_unknown_wait_kind():
    from memory.task_store import TaskStore
    from tools.task_ops import task_wait

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "wait-kind.db")
        await store.open()
        task_id = await store.add_task("非法等待类型", goal="验证 task.wait 只接受受支持的等待类型")

        ctx = _tool_ctx(task_store=store)
        wait_res = await task_wait(
            {
                "task_id": task_id,
                "wait_kind": "missing-evidence",
                "wait_key": "source-path",
            },
            ctx,
        )

        assert wait_res.skipped is True
        assert "不支持的 wait_kind" in wait_res.summary

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "pending"

        await store.close()


async def _chat_messages_are_sanitized_on_write():
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "chat.db")
        await store.open()
        await store.add_chat_message("user", "\x1b[31mhi\x1b[0m\ufeff\u200b\ufffd\x07there\r\n")
        await store.add_chat_message("user", "删 掉中文 后 就会 多 出 空格")
        msgs = await store.get_chat_messages_since(0)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "hithere"
        assert msgs[1]["content"] == "删掉中文后就会多出空格"
        await store.close()


def test_task_store_migration():
    asyncio.run(_task_store_migration())

async def _task_store_migration():
    """旧列式 schema → JSON-first 自动迁移。"""
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "old.db"
        async with aiosqlite.connect(str(db_path)) as db:
            await db.executescript("""
                CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    goal TEXT DEFAULT '',
                    priority TEXT DEFAULT 'normal',
                    status TEXT DEFAULT 'pending',
                    source TEXT DEFAULT 'external',
                    next_step TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    context TEXT DEFAULT '',
                    task_id TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE facts (
                    key TEXT PRIMARY KEY,
                    value TEXT DEFAULT '',
                    scope TEXT DEFAULT 'general',
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                INSERT INTO tasks (title, goal, source, next_step)
                    VALUES ('旧任务', '旧目标', 'external', '旧步骤');
                INSERT INTO failures (kind, summary, context, task_id)
                    VALUES ('old_error', '旧摘要', '旧上下文', '1');
            """)
            await db.commit()

        store = TaskStore(db_path)
        await store.open()

        tasks = await store.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].title == "旧任务"
        assert tasks[0].goal == "旧目标"
        assert tasks[0].source == "external"
        assert tasks[0].next_step == "旧步骤"

        failures = await store.list_failures()
        assert len(failures) == 1
        assert failures[0].summary == "旧摘要"
        assert failures[0].context == "旧上下文"
        assert failures[0].task_id == "1"

        await store.close()


# ══════════════════════════════════════════════════════════════════════════════
# SemanticMemory — Ebbinghaus 衰减
# ══════════════════════════════════════════════════════════════════════════════

def test_semantic_ebbinghaus():
    from memory.semantic import SemanticMemory, MemoryNode, effective_activation

    now_ts = datetime.now(UTC).isoformat()
    old_ts = (datetime.now(UTC) - timedelta(days=7)).isoformat()

    n_new = MemoryNode(id="new", kind="fact", title="python reload",
                       body="importlib", activation=0.8, created_at=now_ts)
    n_old = MemoryNode(id="old", kind="fact", title="python reload",
                       body="importlib", activation=0.8, created_at=old_ts)

    eff_new = effective_activation(n_new, 0.1)
    eff_old = effective_activation(n_old, 0.1)
    expected = 0.8 * math.exp(-0.1 * 7)

    assert eff_new > eff_old
    assert abs(eff_old - expected) < 0.01
    assert effective_activation(n_old, 0.0) == 0.8  # λ=0 不衰减

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.1)
        sm.upsert(n_new)
        sm.upsert(n_old)
        results = sm.retrieve("python reload importlib", top_k=2)
        assert results[0]["id"] == "new"  # 新节点排前


# ══════════════════════════════════════════════════════════════════════════════
# EpisodicMemory — events.jsonl 轮转
# ══════════════════════════════════════════════════════════════════════════════

def test_episodic_rotation():
    from memory.episodic import EpisodicMemory

    with tempfile.TemporaryDirectory() as d:
        ep = EpisodicMemory(Path(d), max_events=10)
        for i in range(20):
            ep.record_event("perception", {"seq": i})

        events = ep.list_events("perception", limit=100)
        assert len(events) <= 10
        assert events[-1]["seq"] == 19   # 最新
        assert events[0]["seq"] == 10    # 保留最新 10 条


def test_episodic_no_rotation():
    """max_events=0 时不做任何裁剪。"""
    from memory.episodic import EpisodicMemory

    with tempfile.TemporaryDirectory() as d:
        ep = EpisodicMemory(Path(d), max_events=0)
        for i in range(20):
            ep.record_event("perception", {"seq": i})
        events = ep.list_events("perception", limit=100)
        assert len(events) == 20


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap 注入
# ══════════════════════════════════════════════════════════════════════════════

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
                wm.add(WMItem(kind="bootstrap_identity",
                               content=f"[{fname}]\n{content[:400]}", priority=1.0))

        items = wm.get_top(10)
        assert sum(1 for i in items if i["kind"] == "bootstrap_identity") == 2


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
            wm_top = loop._wm.get_top(10)
            assert any("好奇心信号" in item["content"] for item in wm_top)
        finally:
            await loop.task_store.close()
            await loop.provider.close()


def test_auth_store_profile_roundtrip(tmp_path):
    from auth_store import load_auth_profiles, set_token_profile

    path = tmp_path / "auth-profiles.json"
    set_token_profile(profile_id="copilot:default", provider="copilot", token="tok-123456", path=path)
    data = load_auth_profiles(path)
    assert data["version"] == 1
    assert data["profiles"]["copilot:default"]["provider"] == "copilot"
    assert data["profiles"]["copilot:default"]["token"] == "tok-123456"


def test_copilot_token_resolution_prefers_auth_profile(monkeypatch, tmp_path):
    from auth_store import resolve_copilot_token, set_token_profile, save_legacy_credentials

    monkeypatch.setenv("GH_TOKEN", "env-gh-token")
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    set_token_profile(profile_id="copilot:default", provider="copilot", token="profile-token", path=tmp_path / "auth-profiles.json")
    save_legacy_credentials({"GITHUB_TOKEN": "legacy-token"}, path=tmp_path / "credentials.json")

    import auth_store as auth_mod
    monkeypatch.setattr(auth_mod, "AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    monkeypatch.setattr(auth_mod, "LEGACY_CREDENTIALS_PATH", tmp_path / "credentials.json")

    resolved = resolve_copilot_token()
    assert resolved is not None
    assert resolved.token == "profile-token"
    assert resolved.source == "auth-profile"


def test_github_device_client_id_prefers_env(monkeypatch, tmp_path):
    import json
    import auth_store as auth_mod

    state_file = tmp_path / "github-device.json"
    state_file.write_text(json.dumps({"client_id": "Iv1.file-client"}), encoding="utf-8")

    monkeypatch.setattr(auth_mod, "GITHUB_DEVICE_AUTH_PATH", state_file)
    monkeypatch.setenv("LINGZHOU_GITHUB_CLIENT_ID", "Iv1.env-client")

    assert auth_mod.load_github_device_client_id() == "Iv1.env-client"


def test_copilot_gpt5_uses_max_completion_tokens():
    from provider.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
    provider._provider_mode = "copilot"
    provider._model = "gpt-5.4"

    payload = {}
    provider._inject_completion_limits(payload)

    assert payload["max_completion_tokens"] == 65536  # gpt-5.4 在 models.json 中的 max_tokens


def test_copilot_base_url_derives_from_proxy_ep():
    from provider.openai_compat import _derive_copilot_api_base_url_from_token

    token = "ghu_xxx; proxy-ep=proxy.business.githubcopilot.com; tid=abc"
    assert _derive_copilot_api_base_url_from_token(token) == "https://api.business.githubcopilot.com"


def test_copilot_normalize_base_url_uses_openclaw_default():
    from provider.openai_compat import _normalize_copilot_api_base_url, DEFAULT_COPILOT_API_BASE_URL

    assert _normalize_copilot_api_base_url("") == DEFAULT_COPILOT_API_BASE_URL
    assert _normalize_copilot_api_base_url("https://api.githubcopilot.com") == DEFAULT_COPILOT_API_BASE_URL


def test_login_copilot_help_is_registered():
    from typer.testing import CliRunner
    from lingzhou import app

    runner = CliRunner()
    result = runner.invoke(app, ["auth", "login-copilot", "--help"])
    assert result.exit_code == 0
    assert "专用 Copilot 登录命令" in result.stdout
    assert "--method" in result.stdout
    assert "--oauth-client-id" in result.stdout


# ══════════════════════════════════════════════════════════════════════════════
# SemanticMemory — 多锚点情境召回（ACT-R 收敛激活）
# ══════════════════════════════════════════════════════════════════════════════

def test_semantic_multi_anchor_convergence_bonus():
    """多锚点命中同一节点时 convergence_bonus 使其排名高于单锚点命中节点。

    设计原理：两节点在主锚点 "importlib" 上得分相近，但 node_ab 的 body
    同时命中第二锚点 "热加载 reload"，因此多锚点命中使其 final_score 更高。
    """
    from memory.semantic import SemanticMemory, MemoryNode

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.0)

        # node_ab: title 含主锚点 "importlib"，body 含第二锚点 "热加载 reload"
        node_ab = MemoryNode(id="ab", kind="fact",
                             title="importlib",
                             body="热加载 reload 模块替换",
                             activation=0.0)
        # node_a: 同样含主锚点 "importlib"，body 不含第二锚点
        node_a = MemoryNode(id="a", kind="fact",
                            title="importlib",
                            body="模块导入",
                            activation=0.0)
        sm.upsert(node_ab)
        sm.upsert(node_a)

        results = sm.retrieve_multi_anchor(
            ["importlib", "热加载 reload"],
            top_k=2,
            convergence_bonus=0.3,
        )
        ids = [r["id"] for r in results]
        # node_ab 被两个锚点命中（convergence_bonus 加分），应排在第一位
        assert ids[0] == "ab", f"期望 ab 排第一，实际顺序: {ids}"


def test_semantic_multi_anchor_empty_anchors():
    """空锚点列表应返回空结果，不崩溃。"""
    from memory.semantic import SemanticMemory, MemoryNode

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.0)
        sm.upsert(MemoryNode(id="x", kind="fact", title="test", body="body", activation=0.5))

        assert sm.retrieve_multi_anchor([]) == []
        assert sm.retrieve_multi_anchor(["", "  "]) == []


# ══════════════════════════════════════════════════════════════════════════════
# 今日新增功能验证
# ══════════════════════════════════════════════════════════════════════════════

def test_model_health_circuit_breaker_blocks_and_clears():
    """ModelHealth 断路器：标记冷却后 _is_model_available 返回 False，
    recover 后返回 True；fallback tier 在主 tier 冷却时被选中。"""
    import time
    from core.config import Config
    from core.judgment import JudgmentLayer, ModelHealth
    from tools.registry import ToolRegistry

    class _Dummy:
        async def chat(self, messages, **kw):
            return '{"decision":"wait"}'
        async def close(self):
            pass

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
    layer = JudgmentLayer(_Dummy(), ToolRegistry(), cfg)

    # 初始状态：模型可用
    assert layer._is_model_available("bailian/qwen3.6-plus") is True

    # 标记 429 错误 → 进入冷却
    layer._mark_model_failure("bailian/qwen3.6-plus", "Client error '429 Too Many Requests'")
    assert layer._is_model_available("bailian/qwen3.6-plus") is False

    # recover → 可用
    health = layer._get_health("bailian/qwen3.6-plus")
    health.cooldown_until = time.time() - 1  # 手动过期
    assert layer._is_model_available("bailian/qwen3.6-plus") is True


def test_select_tier_logic():
    """_select_tier 按 phase 和 prefer_tier 正确返回 tier。"""
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _Dummy:
        async def chat(self, messages, **kw):
            return '{"decision":"wait"}'
        async def close(self):
            pass

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
    layer = JudgmentLayer(_Dummy(), ToolRegistry(), cfg)

    # initial phase → reasoner
    assert layer._select_tier(phase="initial", user_message="hello") == "reasoner"

    # repair phase → repair
    assert layer._select_tier(phase="repair", user_message="") == "repair"

    # prefer_tier 优先
    assert layer._select_tier(phase="initial", user_message="", prefer_tier="reader") == "reader"

    # continue + reader tool + no error → reader
    tier = layer._select_tier(
        phase="continue", user_message="",
        current_action="file.read", tool_history=[],
    )
    assert tier == "reader"

    # continue + reasoner tool → reasoner
    tier2 = layer._select_tier(
        phase="continue", user_message="",
        current_action="shell.run", tool_history=[],
    )
    assert tier2 == "reasoner"


def test_prefer_tier_for_task_uses_pending_then_task_default():
    from core.loop import _prefer_tier_for_task
    from memory.task_store import Task

    task = Task(
        id=1,
        title="任务",
        status="pending",
        priority="normal",
        created_at="2026-05-15T00:00:00Z",
        model_tier="reader",
    )

    assert _prefer_tier_for_task(None, task) == "reader"
    assert _prefer_tier_for_task("repair", task) == "repair"

    task.model_tier = "invalid"
    assert _prefer_tier_for_task(None, task) is None


def test_behavior_gate_passthrough():
    """apply_execution_gate 为纯透传：决策权归 LLM，不做硬拦截。

    重复行为信号由 on_act/on_read/on_list 以 WMItem 形式注入工作记忆，
    LLM 在下一轮 judgment 时自主决定是否改变策略。
    """
    from core.behavior_tracker import BehaviorTracker
    from core.judgment import JudgmentOutput

    tracker = BehaviorTracker()

    class _Signals:
        repeat_action_count = 3
        repeat_action_tool = "shell.run"
        repeat_action_key = "ls"
        repeat_read_count = 0
        repeat_read_path = ""
        loop_probe_version = 5

    action = _judgment_output(
        decision="act",
        chosen_action_id="shell.run",
        params={"command": "ls"},
        rationale="再跑一次",
    )
    # 透传：gate 不改变决策，信号已通过 WM 注入交由 LLM 判断
    gated = tracker.apply_execution_gate(action, _Signals())
    assert gated.decision == "act", "apply_execution_gate 应透传，不强制改变决策"
    assert gated is action, "apply_execution_gate 应返回原对象（零拷贝）"

    # on_act 连续相同行为时应生成 WMItem 信号
    items = []
    for _ in range(3):
        items = tracker.on_act("shell.run", "ls", task_id="t1")
    assert any("行为信号" in i.content for i in items), "连续 3 次相同行为应注入 WM 行为信号"

    # on_act 连续不同命令（key_param 不同）不应触发 streak
    tracker2 = BehaviorTracker()
    tracker2.on_act("shell.run", "cat USER.md", task_id="t2")
    tracker2.on_act("shell.run", "cat SOUL.md", task_id="t2")
    items2 = tracker2.on_act("shell.run", "sed -n '1p' TOOLS.md", task_id="t2")
    assert not any("行为信号" in i.content for i in items2), (
        "不同 shell.run 命令不应触发 streak（key_param 已区分命令内容）"
    )


def test_behavior_list_result_aware():
    """file.list 应按“结果是否相同”而不是仅按路径判定重复。"""
    from core.behavior_tracker import BehaviorTracker

    tracker = BehaviorTracker()

    # 同一路径，但目录结果不同：不应触发重复警告
    for _ in range(3):
        tracker.on_act("file.list", "/root", task_id="t-list")
    items = tracker.on_list("/root", "a.txt\n")
    items = tracker.on_list("/root", "a.txt\nb.txt\n")
    items = tracker.on_list("/root", "a.txt\nb.txt\nc.txt\n")
    assert not any("行为信号" in i.content for i in items), "同路径但结果变化，不应判定为无效重复"

    # 同一路径且结果相同：才触发重复警告
    tracker2 = BehaviorTracker()
    for _ in range(3):
        tracker2.on_act("file.list", "/root", task_id="t-list-2")
    same = []
    for _ in range(3):
        same = tracker2.on_list("/root", "a.txt\nb.txt\n")
    assert any("行为信号" in i.content for i in same), "同路径且结果相同，才应触发 file.list 重复信号"


def test_behavior_explore_awareness_requires_task_context():
    from core.behavior_tracker import BehaviorTracker

    tracker = BehaviorTracker()
    items = []
    for _ in range(10):
        items = tracker.on_act("file.list", "/root", task_id=None)

    assert items == []


def test_next_thinking_override_is_one_shot_and_strict():
    from core.loop import _next_thinking_override

    assert _next_thinking_override({"thinking_override": "low"}) == "low"
    assert _next_thinking_override({"thinking_override": "invalid"}) is None
    assert _next_thinking_override({}) is None
    assert _next_thinking_override(None) is None


def test_resolve_thinking_override_uses_mode_defaults_and_strategy():
    from core.loop import _resolve_thinking_override

    cfg = cast(Any, SimpleNamespace(
        thinking="off",
        loop=SimpleNamespace(chat_thinking="low", autonomous_thinking="medium"),
    ))

    assert _resolve_thinking_override(cfg, user_message="hi") == "low"
    assert _resolve_thinking_override(cfg, user_message="") == "medium"
    assert _resolve_thinking_override(cfg, user_message="", pending_override="high") == "high"
    assert _resolve_thinking_override(cfg, user_message="", model_strategy={"thinking_override": "minimal"}) == "minimal"


def test_thinking_floor_respects_chat_minimum_for_user_message():
    from core.loop import _thinking_floor

    assert _thinking_floor("off", "low") == "low"
    assert _thinking_floor("minimal", "low") == "low"
    assert _thinking_floor("high", "low") == "high"
    assert _thinking_floor(None, "low") == "low"


def test_recent_runs_summary_prefers_output_and_progress():
    from core.judgment import _fmt_recent_runs
    from memory.task_store import Run

    runs = [
        Run(
            id=12,
            task_id=7,
            run_type="tool_chain",
            worker_type="tool-chain-worker",
            status="done",
            created_at="2026-05-15T14:00:00+00:00",
            tool_name="file.list",
            model_tier="reader",
            progress="列出目录，确认技能清单",
            output_json={"summary": "index.ts\npackage.json\nSKILL.md"},
        )
    ]

    text = _fmt_recent_runs(runs)
    assert "run#12 [done]" in text
    assert "tool=file.list" in text
    assert "progress=列出目录，确认技能清单" in text
    assert "summary=index.ts package.json SKILL.md" in text


def test_waiting_tasks_section_exposes_wait_reason_and_next_step():
    from core.judgment import _fmt_waiting_tasks
    from memory.task_store import Task

    tasks = [
        Task(
            id=27,
            title="等待用户补源路径",
            status="waiting",
            priority="normal",
            created_at="2026-05-15T14:00:00+00:00",
            goal="等待用户提供源路径后继续",
            next_step="拿到源路径后恢复任务并重新验证目录结构",
            wait_kind="external",
            wait_key="source-path",
        )
    ]

    text = _fmt_waiting_tasks(tasks)
    assert "task#27 [waiting] 等待用户补源路径" in text
    assert "wait=external/source-path" in text
    assert "next=拿到源路径后恢复任务并重新验证目录结构" in text


def test_model_routing_section_uses_effective_thinking():
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
            },
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "routing": {
            "reader": "bailian/qwen3.6-plus",
            "reasoner": "copilot/gpt-5.4",
        },
        "thinking": "high",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
    payload = json.loads(layer._build_model_routing_section(
        phase="initial",
        user_message="继续",
        current_action="",
        tool_history=None,
        effective_thinking="low",
    ))

    assert payload["primary_provider"]["model"] == "copilot/gpt-5.4"
    assert payload["primary_provider"]["current_thinking"] == "low"
    assert payload["reference_resolution"]["uses_primary_provider"] is True
    assert payload["reference_resolution"]["llm_available"] is True
    assert payload["available_models"][0]["current_thinking"] == "low"
    assert "tool_tier_mapping" in payload
    assert "schedule.add" in payload["tool_tier_mapping"]["reasoner"]
    assert "schedule.list" in payload["tool_tier_mapping"]["reader"]
    assert payload["implicit_next_phase_default"] is None


async def test_reference_failure_is_exposed_in_model_routing_section():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _FailingProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            raise RuntimeError("400 Bad Request")

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4-mini",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_FailingProvider(), ToolRegistry(), cfg)
    await layer._ref_resolver._llm_reason(
        "继续上次的话题",
        {"n1": {"kind": "task", "title": "旧任务", "body": "body"}},
    )
    payload = json.loads(layer._build_model_routing_section(
        phase="initial",
        user_message="继续上次的话题",
        current_action="",
        tool_history=None,
        effective_thinking="low",
    ))

    assert payload["primary_provider"]["model"] == "copilot/gpt-5.4-mini"
    assert payload["reference_resolution"]["llm_available"] is False
    assert payload["reference_resolution"]["last_error_code"] == "400"
    assert "400 Bad Request" in payload["reference_resolution"]["last_error"]


def test_model_routing_section_exposes_implicit_reader_default():
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
    payload = json.loads(layer._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="file.read",
        tool_history=[{"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "ok"}],
        effective_thinking="low",
    ))

    assert payload["implicit_next_phase_default"]["tier"] == "reader"
    assert payload["implicit_next_phase_default"]["trigger"] == "last_action=file.read"


def test_fmt_durable_failures_exposes_policy_and_muted_actions():
    from core.judgment import _fmt_durable_failures

    text = _fmt_durable_failures({
        "threshold": 3,
        "ttl_sec": 7200,
        "muted_actions": [
            {
                "tool": "file.read",
                "key": "/tmp/missing.txt",
                "reason": "missing_path",
                "count": 4,
                "remaining_sec": 119,
            }
        ],
    })

    assert "policy: threshold=3 ttl_sec=7200" in text
    assert "file.read /tmp/missing.txt" in text
    assert "reason=missing_path" in text
    assert "remaining=119s" in text


async def test_load_durable_failure_snapshot_reads_policy_and_active_mutes():
    from core.judgment import _load_durable_failure_snapshot
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            await store.set_fact(
                "control:durable_failure_policy",
                json.dumps({"threshold": 5, "ttl_sec": 1800}, ensure_ascii=False),
                scope="system",
            )
            await store.set_fact(
                "durable_failure:active",
                json.dumps({
                    "tool": "file.read",
                    "key": "/tmp/missing.txt",
                    "reason": "missing_path",
                    "count": 5,
                    "muted_until": time.time() + 90,
                }, ensure_ascii=False),
                scope="system",
            )
            await store.set_fact(
                "durable_failure:expired",
                json.dumps({
                    "tool": "file.read",
                    "key": "/tmp/old.txt",
                    "reason": "missing_path",
                    "count": 3,
                    "muted_until": time.time() - 10,
                }, ensure_ascii=False),
                scope="system",
            )

            snapshot = await _load_durable_failure_snapshot(store)
            assert snapshot["threshold"] == 5
            assert snapshot["ttl_sec"] == 1800
            assert len(snapshot["muted_actions"]) == 1
            assert snapshot["muted_actions"][0]["tool"] == "file.read"
            assert snapshot["muted_actions"][0]["key"] == "/tmp/missing.txt"
        finally:
            await store.close()


async def test_decide_continue_uses_passed_thinking_override():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        def __init__(self) -> None:
            self.last_thinking_override: str | None = None

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.last_thinking_override = thinking_override
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

    provider = _DummyProvider()
    layer = JudgmentLayer(provider, ToolRegistry(), cfg)
    layer._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "file.list", "params": {"path": "/tmp"}, "result": "ok"}],
        user_message="继续",
        prefer_tier="reasoner",
        thinking_override="low",
    )

    assert out.decision == "wait"
    assert provider.last_thinking_override == "low"
    assert layer.last_call_meta["thinking"] == "low"


def test_action_made_progress_result_aware():
    from core.judgment import JudgmentOutput
    from core.loop import _action_made_progress, _result_fingerprint
    from tools.registry import ToolResult

    list_action = _judgment_output(decision="act", chosen_action_id="file.list", params={"path": "/tmp"})
    list_res = ToolResult(summary="a.txt\nb.txt\n")
    assert _action_made_progress(list_action, list_res, prev_sig="", prev_fp="") is True
    assert _action_made_progress(
        list_action,
        list_res,
        prev_sig="file.list|/tmp",
        prev_fp=_result_fingerprint(list_res.summary),
    ) is False

    write_action = _judgment_output(decision="act", chosen_action_id="file.write", params={"path": "/tmp/x"})
    write_res = ToolResult(summary="写入成功: /tmp/x")
    assert _action_made_progress(write_action, write_res) is True

    fail_action = _judgment_output(decision="act", chosen_action_id="file.read", params={"path": "/tmp/missing"})
    fail_res = ToolResult(summary="文件不存在: /tmp/missing", error="FileNotFound")
    assert _action_made_progress(fail_action, fail_res) is False

    unknown_action = _judgment_output(decision="act", chosen_action_id="custom.unknown", params={"id": "42"})
    empty_unknown = ToolResult(summary="")
    assert _action_made_progress(unknown_action, empty_unknown) is False

    unknown_res = ToolResult(summary="no-op result")
    assert _action_made_progress(unknown_action, unknown_res, prev_sig="", prev_fp="") is True
    assert _action_made_progress(
        unknown_action,
        unknown_res,
        prev_sig="custom.unknown|42",
        prev_fp=_result_fingerprint(unknown_res.summary),
    ) is False

    unknown_with_delta = ToolResult(summary="", state_delta={"updated": True})
    assert _action_made_progress(unknown_action, unknown_with_delta) is True


def test_fallback_reply_for_user_describes_waiting_state():
    from core.loop import _fallback_reply_for_user
    from tools.registry import ToolResult
    from memory.task_store import Task

    action = _judgment_output(decision="act", chosen_action_id="task.wait", next_step="等用户补充路径后重新验证目录")
    result = ToolResult(
        summary="任务 [27] 已进入 waiting: external/source-path",
        state_delta={"task_status": "waiting", "wait_kind": "external", "wait_key": "source-path"},
    )
    task = Task(id=27, title="等待路径", status="in_progress", priority="normal", created_at="2026-05-15T14:00:00+00:00")

    reply = _fallback_reply_for_user(action, result, task)
    assert reply.startswith("状态: waiting")
    assert "waiting" in reply
    assert "external/source-path" in reply
    assert "等用户补充路径后重新验证目录" in reply


def test_fallback_reply_for_user_uses_real_error_instead_of_background_ack():
    from core.loop import _fallback_reply_for_user
    from tools.registry import ToolResult

    action = _judgment_output(decision="pause", rationale="源路径证据不存在，需要用户补充。")
    result = ToolResult(summary="路径不存在: /root/.openclaw/source", error="FileNotFound")

    reply = _fallback_reply_for_user(action, result, None)
    assert reply.startswith("状态: error")
    assert "detail:" in reply
    assert "路径不存在" in reply
    assert "后台继续处理" not in reply
    assert "我这轮" not in reply


def test_should_continue_within_tick_for_autonomous_act():
    from core.judgment import JudgmentOutput
    from core.loop import _should_continue_within_tick

    assert _should_continue_within_tick(_judgment_output(decision="act", chosen_action_id="file.read")) is True
    assert _should_continue_within_tick(_judgment_output(decision="wait")) is False
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.read"),
        user_message="帮我看下 mini 为什么 400",
        has_active_task=True,
    ) is False
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.read"),
        user_message="帮我看下 mini 为什么 400",
        has_active_task=False,
    ) is True


async def test_sync_task_progress_state_promotes_previous_next_step():
    from core.loop import _sync_task_progress_state
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("推进任务", goal="验证 current_step/next_step 同步", next_step="先读文件")
        task = await store.get_task_by_id(task_id)
        assert task is not None

        updated = await _sync_task_progress_state(
            store,
            task,
            previous_next_step="先读文件",
            action=_judgment_output(decision="act", chosen_action_id="file.read", next_step="再总结结论"),
            progressful=True,
        )
        assert updated is not None
        assert updated.current_step == "先读文件"
        assert updated.next_step == "再总结结论"

        updated2 = await _sync_task_progress_state(
            store,
            updated,
            previous_next_step="再总结结论",
            action=_judgment_output(decision="act", chosen_action_id="file.read", next_step=""),
            progressful=True,
        )
        assert updated2 is not None
        assert updated2.current_step == "再总结结论"
        assert updated2.next_step == ""
        await store.close()


async def test_sync_task_progress_state_preserves_explicit_current_step_from_state_delta():
    from core.loop import _sync_task_progress_state
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime-explicit.db")
        await store.open()
        task_id = await store.add_task("迁移任务", goal="验证显式 current_step 优先", next_step="继续旧技能")
        task = await store.get_task_by_id(task_id)
        assert task is not None

        await store.sync_task_progress(task_id, current_step="收到新迁移指令", next_step="开始盘点 openclaw 记忆")
        updated = await _sync_task_progress_state(
            store,
            task,
            previous_next_step="继续旧技能",
            action=_judgment_output(decision="act", chosen_action_id="task.update", next_step="开始盘点 openclaw 记忆"),
            progressful=True,
            state_delta={"current_step": "收到新迁移指令", "next_step": "开始盘点 openclaw 记忆"},
        )

        assert updated is not None
        assert updated.current_step == "收到新迁移指令"
        assert updated.next_step == "开始盘点 openclaw 记忆"
        await store.close()


def test_fmt_task_exposes_runtime_state_to_llm():
    from core.judgment import _fmt_task
    from memory.task_store import Task

    task = Task(
        id=7,
        title="测试任务",
        status="active",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="验证状态可见性",
        next_step="继续修复",
        current_step="检查 run monitor",
        model_tier="repair",
        result_json={"last_run_status": "failed"},
    )
    section = _fmt_task(task)
    assert "状态: active" in section
    assert "模型层级: repair" in section
    assert "当前步骤: 检查 run monitor" in section
    assert "最近运行状态: failed" in section


def test_tool_result_log_fields_include_state_delta():
    from core.execution import _tool_result_log_fields
    from tools.registry import ToolResult

    summary, error, state = _tool_result_log_fields(ToolResult(
        summary="工具完成\n含多行",
        error="",
        state_delta={"task_status": "waiting", "wait_key": "exec-1"},
    ))

    assert summary == "工具完成\\n含多行"
    assert error == ""
    assert '"task_status": "waiting"' in state
    assert '"wait_key": "exec-1"' in state


def test_tool_result_log_fields_prefer_log_summary_over_raw_text():
    from core.execution import _tool_result_log_fields
    from tools.registry import ToolResult

    summary, error, state = _tool_result_log_fields(ToolResult(
        summary="---\nlicense: Proprietary\nname: error-handling\n...",
        error="",
        metadata={"log_summary": "file.read path=/tmp/skill.md chars=2048 preview='---'"},
    ))

    assert summary == "file.read path=/tmp/skill.md chars=2048 preview='---'"
    assert error == ""
    assert state == ""


def test_clip_reply_for_log_strips_memory_context():
    from core.loop import _clip_reply_for_log

    clipped = _clip_reply_for_log("<memory-context>hidden</memory-context>\n用户可见回复")
    assert clipped == "用户可见回复"


# ══════════════════════════════════════════════════════════════════════════════
# 新增工具测试（file.edit / skill_ops / exec 覆盖）
# ══════════════════════════════════════════════════════════════════════════════

def test_file_edit_single_replace():
    """file.edit 单处替换成功。"""
    asyncio.run(_file_edit_single_replace())

async def _file_edit_single_replace():
    from tools.file import file_write, file_read, file_edit

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ctx = _tool_ctx(workspace_dir=d)
        fpath = root / "test.py"
        await file_write({"path": str(fpath), "content": "x = 1\ny = 2\nz = 3\n"}, ctx)

        # 单处替换
        res = await file_edit({"path": str(fpath), "edits": [{"oldText": "y = 2", "newText": "y = 20"}]}, ctx)
        assert res.error is None
        assert "1 处替换" in res.summary

        # 验证内容
        content = await file_read({"path": str(fpath)}, ctx)
        assert content.summary == "x = 1\ny = 20\nz = 3\n"


def test_file_edit_multiple_replace():
    """file.edit 多处替换成功。"""
    asyncio.run(_file_edit_multiple_replace())

async def _file_edit_multiple_replace():
    from tools.file import file_write, file_read, file_edit

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ctx = _tool_ctx(workspace_dir=d)
        fpath = root / "multi.py"
        await file_write({"path": str(fpath), "content": "a = 1\nb = 2\nc = 3\n"}, ctx)

        res = await file_edit({"path": str(fpath), "edits": [
            {"oldText": "a = 1", "newText": "a = 10"},
            {"oldText": "c = 3", "newText": "c = 30"},
        ]}, ctx)
        assert res.error is None
        assert "2 处替换" in res.summary

        content = await file_read({"path": str(fpath)}, ctx)
        assert "a = 10" in content.summary
        assert "c = 30" in content.summary


def test_file_edit_errors():
    """file.edit 错误处理：oldText 不唯一 / 不存在 / 空 edits / 文件不存在。"""
    asyncio.run(_file_edit_errors())

async def _file_edit_errors():
    from tools.file import file_write, file_edit

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ctx = _tool_ctx(workspace_dir=d)
        fpath = root / "err.py"
        await file_write({"path": str(fpath), "content": "x = 1\nx = 1\ny = 2\n"}, ctx)

        # 文件不存在
        r = await file_edit({"path": str(root / "nonexistent.py"), "edits": [{"oldText": "a", "newText": "b"}]}, ctx)
        assert r.error == "FileNotFound"

        # 空 edits
        r2 = await file_edit({"path": str(fpath), "edits": []}, ctx)
        assert r2.skipped is True
        assert r2.error == "EmptyEdits"

        # oldText 不存在
        r3 = await file_edit({"path": str(fpath), "edits": [{"oldText": "ZZZ", "newText": "b"}]}, ctx)
        assert r3.skipped is True
        assert r3.error == "OldTextNotFound"

        # oldText 不唯一
        r4 = await file_edit({"path": str(fpath), "edits": [{"oldText": "x = 1", "newText": "x = 10"}]}, ctx)
        assert r4.skipped is True
        assert r4.error == "NonUniqueOldText"


def test_skill_list_and_search():
    """skill.list 和 skill.search 工具正常返回。"""
    asyncio.run(_skill_list_and_search())

async def _skill_list_and_search():
    from tools.skill_ops import skill_list, skill_search

    ws = _proj_root() / "workspace"
    ctx = _tool_ctx(workspace_dir=str(ws))

    r = await skill_list({}, ctx)
    assert r.error is None
    # 至少有 builtin skills
    assert "runtime.bootstrap" in r.summary

    r2 = await skill_search({"query": "失败"}, ctx)
    assert r2.error is None
    # 搜索 "失败" 应匹配 failure.reflection
    assert "failure.reflection" in r2.summary

    # 搜索不存在的词 → 返回"未找到"，不是 skipped
    r3 = await skill_search({"query": "zxcvbnm_nonexistent_skill_query"}, ctx)
    assert r3.error is None
    assert "没有找到" in r3.summary


def test_exec_empty_command():
    """exec 空命令应被拒绝。"""
    asyncio.run(_exec_empty_command())

async def _exec_empty_command():
    from tools.exec import exec_run

    ctx = _tool_ctx()
    res = await exec_run({"command": ""}, ctx)
    assert res.skipped is True
    assert res.error == "EmptyCommand"


def test_process_kill():
    """process.kill 可以终止后台进程。"""
    asyncio.run(_process_kill())

async def _process_kill():
    import json
    from tools.exec import exec_run, process_kill, process_poll, process_list, _MANAGER

    _MANAGER.clear()
    ctx = _tool_ctx()

    res = await exec_run({"command": "sleep 60", "background": True, "timeout": 60}, ctx)
    sid = json.loads(res.evidence)["session_id"]

    # 确认进程存在
    poll1 = await process_poll({"session_id": sid}, ctx)
    status = json.loads(poll1.summary)
    assert status["status"] == "running"

    # kill
    kill_res = await process_kill({"session_id": sid}, ctx)
    assert kill_res.error is None
    assert "已终止" in kill_res.summary

    # 确认已终止
    poll2 = await process_poll({"session_id": sid}, ctx)
    status2 = json.loads(poll2.summary)
    assert status2["status"] == "finished"


def test_process_list():
    """process.list 返回通过 exec 启动的进程。"""
    asyncio.run(_process_list())

async def _process_list():
    import json
    from tools.exec import exec_run, process_list, _MANAGER

    _MANAGER.clear()
    ctx = _tool_ctx()

    # 空列表
    r = await process_list({"state": "all"}, ctx)
    assert "无进程" in r.summary

    # 启动一个后台进程
    res = await exec_run({"command": "sleep 5", "background": True, "timeout": 10}, ctx)
    sid = json.loads(res.evidence)["session_id"]

    r2 = await process_list({"state": "running"}, ctx)
    assert sid in r2.summary


def test_process_write_to_finished():
    """向已结束的进程写入应被拒绝。"""
    asyncio.run(_process_write_to_finished())

async def _process_write_to_finished():
    import json
    from tools.exec import exec_run, process_write, _MANAGER

    _MANAGER.clear()
    ctx = _tool_ctx()

    res = await exec_run({"command": "echo done"}, ctx)  # 前台，立即结束
    assert res.error is None

    # 前台进程不在 _MANAGER 中，所以写一个短命令后台
    res2 = await exec_run({"command": "echo hi", "background": True, "timeout": 2}, ctx)
    sid = json.loads(res2.evidence)["session_id"]
    await asyncio.sleep(0.5)  # 等待完成

    # 写入已结束进程
    w = await process_write({"session_id": sid, "data": "hello"}, ctx)
    assert w.skipped is True
    assert w.error == "ProcessFinished"


def test_process_poll_exposes_handle_lost_interaction_state():
    asyncio.run(_process_poll_exposes_handle_lost_interaction_state())


async def _process_poll_exposes_handle_lost_interaction_state():
    import json
    import os
    import time

    from tools.exec import ProcessInfo, process_poll, process_write, _MANAGER

    _MANAGER.clear()
    info = ProcessInfo(
        session_id="restored-1",
        command="python -i",
        pid=os.getpid(),
        started_at=time.time() - 5,
        background=True,
        restored=True,
        handle_lost=True,
    )
    _MANAGER.register(info)

    ctx = _tool_ctx()
    poll = await process_poll({"session_id": "restored-1"}, ctx)
    status = json.loads(poll.summary)
    assert status["restored"] is True
    assert status["handle_lost"] is True
    assert status["interaction_available"] is False

    write = await process_write({"session_id": "restored-1", "data": "hello"}, ctx)
    assert write.error == "ProcessHandleLost"
    assert write.metadata["handle_lost"] is True


def test_file_edit_json_string_edits():
    """file.edit 支持 edits 为 JSON 字符串。"""
    asyncio.run(_file_edit_json_string_edits())

async def _file_edit_json_string_edits():
    import json as _json
    from tools.file import file_write, file_read, file_edit

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ctx = _tool_ctx(workspace_dir=d)
        fpath = root / "jsontest.py"
        await file_write({"path": str(fpath), "content": "v = 1\n"}, ctx)

        edits_str = _json.dumps([{"oldText": "v = 1", "newText": "v = 2"}])
        res = await file_edit({"path": str(fpath), "edits": edits_str}, ctx)
        assert res.error is None

        content = await file_read({"path": str(fpath)}, ctx)
        assert content.summary == "v = 2\n"


def test_file_read_max_chars():
    """file.read max_chars 参数正确截断。"""
    asyncio.run(_file_read_max_chars())

async def _file_read_max_chars():
    from tools.file import file_write, file_read

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ctx = _tool_ctx(workspace_dir=d)
        fpath = root / "big.txt"
        await file_write({"path": str(fpath), "content": "abcdefghij" * 100}, ctx)  # 1000 chars

        r = await file_read({"path": str(fpath), "max_chars": 20}, ctx)
        assert len(r.summary) == 20

