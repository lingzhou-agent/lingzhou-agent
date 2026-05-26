"""文件、进程、shell 工具测试"""
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
# 新增工具测试（file.edit / skill_ops / exec 覆盖）
# ══════════════════════════════════════════════════════════════════════════════


def test_task_store_get_active_prefers_started_task_over_pending():
    asyncio.run(_task_store_get_active_prefers_started_task_over_pending())


async def _task_store_get_active_prefers_started_task_over_pending():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            pending_id = await store.add_task(
                "pending critical",
                goal="should not outrank running task",
                priority="critical",
                status="pending",
            )
            running_id = await store.add_task(
                "running low",
                goal="must stay active",
                priority="low",
                status="in_progress",
            )

            active = await store.get_active()

            assert active is not None
            assert active.id == running_id
            assert active.id != pending_id
            assert active.status == "in_progress"
        finally:
            await store.close()


def test_task_tools_do_not_reenter_terminal_tasks():
    asyncio.run(_task_tools_do_not_reenter_terminal_tasks())


async def _task_tools_do_not_reenter_terminal_tasks():
    from store.task import TaskStore
    from tools.task_ops import task_advance, task_complete

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            done_id = await store.add_task("done task", goal="already finished", status="done")
            cancelled_id = await store.add_task("cancelled task", goal="already cancelled", status="cancelled")
            ctx = _tool_ctx(task_store=store)

            done_advance = await task_advance({"task_id": done_id}, ctx)
            cancelled_advance = await task_advance({"task_id": cancelled_id}, ctx)
            done_complete = await task_complete({"task_id": done_id}, ctx)
            cancelled_complete = await task_complete({"task_id": cancelled_id}, ctx)

            assert done_advance.skipped is True
            assert done_advance.summary == f"任务 [{done_id}] 已完成，不能再次推进"
            assert done_advance.metadata["task_id"] == done_id

            assert cancelled_advance.skipped is True
            assert cancelled_advance.summary == f"任务 [{cancelled_id}] 已取消，不能再次推进"
            assert cancelled_advance.metadata["task_id"] == cancelled_id

            assert done_complete.skipped is True
            assert done_complete.summary == f"任务 [{done_id}] 已完成"
            assert done_complete.metadata["task_id"] == done_id

            assert cancelled_complete.skipped is True
            assert cancelled_complete.summary == f"任务 [{cancelled_id}] 已取消，不能完成"
            assert cancelled_complete.metadata["task_id"] == cancelled_id

            done_task = await store.get_task_by_id(done_id)
            cancelled_task = await store.get_task_by_id(cancelled_id)
            assert done_task is not None and done_task.status == "done"
            assert cancelled_task is not None and cancelled_task.status == "cancelled"
        finally:
            await store.close()

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

    r = await skill_list({"scope": "seed"}, ctx)
    assert r.error is None
    # 至少有 seed skills
    assert "runtime-bootstrap [seed]" in r.summary

    r2 = await skill_search({"query": "失败"}, ctx)
    assert r2.error is None
    # 搜索 "失败" 应匹配 failure-reflection
    assert "failure-reflection" in r2.summary

    # 搜索不存在的词 → 返回"未找到"，不是 skipped
    r3 = await skill_search({"query": "zxcvbnm_nonexistent_skill_query"}, ctx)
    assert r3.error is None
    assert "没有找到" in r3.summary


def test_skill_activate_reads_skill_markdown_and_resources():
    asyncio.run(_skill_activate_reads_skill_markdown_and_resources())


async def _skill_activate_reads_skill_markdown_and_resources():
    from tools.skill_ops import skill_activate

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        skill_dir = root / "skills" / "sample-skill"
        (skill_dir / "references").mkdir(parents=True)
        (skill_dir / "references" / "REFERENCE.md").write_text("reference body", encoding="utf-8")
        (skill_dir / "SKILL.md").write_text(
            """---
name: sample-skill
description: Use when testing activation.
---
先读取 references/REFERENCE.md，再执行下一步。
""",
            encoding="utf-8",
        )

        ctx = _tool_ctx(workspace_dir=d)
        res = await skill_activate({"name": "sample-skill"}, ctx)

        assert res.error is None
        assert "<skill_content name=\"sample-skill\">" in res.summary
        assert "references/REFERENCE.md" in res.summary
        assert res.metadata["skill"] == "sample-skill"


def test_skill_evolve_uses_judgment_provider_and_registry(monkeypatch):
    asyncio.run(_skill_evolve_uses_judgment_provider_and_registry(monkeypatch))


async def _skill_evolve_uses_judgment_provider_and_registry(monkeypatch):
    import core.evolution as evolution_mod
    from tools.registry import ToolContext, ToolRegistry
    from tools.skill_ops import skill_evolve

    provider = object()
    registry = ToolRegistry()
    observed: dict[str, Any] = {}

    class _FakeEngine:
        def __init__(self, cfg: Any, provider_arg: Any, registry_arg: Any) -> None:
            observed["cfg"] = cfg
            observed["provider"] = provider_arg
            observed["registry"] = registry_arg

        async def evolve_skill(self, name: str, feedback: str, ctx: Any = None) -> Any:
            observed["name"] = name
            observed["feedback"] = feedback
            observed["ctx"] = ctx
            return SimpleNamespace(success=True, target="sample-skill", new_code="# updated")

    monkeypatch.setattr(evolution_mod, "EvolutionEngine", _FakeEngine)

    ctx = ToolContext(
        config=_test_config(),
        wm=cast(Any, None),
        task_store=cast(Any, None),
        episodic=cast(Any, None),
        semantic=cast(Any, None),
        emotion=cast(Any, None),
        judgment=SimpleNamespace(_provider=provider, _registry=registry),
    )

    res = await skill_evolve({"name": "sample-skill", "feedback": "tighten guardrails"}, ctx)

    assert res.error is None
    assert observed["provider"] is provider
    assert observed["registry"] is registry
    assert observed["name"] == "sample-skill"
    assert observed["feedback"] == "tighten guardrails"


def test_config_set_rejects_unknown_interval_key(monkeypatch):
    asyncio.run(_config_set_rejects_unknown_interval_key(monkeypatch))


async def _config_set_rejects_unknown_interval_key(monkeypatch):
    from tools.config_ops import config_set
    import tools.config_ops as config_mod

    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "lingzhou.json"
        cfg_path.write_text((_proj_root() / "lingzhou.json.example").read_text(encoding="utf-8"), encoding="utf-8")
        before = cfg_path.read_text(encoding="utf-8")

        monkeypatch.setattr(config_mod, "_resolve_config_path", lambda ctx=None: cfg_path)

        res = await config_set({"key": "loop.interval", "value": "100"}, _tool_ctx())

        assert res.error == "UnknownConfigKey"
        assert "固定 tick interval 已废弃" in res.summary
        assert cfg_path.read_text(encoding="utf-8") == before


def test_config_set_accepts_duration_string_for_millisecond_fields(monkeypatch):
    asyncio.run(_config_set_accepts_duration_string_for_millisecond_fields(monkeypatch))


async def _config_set_accepts_duration_string_for_millisecond_fields(monkeypatch):
    from tools.config_ops import config_set
    import tools.config_ops as config_mod

    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "lingzhou.json"
        cfg_path.write_text((_proj_root() / "lingzhou.json.example").read_text(encoding="utf-8"), encoding="utf-8")

        monkeypatch.setattr(config_mod, "_resolve_config_path", lambda ctx=None: cfg_path)

        res = await config_set({"key": "loop.wake_poll_interval", "value": "100ms"}, _tool_ctx())

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        assert res.error is None
        assert "✅ loop.wake_poll_interval" in res.summary
        assert cfg["loop"]["wake_poll_interval"] == 100


def test_subagent_filtered_registry_blocks_parent_mutations():
    from core.subagent import _DEFAULT_BLOCKED_TOOLS, _FilteredRegistry
    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.discover(_proj_root() / "tools")
    filtered = _FilteredRegistry(registry, None, set(_DEFAULT_BLOCKED_TOOLS))

    assert filtered.get("memory.set_fact") is None
    assert filtered.get("schedule.add") is None
    assert filtered.get("task.plan") is None
    assert filtered.get("memory.search") is not None
    assert filtered.get("memory.add_wm") is not None
    assert filtered.get("task.ask") is not None


def test_tool_registry_discover_skips_hidden_smoke_failed_modules():
    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = f"visible_tool_{time.time_ns()}"
        manifest_name = f"probe.hidden_skip.{time.time_ns()}"
        (root / f"{stem}.py").write_text(
            "from tools.registry import ToolManifest, ToolResult, tool\n"
            f"@tool(ToolManifest(name={manifest_name!r}, description='visible probe'))\n"
            "async def _visible_probe(params, ctx):\n"
            "    return ToolResult(summary='ok')\n",
            encoding="utf-8",
        )
        (root / f".{stem}.smoke-failed.py").write_text(
            "raise RuntimeError('hidden smoke-failed artifact must not be imported')\n",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.discover(root)

        assert registry.get(manifest_name) is not None


def test_tool_registry_discover_accepts_legacy_manifest_kwargs():
    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = f"legacy_tool_{time.time_ns()}"
        manifest_name = f"probe.legacy_compat.{time.time_ns()}"
        (root / f"{stem}.py").write_text(
            "from tools.registry import ToolManifest, ToolParam, ToolResult, tool\n"
            f"@tool(ToolManifest(name={manifest_name!r}, description='legacy probe', parameters=[ToolParam(name='path', type='string', description='目录路径', default='.')], required_caps=('plan_bootstrap_exempt',)))\n"
            "async def _legacy_probe(params, ctx):\n"
            "    return ToolResult(summary='ok')\n",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.discover(root)

        entry = registry.get(manifest_name)
        assert entry is not None
        assert entry.manifest.required_caps == ("plan_bootstrap_exempt",)
        assert entry.manifest.parameters[0].default == "."
        assert entry.manifest.params[0].dtype == "string"


def test_tool_registry_discover_cleans_partial_module_after_import_failure():
    import sys

    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        broken_stem = f"broken_tool_{time.time_ns()}"
        dependent_stem = f"dependent_tool_{time.time_ns()}"
        broken_module_name = f"tools.{broken_stem}"
        dependent_manifest = f"probe.dependent_recover.{time.time_ns()}"

        (root / f"{broken_stem}.py").write_text(
            "PARTIAL = True\n"
            "raise RuntimeError('boom during import')\n"
            "def resolve_read_path():\n"
            "    return 'ok'\n",
            encoding="utf-8",
        )
        (root / f"{dependent_stem}.py").write_text(
            f"from tools.{broken_stem} import resolve_read_path\n"
            "from tools.registry import ToolManifest, ToolResult, tool\n"
            f"@tool(ToolManifest(name={dependent_manifest!r}, description='dependent probe'))\n"
            "async def _dependent_probe(params, ctx):\n"
            "    return ToolResult(summary=resolve_read_path())\n",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        with pytest.raises(RuntimeError, match="boom during import"):
            registry.discover(root)

        assert broken_module_name not in sys.modules

        (root / f"{broken_stem}.py").write_text(
            "def resolve_read_path():\n"
            "    return 'ok'\n",
            encoding="utf-8",
        )

        registry.discover(root)

        assert registry.get(dependent_manifest) is not None
        sys.modules.pop(broken_module_name, None)
        sys.modules.pop(f"tools.{dependent_stem}", None)


def test_tool_registry_reload_restores_previous_module_after_failure():
    import sys

    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = f"reloadable_tool_{time.time_ns()}"
        module_name = f"tools.{stem}"
        manifest_name = f"probe.reload_restore.{time.time_ns()}"

        (root / f"{stem}.py").write_text(
            "def resolve_read_path():\n"
            "    return 'ok'\n"
            "from tools.registry import ToolManifest, ToolResult, tool\n"
            f"@tool(ToolManifest(name={manifest_name!r}, description='reload probe'))\n"
            "async def _reload_probe(params, ctx):\n"
            "    return ToolResult(summary=resolve_read_path())\n",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.discover(root)
        baseline = sys.modules[module_name]
        assert getattr(baseline, "resolve_read_path")() == "ok"

        (root / f"{stem}.py").write_text(
            "PARTIAL = True\n"
            "raise RuntimeError('boom during reload')\n"
            "def resolve_read_path():\n"
            "    return 'broken'\n",
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="boom during reload"):
            registry.reload_tool(stem, root)

        restored = sys.modules[module_name]
        assert restored is baseline
        assert getattr(restored, "resolve_read_path")() == "ok"
        sys.modules.pop(module_name, None)


def test_subagent_runner_restores_parent_registry_after_child_exception():
    asyncio.run(_subagent_runner_restores_parent_registry_after_child_exception())


async def _subagent_runner_restores_parent_registry_after_child_exception():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, tool

    @tool(ToolManifest(
        name="probe.raise_registry",
        description="测试子灵异常后 registry 是否恢复",
        progress_category="info",
    ))
    async def _probe_raise_registry(params: dict[str, Any], ctx: Any) -> Any:
        raise RuntimeError("registry restore probe")

    class _FakeJudgment:
        def __init__(self) -> None:
            self._calls = 0

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.raise_registry",
                    params={},
                    rationale="probe registry restore",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast(Any, SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=SimpleNamespace(ethos_baseline={
                    "truth": 0.91,
                    "caution": 0.81,
                    "continuity": 0.71,
                    "curiosity": 0.61,
                    "care": 0.51,
                }),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast(Any, SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="恢复 registry", max_ticks=2, allowed_tools=["probe.raise_registry"]),
                parent_ctx,
                cast(Any, _FakeJudgment()),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert "工具执行异常: registry restore probe" in result.last_summary
            assert execution._registry is registry  # type: ignore[attr-defined]
            assert execution._registry.get("task.ask") is not None  # type: ignore[attr-defined]
            assert execution._registry.get("probe.raise_registry") is not None  # type: ignore[attr-defined]
        finally:
            await store.close()


def test_subagent_runner_passes_filtered_registry_to_judgment():
    asyncio.run(_subagent_runner_passes_filtered_registry_to_judgment())


async def _subagent_runner_passes_filtered_registry_to_judgment():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolContext, ToolRegistry

    captured_visible_tools: list[str] = []

    class _RecordingJudgment:
        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            registry_override = kwargs.get("registry_override")
            assert registry_override is not None
            captured_visible_tools.extend(sorted(m.name for m in registry_override.list_manifests()))
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast(Any, SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=SimpleNamespace(ethos_baseline={
                    "truth": 0.91,
                    "caution": 0.81,
                    "continuity": 0.71,
                    "curiosity": 0.61,
                    "care": 0.51,
                }),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast(Any, SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="检查子灵可见工具", max_ticks=1, allowed_tools=["task.list"]),
                parent_ctx,
                cast(Any, _RecordingJudgment()),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert "task.list" in captured_visible_tools
            assert "shell.run" not in captured_visible_tools
            assert "subagent.run" not in captured_visible_tools
        finally:
            await store.close()


def test_subagent_task_store_view_exposes_local_state_to_subsequent_ticks():
    asyncio.run(_subagent_task_store_view_exposes_local_state_to_subsequent_ticks())


async def _subagent_task_store_view_exposes_local_state_to_subsequent_ticks():
    from core.subagent import _SubagentTaskStoreView
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "child active task",
                goal="verify local task result overlay",
                status="in_progress",
                result_json={"summary": "before"},
            )
            view = _SubagentTaskStoreView(store)

            await view.set_fact("control:durable_failure_policy", json.dumps({"threshold": 5}), scope="system")
            fact, found = await view.get_fact("control:durable_failure_policy")
            assert found is True
            assert json.loads(fact)["threshold"] == 5
            facts = await view.list_facts(prefix="control:", limit=5)
            assert any(key == "control:durable_failure_policy" for key, _ in facts)

            await view.update_task_result(task_id, {"last_run_status": "failed", "summary": "child summary"})
            active_task = await view.get_active()
            assert active_task is not None
            assert active_task.id == task_id
            assert active_task.result_json["last_run_status"] == "failed"
            assert active_task.result_json["summary"] == "child summary"
            fetched_task = await view.get_task_by_id(task_id)
            assert fetched_task is not None
            assert fetched_task.result_json["last_run_status"] == "failed"
            listed_tasks = await view.list_tasks(status="in_progress", limit=5)
            assert any(item.id == task_id and item.result_json.get("last_run_status") == "failed" for item in listed_tasks)

            run_id = await view.add_run(
                task_id=7,
                run_type="llm",
                worker_type="llm-worker",
                status="running",
                tool_name="probe.local",
                input_json={"query": "child"},
            )
            await view.update_run(run_id, status="failed", progress="phase-2", error_text="boom")

            run = await view.get_run_by_id(run_id)
            assert run is not None
            assert run.status == "failed"
            assert run.progress == "phase-2"
            runs = await view.list_runs(task_id=7, limit=5)
            assert any(item.id == run_id and item.error_text == "boom" for item in runs)

            await view.add_meta_reflection(
                reflection_id="local-r1",
                target_kind="threshold",
                trigger="failure_pattern",
                loop_level="single",
                diagnosis="child diagnosis",
                proposal="child proposal",
                verification_plan="rerun once",
                decision="apply",
                task_id=7,
                run_id=run_id,
                tool_name="probe.local",
            )

            reflections = await view.list_meta_reflections(limit=5)
            assert any(item.id == "local-r1" and item.run_id == run_id for item in reflections)
            filtered = await view.list_meta_reflections(limit=5, loop_level="single")
            assert any(item.id == "local-r1" for item in filtered)
        finally:
            await store.close()


def test_subagent_task_store_view_hides_parent_waiting_tasks_from_child_context():
    asyncio.run(_subagent_task_store_view_hides_parent_waiting_tasks_from_child_context())


async def _subagent_task_store_view_hides_parent_waiting_tasks_from_child_context():
    from core.subagent import _SubagentTaskStoreView
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            waiting_task_id = await store.add_task(
                "parent waiting task",
                goal="should not leak into child waiting context",
                status="waiting",
                wait_kind="external",
                wait_key="user-input",
                next_step="wait for parent input",
            )
            view = _SubagentTaskStoreView(store)

            waiting_tasks = await view.list_tasks(status="waiting", limit=5)

            assert waiting_tasks == []
            parent_waiting = await store.list_tasks(status="waiting", limit=5)
            assert any(item.id == waiting_task_id for item in parent_waiting)
        finally:
            await store.close()


def test_subagent_runner_uses_virtual_active_task_instead_of_parent_task():
    asyncio.run(_subagent_runner_uses_virtual_active_task_instead_of_parent_task())


async def _subagent_runner_uses_virtual_active_task_instead_of_parent_task():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    observed: dict[str, Any] = {}

    @tool(ToolManifest(
        name="probe.capture_active_task",
        description="测试子灵 active task 使用本地虚拟 task 而非父灵 task",
        progress_category="info",
    ))
    async def _probe_capture_active_task(params: dict[str, Any], ctx: Any) -> Any:
        task = await ctx.task_store.get_active()
        observed["tool_task"] = task
        assert task is not None
        await ctx.task_store.update_task_result(task.id, {"probe_marker": "child-local"})
        return ToolResult(summary=f"active-task={task.id}:{task.title}", kind="execute_result", priority=0.5)

    class _FakeJudgment:
        def __init__(self) -> None:
            self._calls = 0
            self._seen_tasks: list[Any] = []

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            task = await args[2].get_active()
            self._seen_tasks.append(task)
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.capture_active_task",
                    params={},
                    rationale="probe virtual child task",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            parent_task_id = await store.add_task(
                "parent active task",
                goal="should stay in parent only",
                status="in_progress",
                current_step="parent-step",
                next_step="parent-next",
                result_json={"summary": "parent-summary"},
            )
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast(Any, SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=SimpleNamespace(ethos_baseline={
                    "truth": 0.91,
                    "caution": 0.81,
                    "continuity": 0.71,
                    "curiosity": 0.61,
                    "care": 0.51,
                }),
            ))
            execution = ExecutionLayer(registry, cfg)
            judgment = _FakeJudgment()
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast(Any, SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="隔离子灵 active task", max_ticks=2, allowed_tools=["probe.capture_active_task"]),
                parent_ctx,
                cast(Any, judgment),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert len(judgment._seen_tasks) == 2
            first_task = judgment._seen_tasks[0]
            second_task = judgment._seen_tasks[1]
            assert first_task is not None
            assert first_task.id < 0
            assert first_task.id != parent_task_id
            assert first_task.title.startswith("子灵任务: ")
            assert first_task.goal == "隔离子灵 active task"
            assert first_task.current_step == ""
            assert first_task.next_step == ""
            assert first_task.result_json == {}
            assert second_task is not None
            assert second_task.id == first_task.id
            assert second_task.result_json["last_run_status"] == "succeeded"
            assert second_task.result_json["probe_marker"] == "child-local"

            tool_task = observed["tool_task"]
            assert tool_task is not None
            assert tool_task.id == first_task.id
            assert tool_task.title == first_task.title
            assert result.last_summary == f"active-task={first_task.id}:{first_task.title}"

            parent_active = await store.get_active()
            assert parent_active is not None
            assert parent_active.id == parent_task_id
            assert parent_active.title == "parent active task"
            assert parent_active.result_json["summary"] == "parent-summary"
        finally:
            await store.close()


def test_subagent_task_list_does_not_expose_parent_tasks():
    asyncio.run(_subagent_task_list_does_not_expose_parent_tasks())


async def _subagent_task_list_does_not_expose_parent_tasks():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolContext, ToolRegistry

    class _FakeJudgment:
        def __init__(self) -> None:
            self._calls = 0

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="task.list",
                    params={"limit": 5},
                    rationale="probe task list isolation",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            await store.add_task(
                "parent active task",
                goal="should not show in child task.list",
                status="in_progress",
                current_step="parent-step",
            )
            await store.add_task(
                "parent pending task",
                goal="should also stay hidden",
                status="pending",
            )
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast(Any, SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=SimpleNamespace(ethos_baseline={
                    "truth": 0.91,
                    "caution": 0.81,
                    "continuity": 0.71,
                    "curiosity": 0.61,
                    "care": 0.51,
                }),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast(Any, SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="列出子灵可见任务", max_ticks=2, allowed_tools=["task.list"]),
                parent_ctx,
                cast(Any, _FakeJudgment()),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert "子灵任务: 列出子灵可见任务" in result.last_summary
            assert "parent active task" not in result.last_summary
            assert "parent pending task" not in result.last_summary
        finally:
            await store.close()


def test_subagent_explicit_task_id_does_not_expose_parent_task():
    asyncio.run(_subagent_explicit_task_id_does_not_expose_parent_task())


async def _subagent_explicit_task_id_does_not_expose_parent_task():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    @tool(ToolManifest(
        name="probe.read_task_by_id",
        description="测试子灵不能通过显式 task_id 读取父灵任务",
        progress_category="info",
    ))
    async def _probe_read_task_by_id(params: dict[str, Any], ctx: Any) -> Any:
        task = await ctx.task_store.get_task_by_id(int(params.get("task_id") or 0))
        if task is None:
            return ToolResult(summary="task-by-id=not-found", kind="execute_result", priority=0.5)
        return ToolResult(summary=f"task-by-id={task.id}:{task.title}", kind="execute_result", priority=0.5)

    class _FakeJudgment:
        def __init__(self, parent_task_id: int) -> None:
            self._calls = 0
            self._parent_task_id = parent_task_id

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.read_task_by_id",
                    params={"task_id": self._parent_task_id},
                    rationale="probe explicit task id isolation",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            parent_task_id = await store.add_task(
                "parent secret task",
                goal="should not be readable by child via explicit task_id",
                status="in_progress",
            )
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast(Any, SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=SimpleNamespace(ethos_baseline={
                    "truth": 0.91,
                    "caution": 0.81,
                    "continuity": 0.71,
                    "curiosity": 0.61,
                    "care": 0.51,
                }),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast(Any, SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="阻断显式 task_id 泄漏", max_ticks=2, allowed_tools=["probe.read_task_by_id"]),
                parent_ctx,
                cast(Any, _FakeJudgment(parent_task_id)),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert result.last_summary == "task-by-id=not-found"
        finally:
            await store.close()


def test_subagent_run_history_does_not_expose_parent_runs():
    asyncio.run(_subagent_run_history_does_not_expose_parent_runs())


async def _subagent_run_history_does_not_expose_parent_runs():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    @tool(ToolManifest(
        name="probe.read_runs",
        description="测试子灵不能读取父灵 runs 历史",
        progress_category="info",
    ))
    async def _probe_read_runs(params: dict[str, Any], ctx: Any) -> Any:
        runs = await ctx.task_store.list_runs(limit=5)
        parent_run = await ctx.task_store.get_run_by_id(int(params.get("run_id") or 0))
        first_tool = runs[0].tool_name if runs else "-"
        parent_state = "hit" if parent_run is not None else "miss"
        return ToolResult(
            summary=f"runs={len(runs)} first={first_tool} parent-run={parent_state}",
            kind="execute_result",
            priority=0.5,
        )

    class _FakeJudgment:
        def __init__(self, parent_run_id: int) -> None:
            self._calls = 0
            self._parent_run_id = parent_run_id

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.read_runs",
                    params={"run_id": self._parent_run_id},
                    rationale="probe run history isolation",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            parent_task_id = await store.add_task(
                "parent task with run",
                goal="should not leak runs into child",
                status="in_progress",
            )
            parent_run_id = await store.add_run(
                task_id=parent_task_id,
                tool_name="probe.parent",
                status="succeeded",
                output_json={"summary": "parent run"},
            )
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast(Any, SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=SimpleNamespace(ethos_baseline={
                    "truth": 0.91,
                    "caution": 0.81,
                    "continuity": 0.71,
                    "curiosity": 0.61,
                    "care": 0.51,
                }),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast(Any, SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="阻断 parent run 泄漏", max_ticks=2, allowed_tools=["probe.read_runs"]),
                parent_ctx,
                cast(Any, _FakeJudgment(parent_run_id)),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert result.last_summary == "runs=1 first=probe.read_runs parent-run=miss"
        finally:
            await store.close()


def test_subagent_failure_and_reflection_history_do_not_expose_parent_state():
    asyncio.run(_subagent_failure_and_reflection_history_do_not_expose_parent_state())


async def _subagent_failure_and_reflection_history_do_not_expose_parent_state():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    @tool(ToolManifest(
        name="probe.read_failure_state",
        description="测试子灵不能读取父灵 failures/meta reflections",
        progress_category="info",
    ))
    async def _probe_read_failure_state(params: dict[str, Any], ctx: Any) -> Any:
        failures = await ctx.task_store.list_failures(limit=5)
        task_failures = await ctx.task_store.list_failures_for_task(str(params.get("task_id") or ""), limit=5)
        reflections = await ctx.task_store.list_meta_reflections(limit=5)
        return ToolResult(
            summary=(
                f"failures={len(failures)} "
                f"task-failures={len(task_failures)} "
                f"reflections={len(reflections)}"
            ),
            kind="execute_result",
            priority=0.5,
        )

    class _FakeJudgment:
        def __init__(self, parent_task_id: int) -> None:
            self._calls = 0
            self._parent_task_id = parent_task_id

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.read_failure_state",
                    params={"task_id": self._parent_task_id},
                    rationale="probe failure/reflection isolation",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            parent_task_id = await store.add_task(
                "parent task with failures",
                goal="should not leak failure state into child",
                status="in_progress",
            )
            await store.record_failure(
                kind="probe.parent_failure",
                summary="parent failure",
                context="parent failure context",
                task_id=str(parent_task_id),
            )
            await store.add_meta_reflection(
                reflection_id="parent-r1",
                target_kind="tool",
                trigger="failure_pattern",
                loop_level="single",
                diagnosis="parent diagnosis",
                proposal="parent proposal",
                verification_plan="parent rerun",
                decision="apply",
                task_id=parent_task_id,
                run_id=1,
                tool_name="probe.parent_failure",
            )
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast(Any, SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=SimpleNamespace(ethos_baseline={
                    "truth": 0.91,
                    "caution": 0.81,
                    "continuity": 0.71,
                    "curiosity": 0.61,
                    "care": 0.51,
                }),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast(Any, SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="阻断 parent failure 泄漏", max_ticks=2, allowed_tools=["probe.read_failure_state"]),
                parent_ctx,
                cast(Any, _FakeJudgment(parent_task_id)),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert result.last_summary == "failures=0 task-failures=0 reflections=0"
        finally:
            await store.close()


def test_subagent_runner_does_not_pollute_parent_store():
    asyncio.run(_subagent_runner_does_not_pollute_parent_store())


async def _subagent_runner_does_not_pollute_parent_store():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.perception import EmotionState
    from core.perception.ethos import EthosState
    from core.subagent import SubagentConfig, make_subagent_runner
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolContext, ToolRegistry

    class _FakeJudgment:
        def __init__(self) -> None:
            self._calls = 0
            self._last_emotion: EmotionState | None = None
            self._last_ethos: EthosState | None = None

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            self._last_emotion = cast(EmotionState, args[5])
            self._last_ethos = cast(EthosState | None, kwargs.get("ethos_state"))
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="task.ask",
                    params={"question": "请确认只读子灵是否生效？"},
                    rationale="ask once",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            cfg = cast(Any, SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.23, baseline_arousal=0.34),
                soul=SimpleNamespace(ethos_baseline={
                    "truth": 0.91,
                    "caution": 0.81,
                    "continuity": 0.71,
                    "curiosity": 0.61,
                    "care": 0.51,
                }),
            ))
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            execution = ExecutionLayer(registry, cfg)
            judgment = _FakeJudgment()
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast(Any, SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            runner = make_subagent_runner(
                SubagentConfig(goal="只读检查", max_ticks=2),
                parent_ctx,
                cast(Any, judgment),
                execution,
                registry,
            )

            result = await runner.run()

            assert result.completed is True
            assert "已登记用户澄清请求" in result.last_summary
            assert await store.list_runs(limit=10) == []
            assert await store.list_failures(limit=10) == []
            assert await store.list_facts(prefix="durable_failure:", limit=10) == []
            assert await store.list_meta_reflections(limit=10) == []
            assert judgment._last_emotion is not None
            assert judgment._last_emotion.valence == pytest.approx(0.23)
            assert judgment._last_emotion.arousal == pytest.approx(0.34)
            assert judgment._last_ethos is not None
            assert judgment._last_ethos.values.truth == pytest.approx(0.91)
            assert judgment._last_ethos.values.caution == pytest.approx(0.81)
        finally:
            await store.close()


def test_subagent_runner_shared_memory_does_not_write_parent_episodic():
    asyncio.run(_subagent_runner_shared_memory_does_not_write_parent_episodic())


async def _subagent_runner_shared_memory_does_not_write_parent_episodic():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    @tool(ToolManifest(
        name="probe.ep_write",
        description="测试子灵 shared-memory 是否会污染父灵 episodic",
        progress_category="info",
    ))
    async def _probe_ep_write(params: dict[str, Any], ctx: Any) -> Any:
        ctx.episodic.record(role="reflection", content="shared-memory-write")
        return ToolResult(summary="wrote episodic", kind="execute_result", priority=0.5)

    class _FakeJudgment:
        def __init__(self) -> None:
            self._calls = 0

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.ep_write",
                    params={},
                    rationale="probe episodic write",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            episodic = EpisodicMemory(root / "episodic")
            cfg = cast(Any, SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=SimpleNamespace(ethos_baseline={
                    "truth": 0.91,
                    "caution": 0.81,
                    "continuity": 0.71,
                    "curiosity": 0.61,
                    "care": 0.51,
                }),
            ))
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=episodic,
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast(Any, SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            runner = make_subagent_runner(
                SubagentConfig(goal="检查 shared memory episodic 只读", max_ticks=2, allowed_tools=["probe.ep_write"]),
                parent_ctx,
                cast(Any, _FakeJudgment()),
                execution,
                registry,
            )

            result = await runner.run()

            assert result.completed is True
            assert result.last_summary == "wrote episodic"
            assert episodic.load_for_context(None, max_chars=4000) == ""
            assert episodic.get_recent_turns(task_id=None, limit=5) == []
        finally:
            await store.close()


def test_subagent_absorb_persists_parent_semantic_node_with_provenance():
    asyncio.run(_subagent_absorb_persists_parent_semantic_node_with_provenance())


async def _subagent_absorb_persists_parent_semantic_node_with_provenance():
    from store.semantic import SemanticMemory
    from tools.subagent_ops import subagent_absorb

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        semantic = SemanticMemory(root / "semantic")
        ctx = _tool_ctx(semantic=semantic)

        res = await subagent_absorb(
            {
                "subagent_id": "sub-a1",
                "memories_json": json.dumps([
                    {
                        "id": "note-1",
                        "kind": "learned_insight",
                        "title": "定位异常根因",
                        "body": "execution 与 semantic 接口未对齐。",
                        "activation": 0.76,
                        "valence": 0.61,
                        "importance": 0.88,
                        "tags": ["reflection", "execution"],
                        "created_at": "2026-05-22T10:00:00+00:00",
                    }
                ], ensure_ascii=False),
            },
            ctx,
        )

        assert res.error is None
        assert res.metadata["absorbed"] == 1
        assert res.metadata["requested_total"] == 1
        node = semantic.get("absorbed-sub-a1-note-1")
        assert node is not None
        assert node.title == "定位异常根因"
        assert node.body == "execution 与 semantic 接口未对齐。"
        assert node.importance == pytest.approx(0.88)
        assert node.source == "subagent:sub-a1"
        assert "reflection" in node.tags
        assert "subagent:sub-a1" in node.tags


def test_subagent_run_isolated_memory_returns_absorbable_memories_without_parent_semantic_pollution():
    asyncio.run(_subagent_run_isolated_memory_returns_absorbable_memories_without_parent_semantic_pollution())


async def _subagent_run_isolated_memory_returns_absorbable_memories_without_parent_semantic_pollution():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from store.semantic import MemoryNode, SemanticMemory
    from store.task import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool
    from tools.subagent_ops import subagent_run

    @tool(ToolManifest(
        name="probe.semantic_note",
        description="测试 isolated-memory 子灵写入独立 semantic",
        progress_category="info",
    ))
    async def _probe_semantic_note(params: dict[str, Any], ctx: Any) -> Any:
        ctx.semantic.upsert(MemoryNode(
            id="sub-note-1",
            kind="learned_insight",
            title="隔离语义吸收测试",
            body="isolated-memory 子灵应返回 absorbable memories 且不污染父灵。",
            activation=0.73,
            valence=0.58,
            importance=0.81,
            tags=["subagent", "isolated"],
            source="child-runtime",
        ))
        return ToolResult(summary="wrote isolated semantic", kind="execute_result", priority=0.5)

    class _FakeJudgment:
        def __init__(self) -> None:
            self._calls = 0

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.semantic_note",
                    params={},
                    rationale="probe isolated semantic write",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast(Any, SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12, max_events=20),
                memory_dir=root / "memory",
                emotion=SimpleNamespace(baseline_valence=0.4, baseline_arousal=0.3),
                soul=SimpleNamespace(ethos_baseline={
                    "truth": 0.91,
                    "caution": 0.81,
                    "continuity": 0.71,
                    "curiosity": 0.61,
                    "care": 0.51,
                }),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_semantic = SemanticMemory(root / "semantic")
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=cast(Any, SimpleNamespace(record=lambda *args, **kwargs: None, record_event=lambda *args, **kwargs: None)),
                semantic=parent_semantic,
                emotion=cast(Any, SimpleNamespace()),
                judgment=cast(Any, _FakeJudgment()),
                execution=execution,
                registry=registry,
            )

            res = await subagent_run(
                {
                    "goal": "隔离语义吸收",
                    "max_ticks": 2,
                    "allowed_tools": "probe.semantic_note",
                    "isolated_memory": True,
                },
                parent_ctx,
            )

            assert res.error is None
            assert res.metadata["absorbed_memories_count"] == 1
            assert res.metadata["memory_dir"]
            assert parent_semantic.get("sub-note-1") is None
            absorbed = res.metadata["absorbed_memories"]
            assert len(absorbed) == 1
            assert absorbed[0]["title"] == "隔离语义吸收测试"
            assert absorbed[0]["body"] == "isolated-memory 子灵应返回 absorbable memories 且不污染父灵。"
            assert absorbed[0]["source"] == "child-runtime"
            assert absorbed[0]["importance"] == pytest.approx(0.81)
        finally:
            await store.close()


def test_subagent_absorb_surfaces_truncation_and_invalid_nodes():
    asyncio.run(_subagent_absorb_surfaces_truncation_and_invalid_nodes())


async def _subagent_absorb_surfaces_truncation_and_invalid_nodes():
    from store.semantic import SemanticMemory
    from tools.subagent_ops import subagent_absorb

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        semantic = SemanticMemory(root / "semantic")
        ctx = _tool_ctx(semantic=semantic)

        nodes = [
            {"id": "good-1", "title": "A", "body": "alpha"},
            {"id": "good-2", "title": "B", "body": "beta"},
            {"id": "bad-3", "title": "", "body": "missing title"},
            {"id": "good-4", "title": "D", "body": "delta"},
            {"id": "good-5", "title": "E", "body": "epsilon"},
            {"id": "good-6", "title": "F", "body": "zeta"},
        ]

        res = await subagent_absorb(
            {
                "subagent_id": "sub-b2",
                "memories_json": json.dumps(nodes, ensure_ascii=False),
                "max_absorb": 5,
            },
            ctx,
        )

        assert res.error is None
        assert res.metadata["requested_total"] == 6
        assert res.metadata["selected_total"] == 5
        assert res.metadata["truncated"] == 1
        assert res.metadata["invalid"] == 1
        assert "另有 1 条因 max_absorb 未吸收" in res.summary
        assert "1 条缺少标题或正文已跳过" in res.summary
        assert semantic.get("absorbed-sub-b2-good-6") is None


def test_browser_navigate_failure_uses_stdout_when_stderr_empty(monkeypatch):
    asyncio.run(_browser_navigate_failure_uses_stdout_when_stderr_empty(monkeypatch))


async def _browser_navigate_failure_uses_stdout_when_stderr_empty(monkeypatch):
    import tools.browser as browser_mod

    monkeypatch.setattr(browser_mod, "_find_browser", lambda: "agent-browser")

    async def _fake_browser_run(*args: str, timeout: int = 30):
        return 7, "blocked by upstream gateway", ""

    monkeypatch.setattr(browser_mod, "_browser_run", _fake_browser_run)
    res = await browser_mod.browser_navigate({"url": "https://example.com"}, _tool_ctx())

    assert res.error == "NavigateTargetBlocked"
    assert "exit=7" in res.summary
    assert "blocked by upstream gateway" in res.summary


def test_browser_navigate_timeout_classified(monkeypatch):
    asyncio.run(_browser_navigate_timeout_classified(monkeypatch))


async def _browser_navigate_timeout_classified(monkeypatch):
    import tools.browser as browser_mod

    monkeypatch.setattr(browser_mod, "_find_browser", lambda: "agent-browser")

    async def _fake_browser_run(*args: str, timeout: int = 30):
        return -1, "", "操作超时"

    monkeypatch.setattr(browser_mod, "_browser_run", _fake_browser_run)
    res = await browser_mod.browser_navigate({"url": "https://example.com"}, _tool_ctx())

    assert res.error == "NavigateTimeout"
    assert "操作超时" in res.summary


def test_browser_navigate_network_unreachable_classified(monkeypatch):
    asyncio.run(_browser_navigate_network_unreachable_classified(monkeypatch))


async def _browser_navigate_network_unreachable_classified(monkeypatch):
    import tools.browser as browser_mod

    monkeypatch.setattr(browser_mod, "_find_browser", lambda: "agent-browser")

    async def _fake_browser_run(*args: str, timeout: int = 30):
        return 2, "", "net::ERR_NAME_NOT_RESOLVED"

    monkeypatch.setattr(browser_mod, "_browser_run", _fake_browser_run)
    res = await browser_mod.browser_navigate({"url": "https://example.com"}, _tool_ctx())

    assert res.error == "NavigateNetworkUnreachable"
    assert "网络不可达" in res.summary


def test_browser_navigate_dependency_missing_classified(monkeypatch):
    asyncio.run(_browser_navigate_dependency_missing_classified(monkeypatch))


async def _browser_navigate_dependency_missing_classified(monkeypatch):
    import tools.browser as browser_mod

    monkeypatch.setattr(browser_mod, "_find_browser", lambda: "agent-browser")

    async def _fake_browser_run(*args: str, timeout: int = 30):
        return 3, "", "Failed to launch browser process! libnss3.so: cannot open shared object file"

    monkeypatch.setattr(browser_mod, "_browser_run", _fake_browser_run)
    res = await browser_mod.browser_navigate({"url": "https://example.com"}, _tool_ctx())

    assert res.error == "NavigateDependencyMissing"
    assert "浏览器依赖缺失" in res.summary


def test_browser_navigate_blank_page_classified(monkeypatch):
    asyncio.run(_browser_navigate_blank_page_classified(monkeypatch))


async def _browser_navigate_blank_page_classified(monkeypatch):
    import tools.browser as browser_mod

    monkeypatch.setattr(browser_mod, "_find_browser", lambda: "agent-browser")

    async def _fake_browser_run(*args: str, timeout: int = 30):
        return 0, "   \n   ", ""

    monkeypatch.setattr(browser_mod, "_browser_run", _fake_browser_run)
    res = await browser_mod.browser_navigate({"url": "https://example.com"}, _tool_ctx())

    assert res.error == "NavigateBlankPage"
    assert "页面空白" in res.summary


def test_web_fetch_recreates_closed_shared_client(monkeypatch):
    asyncio.run(_web_fetch_recreates_closed_shared_client(monkeypatch))


async def _web_fetch_recreates_closed_shared_client(monkeypatch):
    import tools.web as web_mod

    class _ClosedClient:
        is_closed = True

    class _FakeResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.text = "<html><body>hello web</body></html>"
            self.headers = {"content-type": "text/html; charset=utf-8"}

        def raise_for_status(self) -> None:
            return None

    class _FreshClient:
        is_closed = False

        async def get(self, url: str):
            assert url == "https://example.com"
            return _FakeResponse()

    created: dict[str, Any] = {}

    def _factory(**kwargs: Any):
        created["kwargs"] = kwargs
        return _FreshClient()

    monkeypatch.setattr(web_mod.httpx, "AsyncClient", _factory)
    monkeypatch.setattr(web_mod, "_http_client", _ClosedClient())

    res = await web_mod.web_fetch({"url": "https://example.com"}, _tool_ctx())

    assert created["kwargs"]["follow_redirects"] is True
    assert res.error is None
    assert res.skipped is False
    assert "获取成功" in res.summary


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
    sid = json.loads(res.evidence)["process_id"]

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
    sid = json.loads(res.evidence)["process_id"]

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
    sid = json.loads(res2.evidence)["process_id"]
    await asyncio.sleep(0.5)  # 等待完成

    # 写入已结束进程
    w = await process_write({"session_id": sid, "data": "hello"}, ctx)
    assert w.skipped is True
    assert w.error == "ProcessFinished"


def test_exec_foreground_success_summary_is_exit_code_neutral():
    asyncio.run(_exec_foreground_success_summary_is_exit_code_neutral())


async def _exec_foreground_success_summary_is_exit_code_neutral():
    from tools.exec import exec_run

    ctx = _tool_ctx()
    res = await exec_run({"command": "printf 'payload-with-error-word'"}, ctx)

    assert res.error is None
    assert res.skipped is False
    assert res.summary.startswith("命令完成 (exit=0):")
    assert "payload-with-error-word" in res.summary


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


def test_file_edit_resolves_workspace_logical_path_for_existing_file():
    asyncio.run(_file_edit_resolves_workspace_logical_path_for_existing_file())


async def _file_edit_resolves_workspace_logical_path_for_existing_file():
    from tools.file import file_edit

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        workspace = root / ".lingzhou" / "workspace"
        workspace.mkdir(parents=True)
        target = workspace / "MEMORY.md"
        target.write_text("hello\nworld\n", encoding="utf-8")

        wrong_path = root / "root" / "lingzhou" / "MEMORY.md"
        ctx = _tool_ctx(workspace_dir=str(workspace))

        res = await file_edit(
            {"path": str(wrong_path), "edits": [{"oldText": "world", "newText": "dad"}]},
            ctx,
        )

        assert res.error is None
        assert target.read_text(encoding="utf-8") == "hello\ndad\n"
        assert not wrong_path.exists()


def test_file_write_resolves_workspace_logical_path_for_existing_file():
    asyncio.run(_file_write_resolves_workspace_logical_path_for_existing_file())


async def _file_write_resolves_workspace_logical_path_for_existing_file():
    from tools.file import file_write

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        workspace = root / ".lingzhou" / "workspace"
        workspace.mkdir(parents=True)
        target = workspace / "MEMORY.md"
        target.write_text("old\n", encoding="utf-8")

        wrong_path = root / "root" / "lingzhou" / "MEMORY.md"
        ctx = _tool_ctx(workspace_dir=str(workspace))

        res = await file_write({"path": str(wrong_path), "content": "new\n"}, ctx)

        assert res.error is None
        assert target.read_text(encoding="utf-8") == "new\n"
        assert not wrong_path.exists()


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


