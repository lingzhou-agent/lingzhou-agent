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


