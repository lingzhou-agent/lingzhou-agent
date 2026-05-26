"""TaskStore 持久化测试"""
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
# TaskStore — JSON-first
# ══════════════════════════════════════════════════════════════════════════════

def test_task_store_basic():
    asyncio.run(_task_store_basic())

async def _task_store_basic():
    from store.task import TaskStore
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
        assert failures[0].summary == "网络"  # DESC: 最新在前

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


def test_task_store_find_similar_open_tasks():
    asyncio.run(_task_store_find_similar_open_tasks())


async def _task_store_find_similar_open_tasks():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "similar.db")
        await store.open()

        open_id = await store.add_task(
            "排查远程运行重启循环",
            goal="分析 crash.log 并修复远程服务重启问题",
            source="external",
        )
        await store.add_task(
            "排查远程运行重启循环",
            goal="历史已完成任务",
            status="done",
            source="external",
        )

        hits = await store.find_similar_open_tasks("解决远程运行重启循环", limit=3)

        assert hits
        assert hits[0][0].id == open_id
        assert all(task.status != "done" for task, _ in hits)

        await store.close()


def test_task_add_reuses_similar_open_task():
    asyncio.run(_task_add_reuses_similar_open_task())


async def _task_add_reuses_similar_open_task():
    from store.task import TaskStore
    from tools.task_ops import task_add

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "task-add-reuse.db")
        await store.open()

        existing_id = await store.add_task(
            "排查远程运行重启循环",
            goal="分析 crash.log 并修复远程运行重启循环",
            source="external",
        )
        ctx = _tool_ctx(task_store=store)

        result = await task_add(
            {
                "title": "解决远程运行一直重启",
                "goal": "分析 crash.log 并修复远程运行重启循环",
            },
            ctx,
        )

        assert result.skipped is True
        assert result.metadata.get("task_id") == existing_id
        assert result.metadata.get("reused_existing_task") is True
        tasks = await store.list_tasks(limit=10)
        assert len(tasks) == 1

        await store.close()


async def _task_store_run_lifecycle():
    from store.task import TaskStore

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
    from store.task import TaskStore
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


def test_update_status_can_patch_runtime_fields_in_one_call():
    asyncio.run(_update_status_can_patch_runtime_fields_in_one_call())


async def _update_status_can_patch_runtime_fields_in_one_call():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "status-patch.db")
        await store.open()
        task_id = await store.add_task(
            "状态补丁任务",
            goal="验证单次 update_status 可同步 runtime 字段",
            next_step="旧下一步",
            current_step="旧当前步骤",
            model_tier="reader",
        )

        await store.update_status(
            task_id,
            "in_progress",
            "",
            current_step="",
            model_tier="",
        )

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
    from store.task import TaskStore
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


def test_status_transitions_can_merge_result_json_in_one_call():
    asyncio.run(_status_transitions_can_merge_result_json_in_one_call())


async def _status_transitions_can_merge_result_json_in_one_call():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "status-result.db")
        await store.open()
        task_id = await store.add_task(
            "状态结果合并",
            goal="验证 status/waiting 可顺带合并 result_json",
            result_json={"seed": "present"},
        )

        await store.update_status(task_id, "done", result_json={"summary": "ok"})
        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "done"
        assert task.result_json == {"seed": "present", "summary": "ok"}

        await store.mark_waiting(
            task_id,
            wait_kind="task",
            wait_key="parent-1",
            wait_json={"terminal_decision": "wait"},
            result_json={"terminal_decision": "wait"},
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "waiting"
        assert task.wait_key == "parent-1"
        assert task.result_json == {
            "seed": "present",
            "summary": "ok",
            "terminal_decision": "wait",
        }

        await store.close()


def test_chat_messages_are_sanitized_on_write():
    asyncio.run(_chat_messages_are_sanitized_on_write())


def test_chat_pending_messages_are_recoverable_until_processed():
    asyncio.run(_chat_pending_messages_are_recoverable_until_processed())


def test_task_wait_allows_external_wait_without_wait_key():
    asyncio.run(_task_wait_allows_external_wait_without_wait_key())


async def _task_wait_allows_external_wait_without_wait_key():
    from store.task import TaskStore
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
    from store.task import TaskStore
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


def test_task_steer_inbox_is_consumed_once():
    asyncio.run(_task_steer_inbox_is_consumed_once())


async def _task_steer_inbox_is_consumed_once():
    from store.task import TaskStore
    from tools.task_ops import task_steer

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "steer.db")
        await store.open()
        task_id = await store.add_task("用户消息 inbox 任务", goal="验证用户消息 inbox 只消费一次")

        ctx = _tool_ctx(task_store=store)
        steer_res = await task_steer(
            {
                "task_id": task_id,
                "message": "改成先读取配置文件，再决定是否继续旧计划",
            },
            ctx,
        )

        assert steer_res.error is None

        steered = await store.get_task_by_id(task_id)
        assert steered is not None
        assert steered.extras["inbox_messages"] == ["改成先读取配置文件，再决定是否继续旧计划"]

        inbox = await store.pop_task_inbox(task_id)
        assert inbox == ["改成先读取配置文件，再决定是否继续旧计划"]

        consumed = await store.get_task_by_id(task_id)
        assert consumed is not None
        assert consumed.extras.get("inbox_messages") == []

        inbox_again = await store.pop_task_inbox(task_id)
        assert inbox_again == []

        await store.close()


async def _chat_messages_are_sanitized_on_write():
    from store.task import TaskStore

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


async def _chat_pending_messages_are_recoverable_until_processed():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "chat-processing.db"
        store = TaskStore(db_path)
        await store.open()
        await store.add_chat_message("user", "hello")

        reserved = await store.pop_pending_chat_message()
        assert reserved is not None
        assert reserved["content"] == "hello"
        assert await store.has_pending_chat_message() is False

        await store.release_chat_messages([reserved["id"]])
        retried = await store.pop_pending_chat_message()
        assert retried is not None
        assert retried["id"] == reserved["id"]

        await store.close()

        reopened = TaskStore(db_path)
        await reopened.open()
        recovered = await reopened.pop_pending_chat_message()
        assert recovered is not None
        assert recovered["id"] == reserved["id"]

        await reopened.mark_chat_messages_processed([recovered["id"]])
        assert await reopened.has_pending_chat_message() is False
        await reopened.close()


async def _ingress_store_unifies_external_chat_and_task_writes():
    from store.task import TaskStore
    from store.task.ingress import IngressStore

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "ingress.db"
        store = TaskStore(db_path)
        await store.open()
        await store.close()

        ingress = IngressStore(db_path)
        msg_id = ingress.ingest_user_message(
            "\x1b[31mhi\x1b[0m\ufeff\u200b\ufffdthere",
            chat_id="wechat:user-1",
            facts={
                "wechat:ctx:user-1": "ctx-1",
                "wechat:last_user": "user-1",
            },
        )
        task_id = ingress.add_task(
            " webhook: inbound ",
            goal="来自 webhook 的外部任务",
            priority="high",
            source="gateway:webhook",
        )

        verify = TaskStore(db_path)
        await verify.open()
        messages = await verify.get_chat_messages_since(0, chat_id="wechat:user-1")
        assert len(messages) == 1
        assert messages[0]["id"] == msg_id
        assert messages[0]["content"] == "hithere"
        ctx_value, ctx_found = await verify.get_fact("wechat:ctx:user-1")
        assert ctx_found is True
        assert ctx_value == "ctx-1"
        task = await verify.get_task_by_id(task_id)
        assert task is not None
        assert task.title == "webhook: inbound"
        assert task.goal == "来自 webhook 的外部任务"
        assert task.source == "gateway:webhook"
        assistant_id = await verify.add_chat_message("assistant", "已收到", chat_id="wechat:user-1")
        await verify.close()

        outbound = ingress.list_pending_assistant_messages(chat_prefix="wechat:", limit=10)
        assert len(outbound) == 1
        assert outbound[0]["id"] == assistant_id
        assert outbound[0]["chat_id"] == "wechat:user-1"
        ingress.mark_chat_message_delivered(assistant_id)
        assert ingress.list_pending_assistant_messages(chat_prefix="wechat:", limit=10) == []


def test_task_store_migration():
    asyncio.run(_task_store_migration())


def test_ingress_store_unifies_external_chat_and_task_writes():
    asyncio.run(_ingress_store_unifies_external_chat_and_task_writes())

async def _task_store_migration():
    """旧列式 schema → JSON-first 自动迁移。"""
    from store.task import TaskStore

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


def test_task_store_migrates_legacy_person_profile_facts_to_interlocutor_scope():
    asyncio.run(_task_store_migrates_legacy_person_profile_facts_to_interlocutor_scope())


async def _task_store_migrates_legacy_person_profile_facts_to_interlocutor_scope():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "legacy-facts.db"
        async with aiosqlite.connect(str(db_path)) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT '',
                    scope TEXT NOT NULL DEFAULT 'general',
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
            """)
            await db.execute(
                "INSERT INTO facts (key, value, scope, updated_at) VALUES (?, ?, ?, datetime('now'))",
                ("chat:wechat:chat-1:person_profile_id", "person-bat", "profile"),
            )
            await db.execute(
                "INSERT INTO facts (key, value, scope, updated_at) VALUES (?, ?, ?, datetime('now'))",
                ("user:person-bat:display_name", "bat", "profile"),
            )
            await db.commit()

        store = TaskStore(db_path)
        await store.open()
        try:
            new_chat_key = await store.get_fact("chat:wechat:chat-1:interlocutor_profile_id")
            new_display_key = await store.get_fact("interlocutor:person-bat:display_name")
            old_chat_key = await store.get_fact("chat:wechat:chat-1:person_profile_id")
            old_display_key = await store.get_fact("user:person-bat:display_name")

            assert new_chat_key == ("person-bat", True)
            assert new_display_key == ("bat", True)
            assert old_chat_key == ("", False)
            assert old_display_key == ("", False)
        finally:
            await store.close()


# ══════════════════════════════════════════════════════════════════════════════
