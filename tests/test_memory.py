"""语义记忆（semantic）与情节记忆（episodic）测试"""
import asyncio
import builtins
import io
import json
import logging
import math
import os
import sqlite3
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


def test_semantic_importance_slows_decay():
    from memory.semantic import MemoryNode, effective_activation

    old_ts = (datetime.now(UTC) - timedelta(days=30)).isoformat()

    ordinary = MemoryNode(
        id="ordinary",
        kind="fact",
        title="ordinary",
        body="importlib",
        activation=0.8,
        importance=0.0,
        created_at=old_ts,
    )
    important = MemoryNode(
        id="important",
        kind="fact",
        title="important",
        body="importlib",
        activation=0.8,
        importance=0.9,
        created_at=old_ts,
    )

    ordinary_eff = effective_activation(ordinary, 0.1)
    important_eff = effective_activation(important, 0.1)

    assert important_eff > ordinary_eff
    assert important_eff >= 0.5


def test_semantic_retrieve_ranking_uses_effective_activation():
    from memory.semantic import SemanticMemory, MemoryNode

    old_ts = (datetime.now(UTC) - timedelta(days=30)).isoformat()

    ordinary = MemoryNode(
        id="ordinary-rank",
        kind="fact",
        title="python importlib",
        body="",
        activation=0.8,
        importance=0.0,
        created_at=old_ts,
    )
    important = MemoryNode(
        id="important-rank",
        kind="fact",
        title="python",
        body="",
        activation=0.8,
        importance=0.9,
        created_at=old_ts,
    )

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.1)
        sm.upsert(ordinary)
        sm.upsert(important)

        results = sm.retrieve("python reload importlib hot swap", top_k=2)

        assert results[0]["id"] == "important-rank"


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


def test_semantic_store_reflection_title_uses_unique_suffix():
    from memory.semantic import SemanticMemory

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.0)
        first_id = sm.store_reflection('baseline', '洞察A')
        second_id = sm.store_reflection('baseline', '洞察B')

        first = sm.get(first_id)
        second = sm.get(second_id)

        assert first is not None
        assert second is not None
        assert first.title.startswith('[baseline] [')
        assert second.title.startswith('[baseline] [')
        assert first.title != second.title


# ══════════════════════════════════════════════════════════════════════════════
# EpisodicMemory — search() 质量验证
# ══════════════════════════════════════════════════════════════════════════════

def test_episodic_search_finds_chinese_narrative():
    """search() 通过 FTS5 能召回中文 narrative 条目。"""
    from memory.episodic import EpisodicMemory

    with tempfile.TemporaryDirectory() as d:
        ep = EpisodicMemory(Path(d), max_events=0)
        ep.record("user", "灵舟正在阅读语义记忆模块", task_id="task-1")
        ep.record("assistant", "已完成模块分析，发现激活衰减逻辑", task_id="task-1")

        result = ep.search("阅读语义记忆模块", max_chars=500)
        assert "语义记忆" in result or "激活衰减" in result, f"FTS5 未命中，result={result!r}"


def test_episodic_search_short_ascii_not_overmatching():
    """短 ASCII 词（如 'core'）不应导致 OR 查询泛滥命中不相关条目。

    查询 "阅读 core/ 中的关键模块" 时 "core" 被过滤掉（ASCII len=4 < 5）；
    只用中文词检索，task-3（"今天天气不错"）不应被召回。
    """
    from memory.episodic import EpisodicMemory

    with tempfile.TemporaryDirectory() as d:
        ep = EpisodicMemory(Path(d), max_events=0)
        ep.record("user", "阅读 core/ 中的关键模块，理解架构", task_id="task-1")
        ep.record("user", "检查 core/config.py 文件权限", task_id="task-2")
        ep.record("user", "今天天气不错，适合散步", task_id="task-3")

        result = ep.search("阅读 core/ 中的关键模块", max_chars=2000)
        assert "关键模块" in result, "相关条目应被检索到"
        assert "散步" not in result, "不相关条目不应被召回（core 被过滤，不应 OR 泛命中）"


def test_episodic_search_cross_task_returns_different_task():
    """跨任务检索：search() 能返回来自其他任务的相关内容。"""
    from memory.episodic import EpisodicMemory

    with tempfile.TemporaryDirectory() as d:
        ep = EpisodicMemory(Path(d), max_events=0)
        ep.record("user", "深度理解记忆衰减模型 Ebbinghaus", task_id="old-task")
        ep.record("assistant", "衰减曲线已分析完毕", task_id="old-task")
        ep.record("user", "开始新任务", task_id="current-task")

        result = ep.search("记忆衰减模型 Ebbinghaus", max_chars=2000)
        assert "衰减" in result, f"应从旧任务召回相关内容，result={result!r}"


def test_episodic_search_exclude_task_id_blocks_self_echo():
    """exclude_task_id 过滤：当前任务的 narrative 不应作为跨任务命中返回。

    场景：同一目标被多个任务运行过（goal echo）；
    传入 exclude_task_id 后，旧任务中 content ≈ 查询文本的条目被过滤掉。
    """
    from memory.episodic import EpisodicMemory

    goal = "阅读 core/ 中的关键模块，理解架构和可改进点。选择你之前没细读过的文件开始。"
    with tempfile.TemporaryDirectory() as d:
        ep = EpisodicMemory(Path(d), max_events=0)
        # 旧任务写入了相同目标文本
        ep.record("user", goal, task_id="old-task-1")
        ep.record("assistant", "已读取 core/loop/runtime.py", task_id="old-task-1")
        # 当前任务写入不同内容
        ep.record("user", "继续执行下一步", task_id="cur-task")
        ep.record("assistant", "正在分析 core/evolution.py", task_id="cur-task")

        # 不传 exclude_task_id：old-task-1 的 goal echo 可能命中
        result_no_excl = ep.search(goal, max_chars=4000)

        # 传入 exclude_task_id：goal echo（content 含 goal 前 40 字符）应被过滤
        result_excl = ep.search(goal, max_chars=4000, exclude_task_id="cur-task")
        # goal 文本本身不应出现（被 _query_head 过滤）
        assert goal[:30] not in result_excl, \
            f"旧任务的目标文本回显应被过滤，实际: {result_excl!r}"


def test_episodic_record_keeps_narrative_when_fts_sync_fails():
    """FTS 同步失败时，.md 和 narrative 表仍应保持一致。"""
    from memory.episodic import EpisodicMemory

    with tempfile.TemporaryDirectory() as d:
        ep = EpisodicMemory(Path(d), max_events=0)

        def _boom(*args, **kwargs):
            raise sqlite3.OperationalError("fts broken")

        ep._sync_narrative_fts = _boom  # type: ignore[method-assign]
        ep.record("user", "记录一条需要保留的情节", task_id="task-1")

        md_text = EpisodicMemory.narrative_path_for_dir(Path(d), "task-1").read_text(encoding="utf-8")
        assert "记录一条需要保留的情节" in md_text

        rows = ep.query_recent_narrative(hours=24, limit=10)
        assert any(row["content"] == "记录一条需要保留的情节" for row in rows)

        turns = ep.get_recent_turns("task-1", limit=5)
        assert any(turn["content"] == "记录一条需要保留的情节" for turn in turns)


def test_episodic_migrates_legacy_root_narrative_files():
    """旧版根目录 task/global narrative 文件应迁移到 episodic/ 子目录。"""
    from memory.episodic import EpisodicMemory

    with tempfile.TemporaryDirectory() as d:
        memory_dir = Path(d)
        legacy_task = memory_dir / "task-legacy.md"
        legacy_global = memory_dir / "global.md"
        legacy_task.write_text("旧任务叙事", encoding="utf-8")
        legacy_global.write_text("旧全局叙事", encoding="utf-8")

        ep = EpisodicMemory(memory_dir, max_events=0)

        assert not legacy_task.exists()
        assert not legacy_global.exists()
        assert EpisodicMemory.narrative_path_for_dir(memory_dir, "legacy").read_text(encoding="utf-8") == "旧任务叙事"
        assert EpisodicMemory.narrative_path_for_dir(memory_dir, None).read_text(encoding="utf-8") == "旧全局叙事"
        assert ep.load_for_context("legacy", max_chars=200) == "旧任务叙事"
        assert "legacy" in ep.list_tasks()


# ══════════════════════════════════════════════════════════════════════════════
# SemanticMemory — retrieve() 向量路径 & retrieve_multi_anchor 向量对齐
# ══════════════════════════════════════════════════════════════════════════════

def test_semantic_retrieve_with_mock_embedding():
    """embed_fn 配置后 retrieve() 使用向量混合评分。

    mock embed_fn：含 'python' → [1,0]，否则 [0,1]；
    查询向量 [1,0] → python 节点相似度高 → 应排第一。
    """
    from memory.semantic import SemanticMemory, MemoryNode

    def _mock_embed(text: str) -> list[float]:
        return [1.0, 0.0] if "python" in text.lower() else [0.0, 1.0]

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.0, embed_fn=_mock_embed)
        sm.upsert(MemoryNode(id="py", kind="fact", title="python reload",
                             body="importlib 热加载", activation=0.5))
        sm.upsert(MemoryNode(id="sql", kind="fact", title="数据库查询",
                             body="sqlite 索引优化", activation=0.5))

        results = sm.retrieve("python importlib", top_k=2)
        assert results, "应有结果"
        assert results[0]["id"] == "py", \
            f"python 节点向量对齐，应排第一，实际: {[r['id'] for r in results]}"


def test_semantic_multi_anchor_uses_embedding_when_available():
    """retrieve_multi_anchor 有 embed_fn 时启用向量评分（修复：原实现未传 query_vec）。

    两节点内容完全相同（关键词得分相等），但 embedding 方向不同；
    embedding 对齐查询方向的节点应得分更高 → 验证向量路径生效。
    """
    from memory.semantic import SemanticMemory, MemoryNode

    # embed_fn 统一返回 [1,0]，保证 query_vec = [1,0]
    def _mock_embed(text: str) -> list[float]:
        return [1.0, 0.0]

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.0, embed_fn=_mock_embed)
        # 两节点关键词内容相同 → 关键词得分相等
        sm.upsert(MemoryNode(id="match", kind="fact", title="检索模块",
                             body="功能测试", activation=0.0))
        sm.upsert(MemoryNode(id="nomatch", kind="fact", title="检索模块",
                             body="功能测试", activation=0.0))
        # 手动覆盖 embedding：match 与 query_vec [1,0] 对齐；nomatch 垂直
        sm.set_embedding("match", [1.0, 0.0])
        sm.set_embedding("nomatch", [0.0, 1.0])

        results = sm.retrieve_multi_anchor(["检索模块 功能测试"], top_k=2)
        assert results, "应有结果"
        assert results[0]["id"] == "match", \
            f"向量对齐的节点应排第一（score 更高），实际: {[r['id'] for r in results]}"


def test_semantic_fts_short_ascii_filtered():
    """FTS5 短 ASCII 词（≤4字符）被过滤后，中文词主导检索排序。"""
    from memory.semantic import SemanticMemory, MemoryNode

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.0)
        sm.upsert(MemoryNode(id="cn", kind="fact", title="模块架构分析",
                             body="阅读 core 模块，发现架构分层清晰", activation=0.5))
        sm.upsert(MemoryNode(id="en", kind="fact", title="core loop",
                             body="core task loop 基础结构", activation=0.5))

        # 全短 ASCII → fallback 行为，至少不崩溃
        results_short = sm.retrieve("loop core task", top_k=5)
        assert isinstance(results_short, list)

        # 含中文词 → "架构" 主导，cn 节点应排第一
        results_mixed = sm.retrieve("阅读 core 模块架构", top_k=2)
        if results_mixed:
            assert results_mixed[0]["id"] == "cn", \
                f"含中文关键词的节点应排第一，实际: {[r['id'] for r in results_mixed]}"


def test_semantic_upsert_disables_fts_when_sync_fails_and_retrieval_falls_back():
    """FTS 同步失败后，不应继续依赖残缺索引。"""
    from memory.semantic import SemanticMemory, MemoryNode

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.0)

        def _boom(*args, **kwargs):
            raise sqlite3.OperationalError("fts broken")

        sm._sync_node_fts = _boom  # type: ignore[method-assign]
        sm.upsert(MemoryNode(id="node-1", kind="fact", title="模块架构分析", body="关键检索路径", activation=0.5))

        assert sm.fts5_ok is False

        results = sm.retrieve("模块架构 关键检索", top_k=3)
        assert results
        assert results[0]["id"] == "node-1"


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap 注入
# ══════════════════════════════════════════════════════════════════════════════
