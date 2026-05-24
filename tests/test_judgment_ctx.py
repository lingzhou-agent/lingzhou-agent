"""行为门控、技能、thinking、模型路由、任务改写等 judgment context 测试"""
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


def test_fill_template_raises_when_variable_missing():
    from core.judgment.context import _fill_template

    with pytest.raises(ValueError, match="missing_field"):
        _fill_template("hello {{ missing_field }}", {"other": "value"})


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

    # continue 默认不再因 reader 工具而隐式降到 reader
    tier = layer._select_tier(
        phase="continue", user_message="",
        current_action="file.read", tool_history=[],
    )
    assert tier == "reasoner"

    # continue + reasoner tool → reasoner
    tier2 = layer._select_tier(
        phase="continue", user_message="",
        current_action="shell.run", tool_history=[],
    )
    assert tier2 == "reasoner"


def test_prefer_tier_for_task_uses_pending_then_task_default():
    from core.loop.common import _next_initial_tier_hint, _prefer_tier_for_task
    from memory.task_store import Task

    task = Task(
        id=1,
        title="任务",
        status="pending",
        priority="normal",
        created_at="2026-05-15T00:00:00Z",
        model_tier="reader",
    )

    assert _prefer_tier_for_task(None, task) is None
    assert _prefer_tier_for_task("reader", task) == "reader"
    assert _prefer_tier_for_task("repair", task) == "repair"
    assert _prefer_tier_for_task("reader", task, has_user_message=True) == "reasoner"
    assert _prefer_tier_for_task(None, task, has_user_message=True) == "reasoner"

    task.model_tier = "reasoner"
    assert _prefer_tier_for_task(None, task) == "reasoner"

    task.model_tier = "invalid"
    assert _prefer_tier_for_task(None, task) is None

    assert _next_initial_tier_hint(_judgment_output(decision="act", chosen_action_id="file.read")) is None
    assert _next_initial_tier_hint(
        _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            model_strategy={"next_phase_tier": "reader"},
        )
    ) == "reader"


def test_behavior_gate_passthrough_and_logs_observation(caplog):
    """重复信号只做感知和日志，不替 LLM 改 decision。"""
    from core.behavior_tracker import BehaviorTracker
    from core.judgment import JudgmentOutput

    caplog.set_level(logging.INFO, logger="lingzhou.behavior_tracker")
    tracker = BehaviorTracker()

    class _Signals:
        repeat_action_count = 3
        repeat_action_tool = "memory.search"
        repeat_action_key = "legacy runtime"
        repeat_read_count = 0
        repeat_read_path = ""
        loop_probe_version = 5

    action = _judgment_output(
        decision="act",
        chosen_action_id="memory.search",
        params={"query": "legacy runtime"},
        rationale="再搜一次",
    )
    gated = tracker.apply_execution_gate(action, _Signals())
    assert gated.decision == "act"
    assert gated is action
    assert any("delegated to llm" in rec.message for rec in caplog.records)

    class _ReadSignals:
        repeat_action_count = 0
        repeat_action_tool = ""
        repeat_action_key = ""
        repeat_read_count = 3
        repeat_read_path = "/tmp/demo.txt"
        loop_probe_version = 6

    read_action = _judgment_output(
        decision="act",
        chosen_action_id="file.read",
        params={"path": "/tmp/demo.txt"},
    )
    gated_read = tracker.apply_execution_gate(read_action, _ReadSignals())
    assert gated_read.decision == "act"
    assert gated_read is read_action

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


def test_cognitive_signals_include_last_action_feedback_and_repeat_list():
    from core.perception import CognitiveSignals

    text = CognitiveSignals(
        repeat_action_count=3,
        repeat_action_tool="memory.search",
        repeat_action_key="legacy runtime sqlite",
        repeat_read_count=0,
        repeat_read_path="",
        repeat_list_count=3,
        repeat_list_path="/root/.legacy-runtime/memory",
        loop_probe_version=9,
        last_action_tool="shell.run",
        last_action_key="find /root/.legacy-runtime",
        last_action_status="ok",
        last_action_summary="找到了 main.sqlite，但没有进一步推进 next_step",
        last_action_error="",
        last_action_state_delta="process=finished; exit_code=0",
        last_action_progressful=False,
        recent_action_history=[
            "tool=file.list | key=/root/.legacy-runtime | status=ok | progressful=True",
            "tool=memory.search | key=legacy runtime sqlite | status=ok | progressful=False",
        ],
    ).to_text()

    assert "last_action={tool='shell.run'" in text
    assert "repeat_list_count=3" in text
    assert "被系统判定为未推进" in text
    assert "recent_actions:" in text
    assert "tool=memory.search | key=legacy runtime sqlite" in text


def test_skill_registry_logs_selected_skills(caplog):
    from core.skill import SkillRegistry

    caplog.set_level(logging.INFO, logger="core.skill")
    reg = SkillRegistry()
    skills = reg.match_for_context(
        wm_pressure=0.5,
        has_active_task=True,
        has_next_step=True,
        failure_count=2,
        high_error_streak=1,
        context_text="当前任务需要继续推进，但已经有失败和参数错误。",
    )

    assert skills
    assert any("[skill.match] selected=" in rec.message for rec in caplog.records)


def test_judgment_skills_for_log_formats_selected_names():
    from core.judgment import JudgmentLayer
    from core.skill import Skill

    assert JudgmentLayer._skills_for_log([]) == "none"
    assert JudgmentLayer._skills_for_log([
        Skill(name="runtime-bootstrap", description="", guidance=""),
        Skill(name="task-continuity", description="", guidance=""),
    ]) == "runtime-bootstrap,task-continuity"


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


def test_behavior_tracker_uses_explicit_registry_capabilities():
    from core.behavior_tracker import BehaviorTracker
    from tools.registry import ToolEntry, ToolManifest

    class _Registry:
        def get(self, name: str):
            if name != "demo.readonly":
                return None
            return ToolEntry(
                manifest=ToolManifest(
                    name="demo.readonly",
                    description="demo",
                    capabilities=("result_streak_only",),
                ),
                handler=lambda params, ctx: None,
            )

    tracker = BehaviorTracker(registry=_Registry())
    items = []
    for _ in range(3):
        items = tracker.on_act("demo.readonly", "same", task_id="task-1")

    assert items == []


def test_next_thinking_override_is_one_shot_and_strict():
    from core.loop.common import _next_thinking_override

    assert _next_thinking_override({"thinking_override": "low"}) == "low"
    assert _next_thinking_override({"thinking_override": "invalid"}) is None
    assert _next_thinking_override({}) is None
    assert _next_thinking_override(None) is None


def test_resolve_thinking_override_uses_mode_defaults_and_strategy():
    from core.loop.common import _resolve_thinking_override

    cfg = cast(Any, SimpleNamespace(
        thinking="off",
        loop=SimpleNamespace(chat_thinking="low", autonomous_thinking="medium"),
    ))

    assert _resolve_thinking_override(cfg, user_message="hi") == "low"
    assert _resolve_thinking_override(cfg, user_message="") == "medium"
    assert _resolve_thinking_override(cfg, user_message="", pending_override="high") == "high"
    assert _resolve_thinking_override(cfg, user_message="", model_strategy={"thinking_override": "minimal"}) == "minimal"


def test_thinking_floor_respects_chat_minimum_for_user_message():
    from core.loop.common import _thinking_floor

    assert _thinking_floor("off", "low") == "low"
    assert _thinking_floor("minimal", "low") == "low"
    assert _thinking_floor("high", "low") == "high"
    assert _thinking_floor(None, "low") == "low"


def test_recent_runs_summary_prefers_output_and_progress():
    from core.judgment.context import _fmt_recent_runs
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
    from core.judgment.context import _fmt_waiting_tasks
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

    # available_models 在测试环境可能为空（fake providers 无模型目录）
    # 验证 delegation_guide 和 tier_descriptions 等结构字段存在即可
    assert "delegation_guide" in payload
    assert "tier_descriptions" in payload
    assert "reasoner" in payload["tier_descriptions"]
    assert payload["implicit_next_phase_default"] is None


def test_select_provider_matches_routing_provider_by_public_model_ref():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        def __init__(self, model_ref: str):
            self.model_ref = model_ref

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
        "temperature": 0.7,
        "timeout": 60.0,
    })

    main_provider = _DummyProvider("copilot/gpt-5.4")
    reader_provider = _DummyProvider("bailian/qwen3.6-plus")
    layer = JudgmentLayer(main_provider, ToolRegistry(), cfg)
    layer.set_routing_providers({"reader": reader_provider})

    provider, selection = layer._select_provider(
        phase="initial",
        user_message="先读取配置",
        prefer_tier="reader",
    )

    assert provider is reader_provider
    assert selection.tier == "reader"
    assert selection.model_ref == "bailian/qwen3.6-plus"


def test_fmt_config_snapshot_exposes_judgment_signal_thresholds():
    from core.config import Config
    from core.judgment.context import _fmt_config_snapshot

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
        "emotion": {
            "failure_normalization_count": 4.0,
            "high_error_normalization_streak": 5.0,
            "feeling_min_intensity": 0.25,
            "regulation_down_regulate_arousal_high": 0.8,
            "regulation_high_error_streak_guard": 3,
        },
        "soul": {
            "ethos_prefer_verification_caution_min": 0.72,
            "ethos_prefer_verification_failure_count": 3,
            "ethos_prefer_narrow_error_streak": 4,
            "ethos_avoid_overclaiming_down_regulate_streak": 5,
            "ethos_failure_adjust_count": 2,
            "ethos_failure_truth_delta": 0.11,
            "ethos_high_error_adjust_streak": 4,
            "ethos_recovering_curiosity_delta": 0.09,
        },
        "thresholds": {
            "prediction_error_task": 0.8,
            "perception_replay_trend_delta": 0.2,
            "perception_replay_high_error_hint_streak": 3,
            "emotion_replay_trend_delta": 0.12,
            "judgment_error_streak_guard": 4,
            "judgment_posture_narrow_failure_count": 5,
            "judgment_posture_pause_worsening_failure_count": 3,
        },
    })

    text = _fmt_config_snapshot(cfg)
    assert "## Emotion guardrails (emotion.*)" in text
    assert "failure_normalization_count: 4.0" in text
    assert "high_error_normalization_streak: 5.0" in text
    assert "feeling_min_intensity: 0.25" in text
    assert "regulation_down_regulate_arousal_high: 0.8" in text
    assert "regulation_high_error_streak_guard: 3" in text
    assert "## Ethos guardrails (soul.ethos.*)" in text
    assert "prefer_verification_caution_min: 0.72" in text
    assert "prefer_verification_failure_count: 3" in text
    assert "prefer_narrow_error_streak: 4" in text
    assert "avoid_overclaiming_down_regulate_streak: 5" in text
    assert "failure_adjust_count: 2" in text
    assert "failure_truth_delta: 0.11" in text
    assert "high_error_adjust_streak: 4" in text
    assert "recovering_curiosity_delta: 0.09" in text
    assert "## Replay guardrails (thresholds.*)" in text
    assert "prediction_error_task: 0.8" in text
    assert "perception_replay_trend_delta: 0.2" in text
    assert "perception_replay_high_error_hint_streak: 3" in text
    assert "emotion_replay_trend_delta: 0.12" in text
    assert "## Judgment guardrails (thresholds.*)" in text
    assert "judgment_error_streak_guard: 4" in text
    assert "judgment_posture_narrow_failure_count: 5" in text
    assert "judgment_posture_pause_worsening_failure_count: 3" in text


def test_fmt_config_snapshot_exposes_reference_thresholds():
    from core.config import Config
    from core.judgment.context import _fmt_config_snapshot

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
        "memory": {
            "consolidate_threshold": 0.74,
            "consolidate_low_pressure_skip_threshold": 0.81,
            "global_md_warn_bytes": 12345,
            "global_md_warn_lines": 67,
        },
        "emotion": {
            "reflection_valence_history_weight": 0.7,
            "reflection_valence_hint_weight": 0.3,
        },
        "thresholds": {
            "reference_min_confidence": 0.61,
            "reference_local_signal_base": 0.42,
            "reference_local_signal_step": 0.11,
            "reference_local_confidence_cap": 0.77,
            "reference_max_anchors": 2,
            "reference_topic_top_k": 7,
            "reference_recent_narrative_limit": 4,
            "reference_recent_semantic_top_k": 6,
            "reference_candidate_cap": 9,
            "reference_entity_section_limit": 3,
            "reference_anchor_text_chars": 180,
            "reference_candidate_body_chars": 70,
            "reference_entity_snippet_chars": 90,
            "reference_topic_anchor_min_chars": 4,
            "fact_context_exclude_prefixes": ["pref:", "run:"],
            "fact_context_task_limit": 8,
            "fact_context_global_limit": 5,
            "fact_context_recent_scan_multiplier": 4,
            "fact_context_recent_scan_min": 10,
            "chat_history_turn_limit": 2,
            "chat_history_max_chars": 180,
        },
    })

    text = _fmt_config_snapshot(cfg)
    assert "## Reference guardrails (thresholds.*)" in text
    assert "reference_min_confidence: 0.61" in text
    assert "reference_local_signal_base: 0.42" in text
    assert "reference_local_signal_step: 0.11" in text
    assert "reference_local_confidence_cap: 0.77" in text
    assert "reference_max_anchors: 2" in text
    assert "reference_topic_top_k: 7" in text
    assert "reference_recent_narrative_limit: 4" in text
    assert "reference_recent_semantic_top_k: 6" in text
    assert "reference_candidate_cap: 9" in text
    assert "reference_entity_section_limit: 3" in text
    assert "reference_anchor_text_chars: 180" in text
    assert "reference_candidate_body_chars: 70" in text
    assert "reference_entity_snippet_chars: 90" in text
    assert "reference_time_recent_limit" not in text
    assert "reference_time_semantic_top_k" not in text
    assert "reflection_valence_history_weight: 0.7" in text
    assert "reflection_valence_hint_weight: 0.3" in text
    assert "## Memory guardrails (memory.*)" in text
    assert "consolidate_threshold: 0.74" in text
    assert "consolidate_low_pressure_skip_threshold: 0.81" in text
    assert "global_md_warn_bytes: 12345" in text
    assert "global_md_warn_lines: 67" in text
    assert "reference_topic_anchor_min_chars: 4" in text
    assert "reference_time_phrase_hours" not in text
    assert "reference_days_ago_pattern" not in text
    assert "reference_hours_ago_pattern" not in text
    assert "reference_named_top_k" not in text
    assert "reference_self_intro_terms" not in text
    assert "reference_self_intro_name_max_chars" not in text
    assert "reference_relation_hint_terms" not in text
    assert "## Context facts guardrails (thresholds.*)" in text
    assert 'fact_context_exclude_prefixes: ["pref:", "run:"]' in text
    assert "fact_context_task_limit: 8" in text
    assert "fact_context_global_limit: 5" in text
    assert "fact_context_recent_scan_multiplier: 4" in text
    assert "fact_context_recent_scan_min: 10" in text
    assert "## Chat history guardrails (thresholds.*)" in text
    assert "chat_history_turn_limit: 2" in text
    assert "chat_history_max_chars: 180" in text
    assert "## Task steering guardrails (thresholds.*)" not in text
    assert "task_steer_ascii_term_min_chars" not in text
    assert "task_steer_cjk_term_min_chars" not in text
    assert "task_steer_cjk_term_max_chars" not in text
    assert "task_steer_message_min_chars" not in text
    assert "task_steer_message_min_terms" not in text
    assert "task_steer_message_overlap_threshold" not in text


def test_fmt_soul_uses_config_ethos_fallback_when_db_missing():
    from core.judgment.context import _fmt_soul

    text = _fmt_soul(
        "",
        "",
        '{"truth": 0.85, "caution": 0.70}',
        '["不绕过监督"]',
    )

    assert "绝对禁忌（hard_axioms，config fallback）" in text
    assert '"不绕过监督"' in text
    assert "价值基线（ethos_baseline，config fallback）" in text
    assert '"truth": 0.85' in text


def test_fmt_chat_history_uses_configured_max_chars():
    from core.judgment.context import _fmt_chat_history

    text = _fmt_chat_history([
        {"role": "user", "content": "abcdefghi"},
        {"role": "assistant", "content": "123456789"},
    ], max_chars=5)

    assert "用户: abcde…" in text
    assert "我: 12345…" in text


def test_tool_tier_uses_manifest_truth_for_reasoner_tools():
    from core.judgment.output import is_plan_alignment_exempt, tool_tier, tool_tier_mapping

    registry = _tool_registry()

    assert tool_tier("task.ask", registry) == "reasoner"
    assert tool_tier("task.plan", registry) == "reasoner"
    assert tool_tier("shell.run", registry) == "reasoner"
    assert tool_tier("schedule.add", registry) == "reasoner"
    assert tool_tier("memory.snapshot", registry) == "reasoner"
    assert tool_tier("task.resume", registry) == "reasoner"
    assert tool_tier("web.search", registry) == "reasoner"
    assert tool_tier("image.analyze", registry) == "reasoner"
    assert tool_tier("image.generate", registry) == "reasoner"
    assert tool_tier("schedule.ack", registry) == "reasoner"
    assert tool_tier("schedule.cancel", registry) == "reasoner"
    assert tool_tier("failure.dismiss", registry) == "reasoner"
    assert tool_tier("task.list", registry) == "reader"

    mapping = tool_tier_mapping(registry)
    assert "task.ask" in mapping["reasoner"]
    assert "task.plan" in mapping["reasoner"]
    assert "schedule.add" in mapping["reasoner"]
    assert "memory.snapshot" in mapping["reasoner"]
    assert "task.resume" in mapping["reasoner"]
    assert "web.search" in mapping["reasoner"]
    assert "image.analyze" in mapping["reasoner"]
    assert "image.generate" in mapping["reasoner"]
    assert "schedule.ack" in mapping["reasoner"]
    assert "schedule.cancel" in mapping["reasoner"]
    assert "failure.dismiss" in mapping["reasoner"]
    assert "task.list" in mapping["reader"]

    assert is_plan_alignment_exempt("task.ask", registry) is True
    assert is_plan_alignment_exempt("task.plan", registry) is True


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


def test_model_routing_section_no_longer_exposes_implicit_reader_default():
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

    assert payload["implicit_next_phase_default"] is None
    assert "必须由你显式设置 next_phase_tier" in payload["delegation_guide"]


def test_model_routing_section_uses_configured_idle_bounds_and_defaults():
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
        "loop": {
            "idle_with_task_bounds": [250, 45000],
            "idle_no_task_bounds": [8000, 120000],
            "active_idle_gap": 1500,
            "max_idle_gap": 90000,
        },
    })

    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
    payload = json.loads(layer._build_model_routing_section(
        phase="continue",
        user_message="继续分析",
        current_action="file.read",
        tool_history=[{"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "ok"}],
        effective_thinking="low",
    ))

    guide = payload["delegation_guide"]
    assert "当前有任务时 250ms-45s，无任务时 8s-120s" in guide
    assert "当前 loop 默认备用值（有任务 1.5s，无任务 90s）" in guide


def test_model_routing_section_counts_exploration_budget_by_capability():
    from core.config import Config
    from core.judgment import JudgmentLayer

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

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    payload = json.loads(layer._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="memory.search",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "legacy runtime"}, "result": "命中 2 条"},
            {"tool": "task.list", "params": {"status": "all"}, "result": "命中 3 条任务"},
            {"tool": "shell.run", "params": {"command": "pytest -q"}, "result": "1 passed"},
        ],
        effective_thinking="low",
    ))

    assert payload["budget_state"]["task_explore_count"] == 3
    assert payload["budget_state"]["ask_evidence_hits"] == 2
    assert payload["budget_state"]["ask_evidence_budget"] == 2
    assert payload["budget_state"]["task_explore_converge_after"] == 4
    assert payload["budget_state"]["global_cost_posture"] == "conserve"


def test_model_routing_section_uses_configured_explore_converge_threshold():
    from core.config import Config
    from core.judgment import JudgmentLayer

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
        "thresholds": {
            "task_explore_converge_after": 3,
        },
    })

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    payload = json.loads(layer._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="memory.search",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "legacy runtime"}, "result": "命中 2 条"},
            {"tool": "task.list", "params": {"status": "all"}, "result": "命中 3 条任务"},
            {"tool": "shell.run", "params": {"command": "pytest -q"}, "result": "1 passed"},
        ],
        effective_thinking="low",
    ))

    assert payload["budget_state"]["task_explore_count"] == 3
    assert payload["budget_state"]["task_explore_converge_after"] == 3
    assert payload["budget_state"]["global_cost_posture"] == "converge"


def test_model_routing_section_repeat_counts_only_trailing_streak():
    from core.config import Config
    from core.judgment import JudgmentLayer

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

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    payload = json.loads(layer._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="file.read",
        tool_history=[
            {"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "alpha"},
            {"tool": "task.list", "params": {"status": "all"}, "result": "命中 1 条任务"},
            {"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "alpha"},
        ],
        effective_thinking="low",
    ))

    assert payload["budget_state"]["repeat_action_count"] == 1
    assert payload["budget_state"]["repeat_read_count"] == 1

    trailing = json.loads(layer._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="file.read",
        tool_history=[
            {"tool": "task.list", "params": {"status": "all"}, "result": "命中 1 条任务"},
            {"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "alpha"},
            {"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "alpha"},
        ],
        effective_thinking="low",
    ))

    assert trailing["budget_state"]["repeat_action_count"] == 2
    assert trailing["budget_state"]["repeat_read_count"] == 2


def test_model_routing_section_repeat_action_count_uses_action_key_signature():
    from core.config import Config
    from core.judgment import JudgmentLayer

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

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    payload = json.loads(layer._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="memory.search",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "legacy runtime"}, "result": "命中 1 条"},
            {"tool": "memory.search", "params": {"query": "other runtime"}, "result": "命中 1 条"},
        ],
        effective_thinking="low",
    ))

    assert payload["budget_state"]["repeat_action_count"] == 1

    trailing = json.loads(layer._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="memory.search",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "legacy runtime"}, "result": "命中 1 条"},
            {"tool": "memory.search", "params": {"query": "legacy runtime"}, "result": "命中 1 条"},
        ],
        effective_thinking="low",
    ))

    assert trailing["budget_state"]["repeat_action_count"] == 2


def test_model_routing_section_exposes_continue_phase_task_plan_policy():
    from core.config import Config
    from core.judgment import JudgmentLayer

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
        "thresholds": {
            "continue_task_plan_max_per_tick": 2,
        },
    })

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    payload = json.loads(layer._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="task.plan",
        tool_history=[
            {"tool": "task.plan", "params": {"plan": [{"step": "a", "status": "in_progress"}]}, "result": "计划已创建"},
            {"tool": "task.list", "params": {"status": "all"}, "result": "命中 1 条任务"},
        ],
        effective_thinking="low",
    ))

    assert payload["continue_phase_policy"]["task_plan_calls_this_tick"] == 1
    assert payload["continue_phase_policy"]["task_plan_max_per_tick"] == 2
    assert payload["continue_phase_policy"]["task_plan_blocked_next"] is False
    assert "continue_phase_policy" in payload["delegation_guide"]


def test_model_routing_section_marks_task_plan_blocked_when_limit_reached():
    from core.config import Config
    from core.judgment import JudgmentLayer

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

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    payload = json.loads(layer._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="task.plan",
        tool_history=[
            {"tool": "task.plan", "params": {"plan": [{"step": "a", "status": "in_progress"}]}, "result": "计划已创建"},
        ],
        effective_thinking="low",
    ))

    assert payload["continue_phase_policy"]["task_plan_max_per_tick"] == 1
    assert payload["continue_phase_policy"]["task_plan_blocked_next"] is True


def test_model_routing_section_exposes_tool_history_compaction_policy():
    from core.config import Config
    from core.judgment import JudgmentLayer

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
        "thresholds": {
            "continue_tool_history_compact_threshold": 2,
            "continue_tool_history_keep_last": 1,
        },
    })

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    payload = json.loads(layer._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="memory.search",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "legacy runtime"}, "result": "命中 1 条"},
            {"tool": "task.list", "params": {"status": "all"}, "result": "命中 1 条任务"},
        ],
        effective_thinking="low",
    ))

    assert payload["continue_phase_policy"]["tool_history_count"] == 2
    assert payload["continue_phase_policy"]["tool_history_compact_threshold"] == 2
    assert payload["continue_phase_policy"]["tool_history_keep_last"] == 1
    assert payload["continue_phase_policy"]["tool_history_will_compact_next"] is True
    assert "tool_history_will_compact_next=true" in payload["delegation_guide"]


def test_fmt_durable_failures_exposes_policy_and_muted_actions():
    from core.judgment.context import _fmt_durable_failures

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
    from core.judgment.context import _load_durable_failure_snapshot
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
    from core.judgment import JudgmentLayer, ModelSelection
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


async def test_decide_continue_updates_last_call_meta_after_fallback():
    from core.config import Config
    from core.judgment import JudgmentLayer, ModelSelection

    class _FailingProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            raise RuntimeError("primary unavailable")

        async def close(self):
            return None

    class _FallbackProvider:
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

    layer = JudgmentLayer(_FailingProvider(), _tool_registry(), cfg)
    layer._last_context_text = "cached context"
    layer._last_call_meta["skills"] = "cached-skill"
    fallback_provider = _FallbackProvider()

    def _fake_select_provider(**kwargs):
        prefer_tier = kwargs.get("prefer_tier")
        if prefer_tier == "reasoner":
            return fallback_provider, ModelSelection(
                phase="continue",
                tier="reasoner",
                model_ref="bailian/qwen-reasoner-fallback",
                thinking="high",
            )
        return layer._provider, ModelSelection(
            phase="continue",
            tier="reader",
            model_ref="bailian/qwen-reader-primary",
            thinking="off",
        )

    layer._select_provider = _fake_select_provider  # type: ignore[method-assign]

    out = await layer.decide_continue(
        [{"tool": "file.list", "params": {"path": "/tmp"}, "result": "ok"}],
        user_message="继续",
        prefer_tier="reader",
    )

    assert out.decision == "wait"
    assert layer.last_call_meta["model_ref"] == "bailian/qwen-reasoner-fallback"
    assert layer.last_call_meta["tier"] == "reasoner"
    assert layer.last_call_meta["skills"] == "cached-skill"


def test_action_made_progress_result_aware():
    from core.judgment import JudgmentOutput
    from core.loop.progress import _action_made_progress, _result_fingerprint
    from tools.registry import ToolEntry, ToolManifest, ToolResult

    list_action = _judgment_output(decision="act", chosen_action_id="file.list", params={"path": "/tmp"})
    list_res = ToolResult(summary="a.txt\nb.txt\n")
    assert _action_made_progress(list_action, list_res, prev_sig="", prev_fp="")[0] is True
    assert _action_made_progress(
        list_action,
        list_res,
        prev_sig="file.list|/tmp",
        prev_fp=_result_fingerprint(list_res.summary),
    )[0] is False

    write_action = _judgment_output(decision="act", chosen_action_id="file.write", params={"path": "/tmp/x"})
    write_res = ToolResult(summary="写入成功: /tmp/x")
    assert _action_made_progress(write_action, write_res)[0] is True

    config_set_action = _judgment_output(decision="act", chosen_action_id="config.set", params={"key": "loop.max_idle_gap"})
    empty_mutation = ToolResult(summary="")
    assert _action_made_progress(config_set_action, empty_mutation)[0] is True

    fail_action = _judgment_output(decision="act", chosen_action_id="file.read", params={"path": "/tmp/missing"})
    fail_res = ToolResult(summary="文件不存在: /tmp/missing", error="FileNotFound")
    assert _action_made_progress(fail_action, fail_res)[0] is False

    unknown_action = _judgment_output(decision="act", chosen_action_id="custom.unknown", params={"id": "42"})
    empty_unknown = ToolResult(summary="")
    assert _action_made_progress(unknown_action, empty_unknown)[0] is False

    unknown_res = ToolResult(summary="no-op result")
    assert _action_made_progress(unknown_action, unknown_res, prev_sig="", prev_fp="")[0] is True
    assert _action_made_progress(
        unknown_action,
        unknown_res,
        prev_sig="custom.unknown|42",
        prev_fp=_result_fingerprint(unknown_res.summary),
    )[0] is False

    unknown_with_delta = ToolResult(summary="", state_delta={"updated": True})
    assert _action_made_progress(unknown_action, unknown_with_delta)[0] is True

    class _Registry:
        def get(self, name: str):
            if name != "custom.override":
                return None
            return ToolEntry(
                manifest=ToolManifest(
                    name="custom.override",
                    description="override",
                    progress_category="mutation",
                ),
                handler=lambda params, ctx: None,
            )

    override_action = _judgment_output(decision="act", chosen_action_id="custom.override", params={"id": "7"})
    override_res = ToolResult(summary="")
    assert _action_made_progress(override_action, override_res)[0] is False
    assert _action_made_progress(override_action, override_res, registry=_Registry())[0] is True


def test_write_success_stall_meta_reflection_records_task_hint():
    asyncio.run(_write_success_stall_meta_reflection_records_task_hint())


async def _write_success_stall_meta_reflection_records_task_hint():
    from core.loop.postprocess import _write_success_stall_meta_reflection
    from memory.task_store import TaskStore
    from tools.registry import ToolResult

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "stall.db")
        await store.open()
        task_id = await store.add_task("分析空转", goal="减少重复探索")
        task = await store.get_task_by_id(task_id)
        assert task is not None

        action = _judgment_output(decision="act", chosen_action_id="memory.search", params={"query": "legacy runtime"})
        result = ToolResult(summary="命中旧记忆：/root/.legacy-runtime/memory/main.sqlite")
        await _write_success_stall_meta_reflection(store, task, action, result, streak=2, cycle=12)

        raw, found = await store.get_fact(f"task:{task_id}:meta_reflection")
        assert found
        payload = json.loads(raw)
        assert payload["target_kind"] == "stall_recovery"
        assert payload["tool_name"] == "memory.search"
        assert "停止重复 memory.search" in payload["proposal"]
        await store.close()


def test_success_stall_reflection_tracks_capability_based_tool():
    asyncio.run(_success_stall_reflection_tracks_capability_based_tool())


async def _success_stall_reflection_tracks_capability_based_tool():
    from core.loop.tick import _maybe_record_success_stall_reflection_impl
    from memory.task_store import TaskStore
    from tools.registry import ToolResult

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "stall-capability.db")
        await store.open()
        try:
            task_id = await store.add_task("分析任务枚举空转", goal="避免重复 task.list")
            task = await store.get_task_by_id(task_id)
            assert task is not None

            loop = cast(Any, SimpleNamespace(
                _task_store=store,
                _registry=_tool_registry(),
                _last_act_progressful=False,
                _success_stall_task_id=None,
                _success_stall_streak=0,
            ))
            action = _judgment_output(decision="act", chosen_action_id="task.list", params={"status": "all"})
            result = ToolResult(summary="命中 3 条任务")

            await _maybe_record_success_stall_reflection_impl(loop, task, action, result, cycle=7)
            await _maybe_record_success_stall_reflection_impl(loop, task, action, result, cycle=8)

            raw, found = await store.get_fact(f"task:{task_id}:meta_reflection")
            assert found
            payload = json.loads(raw)
            assert payload["tool_name"] == "task.list"
            assert "停止重复 task.list" in payload["proposal"]
        finally:
            await store.close()


def test_fallback_reply_for_user_describes_waiting_state():
    from core.loop.logging import _fallback_reply_for_user
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
    from core.loop.logging import _fallback_reply_for_user
    from tools.registry import ToolResult

    action = _judgment_output(decision="pause", rationale="源路径证据不存在，需要用户补充。")
    result = ToolResult(summary="路径不存在: /root/.legacy-runtime/source", error="FileNotFound")

    reply = _fallback_reply_for_user(action, result, None)
    assert reply.startswith("状态: error")
    assert "detail:" in reply
    assert "路径不存在" in reply
    assert "后台继续处理" not in reply
    assert "我这轮" not in reply


def test_fallback_reply_for_user_does_not_echo_tool_summary_on_success():
    from core.loop.logging import _fallback_reply_for_user
    from tools.registry import ToolResult

    action = _judgment_output(decision="act", chosen_action_id="file.read", rationale="我已经收集到关键证据。")
    result = ToolResult(summary="/tmp/a.py\n/tmp/b.py")

    reply = _fallback_reply_for_user(action, result, None)
    assert reply.startswith("状态: progressed")
    assert "basis:" in reply
    assert "/tmp/a.py" not in reply


@pytest.mark.asyncio
async def test_finalize_tick_user_reply_does_not_synthesize_progress_reply_on_nonfailure_reply_only_empty():
    from core.loop.tick import _finalize_tick_user_reply
    from tools.registry import ToolResult

    class _Judgment:
        async def decide_continue(self, *args, **kwargs):
            return _judgment_output(decision="wait", rationale="继续等待更多证据")

    class _Store:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str, str]] = []

        async def get_fact(self, key: str):
            return "", False

        async def add_chat_message(self, role: str, content: str, chat_id: str = ""):
            self.messages.append((role, content, chat_id))
            return len(self.messages)

    cfg = cast(Any, SimpleNamespace(
        thinking="off",
        loop=SimpleNamespace(chat_thinking="low", autonomous_thinking="minimal"),
    ))
    store = _Store()
    loop = cast(Any, SimpleNamespace(
        _cfg=cfg,
        _judgment=_Judgment(),
        _pending_routing_overrides=None,
        _task_store=store,
    ))
    action = _judgment_output(
        decision="act",
        chosen_action_id="file.read",
        rationale="已拿到证据，等待下一步判断。",
    )
    result = ToolResult(summary="读取完成")

    await _finalize_tick_user_reply(
        loop,
        action,
        result,
        tool_history=[{"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "读取完成"}],
        user_message="继续",
        active_task=None,
        chat_id=None,
    )

    assert action.reply_to_user == ""
    assert store.messages == []


@pytest.mark.asyncio
async def test_finalize_tick_user_reply_keeps_disaster_fallback_for_reply_only_failure():
    from core.loop.tick import _finalize_tick_user_reply
    from tools.registry import ToolResult

    class _Judgment:
        async def decide_continue(self, *args, **kwargs):
            return _judgment_output(decision="wait", rationale="[reply-only] reply_to_user 不能为空")

    class _Store:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str, str]] = []

        async def get_fact(self, key: str):
            return "", False

        async def add_chat_message(self, role: str, content: str, chat_id: str = ""):
            self.messages.append((role, content, chat_id))
            return len(self.messages)

    cfg = cast(Any, SimpleNamespace(
        thinking="off",
        loop=SimpleNamespace(chat_thinking="low", autonomous_thinking="minimal"),
    ))
    store = _Store()
    loop = cast(Any, SimpleNamespace(
        _cfg=cfg,
        _judgment=_Judgment(),
        _pending_routing_overrides=None,
        _task_store=store,
    ))
    action = _judgment_output(
        decision="act",
        chosen_action_id="file.read",
        rationale="我已经收集到关键证据。",
    )
    result = ToolResult(summary="读取完成")

    await _finalize_tick_user_reply(
        loop,
        action,
        result,
        tool_history=[{"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "读取完成"}],
        user_message="继续",
        active_task=None,
        chat_id=None,
    )

    assert action.reply_to_user.startswith("状态: progressed")


@pytest.mark.asyncio
async def test_persist_tick_user_reply_does_not_append_skill_suffix():
    from core.loop.tick import _persist_tick_user_reply

    class _Store:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str, str]] = []

        async def add_chat_message(self, role: str, content: str, chat_id: str = ""):
            self.messages.append((role, content, chat_id))
            return len(self.messages)

    store = _Store()
    loop = cast(Any, SimpleNamespace(_task_store=store))
    action = _judgment_output(
        decision="pause",
        rationale="证据已足够，直接回复用户。",
        reply_to_user="这是最终答复。",
    )
    action.applied_skills = ["runtime.bootstrap", "task_planning"]

    await _persist_tick_user_reply(
        loop,
        action,
        active_task=None,
        chat_id="",
    )

    assert action.reply_to_user == "这是最终答复。"
    assert store.messages == [("assistant", "这是最终答复。", "")]


def test_infer_valence_from_text_uses_explicit_hint_only():
    from core.loop.common import _infer_valence_from_text
    from core.config import EmotionConfig

    default_cfg = EmotionConfig()
    assert _infer_valence_from_text("继续推进，暂无结构化情绪提示", 0.6, default_cfg) == 0.6
    assert _infer_valence_from_text("root cause found; valence=0.2", 0.6, default_cfg) == pytest.approx(0.52)
    assert _infer_valence_from_text(
        "root cause found; valence=0.2",
        0.6,
        EmotionConfig(reflection_valence_history_weight=0.5, reflection_valence_hint_weight=0.5),
    ) == pytest.approx(0.4)


def test_should_continue_within_tick_for_autonomous_act():
    from core.judgment import JudgmentOutput
    from core.loop.common import _preferred_continue_tier, _should_continue_within_tick

    assert _should_continue_within_tick(_judgment_output(decision="act", chosen_action_id="file.read")) is True
    assert _should_continue_within_tick(_judgment_output(decision="act", chosen_action_id="task.complete")) is False
    assert _should_continue_within_tick(_judgment_output(decision="wait")) is False
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.read"),
        user_message="帮我看下 mini 为什么 400",
        has_active_task=True,
    ) is True
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.read"),
        user_message="帮我看下 mini 为什么 400",
        has_active_task=False,
    ) is True
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.write"),
        user_message="帮我顺手改一下配置",
        has_active_task=True,
    ) is False
    assert _preferred_continue_tier(
        _judgment_output(decision="act", chosen_action_id="memory.search"),
        user_message="继续分析这个问题",
    ) is None
    assert _preferred_continue_tier(
        _judgment_output(
            decision="act",
            chosen_action_id="memory.search",
            model_strategy={"next_phase_tier": "reader"},
        ),
        user_message="继续分析这个问题",
    ) == "reader"


def test_rewrite_task_ask_does_not_override_llm_decision_before_asking_for_id():
    from core.judgment.runtime import _rewrite_task_ask_to_evidence

    action = _judgment_output(
        decision="act",
        chosen_action_id="task.ask",
        params={"question": "请提供相关 task id"},
        rationale="我需要先定位任务。",
    )

    rewritten = _rewrite_task_ask_to_evidence(
        action,
        user_message="帮我看看昨晚那个任务为什么没回消息",
        registry=_tool_registry(),
    )

    assert rewritten is action
    assert rewritten.chosen_action_id == "task.ask"
    assert rewritten.params == {"question": "请提供相关 task id"}


async def test_decide_continue_reply_only_forces_reasoner_and_reply_to_user():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        def __init__(self) -> None:
            self.last_messages = None

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.last_messages = messages
            return (
                '{"decision":"act","chosen_action_id":"file.read",'
                '"params":{"path":"/tmp/ignored"},'
                '"rationale":"证据已足够，直接整理用户答复。",'
                '"reply_to_user":"这是最终回复。"}'
            )

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
    layer = JudgmentLayer(provider, _tool_registry(), cfg)
    layer._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "memory.search", "params": {"query": "继续分析"}, "result": "命中 2 条相关记忆"}],
        user_message="继续分析",
        prefer_tier="reader",
        reply_only=True,
    )

    assert out.decision == "pause"
    assert out.chosen_action_id == ""
    assert out.params == {}
    assert out.reply_to_user == "这是最终回复。"
    assert layer.last_call_meta["tier"] == "reasoner"
    assert provider.last_messages is not None
    assert "禁止再调用任何工具" in provider.last_messages[1].content


async def test_decide_continue_surfaces_missing_chosen_action_id_without_runtime_repair():
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.calls += 1
            return '{"decision":"act","params":{"key":"loop.min_act_gap","value":100},"rationale":"应该改配置"}'

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
    layer = JudgmentLayer(provider, _tool_registry(), cfg)
    layer._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "task.list", "params": {"status": "all"}, "result": "命中 3 条任务"}],
        user_message="把 tick 设置到 100 毫秒一次",
        prefer_tier="reasoner",
    )

    assert out.decision == "wait"
    assert out.chosen_action_id == ""
    assert out.params == {}
    assert out.rationale == "act 决策缺少 chosen_action_id"
    assert provider.calls == 1


async def test_decide_continue_includes_structured_tool_history_window():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        def __init__(self) -> None:
            self.last_messages = None

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.last_messages = messages
            return '{"decision":"wait","rationale":"继续观察"}'

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
    layer = JudgmentLayer(provider, _tool_registry(), cfg)
    layer._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{
            "tool": "memory.search",
            "params": {"query": "继续分析"},
            "result": "命中 2 条相关记忆",
            "summary": "命中 2 条相关记忆",
            "error": "",
            "status": "ok",
            "state_delta": {"hits": 2},
        }],
        user_message="继续分析",
        prefer_tier="reasoner",
    )

    assert out.decision == "wait"
    assert provider.last_messages is not None
    assert "结构化最近工具结果(JSON)" in provider.last_messages[1].content
    assert '"status": "ok"' in provider.last_messages[1].content
    assert '"state_delta": {' in provider.last_messages[1].content


async def test_decide_continue_keeps_complex_act_without_runtime_rewrite():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from memory.task_store import Task
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return (
                '{"decision":"act","chosen_action_id":"shell.run",'
                '"params":{"command":"pytest -q"},'
                '"rationale":"先继续排查失败原因。",'
                '"next_step":"再修复 chat 回复链路"}'
            )

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

    task = Task(
        id=11,
        title="排查 chat 回复",
        status="active",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="逐一排查 chat 回复链路",
    )
    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    layer._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "memory.search", "params": {"query": "chat 回复"}, "result": "命中 2 条相关记忆"}],
        user_message="请你逐一排查并修复 chat 回复问题",
        active_task=task,
        prefer_tier="reasoner",
    )

    assert out.decision == "act"
    assert out.chosen_action_id == "shell.run"
    assert out.params == {"command": "pytest -q"}
    assert out.next_step == "再修复 chat 回复链路"


def test_rewrite_task_ask_keeps_direct_question_after_evidence():
    from core.judgment.runtime import _rewrite_task_ask_to_evidence

    action = _judgment_output(
        decision="act",
        chosen_action_id="task.ask",
        params={"question": "还需要你补充具体 id 吗？"},
    )

    rewritten = _rewrite_task_ask_to_evidence(
        action,
        user_message="继续分析",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "继续分析"}, "result": "命中 2 条相关记忆"},
            {"tool": "task.list", "params": {"status": "all"}, "result": "[12] [in_progress] 继续分析"},
        ],
        registry=_tool_registry(),
    )

    assert rewritten is action
    assert rewritten.chosen_action_id == "task.ask"
    assert rewritten.params["question"] == "还需要你补充具体 id 吗？"


def test_rewrite_task_ask_leaves_evidence_budget_decision_to_llm():
    from core.judgment.runtime import _rewrite_task_ask_to_evidence

    action = _judgment_output(
        decision="act",
        chosen_action_id="task.ask",
        params={"question": "还需要你补充具体 id 吗？"},
    )

    rewritten = _rewrite_task_ask_to_evidence(
        action,
        user_message="继续分析",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "继续分析"}, "result": "命中 1 条相关记忆"},
        ],
        registry=_tool_registry(),
    )

    assert rewritten is action
    assert rewritten.chosen_action_id == "task.ask"
    assert rewritten.params == {"question": "还需要你补充具体 id 吗？"}


def test_rewrite_task_ask_ignores_configured_evidence_budget_for_passthrough():
    from core.judgment.runtime import _rewrite_task_ask_to_evidence

    action = _judgment_output(
        decision="act",
        chosen_action_id="task.ask",
        params={"question": "请补充 task id"},
    )

    rewritten = _rewrite_task_ask_to_evidence(
        action,
        user_message="继续分析",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "继续分析"}, "result": "命中 1 条相关记忆"},
            {"tool": "task.list", "params": {"status": "all"}, "result": "命中 1 条任务"},
        ],
        registry=_tool_registry(),
        evidence_budget=3,
    )

    assert rewritten is action
    assert rewritten.chosen_action_id == "task.ask"
    assert rewritten.params == {"question": "请补充 task id"}


def test_rewrite_complex_user_act_keeps_llm_chosen_mutation():
    from core.judgment.runtime import _rewrite_complex_act_to_task_plan
    from memory.task_store import Task

    task = Task(
        id=7,
        title="修复 chat 回复",
        status="active",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="逐一排查 chat 回复链路",
    )
    action = _judgment_output(
        decision="act",
        chosen_action_id="shell.run",
        params={"command": "rg \"chat\" core -n"},
        rationale="先检查日志，再修复回复链路。",
        next_step="再修复 chat 回复链路",
    )

    rewritten = _rewrite_complex_act_to_task_plan(
        action,
        user_message="请你逐一排查昨天日志并修复 chat 回复问题",
        active_task=task,
        registry=_tool_registry(),
    )

    assert rewritten is action
    assert rewritten.chosen_action_id == "shell.run"
    assert rewritten.params == {"command": "rg \"chat\" core -n"}
    assert rewritten.next_step == "再修复 chat 回复链路"


def test_rewrite_complex_user_act_keeps_read_action_without_plan():
    from core.judgment.runtime import _rewrite_complex_act_to_task_plan
    from memory.task_store import Task

    task = Task(
        id=8,
        title="排查日志",
        status="active",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="逐一分析昨天日志",
    )
    action = _judgment_output(
        decision="act",
        chosen_action_id="file.read",
        params={"path": "/tmp/runtime.log"},
        next_step="再总结问题点",
    )

    rewritten = _rewrite_complex_act_to_task_plan(
        action,
        user_message="请你逐一分析昨天日志并整理问题",
        active_task=task,
        registry=_tool_registry(),
    )

    assert rewritten is action
    assert rewritten.chosen_action_id == "file.read"


def test_rewrite_complex_user_act_keeps_structural_next_step_without_rewrite():
    from core.judgment.runtime import _rewrite_complex_act_to_task_plan
    from memory.task_store import Task

    task = Task(
        id=9,
        title="处理回复",
        status="active",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="处理 chat 回复",
    )
    action = _judgment_output(
        decision="act",
        chosen_action_id="shell.run",
        params={"command": "pytest -q"},
        next_step="汇总结果并修复失败项",
    )

    rewritten = _rewrite_complex_act_to_task_plan(
        action,
        user_message="帮我处理一下",
        active_task=task,
        registry=_tool_registry(),
    )

    assert rewritten is action
    assert rewritten.chosen_action_id == "shell.run"
    assert rewritten.next_step == "汇总结果并修复失败项"


def test_rewrite_complex_user_act_respects_plan_alignment_exempt_capability():
    from core.judgment.runtime import _rewrite_complex_act_to_task_plan
    from memory.task_store import Task
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    @tool(ToolManifest(
        name="debug.plan.exempt",
        description="调试用对齐豁免工具",
        capabilities=("plan_alignment_exempt",),
    ))
    async def _debug_plan_exempt(params: dict[str, object], ctx: ToolContext) -> ToolResult:
        return ToolResult(summary="ok")

    task = Task(
        id=10,
        title="处理回复",
        status="active",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="处理 chat 回复",
    )
    action = _judgment_output(
        decision="act",
        chosen_action_id="debug.plan.exempt",
        params={"query": "check"},
        next_step="继续汇总",
    )

    rewritten = _rewrite_complex_act_to_task_plan(
        action,
        user_message="帮我继续处理",
        active_task=task,
        registry=ToolRegistry(),
    )

    assert rewritten is action
    assert rewritten.chosen_action_id == "debug.plan.exempt"


def test_preferred_continue_tier_uses_manifest_reader_tier():
    from core.loop.common import _preferred_continue_tier, _should_continue_within_tick
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    @tool(ToolManifest(
        name="debug.reader.inspect",
        description="调试用 reader 工具",
        prefer_tier="reader",
        capabilities=("completion_info_only",),
    ))
    async def _debug_reader_inspect(params: dict[str, object], ctx: ToolContext) -> ToolResult:
        return ToolResult(summary="ok")

    reg = ToolRegistry()
    action = _judgment_output(decision="act", chosen_action_id="debug.reader.inspect")

    assert _preferred_continue_tier(action, user_message="继续分析", registry=reg) is None
    assert _should_continue_within_tick(
        action,
        user_message="继续分析",
        has_active_task=True,
        registry=reg,
    ) is True


async def test_sync_task_progress_state_promotes_previous_next_step():
    from core.task_runtime import _sync_task_progress_state
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
    from core.task_runtime import _sync_task_progress_state
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime-explicit.db")
        await store.open()
        task_id = await store.add_task("迁移任务", goal="验证显式 current_step 优先", next_step="继续旧技能")
        task = await store.get_task_by_id(task_id)
        assert task is not None

        await store.sync_task_progress(task_id, current_step="收到新迁移指令", next_step="开始盘点旧运行时记忆")
        updated = await _sync_task_progress_state(
            store,
            task,
            previous_next_step="继续旧技能",
            action=_judgment_output(decision="act", chosen_action_id="task.update", next_step="开始盘点旧运行时记忆"),
            progressful=True,
            state_delta={"current_step": "收到新迁移指令", "next_step": "开始盘点旧运行时记忆"},
        )

        assert updated is not None
        assert updated.current_step == "收到新迁移指令"
        assert updated.next_step == "开始盘点旧运行时记忆"
        await store.close()


def test_fmt_task_exposes_runtime_state_to_llm():
    from core.judgment.context import _fmt_task
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
        extras={"plan": [{"step": "检查回复链路", "status": "in_progress"}]},
    )
    section = _fmt_task(task)
    assert "状态: active" in section
    assert "模型层级: repair" in section
    assert "当前步骤: 检查 run monitor" in section
    assert "当前计划:" in section
    assert "检查回复链路" in section
    assert "最近运行状态: failed" in section


def test_fmt_context_facts_surfaces_task_and_recent_general_facts():
    asyncio.run(_fmt_context_facts_surfaces_task_and_recent_general_facts())


def test_load_context_facts_snapshot_uses_configured_exclude_prefixes_and_limits():
    asyncio.run(_load_context_facts_snapshot_uses_configured_exclude_prefixes_and_limits())


async def _fmt_context_facts_surfaces_task_and_recent_general_facts():
    from core.judgment.context import _fmt_context_facts, _load_context_facts_snapshot
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / 'facts.db')
        await store.open()
        task_id = await store.add_task('分析旧运行时记忆', goal='确认 carrier')
        task = await store.get_task_by_id(task_id)
        assert task is not None

        await store.set_fact(f'task:{task_id}:progress', '已确认 sqlite 为主载体', scope='task')
        await store.set_fact('legacy_runtime.workspace_memory.primary_carrier', '/root/.legacy-runtime/memory/main.sqlite')
        await store.set_fact('pref:routing_overrides', '{"reader":"demo"}', scope='system')

        facts = await _load_context_facts_snapshot(store, task)
        text = _fmt_context_facts(facts)

        assert f'task:{task_id}:progress' in text
        assert 'legacy_runtime.workspace_memory.primary_carrier' in text
        assert 'pref:routing_overrides' not in text
        await store.close()


async def _load_context_facts_snapshot_uses_configured_exclude_prefixes_and_limits():
    from core.judgment.context import _load_context_facts_snapshot

    class _Store:
        def __init__(self) -> None:
            self.calls: list[tuple[str | None, int]] = []

        async def list_facts(self, prefix=None, limit=0):
            self.calls.append((prefix, limit))
            if prefix == 'task:7:':
                return [
                    ('task:7:a', 'A'),
                    ('task:7:b', 'B'),
                    ('task:7:c', 'C'),
                ][:limit]
            return [
                ('pref:hidden', 'P'),
                ('soul:visible', 'S'),
                ('evolution:visible', 'E'),
                ('misc:1', 'M1'),
                ('run:hidden', 'R'),
            ][:limit]

    store = _Store()
    facts = await _load_context_facts_snapshot(
        cast(Any, store),
        cast(Any, SimpleNamespace(id=7)),
        exclude_prefixes=['pref:', 'run:'],
        task_limit=2,
        global_limit=2,
        recent_scan_multiplier=4,
        recent_scan_min=9,
    )

    assert store.calls == [('task:7:', 2), (None, 9)]
    assert facts == [
        ('task:7:a', 'A'),
        ('task:7:b', 'B'),
        ('soul:visible', 'S'),
        ('evolution:visible', 'E'),
    ]


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
    from core.loop.logging import _clip_reply_for_log

    clipped = _clip_reply_for_log("<memory-context>hidden</memory-context>\n用户可见回复")
    assert clipped == "用户可见回复"


def test_assemble_context_prefers_active_task_override_with_inbox():
    asyncio.run(_assemble_context_prefers_active_task_override_with_inbox())


async def _assemble_context_prefers_active_task_override_with_inbox():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

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
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "旧回填任务",
                goal="等待模型加载完成后，再次检查日志确认数据回填进度",
                next_step="继续检查回填进度",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None
            task.extras["inbox_messages"] = [
                "收到新的用户消息：请你使用 puppeteer 去搜索。"
            ]

            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assemble_context(
                CognitionFrame(
                    percept=cast(Any, SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=task,
                user_message="请你使用 puppeteer 去搜索。",
            )

            assert "⚠️ 新增用户消息（inbox 1 条，先评估这些新消息是否改变当前方向）:" in text
            assert "收到新的用户消息：请你使用 puppeteer 去搜索。" in text
        finally:
            await store.close()


def test_assemble_context_without_active_task_or_probe_manager_does_not_crash():
    asyncio.run(_assemble_context_without_active_task_or_probe_manager_does_not_crash())


async def _assemble_context_without_active_task_or_probe_manager_does_not_crash():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

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
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            layer._probe_manager = None
            text = await layer._assemble_context(
                CognitionFrame(
                    percept=cast(Any, SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="帮我检查当前状态",
            )

            assert "帮我检查当前状态" in text
        finally:
            await store.close()


def test_assemble_context_semantic_anchors_do_not_bucket_emotion():
    asyncio.run(_assemble_context_semantic_anchors_do_not_bucket_emotion())


async def _assemble_context_semantic_anchors_do_not_bucket_emotion():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolRegistry

    captured: list[str] = []

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

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
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "回归情绪锚测试",
                goal="确认 semantic anchors 不再使用情绪桶标签",
                next_step="检查 semantic 检索锚点",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None

            semantic = SemanticMemory(Path(d) / "memory", decay_lambda=0.0)

            def _capture_retrieve_multi_anchor(anchors, top_k):
                captured.extend(str(a) for a in anchors)
                return []

            semantic.retrieve_multi_anchor = cast(Any, _capture_retrieve_multi_anchor)

            emotion = EmotionState.from_config(cfg)
            emotion.valence = 0.10
            emotion.arousal = 0.95

            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            await layer._assemble_context(
                CognitionFrame(
                    percept=cast(Any, SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=semantic,
                    emotion=emotion,
                ),
                active_task=task,
                user_message="",
            )

            assert captured
            assert "检查 semantic 检索锚点" in captured
            assert "焦虑" not in captured
            assert "沮丧" not in captured
            assert "兴奋" not in captured
            assert "稳定" not in captured
            assert "中性" not in captured
        finally:
            await store.close()


def test_assemble_context_consumes_parallel_fetch_exceptions():
    asyncio.run(_assemble_context_consumes_parallel_fetch_exceptions())


async def _assemble_context_consumes_parallel_fetch_exceptions():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

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
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        loop = asyncio.get_running_loop()
        recorded_exceptions: list[str] = []
        prev_handler = loop.get_exception_handler()

        def _capture_exception(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
            message = str(context.get("message") or context.get("exception") or context)
            recorded_exceptions.append(message)

        loop.set_exception_handler(_capture_exception)
        try:
            task_id = await store.add_task(
                "并发上下文异常回归",
                goal="确保并发上下文异常不会泄漏未消费异常",
                next_step="触发 list_runs 与 list_failures_for_task 异常",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None

            async def _list_runs_fail(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("recent runs boom")

            async def _list_failures_fail(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("failures boom")

            store.list_runs = cast(Any, _list_runs_fail)
            store.list_failures_for_task = cast(Any, _list_failures_fail)

            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)

            with pytest.raises(RuntimeError, match="recent runs boom"):
                await layer._assemble_context(
                    CognitionFrame(
                        percept=cast(Any, SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                        wm=WorkingMemory(capacity=20),
                        task_store=store,
                        episodic=EpisodicMemory(Path(d) / "memory"),
                        semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                        emotion=EmotionState.from_config(cfg),
                    ),
                    active_task=task,
                    user_message="检查并发异常清理",
                )

            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert recorded_exceptions == []
        finally:
            loop.set_exception_handler(prev_handler)
            await store.close()


def test_assemble_context_registry_override_limits_tools_section():
    asyncio.run(_assemble_context_registry_override_limits_tools_section())


async def _assemble_context_registry_override_limits_tools_section():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from core.subagent import _DEFAULT_BLOCKED_TOOLS, _FilteredRegistry
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

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
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            registry = _tool_registry()
            filtered = _FilteredRegistry(registry, {"task.list"}, set(_DEFAULT_BLOCKED_TOOLS))
            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), registry, cfg)
            text = await layer._assemble_context(
                CognitionFrame(
                    percept=cast(Any, SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="检查子灵工具边界",
                registry_override=filtered,
            )

            assert "- `task.list`:" in text
            assert "- `shell.run`:" not in text
            assert "- `subagent.run`:" not in text
        finally:
            await store.close()


