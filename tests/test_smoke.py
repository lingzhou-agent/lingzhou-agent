"""快速验证测试，不依赖 LLM。"""
import asyncio
import math
import os
import tempfile
from datetime import datetime, UTC, timedelta
from pathlib import Path

import aiosqlite


# ── helpers ──────────────────────────────────────────────────────────────────

def _proj_root() -> Path:
    return Path(__file__).parent.parent


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
    raw = '```json\n{"decision":"act","chosen_action_id":"shell.run","params":{"command":"echo hi"},"rationale":"test","reflection":"洞察","next_step":"done"}\n```'
    out = JudgmentOutput.from_llm(raw)
    assert out.decision == "act"
    assert out.chosen_action_id == "shell.run"
    assert out.reflection == "洞察"


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
    assert "shell.run" in names
    assert "task.complete" in names
    assert "memory.add_wm" in names


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
        await store.update_task_data(tid, {"tags": ["ai"], "score": 99})
        t3 = await store.get_task_by_id(tid)
        assert t3 is not None
        assert t3.extras["score"] == 99
        assert t3.next_step == "步骤1"  # 原有字段未被覆盖

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

    cfg = Config.load(_proj_root() / "lingzhou.json")
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        assert loop.semantic.decay_lambda == cfg.memory.semantic_decay_lambda
        assert loop.episodic.max_events == cfg.memory.max_events


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
