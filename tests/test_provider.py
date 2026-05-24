"""Auth / Copilot provider 测试"""
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
def test_auth_store_profile_roundtrip(tmp_path):
    from store.auth import load_auth_profiles, set_token_profile

    path = tmp_path / "auth-profiles.json"
    set_token_profile(profile_id="copilot:default", provider="copilot", token="tok-123456", path=path)
    data = load_auth_profiles(path)
    assert data["version"] == 1
    assert data["profiles"]["copilot:default"]["provider"] == "copilot"
    assert data["profiles"]["copilot:default"]["token"] == "tok-123456"


def test_copilot_token_resolution_prefers_auth_profile(monkeypatch, tmp_path):
    from store.auth import resolve_copilot_token, set_token_profile, save_legacy_credentials

    monkeypatch.setenv("GH_TOKEN", "env-gh-token")
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    set_token_profile(profile_id="copilot:default", provider="copilot", token="profile-token", path=tmp_path / "auth-profiles.json")
    save_legacy_credentials({"GITHUB_TOKEN": "legacy-token"}, path=tmp_path / "credentials.json")

    import store.auth as auth_mod
    monkeypatch.setattr(auth_mod, "AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    monkeypatch.setattr(auth_mod, "LEGACY_CREDENTIALS_PATH", tmp_path / "credentials.json")

    resolved = resolve_copilot_token()
    assert resolved is not None
    assert resolved.token == "profile-token"
    assert resolved.source == "auth-profile"


def test_github_device_client_id_prefers_env(monkeypatch, tmp_path):
    import json
    import store.auth as auth_mod

    state_file = tmp_path / "github-device.json"
    state_file.write_text(json.dumps({"client_id": "Iv1.file-client"}), encoding="utf-8")

    monkeypatch.setattr(auth_mod, "GITHUB_DEVICE_AUTH_PATH", state_file)
    monkeypatch.setenv("LINGZHOU_GITHUB_CLIENT_ID", "Iv1.env-client")

    assert auth_mod.load_github_device_client_id() == "Iv1.env-client"


def _write_hot_reload_config(
    path: Path,
    *,
    model: str,
    mtime: float,
    embedding_model: str | None = "text-embedding-v3",
) -> None:
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "bailian": {
                        "type": "openai_compat",
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": "DASHSCOPE_API_KEY",
                    },
                    "copilot": {
                        "type": "openai_compat",
                        "mode": "copilot",
                        "base_url": "https://api.githubcopilot.com",
                        "api_key_env": "GITHUB_TOKEN",
                    },
                },
                "model": model,
                "routing": {"reader": "bailian/qwen-plus"},
                "loop": {
                    "db_path": str(path.parent / "state" / "runtime.db"),
                    "memory_dir": str(path.parent / "memory"),
                    "state_dir": str(path.parent / "state"),
                    "workspace_dir": str(path.parent / "workspace"),
                },
                "memory": {
                    "embedding_model": embedding_model,
                    "embedding_weight": 0.45,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.utime(path, (mtime, mtime))


class _ReloadClosable:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    def embed(self, text: str) -> list[float]:
        return [float(len(text) or 1)]

    async def close(self) -> None:
        self.closed = True


class _ReloadSelfModel:
    def __init__(self) -> None:
        self.last_cfg = None

    def set_routing(self, cfg: Any) -> None:
        self.last_cfg = cfg


class _ReloadJudgment:
    def __init__(self, provider: Any, registry: Any, cfg: Any) -> None:
        self.provider = provider
        self.registry = registry
        self.cfg = cfg
        self.self_model = _ReloadSelfModel()
        self.routing_providers: dict[str, Any] = {}

    def set_routing_providers(self, providers: dict[str, Any]) -> None:
        self.routing_providers = providers


class _ReloadExecution:
    def __init__(self, registry: Any, cfg: Any) -> None:
        self.registry = registry
        self.cfg = cfg


class _ReloadEvolution:
    def __init__(self, cfg: Any, provider: Any, registry: Any) -> None:
        self.cfg = cfg
        self.provider = provider
        self.registry = registry


class _ReloadPerception:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg


class _ReloadSoul:
    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        self.refresh_calls = 0

    async def refresh_identity(self, judgment: Any = None) -> None:
        self.refresh_calls += 1


class _ReloadTaskStore:
    def __init__(self, *, value: str = "", found: bool = False) -> None:
        self.value = value
        self.found = found

    async def get_fact(self, key: str) -> tuple[str, bool]:
        assert key == "pref:routing_overrides"
        return self.value, self.found


def test_hot_reload_build_failure_keeps_old_runtime(monkeypatch, tmp_path):
    asyncio.run(_hot_reload_build_failure_keeps_old_runtime(monkeypatch, tmp_path))


async def _hot_reload_build_failure_keeps_old_runtime(monkeypatch, tmp_path):
    from core.config import Config
    import core.loop.reload as reload_mod

    cfg_path = tmp_path / "lingzhou.json"
    old_mtime = time.time() - 10
    new_mtime = old_mtime + 5
    _write_hot_reload_config(cfg_path, model="bailian/qwen-plus", mtime=old_mtime)
    old_cfg = Config.load(cfg_path)
    _write_hot_reload_config(cfg_path, model="copilot/gpt-5.4", mtime=new_mtime)

    old_provider = _ReloadClosable("old-main")
    old_reader = _ReloadClosable("old-reader")
    self_model = _ReloadSelfModel()
    loop = cast(
        Any,
        SimpleNamespace(
            _cfg=old_cfg,
            _cfg_file=cfg_path,
            _cfg_mtime=old_mtime,
            _auth_profiles_path=tmp_path / "auth-profiles.json",
            _auth_profiles_mtime=0.0,
            _provider=old_provider,
            _routing_providers={"reader": old_reader},
            _registry=object(),
            _judgment=SimpleNamespace(self_model=self_model),
            _execution=object(),
            _evolution=object(),
            _perception=object(),
            _semantic=SimpleNamespace(_embed_fn=None, _embedding_weight=0.0),
            _soul=_ReloadSoul(old_cfg),
        ),
    )

    monkeypatch.setattr(reload_mod, "create_provider", lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")))

    await reload_mod._maybe_hot_reload_provider_impl(loop)

    assert loop._cfg is old_cfg
    assert loop._provider is old_provider
    assert loop._routing_providers["reader"] is old_reader
    assert loop._cfg_mtime == old_mtime
    assert loop._auth_profiles_mtime == 0.0
    assert old_provider.closed is False
    assert old_reader.closed is False
    assert loop._soul._cfg is old_cfg


def test_hot_reload_success_atomically_replaces_runtime(monkeypatch, tmp_path):
    asyncio.run(_hot_reload_success_atomically_replaces_runtime(monkeypatch, tmp_path))


async def _hot_reload_success_atomically_replaces_runtime(monkeypatch, tmp_path):
    from core.config import Config
    import core.loop.reload as reload_mod

    cfg_path = tmp_path / "lingzhou.json"
    old_mtime = time.time() - 10
    new_mtime = old_mtime + 5
    _write_hot_reload_config(cfg_path, model="bailian/qwen-plus", mtime=old_mtime)
    old_cfg = Config.load(cfg_path)
    _write_hot_reload_config(cfg_path, model="copilot/gpt-5.4", mtime=new_mtime)

    old_provider = _ReloadClosable("old-main")
    old_reader = _ReloadClosable("old-reader")
    new_provider = _ReloadClosable("new-main")
    new_reader = _ReloadClosable("new-reader")
    self_model = _ReloadSelfModel()
    loop = cast(
        Any,
        SimpleNamespace(
            _cfg=old_cfg,
            _cfg_file=cfg_path,
            _cfg_mtime=old_mtime,
            _auth_profiles_path=tmp_path / "auth-profiles.json",
            _auth_profiles_mtime=0.0,
            _provider=old_provider,
            _routing_providers={"reader": old_reader},
            _registry=object(),
            _judgment=SimpleNamespace(self_model=self_model),
            _execution="old-execution",
            _evolution="old-evolution",
            _perception="old-perception",
            _semantic=SimpleNamespace(_embed_fn=None, _embedding_weight=0.0),
            _soul=_ReloadSoul(old_cfg),
        ),
    )

    monkeypatch.setattr(reload_mod, "create_provider", lambda cfg: new_provider)
    monkeypatch.setattr(reload_mod, "_build_routing_providers", lambda cfg: {"reader": new_reader})
    monkeypatch.setattr(reload_mod, "JudgmentLayer", _ReloadJudgment)
    monkeypatch.setattr(reload_mod, "ExecutionLayer", _ReloadExecution)
    monkeypatch.setattr(reload_mod, "EvolutionEngine", _ReloadEvolution)
    monkeypatch.setattr(reload_mod, "PerceptionLayer", _ReloadPerception)

    await reload_mod._maybe_hot_reload_provider_impl(loop)

    assert loop._cfg.model == "copilot/gpt-5.4"
    assert loop._provider is new_provider
    assert loop._routing_providers["reader"] is new_reader
    assert isinstance(loop._judgment, _ReloadJudgment)
    assert isinstance(loop._execution, _ReloadExecution)
    assert isinstance(loop._evolution, _ReloadEvolution)
    assert isinstance(loop._perception, _ReloadPerception)
    assert loop._cfg_mtime == new_mtime
    assert old_provider.closed is True
    assert old_reader.closed is True
    assert loop._soul._cfg is loop._cfg
    assert loop._soul.refresh_calls == 1
    assert callable(loop._semantic._embed_fn)
    assert loop._semantic._embedding_weight == loop._cfg.memory.embedding_weight
    assert self_model.last_cfg is loop._cfg


def test_hot_reload_refreshes_runtime_routing_overrides_from_db(monkeypatch, tmp_path):
    asyncio.run(_hot_reload_refreshes_runtime_routing_overrides_from_db(monkeypatch, tmp_path))


async def _hot_reload_refreshes_runtime_routing_overrides_from_db(monkeypatch, tmp_path):
    from core.config import Config
    import core.loop.reload as reload_mod

    cfg_path = tmp_path / "lingzhou.json"
    old_mtime = time.time() - 10
    new_mtime = old_mtime + 5
    _write_hot_reload_config(cfg_path, model="bailian/qwen-plus", mtime=old_mtime)
    old_cfg = Config.load(cfg_path)
    _write_hot_reload_config(cfg_path, model="copilot/gpt-5.4", mtime=new_mtime)

    old_provider = _ReloadClosable("old-main")
    old_reader = _ReloadClosable("old-reader")
    new_provider = _ReloadClosable("new-main")
    new_reader = _ReloadClosable("new-reader")
    self_model = _ReloadSelfModel()
    loop = cast(
        Any,
        SimpleNamespace(
            _cfg=old_cfg,
            _cfg_file=cfg_path,
            _cfg_mtime=old_mtime,
            _auth_profiles_path=tmp_path / "auth-profiles.json",
            _auth_profiles_mtime=0.0,
            _provider=old_provider,
            _routing_providers={"reader": old_reader},
            _registry=object(),
            _judgment=SimpleNamespace(self_model=self_model),
            _execution="old-execution",
            _evolution="old-evolution",
            _perception="old-perception",
            _semantic=SimpleNamespace(_embed_fn=None, _embedding_weight=0.0),
            _soul=_ReloadSoul(old_cfg),
            _task_store=_ReloadTaskStore(value='{"reasoner":"copilot/gpt-5.4"}', found=True),
            _pending_routing_overrides={"reasoner": "bailian/qwen-plus"},
        ),
    )

    monkeypatch.setattr(reload_mod, "create_provider", lambda cfg: new_provider)
    monkeypatch.setattr(reload_mod, "_build_routing_providers", lambda cfg: {"reader": new_reader})
    monkeypatch.setattr(reload_mod, "JudgmentLayer", _ReloadJudgment)
    monkeypatch.setattr(reload_mod, "ExecutionLayer", _ReloadExecution)
    monkeypatch.setattr(reload_mod, "EvolutionEngine", _ReloadEvolution)
    monkeypatch.setattr(reload_mod, "PerceptionLayer", _ReloadPerception)

    await reload_mod._maybe_hot_reload_provider_impl(loop)

    assert loop._pending_routing_overrides == {"reasoner": "copilot/gpt-5.4"}


def test_copilot_gpt5_does_not_auto_inject_max_completion_tokens():
    from provider.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
    provider._provider_mode = "copilot"
    provider._model = "gpt-5.4"

    payload = {}
    provider._inject_completion_limits(payload)

    assert "max_completion_tokens" not in payload


def test_copilot_o_series_uses_max_completion_tokens():
    from provider.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
    provider._provider_mode = "copilot"
    provider._model = "o3"

    payload = {}
    provider._inject_completion_limits(payload)

    assert payload["max_completion_tokens"] == 100000


def test_copilot_transport_selection_and_limits_are_metadata_driven(monkeypatch):
    import provider.openai_compat as mod

    provider = mod.OpenAICompatProvider.__new__(mod.OpenAICompatProvider)
    provider._provider_mode = "copilot"
    provider._model = "future-reasoner"
    provider._temperature = 0.7
    provider._thinking_level = "high"
    provider._extra_body = {}

    monkeypatch.setattr(mod, "lookup_model", lambda model_id, catalog_path=None: {
        "api": "responses",
        "reasoning": True,
        "max_tokens": 1234,
        "request_params": {
            "unsupported": ["temperature"],
            "completion_limit_param": "max_completion_tokens",
        },
    } if model_id == "future-reasoner" else None)

    payload = provider._build_responses_payload(  # type: ignore[attr-defined]
        [mod.Message(role="system", content="sys"), mod.Message(role="user", content="u")],
        temperature=0.0,
    )
    limits_payload: dict[str, Any] = {}
    provider._inject_completion_limits(limits_payload)

    assert provider._uses_responses_api() is True
    assert payload["reasoning"] == {"effort": "high"}
    assert "temperature" not in payload
    assert limits_payload["max_completion_tokens"] == 1234


def test_models_gen_merges_provider_model_overrides_modalities_and_capabilities(tmp_path):
    import json as _json

    from core.config import Config
    from provider import catalog as catalog_mod
    from provider.models_gen import ensure_models_json

    cfg_path = tmp_path / "lingzhou.json"
    cfg_path.write_text(
        _json.dumps(
            {
                "providers": {
                    "copilot": {
                        "type": "openai_compat",
                        "mode": "copilot",
                        "base_url": "https://api.individual.githubcopilot.com",
                        "api_key_env": "GITHUB_TOKEN",
                        "models": [
                            {
                                "id": "gpt-5.4",
                                "input": ["text"],
                                "capabilities": ["text_generation", "thinking"],
                                "request_params": {
                                    "unsupported": ["temperature", "top_p"]
                                },
                            },
                            {
                                "id": "future-vision",
                                "api": "responses",
                                "input": ["text", "image"],
                                "capabilities": ["text_generation", "vision"],
                                "context_window": 123456,
                                "max_tokens": 4096,
                            },
                        ],
                    }
                },
                "model": "copilot/gpt-5.4",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = Config.load(cfg_path)
    cfg.loop.workspace_dir = str(tmp_path / "workspace")

    asyncio.run(ensure_models_json(cfg))

    runtime_path = tmp_path / "workspace" / "models.json"
    runtime_catalog = _json.loads(runtime_path.read_text(encoding="utf-8"))
    copilot_models = {m["id"]: m for m in runtime_catalog["copilot"]["models"]}

    assert copilot_models["gpt-5.4"]["api"] == "responses"
    assert copilot_models["gpt-5.4"]["max_tokens"] == 65536
    assert copilot_models["gpt-5.4"]["input"] == ["text"]
    assert copilot_models["gpt-5.4"]["capabilities"] == ["text_generation", "thinking"]
    assert copilot_models["gpt-5.4"]["request_params"]["unsupported"] == ["temperature", "top_p"]
    assert copilot_models["future-vision"]["input"] == ["text", "image"]
    assert copilot_models["future-vision"]["capabilities"] == ["text_generation", "vision"]
    assert copilot_models["future-vision"]["api"] == "responses"
    assert catalog_mod.lookup_model("future-vision") is None
    explicit_model = catalog_mod.lookup_model("future-vision", catalog_path=runtime_path)
    assert explicit_model is not None
    assert explicit_model["api"] == "responses"


def test_models_gen_ready_cache_is_bounded():
    from provider import models_gen as models_gen_mod

    previous_cache = dict(models_gen_mod._READY_CACHE)
    try:
        models_gen_mod._READY_CACHE.clear()
        for index in range(models_gen_mod._READY_CACHE_MAX + 5):
            models_gen_mod._remember_ready_fingerprint(f"workspace-{index}", f"fp-{index}")

        assert len(models_gen_mod._READY_CACHE) == models_gen_mod._READY_CACHE_MAX
        assert "workspace-0" not in models_gen_mod._READY_CACHE
        assert f"workspace-{models_gen_mod._READY_CACHE_MAX + 4}" in models_gen_mod._READY_CACHE
    finally:
        models_gen_mod._READY_CACHE.clear()
        models_gen_mod._READY_CACHE.update(previous_cache)


def test_copilot_o_series_chat_retries_without_reasoning_fields_after_400():
    import httpx
    from provider.base import Message
    from provider.openai_compat import OpenAICompatProvider

    class _FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.timeout = SimpleNamespace(read=30.0, connect=30.0)
            self._responses = [
                httpx.Response(400, text='{"error":"bad request"}', request=httpx.Request("POST", "https://api.individual.githubcopilot.com/chat/completions")),
                httpx.Response(400, text='{"error":"unsupported field"}', request=httpx.Request("POST", "https://api.individual.githubcopilot.com/chat/completions")),
                httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=httpx.Request("POST", "https://api.individual.githubcopilot.com/chat/completions")),
            ]

        async def post(self, url, *, content=None, headers=None, timeout=None):
            self.calls.append({
                "url": url,
                "payload": json.loads(content or "{}"),
                "headers": headers,
                "timeout": timeout,
            })
            return self._responses.pop(0)

    fake_client = _FakeAsyncClient()
    provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
    provider._provider_mode = "copilot"
    provider._model = "o3"
    provider._temperature = 0.7
    provider._thinking_level = "high"
    provider._extra_body = {}
    provider._client = cast(Any, fake_client)
    provider._copilot_api_base_url = "https://api.individual.githubcopilot.com"  # type: ignore[assignment]

    async def _ensure_token(*, force_refresh: bool = False) -> str:
        return "copilot-token-2" if force_refresh else "copilot-token-1"

    provider._ensure_copilot_token = _ensure_token  # type: ignore[assignment]
    provider._copilot_request_headers = lambda token: {"Authorization": f"Bearer {token}"}  # type: ignore[assignment]
    provider._copilot_url = lambda path: f"https://api.individual.githubcopilot.com{path}"  # type: ignore[assignment]

    result = asyncio.run(provider.chat(
        [Message(role="system", content="s"), Message(role="user", content="u")],
        temperature=0.0,
    ))

    assert result == "ok"
    assert len(fake_client.calls) == 3
    first_payload = fake_client.calls[0]["payload"]
    third_payload = fake_client.calls[2]["payload"]
    assert first_payload["reasoning_effort"] == "high"
    assert first_payload["max_completion_tokens"] == 100000
    assert first_payload["temperature"] == 1
    assert "reasoning_effort" not in third_payload
    assert "max_completion_tokens" not in third_payload
    assert third_payload["temperature"] == 0.0


def test_copilot_gpt5_uses_responses_endpoint_and_parses_output_text():
    import httpx
    from provider.base import Message
    from provider.openai_compat import OpenAICompatProvider

    class _FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.timeout = SimpleNamespace(read=30.0, connect=30.0)
            self._responses = [
                httpx.Response(
                    200,
                    json={
                        "output_text": "ok from responses",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "ok from responses"}],
                            }
                        ],
                    },
                    request=httpx.Request("POST", "https://api.individual.githubcopilot.com/responses"),
                ),
            ]

        async def post(self, url, *, content=None, headers=None, timeout=None):
            self.calls.append({
                "url": url,
                "payload": json.loads(content or "{}"),
                "headers": headers,
                "timeout": timeout,
            })
            return self._responses.pop(0)

    fake_client = _FakeAsyncClient()
    provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
    provider._provider_mode = "copilot"
    provider._model = "gpt-5.4-mini"
    provider._temperature = 0.7
    provider._thinking_level = "high"
    provider._extra_body = {}
    provider._client = cast(Any, fake_client)
    provider._copilot_api_base_url = "https://api.individual.githubcopilot.com"  # type: ignore[assignment]

    async def _ensure_token(*, force_refresh: bool = False) -> str:
        return "copilot-token-1"

    provider._ensure_copilot_token = _ensure_token  # type: ignore[assignment]
    provider._copilot_request_headers = lambda token: {"Authorization": f"Bearer {token}"}  # type: ignore[assignment]
    provider._copilot_url = lambda path: f"https://api.individual.githubcopilot.com{path}"  # type: ignore[assignment]

    result = asyncio.run(provider.chat(
        [Message(role="system", content="sys"), Message(role="user", content="u")],
        temperature=0.0,
    ))

    assert result == "ok from responses"
    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["url"].endswith("/responses")
    assert call["payload"]["instructions"] == "sys"
    assert call["payload"]["input"] == [{"role": "user", "content": "u"}]
    assert call["payload"]["reasoning"] == {"effort": "high"}
    assert "temperature" not in call["payload"]
    assert "messages" not in call["payload"]


def test_copilot_gpt5_responses_payload_omits_temperature():
    from provider.base import Message
    from provider.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
    provider._provider_mode = "copilot"
    provider._model = "gpt-5.4-mini"
    provider._temperature = 0.7
    provider._thinking_level = "high"
    provider._extra_body = {}

    payload = provider._build_responses_payload(  # type: ignore[attr-defined]
        [Message(role="system", content="sys"), Message(role="user", content="u")],
        temperature=0.0,
    )

    assert payload["model"] == "gpt-5.4-mini"
    assert payload["instructions"] == "sys"
    assert payload["input"] == [{"role": "user", "content": "u"}]
    assert payload["reasoning"] == {"effort": "high"}
    assert "temperature" not in payload


def test_copilot_gpt5_responses_400_surfaces_error_body():
    import httpx
    from provider.base import Message
    from provider.openai_compat import OpenAICompatProvider

    class _FakeAsyncClient:
        def __init__(self) -> None:
            self.timeout = SimpleNamespace(read=30.0, connect=30.0)
            self._responses = [
                httpx.Response(
                    400,
                    text='{"error":{"message":"model \\"gpt-5.4-mini\\" is not accessible via the /responses endpoint","code":"unsupported_api_for_model"}}',
                    request=httpx.Request("POST", "https://api.individual.githubcopilot.com/responses"),
                ),
                httpx.Response(
                    400,
                    text='{"error":{"message":"model \\"gpt-5.4-mini\\" is not accessible via the /responses endpoint","code":"unsupported_api_for_model"}}',
                    request=httpx.Request("POST", "https://api.individual.githubcopilot.com/responses"),
                ),
            ]

        async def post(self, url, *, content=None, headers=None, timeout=None):
            return self._responses.pop(0)

    fake_client = _FakeAsyncClient()
    provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
    provider._provider_mode = "copilot"
    provider._model = "gpt-5.4-mini"
    provider._temperature = 0.7
    provider._thinking_level = "high"
    provider._extra_body = {}
    provider._client = cast(Any, fake_client)
    provider._copilot_api_base_url = "https://api.individual.githubcopilot.com"  # type: ignore[assignment]

    async def _ensure_token(*, force_refresh: bool = False) -> str:
        return "copilot-token-2" if force_refresh else "copilot-token-1"

    provider._ensure_copilot_token = _ensure_token  # type: ignore[assignment]
    provider._copilot_request_headers = lambda token: {"Authorization": f"Bearer {token}"}  # type: ignore[assignment]
    provider._copilot_url = lambda path: f"https://api.individual.githubcopilot.com{path}"  # type: ignore[assignment]

    with pytest.raises(httpx.HTTPStatusError, match="unsupported_api_for_model"):
        asyncio.run(provider.chat(
            [Message(role="system", content="sys"), Message(role="user", content="u")],
            temperature=0.0,
        ))


def test_copilot_base_url_derives_from_proxy_ep():
    from provider.openai_compat import _derive_copilot_api_base_url_from_token

    token = "ghu_xxx; proxy-ep=proxy.business.githubcopilot.com; tid=abc"
    assert _derive_copilot_api_base_url_from_token(token) == "https://api.business.githubcopilot.com"


def test_copilot_normalize_base_url_uses_default_base_url():
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


