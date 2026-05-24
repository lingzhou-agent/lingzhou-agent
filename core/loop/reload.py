from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from core.config import Config
from core.evolution import EvolutionEngine
from core.execution import ExecutionLayer
from core.judgment import JudgmentLayer
from core.perception import PerceptionLayer
from provider import create_provider
from provider.base import EmbeddingProvider

from .startup import _build_routing_providers

if TYPE_CHECKING:
    from .runtime import CognitionLoop

_log = logging.getLogger("lingzhou.loop")


@dataclass
class _HotReloadCandidate:
    cfg: Config
    provider: Any
    routing_providers: dict[str, Any]
    judgment: JudgmentLayer
    execution: ExecutionLayer
    evolution: EvolutionEngine
    perception: PerceptionLayer
    replaced_provider_stack: bool

    async def close_new_stack(self) -> None:
        if not self.replaced_provider_stack:
            return
        await _close_provider_stack(self.provider, self.routing_providers)


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _provider_reload_signature(cfg: Config) -> str:
    payload = {
        "model": cfg.model,
        "routing": cfg.routing,
        "providers": {
            name: provider.model_dump(mode="json")
            for name, provider in sorted(cfg.providers.items())
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


async def _close_provider_stack(provider: Any, routing_providers: dict[str, Any]) -> None:
    try:
        await provider.close()
    except Exception:
        pass
    for routing_provider in routing_providers.values():
        try:
            await routing_provider.close()
        except Exception:
            pass


def _refresh_semantic_embed_runtime(loop: "CognitionLoop") -> None:
    semantic = cast(Any, loop._semantic)
    semantic._embed_fn = loop._provider.embed if loop._cfg.memory.embedding_model and isinstance(loop._provider, EmbeddingProvider) else None
    semantic._embedding_weight = loop._cfg.memory.embedding_weight


async def _refresh_runtime_routing_overrides(loop: "CognitionLoop") -> None:
    task_store = getattr(loop, "_task_store", None)
    if task_store is None:
        return

    refreshed: dict[str, str] | None = None
    try:
        raw, found = await task_store.get_fact("pref:routing_overrides")
    except Exception as exc:
        _log.warning("[hot-reload] 读取 DB routing_overrides 失败,保留当前内存态: %s", exc)
        return

    if found and raw:
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                refreshed = {
                    key: value
                    for key, value in payload.items()
                    if key in {"reader", "reasoner", "repair"} and isinstance(value, str) and value
                } or None
        except Exception as exc:
            _log.warning("[hot-reload] DB routing_overrides 解析失败,已清空内存态: %s", exc)

    previous = getattr(loop, "_pending_routing_overrides", None)
    loop._pending_routing_overrides = refreshed
    if previous != refreshed:
        _log.info("[hot-reload] routing_overrides 已同步: %s", refreshed or {})


async def _build_reload_candidate(
    loop: "CognitionLoop",
    new_cfg: Config,
    *,
    auth_changed: bool,
) -> _HotReloadCandidate:
    replaced_provider_stack = auth_changed or _provider_reload_signature(loop._cfg) != _provider_reload_signature(new_cfg)
    provider = loop._provider
    routing_providers = loop._routing_providers
    new_provider: Any | None = None
    new_routing_providers: dict[str, Any] | None = None
    try:
        if replaced_provider_stack:
            provider = create_provider(new_cfg)
            new_provider = create_provider(new_cfg)
            new_routing_providers = _build_routing_providers(new_cfg)
            provider = new_provider
            routing_providers = new_routing_providers
        judgment = JudgmentLayer(provider, loop._registry, new_cfg)
        judgment.self_model = loop._judgment.self_model
        judgment.self_model.set_routing(new_cfg)
        judgment.set_routing_providers(routing_providers)

        return _HotReloadCandidate(
            cfg=new_cfg,
            provider=provider,
            routing_providers=routing_providers,
            judgment=judgment,
            execution=ExecutionLayer(loop._registry, new_cfg),
            evolution=EvolutionEngine(new_cfg, provider, loop._registry),
            perception=PerceptionLayer(new_cfg),
            replaced_provider_stack=replaced_provider_stack,
        )
    except Exception:
        if new_provider is not None:
            await _close_provider_stack(new_provider, new_routing_providers or {})
        raise


async def _commit_hot_reload_candidate(
    loop: "CognitionLoop",
    candidate: _HotReloadCandidate,
    *,
    cfg_mtime: float,
    auth_mtime: float,
) -> None:
    old_cfg = loop._cfg
    old_provider = loop._provider
    old_routing_providers = loop._routing_providers

    loop._cfg = candidate.cfg
    loop._provider = candidate.provider
    loop._routing_providers = candidate.routing_providers
    loop._judgment = candidate.judgment
    loop._execution = candidate.execution
    loop._evolution = candidate.evolution
    loop._perception = candidate.perception
    loop._cfg_mtime = cfg_mtime
    loop._auth_profiles_mtime = auth_mtime
    loop._soul._cfg = candidate.cfg
    _refresh_semantic_embed_runtime(loop)
    await _refresh_runtime_routing_overrides(loop)

    try:
        await loop._soul.refresh_identity(loop._judgment)
    except Exception as exc:
        _log.warning("[hot-reload] 身份前缀刷新失败,保留新运行时: %s", exc)

    if candidate.replaced_provider_stack:
        await _close_provider_stack(old_provider, old_routing_providers)


async def _maybe_hot_reload_provider_impl(loop: "CognitionLoop") -> None:
    if not loop._cfg_file.exists():
        return

    cfg_mtime = _file_mtime(loop._cfg_file)
    auth_mtime = _file_mtime(loop._auth_profiles_path)
    cfg_changed = cfg_mtime > loop._cfg_mtime
    auth_changed = auth_mtime > loop._auth_profiles_mtime
    if not cfg_changed and not auth_changed:
        return

    try:
        new_cfg = Config.load(loop._cfg_file)
    except Exception as exc:
        _log.warning("[hot-reload] 配置解析失败,保留旧运行时: %s", exc)
        return

    old_model = loop._cfg.model
    try:
        candidate = await _build_reload_candidate(loop, new_cfg, auth_changed=auth_changed)
    except Exception as exc:
        _log.warning("[hot-reload] 构建候选运行时失败,保留旧运行时: %s", exc)
        return

    try:
        await _commit_hot_reload_candidate(loop, candidate, cfg_mtime=cfg_mtime, auth_mtime=auth_mtime)
    except Exception as exc:
        await candidate.close_new_stack()
        _log.warning("[hot-reload] 提交新运行时失败,已回退旧运行时: %s", exc)
        return

    if auth_changed and not cfg_changed:
        _log.info("[hot-reload] auth profiles 更新,已原子重建 provider 栈")
    elif candidate.replaced_provider_stack and old_model != new_cfg.model:
        _log.info("[hot-reload] 模型热换完成: %s -> %s", old_model, new_cfg.model)
    else:
        _log.info("[hot-reload] 配置热加载完成")
