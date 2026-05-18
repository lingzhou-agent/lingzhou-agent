"""core/probe/manager.py — 探针系统外部 API。

ProbeManager 是探针系统的唯一对外接口，由 CognitionLoop 持有。
它封装了 ProbeStore（JSON 文件持久化）和 ProbeRunner（调度执行）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .store import ProbeStore
from .runner import ProbeRunner
from .types import ProbeConfig, ProbeResult

_log = logging.getLogger("lingzhou.probe")


class ProbeManager:
    """探针系统管理器（由 CognitionLoop 持有）。

    生命周期：
    1. ProbeManager(probe_file) — 创建；探针配置从 JSON 文件加载
    2. await manager.start(wm, loop_ref) — 启动所有调度 Task
    3. manager.stop() — 优雅停止所有调度 Task
    """

    def __init__(self, probe_file: Path) -> None:
        self._store = ProbeStore(probe_file)
        self._store.load()  # 同步加载，在事件循环前就就绪
        self._runner = ProbeRunner(self._store)

    async def start(self, wm: Any, loop_ref: Any | None = None) -> None:
        """注入运行时依赖，启动所有调度任务。"""
        self._runner.attach(wm, loop_ref)
        await self._runner.start_all()
        _log.info("[probe] ProbeManager started")

    def stop(self) -> None:
        """取消所有调度 Task（shutdown 时调用）。"""
        for name in list(self._runner._tasks):
            self._runner.unschedule(name)
        _log.info("[probe] ProbeManager stopped")

    # ── CRUD API（供 tools/probe_ops.py 调用） ─────────────────────────────────

    async def install(self, cfg: ProbeConfig) -> ProbeConfig:
        """安装或更新探针，立即启动调度。"""
        await self._store.upsert(cfg)
        saved = await self._store.get(cfg.name)
        assert saved is not None
        self._runner._schedule(saved)
        _log.info("[probe] installed probe=%s trigger=%s data_back=%s", cfg.name, cfg.trigger, cfg.data_back)
        return saved

    async def remove(self, name: str) -> bool:
        """移除探针（取消调度 + 删除 DB）。"""
        self._runner.unschedule(name)
        deleted = await self._store.delete(name)
        if deleted:
            _log.info("[probe] removed probe=%s", name)
        return deleted

    async def run_now(self, name: str) -> ProbeResult | None:
        """立即执行指定探针，返回结果。"""
        cfg = await self._store.get(name)
        if cfg is None:
            return None
        return await self._runner.run_now(cfg)

    async def list_probes(self) -> list[ProbeConfig]:
        return await self._store.list_all()

    async def get_probe(self, name: str) -> ProbeConfig | None:
        return await self._store.get(name)

    def runner_status(self) -> dict[str, str]:
        """各探针调度 Task 的运行状态。"""
        return self._runner.status()
