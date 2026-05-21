"""core/loop/runtime.py - 认知主循环(CognitionLoop)。

一个 tick 的流程:
  感知 → 情绪更新 → 伦理评估 → 判断信号生成 → LLM 判断 → 工具执行 → 记忆整合
  每 consolidate_every 轮:WM 内容写入情节记忆
  每 evolve_every 轮:触发自进化检查

解耦原则:loop 只编排,不包含业务逻辑;各层职责内聚。
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

_log = logging.getLogger("lingzhou.loop")

from core.config import Config
from core.perception import (
    PerceptionLayer, EmotionState,
)
from core.judgment import JudgmentLayer, JudgmentOutput
from core.execution import (
    ExecutionLayer,
)
from core.evolution import EvolutionEngine
from .tick import (
    _maybe_record_success_stall_reflection_impl,
    _post_tick_memory_impl,
    _tick_finalize_impl,
    _tick_impl,
)
from memory.working import WorkingMemory, WMItem
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.task_store import TaskStore, Task
from provider import create_provider
from tools.registry import ToolRegistry, ToolContext, ToolResult
from core.behavior_tracker import BehaviorTracker
from core.soul import SoulManager
from core.probe import ProbeManager
from .driver import _run_cycle_impl, _wait_after_cycle_impl, _wait_for_event_impl
from .chat import _process_pending_chat_turn, _tick_interact_impl
from .reload import _maybe_hot_reload_provider_impl
from .startup import (
    _open_runtime_impl,
    _prepare_runtime_run_impl,
    _restore_self_model_impl,
    _restore_state_from_db_impl,
)

console = Console()



class CognitionLoop:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

        # 工具注册
        self._registry = ToolRegistry()
        tools_dir = Path(__file__).parent.parent.parent / "tools"
        self._registry.discover(tools_dir)

        # 插件系统：发现并加载插件
        from core.plugin import PluginManager
        plugins_dir = Path(__file__).parent.parent.parent / "plugins"
        self._plugin_manager = PluginManager(plugins_dir)
        self._plugin_manager.discover()
        self._plugin_manager.load_all()
        self._plugin_manager.register_all(tool_registry=self._registry)
        self._plugin_manager.start_all()
        _log.info("[plugin] 已加载 %d 个插件", len(self._plugin_manager.list_plugins()))

        # 记忆层
        self._wm = WorkingMemory(capacity=cfg.memory.working_capacity, token_budget=cfg.effective_wm_token_budget(), item_max_tokens=cfg.memory.wm_item_max_tokens)
        self._episodic = EpisodicMemory(cfg.memory_dir, max_events=cfg.memory.max_events)
        self._task_store = TaskStore(Path(cfg.db_path))

        # 情绪状态(初始值来自 config)
        self._emotion = EmotionState.from_config(cfg)

        # 认知组件
        self._provider = create_provider(cfg)
        self._perception = PerceptionLayer(cfg)
        self._judgment = JudgmentLayer(self._provider, self._registry, cfg)
        self._execution = ExecutionLayer(self._registry, cfg)
        self._evolution = EvolutionEngine(cfg, self._provider, self._registry)
        # 分层路由 providers({"simple": p1, "complex": p2},由 open() 注入 JudgmentLayer)
        self._routing_providers: dict[str, Any] = {}
        # embedding 混合检索(embed_fn=None 则纯关键词模式)
        _embed_fn = None
        if cfg.memory.local_embed_model:
            try:
                import os as _os
                import importlib

                _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                _os.environ.setdefault("HF_HUB_OFFLINE", "1")
                _st_kwargs: dict = {}
                if cfg.memory.local_embed_cache_dir:
                    _st_kwargs["cache_folder"] = cfg.memory.local_embed_cache_dir
                _st_module = importlib.import_module("sentence_transformers")
                _ST = getattr(_st_module, "SentenceTransformer")
                _local_st = _ST(cfg.memory.local_embed_model, **_st_kwargs)
                _embed_fn = lambda texts: _local_st.encode(texts, normalize_embeddings=True).tolist()
            except Exception as _e:
                import logging as _lg
                _lg.getLogger("lingzhou.loop").warning("[loop] 本地 embedding 模型加载失败，回退到 API: %s", _e)
                _embed_fn = getattr(self._provider, "embed", None) if cfg.memory.embedding_model else None
        elif cfg.memory.embedding_model:
            _embed_fn = getattr(self._provider, "embed", None)
        self._semantic = SemanticMemory(
            cfg.memory_dir,
            decay_lambda=cfg.memory.semantic_decay_lambda,
            embed_fn=_embed_fn,
            embedding_weight=cfg.memory.embedding_weight,
        )

        # 子系统:Soul 文件管理 + 行为模式追踪
        self._soul = SoulManager(self._cfg, self._task_store, self._wm)
        self._behavior = BehaviorTracker(
            wait_streak_notify=list(cfg.loop.wait_streak_notify),
            streak_threshold=cfg.loop.behavior_streak_threshold,
            wm_priorities={
                "behavior_loop": cfg.thresholds.wm_pri_user_msg,
                "edit_caution": cfg.thresholds.wm_pri_self_aware,
                "belief_stale": cfg.thresholds.wm_pri_critical,
            },
        )

        # 自驱力引擎 (Active Inference + Intrinsic Motivation)
        from core.self_drive import SelfDriveEngine
        self._self_drive = SelfDriveEngine(str(cfg.db_path))

        # tick 间连续性追踪(预测误差 + 认知信号计算用)
        self._last_next_step: str = ""
        self._last_decision: str = "wait"
        self._last_act_error: bool = False   # 兼容旧信号:上轮 act 是否以工具错误结束
        self._last_act_progressful: bool = False
        self._last_act_progress_reason: str = ""  # LLM 可见的进展判断原因
        self._last_action_tool: str = ""
        self._last_action_key: str = ""
        self._last_action_status: str = ""
        self._last_action_summary: str = ""
        self._last_action_error: str = ""
        self._last_action_state_delta: str = ""
        self._success_stall_task_id: str | None = None
        self._success_stall_streak: int = 0
        self._recent_action_feedback: deque[str] = deque(maxlen=3)
        self._last_action_sig: str = ""
        self._last_result_fp: str = ""
        self._idle_cycles: int = 0
        self._last_curiosity_signal_idle_cycle: int = 0

        # 多轮对话历史(最多保留 6 轮 user/assistant 对)
        self._conv_history: deque[tuple[str, str]] = deque(maxlen=6)
        # 心跳计时(monotonic,独立于用户 cron,不存 DB)
        self._last_heartbeat_at: float = 0.0
        # bootstrap 模式（由 soul.bootstrap() 在 open/run 时写入）
        # "full" = 首次运行；"none" = 正常运行（BOOTSTRAP.md 已删除）
        self._bootstrap_mode: str = "none"
        # 探针系统：配置来自工作区 probes.json（与主 DB 完全解耦）
        _probe_file = Path(cfg.loop.workspace_dir).expanduser() / "probes.json"
        self._probe_manager: ProbeManager = ProbeManager(_probe_file)
        self._judgment._probe_manager = self._probe_manager
        # 按请求计费聚合:追踪距上次真正调用 LLM 已经过了几轮
        self._ticks_since_judge: int = 0
        # LLM 通过 model_strategy.next_phase_tier 跨 tick 传递的 tier 偏好
        self._pending_tier: str | None = None
        self._pending_idle_gap: float | None = None  # LLM 通过 model_strategy.next_idle_gap_secs 动态调控等待时长
        self._pending_routing_overrides: dict[str, str] | None = None  # LLM 通过 routing_overrides 临时覆盖 tier→model
        self._pending_thinking_override: str | None = None  # LLM 通过 thinking_override 覆盖下轮 thinking 等级
        _cfg_file = cfg._base_dir / "lingzhou.json"
        self._cfg_file: Path = _cfg_file
        self._cfg_mtime: float = _cfg_file.stat().st_mtime if _cfg_file.exists() else 0.0
        # 同时监听 auth-profiles.json(token 更新时重建 provider)
        from store.auth import AUTH_PROFILES_PATH as _AUTH_PROFILES_PATH
        self._auth_profiles_path: Path = _AUTH_PROFILES_PATH
        self._auth_profiles_mtime: float = _AUTH_PROFILES_PATH.stat().st_mtime if _AUTH_PROFILES_PATH.exists() else 0.0

    @property
    def probe_manager(self) -> ProbeManager:
        return self._probe_manager

    @property
    def semantic(self) -> SemanticMemory:
        return self._semantic

    @property
    def episodic(self) -> EpisodicMemory:
        return self._episodic

    async def _maybe_hot_reload_provider(self) -> None:
        await _maybe_hot_reload_provider_impl(self)

    def _make_ctx(self) -> ToolContext:
        return ToolContext(
            config=self._cfg,
            wm=self._wm,
            task_store=self._task_store,
            episodic=self._episodic,
            semantic=self._semantic,
            emotion=self._emotion,
            probe_manager=self._probe_manager,
            judgment=self._judgment,
        )

    async def open(self) -> None:
        """打开数据库连接、执行启动引导和状态恢复。interact 模式下替代 run() 前两步。"""
        await _open_runtime_impl(self)

    async def run(self) -> None:
        cfg, _routing_summary = await _prepare_runtime_run_impl(self)

        console.print(Panel(
            f"[bold green]lingzhou[/bold green] 启动\n"
            f"provider={cfg.model}  idle_gap={cfg.loop.max_idle_gap}ms  "
            f"act={'yes' if cfg.loop.act else 'dry-run'}\n"
            f"routing:\n{_routing_summary}",
            title="🌱 认知循环"
        ))

        cycle = 0
        consecutive_errors = 0

        try:
            while True:
                try:
                    cycle = await _run_cycle_impl(self, cycle)
                    consecutive_errors = 0
                except Exception:
                    consecutive_errors += 1
                    console.print_exception(max_frames=5)
                    if consecutive_errors >= cfg.loop.max_consecutive_errors:
                        console.print(
                            f"[red]连续错误 {consecutive_errors} 次,暂停循环[/red]"
                        )
                        break

                try:
                    await _wait_after_cycle_impl(self)
                except Exception:
                    _log.exception("[loop] _wait_after_cycle_impl 异常，跳过本次等待")
                    await asyncio.sleep(1.0)  # 防止异常紧循环消耗 CPU
                cfg = self._cfg  # 可能已更新
        finally:
            self._probe_manager.stop()
            await self._task_store.close()
            await self._provider.close()
            for _rp in self._routing_providers.values():
                try:
                    await _rp.close()
                except Exception:
                    _log.exception("[loop] 关闭 routing provider 失败")
            # 干净退出：更新 survival.json 的 exit_type，下次启动不触发崩溃注入
            try:
                import json as _json
                _sp = self._cfg.state_dir / "survival.json"
                if _sp.exists():
                    _snap = _json.loads(_sp.read_text(encoding="utf-8"))
                    _snap["exit_type"] = "clean"
                    _sp.write_text(_json.dumps(_snap, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

    async def _wait_for_event(self, max_wait: float, before_task) -> None:
        """事件驱动等待:chat 消息、task 状态变化、超时三类事件任一发生即唤醒。"""
        await _wait_for_event_impl(self, max_wait, before_task)

    async def _process_pending_chat_turn(self, cycle: int) -> tuple[int, bool]:
        return await _process_pending_chat_turn(self, cycle)

    async def _tick(
        self,
        cycle: int,
        user_message: str = "",
        chat_id: str | None = None,
    ) -> str:
        return await _tick_impl(self, cycle, user_message=user_message, chat_id=chat_id)

    async def _tick_finalize(
        self,
        action: JudgmentOutput,
        result: ToolResult | Any,
        active_task: Task | None,
        cycle: int,
        user_message: str,
        cognitive_signals: Any,
        reply: str,
        chat_id: str | None = None,
        perception_replay: Any = None,
    ) -> str:
        return await _tick_finalize_impl(
            self,
            action,
            result,
            active_task,
            cycle,
            user_message,
            cognitive_signals,
            reply,
            chat_id,
            perception_replay,
        )

    async def _maybe_record_success_stall_reflection(
        self,
        active_task: Task | None,
        action: JudgmentOutput,
        result: ToolResult,
        cycle: int,
    ) -> None:
        await _maybe_record_success_stall_reflection_impl(self, active_task, action, result, cycle)

    async def _restore_state_from_db(self) -> None:
        await _restore_state_from_db_impl(self)


    async def _restore_self_model(self) -> None:
        await _restore_self_model_impl(self)

    async def _save_self_model(self) -> None:
        """持久化自我模型到 DB(每 tick 调用)。"""
        await self._task_store.set_fact("self:model", self._judgment.self_model.to_json(), scope="system")

    def _maybe_inject_budget_warning(self) -> None:
        """Token 预算记录：仅日志，不向 WM 注入任何建议。"""
        tokens = self._judgment.self_model.total_tokens
        if tokens > 8_000_000:
            _log.debug("[budget] 今日 token=%.1fM", tokens / 1e6)

    async def _maybe_inject_self_drive(self) -> None:
        """自驱力引擎：空闲或探索卡住时注入自主探索目标到 WM。

        基于 Active Inference + Intrinsic Motivation:
        - 好奇心 C(t) > 阈值 → 生成探索目标
        - 长时间空闲 → 强制探索
        - 探索卡住（explore-awareness 触发）→ 建议换策略
        """
        # 检查是否有真的活跃任务（非 waiting 状态）
        has_real_work = (
            self._last_decision == "act"
            and self._last_action_tool
            and not self._last_action_tool.startswith("task.update")
        )

        # 检查是否探索卡住（streak 超过窗口大小 + 2，使用公开属性）
        _stuck_gate = self._cfg.loop.behavior_streak_threshold + 2
        explore_stuck = (
            self._behavior.list_streak_count >= _stuck_gate
            or self._behavior.read_streak_count >= _stuck_gate
        )

        signal = self._self_drive.compute_signal(
            idle_ticks=self._behavior.wait_streak,
            has_user_message=False,
            has_active_task=bool(has_real_work and not explore_stuck),
            tick=self._judgment.self_model.tick_count,
            force_explore_idle=self._cfg.thresholds.curiosity_idle_min_cycles,
        )
        if not signal.should_explore:
            return

        # 去重①：队列中已存在未完成的 self_drive 任务时跳过
        runnable = await self._task_store.list_runnable_tasks(limit=20)
        if any(t.source == "self_drive" for t in runnable):
            return

        # 去重②：最近 2 小时内已有 self_drive 任务完成，跳过（防止刚完成就重新触发）
        import time as _time
        from datetime import datetime as _dt, timezone as _tz
        _recent_done = await self._task_store.list_tasks(status="done", limit=10)
        _two_hours_ago = _time.time() - 7200
        for _t in _recent_done:
            if getattr(_t, "source", None) != "self_drive":
                continue
            try:
                _ts = _dt.fromisoformat(_t.created_at.replace("Z", "+00:00")).timestamp()
            except Exception:
                continue
            if _ts >= _two_hours_ago:
                return

        task_template = self._self_drive.generate_exploration_task(
            signal.suggested_domain or "self_evolution"
        )
        await self._task_store.add_task(
            title=task_template["title"],
            goal=task_template["goal"],
            next_step=task_template.get("next_step", ""),
            source="self_drive",
            priority="low",
        )

        _log.info(
            "[self_drive] 探索触发 C=%.2f domain=%s idle=%d rationale=%s",
            signal.curiosity_score,
            signal.suggested_domain,
            self._behavior.wait_streak,
            signal.rationale,
        )

    async def _post_tick_memory(
        self,
        action: JudgmentOutput,
        result: Any,
        active_task: Any,
        cycle: int,
        user_message: str,
    ) -> None:
        await _post_tick_memory_impl(self, action, result, active_task, cycle, user_message)

    @property
    def task_store(self) -> TaskStore:
        return self._task_store

    @property
    def provider(self):
        return self._provider

    async def tick_interact(self, cycle: int, user_message: str) -> str:
        return await _tick_interact_impl(self, cycle, user_message)

    async def state_snapshot(self) -> dict[str, Any]:
        """返回当前可见状态快照,供 interact REPL 渲染(Clark & Schaefer 1989 基础共识)。

        P2-A: 扩展字段,包含行为循环探针、空闲计数、WM 压力等诊断信息。
        """
        active_task = await self._task_store.get_active()
        running_runs = await self._task_store.list_runs(status="running", limit=5)
        wm_items = self._wm.get_top(3)
        _bt = self._behavior.snapshot()
        return {
            "valence": round(self._emotion.valence, 4),
            "arousal": round(self._emotion.arousal, 4),
            "dominance": round(self._emotion.dominance, 4),
            "dominant_emotion": self._emotion.dominant,
            "task_title": active_task.title if active_task else None,
            "task_id": str(active_task.id) if active_task else None,
            "task_status": active_task.status if active_task else None,
            "wm_size": len(self._wm.get_top(100)),
            "wm_pressure": round(self._wm.pressure, 4),
            "wm_top": [i.get("content", "")[:60] for i in wm_items],
            "idle_cycles": self._idle_cycles,
            "running_runs": [
                {
                    "id": r.id,
                    "task_id": r.task_id,
                    "tool": r.tool_name,
                    "worker": r.worker_type,
                    "session_id": r.session_id,
                }
                for r in running_runs
            ],
            "action_streak": _bt["action_streak"],
            "read_streak": _bt["read_streak"],
            "loop_probe_version": _bt["loop_probe_version"],
            "conv_history_len": len(self._conv_history),
            "fts5_ok": self._semantic.fts5_ok,
        }

    async def _maybe_curiosity_task(self, ethos_state: Any) -> None:
        """P1-C: 好奇心阈值驱动的探索信号注入。

        触发条件(全部满足):
        1. 当前无活跃任务
        2. 空闲周期 >= thresholds.curiosity_idle_min_cycles
        3. ethos.curiosity >= thresholds.curiosity_idle_task
        4. 每个 idle 周期段最多提示一次,由 LLM 决定是否创建任务
        """
        cfg = self._cfg
        if self._idle_cycles < cfg.thresholds.curiosity_idle_min_cycles:
            return
        curiosity = getattr(ethos_state.values, "curiosity", 0.0) if ethos_state else 0.0
        if curiosity < cfg.thresholds.curiosity_idle_task:
            return
        if self._idle_cycles - self._last_curiosity_signal_idle_cycle < cfg.thresholds.curiosity_idle_min_cycles:
            return

        recent = await self._task_store.list_tasks(limit=10)
        pending_curiosity = [
            t for t in recent
            if getattr(t, "source", None) == "curiosity"
            and getattr(t, "status", "done") not in ("done", "failed")
        ]
        self._last_curiosity_signal_idle_cycle = self._idle_cycles
        _log.info(
            "[curiosity] idle=%d curiosity=%.2f pending_tasks=%d",
            self._idle_cycles, curiosity, len(pending_curiosity),
        )
        # 无待处理的好奇心任务时，向 WM 注入信号，让 LLM 感知到好奇心触发并决定是否创建任务
        if not pending_curiosity:
            self._wm.add(WMItem(
                kind="curiosity",
                content=(
                    f"[好奇心] 已空闲 {self._idle_cycles} 轮，好奇心 {curiosity:.2f} "
                    f"> 阈值 {cfg.thresholds.curiosity_idle_task}。"
                    "建议发起自主探索任务（task.add source=curiosity）或深化当前认知。"
                ),
                priority=0.7,
            ))

    async def _consolidate(self, active_task: Task | None) -> None:
        """将 WM 高优先级条目写入情节记忆,然后清空 WM,保留身份锚点。"""
        items = self._wm.get_top(25)
        if not items:
            return
        task_id = str(active_task.id) if active_task else None
        summary = "\n".join(f"- [{i['kind']}] {i['content']}" for i in items)
        self._episodic.record(role="consolidation", content=summary, task_id=task_id)
        # 保留身份锚点(bootstrap_identity)和自我感知信号(self_awareness)
        # self_awareness 包含行为循环检测等信号，清除后 LLM 会失去对空转的感知
        self._wm.clear(preserve_kinds={"bootstrap_identity", "self_awareness"})
        # 清空后注入任务锚点,避免下一轮因 WM 为空而丢失任务上下文
        if active_task:
            _progress_line = ""
            try:
                _prog, _prog_found = await self._task_store.get_fact(f"task:{active_task.id}:progress")
                if _prog_found and _prog:
                    _progress_line = f"\n进度: {_prog}"
            except Exception:
                pass
            self._wm.add(WMItem(
                kind="task_anchor",
                content=(
                    f"[任务锚点] {active_task.title}\n"
                    f"目标: {active_task.goal or '(未指定)'}\n"
                    f"下一步: {active_task.next_step or '(未指定)'}"
                    f"{_progress_line}"
                ),
                priority=0.95,
            ))
        # 同步感知基准,避免下一轮因 WM 大小骤降产生假预测误差
        self._perception.reset_wm_baseline(len(self._wm))
        _log.info("[consolidate] WM→episodic %d items, WM cleared (bootstrap+task_anchor preserved)", len(items))
