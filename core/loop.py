"""core/loop.py — 认知主循环（CognitionLoop）。

一个 tick 的流程：
  感知 → 情绪更新 → 伦理评估 → 判断信号生成 → LLM 判断 → 工具执行 → 记忆整合
  每 consolidate_every 轮：WM 内容写入情节记忆
  每 evolve_every 轮：触发自进化检查

解耦原则：loop 只编排，不包含业务逻辑；各层职责内聚。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import deque
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

_log = logging.getLogger("lingzhou.loop")

from core.config import Config
from core.perception import (
    PerceptionLayer, EmotionState,
    build_perception_replay, build_emotion_replay,
    derive_ethos_state, compute_judgment_signals,
)
from core.judgment import JudgmentLayer, JudgmentOutput
from core.execution import ExecutionLayer
from core.evolution import EvolutionEngine
from memory.working import WorkingMemory, WMItem
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory, MemoryNode
from memory.task_store import TaskStore, Task
from provider import create_provider, create_provider_with_model
from provider.models_gen import ensure_models_json
from tools.registry import ToolRegistry, ToolContext
from core.behavior_tracker import BehaviorTracker
from core.soul import SoulManager

console = Console()

# WM 优先级具名常量（集中定义，避免散落魔数）
_WM_PRI_SIGNAL   = 0.90   # 调度信号、执行成功结果
_WM_PRI_HISTORY  = 0.88   # 近期对话历史
_WM_PRI_IDENTITY = 0.85   # 身份/Soul 文件（bootstrap/init 与 core.soul 同步使用）
_WM_PRI_ERROR    = 0.30   # 工具失败结果

# P1-B: reflection → 情绪效价的关键词启发式推断（模块级，无 LLM 依赖）
_VALENCE_POS = frozenset(["完成", "成功", "理解", "学到", "进步", "有效", "清晰", "好", "正确", "解决", "突破"])
_VALENCE_NEG = frozenset(["失败", "错误", "困惑", "卡住", "无法", "问题", "不对", "不清", "循环", "重复", "卡顿"])


def _infer_valence_from_text(text: str, current: float) -> float:
    """从 reflection 文本推断情绪效价倾向。

    只做轻度修正（±0.05 上限在调用处控制）；
    关键词命中越多越偏向极性，无命中时返回 current（不产生噪声）。
    """
    pos = sum(1 for w in _VALENCE_POS if w in text)
    neg = sum(1 for w in _VALENCE_NEG if w in text)
    if pos + neg == 0:
        return current
    ratio = pos / (pos + neg)
    # 映射到 [0.3, 1.0] 再与 current 混合 (权重 0.2)
    target = 0.3 + ratio * 0.7
    return current * 0.8 + target * 0.2


def _strip_memory_context(text: str) -> str:
    """剥离 LLM 输出中意外泄露的 <memory-context>...</memory-context> 内容（Hermes 借鉴）。

    Hermes 使用 StreamingContextScrubber 防止 memory fencing 标签泄露给用户。
    lingzhou 在 tick_interact() 的 reply 返回前做一次性清洗。
    """
    import re as _re
    cleaned = _re.sub(r"<memory-context>.*?</memory-context>", "", text, flags=_re.DOTALL)
    return cleaned.strip() or text.strip()


class CognitionLoop:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

        # 工具注册
        self._registry = ToolRegistry()
        tools_dir = Path(__file__).parent.parent / "tools"
        self._registry.discover(tools_dir)

        # 记忆层
        self._wm = WorkingMemory(capacity=cfg.memory.working_capacity)
        self._episodic = EpisodicMemory(cfg.memory_dir, max_events=cfg.memory.max_events)
        self._task_store = TaskStore(cfg.db_path)

        # 情绪状态（初始值来自 config）
        self._emotion = EmotionState.from_config(cfg)

        # 认知组件
        self._provider = create_provider(cfg)
        self._perception = PerceptionLayer(cfg)
        self._judgment = JudgmentLayer(self._provider, self._registry, cfg)
        self._execution = ExecutionLayer(self._registry, cfg)
        self._evolution = EvolutionEngine(cfg, self._provider, self._registry)
        # 分层路由 providers（{"simple": p1, "complex": p2}，由 open() 注入 JudgmentLayer）
        self._routing_providers: dict[str, Any] = {}
        # Hermes/OpenClaw 借鉴：embedding 混合检索（embed_fn=None 则纯关键词模式）
        _embed_fn = getattr(self._provider, "embed", None) if cfg.memory.embedding_model else None
        self._semantic = SemanticMemory(
            cfg.memory_dir,
            decay_lambda=cfg.memory.semantic_decay_lambda,
            embed_fn=_embed_fn,
            embedding_weight=cfg.memory.embedding_weight,
        )

        # 子系统：Soul 文件管理 + 行为模式追踪
        self._soul = SoulManager(self._cfg, self._task_store, self._wm)
        self._behavior = BehaviorTracker()

        # tick 间连续性追踪（预测误差 + 认知信号计算用）
        self._last_next_step: str = ""
        self._last_decision: str = "wait"
        self._idle_cycles: int = 0

        # 多轮对话历史（最多保留 6 轮 user/assistant 对）
        self._conv_history: deque[tuple[str, str]] = deque(maxlen=6)
        # 心跳计时（monotonic，独立于用户 cron，不存 DB）
        self._last_heartbeat_at: float = 0.0
        # 按请求计费聚合：追踪距上次真正调用 LLM 已经过了几轮
        self._ticks_since_judge: int = 0
        # 上一轮 tick 的决策类型（act/wait/fallback），用于动态 sleep 计算
        self._last_decision: str = "wait"
        # 配置文件热重载：记录初始 mtime，每轮 sleep 后检查
        _cfg_file = cfg._base_dir / "lingzhou.json"
        self._cfg_file: Path = _cfg_file
        self._cfg_mtime: float = _cfg_file.stat().st_mtime if _cfg_file.exists() else 0.0
        # 同时监听 auth-profiles.json（token 更新时重建 provider）
        from auth_store import AUTH_PROFILES_PATH as _AUTH_PROFILES_PATH
        self._auth_profiles_path: Path = _AUTH_PROFILES_PATH
        self._auth_profiles_mtime: float = _AUTH_PROFILES_PATH.stat().st_mtime if _AUTH_PROFILES_PATH.exists() else 0.0

    @property
    def semantic(self) -> SemanticMemory:
        return self._semantic

    @property
    def episodic(self) -> EpisodicMemory:
        return self._episodic

    async def _maybe_hot_reload_provider(self) -> None:
        """检测 lingzhou.json 和 auth-profiles.json mtime；若已改变则热换 provider 和相关组件。"""
        if not self._cfg_file.exists():
            return
        mtime = self._cfg_file.stat().st_mtime
        auth_mtime = self._auth_profiles_path.stat().st_mtime if self._auth_profiles_path.exists() else 0.0
        cfg_changed = mtime > self._cfg_mtime
        auth_changed = auth_mtime > self._auth_profiles_mtime
        if not cfg_changed and not auth_changed:
            return
        self._cfg_mtime = mtime
        self._auth_profiles_mtime = auth_mtime
        try:
            new_cfg = Config.load(self._cfg_file)
        except Exception as e:
            _log.warning("[hot-reload] 配置解析失败，跳过热换: %s", e)
            return
        old_model = self._cfg.model
        new_model = new_cfg.model
        if not cfg_changed and auth_changed:
            # 仅 token 变更，重建 provider（保持模型不变）
            _log.info("[hot-reload] 检测到 auth-profiles.json 变更，重建 provider")
            try:
                await self._provider.close()
            except Exception:
                pass
            for _rp in self._routing_providers.values():
                try: await _rp.close()
                except Exception: pass
            self._cfg = new_cfg
            self._provider = create_provider(new_cfg)
            self._judgment = JudgmentLayer(self._provider, self._registry, new_cfg)
            self._evolution = EvolutionEngine(new_cfg, self._provider, self._registry)
            self._routing_providers = _build_routing_providers(new_cfg)
            self._judgment.set_routing_providers(self._routing_providers)
            console.print("[green]✓ 检测到 token 更新，provider 已重建[/green]")
            return
        if old_model == new_model:
            # 其他配置变更；静默更新 cfg 引用
            self._cfg = new_cfg
            return
        _log.info("[hot-reload] 检测到模型变更: %s → %s，开始热换 provider", old_model, new_model)
        try:
            await self._provider.close()
        except Exception:
            pass
        for _rp in self._routing_providers.values():
            try: await _rp.close()
            except Exception: pass
        self._cfg = new_cfg
        self._provider = create_provider(new_cfg)
        self._judgment = JudgmentLayer(self._provider, self._registry, new_cfg)
        self._evolution = EvolutionEngine(new_cfg, self._provider, self._registry)
        self._routing_providers = _build_routing_providers(new_cfg)
        self._judgment.set_routing_providers(self._routing_providers)
        console.print(f"[green]✓ 模型热换完成:[/green] {old_model} → [bold cyan]{new_model}[/bold cyan]")

    def _make_ctx(self) -> ToolContext:
        return ToolContext(
            config=self._cfg,
            wm=self._wm,
            task_store=self._task_store,
            episodic=self._episodic,
            semantic=self._semantic,
            emotion=self._emotion,
        )

    async def open(self) -> None:
        """打开数据库连接、执行启动引导和状态恢复。interact 模式下替代 run() 前两步。"""
        await self._task_store.open()
        await ensure_models_json(self._cfg)
        self._routing_providers = _build_routing_providers(self._cfg)
        self._judgment.set_routing_providers(self._routing_providers)
        await self._soul.bootstrap(self._judgment)
        await self._restore_state_from_db()

    async def run(self) -> None:
        await self._task_store.open()
        cfg = self._cfg

        await ensure_models_json(cfg)
        self._routing_providers = _build_routing_providers(cfg)
        self._judgment.set_routing_providers(self._routing_providers)
        await self._soul.bootstrap(self._judgment)
        await self._restore_state_from_db()

        console.print(Panel(
            f"[bold green]lingzhou[/bold green] 启动\n"
            f"provider={cfg.model}  idle_gap={cfg.loop.max_idle_gap}s  "
            f"act={'yes' if cfg.loop.act else 'dry-run'}",
            title="🌱 认知循环"
        ))

        cycle = 0
        consecutive_errors = 0

        try:
            while True:
                cycle += 1

                try:
                    # 检查是否有待处理的 chat 消息（CLI chat 命令注入）
                    chat_msg = await self._task_store.pop_pending_chat_message()
                    if chat_msg:
                        reply = await self._tick(cycle, user_message=chat_msg["content"])
                        # 内层循环已兜底兜底 reply；若 tick 意外返回空串，也写一条 ACK 防超时
                        await self._task_store.add_chat_message(
                            "assistant",
                            reply or "（请求已处理，任务正在后台继续）",
                            chat_msg["session_id"],
                        )
                    else:
                        await self._tick(cycle)
                    consecutive_errors = 0
                except Exception:
                    consecutive_errors += 1
                    console.print_exception(max_frames=5)
                    if consecutive_errors >= cfg.loop.max_consecutive_errors:
                        console.print(
                            f"[red]连续错误 {consecutive_errors} 次，暂停循环[/red]"
                        )
                        break

                # 事件驱动时序（Active Inference 理念：由事件/预测误差驱动，而非时钟节拍）
                # act + 有任务 → min_act_gap（2s）让工具副作用落地，立即续跑
                # 其他         → _wait_for_event()，chat/task/heartbeat 任一事件即唤醒
                after_task = await self._task_store.get_active()
                if self._last_decision == "act" and after_task is not None:
                    await asyncio.sleep(cfg.loop.min_act_gap)
                else:
                    await self._wait_for_event(cfg.loop.max_idle_gap, after_task)
                # 事件驱动等待结束后检测配置变更（模型热换）
                await self._maybe_hot_reload_provider()
                cfg = self._cfg  # 可能已更新
        finally:
            await self._task_store.close()
            await self._provider.close()
            for _rp in self._routing_providers.values():
                try:
                    await _rp.close()
                except Exception:
                    pass

    async def _wait_for_event(self, max_wait: float, before_task) -> None:
        """事件驱动等待：chat 消息、task 状态变化、超时三类事件任一发生即唤醒。

        设计依据（Active Inference / Global Workspace Theory）：
        认知唤醒由外部事件或内部预测误差阈值驱动，而非固定时钟节拍。
        max_wait 是兜底超时（防止永久沉睡），不是轮询周期。
        """
        cfg = self._cfg
        poll = cfg.loop.wake_poll_interval
        before_sig = (
            before_task.id if before_task else None,
            before_task.status if before_task else None,
        )
        deadline = asyncio.get_event_loop().time() + max_wait
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(poll, remaining))
            # 事件1：chat 消息到达 → 立即唤醒处理用户输入
            if await self._task_store.has_pending_chat_message():
                _log.debug("[wake] chat 消息到达，提前唤醒")
                break
            # 事件2：task 状态变化 → 立即响应任务推进
            if cfg.loop.wake_on_task_change:
                now = await self._task_store.get_active()
                now_sig = (now.id if now else None, now.status if now else None)
                if now_sig != before_sig:
                    _log.info("[wake] task 状态变化 %s → %s", before_sig, now_sig)
                    break
            # 事件3：max_wait 超时 → 自主思考节律兜底（60s 默认，非固定 30s）

    async def _tick(self, cycle: int, user_message: str = "") -> str:
        """执行一轮完整认知 tick，返回 reply_to_user（interact 模式时非空）。"""
        cfg = self._cfg
        ctx = self._make_ctx()

        # 1. 感知
        active_task = await self._task_store.get_active()

        # 调度器：检查到期用户 cron 信号 → 注入 WM（心跳不走此路径）
        for sig in await self._task_store.due_signals():
            self._wm.add(WMItem(
                kind="scheduler",
                content=f"[提醒] {sig['title']}",
                priority=_WM_PRI_SIGNAL,
            ))
            await self._task_store.ack_signal(sig["id"])
            _log.info("[scheduler] signal fired: %s", sig["title"])

        # 心跳自检：系统级计时（monotonic），独立于用户 cron。
        # HEARTBEAT.md 定义检查清单，LLM 自主决定是否行动（静默回复 HEARTBEAT_OK）。
        # 参考 OpenClaw HeartbeatRunner：heartbeat 是独立定时机制，不是 DB 任务。
        _now = time.monotonic()
        if _now - self._last_heartbeat_at >= self._cfg.loop.heartbeat_interval:
            _hb_path = self._cfg.workspace_dir / "HEARTBEAT.md"
            if _hb_path.exists():
                try:
                    _hb_md = _hb_path.read_text(encoding="utf-8").strip()
                    if _hb_md:
                        self._wm.add(WMItem(
                            kind="heartbeat",
                            content=f"[心跳自检]\n{_hb_md}",
                            priority=_WM_PRI_SIGNAL,
                        ))
                        _log.info("[heartbeat] 注入 WM，间隔 %ds", self._cfg.loop.heartbeat_interval)
                except Exception:
                    pass
            self._last_heartbeat_at = _now

        # tick 间连续性：上轮 next_step 是否被执行？（首轮为 None）
        _next_step_fulfilled: bool | None = None
        if self._last_next_step:
            _next_step_fulfilled = (self._last_decision == "act")
        percept = await self._perception.sense(
            self._wm, active_task,
            last_next_step=self._last_next_step,
            last_decision=self._last_decision,
        )

        # 1b. 持久化感知事件 → episodic events.jsonl（追加式，cat 直读）
        self._episodic.record_event("perception", {
            "prediction_error": round(percept.prediction_error, 4),
            "workspace_dirty": percept.workspace_dirty,
            "wm_pressure": round(self._wm.pressure, 4),
        })

        # 1c. 一次 IO 读取 perception + emotion 事件，减少文件扫描次数
        _events_batch = self._episodic.list_events_multi(["perception", "emotion"], limit=8)
        perception_events = _events_batch["perception"]
        perception_replay = build_perception_replay(
            perception_events,
            high_error_threshold=cfg.thresholds.prediction_error_task,
        )

        # 2. 认知信号计算（只统计内部状态，不产生决策；信号注入 LLM 上下文后由 LLM 自主决定响应方式）
        if active_task is None:
            self._idle_cycles += 1
        else:
            self._idle_cycles = 0

        cognitive_signals = self._perception.derive_cognitive_signals(
            percept, self._wm, self._emotion, cfg,
            has_active_task=active_task is not None,
            idle_cycles=self._idle_cycles,
            next_step_fulfilled=_next_step_fulfilled,
        )
        # 注入结构化循环探针
        self._behavior.apply_cognitive_probe(cognitive_signals)

        # 3a. 情绪更新（在判断前）：OCC 评价理论，感知信号确定性推导
        failures_recent = await self._task_store.list_failures(limit=5)
        self._emotion.derive_from_signals(
            failure_count=len(failures_recent),
            prediction_error=percept.prediction_error,
            wm_pressure=self._wm.pressure,
            workspace_dirty=percept.workspace_dirty,
            alpha=cfg.emotion.ema_alpha,
            high_error_streak=perception_replay.high_error_streak,
            replay_trend=perception_replay.trend,
            has_active_task=active_task is not None,
            has_next_step=bool(active_task and active_task.next_step),
            task_status=active_task.status if active_task else "",
        )

        # 3b. 持久化情绪事件 → episodic events.jsonl
        self._episodic.record_event("emotion", {
            "valence": round(self._emotion.valence, 4),
            "arousal": round(self._emotion.arousal, 4),
            "dominance": round(self._emotion.dominance, 4),
            "dominant": self._emotion.dominant,
            "regulation_strategy": self._emotion.regulation.strategy,
            "regulation_reason": self._emotion.regulation.reason,
        })

        # 3c. 构建情绪重放 + Ethos + JudgmentSignals（复用已读批次，无需再次 IO）
        emotion_replay = build_emotion_replay(_events_batch["emotion"])

        ethos_baseline_json, _ = await self._task_store.get_fact("soul:ethos_baseline")
        ethos_baseline = json.loads(ethos_baseline_json) if ethos_baseline_json else None
        ethos_state = derive_ethos_state(
            failure_count=len(failures_recent),
            high_error_streak=perception_replay.high_error_streak,
            has_active_task=active_task is not None,
            has_next_step=bool(active_task and active_task.next_step),
            perception_trend=perception_replay.trend,
            emotion_down_regulate_streak=emotion_replay.down_regulate_streak,
            baseline=ethos_baseline,
            ema_alpha=cfg.soul.ethos_ema_alpha,
            floor_truth=cfg.soul.ethos_floor_truth,
            floor_caution=cfg.soul.ethos_floor_caution,
        )

        # EMA 写回：灵魂随每次经历缓慢漂移（derive_ethos_state 内已做 EMA 混合，直接持久化结果）
        await self._task_store.set_fact("soul:ethos_baseline", json.dumps({
            "truth":      ethos_state.values.truth,
            "caution":    ethos_state.values.caution,
            "continuity": ethos_state.values.continuity,
            "curiosity":  ethos_state.values.curiosity,
            "care":       ethos_state.values.care,
        }))

        signals = compute_judgment_signals(
            failure_count=len(failures_recent),
            high_error_streak=perception_replay.high_error_streak,
            perception_trend=perception_replay.trend,
            emotion_state=self._emotion,
        )
        axioms_json, _ = await self._task_store.get_fact("soul:hard_axioms")
        hard_boundaries: list[str] = json.loads(axioms_json) if axioms_json else []

        # 3d. 判断（传入 ethos/signals/hard_boundaries/replay）
        # 好奇心驱动的确定性任务生成（空闲 + 高好奇心 → 自动探索任务）
        if active_task is None:
            await self._maybe_curiosity_task(ethos_state)

        # 按请求计费聚合门控：
        # 仅在空闲（无活跃任务、无用户消息、WM 中无高优先级外部信号）且 judge_every > 1 时生效。
        # 有任务或有用户消息时始终调用 LLM，不受此限制。
        _has_external_signal = any(
            item.get("kind") in ("heartbeat", "scheduler") for item in self._wm.get_top(20)
        )
        _skip_llm = (
            cfg.loop.judge_every > 1
            and not user_message
            and active_task is None
            and not _has_external_signal
            and self._ticks_since_judge < cfg.loop.judge_every - 1
        )
        if _skip_llm:
            self._ticks_since_judge += 1
            action = JudgmentOutput.wait(
                reason=f"[按请求聚合] 空闲跳过 LLM（{self._ticks_since_judge}/{cfg.loop.judge_every}）"
            )
            _log.debug(
                "[loop] tick=%d 跳过 LLM 判断（聚合 %d/%d）",
                cycle, self._ticks_since_judge, cfg.loop.judge_every,
            )
        else:
            # thinking 覆盖：
            #   有用户消息 → chat_thinking（默认 low，~3-10s，保证响应性）
            #   自主循环   → autonomous_thinking（默认 medium，~10-20s，平衡质量与速度）
            #   两者均与顶层 thinking 相同时不传 override（保持原有行为）
            _thinking_override: str | None = None
            if user_message:
                if cfg.loop.chat_thinking != cfg.thinking:
                    _thinking_override = cfg.loop.chat_thinking
            else:
                if cfg.loop.autonomous_thinking != cfg.thinking:
                    _thinking_override = cfg.loop.autonomous_thinking
            action = await self._judgment.decide(
                percept, self._wm, self._task_store, self._episodic, self._semantic, self._emotion,
                user_message=user_message,
                ethos_state=ethos_state,
                judgment_signals=signals,
                hard_boundaries=hard_boundaries,
                perception_replay=perception_replay,
                cognitive_signals=cognitive_signals,
                thinking_override=_thinking_override,
            )
            self._ticks_since_judge = 0

        # 决策结果输出到 stdout
        _model_tag = f" model={cfg.model} thinking={cfg.thinking}" if cfg.thinking != "off" else f" model={cfg.model}"
        console.print(
            f"[bold cyan][loop][/bold cyan] tick={cycle} "
            f"decision={action.decision} tool={action.chosen_action_id}"
            f"[dim]{_model_tag}[/dim]"
        )
        _log.info(
            "[loop] tick=%d decision=%s tool=%s model=%s thinking=%s rationale=%s",
            cycle, action.decision, action.chosen_action_id, cfg.model, cfg.thinking,
            (action.rationale or "")[:120],
        )

        # 3.5 行为模式感知（act 时追踪，wait/fallback 不计入）
        if action.decision == "act":
            _tool_id = action.chosen_action_id or ""
            _key_param = (action.params or {}).get("path") or (action.params or {}).get("name") or ""
            _cur_task_id = str(active_task.id) if active_task else None
            for _item in self._behavior.on_act(_tool_id, _key_param, _cur_task_id):
                self._wm.add(_item)

        # 4. 执行前本地硬门控：重复循环时强制 wait
        action = self._behavior.apply_execution_gate(action, cognitive_signals)

        # 5. 执行
        result = await self._execution.dispatch(action, ctx)

        # 5a. file.read 去重感知：只对"读取到相同内容"发出循环警告
        if (
            action.decision == "act"
            and (action.chosen_action_id or "") == "file.read"
            and not result.error
        ):
            _path = (action.params or {}).get("path") or ""
            _max_chars = int((action.params or {}).get("max_chars") or 4000)
            for _item in self._behavior.on_read(_path, _max_chars, result.summary):
                self._wm.add(_item)

        # 5b. 内层工具循环（仅 chat/interact 模式：有 user_message 且首轮决策是 act）
        # 目标：让 LLM 在单次 tick 内连续调用工具直到生成回复，节省 perception 重装 token。
        # 注意：不判断 reply_to_user——首轮 act 可能包含中间 ACK（如"正在扫描..."），
        # 内层循环应继续执行工具调用，最终生成的 reply 会覆盖中间 ACK。
        if user_message and action.decision == "act":
            _tool_history: list[dict] = [{
                "tool":   action.chosen_action_id or "",
                "params": action.params or {},
                "result": result.summary,
            }]
            _affect = {"valence": self._emotion.valence, "arousal": self._emotion.arousal}
            for _inner in range(cfg.loop.max_tool_rounds - 1):
                _cont = await self._judgment.decide_continue(_tool_history, user_message=user_message)

                # 内层行为追踪
                if _cont.decision == "act":
                    _t = _cont.chosen_action_id or ""
                    _kp = (_cont.params or {}).get("path") or (_cont.params or {}).get("name") or ""
                    for _bi in self._behavior.on_act(_t, _kp, str(active_task.id) if active_task else None):
                        self._wm.add(_bi)
                _cont = self._behavior.apply_execution_gate(_cont, cognitive_signals)
                _cont_result = await self._execution.dispatch(_cont, ctx)

                # 内层 WM 写入
                if _cont_result.summary and not _cont_result.skipped:
                    _t = _cont.chosen_action_id or ""
                    _kp2 = (_cont.params or {}).get("path") or (_cont.params or {}).get("name") or (_cont.params or {}).get("title") or ""
                    _pfx = f"[{_t}{'  ' + _kp2[:40] if _kp2 else ''}] "
                    self._wm.add(WMItem(kind=_t or _cont_result.kind, content=_pfx + _cont_result.summary, priority=_cont_result.priority))
                # 内层 rationale → 情节记忆
                if _cont.rationale:
                    self._episodic.record(role="assistant", content=f"[inner-{_inner + 1}] {_cont.rationale}",
                                         task_id=str(active_task.id) if active_task else None, affect=_affect)

                if _cont.decision == "act":
                    # 无论成功还是报错都追加历史，避免 LLM 重复调同一个失败工具
                    _result_text = (
                        f"ERROR: {_cont_result.summary}" if _cont_result.error else _cont_result.summary
                    )
                    _tool_history.append({
                        "tool":   _cont.chosen_action_id or "",
                        "params": _cont.params or {},
                        "result": _result_text,
                    })

                action = _cont
                result = _cont_result
                if action.reply_to_user or action.decision != "act":
                    break

            # 内层循环结束仍无回复时给用户兜底 ACK
            if not action.reply_to_user:
                action.reply_to_user = "（已执行完工具链，任务正在后台继续处理）"

        # 执行后记忆整合（结晶、WM 注入、情节记录、语义结晶、情绪反写）
        await self._post_tick_memory(action, result, active_task, cycle, user_message)

        # 9. 定期：WM → 情节记忆整合（只在 WM 真正有压力时才触发，避免机械周期强制清空）
        if cycle % cfg.loop.consolidate_every == 0:
            if self._wm.pressure >= self._cfg.thresholds.wm_pressure_task:
                await self._consolidate(active_task)
            # 将最新 EMA 值同步写回 SOUL.md（人类可读镜像）
            await self._soul.sync_md()

        # 10. 自进化检查：由内环失败模式驱动（Reflexion 2023 双环纠偏原则）
        _should_evolve = (
            cfg.evolution.enabled and (
                perception_replay.high_error_streak >= 3
                or cycle % cfg.loop.evolve_every == 0
            )
        )
        if _should_evolve:
            results = await self._evolution.run(ctx)
            for r in results:
                if r.success:
                    console.print(f"[green][evolution] {r.target} 已进化[/green]")
                    if r.target.startswith("prompt:"):
                        prompt_key = r.target.split(":", 1)[1]
                        self._judgment.reload_prompt(prompt_key)
            # 进化后刷新身份前缀：evolution 可能已修改 BOOTSTRAP.md / IDENTITY.md
            await self._soul.refresh_identity(self._judgment)

        # tick 间状态更新（下轮感知用）
        self._last_next_step = action.next_step or ""
        self._last_decision = action.decision

        # 情绪状态持久化（跨重启情绪连续性，与 ethos_baseline 对称）
        await self._task_store.set_fact("soul:emotion_state", json.dumps({
            "valence":   round(self._emotion.valence, 4),
            "arousal":   round(self._emotion.arousal, 4),
            "dominance": round(self._emotion.dominance, 4),
        }))

        return action.reply_to_user

    async def _restore_state_from_db(self) -> None:
        """从 DB 恢复上次持久化的情绪状态，实现跨重启情绪连续性。"""
        _em_json, _em_found = await self._task_store.get_fact("soul:emotion_state")
        if _em_found and _em_json:
            try:
                _em = json.loads(_em_json)
                self._emotion.valence   = float(_em.get("valence",   self._emotion.valence))
                self._emotion.arousal   = float(_em.get("arousal",   self._emotion.arousal))
                self._emotion.dominance = float(_em.get("dominance", self._emotion.dominance))
            except Exception:
                pass

    async def _post_tick_memory(
        self,
        action: JudgmentOutput,
        result: Any,
        active_task: Any,
        cycle: int,
        user_message: str,
    ) -> None:
        """执行后记忆整合：结晶、WM 注入、情节记录、语义结晶、情绪反写。

        从 _tick 提取，使主循环只做编排，不包含存储业务逻辑。
        步骤 4b-8（结晶 → WM → episodic → semantic → emotion EMA 反写）。
        """
        # 4b. 任务完成兜底结晶（macro-crystallization）
        # task.complete 工具已对 done 做结晶，此处兜底 failed 或未经工具的 done
        if active_task and active_task.status not in ("done", "failed"):
            refreshed = await self._task_store.get_task_by_id(active_task.id)
            if refreshed and refreshed.status in ("done", "failed"):
                _marker = f"crystallized:{refreshed.id}"
                _, _already = await self._task_store.get_fact(_marker)
                if not _already:
                    _narrative = self._episodic.load_for_context(str(refreshed.id), max_chars=1200)
                    if _narrative.strip():
                        _nid = f"task_summary_{refreshed.id}"
                        self._semantic.upsert(MemoryNode(
                            id=_nid,
                            kind="task_summary",
                            title=f"[{refreshed.status}] {refreshed.title[:60]}",
                            body=_narrative[-800:],
                            activation=0.9 if refreshed.status == "done" else 0.7,
                            valence=self._emotion.valence,
                            tags=["task_summary", refreshed.status, f"task_{refreshed.id}"],
                        ))
                    await self._task_store.set_fact(_marker, "1", scope="system")

        # 5. 结果写入 WM（kind=tool_id，让反循环规则能识别来源）
        if result.summary and not result.skipped:
            tool_id = action.chosen_action_id or ""
            params = action.params or {}
            key_param = params.get("path") or params.get("name") or params.get("title") or ""
            wm_prefix = f"[{tool_id}{'  ' + key_param[:40] if key_param else ''}] "
            self._wm.add(WMItem(
                kind=tool_id or result.kind,
                content=wm_prefix + result.summary,
                priority=result.priority,
            ))

        # 6. 内部独白写入情节记忆（Tulving 1983 四元素绑定：WHAT+WHEN+CONTEXT+AFFECT）
        _affect = {"valence": self._emotion.valence, "arousal": self._emotion.arousal}
        if action.rationale:
            self._episodic.record(
                role="assistant",
                content=f"[cycle={cycle}] {action.rationale}",
                task_id=str(active_task.id) if active_task else None,
                affect=_affect,
            )

        # 7. reflection → 语义记忆 + 情绪效价弱反写（P1-B，delta ≤ 0.05）
        if action.reflection:
            _node_id = f"insight_{hashlib.md5(action.reflection.encode()).hexdigest()[:10]}"
            self._semantic.upsert(MemoryNode(
                id=_node_id,
                kind="learned_insight",
                title=action.reflection[:60],
                body=action.reflection,
                activation=0.9,
                valence=self._emotion.valence,
                tags=["reflection", active_task.title[:20] if active_task else "free"],
            ))
            _ref_valence = _infer_valence_from_text(action.reflection, self._emotion.valence)
            _delta = _ref_valence - self._emotion.valence
            if abs(_delta) > 0.01:
                self._emotion.valence = round(
                    self._emotion.valence + min(max(_delta, -0.05), 0.05), 4
                )

        # 7b. 事件结晶：每 N 轮 reflection → kind="event" 节点（Park et al. 2023 重要性模型）
        #     零额外 LLM call：直接从 LLM 产出的 reflection 蒸馏，积累当天对话摘要
        if action.reflection and active_task:
            _turns_key = f"chat:{active_task.id}:turns"
            _turns_val, _ = await self._task_store.get_fact(_turns_key)
            _turns = int(_turns_val or "0") + 1
            await self._task_store.set_fact(_turns_key, str(_turns), scope="system")
            _crystallize_every = self._cfg.memory.chat_crystallize_every
            if _turns % _crystallize_every == 0:
                _ts_label = datetime.now(UTC).strftime("%Y-%m-%d")
                _evt_id = f"event-task{active_task.id}-{_ts_label}"
                _existing = self._semantic.get(_evt_id)
                if _existing:
                    # 同一天：追加 reflection，保持最近 600 字
                    _existing.body = (_existing.body + f"\n— {action.reflection[:200]}")[-600:]
                    _existing.activation = min(1.0, _existing.activation + 0.05)
                    self._semantic.upsert(_existing)
                else:
                    _source = getattr(active_task, "source", "") or ""
                    _chat_id = _source[5:] if _source.startswith("chat:") else _source
                    _tags = ["event", _ts_label]
                    if _chat_id:
                        _tags.append(_chat_id)
                    self._semantic.upsert(MemoryNode(
                        id=_evt_id,
                        kind="event",
                        title=f"[{_ts_label}] {active_task.title[:40]}",
                        body=action.reflection[:400],
                        activation=0.85,
                        valence=self._emotion.valence,
                        tags=_tags,
                    ))

        # 8. 用户消息 & 回复写入情节记忆（Ricoeur 叙事连续性）
        if user_message:
            self._episodic.record(
                role="user",
                content=user_message,
                task_id=str(active_task.id) if active_task else None,
                source_type="human",
            )
            if action.reply_to_user:
                self._episodic.record(
                    role="assistant_reply",
                    content=action.reply_to_user,
                    task_id=str(active_task.id) if active_task else None,
                    affect=_affect,
                )

    @property
    def task_store(self) -> TaskStore:
        return self._task_store

    @property
    def provider(self):
        return self._provider

    async def tick_interact(self, cycle: int, user_message: str) -> str:
        """interact 命令的单次入口：完整内环 + 返回 reply_to_user。

        P0-C: 将近期对话历史注入 WM，让 LLM 在判断时能回顾上下文。
        每次完整交互后记录 (user, reply) pair，最多保留 6 轮。
        """
        # 将近期对话历史作为高优先级 WM 条目注入
        if self._conv_history:
            hist_text = "\n".join(
                f"[用户] {u}\n[灵舟] {a}" for u, a in self._conv_history
            )
            self._wm.add(WMItem(
                kind="conversation_history",
                content=f"[近期对话记录]\n{hist_text}",
                priority=_WM_PRI_HISTORY,
            ))
        reply = await self._tick(cycle, user_message=user_message)
        # Hermes 借鉴：剥离 LLM 输出中意外泄露的 <memory-context> 标签内容
        if reply:
            reply = _strip_memory_context(reply)
        if reply:
            self._conv_history.append((user_message, reply))
        return reply

    async def state_snapshot(self) -> dict[str, Any]:
        """返回当前可见状态快照，供 interact REPL 渲染（Clark & Schaefer 1989 基础共识）。

        P2-A: 扩展字段，包含行为循环探针、空闲计数、WM 压力等诊断信息。
        """
        active_task = await self._task_store.get_active()
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
            "action_streak": _bt["action_streak"],
            "read_streak": _bt["read_streak"],
            "loop_probe_version": _bt["loop_probe_version"],
            "conv_history_len": len(self._conv_history),
            "fts5_ok": self._semantic.fts5_ok,
        }

    async def _maybe_curiosity_task(self, ethos_state: Any) -> None:
        """P1-C: 好奇心阈值驱动的自主探索任务生成（确定性触发，不依赖 LLM 自发）。

        触发条件（全部满足）：
        1. 当前无活跃任务
        2. 空闲周期 >= thresholds.curiosity_idle_min_cycles
        3. ethos.curiosity >= thresholds.curiosity_idle_task
        4. 最近 10 个任务中无 source=curiosity 且状态未完成的任务（防重复）
        """
        cfg = self._cfg
        if self._idle_cycles < cfg.thresholds.curiosity_idle_min_cycles:
            return
        curiosity = getattr(ethos_state.values, "curiosity", 0.0) if ethos_state else 0.0
        if curiosity < cfg.thresholds.curiosity_idle_task:
            return
        # 防重复：最近 10 任务中若已有未完成的 curiosity 任务则跳过
        recent = await self._task_store.list_tasks(limit=10)
        for t in recent:
            if (
                getattr(t, "source", None) == "curiosity"
                and getattr(t, "status", "done") not in ("done", "failed")
            ):
                return
        await self._task_store.add_task(
            title="自主探索：回顾近期经历并整合语义记忆",
            goal="回顾最近情节记忆和工作记忆中的洞察，提炼新的 reflection 写入语义记忆，更新自我认知",
            priority="low",
            source="curiosity",
        )
        _log.info(
            "[curiosity] idle=%d curiosity=%.2f → 自动生成探索任务",
            self._idle_cycles, curiosity,
        )

    async def _consolidate(self, active_task: Task | None) -> None:
        """将 WM 高优先级条目写入情节记忆，然后清空 WM，保留身份锚点。"""
        items = self._wm.get_top(10)
        if not items:
            return
        task_id = str(active_task.id) if active_task else None
        summary = "\n".join(f"- [{i['kind']}] {i['content']}" for i in items)
        self._episodic.record(role="consolidation", content=summary, task_id=task_id)
        # 保留身份锚点（bootstrap_identity），不参与周期轮换
        self._wm.clear(preserve_kinds={"bootstrap_identity"})
        # 同步感知基准，避免下一轮因 WM 大小骤降产生假预测误差
        self._perception.reset_wm_baseline(len(self._wm))
        _log.info("[consolidate] WM→episodic %d items, WM cleared (bootstrap preserved)", len(items))


# ── 模块级辅助函数 ────────────────────────────────────────────────────────────

def _build_routing_providers(cfg: "Config") -> dict:
    """根据 cfg.routing 构建分层路由 providers 字典。

    routing = {"simple": "bailian/qwen3.6-plus", "complex": "copilot/gpt-5.4"}
    如果某个 tier 的 model_ref 与主模型相同或未配置，则跳过（避免重复创建连接）。
    """
    if not cfg.routing:
        return {}
    providers: dict = {}
    for tier, model_ref in cfg.routing.items():
        if not model_ref or model_ref == cfg.model:
            continue
        try:
            providers[tier] = create_provider_with_model(cfg, model_ref)
            _log.info("[routing] tier=%s model=%s", tier, model_ref)
        except Exception as e:
            _log.warning("[routing] tier=%s model=%s 创建失败，跳过: %s", tier, model_ref, e)
    return providers
