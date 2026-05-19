"""core/loop/startup.py - loop 启动装配与状态恢复。"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.config import Config
from core.self_model import SelfModel
from provider import create_provider_with_model
from provider.models_gen import ensure_models_json

_log = logging.getLogger("lingzhou.loop")

def _build_routing_providers(cfg: Config) -> dict[str, Any]:
    """根据 cfg.routing 构建分层路由 providers 字典。"""
    if not cfg.routing:
        return {}
    providers: dict[str, Any] = {}
    for tier, model_ref in cfg.routing.items():
        if not model_ref or model_ref == cfg.model:
            continue
        try:
            providers[tier] = create_provider_with_model(cfg, model_ref)
            _log.info("[routing] tier=%s model=%s", tier, model_ref)
        except Exception as exc:
            _log.warning("[routing] tier=%s model=%s 创建失败,跳过: %s", tier, model_ref, exc)
    return providers


def _routing_summary_text(cfg: Config, routing_providers: dict[str, Any]) -> str:
    routing_lines: list[str] = []
    for tier, model_ref in cfg.routing.items():
        if model_ref == cfg.model:
            routing_lines.append(f"  {tier}: {model_ref} (= main, no separate provider)")
        elif tier in routing_providers:
            routing_lines.append(f"  {tier}: {model_ref} ✓")
        else:
            routing_lines.append(f"  {tier}: {model_ref} ✗ MISSING - provider 创建失败,实际回退至 {cfg.model}")
    if cfg.routing and not routing_providers:
        _log.warning(
            "[routing] 所有 routing provider 均创建失败,整个 routing 降级为单模型 %s。"
            "请检查各 provider 的 API key 环境变量是否已设置。",
            cfg.model,
        )
    return "\n".join(routing_lines) if routing_lines else "  (无路由配置,全部使用主模型)"


async def _open_runtime_impl(loop: Any) -> None:
    await loop._task_store.open()
    await ensure_models_json(loop._cfg)
    loop._routing_providers = _build_routing_providers(loop._cfg)
    loop._judgment.set_routing_providers(loop._routing_providers)
    loop._bootstrap_mode = await loop._soul.bootstrap(loop._judgment, run_kind="interactive")
    # 探针系统：从 probes.json 加载（已在 ProbeManager.__init__ 同步完成），启动调度 Task
    await loop._probe_manager.start(loop._wm, loop_ref=loop)
    await _restore_state_from_db_impl(loop)


async def _prepare_runtime_run_impl(loop: Any) -> tuple[Config, str]:
    await loop._task_store.open()
    cfg = loop._cfg
    await ensure_models_json(cfg)
    loop._routing_providers = _build_routing_providers(cfg)
    loop._judgment.set_routing_providers(loop._routing_providers)
    loop._bootstrap_mode = await loop._soul.bootstrap(loop._judgment, run_kind="interactive")
    loop._judgment.self_model.record_start(name="lingzhou")
    loop._judgment.self_model.set_routing(cfg)
    await _restore_self_model_impl(loop)
    await _restore_state_from_db_impl(loop)
    return cfg, _routing_summary_text(cfg, loop._routing_providers)


async def _restore_state_from_db_impl(loop: Any) -> None:
    """从 DB 恢复上次持久化的状态，实现跨重启连续性。"""
    emotion_json, emotion_found = await loop._task_store.get_fact("soul:emotion_state")
    if emotion_found and emotion_json:
        try:
            emotion = json.loads(emotion_json)
            loop._emotion.valence = float(emotion.get("valence", loop._emotion.valence))
            loop._emotion.arousal = float(emotion.get("arousal", loop._emotion.arousal))
            loop._emotion.dominance = float(emotion.get("dominance", loop._emotion.dominance))
        except Exception:
            pass

    overrides_json, overrides_found = await loop._task_store.get_fact("pref:routing_overrides")
    if overrides_found and overrides_json:
        try:
            overrides = json.loads(overrides_json)
            if isinstance(overrides, dict) and overrides:
                loop._pending_routing_overrides = {
                    key: value
                    for key, value in overrides.items()
                    if key in {"reader", "reasoner", "repair"} and isinstance(value, str) and value
                } or None
                if loop._pending_routing_overrides:
                    _log.info("[routing] 从 DB 恢复 routing_overrides: %s", loop._pending_routing_overrides)
        except Exception:
            pass

    zombie_count = await loop._task_store.reset_in_progress_tasks()
    if zombie_count > 0:
        _log.info("[restart] 重置 %d 个 in_progress 任务为 pending", zombie_count)

    # ── 崩溃连续性恢复：读取 survival.json，若上次非干净退出则注入 WM 感知 ──
    _inject_crash_recovery(loop)


def _inject_crash_recovery(loop: Any) -> None:
    """若上次为崩溃退出，将崩溃摘要注入 WM，让 LLM 在第一轮 tick 感知到。"""
    try:
        _sp = loop._cfg.state_dir / "survival.json"
        if not _sp.exists():
            return
        snap = json.loads(_sp.read_text(encoding="utf-8"))
        if snap.get("exit_type") == "clean":
            return
        # 上次是 crash 退出
        tick = snap.get("tick", "?")
        ts = snap.get("ts", "未知时间")
        task_title = snap.get("active_task_title") or "无"
        task_goal = snap.get("active_task_goal") or ""
        last_action = snap.get("last_action") or "未知"
        emotion = snap.get("emotion") or {}
        valence = emotion.get("valence", 0)
        arousal = emotion.get("arousal", 0)

        content = (
            f"[崩溃恢复] 上次运行在 tick={tick}（{ts}）异常终止（非干净退出）。\n"
            f"  中断前活跃任务: 「{task_title}」\n"
        )
        if task_goal:
            content += f"  任务目标: {task_goal[:100]}\n"
        content += (
            f"  最后动作: {last_action}\n"
            f"  情绪状态: valence={valence} arousal={arousal}\n"
            "  建议: 先确认中断前的任务是否需要继续，检查是否有遗留副作用，"
            "再决定本轮行动。"
        )
        from memory.working import WMItem
        loop._wm.add(WMItem(
            kind="crash_recovery",
            content=content,
            priority=0.97,
        ))
        _log.info("[startup] 注入崩溃恢复信号: tick=%s ts=%s", tick, ts)
    except Exception as exc:
        _log.debug("[startup] 读取 survival.json 失败: %s", exc)


async def _restore_self_model_impl(loop: Any) -> None:
    """从 DB 恢复自我模型(跨重启连续性)。"""
    raw, found = await loop._task_store.get_fact("self:model")
    if found and raw:
        loop._judgment.self_model = SelfModel.from_json(raw, name="lingzhou")
        loop._judgment.self_model.set_routing(loop._cfg)
        loop._judgment.self_model.tick_count = 0
        _log.info(
            "[self_model] 已恢复: api=%d tokens=%d (tick=0 重置)",
            loop._judgment.self_model.api_call_count,
            loop._judgment.self_model.total_tokens,
        )
