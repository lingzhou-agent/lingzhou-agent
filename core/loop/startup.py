"""core/loop/startup.py - loop 启动装配与状态恢复。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.config import Config
from core.self_model import SelfModel
from provider import create_provider_with_model
from provider.models_gen import ensure_models_json

_log = logging.getLogger("lingzhou.loop")

# ── 运行时启动自检 ────────────────────────────────────────────────────────────
# 与 cli/diag.py doctor 命令共享同一套 patch 定义（字符串 patch 而非 importlib，
# 因为需要在 Config 对象创建之前修复源文件，确保下次 reload 生效）。

_MEMORY_FIELD_PATCHES: dict[str, str] = {
    "wm_item_max_tokens": (
        "    wm_item_max_tokens: int = Field(\n"
        "        default=300, ge=0,\n"
        "        description=(\n"
        "            \"工作记忆单条 content token 上限（估算）；超出时自动截断并追加省略提示。\"\n"
        "            \"0 = 不限制。调优请在 lingzhou.json 的 memory 区块覆盖，不要修改此处 default 值。\"\n"
        "        ),\n"
        "    )\n"
    ),
}

_THRESHOLDS_FIELD_PATCHES: dict[str, str] = {
    "skill_max_inject": (
        "    skill_max_inject: int = Field(\n"
        "        default=3, ge=1, le=8,\n"
        "        description=\"单次 tick 最多注入技能数；压力大时可通过配置增加护栏覆盖\"\n"
        "    )\n"
    ),
    "skill_failure_threshold": (
        "    skill_failure_threshold: int = Field(\n"
        "        default=3, ge=1,\n"
        "        description=\"连续评分函数的失败次数基准点；达到此值时 failure.reflection 技能得分达到峰值\"\n"
        "    )\n"
    ),
    "skill_wm_pressure_threshold": (
        "    skill_wm_pressure_threshold: float = Field(\n"
        "        default=0.4, ge=0.0, le=1.0,\n"
        "        description=\"WM 压力连续评分基准点；达到此值时 evidence-first-change 技能得分达到峰值\"\n"
        "    )\n"
    ),
    "skill_min_budget_tokens": (
        "    skill_min_budget_tokens: int = Field(\n"
        "        default=80, ge=0,\n"
        "        description=\"上下文预算裁剪时 skills_section 保留的最小 token 数；0=可完全裁掉\"\n"
        "    )\n"
    ),
}


def _patch_config_classes(
    config_py: Path,
    class_patches: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    """向 config_py 中的多个 class 末尾注入缺失字段，单次文件 I/O。

    返回 {class_name: [注入的字段名]} 字典；无需注入的 class 不出现在结果中。
    """
    # 1. 确定各 class 的缺失字段
    try:
        import importlib as _il
        mod = _il.import_module("core.config")
    except Exception:
        return {}

    to_inject: dict[str, list[str]] = {}
    for class_name, patches in class_patches.items():
        try:
            cls = getattr(mod, class_name, None)
            if cls is None:
                continue
            instance = cls()
            missing = [f for f in patches if not hasattr(instance, f)]
            if missing:
                to_inject[class_name] = missing
        except Exception:
            continue

    if not to_inject or not config_py.exists():
        return {}

    # 2. 单次读取文件
    lines = config_py.read_text(encoding="utf-8").splitlines(keepends=True)

    # 3. 收集各 class 的插入位置，倒序注入避免行号偏移
    insertions: list[tuple[int, str]] = []
    for class_name in to_inject:
        in_class = False
        insert_at = len(lines)
        for i, line in enumerate(lines):
            if line.startswith(f"class {class_name}"):
                in_class = True
            elif in_class and line.startswith("class "):
                insert_at = i
                break
        if not in_class:
            continue
        while insert_at > 0 and not lines[insert_at - 1].strip():
            insert_at -= 1
        insertions.append((insert_at, class_name))

    if not insertions:
        return {}

    insertions.sort(key=lambda x: x[0], reverse=True)
    for insert_at, class_name in insertions:
        missing = to_inject[class_name]
        inject = ["\n"] + [class_patches[class_name][f] for f in missing]
        lines[insert_at:insert_at] = inject

    # 4. 单次写入 + 热重载
    config_py.write_text("".join(lines), encoding="utf-8")
    try:
        import importlib as _il2
        _il2.reload(_il2.import_module("core.config"))
    except Exception:
        pass

    return to_inject


def _patch_config_class(config_py: Path, class_name: str, patches: dict[str, str]) -> list[str]:
    """单 class 兼容包装，内部复用 _patch_config_classes。"""
    result = _patch_config_classes(config_py, {class_name: patches})
    return result.get(class_name, [])


def _startup_health_check(cfg: Config, project_root: Path) -> None:
    """运行时启动自检（非阻塞，仅 warn 级日志）：

    1. 确保 memory_dir / workspace_dir / db_parent 目录存在
    2. Config schema 兼容性 patch（MemoryConfig + ThresholdsConfig，单次 I/O）
    """
    # 1. 目录自动创建
    for label, raw_path in [
        ("memory_dir",    cfg.memory_dir),
        ("workspace_dir", cfg.workspace_dir),
        ("db_parent",     Path(cfg.db_path).parent),
    ]:
        p = Path(raw_path).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            _log.warning("[startup] %s 无法创建，可能导致运行异常: %s  路径=%s", label, exc, p)

    # 2. Config schema patch（一次读写）
    config_py = project_root / "core" / "config.py"
    patched = _patch_config_classes(config_py, {
        "MemoryConfig":     _MEMORY_FIELD_PATCHES,
        "ThresholdsConfig": _THRESHOLDS_FIELD_PATCHES,
    })
    for class_name, fields in patched.items():
        _log.warning("[startup] %s schema 版本过旧，已自动注入缺失字段: %s", class_name, fields)


def _build_routing_providers(cfg: Config) -> dict[str, Any]:
    """根据 cfg.routing 构建分层路由 providers 字典。"""
    if not cfg.routing:
        return {}
    from provider.catalog import lookup_model_ref, lookup_model
    providers: dict[str, Any] = {}
    for tier, model_ref in cfg.routing.items():
        if not model_ref or model_ref == cfg.model:
            continue
        # 启动期校验：若 model_id 不在指定 provider 的目录中，但存在于其他 provider，提前告警
        if "/" in model_ref:
            if lookup_model_ref(model_ref) is None:
                model_id = model_ref.split("/", 1)[1]
                provider_name = model_ref.split("/", 1)[0]
                alt = lookup_model(model_id)
                if alt is not None:
                    _log.warning(
                        "[routing] tier=%s model=%s: 模型 %r 未在 provider %r 的内置目录中注册，"
                        "但在其他 provider 中存在。请检查 routing 配置是否正确（provider 名写错）。",
                        tier, model_ref, model_id, provider_name,
                    )
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
    from core.paths import project_root as _project_root
    _startup_health_check(loop._cfg, _project_root())
    await loop._task_store.open()
    await ensure_models_json(loop._cfg)
    loop._routing_providers = _build_routing_providers(loop._cfg)
    loop._judgment.set_routing_providers(loop._routing_providers)
    loop._bootstrap_mode = await loop._soul.bootstrap(loop._judgment, run_kind="interactive")
    loop._judgment.self_model.record_start(name="lingzhou")
    loop._judgment.self_model.set_routing(loop._cfg)
    await _restore_self_model_impl(loop)
    # 探针系统：从 probes.json 加载（已在 ProbeManager.__init__ 同步完成），启动调度 Task
    await loop._probe_manager.start(loop._wm, loop_ref=loop)
    await _restore_state_from_db_impl(loop)


async def _prepare_runtime_run_impl(loop: Any) -> tuple[Config, str]:
    from core.paths import project_root as _project_root
    _startup_health_check(loop._cfg, _project_root())
    await loop._task_store.open()
    cfg = loop._cfg
    await ensure_models_json(cfg)
    loop._routing_providers = _build_routing_providers(cfg)
    loop._judgment.set_routing_providers(loop._routing_providers)
    loop._bootstrap_mode = await loop._soul.bootstrap(loop._judgment, run_kind="interactive")
    loop._judgment.self_model.record_start(name="lingzhou")
    loop._judgment.self_model.set_routing(cfg)
    await _restore_self_model_impl(loop)
    # 探针系统：从 probes.json 加载（已在 ProbeManager.__init__ 同步完成），启动调度 Task
    await loop._probe_manager.start(loop._wm, loop_ref=loop)
    await _restore_state_from_db_impl(loop)
    return cfg, _routing_summary_text(cfg, loop._routing_providers)


async def _restore_state_from_db_impl(loop: Any) -> None:
    """从 DB 恢复上次持久化的状态，实现跨重启连续性。"""
    # ── soul:born_at：首次启动写入，后续恢复；记录灵舟的诞生时刻 ──
    import time as _time
    born_json, born_found = await loop._task_store.get_fact("soul:born_at")
    if born_found and born_json:
        try:
            loop._judgment.self_model.born_at = float(born_json)
        except Exception:
            pass
    else:
        _born_ts = _time.time()
        await loop._task_store.set_fact("soul:born_at", str(_born_ts), scope="system")
        loop._judgment.self_model.born_at = _born_ts
        _log.info("[startup] soul:born_at 首次写入: %.0f", _born_ts)

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
        valence = emotion.get("valence", getattr(loop._emotion, "valence", 0))
        arousal = emotion.get("arousal", getattr(loop._emotion, "arousal", 0))

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
