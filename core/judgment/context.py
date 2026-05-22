"""core/judgment/context.py - judgment 上下文组装相关格式化与预算 helper。"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from core.probe.types import PROBE_COVERAGE_HINTS, normalize_probe_coverage_tags

_log = logging.getLogger("lingzhou.judgment")

# --- 纯计算格式化函数缓存（per-tick 粒度）---
# key = tick_id + 函数名 + hash(参数); value = 格式化结果或预算后的上下文字典
_context_fmt_cache: dict[str, Any] = {}


def _clear_context_cache() -> None:
    """在每 tick 开头调用，清除上一 tick 的所有缓存。"""
    _context_fmt_cache.clear()

if TYPE_CHECKING:
    from core.config import Config
    from core.perception import (
        CognitiveSignals,
        EmotionState,
        EthosState,
        JudgmentSignals,
        Percept,
        PerceptionReplaySummary,
    )
    from core.skill import Skill
    from memory.task_store import Failure, Run, Task, TaskStore
    from memory.semantic import SemanticMemory
    from tools.registry import ToolManifest


def _task_narrative(task: "Task | None") -> str:
    """从任务状态构建叙事线：目标 → 当前步骤 → 下一步。"""
    if not task:
        return "无"
    parts = []
    if task.goal:
        parts.append(f"目标: {task.goal[:80]}")
    if task.current_step:
        parts.append(f"进展: {task.current_step[:80]}")
    if task.next_step:
        parts.append(f"下一步: {task.next_step[:80]}")
    return " → ".join(parts) if parts else f"执行中 ({task.status})"


def _fmt_task(task: "Task | None") -> str:
    if not task:
        return "（无活跃任务，可自主探索或等待）"
    age_str = ""
    if task.created_at:
        try:
            created = datetime.fromisoformat(task.created_at.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            elapsed = datetime.now(timezone.utc) - created
            total_secs = int(elapsed.total_seconds())
            if total_secs < 60:
                age_str = f"（已进行 {total_secs}s）"
            elif total_secs < 3600:
                age_str = f"（已进行 {total_secs // 60}m）"
            elif total_secs < 86400:
                hours, minutes = divmod(total_secs // 60, 60)
                age_str = f"（已进行 {hours}h {minutes}m）"
            else:
                days, rem = divmod(total_secs, 86400)
                age_str = f"（已进行 {days}d {rem // 3600}h）"
        except Exception:
            pass
    last_run_status = str((task.result_json or {}).get("last_run_status") or "").strip()
    lines = [
        f"ID: {task.id}",
        f"标题: {task.title}{age_str}",
        f"状态: {task.status}",
        f"目标: {task.goal or '（未指定）'}",
        f"优先级: {task.priority}",
        f"模型层级: {task.model_tier or '（未指定）'}",
        f"当前步骤: {task.current_step or '（未指定）'}",
        f"下一步: {task.next_step or '（未指定）'}",
        f"叙事线: {_task_narrative(task)}",
    ]
    raw_plan = task.extras.get("plan") if isinstance(task.extras, dict) else None
    if isinstance(raw_plan, list) and raw_plan:
        status_icons = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}
        plan_lines: list[str] = []
        for index, item in enumerate(raw_plan[:5], 1):
            if not isinstance(item, dict):
                continue
            step = str(item.get("step") or "").strip()
            if not step:
                continue
            status = str(item.get("status") or "pending").strip()
            icon = status_icons.get(status, "•")
            plan_lines.append(f"  [{index}] {icon} {_clip_text(step, 80)}")
        if plan_lines:
            lines.append("当前计划:")
            lines.extend(plan_lines)
            # 有 in_progress 步骤时注入高优先级事实信号，避免计划态在判断层被忽略。
            in_progress_step = next(
                (str(s.get("step") or "").strip() for s in raw_plan if isinstance(s, dict) and s.get("status") == "in_progress"),
                None,
            )
            if in_progress_step:
                lines.append(
                    f"⚠️ 计划信号：步骤 [{in_progress_step}] 当前处于 in_progress。"
                    "若没有更强的新证据或 inbox 转向，优先直接推进这一步，而不是重新 plan。"
                )
    # inbox_messages：由 task.steer 注入的转向信号，应先评估它是否改变当前方向
    inbox: list = task.extras.get("inbox_messages") or [] if isinstance(task.extras, dict) else []
    if isinstance(inbox, list) and inbox:
        lines.append(f"⚠️ 转向信号（inbox {len(inbox)} 条，先评估这些新信号是否改变当前方向）:")
        for i, msg in enumerate(inbox[:5], 1):
            lines.append(f"  [{i}] {str(msg)[:120]}")
    if last_run_status:
        lines.append(f"最近运行状态: {last_run_status}")
    return "\n".join(lines)


def _fmt_recent_runs(runs: list["Run"]) -> str:
    cache_key = f"_fmt_recent_runs:{hash(tuple(r.id for r in runs)) if runs else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    if not runs:
        result = "（暂无近期运行记录）"
        _context_fmt_cache[cache_key] = result
        return result
    lines: list[str] = []
    for run in runs[:5]:
        summary = _clip_text(_run_summary(run), 120)
        tool = run.tool_name or run.run_type or "-"
        progress = _clip_text(run.progress.strip(), 60) if run.progress else ""
        line = f"- run#{run.id} [{run.status}] tool={tool} tier={run.model_tier or '-'}"
        if progress:
            line += f" progress={progress}"
        if summary:
            line += f" summary={summary}"
        lines.append(line)
    result = "\n".join(lines)
    _context_fmt_cache[cache_key] = result
    return result


_FACT_CONTEXT_EXCLUDE_PREFIXES = (
    "control:",
    "durable_failure:",
    "evolution:",
    "pref:",
    "run:",
    "soul:",
)


async def _load_context_facts_snapshot(
    task_store: "TaskStore",
    task: "Task | None",
    *,
    task_limit: int = 6,
    global_limit: int = 4,
) -> list[tuple[str, str]]:
    seen: set[str] = set()
    selected: list[tuple[str, str]] = []
    task_prefix = f"task:{task.id}:" if task else ""

    async def _add_facts(items: list[tuple[str, str]], limit: int) -> None:
        for key, value in items:
            if key in seen:
                continue
            if key.startswith(_FACT_CONTEXT_EXCLUDE_PREFIXES):
                continue
            if key.startswith("task:") and task_prefix and not key.startswith(task_prefix):
                continue
            if key.startswith("task:") and not task_prefix:
                continue
            seen.add(key)
            selected.append((key, value))
            if limit > 0 and len(selected) >= limit:
                return

    if task_prefix:
        task_facts = await task_store.list_facts(prefix=task_prefix, limit=task_limit)
        await _add_facts(task_facts, task_limit)

    current_global = len(selected)
    if global_limit > 0:
        recent_facts = await task_store.list_facts(limit=max(global_limit * 3, 12))
        before = len(selected)
        await _add_facts(recent_facts, current_global + global_limit)
        if len(selected) == before and not selected:
            return []

    return selected


def _format_fact_value(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "（空）"
    try:
        payload = json.loads(text)
    except Exception:
        return _clip_text(text, 180)
    if isinstance(payload, dict):
        parts = [f"{key}={payload[key]}" for key in sorted(payload)]
        return _clip_text("; ".join(parts), 180)
    if isinstance(payload, list):
        return _clip_text(", ".join(str(item) for item in payload), 180)
    return _clip_text(str(payload), 180)


def _fmt_context_facts(facts: list[tuple[str, str]]) -> str:
    cache_key = f"_fmt_context_facts:{hash(tuple(facts)) if facts else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    if not facts:
        result = "（暂无近期关键事实）"
        _context_fmt_cache[cache_key] = result
        return result
    result = "\n".join(
        f"- {key} = {_format_fact_value(value)}"
        for key, value in facts
    )
    _context_fmt_cache[cache_key] = result
    return result


def _fmt_waiting_tasks(tasks: list["Task"]) -> str:
    cache_key = f"_fmt_waiting_tasks:{hash(tuple(t.id for t in tasks)) if tasks else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    if not tasks:
        result = "（无 waiting 任务）"
        _context_fmt_cache[cache_key] = result
        return result
    lines: list[str] = []
    for task in tasks[:5]:
        wait_desc = task.wait_kind or "unknown"
        if task.wait_key:
            wait_desc += f"/{task.wait_key}"
        line = f"- task#{task.id} [{task.status}] {task.title} wait={wait_desc}"
        if task.next_step:
            line += f" next={_clip_text(task.next_step, 80)}"
        lines.append(line)
    result = "\n".join(lines)
    _context_fmt_cache[cache_key] = result
    return result


def _run_summary(run: "Run") -> str:
    if run.error_text:
        return f"error: {run.error_text.strip()}"
    for key in ("summary", "result", "message", "reply_to_user"):
        value = run.output_json.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if run.log_text.strip():
        return run.log_text.strip()
    return ""


def _clip_text(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _fmt_current_time() -> str:
    cache_key = "_fmt_current_time"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    now = datetime.now(timezone.utc)
    local_iso = now.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    utc_str = now.strftime("%Y-%m-%d %H:%M UTC")
    result = f"当前时间: {local_iso}\n参考 UTC: {utc_str}"
    _context_fmt_cache[cache_key] = result
    return result


def _fmt_chat_history(messages: list[dict[str, Any]], max_chars: int = 300) -> str:
    """将最近 N 条对话消息格式化为 LLM 可读的历史轮次。

    role=user 显示为 '用户:', role=assistant 显示为 '我:',
    每条消息截断到 max_chars，防止 token 爆炸。
    """
    cache_key = f"_fmt_chat_history:{hash(str(messages)) if messages else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    if not messages:
        result = "（暂无对话历史）"
        _context_fmt_cache[cache_key] = result
        return result
    lines = []
    for msg in messages:
        role = msg.get("role", "")
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        label = "用户" if role == "user" else "我"
        snippet = content[:max_chars] + ("…" if len(content) > max_chars else "")
        lines.append(f"{label}: {snippet}")
    result = "\n".join(lines) if lines else "（暂无对话历史）"
    _context_fmt_cache[cache_key] = result
    return result


def _fmt_wm(
    items: list[dict[str, Any]],
    wm_count: int = 0,
    wm_capacity: int = 20,
    wm_tokens: int = 0,
    wm_token_budget: int = 0,
) -> str:
    if wm_token_budget > 0:
        header = f"[{wm_count}/{wm_capacity} 条，~{wm_tokens} tokens / {wm_token_budget} 预算，{wm_tokens / wm_token_budget:.0%}]"
    else:
        header = f"[{wm_count}/{wm_capacity}，{wm_count / wm_capacity:.0%}]"
    if not items:
        return f"{header} （工作记忆为空）"
    anti_loop = [item for item in items if item.get("kind") == "self_awareness"]
    rest = [item for item in items if item.get("kind") != "self_awareness"]
    ordered = anti_loop + rest
    lines = [header] + [f"- [{item['kind']}|p={item.get('priority', 0):.2f}] {item['content']}" for item in ordered]
    # 大条目警告：top-3 超过 100 tokens 的条目，提醒可能坠占预算
    large_items = sorted(ordered, key=lambda x: _estimate_tokens(x.get("content", "")), reverse=True)[:3]
    large_items = [x for x in large_items if _estimate_tokens(x.get("content", "")) > 100]
    if large_items:
        warnings_str = ", ".join(
            f"[{x.get('kind')}] ~{_estimate_tokens(x.get('content', ''))} tokens"
            for x in large_items
        )
        lines.append(f"⚠ 大条目（可能坠占预算）: {warnings_str}")
    return "\n".join(lines)


def _fmt_failures(failures: "list[Failure]") -> str:
    if not failures:
        return "（无近期失败）"
    lines = [f"- [#{failure.id}][{failure.kind}] {failure.summary}" for failure in failures]
    return "\n".join(lines)


async def _load_durable_failure_snapshot(task_store: "TaskStore") -> dict[str, Any]:
    from core.execution import _load_durable_failure_policy

    policy = await _load_durable_failure_policy(task_store)
    muted_actions: list[dict[str, Any]] = []
    now = time.time()
    for _, raw in await task_store.list_facts(prefix="durable_failure:", limit=12):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        muted_until = float(payload.get("muted_until") or 0)
        if muted_until <= now:
            continue
        muted_actions.append({
            "tool": str(payload.get("tool") or ""),
            "key": str(payload.get("key") or "").strip(),
            "reason": str(payload.get("reason") or "stable_failure"),
            "count": int(payload.get("count") or 0),
            "remaining_sec": max(0, int(muted_until - now)),
        })
    muted_actions.sort(key=lambda item: item["remaining_sec"])
    return {
        "threshold": int(policy.get("threshold") or 0),
        "ttl_sec": int(policy.get("ttl_sec") or 0),
        "muted_actions": muted_actions[:5],
    }


def _fmt_durable_failures(snapshot: dict[str, Any]) -> str:
    threshold = int(snapshot.get("threshold") or 0)
    ttl_sec = int(snapshot.get("ttl_sec") or 0)
    lines = [f"policy: threshold={threshold} ttl_sec={ttl_sec}"]
    muted_actions = snapshot.get("muted_actions") or []
    if not muted_actions:
        lines.append("- 当前无稳定失败静默中的动作")
        return "\n".join(lines)
    for item in muted_actions:
        tool = item.get("tool") or "-"
        key = item.get("key") or ""
        reason = item.get("reason") or "stable_failure"
        count = int(item.get("count") or 0)
        remaining_sec = int(item.get("remaining_sec") or 0)
        line = f"- {tool}"
        if key:
            line += f" {key}"
        line += f" reason={reason} failures={count} remaining={remaining_sec}s"
        lines.append(line)
    return "\n".join(lines)


def _fmt_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "（无相关记忆）"
    lines: list[str] = []
    for memory in memories:
        score = memory.get("score")
        score_part = ""
        if isinstance(score, (int, float)):
            score_part = f" (score={float(score):.3f})"
        lines.append(f"- [{memory['kind']}] {memory['title']}{score_part}: {memory['body']}")
    return "\n".join(lines)


def _fmt_memory_system(
    *,
    runtime_db: str,
    memory_dir: str,
    workspace_dir: str,
    semantic: "SemanticMemory",
    max_concurrent_ticks: int = 1,
    max_tick_queue: int = 8,
) -> str:
    stats = semantic.stats()
    lines = [
        f"runtime_db: {runtime_db}",
        f"memory_dir: {memory_dir}",
        f"workspace_dir: {workspace_dir}",
        f"semantic_db: {stats.get('db_path')}",
        f"semantic_nodes_dir: {stats.get('nodes_dir')}",
        f"semantic_nodes: {int(stats.get('nodes') or 0)}",
        f"semantic_fts5_ok: {'yes' if stats.get('fts5_ok') else 'no'}",
        f"embedding_enabled: {'yes' if stats.get('embedding_enabled') else 'no'}",
        f"decay_lambda: {float(stats.get('decay_lambda') or 0.0):.3f}",
        f"tick_dispatch.max_concurrent_ticks: {int(max_concurrent_ticks)}",
        f"tick_dispatch.max_tick_queue: {int(max_tick_queue)}",
    ]
    lines.append("说明: runtime_db 是任务/事实/聊天/运行轨迹主存储；SOUL/IDENTITY/BOOTSTRAP 等 md 是身份与可读镜像层。")
    lines.append("调参提示: 以上 dispatch 上限可通过 config.set 修改 loop.max_concurrent_ticks / loop.max_tick_queue。")
    return "\n".join(lines)


def _fmt_tools(manifests: "list[ToolManifest]") -> str:
    if not manifests:
        return "（无可用工具）"
    lines: list[str] = []
    for manifest in manifests:
        params_str = ", ".join(
            f"{param.name}({'*' if param.required else '?'}): {param.description}"
            for param in manifest.params
        )
        lines.append(f"- `{manifest.name}`: {manifest.description}  参数: [{params_str}]")
    return "\n".join(lines)


def _fmt_config_snapshot(cfg: "Config") -> str:
    """格式化关键配置参数快照，让 LLM 在 judgment 中直接感知当前参数状态。

    只列可通过 config.set 调整的运行时参数，避免冗余路径/凭证信息。
    """
    ev = cfg.evolution
    lo = cfg.loop
    lines = [
        "# 可调运行时参数（通过 config.set <key> <json_value> 修改）",
        "",
        "## 模型",
        f"  model: {cfg.model}",
        f"  temperature: {cfg.temperature}",
        "",
        "## 进化引擎 (evolution.*)",
        f"  enabled: {ev.enabled}",
        f"  competitive_candidates: {ev.competitive_candidates}  # >=2 启用竞争进化",
        f"  trigger_min_failures: {ev.trigger_min_failures}",
        f"  trigger_window_minutes: {ev.trigger_window_minutes}",
        f"  error_streak_evolve: {ev.error_streak_evolve}",
        f"  ethos_max_delta: {ev.ethos_max_delta}",
        "",
        "## 认知循环 (loop.*)",
        f"  wake_poll_interval: {lo.wake_poll_interval}ms",
        f"  min_act_gap: {lo.min_act_gap}ms",
        f"  active_idle_gap: {lo.active_idle_gap}ms",
        f"  max_idle_gap: {lo.max_idle_gap}ms",
        f"  chat_reply_timeout: {lo.chat_reply_timeout}s",
    ]
    return "\n".join(lines)


def _fmt_shell_capabilities() -> str:
    cache_key = "_fmt_shell_capabilities"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    cmds = (
        "python3", "python", "bash", "sh", "grep", "find", "ls", "cat",
        "sqlite3", "git", "sed", "awk", "jq", "rg",
    )
    available = [cmd for cmd in cmds if shutil.which(cmd)]
    payload = {
        "engine": "asyncio.create_subprocess_shell",
        "execution_model": "one-shot-non-persistent",
        "sandbox": False,
        "network_policy": "inherits-host-environment",
        "default_timeout_sec": 30,
        "default_output_preview_chars": None,
        "shell": os.environ.get("SHELL") or "/bin/sh",
        "cwd": os.getcwd(),
        "available_commands": available,
        "missing_commands": [cmd for cmd in cmds if cmd not in available],
    }
    result = json.dumps(payload, ensure_ascii=False, indent=2)
    _context_fmt_cache[cache_key] = result
    return result


def _fmt_percept(percept: "Percept") -> str:
    return (
        f"预测误差: {percept.prediction_error:.2f}  "
        f"工作区变更: {'是' if percept.workspace_dirty else '否'}"
    )


def _fmt_soul(axioms_val: str, ethos_val: str) -> str:
    cache_key = f"_fmt_soul:{hash(axioms_val)}:{hash(ethos_val)}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    parts: list[str] = []
    if axioms_val:
        parts.append(f"绝对禁忌（hard_axioms）: {axioms_val}")
    if ethos_val:
        parts.append(f"价值基线（ethos_baseline）: {ethos_val}")
    result = "\n".join(parts) if parts else "（Soul 未初始化，运行 `init` 命令生成）"
    _context_fmt_cache[cache_key] = result
    return result


def _emotion_label(emotion: "EmotionState", cfg: "Config") -> str:
    ec = cfg.emotion
    valence_high, valence_low = ec.mood_valence_high, ec.mood_valence_low
    arousal_high = ec.mood_arousal_high
    if emotion.valence < valence_low and emotion.arousal > arousal_high:
        return "焦虑"
    if emotion.valence < valence_low:
        return "沮丧"
    if emotion.valence > valence_high and emotion.arousal > arousal_high:
        return "兴奋"
    if emotion.valence > valence_high:
        return "稳定"
    return "中性"


def _fill_template(template: str, ctx: dict[str, Any]) -> str:
    missing = sorted({
        match.group(1).strip()
        for match in re.finditer(r"\{\{([^}]+)\}\}", template)
        if match.group(1).strip() not in ctx
    })
    if missing:
        msg = "[judgment] 模板变量缺失: " + ", ".join(missing)
        _log.error("%s（judgment.md 与 context 组装已失配）", msg)
        raise ValueError(msg)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return str(ctx[key])

    return re.sub(r"\{\{([^}]+)\}\}", replace, template)


def _fmt_ethos(ethos_state: "EthosState | None") -> str:
    # 缓存：纯计算函数，同一 tick 内不重复计算
    cache_key = f"_fmt_ethos:{hash(ethos_state) if ethos_state else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    if not ethos_state:
        result = "（Ethos 未计算）"
        _context_fmt_cache[cache_key] = result
        return result
    values = ethos_state.values
    bias = ethos_state.bias
    lines: list[str] = [
        f"价値图式  truth={values.truth:.2f}  caution={values.caution:.2f}  continuity={values.continuity:.2f}  curiosity={values.curiosity:.2f}  care={values.care:.2f}",
    ]
    biases: list[str] = []
    if bias.prefer_verification:
        biases.append("prefer_verification")
    if bias.prefer_narrow_scope:
        biases.append("prefer_narrow_scope")
    if bias.preserve_continuity:
        biases.append("preserve_continuity")
    if bias.avoid_overclaiming:
        biases.append("avoid_overclaiming")
    if biases:
        lines.append(f"行为倾向  {', '.join(biases)}")
    if bias.reasons:
        lines.append(f"理由      {'; '.join(bias.reasons)}")
    return "\n".join(lines)


def apply_context_budget(
    ctx: dict[str, str],
    token_budget: int | None = None,
    max_chars: int | None = None,
    skill_min_tokens: int = 0,
) -> dict[str, str]:
    if token_budget is None:
        token_budget = max_chars
    if token_budget is None:
        raise TypeError("apply_context_budget() missing required argument: 'token_budget'")
    if token_budget <= 0:
        return ctx

    # 增量缓存：若上下文内容未变，直接返回上次预算结果，避免重复估算与截断
    ctx_hash = hashlib.md5("".join(sorted(ctx.values())).encode()).hexdigest()
    cache_key = f"budget:{ctx_hash}:{token_budget}:{skill_min_tokens}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]

    budgeted = dict(ctx)
    priority = [
        "skills_catalog_section",  # 工具目录（字典性信息）最先耸
        "memories_section",
        "episodic_section",
        "skills_section",          # 技能正文次优先
        "wm_section",              # 实时感知保留到倒数第二
        "tools_section",           # 工具定义最后耸（agent 必须知道能用什么）
    ]
    minimum_keep = {
        "skills_section": skill_min_tokens,
        "skills_catalog_section": max(40, skill_min_tokens // 2),
        "memories_section": 1,
        "episodic_section": 2,
        "wm_section": 1,
        "tools_section": 2,
    }

    def total_tokens(items: dict[str, str]) -> int:
        return sum(_estimate_tokens(value) for value in items.values())

    current_total = total_tokens(budgeted)
    if current_total <= token_budget:
        return budgeted

    for key in priority:
        if current_total <= token_budget:
            break
        original = budgeted.get(key, "")
        if not original:
            continue

        keep_floor = minimum_keep.get(key, 0)
        original_tokens = _estimate_tokens(original)
        if original_tokens <= keep_floor:
            continue

        reduction = min(original_tokens - keep_floor, current_total - token_budget)
        keep_tokens = max(keep_floor, original_tokens - reduction)
        trimmed = _compress_text_segments(original, keep_tokens)
        if trimmed != original:
            trimmed = f"（上下文已智能压缩：原 {original_tokens} tokens → {keep_tokens} tokens）\n{trimmed}"
        budgeted[key] = trimmed
        current_total -= _estimate_tokens(original) - _estimate_tokens(trimmed)

    _context_fmt_cache[cache_key] = budgeted
    return budgeted


# 简单缓存，避免重复计算高频短文本
@functools.lru_cache(maxsize=8192)
def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    other = len(text) - cjk - ascii_chars
    
    return max(1, int(cjk * 1.8 + ascii_chars * 0.3 + other * 1.0))


def _compress_text_segments(text: str, keep_tokens: int) -> str:
    if keep_tokens <= 0:
        return ""
    if _estimate_tokens(text) <= keep_tokens:
        return text

    segments = _split_segments(text)
    if not segments:
        return ""

    keep_head: list[str] = []
    keep_tail: list[str] = []
    head_tokens = 0
    tail_tokens = 0
    head_idx = 0
    tail_idx = len(segments) - 1
    turn = 0

    while head_idx <= tail_idx:
        if turn % 2 == 0:
            candidate = segments[head_idx]
            candidate_tokens = _estimate_tokens(candidate)
            if head_tokens + tail_tokens + candidate_tokens <= keep_tokens:
                keep_head.append(candidate)
                head_tokens += candidate_tokens
                head_idx += 1
            elif tail_idx == head_idx and not keep_head and not keep_tail:
                keep_head.append(_compress_single_segment(candidate, keep_tokens))
                break
            else:
                break
        else:
            candidate = segments[tail_idx]
            candidate_tokens = _estimate_tokens(candidate)
            if head_tokens + tail_tokens + candidate_tokens <= keep_tokens:
                keep_tail.append(candidate)
                tail_tokens += candidate_tokens
                tail_idx -= 1
            elif tail_idx == head_idx and not keep_head and not keep_tail:
                keep_tail.append(_compress_single_segment(candidate, keep_tokens))
                break
            else:
                break
        turn += 1

    if not keep_head and not keep_tail:
        return _compress_single_segment(text, keep_tokens)

    body = keep_head + (["\n[...省略...]\n"] if head_idx <= tail_idx else []) + keep_tail[::-1]
    result = "".join(body)
    # 结构安全保护：自动补全未闭合的括号/引号，防止压缩破坏代码/JSON结构
    open_chars = "([{"
    close_chars = ")]}"
    stack = []
    for ch in result:
        if ch in open_chars:
            stack.append(close_chars[open_chars.index(ch)])
        elif ch in close_chars and stack and stack[-1] == ch:
            stack.pop()
    return result + "".join(reversed(stack))


def _split_segments(text: str) -> list[str]:
    parts = re.split(r"(\n\s*\n)", text)
    segments: list[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"\n\s*\n", part):
            if buffer:
                segments.append(buffer)
                buffer = ""
            segments.append(part)
        else:
            buffer += part
    if buffer:
        segments.append(buffer)
    return segments


def _compress_single_segment(text: str, keep_tokens: int) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) <= 1:
        return text[: max(1, min(len(text), keep_tokens * 4))]

    kept: list[str] = []
    token_count = 0
    for line in lines:
        line_tokens = _estimate_tokens(line)
        if token_count + line_tokens > keep_tokens:
            break
        kept.append(line)
        token_count += line_tokens

    if kept:
        return "".join(kept) + ("\n[...省略...]" if len(kept) < len(lines) else "")
    return text[: max(1, min(len(text), keep_tokens * 4))]


def _fmt_judgment_signals(signals: "JudgmentSignals | None") -> str:
    if not signals:
        return "（JudgmentSignals 未计算）"
    return (
        f"posture={signals.posture}  "
        f"require_more_evidence={signals.require_more_evidence}  "
        f"prefer_narrow_scope={signals.prefer_narrow_scope}"
    )


def _fmt_hard_boundaries(hard_boundaries: "list[str] | None") -> str:
    if not hard_boundaries:
        return "（无 hard_boundary 限制）"
    return "\n".join(f"- {boundary}" for boundary in hard_boundaries)


def _fmt_perception_replay(replay: "PerceptionReplaySummary | None") -> str:
    if not replay:
        return "（感知重放不可用）"
    lines = [
        f"样本数={replay.samples}  平均预测误差={replay.avg_prediction_error:.2f}  连续高误差={replay.high_error_streak}  趋势={replay.trend}",
    ]
    if replay.hints:
        for hint in replay.hints:
            lines.append(f"提示: {hint}")
    return "\n".join(lines)


def _short_skill_desc(desc: str, limit: int = 90) -> str:
    text = desc.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _fmt_skill_catalog(skills: "list[Skill]") -> str:
    if not skills:
        return "（暂无 skills）"
    lines = [
        "**AGENT SKILLS CATALOG** — 这里只预加载 metadata，不直接注入完整 instructions。",
        "当某个 skill 的 description 明显匹配当前任务时，用 skill.activate 加载对应 SKILL.md；不要仅凭 skill 名称猜测。",
        "",
        "| 技能 | 触发信号 | 何时使用 |",
        "|------|---------|---------|",
    ]
    for skill in skills:
        triggers_list = getattr(skill, "triggers", None) or []
        trigger_str = "、".join(triggers_list[:3]) if triggers_list else "—"
        desc = _short_skill_desc(skill.description)
        lines.append(f"| `{skill.name}` | {trigger_str} | {desc} |")
    lines.append("")
    lines.append("可调用 skill.list / skill.search 查 catalog；真正使用前优先 skill.activate，而不是把目录摘要当完整规则。")
    return "\n".join(lines)


def _fmt_primary_skill(skill: "Skill | None") -> str:
    if skill is None:
        return "（本轮无明显 skill 候选；按一般 judgment 规则执行。若遇到专业流程或项目特有规则，再查 catalog 并按需 skill.activate。）"
    origin = str(getattr(skill, "origin", "dynamic") or "dynamic")
    if origin == "workspace" and getattr(skill, "source_path", ""):
        origin = skill.source_path
    return (
        f"**{skill.name}** — {skill.description}\n"
        f"> 候选 skill，不代表已激活。source: {origin}\n"
        f"> 若你判断它与当前任务相关，先调用 skill.activate(name=\"{skill.name}\") 读取完整 SKILL.md，再决定是否遵循。"
    )


def _fmt_skills(skills: "list[Skill]") -> str:
    if not skills:
        return "（当前没有候选 skill 被高亮；可按需查阅 catalog）"
    parts: list[str] = [
        "以下是当前上下文下较相关的候选 skills。它们目前仍只是 metadata 线索，不是已注入的完整 instructions。",
    ]
    for skill in skills:
        origin = str(getattr(skill, "origin", "dynamic") or "dynamic")
        if origin == "workspace" and getattr(skill, "source_path", ""):
            origin = skill.source_path
        parts.append(f"**{skill.name}** [{origin}] — {skill.description}")
        parts.append(f"> activation: skill.activate(name=\"{skill.name}\")")
    return "\n".join(parts)


def _fmt_cognitive_signals(signals: "CognitiveSignals | None") -> str:
    if signals is None:
        return "（认知信号暂不可用）"
    return signals.to_text()



# --- 严格 Schema 校验 ---
_CONTEXT_SCHEMA_KEYS = ["identity", "tasks", "memory", "perception"]

def _validate_context_schema(ctx: dict) -> tuple[bool, str]:
    """严格校验上下文结构，防止畸形数据注入。"""
    missing = [k for k in _CONTEXT_SCHEMA_KEYS if k not in ctx]
    if missing:
        return False, f"缺少必需字段: {', '.join(missing)}"
    return True, "ok"

def _fmt_blind_spots(probes: list[Any], self_model_tokens: int = 0) -> str:
    """计算当前可能存在的感知盲点——LLM 意识不到的缺失。

    不是命令，是让 LLM 自己决定是否需要关注这些潜在盲区。
    """
    coverage_tags = {
        tag
        for p in probes
        for tag in normalize_probe_coverage_tags(getattr(p, "coverage_tags", []))
    }
    has_channel_health = "ops:channel_health" in coverage_tags
    has_api_quota = "ops:api_quota" in coverage_tags
    has_git = "workspace:git_state" in coverage_tags

    gaps = []
    if not has_channel_health:
        gaps.append("- 关键外部通道健康未监控 → 依赖链路中断时你可能无法及时感知（例如消息网关/API 代理不可用）")
    if not has_api_quota:
        gaps.append("- API 调用量/额度未追踪 → 你可能在悄悄耗尽配额而不自知")
    if not has_git:
        gaps.append("- git 变更未追踪 → evolution 改了代码你不知道改了什么")

    if not gaps:
        return "当前感知覆盖良好，暂无明显盲点。"

    coverage_legend = "；".join(f"{tag}={desc}" for tag, desc in PROBE_COVERAGE_HINTS.items())
    return (
        "以下是你当前**没有在监控**的东西——不是要求你立即行动，只是提醒你可能忽略了：\n"
        + "\n".join(gaps)
        + f"\n\n可用 coverage_tags: {coverage_legend}"
    )

def _fmt_probe_sensors(probes: list[Any]) -> str:
    """将当前已部署的探针传感器网络格式化为 LLM 可读的感知面板。

    每个探针显示：状态 / 名称 / 部署目的 / 执行规格 / 最近读数。
    让 LLM 随时知道自己的感知网络状态及每个探针的意义。
    """
    if not probes:
        return (
            "⚠️ 你目前没有部署任何探针。探针是你的『感知触手』——采集外部信息，结果自动注入工作记忆。\n"
            "建议安装以下自我监控探针（用 probe.install）：\n"
            "  1. 磁盘使用率 → kind=shell spec='df -h / | tail -1' trigger=interval:600 purpose='磁盘超85%需清理' coverage_tags=[]\n"
            "  2. 内存 → kind=shell spec='free -m | grep Mem' trigger=interval:300 purpose='内存压力预警' coverage_tags=[]\n"
            "  3. 自身进程 → kind=shell spec='ps aux | grep lingzhou | grep -v grep | wc -l' trigger=interval:120 purpose='确认自身存活' coverage_tags=[]\n"
            "  4. 外部通道健康 → kind=shell spec='curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8080/health' trigger=interval:300 purpose='关键通道健康，非200说明链路异常' coverage_tags=['ops:channel_health']\n"
        )
    lines: list[str] = [
        "探针结果不是绝对真相：confidence<0.60 或标记为布放可疑时，先校验探针布放（spec/target/trigger），再据此决策。",
        "盲点推断只读取显式 coverage_tags，不再从 purpose/spec 猜测；未声明 coverage_tags 的探针不会计入覆盖。",
    ]
    for p in probes:
        mark = "✓" if p.enabled else "⊘"
        trigger_desc = p.trigger or "manual"
        alert_mark = " 🔔" if p.alert_expr else ""
        confidence = getattr(p, "last_confidence", None)
        confidence_mark = ""
        if isinstance(confidence, (int, float)):
            confidence_mark = f" confidence={float(confidence):.2f}"
        suspect_mark = " ⚠️布放可疑" if getattr(p, "last_suspect", False) else ""
        # 目的说明
        purpose_line = f"  └ 目的: {p.purpose}" if getattr(p, "purpose", "") else ""
        # 最近读数
        reading_line = ""
        if p.last_run_at:
            t = p.last_run_at.split("T")[-1][:5] if "T" in p.last_run_at else p.last_run_at[:16]
            if p.last_error:
                reading_line = f"  └ @{t} ❌ {p.last_error[:80]}"
            elif p.last_result:
                result_text = p.last_result.strip().replace("\n", " ")[:120]
                reading_line = f"  └ @{t} → {result_text}"
            else:
                reading_line = f"  └ @{t} (无输出)"
        else:
            reading_line = "  └ 尚未执行"
        conf_reason = str(getattr(p, "last_confidence_reason", "") or "").strip()
        conf_line = ""
        if conf_reason:
            conf_line = f"  └ 可信度依据: {conf_reason[:120]}"
        coverage_tags = normalize_probe_coverage_tags(getattr(p, "coverage_tags", []))
        coverage_line = (
            f"  └ coverage: {', '.join(coverage_tags)}"
            if coverage_tags else
            "  └ coverage: （未声明，不计入盲点覆盖）"
        )
        header = (
            f"  {mark} [{p.name}] {p.kind}/{trigger_desc} →{p.data_back}{alert_mark}"
            f"{confidence_mark}{suspect_mark}"
        )
        entry = header
        if purpose_line:
            entry += "\n" + purpose_line
        entry += "\n" + coverage_line
        entry += "\n" + reading_line
        if conf_line:
            entry += "\n" + conf_line
        lines.append(entry)
    return "\n".join(lines)
