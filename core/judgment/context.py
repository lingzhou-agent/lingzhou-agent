"""core/judgment/context.py - judgment 上下文组装相关格式化与预算 helper。"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

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
    if last_run_status:
        lines.append(f"最近运行状态: {last_run_status}")
    return "\n".join(lines)


def _fmt_recent_runs(runs: list["Run"]) -> str:
    if not runs:
        return "（暂无近期运行记录）"
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
    return "\n".join(lines)


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
    if not facts:
        return "（暂无近期关键事实）"
    return "\n".join(
        f"- {key} = {_format_fact_value(value)}"
        for key, value in facts
    )


def _fmt_waiting_tasks(tasks: list["Task"]) -> str:
    if not tasks:
        return "（无 waiting 任务）"
    lines: list[str] = []
    for task in tasks[:5]:
        wait_desc = task.wait_kind or "unknown"
        if task.wait_key:
            wait_desc += f"/{task.wait_key}"
        line = f"- task#{task.id} [{task.status}] {task.title} wait={wait_desc}"
        if task.next_step:
            line += f" next={_clip_text(task.next_step, 80)}"
        lines.append(line)
    return "\n".join(lines)


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
    now = datetime.now(timezone.utc)
    local_iso = now.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    utc_str = now.strftime("%Y-%m-%d %H:%M UTC")
    return f"当前时间: {local_iso}\n参考 UTC: {utc_str}"


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
    lines = [header] + [f"- [{item['kind']}] {item['content']}" for item in ordered]
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
    lines = [f"- [{memory['kind']}] {memory['title']}: {memory['body']}" for memory in memories]
    return "\n".join(lines)


def _fmt_tools(manifests: "list[ToolManifest]") -> str:
    if not manifests:
        return "（无可用工具）"
    lines: list[str] = []
    for manifest in manifests:
        params_str = ", ".join(
            f"{param.name}({'*' if param.required else '?'})" for param in manifest.params
        )
        lines.append(f"- `{manifest.name}`: {manifest.description}  参数: [{params_str}]")
    return "\n".join(lines)


def _fmt_shell_capabilities() -> str:
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
        "default_output_preview_chars": 500,
        "shell": os.environ.get("SHELL") or "/bin/sh",
        "cwd": os.getcwd(),
        "available_commands": available,
        "missing_commands": [cmd for cmd in cmds if cmd not in available],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _fmt_percept(percept: "Percept") -> str:
    return (
        f"预测误差: {percept.prediction_error:.2f}  "
        f"工作区变更: {'是' if percept.workspace_dirty else '否'}"
    )


def _fmt_soul(axioms_val: str, ethos_val: str) -> str:
    parts: list[str] = []
    if axioms_val:
        parts.append(f"绝对禁忌（hard_axioms）: {axioms_val}")
    if ethos_val:
        parts.append(f"价值基线（ethos_baseline）: {ethos_val}")
    return "\n".join(parts) if parts else "（Soul 未初始化，运行 `init` 命令生成）"


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
    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return str(ctx.get(key, f"[未知字段: {key}]"))

    return re.sub(r"\{\{([^}]+)\}\}", replace, template)


def _fmt_ethos(ethos_state: "EthosState | None") -> str:
    if not ethos_state:
        return "（Ethos 未计算）"
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

    budgeted = dict(ctx)
    priority = [
        "skills_section",
        "skills_catalog_section",
        "memories_section",
        "episodic_section",
        "wm_section",
        "tools_section",
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

    return budgeted


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_chars = sum(1 for char in text if ord(char) < 128 and not char.isspace())
    other = sum(1 for char in text if ord(char) >= 128 and not ("\u4e00" <= char <= "\u9fff"))
    return cjk + max(1, ascii_chars // 4) + max(1, other // 2)


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

    body = keep_head + (["\n[...省略...]\n"] if head_idx <= tail_idx else []) + list(reversed(keep_tail))
    return "".join(body)


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
    lines = ["你当前可用的 active skills 摘要如下；当当前命中的 skills 不够时，可以据此调整判断或调用 skill.search/skill.list："]
    for skill in skills:
        origin = "builtin" if not getattr(skill, "source_path", "") else "workspace"
        triggers = f" | triggers: {', '.join(skill.triggers[:4])}" if getattr(skill, "triggers", None) else ""
        lines.append(f"- {skill.name} [{origin}] — {_short_skill_desc(skill.description)}{triggers}")
    return "\n".join(lines)


def _fmt_primary_skill(skill: "Skill | None") -> str:
    if skill is None:
        return "（本轮未命中主技能；按一般 judgment 规则执行。如判断受阻，再参考下方 activated skills 或调用 skill.search）"
    origin = "builtin" if not getattr(skill, "source_path", "") else skill.source_path
    return (
        f"**{skill.name}** — {skill.description}\n"
        f"> 本轮主技能，优先遵守。source: {origin}\n"
        f"> {skill.guidance}"
    )


def _fmt_skills(skills: "list[Skill]") -> str:
    if not skills:
        return "（除主技能外，本轮无其他补充技能；如判断受阻，可参考上方 skill catalog 或调用 skill.search/skill.list）"
    parts: list[str] = []
    for skill in skills:
        origin = "builtin" if not getattr(skill, "source_path", "") else skill.source_path
        triggers = f" | triggers: {', '.join(skill.triggers[:4])}" if getattr(skill, "triggers", None) else ""
        parts.append(f"- {skill.name} [{origin}] — {_short_skill_desc(skill.description)}{triggers}")
    return "（以下为本轮的补充技能摘要；主技能优先，补充技能仅在主技能不足时参考）\n" + "\n".join(parts)


def _fmt_cognitive_signals(signals: "CognitiveSignals | None") -> str:
    if signals is None:
        return "（认知信号暂不可用）"
    return signals.to_text()


def _fmt_probe_sensors(probes: list[Any]) -> str:
    """将当前已部署的探针传感器网络格式化为 LLM 可读的感知面板。

    每行一个探针：状态标记 / 名称 / 执行规格 / 回传策略 / 最近读数摘要。
    让 LLM 随时知道自己的感知网络状态，无需主动查询。
    """
    if not probes:
        return "（无部署的探针——可用 probe.install 工具布控传感器）"
    lines: list[str] = []
    for p in probes:
        mark = "✓" if p.enabled else "⊘"
        # 触发规格
        trigger_desc = p.trigger or "manual"
        # 数据回传策略 + 告警标记
        back_desc = f"→{p.data_back}"
        alert_mark = " 🔔" if p.alert_expr else ""
        # 最近读数
        reading = ""
        if p.last_run_at:
            t = p.last_run_at.split("T")[-1][:5] if "T" in p.last_run_at else p.last_run_at[:16]
            result_text = (p.last_result or "").strip().replace("\n", " ")[:80]
            reading = f" | @{t} {result_text!r}"
            if p.last_error:
                reading += f" [err:{p.last_error[:40]}]"
        lines.append(
            f"  {mark} [{p.name}] {p.kind}/{trigger_desc} {back_desc}{alert_mark}{reading}"
        )
    return "\n".join(lines)
