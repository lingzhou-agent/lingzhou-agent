"""core/loop/progress.py - loop 内的任务推进判定 helper。"""

from __future__ import annotations

import hashlib

from core.execution import action_key_param
from core.judgment import JudgmentOutput
from tools.registry import ToolResult

_PROGRESS_MUTATION_TOOLS = frozenset({
    "file.write", "file.edit",
    "exec", "process.write", "process.kill",
    "task.add", "task.update", "task.advance", "task.complete", "task.fail",
    "memory.add_wm", "memory.add_semantic", "memory.set_fact",
    "schedule.add", "schedule.ack", "schedule.cancel",
    "failure.dismiss",
})

_PROGRESS_INFO_TOOLS = frozenset({
    "file.read", "file.list",
    "memory.search", "memory.get_fact",
    "task.list", "schedule.list",
    "skill.list", "skill.search",
    "process.poll", "process.log",
    "shell.capabilities",
})


def _result_fingerprint(summary: str) -> str:
    text = (summary or "").strip()
    if not text:
        return ""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _looks_like_path_probe_output(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or "\n" in stripped:
        return False
    if " " in stripped or "\t" in stripped:
        return False
    return stripped.startswith(("/", "./", "../", "~"))


def _has_failure_markers(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in (
        "traceback",
        "runtimewarning",
        "exception",
        "syntaxerror",
        "attributeerror",
        "typeerror",
        "filenotfound",
        "error:",
        "warning:",
    ))


def _shell_run_made_progress(
    action: JudgmentOutput,
    result: ToolResult,
    *,
    prev_sig: str = "",
    prev_fp: str = "",
) -> tuple[bool, str]:
    metadata = result.metadata or {}
    stdout_preview = str(metadata.get("stdout_preview") or "")
    stderr_preview = str(metadata.get("stderr_preview") or "")
    output_preview = str(metadata.get("output_preview") or result.summary or "")

    if _has_failure_markers(stderr_preview) or _has_failure_markers(output_preview):
        return False, "shell.run 输出包含错误标记(Traceback/RuntimeWarning)"
    probe_text = stdout_preview.strip() or output_preview.strip()
    if _looks_like_path_probe_output(probe_text):
        return False, "shell.run 仅探测路径存在,非实质推进"

    if result.state_delta or result.artifact_paths:
        return True, "shell.run 产生副作用或产出文件"

    fp = _result_fingerprint(probe_text)
    if not fp:
        return False, "shell.run 无有效输出"
    cur_sig = f"{action.chosen_action_id or ''}|{action_key_param(action.params)}"
    if cur_sig == prev_sig and fp == prev_fp:
        return False, "shell.run 结果与上轮相同"
    return True, "shell.run 获得新输出"


def _action_made_progress(
    action: JudgmentOutput,
    result: ToolResult,
    *,
    prev_sig: str = "",
    prev_fp: str = "",
) -> tuple[bool, str]:
    """判断动作是否真实推进了任务。"""
    if action.decision != "act" or result.error or result.skipped:
        return False, f"decision={action.decision} error={bool(result.error)} skipped={result.skipped}"

    tool = action.chosen_action_id or ""
    if tool == "shell.run":
        return _shell_run_made_progress(action, result, prev_sig=prev_sig, prev_fp=prev_fp)

    if tool in _PROGRESS_MUTATION_TOOLS:
        return True, f"{tool} 是变更类工具,成功执行即视为推进"

    if tool in _PROGRESS_INFO_TOOLS:
        fp = _result_fingerprint(result.summary)
        if not fp:
            return False, f"{tool} 返回空结果"
        cur_sig = f"{tool}|{action_key_param(action.params)}"
        if cur_sig == prev_sig and fp == prev_fp:
            return False, f"{tool} 结果与上轮相同(重复操作)"
        return True, f"{tool} 获得新信息(结果指纹变化)"

    if result.state_delta or result.artifact_paths or result.resource_key:
        return True, f"{tool} 产生副作用(state_delta/artifact)"
    fp = _result_fingerprint(result.summary)
    if not fp:
        return False, f"{tool} 无有效输出"
    cur_sig = f"{tool}|{action_key_param(action.params)}"
    if cur_sig == prev_sig and fp == prev_fp:
        return False, f"{tool} 结果与上轮相同"
    return True, f"{tool} 输出与上轮不同"