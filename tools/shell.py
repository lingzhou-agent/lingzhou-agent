"""tools/shell.py — shell.run 工具。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_PREVIEW_CHARS = 0  # 0 = 不限制输出长度

_MANIFEST = ToolManifest(
    name="shell.run",
    description=(
        "在当前宿主环境中执行一次性 shell 命令（非持久会话）。"
        "返回 stdout+stderr 合并输出摘要，并受 timeout 与输出截断限制。"
    ),
    progress_category="mutation",
    capabilities=("completion_verify",),
        params=[
        ToolParam("command", "string", "要执行的 bash 命令", required=True),
        ToolParam("timeout", "number", "超时秒数，默认 30", required=False),
        ToolParam("workdir", "string", "工作目录，默认项目根目录", required=False),
        ToolParam("max_output_chars", "number", "返回摘要最大字符数，默认不限制（0=不限制，传正整数截断）", required=False),
    ],
)

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_workdir(raw: Any, ctx: ToolContext | None = None) -> Path:
    if raw is None or raw == "":
        repo_root = _repo_root()
        if repo_root.exists():
            return repo_root
        return Path.cwd()
    return Path(str(raw)).expanduser()


def _threshold_value(ctx: ToolContext, attr: str, default: Any) -> Any:
    config = getattr(ctx, "config", None)
    thresholds = getattr(config, "thresholds", None)
    return getattr(thresholds, attr, default)


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0:  # 0 或负数 = 不限制
        return text
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _compact_summary_text(text: str, limit: int) -> str:
    compact = " ".join(text.replace("\r", "\n").splitlines()).strip()
    return _truncate_text(compact, limit)


def _decode_output(data: bytes | None) -> str:
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")


def _fingerprint(command: str, workdir: Path, returncode: int, output: str) -> str:
    digest = hashlib.sha256()
    digest.update(command.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(str(workdir).encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(str(returncode).encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(output.encode("utf-8", errors="replace"))
    return f"shell:{digest.hexdigest()[:16]}"


@tool(_MANIFEST)
async def shell_run(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = (params.get("command") or "").strip()
    if not command:
        return ToolResult(summary="命令为空", skipped=True)

    timeout_raw = params.get("timeout")
    timeout = float(
        _threshold_value(ctx, "shell_timeout", _DEFAULT_TIMEOUT)
        if timeout_raw is None
        else timeout_raw
    )

    preview_raw = params.get("max_output_chars")
    preview_limit = int(
        _threshold_value(ctx, "shell_max_output_chars", _DEFAULT_PREVIEW_CHARS)
        if preview_raw is None
        else preview_raw
    )

    workdir = _resolve_workdir(params.get("workdir"), ctx)

    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(workdir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )

    timed_out = False
    stdout_b = b""
    stderr_b = b""

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        stdout_b, stderr_b = await proc.communicate()

    returncode = proc.returncode if proc.returncode is not None else -1
    stdout = _decode_output(stdout_b)
    stderr = _decode_output(stderr_b)
    combined = stdout
    if stdout and stderr:
        combined += "\n"
    combined += stderr

    preview = _truncate_text(combined, max(preview_limit, 0))
    preview_text = preview or "(无输出)"
    summary_body = _compact_summary_text(preview_text, 120)
    status = "timeout" if timed_out else f"exit={returncode}"
    summary = f"{status} cwd={workdir}"
    if summary_body:
        summary += f" | {summary_body}"

    payload = {
        "command": command,
        "workdir": str(workdir),
        "timeout_sec": timeout,
        "timed_out": timed_out,
        "returncode": returncode,
        "stdout_chars": len(stdout),
        "stderr_chars": len(stderr),
        "output_chars": len(combined),
        "output_preview": preview,
        "stdout_preview": _truncate_text(stdout, preview_limit),
        "stderr_preview": _truncate_text(stderr, preview_limit),
        "log_summary": f"shell.run {'timeout' if timed_out else f'exit={returncode}'} chars={len(combined)}",
    }

    return ToolResult(
        summary=summary,
        evidence=json.dumps(payload, ensure_ascii=False),
        resource_key=str(workdir),
        fingerprint=_fingerprint(command, workdir, returncode, combined),
        metadata=payload,
        state_delta={
            "process": "finished",
            "exit_code": returncode,
            "timed_out": timed_out,
        },
    )
