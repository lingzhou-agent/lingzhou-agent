"""tools/shell.py — shell.run 工具。"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_PREVIEW_CHARS = 500
_COMMON_COMMANDS = (
    "python3", "python", "bash", "sh", "grep", "find", "ls", "cat",
    "sqlite3", "git", "sed", "awk", "jq", "rg",
)

_MANIFEST = ToolManifest(
    name="shell.run",
    description=(
        "在当前宿主环境中执行一次性 shell 命令（非持久会话）。"
        "返回 stdout+stderr 合并输出摘要，并受 timeout 与输出截断限制。"
    ),
    params=[
        ToolParam("command", "string", "要执行的 bash 命令", required=True),
        ToolParam("timeout", "number", "超时秒数，默认 30", required=False),
        ToolParam("workdir", "string", "工作目录，默认当前目录", required=False),
        ToolParam("max_output_chars", "number", "返回摘要最大字符数，默认 500", required=False),
    ],
)

_CAP_MANIFEST = ToolManifest(
    name="shell.capabilities",
    description="返回 shell 执行能力画像（可用命令、默认限制、环境语义）",
    params=[],
)


def _resolve_workdir(raw: Any) -> str:
    if raw:
        return str(Path(str(raw)).expanduser())
    return str(Path.cwd())


def _build_capabilities(workdir: str, timeout: float = _DEFAULT_TIMEOUT, preview: int = _DEFAULT_PREVIEW_CHARS) -> dict[str, Any]:
    available = [cmd for cmd in _COMMON_COMMANDS if shutil.which(cmd)]
    return {
        "engine": "asyncio.create_subprocess_shell",
        "execution_model": "one-shot-non-persistent",
        "sandbox": False,
        "network_policy": "inherits-host-environment",
        "default_timeout_sec": timeout,
        "default_output_preview_chars": preview,
        "workdir": workdir,
        "shell": os.environ.get("SHELL") or "/bin/sh",
        "available_commands": available,
        "missing_commands": [cmd for cmd in _COMMON_COMMANDS if cmd not in available],
    }


@tool(_CAP_MANIFEST)
async def shell_capabilities(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    workdir = _resolve_workdir(params.get("workdir"))
    caps = _build_capabilities(workdir, ctx.config.thresholds.shell_timeout, ctx.config.thresholds.shell_max_output_chars)
    summary = (
        "shell.capabilities: "
        f"sandbox={caps['sandbox']} mode={caps['execution_model']} "
        f"timeout={caps['default_timeout_sec']}s cmds={len(caps['available_commands'])}"
    )
    return ToolResult(summary=summary, evidence=json.dumps(caps, ensure_ascii=False))


@tool(_MANIFEST)
async def shell_run(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = (params.get("command") or "").strip()
    if not command:
        return ToolResult(summary="命令为空", skipped=True)

    timeout = float(params.get("timeout") or ctx.config.thresholds.shell_timeout)
    preview_limit = int(params.get("max_output_chars") or ctx.config.thresholds.shell_max_output_chars)
    workdir = _resolve_workdir(params.get("workdir"))

    if ctx.dry_run:
        caps = _build_capabilities(workdir, timeout, preview_limit)
        return ToolResult(
            summary=f"[dry-run] shell.run: {command[:200]}",
            evidence=json.dumps({
                "dry_run": True,
                "command": command[:120],
                "timeout": timeout,
                "workdir": workdir,
                "capabilities": caps,
            }, ensure_ascii=False),
            skipped=True,
        )

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=workdir,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ToolResult(
                summary=f"执行超时（{timeout}s）: {command[:100]}",
                evidence=json.dumps({
                    "timeout": timeout,
                    "command": command[:120],
                    "workdir": workdir,
                    "timed_out": True,
                }, ensure_ascii=False),
                error="TimeoutError",
            )

        output = stdout.decode(errors="replace").strip()
        truncated = output[:preview_limit] + ("..." if len(output) > preview_limit else "")
        evidence = json.dumps({
            "command": command[:120],
            "exit_code": proc.returncode,
            "timeout": timeout,
            "workdir": workdir,
            "output_chars": len(output),
            "preview_chars": min(len(output), preview_limit),
        }, ensure_ascii=False)
        if proc.returncode == 0:
            return ToolResult(
                summary=f"执行成功:\n{truncated}",
                evidence=evidence,
            )
        else:
            return ToolResult(
                summary=f"执行出错 (exit={proc.returncode}):\n{truncated}",
                evidence=evidence,
                error=output[:300],
            )
    except Exception as exc:
        return ToolResult(
            summary=f"执行异常: {exc}",
            evidence=str(exc),
            error=str(exc),
        )
