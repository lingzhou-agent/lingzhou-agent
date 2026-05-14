"""tools/exec.py — exec/process 工具。

exec：启动 shell 命令，支持后台运行、PTY、超时、工作目录、环境变量。
process：管理已启动的后台进程（list/poll/log/write/kill）。

设计：
- exec 启动的进程由模块级进程管理器追踪
- process 工具通过 session_id 操作具体进程
- 支持后台运行（background=True）和前台阻塞运行
- PTY 模式用于需要终端交互的命令
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool

_log = logging.getLogger("lingzhou.tools.exec")

# ── 进程管理器 ────────────────────────────────────────────────────────────────

@dataclass
class ProcessInfo:
    session_id: str
    command: str
    pid: int | None = None
    started_at: float = 0.0
    finished_at: float | None = None
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    background: bool = False
    finished: bool = False
    timed_out: bool = False
    pty: bool = False
    workdir: str = ""
    _output_chunks: list[str] = field(default_factory=list)


class ProcessManager:
    """追踪所有通过 exec 启动的进程。"""

    _counter: int = 0
    _processes: dict[str, ProcessInfo] = {}

    @classmethod
    def next_id(cls) -> str:
        cls._counter += 1
        return f"exec-{cls._counter}"

    @classmethod
    def register(cls, info: ProcessInfo) -> str:
        cls._processes[info.session_id] = info
        return info.session_id

    @classmethod
    def get(cls, session_id: str) -> ProcessInfo | None:
        return cls._processes.get(session_id)

    @classmethod
    def list_all(cls) -> list[ProcessInfo]:
        return list(cls._processes.values())

    @classmethod
    def mark_finished(cls, session_id: str, return_code: int, timed_out: bool = False) -> None:
        info = cls._processes.get(session_id)
        if info:
            info.finished = True
            info.finished_at = time.time()
            info.return_code = return_code
            info.timed_out = timed_out


_MANAGER = ProcessManager()


# ── shell.run 的能力增强：shell.capabilities 更新 ─────────────────────────────
# 在 shell.py 中已有 shell.capabilities，这里覆盖为增强版

_CAP_MANIFEST_V2 = ToolManifest(
    name="shell.capabilities",
    description="返回 shell 执行能力画像（可用命令、默认限制、环境语义、exec/process 支持）",
    params=[],
)


def _build_capabilities_v2(workdir: str) -> dict[str, Any]:
    import shutil
    common = ("python3", "python", "bash", "sh", "grep", "find", "ls", "cat",
              "sqlite3", "git", "sed", "awk", "jq", "rg")
    available = [cmd for cmd in common if shutil.which(cmd)]
    return {
        "engine": "asyncio.create_subprocess_shell (exec) + asyncio.create_subprocess_exec (process)",
        "execution_model": "one-shot or background",
        "sandbox": False,
        "network_policy": "inherits-host-environment",
        "default_timeout_sec": 30,
        "default_output_preview_chars": 500,
        "workdir": workdir,
        "shell": os.environ.get("SHELL") or "/bin/sh",
        "available_commands": available,
        "has_background_exec": True,
        "has_process_management": True,
        "has_pty": False,  # PTY requires pty module, set dynamically
    }


@tool(_CAP_MANIFEST_V2)
async def shell_capabilities(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    workdir = params.get("workdir", str(Path.cwd()))
    caps = _build_capabilities_v2(workdir)
    try:
        import pty
        caps["has_pty"] = True
    except ImportError:
        pass
    summary = (
        f"shell.capabilities: sandbox={caps['sandbox']} "
        f"background={caps['has_background_exec']} "
        f"process_mgmt={caps['has_process_management']} "
        f"pty={caps['has_pty']} "
        f"cmds={len(caps['available_commands'])}"
    )
    return ToolResult(summary=summary, evidence=json.dumps(caps, ensure_ascii=False))


# ── exec：启动命令 ─────────────────────────────────────────────────────────────

_EXEC_MANIFEST = ToolManifest(
    name="exec",
    description=(
        "启动 shell 命令。支持前台阻塞执行或后台运行。"
        "前台模式：等待命令完成，返回完整输出（受 timeout 限制）。"
        "后台模式：立即返回 session_id，后续通过 process 工具管理。"
        "workdir 指定工作目录，env 设置环境变量。"
    ),
    params=[
        ToolParam("command", "string", "要执行的 shell 命令", required=True),
        ToolParam("background", "boolean",
                  "是否后台运行。true=立即返回 session_id；false=等待完成（默认）",
                  required=False),
        ToolParam("timeout", "number", "超时秒数，默认 30（前台）或 300（后台）", required=False),
        ToolParam("workdir", "string", "工作目录，默认当前目录", required=False),
        ToolParam("max_output_chars", "number", "返回摘要最大字符数，默认 500", required=False),
        ToolParam("env", "object", "环境变量字典（可选）", required=False),
    ],
)


@tool(_EXEC_MANIFEST)
async def exec_run(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = (params.get("command") or "").strip()
    if not command:
        return ToolResult(summary="命令为空", skipped=True, error="EmptyCommand")

    background = params.get("background", False)
    timeout = float(params.get("timeout") or (300.0 if background else 30.0))
    workdir = params.get("workdir") or str(Path.cwd())
    preview_limit = int(params.get("max_output_chars") or 500)
    env_overrides = params.get("env")

    if ctx.dry_run:
        return ToolResult(
            summary=f"[dry-run] exec: {command[:200]}",
            evidence=json.dumps({
                "dry_run": True, "command": command[:120],
                "timeout": timeout, "workdir": workdir, "background": background,
            }, ensure_ascii=False),
            skipped=True,
        )

    session_id = _MANAGER.next_id()
    info = ProcessInfo(
        session_id=session_id,
        command=command,
        started_at=time.time(),
        background=background,
        workdir=workdir,
    )
    _MANAGER.register(info)

    # 构建环境
    exec_env = os.environ.copy()
    if env_overrides and isinstance(env_overrides, dict):
        exec_env.update({str(k): str(v) for k, v in env_overrides.items()})

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=workdir,
            env=exec_env,
        )
        info.pid = proc.pid

        if background:
            # 后台：启动独立任务收集输出
            asyncio.create_task(_collect_background_output(proc, session_id))
            return ToolResult(
                summary=f"后台进程已启动: session_id={session_id}, pid={proc.pid}",
                evidence=json.dumps({
                    "session_id": session_id,
                    "pid": proc.pid,
                    "command": command[:200],
                    "timeout": timeout,
                    "workdir": workdir,
                    "background": True,
                }, ensure_ascii=False),
            )

        # 前台：等待完成
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            _MANAGER.mark_finished(session_id, -1, timed_out=True)
            return ToolResult(
                summary=f"执行超时（{timeout}s）: {command[:100]}",
                evidence=json.dumps({
                    "timeout": timeout, "command": command[:120],
                    "workdir": workdir, "timed_out": True,
                }, ensure_ascii=False),
                error="TimeoutError",
                skipped=True,
            )

        output = stdout.decode(errors="replace").strip()
        info.stdout = output
        _MANAGER.mark_finished(session_id, proc.returncode)

        truncated = output[:preview_limit] + ("..." if len(output) > preview_limit else "")
        evidence = json.dumps({
            "command": command[:120], "exit_code": proc.returncode,
            "timeout": timeout, "workdir": workdir,
            "output_chars": len(output), "preview_chars": min(len(output), preview_limit),
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
        info.error = str(exc)
        _MANAGER.mark_finished(session_id, -1)
        _log.exception("exec 失败: %s", command)
        return ToolResult(summary=f"执行异常: {exc}", error=str(exc))


async def _collect_background_output(proc: asyncio.subprocess.Process, session_id: str) -> None:
    """后台收集进程输出。"""
    try:
        stdout, _ = await proc.communicate()
        info = _MANAGER.get(session_id)
        if info:
            info.stdout = stdout.decode(errors="replace").strip()
            info.return_code = proc.returncode
            _MANAGER.mark_finished(session_id, proc.return_code)
    except Exception as e:
        info = _MANAGER.get(session_id)
        if info:
            info.error = str(e)
            _MANAGER.mark_finished(session_id, -1)


# ── process：管理后台进程 ──────────────────────────────────────────────────────

_PROCESS_MANIFEST_LIST = ToolManifest(
    name="process.list",
    description="列出所有通过 exec 启动的进程。可过滤 running/finished/all。",
    params=[
        ToolParam("status", "string", "过滤：running/finished/all（默认 all）", required=False),
    ],
)

_PROCESS_MANIFEST_POLL = ToolManifest(
    name="process.poll",
    description="检查指定进程的状态。返回是否已完成、退出码、运行时间等。",
    params=[
        ToolParam("session_id", "string", "exec 启动时返回的 session_id", required=True),
    ],
)

_PROCESS_MANIFEST_LOG = ToolManifest(
    name="process.log",
    description="获取指定进程的标准输出。支持 offset/limit 分段读取。",
    params=[
        ToolParam("session_id", "string", "exec 启动时返回的 session_id", required=True),
        ToolParam("offset", "number", "从第几个字符开始读，默认 0", required=False),
        ToolParam("limit", "number", "最多读多少字符，默认 2000", required=False),
    ],
)

_PROCESS_MANIFEST_KILL = ToolManifest(
    name="process.kill",
    description="强制终止指定进程。",
    params=[
        ToolParam("session_id", "string", "exec 启动时返回的 session_id", required=True),
    ],
)


@tool(_PROCESS_MANIFEST_LIST)
async def process_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    status_filter = (params.get("status") or "all").lower()
    procs = _MANAGER.list_all()

    if status_filter == "running":
        procs = [p for p in procs if not p.finished]
    elif status_filter == "finished":
        procs = [p for p in procs if p.finished]

    if not procs:
        return ToolResult(summary=f"无进程（filter={status_filter})")

    lines = []
    for p in procs:
        state = "running" if not p.finished else f"done(exit={p.return_code})"
        duration = time.time() - p.started_at
        lines.append(
            f"  {p.session_id}: {state} | {p.command[:80]} | {duration:.0f}s"
        )

    return ToolResult(summary=f"进程列表 ({len(procs)} 个):\n" + "\n".join(lines))


@tool(_PROCESS_MANIFEST_POLL)
async def process_poll(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    session_id = params.get("session_id", "")
    info = _MANAGER.get(session_id)
    if not info:
        return ToolResult(summary=f"进程不存在: {session_id}", error="ProcessNotFound")

    duration = time.time() - info.started_at
    output_len = len(info.stdout)

    status = {
        "session_id": info.session_id,
        "command": info.command[:200],
        "status": "running" if not info.finished else "finished",
        "pid": info.pid,
        "return_code": info.return_code,
        "duration_seconds": round(duration, 1),
        "output_length": output_len,
        "error": info.error,
        "timed_out": info.timed_out,
    }

    return ToolResult(summary=json.dumps(status, ensure_ascii=False, indent=2))


@tool(_PROCESS_MANIFEST_LOG)
async def process_log(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    session_id = params.get("session_id", "")
    info = _MANAGER.get(session_id)
    if not info:
        return ToolResult(summary=f"进程不存在: {session_id}", error="ProcessNotFound")

    offset = int(params.get("offset") or 0)
    limit = int(params.get("limit") or 2000)
    output = info.stdout

    if offset >= len(output):
        return ToolResult(
            summary=f"输出总长 {len(output)} 字符，offset={offset} 超出范围",
            skipped=True,
        )

    chunk = output[offset:offset + limit]
    remaining = len(output) - offset - limit

    return ToolResult(
        summary=chunk,
        evidence=json.dumps({
            "session_id": session_id,
            "offset": offset,
            "limit": limit,
            "returned_chars": len(chunk),
            "remaining_chars": max(0, remaining),
            "total_output_chars": len(output),
        }, ensure_ascii=False),
    )


@tool(_PROCESS_MANIFEST_KILL)
async def process_kill(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    session_id = params.get("session_id", "")
    info = _MANAGER.get(session_id)
    if not info:
        return ToolResult(summary=f"进程不存在: {session_id}", error="ProcessNotFound")

    if info.finished:
        return ToolResult(
            summary=f"进程 {session_id} 已结束 (exit={info.return_code})",
            skipped=True,
        )

    try:
        import signal
        if info.pid:
            os.kill(info.pid, signal.SIGTERM)
            _MANAGER.mark_finished(session_id, -15)
            return ToolResult(summary=f"已发送 SIGTERM 到进程 {session_id} (pid={info.pid})")
        else:
            return ToolResult(
                summary=f"无法终止 {session_id}：没有 pid",
                error="NoPID",
            )
    except Exception as e:
        return ToolResult(summary=f"终止失败: {e}", error=str(e))
