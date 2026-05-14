"""tools/exec.py — exec/process 工具。

目标：
- exec：启动 shell 命令，支持前台/后台、PTY、超时、工作目录、环境变量
- process：管理已启动的后台进程（list/poll/log/write/kill）

注意：
- approval / security 语义刻意不对标 OpenClaw；这里只补能力本体
- 进程状态当前为进程内内存态，runtime 重启后不会恢复（后续可持久化）
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import select
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    timeout_seconds: float | None = None
    proc: Any | None = None
    master_fd: int | None = None
    watch_task: asyncio.Task | None = None
    log_path: str = ""
    meta_path: str = ""
    restored: bool = False
    handle_lost: bool = False
    _output_chunks: list[str] = field(default_factory=list)


class ProcessManager:
    """追踪所有通过 exec 启动的进程，并把最小状态持久化到磁盘。"""

    _counter: int = 0
    _processes: dict[str, ProcessInfo] = {}
    _loaded: bool = False

    @classmethod
    def _state_root(cls) -> Path:
        root = Path(os.environ.get("LINGZHOU_PROCESS_STATE_DIR") or (Path.home() / ".lingzhou/state/processes"))
        root.mkdir(parents=True, exist_ok=True)
        return root

    @classmethod
    def _meta_path(cls, session_id: str) -> Path:
        return cls._state_root() / f"{session_id}.json"

    @classmethod
    def _log_path(cls, session_id: str) -> Path:
        return cls._state_root() / f"{session_id}.log"

    @classmethod
    def _pid_alive(cls, pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False

    @classmethod
    def _persist(cls, info: ProcessInfo) -> None:
        if not info.meta_path:
            info.meta_path = str(cls._meta_path(info.session_id))
        if not info.log_path:
            info.log_path = str(cls._log_path(info.session_id))
        payload = {
            "session_id": info.session_id,
            "command": info.command,
            "pid": info.pid,
            "started_at": info.started_at,
            "finished_at": info.finished_at,
            "return_code": info.return_code,
            "error": info.error,
            "background": info.background,
            "finished": info.finished,
            "timed_out": info.timed_out,
            "pty": info.pty,
            "workdir": info.workdir,
            "timeout_seconds": info.timeout_seconds,
            "log_path": info.log_path,
            "meta_path": info.meta_path,
            "restored": info.restored,
            "handle_lost": info.handle_lost,
        }
        path = Path(info.meta_path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def _load_stdout(cls, info: ProcessInfo) -> None:
        if not info.log_path:
            return
        path = Path(info.log_path)
        if not path.exists():
            return
        try:
            info.stdout = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    @classmethod
    def _refresh_liveness(cls, info: ProcessInfo) -> None:
        if info.finished:
            return
        if info.proc is not None:
            return
        if info.pid and not cls._pid_alive(info.pid):
            info.finished = True
            info.finished_at = info.finished_at or time.time()
            if info.return_code is None:
                info.return_code = -1
            cls._persist(info)

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        root = cls._state_root()
        for meta in sorted(root.glob("exec-*.json")):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                continue
            sid = str(data.get("session_id") or meta.stem)
            info = ProcessInfo(
                session_id=sid,
                command=str(data.get("command") or ""),
                pid=data.get("pid"),
                started_at=float(data.get("started_at") or 0.0),
                finished_at=data.get("finished_at"),
                return_code=data.get("return_code"),
                error=data.get("error"),
                background=bool(data.get("background", False)),
                finished=bool(data.get("finished", False)),
                timed_out=bool(data.get("timed_out", False)),
                pty=bool(data.get("pty", False)),
                workdir=str(data.get("workdir") or ""),
                timeout_seconds=data.get("timeout_seconds"),
                log_path=str(data.get("log_path") or cls._log_path(sid)),
                meta_path=str(data.get("meta_path") or meta),
                restored=True,
                handle_lost=not bool(data.get("finished", False)),
            )
            cls._load_stdout(info)
            cls._refresh_liveness(info)
            cls._processes[sid] = info
        cls._loaded = True

    @classmethod
    def next_id(cls) -> str:
        cls._ensure_loaded()
        cls._counter += 1
        return f"exec-{int(time.time() * 1000)}-{cls._counter}"

    @classmethod
    def register(cls, info: ProcessInfo) -> str:
        cls._ensure_loaded()
        info.meta_path = str(cls._meta_path(info.session_id))
        info.log_path = str(cls._log_path(info.session_id))
        Path(info.log_path).touch(exist_ok=True)
        cls._processes[info.session_id] = info
        cls._persist(info)
        return info.session_id

    @classmethod
    def get(cls, session_id: str) -> ProcessInfo | None:
        cls._ensure_loaded()
        info = cls._processes.get(session_id)
        if info:
            cls._refresh_liveness(info)
        return info

    @classmethod
    def list_all(cls) -> list[ProcessInfo]:
        cls._ensure_loaded()
        for info in cls._processes.values():
            cls._refresh_liveness(info)
        return list(cls._processes.values())

    @classmethod
    def mark_finished(cls, session_id: str, return_code: int, timed_out: bool = False) -> None:
        cls._ensure_loaded()
        info = cls._processes.get(session_id)
        if info:
            info.finished = True
            info.finished_at = time.time()
            info.return_code = return_code
            info.timed_out = timed_out
            info.handle_lost = False
            cls._persist(info)

    @classmethod
    def clear(cls) -> None:
        cls._processes.clear()
        cls._counter = 0
        cls._loaded = True
        root = cls._state_root()
        for p in root.glob("exec-*"):
            try:
                p.unlink()
            except Exception:
                pass


_MANAGER = ProcessManager()


# ── 辅助 ────────────────────────────────────────────────────────────────────

def _append_output(info: ProcessInfo, text: str) -> None:
    if not text:
        return
    info._output_chunks.append(text)
    info.stdout += text
    if info.log_path:
        try:
            with open(info.log_path, "a", encoding="utf-8") as fh:
                fh.write(text)
        except Exception:
            pass


def _preview(text: str, limit: int) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def _terminate_info(info: ProcessInfo, *, force: bool = False) -> None:
    proc = info.proc
    try:
        if proc is None:
            return
        if isinstance(proc, asyncio.subprocess.Process):
            if proc.returncode is None:
                (proc.kill if force else proc.terminate)()
        elif isinstance(proc, subprocess.Popen):
            if proc.poll() is None:
                (proc.kill if force else proc.terminate)()
        elif info.pid:
            os.kill(info.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception as e:
        info.error = str(e)


def _build_capabilities_v2(workdir: str) -> dict[str, Any]:
    common = (
        "python3", "python", "bash", "sh", "grep", "find", "ls", "cat",
        "sqlite3", "git", "sed", "awk", "jq", "rg",
    )
    available = [cmd for cmd in common if shutil.which(cmd)]
    try:
        import pty as _pty  # noqa: F401
        has_pty = True
    except Exception:
        has_pty = False
    return {
        "engine": "exec/process runtime",
        "execution_model": "foreground or background",
        "sandbox": False,
        "network_policy": "inherits-host-environment",
        "default_timeout_sec": 30,
        "default_output_preview_chars": 500,
        "workdir": workdir,
        "shell": os.environ.get("SHELL") or "/bin/sh",
        "available_commands": available,
        "has_background_exec": True,
        "has_process_management": True,
        "has_pty": has_pty,
        "has_process_write": True,
    }


# ── shell.capabilities 覆盖增强版 ─────────────────────────────────────────────

_CAP_MANIFEST_V2 = ToolManifest(
    name="shell.capabilities",
    description="返回 shell 执行能力画像（可用命令、默认限制、环境语义、exec/process 支持）",
    params=[],
)


@tool(_CAP_MANIFEST_V2)
async def shell_capabilities(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    workdir = params.get("workdir", str(Path.cwd()))
    caps = _build_capabilities_v2(workdir)
    summary = (
        f"shell.capabilities: sandbox={caps['sandbox']} "
        f"background={caps['has_background_exec']} "
        f"process_mgmt={caps['has_process_management']} "
        f"process_write={caps['has_process_write']} "
        f"pty={caps['has_pty']} "
        f"cmds={len(caps['available_commands'])}"
    )
    return ToolResult(
        summary=summary,
        evidence=json.dumps(caps, ensure_ascii=False),
        resource_key=workdir,
        fingerprint=f"caps:{len(caps['available_commands'])}:{int(caps['has_pty'])}",
        metadata={"caps": caps},
    )


# ── exec：启动命令 ───────────────────────────────────────────────────────────

_EXEC_MANIFEST = ToolManifest(
    name="exec",
    description=(
        "启动 shell 命令。支持前台阻塞执行或后台运行。"
        "前台模式：等待命令完成，返回完整输出（受 timeout 限制）。"
        "后台模式：立即返回 session_id，后续通过 process 工具管理。"
        "支持 pty=true 运行需要 TTY 的程序。"
    ),
    params=[
        ToolParam("command", "string", "要执行的 shell 命令", required=True),
        ToolParam("background", "boolean", "是否后台运行（默认 false）", required=False),
        ToolParam("pty", "boolean", "是否使用 PTY（适合交互式程序）", required=False),
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

    background = bool(params.get("background", False))
    use_pty = bool(params.get("pty", False))
    timeout = float(params.get("timeout") or (300.0 if background else 30.0))
    workdir = str(params.get("workdir") or Path.cwd())
    preview_limit = int(params.get("max_output_chars") or 500)
    env_overrides = params.get("env")

    if ctx.dry_run:
        return ToolResult(
            summary=f"[dry-run] exec: {command[:200]}",
            evidence=json.dumps({
                "dry_run": True,
                "command": command[:120],
                "timeout": timeout,
                "workdir": workdir,
                "background": background,
                "pty": use_pty,
            }, ensure_ascii=False),
            skipped=True,
        )

    exec_env = os.environ.copy()
    if env_overrides and isinstance(env_overrides, dict):
        exec_env.update({str(k): str(v) for k, v in env_overrides.items()})

    session_id = _MANAGER.next_id()
    info = ProcessInfo(
        session_id=session_id,
        command=command,
        started_at=time.time(),
        background=background,
        workdir=workdir,
        timeout_seconds=timeout,
        pty=use_pty,
    )
    _MANAGER.register(info)

    try:
        if use_pty:
            proc, master_fd = _spawn_pty_process(command, workdir, exec_env)
            info.proc = proc
            info.pid = proc.pid
            info.master_fd = master_fd
            if background:
                info.watch_task = asyncio.create_task(_watch_pty_process(info))
                payload = {
                    "session_id": session_id,
                    "pid": proc.pid,
                    "command": command[:200],
                    "timeout": timeout,
                    "workdir": workdir,
                    "background": True,
                    "pty": True,
                }
                return ToolResult(
                    summary=f"后台 PTY 进程已启动: session_id={session_id}, pid={proc.pid}",
                    evidence=json.dumps(payload, ensure_ascii=False),
                    resource_key=session_id,
                    artifact_paths=[info.meta_path, info.log_path],
                    state_delta={"process": "started", "background": True, "pty": True},
                    metadata=payload,
                )
            await _watch_pty_process(info)
        else:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=workdir,
                env=exec_env,
            )
            info.proc = proc
            info.pid = proc.pid
            if background:
                info.watch_task = asyncio.create_task(_watch_pipe_process(info))
                payload = {
                    "session_id": session_id,
                    "pid": proc.pid,
                    "command": command[:200],
                    "timeout": timeout,
                    "workdir": workdir,
                    "background": True,
                    "pty": False,
                }
                return ToolResult(
                    summary=f"后台进程已启动: session_id={session_id}, pid={proc.pid}",
                    evidence=json.dumps(payload, ensure_ascii=False),
                    resource_key=session_id,
                    artifact_paths=[info.meta_path, info.log_path],
                    state_delta={"process": "started", "background": True, "pty": False},
                    metadata=payload,
                )
            await _watch_pipe_process(info)

        output = info.stdout.strip()
        if info.timed_out:
            payload = {
                "timeout": timeout,
                "command": command[:120],
                "workdir": workdir,
                "timed_out": True,
                "pty": use_pty,
                "session_id": session_id,
            }
            return ToolResult(
                summary=f"执行超时（{timeout}s）: {command[:100]}",
                evidence=json.dumps(payload, ensure_ascii=False),
                error="TimeoutError",
                skipped=True,
                resource_key=session_id,
                artifact_paths=[info.meta_path, info.log_path],
                state_delta={"process": "timed_out"},
                metadata=payload,
            )

        preview_text = output or "(无输出)"
        truncated = _preview(preview_text, preview_limit)
        evidence = json.dumps({
            "command": command[:120],
            "exit_code": info.return_code,
            "timeout": timeout,
            "workdir": workdir,
            "output_chars": len(output),
            "preview_chars": min(len(output), preview_limit),
            "pty": use_pty,
        }, ensure_ascii=False)
        payload = json.loads(evidence)
        payload.update({"session_id": session_id, "meta_path": info.meta_path, "log_path": info.log_path})
        if info.return_code == 0:
            return ToolResult(
                summary=f"执行成功:\n{truncated}",
                evidence=json.dumps(payload, ensure_ascii=False),
                resource_key=session_id,
                fingerprint=f"exec:{info.return_code}:{payload['output_chars']}",
                artifact_paths=[info.meta_path, info.log_path],
                state_delta={"process": "finished", "exit_code": info.return_code},
                metadata=payload,
            )
        return ToolResult(
            summary=f"执行出错 (exit={info.return_code}):\n{truncated}",
            evidence=json.dumps(payload, ensure_ascii=False),
            error=(output[:300] or info.error or f"exit={info.return_code}"),
            resource_key=session_id,
            fingerprint=f"exec:{info.return_code}:{payload['output_chars']}",
            artifact_paths=[info.meta_path, info.log_path],
            state_delta={"process": "finished", "exit_code": info.return_code},
            metadata=payload,
        )
    except Exception as exc:
        info.error = str(exc)
        _MANAGER.mark_finished(session_id, -1)
        _log.exception("exec 失败: %s", command)
        ProcessManager._persist(info)
        return ToolResult(
            summary=f"执行异常: {exc}",
            error=str(exc),
            resource_key=session_id,
            artifact_paths=[info.meta_path, info.log_path] if info.meta_path or info.log_path else [],
            state_delta={"process": "failed_to_start"},
            metadata={"session_id": session_id, "command": command[:200], "workdir": workdir},
        )


def _spawn_pty_process(command: str, workdir: str, env: dict[str, str]) -> tuple[subprocess.Popen[Any], int]:
    import pty

    master_fd, slave_fd = pty.openpty()
    os.set_blocking(master_fd, False)
    proc = subprocess.Popen(
        [os.environ.get("SHELL") or "/bin/bash", "-lc", command],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd=workdir,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)
    return proc, master_fd


async def _watch_pipe_process(info: ProcessInfo) -> None:
    proc: asyncio.subprocess.Process = info.proc
    assert proc is not None

    async def _reader() -> None:
        if proc.stdout is None:
            return
        while True:
            chunk = await proc.stdout.read(1024)
            if not chunk:
                break
            _append_output(info, chunk.decode(errors="replace"))

    reader_task = asyncio.create_task(_reader())
    try:
        await asyncio.wait_for(proc.wait(), timeout=info.timeout_seconds)
    except asyncio.TimeoutError:
        _terminate_info(info)
        await asyncio.sleep(0.1)
        _terminate_info(info, force=True)
        info.error = "TimeoutError"
        _MANAGER.mark_finished(info.session_id, -1, timed_out=True)
        _MANAGER._persist(info)
    except Exception as e:
        info.error = str(e)
        _MANAGER.mark_finished(info.session_id, -1)
        _MANAGER._persist(info)
    else:
        _MANAGER.mark_finished(info.session_id, proc.returncode if proc.returncode is not None else -1)
        _MANAGER._persist(info)
    finally:
        try:
            await asyncio.wait_for(reader_task, timeout=1.0)
        except Exception:
            reader_task.cancel()
        try:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()
            if proc.stdout:
                await proc.stdout.read()
        except Exception:
            pass
        info.proc = None


def _run_pty_until_exit(info: ProcessInfo) -> tuple[int, bool, str | None]:
    proc: subprocess.Popen[Any] = info.proc
    master_fd = info.master_fd
    assert proc is not None and master_fd is not None

    timed_out = False
    err: str | None = None
    start = time.time()
    try:
        while True:
            if info.timeout_seconds and (time.time() - start) > info.timeout_seconds and proc.poll() is None:
                proc.terminate()
                time.sleep(0.1)
                if proc.poll() is None:
                    proc.kill()
                timed_out = True

            try:
                ready, _, _ = select.select([master_fd], [], [], 0.1)
            except (OSError, ValueError):
                ready = []

            if ready:
                try:
                    chunk = os.read(master_fd, 1024)
                    if chunk:
                        _append_output(info, chunk.decode(errors="replace"))
                except BlockingIOError:
                    pass
                except OSError:
                    pass

            rc = proc.poll()
            if rc is not None:
                for _ in range(5):
                    try:
                        chunk = os.read(master_fd, 1024)
                        if not chunk:
                            break
                        _append_output(info, chunk.decode(errors="replace"))
                    except Exception:
                        break
                return rc, timed_out, err
    except Exception as e:
        err = str(e)
        return -1, timed_out, err
    finally:
        try:
            os.close(master_fd)
        except Exception:
            pass
        info.master_fd = None


async def _watch_pty_process(info: ProcessInfo) -> None:
    rc, timed_out, err = await asyncio.to_thread(_run_pty_until_exit, info)
    if err:
        info.error = err
    info.proc = None
    _MANAGER.mark_finished(info.session_id, rc, timed_out=timed_out)
    _MANAGER._persist(info)


# ── process：管理后台进程 ────────────────────────────────────────────────────

_PROCESS_MANIFEST_LIST = ToolManifest(
    name="process.list",
    description="列出所有通过 exec 启动的进程。可过滤 running/finished/all。",
    params=[ToolParam("status", "string", "过滤：running/finished/all（默认 all）", required=False)],
)

_PROCESS_MANIFEST_POLL = ToolManifest(
    name="process.poll",
    description="检查指定进程的状态。返回是否已完成、退出码、运行时间等。",
    params=[ToolParam("session_id", "string", "exec 启动时返回的 session_id", required=True)],
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

_PROCESS_MANIFEST_WRITE = ToolManifest(
    name="process.write",
    description="向后台进程写入 stdin / PTY 输入。可选 eof=true 关闭输入。",
    params=[
        ToolParam("session_id", "string", "exec 启动时返回的 session_id", required=True),
        ToolParam("data", "string", "要写入的文本", required=False),
        ToolParam("eof", "boolean", "写入后是否关闭输入（默认 false）", required=False),
    ],
)

_PROCESS_MANIFEST_KILL = ToolManifest(
    name="process.kill",
    description="强制终止指定进程。",
    params=[ToolParam("session_id", "string", "exec 启动时返回的 session_id", required=True)],
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
        mode = "pty" if p.pty else "pipe"
        duration = time.time() - p.started_at
        lines.append(f"  {p.session_id}: {state} [{mode}] | {p.command[:80]} | {duration:.0f}s")
    return ToolResult(
        summary=f"进程列表 ({len(procs)} 个):\n" + "\n".join(lines),
        metadata={"count": len(procs), "status_filter": status_filter},
    )


@tool(_PROCESS_MANIFEST_POLL)
async def process_poll(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    session_id = params.get("session_id", "")
    info = _MANAGER.get(session_id)
    if not info:
        return ToolResult(summary=f"进程不存在: {session_id}", error="ProcessNotFound")
    duration = time.time() - info.started_at
    status = {
        "session_id": info.session_id,
        "command": info.command[:200],
        "status": "running" if not info.finished else "finished",
        "pid": info.pid,
        "pty": info.pty,
        "return_code": info.return_code,
        "duration_seconds": round(duration, 1),
        "output_length": len(info.stdout),
        "error": info.error,
        "timed_out": info.timed_out,
        "restored": info.restored,
        "handle_lost": info.handle_lost,
        "meta_path": info.meta_path,
        "log_path": info.log_path,
    }
    return ToolResult(
        summary=json.dumps(status, ensure_ascii=False, indent=2),
        resource_key=session_id,
        fingerprint=f"poll:{status['status']}:{status['return_code']}",
        artifact_paths=[p for p in [info.meta_path, info.log_path] if p],
        metadata=status,
    )


@tool(_PROCESS_MANIFEST_LOG)
async def process_log(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    session_id = params.get("session_id", "")
    info = _MANAGER.get(session_id)
    if not info:
        return ToolResult(summary=f"进程不存在: {session_id}", error="ProcessNotFound")
    offset = int(params.get("offset") or 0)
    limit = int(params.get("limit") or 2000)
    if info.log_path and Path(info.log_path).exists():
        try:
            output = Path(info.log_path).read_text(encoding="utf-8", errors="replace")
            info.stdout = output
        except Exception:
            output = info.stdout
    else:
        output = info.stdout
    if offset >= len(output):
        return ToolResult(summary=f"输出总长 {len(output)} 字符，offset={offset} 超出范围", skipped=True)
    chunk = output[offset:offset + limit]
    remaining = len(output) - offset - len(chunk)
    payload = {
        "session_id": session_id,
        "offset": offset,
        "limit": limit,
        "returned_chars": len(chunk),
        "remaining_chars": max(0, remaining),
        "total_output_chars": len(output),
        "log_path": info.log_path,
    }
    return ToolResult(
        summary=chunk,
        evidence=json.dumps(payload, ensure_ascii=False),
        resource_key=session_id,
        fingerprint=f"log:{offset}:{len(chunk)}",
        artifact_paths=[p for p in [info.meta_path, info.log_path] if p],
        metadata=payload,
    )


@tool(_PROCESS_MANIFEST_WRITE)
async def process_write(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    session_id = params.get("session_id", "")
    data = str(params.get("data") or "")
    eof = bool(params.get("eof", False))
    info = _MANAGER.get(session_id)
    if not info:
        return ToolResult(summary=f"进程不存在: {session_id}", error="ProcessNotFound")
    if info.finished:
        return ToolResult(summary=f"进程 {session_id} 已结束，不能再写入", error="ProcessFinished", skipped=True)
    if info.handle_lost or info.proc is None and info.pid and info.restored:
        return ToolResult(
            summary=f"进程 {session_id} 来自重启前的持久状态，当前无法恢复 stdin/PTY 写入句柄；可继续 poll/log/kill。",
            error="ProcessHandleLost",
            skipped=True,
            resource_key=session_id,
            artifact_paths=[p for p in [info.meta_path, info.log_path] if p],
        )

    try:
        if info.pty and info.master_fd is not None:
            if data:
                os.write(info.master_fd, data.encode())
            if eof:
                os.write(info.master_fd, b"\x04")
        else:
            proc: asyncio.subprocess.Process = info.proc
            if proc is None or proc.stdin is None:
                return ToolResult(summary=f"进程 {session_id} 没有可写 stdin", error="NoStdin")
            if data:
                proc.stdin.write(data.encode())
                await proc.stdin.drain()
            if eof:
                proc.stdin.close()
        _MANAGER._persist(info)
        return ToolResult(
            summary=f"已写入进程 {session_id}: {len(data)} 字符{' + EOF' if eof else ''}",
            resource_key=session_id,
            state_delta={"stdin_write_chars": len(data), "eof": eof},
            artifact_paths=[p for p in [info.meta_path, info.log_path] if p],
            metadata={"session_id": session_id, "chars": len(data), "eof": eof},
        )
    except Exception as e:
        info.error = str(e)
        _MANAGER._persist(info)
        return ToolResult(summary=f"写入失败: {e}", error=str(e), resource_key=session_id)


@tool(_PROCESS_MANIFEST_KILL)
async def process_kill(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    session_id = params.get("session_id", "")
    info = _MANAGER.get(session_id)
    if not info:
        return ToolResult(summary=f"进程不存在: {session_id}", error="ProcessNotFound")
    if info.finished:
        return ToolResult(summary=f"进程 {session_id} 已结束 (exit={info.return_code})", skipped=True)
    try:
        _terminate_info(info)
        await asyncio.sleep(0.1)
        if not info.finished:
            _terminate_info(info, force=True)
        _MANAGER.mark_finished(session_id, -15)
        _MANAGER._persist(info)
        return ToolResult(
            summary=f"已终止进程 {session_id} (pid={info.pid})",
            resource_key=session_id,
            state_delta={"process": "killed"},
            artifact_paths=[p for p in [info.meta_path, info.log_path] if p],
            metadata={"session_id": session_id, "pid": info.pid},
        )
    except Exception as e:
        info.error = str(e)
        return ToolResult(summary=f"终止失败: {e}", error=str(e))
