"""cli/gateway.py — run / gateway 命令组（消息网关与认知循环启动）。"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from cli._common import console, load_cfg, DEFAULT_CONFIG_PATH
from cli.logs import logs_tail, logs_errors, logs_crash, logs_wechat, logs_stats
from cli.plugin import plugin_app

_PID_FILE = Path("~/.lingzhou/lingzhou.pid").expanduser()


def _configure_lingzhou_logging(
    log_dir: Path,
    log_level: int,
    *,
    logger_name: str = "lingzhou",
) -> tuple[Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"lingzhou-{datetime.now().strftime('%Y-%m-%d')}.log"
    console_log_file = log_dir / "console.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    keep_handlers: list[logging.Handler] = []
    for handler in logger.handlers:
        base = getattr(handler, "baseFilename", "")
        if base in {str(log_file), str(console_log_file)}:
            handler.close()
            continue
        keep_handlers.append(handler)
    logger.handlers = keep_handlers

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S"))
    file_handler.setLevel(log_level)

    console_handler = logging.FileHandler(console_log_file, mode="w", encoding="utf-8")
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
    console_handler.setLevel(log_level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return log_file, console_log_file


def _daemonize(argv: list[str]) -> None:
    """fork 子进程后台运行，父进程立即退出。"""
    # 去掉 --daemon / -d 标志，避免子进程再次 fork
    clean_argv = [a for a in argv if a not in ("--daemon", "-d")]
    pid = os.fork()
    if pid > 0:
        # 父进程：打印 PID 后退出
        console.print(f"[green]✓ 已后台启动[/green]  PID={pid}  日志: ~/.lingzhou/logs/")
        console.print(f"  停止: [bold]lingzhou stop[/bold]")
        raise typer.Exit(0)
    # 子进程：脱离终端
    os.setsid()
    # stdin → /dev/null, stdout/stderr → 崩溃日志（关键：保留 traceback）
    log_dir = Path("~/.lingzhou/logs").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    crash_log = os.open(str(log_dir / "crash.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)   # stdin
    os.dup2(crash_log, 1)  # stdout → crash.log
    os.dup2(crash_log, 2)  # stderr → crash.log
    os.close(crash_log)
    os.close(devnull)
    # 写 PID
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


# channel名称 → (描述, 是否需要 setup 配置)
_GATEWAY_CHANNELS: dict[str, tuple[str, bool]] = {
    "local":    ("本地终端 — 直接在当前终端运行，无需额外配置", False),
    "webhook":  ("HTTP Webhook — 对外暴露 /message 端点（适合集成其他系统）", True),
    "telegram": ("Telegram Bot — 需要 BOT_TOKEN", True),
    "wechat":   ("微信公众号 / 企业微信 — 开发中", True),
    "qq":       ("QQ Bot — 开发中", True),
}

# 已实现的渠道
_GATEWAY_READY = {"local", "webhook", "wechat"}

gateway_app = typer.Typer(name="gateway", help="消息网关（Telegram、Webhook 等）", no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})

# ── 日志子命令（注册在 gateway 下）─────────────────────────────────────────────
logs_group = typer.Typer(name="logs", help="快速查看运行日志")
logs_group.command("tail")(logs_tail)
logs_group.command("errors")(logs_errors)
logs_group.command("crash")(logs_crash)
logs_group.command("wechat")(logs_wechat)
logs_group.command("stats")(logs_stats)
gateway_app.add_typer(logs_group)
gateway_app.add_typer(plugin_app)


def _kill_existing_loop(quiet: bool = False) -> None:
    """杀掉 PID 文件中记录的旧进程，等待它真正退出（最长 8s）。"""
    import time as _t
    if not _PID_FILE.exists():
        return
    pid_str = _PID_FILE.read_text(encoding="utf-8").strip()
    _PID_FILE.unlink(missing_ok=True)
    try:
        old_pid = int(pid_str)
        os.kill(old_pid, 0)  # 先检查进程是否存在
    except (ProcessLookupError, ValueError, PermissionError):
        return  # 进程已不存在，跳过

    if not quiet:
        console.print(f"[yellow]正在关闭旧进程[/yellow]  PID={old_pid} …")
    try:
        os.kill(old_pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    # 等待进程退出（最长 8s），避免新进程与旧进程 overlap
    deadline = _t.monotonic() + 8.0
    while _t.monotonic() < deadline:
        _t.sleep(0.3)
        try:
            os.kill(old_pid, 0)
        except ProcessLookupError:
            if not quiet:
                console.print(f"[green]旧进程已退出[/green]  PID={old_pid}")
            return

    # 超时：强制 SIGKILL
    try:
        os.kill(old_pid, signal.SIGKILL)
        if not quiet:
            console.print(f"[yellow]旧进程未响应 SIGTERM，已发送 SIGKILL[/yellow]  PID={old_pid}")
    except ProcessLookupError:
        pass


@gateway_app.command("channels")
def gateway_channels() -> None:
    """列出支持的消息渠道。"""
    console.print("[bold]支持的消息渠道[/bold]\n")
    for ch, (desc, needs_setup) in _GATEWAY_CHANNELS.items():
        if ch in _GATEWAY_READY:
            status = "[green]可用[/green]"
            setup_hint = f"  [dim]lingzhou gateway setup --channel {ch}[/dim]" if needs_setup else ""
        else:
            status = "[dim]开发中[/dim]"
            setup_hint = ""
        console.print(f"  [cyan]{ch:<10}[/cyan] {status}  {desc}{setup_hint}")
    console.print(
        "\n[dim]启动: lingzhou gateway start --channel <name>[/dim]\n"
        "[dim]配置: lingzhou gateway setup --channel <name>[/dim]"
    )


@gateway_app.command("setup")
def gateway_setup(
    channel: Annotated[str, typer.Option("--channel", "-ch", help="渠道名称")] = "webhook",
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
) -> None:
    """配置消息渠道（向导模式）。local 渠道无需配置。"""
    from rich.panel import Panel

    if channel not in _GATEWAY_CHANNELS:
        console.print(f"[red]未知渠道: {channel}。支持: {', '.join(_GATEWAY_CHANNELS)}[/red]")
        raise typer.Exit(1)

    if channel == "local":
        console.print("[green]local 渠道无需配置，直接运行: lingzhou gateway start[/green]")
        return

    if channel not in _GATEWAY_READY:
        console.print(f"[yellow]{channel} 渠道尚在开发中，暂不支持配置。[/yellow]")
        raise typer.Exit(1)

    gw_dir = Path("~/.lingzhou/gateway").expanduser()
    gw_dir.mkdir(parents=True, exist_ok=True)
    gw_cfg_path = gw_dir / f"{channel}.json"

    console.print(Panel(
        f"[bold]网关配置向导[/bold]  渠道: [cyan]{channel}[/cyan]",
        border_style="blue",
    ))

    if channel == "telegram":
        console.print("\n  获取 Bot Token: 与 @BotFather 对话 → /newbot → 复制 token")
        token = typer.prompt("  BOT_TOKEN").strip()
        allowed_raw = typer.prompt("  允许的用户 ID（逗号分隔，留空则允许所有人）", default="").strip()
        allowed = [int(x.strip()) for x in allowed_raw.split(",") if x.strip().isdigit()]
        gw_conf: dict[str, Any] = {"channel": "telegram", "bot_token": token, "allowed_user_ids": allowed}
        gw_cfg_path.write_text(_json.dumps(gw_conf, ensure_ascii=False, indent=2), encoding="utf-8")
        gw_cfg_path.chmod(0o600)
        console.print(f"\n[green]✓ Telegram 网关配置已保存: {gw_cfg_path}[/green]")

    elif channel == "wechat":
        console.print("\n  iLink Bot Token（微信开放平台 → bot 管理 → 复制 Token）")
        token = typer.prompt("  ILINK_TOKEN").strip()
        base_url = typer.prompt("  iLink API 地址", default="https://ilinkai.weixin.qq.com").strip()
        poll_sec = typer.prompt("  长轮询超时（秒）", default="35")
        gw_conf: dict[str, Any] = {
            "channel": "wechat",
            "token": token,
            "base_url": base_url,
            "poll_sec": int(poll_sec),
            "reply_poll_sec": 3,
        }
        gw_cfg_path.write_text(_json.dumps(gw_conf, ensure_ascii=False, indent=2), encoding="utf-8")
        gw_cfg_path.chmod(0o600)
        console.print(f"\n[green]✓ 微信网关配置已保存: {gw_cfg_path}[/green]")

    elif channel == "webhook":
        host = typer.prompt("  监听地址", default="0.0.0.0")
        port = int(typer.prompt("  监听端口", default="8765"))
        secret = typer.prompt("  共享 secret（留空则无鉴权）", default="").strip() or None
        gw_conf: dict[str, Any] = {"channel": "webhook", "host": host, "port": port, "secret": secret}
        gw_cfg_path.write_text(_json.dumps(gw_conf, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"\n[green]✓ Webhook 网关配置已保存: {gw_cfg_path}[/green]")

    console.print(f"\n  启动: [bold]lingzhou gateway start --channel {channel}[/bold]")


@gateway_app.command("stop")
def gateway_stop() -> None:
    """停止后台运行的认知循环。"""
    if not _PID_FILE.exists():
        console.print("[yellow]未找到运行中的 lingzhou 进程（~/.lingzhou/lingzhou.pid 不存在）[/yellow]")
        raise typer.Exit(1)
    pid_str = _PID_FILE.read_text(encoding="utf-8").strip()
    try:
        pid = int(pid_str)
        os.kill(pid, signal.SIGTERM)
        _PID_FILE.unlink(missing_ok=True)
        console.print(f"[green]✓ 已发送停止信号[/green]  PID={pid}")
    except ProcessLookupError:
        console.print(f"[yellow]进程 {pid_str} 已不存在，清理 PID 文件[/yellow]")
        _PID_FILE.unlink(missing_ok=True)
        raise typer.Exit(1)
    except ValueError:
        console.print(f"[red]PID 文件内容无效: {pid_str!r}[/red]")


@gateway_app.command("status")
def gateway_status() -> None:
    """查看认知循环运行状态。"""
    import time as _t

    if not _PID_FILE.exists():
        console.print("[yellow]● 未运行[/yellow]  (PID 文件不存在)")
        raise typer.Exit(1)

    pid_str = _PID_FILE.read_text(encoding="utf-8").strip()
    try:
        pid = int(pid_str)
        os.kill(pid, 0)  # 仅探测进程是否存在，不发送真实信号
    except ProcessLookupError:
        console.print(f"[yellow]● 进程已退出[/yellow]  PID={pid_str}（PID 文件残留，可运行 lingzhou gateway stop 清理）")
        raise typer.Exit(1)
    except ValueError:
        console.print(f"[red]PID 文件内容无效: {pid_str!r}[/red]")
        raise typer.Exit(1)

    # 用 PID 文件 mtime 估算运行时长
    try:
        mtime = _PID_FILE.stat().st_mtime
        uptime_s = int(_t.time() - mtime)
        h, rem = divmod(uptime_s, 3600)
        m, s = divmod(rem, 60)
        uptime_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
    except OSError:
        uptime_str = "未知"

    console.print(f"[green]● 运行中[/green]  PID={pid}  已运行 {uptime_str}")
    console.print(f"  日志: [dim]~/.lingzhou/logs/[/dim]")
    console.print(f"  停止: [dim]lingzhou gateway stop[/dim]")
    console.print(f"  重启: [dim]lingzhou gateway restart[/dim]")


@gateway_app.command("restart")
def gateway_restart(
    channel: Annotated[str, typer.Option("--channel", "-ch", help="消息渠道（默认 local）")] = "local",
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    debug: Annotated[Optional[bool], typer.Option("--debug/--no-debug")] = None,
    dry_run: Annotated[Optional[bool], typer.Option("--dry-run/--act")] = None,
) -> None:
    """重启认知循环（stop + start）。"""
    _kill_existing_loop(quiet=False)
    gateway_start(channel=channel, config=config, debug=debug, dry_run=dry_run, daemon=True)


@gateway_app.command("start")
def gateway_start(
    channel: Annotated[Optional[str], typer.Option("--channel", "-ch", help="消息渠道（默认从 lingzhou.json gateway.default_channel 读取）")] = None,
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    debug: Annotated[Optional[bool], typer.Option("--debug/--no-debug")] = None,
    dry_run: Annotated[Optional[bool], typer.Option("--dry-run/--act")] = None,
    daemon: Annotated[bool, typer.Option("--daemon/--no-daemon", "-d/-f", help="后台运行，默认已开启；--no-daemon 前台运行")] = True,
) -> None:
    """启动认知循环 + 消息渠道（loop 是内核，channel 是 I/O 层）。

    local    — 本地终端，无需配置，直接运行
    webhook  — HTTP 接入，loop 与 webhook server 并行
    wechat   — 微信 bot（通过 hermesclaw 代理）
    telegram — Telegram Bot（开发中）
    """
    # 从 config 读取默认渠道（直接读 JSON，避免 Pydantic 模型不包含 gateway 字段）
    if channel is None:
        try:
            _raw = _json.loads(config.read_text(encoding="utf-8"))
            channel = _raw.get("gateway", {}).get("default_channel", "local")
        except Exception:
            channel = "local"

    if channel not in _GATEWAY_CHANNELS:
        console.print(f"[yellow]{channel} 渠道尚在开发中。当前可用: {', '.join(_GATEWAY_READY)}[/yellow]")
        raise typer.Exit(1)

    if daemon and hasattr(os, "fork"):
        # 启动前先杀掉旧进程，避免两个 loop 同时运行
        _kill_existing_loop(quiet=False)
        _daemonize(sys.argv)
        # 子进程从这里继续执行（父进程已退出）
    else:
        # 前台模式：检测是否有旧进程，有则拒绝启动
        _kill_existing_loop(quiet=True)
        # 前台模式也写 PID，防止并发启动
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

    # 非 local 渠道需要提前 setup
    gw_conf: dict[str, Any] = {}
    if channel != "local":
        gw_cfg_path = Path("~/.lingzhou/gateway").expanduser() / f"{channel}.json"
        if not gw_cfg_path.exists():
            console.print(
                f"[red]渠道 {channel!r} 尚未配置，请先运行: "
                f"lingzhou gateway setup --channel {channel}[/red]"
            )
            raise typer.Exit(1)
        gw_conf = _json.loads(gw_cfg_path.read_text(encoding="utf-8"))

    cfg = load_cfg(config)
    # 加载 .env 确保 daemon 进程有 API keys
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv(Path("~/.lingzhou/.env").expanduser())
    except Exception:
        pass
    if debug is not None:
        cfg.loop.debug = debug
    if dry_run is not None:
        cfg.loop.act = not dry_run

    # 日志
    log_dir = Path("~/.lingzhou/logs").expanduser()
    log_level = logging.DEBUG if (debug or cfg.loop.debug) else logging.INFO
    log_file, console_log_file = _configure_lingzhou_logging(log_dir, log_level)
    console.print(f"[dim]渠道: [cyan]{channel}[/cyan]  日志: {log_file}  console: {console_log_file}[/dim]")

    # 启动 channel sidecar（loop 主线程仍是 asyncio）
    if channel == "webhook":
        _start_webhook_sidecar(gw_conf, cfg)
    elif channel == "wechat":
        _start_wechat_sidecar(gw_conf, cfg)

    from core.loop import CognitionLoop
    loop_instance = CognitionLoop(cfg)
    try:
        asyncio.run(loop_instance.run())
    except KeyboardInterrupt:
        console.print("\n[dim]认知循环已停止。[/dim]")
    finally:
        # 前台模式：退出时清理 PID 文件
        if _PID_FILE.exists():
            try:
                if _PID_FILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
                    _PID_FILE.unlink(missing_ok=True)
            except OSError:
                pass


def run(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    channel: Annotated[Optional[str], typer.Option("--channel", "-ch", help="消息渠道（默认从 lingzhou.json gateway.default_channel 读取）")] = None,
    debug: Annotated[Optional[bool], typer.Option("--debug/--no-debug")] = None,
    dry_run: Annotated[Optional[bool], typer.Option("--dry-run/--act")] = None,
    daemon: Annotated[bool, typer.Option("--daemon/--no-daemon", "-d/-f", help="后台运行，默认已开启；--no-daemon 前台运行")] = True,
) -> None:
    """启动认知循环（等同于 gateway start）。"""
    gateway_start(channel=channel, config=config, debug=debug, dry_run=dry_run, daemon=daemon)


def stop() -> None:
    """停止后台运行的认知循环。"""
    if not _PID_FILE.exists():
        console.print("[yellow]未找到运行中的 lingzhou 进程（~/.lingzhou/lingzhou.pid 不存在）[/yellow]")
        raise typer.Exit(1)
    pid_str = _PID_FILE.read_text(encoding="utf-8").strip()
    try:
        pid = int(pid_str)
        os.kill(pid, signal.SIGTERM)
        _PID_FILE.unlink(missing_ok=True)
        console.print(f"[green]✓ 已发送停止信号[/green]  PID={pid}")
    except ProcessLookupError:
        console.print(f"[yellow]进程 {pid_str} 已不存在，清理 PID 文件[/yellow]")
        _PID_FILE.unlink(missing_ok=True)
        raise typer.Exit(1)
    except ValueError:
        console.print(f"[red]PID 文件内容无效: {pid_str!r}[/red]")
        raise typer.Exit(1)


def _start_webhook_sidecar(gw_conf: dict[str, Any], cfg: Any) -> None:
    """在 daemon 线程中启动 webhook HTTP 服务，与主 loop asyncio 并行。

    POST /message  {"message": "...", "priority": "high"}
    → 同步写入 SQLite tasks 表 → loop 下一个 tick 消费
    """
    import sqlite3
    import datetime as _dt

    host = gw_conf.get("host", "0.0.0.0")
    port = int(gw_conf.get("port", 8765))
    secret = gw_conf.get("secret")
    db_path = str(cfg.db_path)

    console.print(
        f"[dim]Webhook 监听: http://{host}:{port}/message"
        f"{'  (Bearer token)' if secret else '  (无鉴权)'}[/dim]"
    )

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # 静默访问日志

        def do_POST(self) -> None:
            if self.path != "/message":
                self.send_response(404); self.end_headers(); return
            if secret:
                if self.headers.get("Authorization", "") != f"Bearer {secret}":
                    self.send_response(401); self.end_headers(); return
            length = min(int(self.headers.get("Content-Length", 0)), 65536)
            body = self.rfile.read(length)
            try:
                payload = _json.loads(body)
                msg = payload.get("message", "").strip()
                priority = payload.get("priority", "high")
            except Exception:
                self.send_response(400); self.end_headers(); return
            if not msg:
                self.send_response(400); self.end_headers()
                self.wfile.write(b'{"error":"empty message"}'); return

            short = msg.replace("\n", " ")[:28] + ("..." if len(msg) > 28 else "")
            now = _dt.datetime.now(_dt.UTC).isoformat()
            try:
                conn = sqlite3.connect(db_path)
                try:
                    data_json = _json.dumps(
                        {"goal": msg, "source": "gateway:webhook", "next_step": ""},
                        ensure_ascii=False,
                    )
                    conn.execute(
                        "INSERT INTO tasks (title, status, priority, created_at, data) VALUES (?,?,?,?,?)",
                        (f"webhook: {short}", "pending", priority, now, data_json),
                    )
                    conn.commit()
                    task_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                finally:
                    conn.close()
                resp = _json.dumps({"ok": True, "task_id": task_id}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp)
                console.print(f"[green][webhook] 注入任务 id={task_id}: {short}[/green]")
            except Exception as e:
                self.send_response(500); self.end_headers()
                self.wfile.write(_json.dumps({"error": str(e)}).encode())

    server = HTTPServer((host, port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="webhook-gateway")
    t.start()


def _start_wechat_sidecar(gw_conf: dict[str, Any], cfg: Any) -> None:
    """启动微信 iLink 通道 sidecar（daemon 线程）。

    iLink long-poll → 写入 SQLite tasks 表 → loop 消费 → 回复通过 iLink 发送
    """
    from core.wechat_channel import start_wechat_channel

    db_path = str(cfg.db_path)
    console.print(
        f"[dim]微信 iLink: {gw_conf.get('base_url', 'https://ilinkai.weixin.qq.com')}"
        f"  poll={gw_conf.get('poll_sec', 35)}s[/dim]"
    )
    start_wechat_channel(gw_conf, db_path)
