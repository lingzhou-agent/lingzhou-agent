"""cli/logs.py — 快速查看日志命令（注册在 gateway_app 下）。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from cli._common import console

LOG_DIR = Path("~/.lingzhou/logs").expanduser()
DAILY_LOG = LOG_DIR / "lingzhou-2026-05-17.log"  # fallback，运行时动态拼接


def _latest_log() -> Path:
    """返回最新的每日日志文件。"""
    from datetime import datetime
    today = LOG_DIR / f"lingzhou-{datetime.now().strftime('%Y-%m-%d')}.log"
    if today.exists():
        return today
    # 回退：找最近的文件
    logs = sorted(LOG_DIR.glob("lingzhou-*.log"), reverse=True)
    return logs[0] if logs else today


def logs_tail(
    lines: Annotated[int, typer.Option("-n", "--lines", help="显示行数")] = 30,
    follow: Annotated[bool, typer.Option("-f", "--follow", help="持续监控（Ctrl-C 退出）")] = False,
    filter_text: Annotated[Optional[str], typer.Option("-g", "--grep", help="过滤关键词")] = None,
) -> None:
    """查看最近日志（默认 30 行）。

    用法:
      lingzhou logs tail              # 最近 30 行
      lingzhou logs tail -n 100       # 最近 100 行
      lingzhou logs tail -f           # 持续监控
      lingzhou logs tail -g ERROR     # 只看含 ERROR 的行
    """
    log_file = _latest_log()
    if not log_file.exists():
        console.print(f"[yellow]日志文件不存在: {log_file}[/yellow]")
        return

    if follow:
        import subprocess
        cmd = ["tail", "-n", str(lines), "-f", str(log_file)]
        if filter_text:
            cmd = ["tail", "-n", str(lines), "-f", str(log_file)]
            subprocess.run(["stdbuf", "-oL"] + cmd, stdout=None)
            # fallback: Python 实现
            import time
            with open(log_file) as f:
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        if not filter_text or filter_text in line:
                            console.print(line, end="")
                    else:
                        time.sleep(0.5)
        else:
            import time
            with open(log_file) as f:
                f.seek(0, 2)  # 跳到文件末尾
                while True:
                    line = f.readline()
                    if line:
                        if not filter_text or filter_text in line:
                            console.print(line, end="")
                    else:
                        time.sleep(0.5)
        return

    # 非 follow 模式
    lines_data = _read_tail(log_file, lines)
    if filter_text:
        lines_data = [l for l in lines_data if filter_text in l]
    for line in lines_data:
        console.print(line, end="")


def logs_errors(
    lines: Annotated[int, typer.Option("-n", "--lines", help="最近行数中搜索")] = 500,
) -> None:
    """只看最近的错误和警告。

    用法:
      lingzhou logs errors           # 最近 500 行中的错误
      lingzhou logs errors -n 2000   # 最近 2000 行
    """
    log_file = _latest_log()
    if not log_file.exists():
        console.print(f"[yellow]日志文件不存在: {log_file}[/yellow]")
        return

    import re
    keywords = ["ERROR", "WARNING", "Traceback", "连续错误", "LLM.*失败", "crash"]
    pattern = re.compile("|".join(keywords))

    tail_lines = _read_tail(log_file, lines)
    matched = [l for l in tail_lines if pattern.search(l)]
    if not matched:
        console.print("[green]✅ 最近无错误[/green]")
        return
    console.print(f"[dim]最近 {lines} 行中找到 {len(matched)} 条异常:[/dim]")
    for line in matched:
        # 颜色标注
        if "ERROR" in line or "Traceback" in line or "连续错误" in line:
            console.print(f"[red]{line}[/red]", end="")
        elif "WARNING" in line:
            console.print(f"[yellow]{line}[/yellow]", end="")
        else:
            console.print(line, end="")


def logs_crash(
    lines: Annotated[int, typer.Option("-n", "--lines", help="显示行数")] = 50,
) -> None:
    """查看崩溃日志（stderr/traceback）。

    用法:
      lingzhou logs crash           # 最近 50 行
      lingzhou logs crash -n 200    # 最近 200 行
    """
    crash_log = LOG_DIR / "crash.log"
    if not crash_log.exists() or crash_log.stat().st_size == 0:
        console.print("[green]✅ 无崩溃记录[/green]")
        return

    tail_lines = _read_tail(crash_log, lines)
    for line in tail_lines:
        console.print(line, end="")


def logs_wechat(
    lines: Annotated[int, typer.Option("-n", "--lines", help="搜索范围行数")] = 200,
) -> None:
    """只看微信 bot 相关日志。

    用法:
      lingzhou logs wechat          # 最近微信消息
      lingzhou logs wechat -n 500   # 更多范围
    """
    log_file = _latest_log()
    if not log_file.exists():
        console.print(f"[yellow]日志文件不存在: {log_file}[/yellow]")
        return

    tail_lines = _read_tail(log_file, lines)
    wechat_lines = [l for l in tail_lines if "[wechat]" in l]
    if not wechat_lines:
        console.print("[dim]最近无微信活动[/dim]")
        return
    console.print(f"[dim]最近 {len(wechat_lines)} 条微信日志:[/dim]")
    for line in wechat_lines:
        if "chat_msg" in line:
            console.print(f"[cyan]{line}[/cyan]", end="")
        elif "→ iLink" in line or "回复成功" in line:
            console.print(f"[green]{line}[/green]", end="")
        else:
            console.print(line, end="")


def logs_stats() -> None:
    """日志统计概览。"""
    log_file = _latest_log()
    if not log_file.exists():
        console.print(f"[yellow]日志文件不存在: {log_file}[/yellow]")
        return

    text = log_file.read_text(encoding="utf-8")
    total = text.count("\n")
    boots = text.count("[boot]")
    warnings = text.count("WARNING")
    errors = text.count("ERROR") + text.count("Traceback")
    chats = text.count("[chat] user")
    wechat_msgs = text.count("chat_msg")
    wechat_sent = text.count("回复成功")
    ticks = text.count("[loop]")
    decisions_act = text.count("decision=act")
    decisions_wait = text.count("decision=wait")
    llm_fails = text.count("LLM.*失败")

    console.print(f"  总行数:    {total}")
    console.print(f"  启动次数:  {boots}")
    console.print(f"  WARNING:   {warnings}")
    console.print(f"  ERROR:     {errors}")
    console.print(f"  tick:      {ticks}  (act:{decisions_act} wait:{decisions_wait})")
    console.print(f"  chat:      {chats} 条用户消息")
    console.print(f"  微信:      {wechat_msgs} 收 / {wechat_sent} 发")
    console.print(f"  LLM失败:   {llm_fails}")
    console.print(f"  日志文件:  {log_file}  ({log_file.stat().st_size:,} bytes)")


# ── helpers ──

def _read_tail(path: Path, n: int) -> list[str]:
    """高效读取文件末尾 n 行。"""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return []

        # 从末尾反向读取，找到第 n 个换行符
        block_size = 4096
        blocks = []
        lines_found = 0
        pos = size
        while pos > 0 and lines_found <= n:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            block = f.read(read_size)
            blocks.append(block)
            lines_found += block.count(b"\n")

        data = b"".join(reversed(blocks))
        all_lines = data.decode("utf-8", errors="replace").splitlines()
        return [l + "\n" for l in all_lines[-n:]]
