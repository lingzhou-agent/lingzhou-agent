"""cli/chat.py — 持续对话窗口命令。

用法：
  lingzhou chat              # 进入交互式对话（自动感知 loop 回复）
  lingzhou chat -a "你好"    # 一次性发送消息，等待回复后退出
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Annotated, Optional  # noqa: F401

import typer
from rich.panel import Panel

from cli._common import console, load_cfg

# 等待回复最长秒数（-a 模式）
_DEFAULT_TIMEOUT = 300


def chat(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("lingzhou.json"),
    ask: Annotated[
        Optional[str],
        typer.Option("--ask", "-a", help="发送一条消息并等待回复，收到回复后退出"),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option("--timeout", "-t", help="等待回复的最长秒数（-a 模式有效）"),
    ] = _DEFAULT_TIMEOUT,
    session: Annotated[
        str,
        typer.Option("--session", "-s", help="会话 ID（留空则使用全局频道）"),
    ] = "",
) -> None:
    """持续对话窗口：发送消息、实时显示 loop 回复。

    交互模式（不带 -a）：
      - 显示最近对话历史
      - 输入消息 → 注入 loop 的下一个 tick → 打印回复
      - Ctrl-C 退出

    一次性模式（-a "消息"）：
      - 发送消息，等待 loop 回复（最长 --timeout 秒），打印后退出
    """
    asyncio.run(_main(config, ask, timeout, session))


async def _main(
    config: Path,
    ask: Optional[str],
    timeout: int,
    session_id: str,
) -> None:
    from memory.task_store import TaskStore

    cfg = load_cfg(config)
    store = TaskStore(cfg.db_path)
    await store.open()

    try:
        # 读取 soul 名称（显示用）
        name_val = "灵舟"
        try:
            v, found = await store.get_fact("soul:name")
            if found and v:
                name_val = v
        except Exception:
            pass

        if ask is not None:
            await _ask_once(store, ask, timeout, session_id, name_val)
        else:
            # 交互模式：从配置读 chat_reply_timeout
            interactive_timeout = cfg.loop.chat_reply_timeout
            await _interactive(store, session_id, name_val, interactive_timeout)
    finally:
        await store.close()


async def _ask_once(
    store: "TaskStore",
    text: str,
    timeout: int,
    session_id: str,
    agent_name: str,
) -> None:
    """一次性模式：发送 → 轮询回复 → 打印 → 退出。"""
    msg_id = await store.add_chat_message("user", text, session_id)
    console.print(f"[dim][你][/dim] {text}")
    console.print(f"[dim]等待 {agent_name} 回复（最长 {timeout}s）…[/dim]")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        msgs = await store.get_chat_messages_since(msg_id, session_id)
        for m in msgs:
            if m["role"] == "assistant":
                console.print(f"[bold green][{agent_name}][/bold green] {m['content']}")
                return

    console.print(
        f"[yellow]等待超时（{timeout}s）。请确认 loop 正在后台运行：[bold]lingzhou run -d[/bold][/yellow]"
    )


async def _interactive(
    store: "TaskStore",
    session_id: str,
    agent_name: str,
    reply_timeout: int = 300,
) -> None:
    """交互式对话 REPL：持续接收用户输入，显示 loop 回复。"""
    # 显示最近历史（最多 10 条）
    history = await store.get_chat_messages_since(0, session_id)
    last_id = 0
    if history:
        recent = history[-10:]
        last_id = history[-1]["id"]
        console.print("[dim]─── 近期对话记录 ───[/dim]")
        for m in recent:
            if m["role"] == "user":
                console.print(f"[你] {m['content']}")
            else:
                console.print(f"[bold green][{agent_name}][/bold green] {m['content']}")
        console.print("[dim]─────────────────────[/dim]")

    console.print(Panel(
        f"已连接到 [bold green]{agent_name}[/bold green]\n"
        "[dim]输入消息后按 Enter 发送。Ctrl-C 退出。[/dim]",
        title="💬 Chat",
        border_style="green",
    ))

    loop = asyncio.get_event_loop()

    try:
        while True:
            # 阻塞等待用户输入（executor 线程里打印提示符 + readline）
            line = await loop.run_in_executor(None, _read_line)
            if not line:  # EOF / Ctrl-D
                break

            user_text = line.strip()
            if not user_text:
                continue

            await store.add_chat_message("user", user_text, session_id)

            # 等待回复（最长 reply_timeout 秒，默认来自 cfg.loop.chat_reply_timeout）
            wait_start = time.monotonic()
            replied = False

            while time.monotonic() - wait_start < reply_timeout:
                await asyncio.sleep(0.5)
                new_msgs = await store.get_chat_messages_since(last_id, session_id)
                for m in new_msgs:
                    last_id = m["id"]
                    if m["role"] == "assistant":
                        console.print(f"[bold green][{agent_name}][/bold green] {m['content']}")
                        replied = True
                if replied:
                    break

            if not replied:
                console.print(
                    f"[yellow](等待超时 {reply_timeout}s。请确认 loop 正在运行：[bold]lingzhou run -d[/bold])[/yellow]"
                )

    except KeyboardInterrupt:
        pass

    console.print("\n[dim]再见。[/dim]")


def _read_line() -> str:
    """在 executor 线程中打印提示符并读取一行输入。"""
    import sys
    try:
        sys.stdout.write("[你] ")
        sys.stdout.flush()
        return sys.stdin.readline()
    except (EOFError, KeyboardInterrupt, OSError):
        return ""



