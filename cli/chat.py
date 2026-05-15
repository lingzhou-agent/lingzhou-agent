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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.task_store import TaskStore

from cli._common import console, load_cfg, DEFAULT_CONFIG_PATH

# 等待回复最长秒数（-a 模式）
_DEFAULT_TIMEOUT = 300
_CHAT_INPUT_PROMPT = "you> "


def chat(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
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
        await asyncio.sleep(0.1)
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
    reply_timeout: int = 300,  # noqa: ARG001 — 保留签名兼容，异步模式不再阻塞等待
) -> None:
    """交互式对话 REPL：发送消息不阻塞，回复异步展示。

    设计：
      - _input_task  读取用户输入，写入 DB，立即返回等待下一条
      - _reply_task  持续轮询 DB，新回复到达时随时打印
      两个 task 并发运行，互不阻塞。LLM 慢/宕机时用户可继续输入。
    """
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
        "[dim]输入消息后按 Enter 发送（无需等待回复）。Ctrl-C 退出。[/dim]",
        title="💬 Chat",
        border_style="green",
    ))

    ev_loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    async def _input_task() -> None:
        """读取用户输入，写入 DB，立即允许下一条输入。"""
        try:
            while not stop.is_set():
                line = await ev_loop.run_in_executor(None, _read_line)
                if not line:  # EOF / Ctrl-D
                    stop.set()
                    break
                user_text = line.strip()
                if user_text:
                    await store.add_chat_message("user", user_text, session_id)
                    # 告知用户消息已入队，loop 在后台处理（异步模式核心体验）
                    console.print("[dim]  ↑ 已发送，等待回复中…（可继续输入下一条）[/dim]")
        except (KeyboardInterrupt, asyncio.CancelledError):
            stop.set()

    async def _reply_task(cur_last_id: int) -> None:
        """持续轮询 DB，新 assistant 回复到达时立即打印。"""
        try:
            while not stop.is_set():
                await asyncio.sleep(0.1)
                new_msgs = await store.get_chat_messages_since(cur_last_id, session_id)
                for m in new_msgs:
                    cur_last_id = m["id"]
                    if m["role"] == "assistant":
                        # \n 前缀避免打断用户正在输入的行
                        console.print(f"\n[bold green][{agent_name}][/bold green] {m['content']}\n")
                        # 回复打印后重补 [你] 提示符，使用户知道可以继续输入
                        import sys as _sys
                        _sys.stdout.write(_CHAT_INPUT_PROMPT)
                        _sys.stdout.flush()
        except asyncio.CancelledError:
            pass

    t_input = asyncio.create_task(_input_task())
    t_reply = asyncio.create_task(_reply_task(last_id))

    try:
        # 等待用户主动退出（EOF / Ctrl-D）；Ctrl-C 由外层 KeyboardInterrupt 捕获
        await t_input
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop.set()
        t_reply.cancel()
        await asyncio.gather(t_input, t_reply, return_exceptions=True)

    console.print("\n[dim]再见。[/dim]")


def _read_line() -> str:
    """在 executor 线程中打印提示符并读取一行输入。

    服务器环境 locale 可能不是 UTF-8（如 POSIX/C locale），
    直接读 buffer 字节后手动 decode(errors='replace') 避免 UnicodeDecodeError。
    """
    import sys
    try:
        return input(_CHAT_INPUT_PROMPT)
    except UnicodeDecodeError:
        try:
            sys.stdout.write(_CHAT_INPUT_PROMPT)
            sys.stdout.flush()
            raw = sys.stdin.buffer.readline()
            return raw.decode("utf-8", errors="replace")
        except (EOFError, KeyboardInterrupt, OSError):
            return ""
    except (EOFError, KeyboardInterrupt, OSError):
        return ""



