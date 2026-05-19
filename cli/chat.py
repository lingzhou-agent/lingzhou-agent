"""cli/chat.py — 持续对话窗口命令。

用法：
  lingzhou chat              # 进入交互式对话（自动感知 loop 回复）
  lingzhou chat -a "你好"    # 一次性发送消息，等待回复后退出
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Annotated, Any, Optional  # noqa: F401

import typer
from rich.panel import Panel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.task_store import TaskStore

from cli._common import console, load_cfg, DEFAULT_CONFIG_PATH
from store.memory import sanitize_chat_content

# 等待回复最长秒数（-a 模式）
_DEFAULT_TIMEOUT = 300
_CHAT_INPUT_PROMPT = "you> "


def _erase_last_input_echo() -> None:
    """尽量擦除用户刚提交的输入行，避免 chat 界面残留一行回显。"""
    import sys

    stdout = sys.stdout
    if not hasattr(stdout, "isatty") or not stdout.isatty():
        return
    try:
        stdout.write("\x1b[1A\r\x1b[2K\r")
        stdout.flush()
    except OSError:
        pass


def _normalize_user_title(raw: str) -> str:
    title = str(raw or "").strip().strip("[]()<>\"'`，,。.!！?？:：")
    if not title or len(title) > 12 or any(ch.isspace() for ch in title):
        return ""
    blocked = {
        "收到", "好的", "明白", "了解", "现在", "当前", "状态", "结果", "下一步", "进展", "回复",
        "assistant", "user", "chat", "you",
    }
    return "" if title.lower() in blocked or title in blocked else title


def _infer_user_title_from_messages(messages: list[dict[str, object]]) -> str:
    """从当前 chat 会话历史中推断用户称谓。"""
    user_patterns = (
        re.compile(r"(?:你可以)?叫我([^\s，,。.!！?？:：]{1,12})"),
        re.compile(r"我是([^\s，,。.!！?？:：]{1,12})"),
        re.compile(r"我的称呼是([^\s，,。.!！?？:：]{1,12})"),
    )
    assistant_pattern = re.compile(r"^([^\s，,：:]{1,12})[，,:：]")

    for msg in reversed(messages[-50:]):
        role = str(msg.get("role") or "")
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            for pattern in user_patterns:
                match = pattern.search(content)
                if match:
                    title = _normalize_user_title(match.group(1))
                    if title:
                        return title
        elif role == "assistant":
            match = assistant_pattern.match(content)
            if match:
                title = _normalize_user_title(match.group(1))
                if title:
                    return title
    return ""


def _chat_input_prompt(user_title: str = "", chat_id: str = "") -> str:
    """优先使用会话中已识别的用户称谓；未知时退回 chat id。"""
    label = str(user_title or "").strip() or str(chat_id or "").strip() or "chat"
    return f"{label}> "


def _print_input_prompt(prompt: str) -> None:
    """在 TTY 上重绘输入提示符，减少异步回复后光标悬空。"""
    import sys

    stdout = sys.stdout
    if not hasattr(stdout, "isatty") or not stdout.isatty():
        return
    try:
        stdout.write(prompt)
        stdout.flush()
    except OSError:
        pass


def _history_excerpt(messages: list[dict[str, object]], limit: int = 12) -> str:
    lines: list[str] = []
    for msg in messages[-limit:]:
        role = str(msg.get("role") or "").strip() or "unknown"
        content = " ".join(str(msg.get("content") or "").split()).strip()
        if not content:
            continue
        if len(content) > 160:
            content = content[:157] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_user_title_from_llm_output(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
        except Exception:
            data = None
        if isinstance(data, dict):
            for key in ("user_title", "title", "appellation"):
                if key in data:
                    return _normalize_user_title(str(data.get(key) or ""))
    head = text.splitlines()[0].strip()
    if head.upper() in {"NONE", "NULL", "UNKNOWN", "UNSURE", "不确定", "未知", "无"}:
        return ""
    return _normalize_user_title(head)


async def _infer_user_title_with_llm(provider: Any, messages: list[dict[str, object]]) -> str:
    excerpt = _history_excerpt(messages)
    if not excerpt:
        return ""

    from provider.base import Message

    raw = await provider.chat(
        [
            Message(
                role="system",
                content=(
                    "你是一个只负责识别聊天中‘assistant 对 user 的当前称谓’的提取器。"
                    "只输出一个称谓本身；若无法确定，输出 NONE。"
                    "不要解释，不要输出多余文字。"
                ),
            ),
            Message(
                role="user",
                content=(
                    "从下面对话中识别 assistant 当前对 user 的称谓。"
                    "优先使用 assistant 已经采用的称呼；若 user 明确说‘叫我 X’，也可用该称呼。"
                    "若仍不明确，输出 NONE。\n\n"
                    f"{excerpt}"
                ),
            ),
        ],
        temperature=0,
        thinking_override="off",
    )
    return _parse_user_title_from_llm_output(raw)


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
    chat_id: str,
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
            await _ask_once(store, ask, timeout, chat_id, name_val)
        else:
            # 交互模式：从配置读 chat_reply_timeout
            interactive_timeout = cfg.loop.chat_reply_timeout
            await _interactive(store, cfg, chat_id, name_val, interactive_timeout)
    finally:
        await store.close()


async def _ask_once(
    store: "TaskStore",
    text: str,
    timeout: int,
    chat_id: str,
    agent_name: str,
) -> None:
    """一次性模式：发送 → 轮询回复 → 打印 → 退出。"""
    msg_id = await store.add_chat_message("user", text, chat_id=chat_id)
    console.print(f"[dim][你][/dim] {text}")
    console.print(f"[dim]等待 {agent_name} 回复（最长 {timeout}s）…[/dim]")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(0.1)
        msgs = await store.get_chat_messages_since(msg_id, chat_id=chat_id)
        for m in msgs:
            if m["role"] == "assistant":
                console.print(f"[bold green][{agent_name}][/bold green] {m['content']}")
                return

    console.print(
        f"[yellow]等待超时（{timeout}s）。请确认 loop 正在后台运行：[bold]lingzhou run -d[/bold][/yellow]"
    )


async def _interactive(
    store: "TaskStore",
    cfg,
    chat_id: str,
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
    history = await store.get_chat_messages_since(0, chat_id=chat_id)
    last_id = 0
    prompt_state = {"value": _chat_input_prompt(_infer_user_title_from_messages(history), chat_id)}
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
    title_task: asyncio.Task | None = None
    title_provider = None

    async def _refresh_prompt_from_history(*, redraw: bool = False) -> None:
        inferred = _infer_user_title_from_messages(history)
        if title_provider is not None:
            try:
                llm_title = await _infer_user_title_with_llm(title_provider, history)
                if llm_title:
                    inferred = llm_title
            except Exception:
                pass
        prompt_state["value"] = _chat_input_prompt(inferred, chat_id)
        if redraw:
            _print_input_prompt(prompt_state["value"])

    def _schedule_prompt_refresh(*, redraw: bool = False) -> None:
        nonlocal title_task
        if title_task is not None and not title_task.done():
            title_task.cancel()
        title_task = asyncio.create_task(_refresh_prompt_from_history(redraw=redraw))

    try:
        from provider import create_provider

        title_provider = create_provider(cfg)
    except Exception:
        title_provider = None

    if history:
        _schedule_prompt_refresh(redraw=False)

    async def _input_task() -> None:
        """读取用户输入，写入 DB，立即允许下一条输入。"""
        try:
            while not stop.is_set():
                line = await ev_loop.run_in_executor(None, _read_line, prompt_state["value"])
                if not line:  # EOF / Ctrl-D
                    stop.set()
                    break
                user_text = line.strip()
                if user_text:
                    await store.add_chat_message("user", user_text, chat_id=chat_id)
                    history.append({"role": "user", "content": user_text})
                    prompt_state["value"] = _chat_input_prompt(_infer_user_title_from_messages(history), chat_id)
                    _schedule_prompt_refresh(redraw=False)
                    _erase_last_input_echo()
                    # 告知用户消息已入队，loop 在后台处理（异步模式核心体验）
                    console.print("[dim]  ↑ 已发送，等待回复中…（可继续输入下一条）[/dim]")
        except (KeyboardInterrupt, asyncio.CancelledError):
            stop.set()

    async def _reply_task(cur_last_id: int) -> None:
        """持续轮询 DB，新 assistant 回复到达时立即打印。"""
        try:
            while not stop.is_set():
                await asyncio.sleep(0.1)
                new_msgs = await store.get_chat_messages_since(cur_last_id, chat_id=chat_id)
                for m in new_msgs:
                    cur_last_id = m["id"]
                    history.append({"role": str(m.get("role") or ""), "content": str(m.get("content") or "")})
                    prompt_state["value"] = _chat_input_prompt(_infer_user_title_from_messages(history), chat_id)
                    if m["role"] == "assistant":
                        # \n 前缀避免把回复直接挤到用户当前输入后面。
                        console.print(f"\n[bold green][{agent_name}][/bold green] {m['content']}\n")
                    _schedule_prompt_refresh(redraw=(m["role"] == "assistant"))
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
        if title_task is not None and not title_task.done():
            title_task.cancel()
        tasks = [t_input, t_reply]
        if title_task is not None:
            tasks.append(title_task)
        await asyncio.gather(*tasks, return_exceptions=True)
        if title_provider is not None:
            await title_provider.close()

    console.print("\n[dim]再见。[/dim]")


def _read_line(prompt: str = _CHAT_INPUT_PROMPT) -> str:
    """在 executor 线程中打印提示符并读取一行输入。

    服务器环境 locale 可能不是 UTF-8（如 POSIX/C locale），
    直接读 buffer 字节后手动 decode(errors='replace') 避免 UnicodeDecodeError。
    """
    import sys
    try:
        return sanitize_chat_content(input(prompt))
    except UnicodeDecodeError:
        try:
            sys.stdout.write(prompt)
            sys.stdout.flush()
            raw = sys.stdin.buffer.readline()
            return sanitize_chat_content(raw.decode("utf-8", errors="replace"))
        except (EOFError, KeyboardInterrupt, OSError):
            return ""
    except (EOFError, KeyboardInterrupt, OSError):
        return ""



