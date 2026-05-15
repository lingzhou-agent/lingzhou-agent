"""provider/base.py — LLM provider 抽象接口。

新增 provider 只需实现 Provider 协议，在 provider/__init__.py 的工厂里注册即可。
上层代码不依赖任何具体实现。
"""
from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


MessageContent = str | list[dict[str, Any]]


@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: MessageContent


@runtime_checkable
class Provider(Protocol):
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        thinking_override: str | None = None,
    ) -> str:
        """发送对话消息，返回 assistant 回复文本。"""
        ...

    async def close(self) -> None:
        """释放连接资源。"""
        ...
