"""core/probe/types.py — 探针系统核心数据类型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# 探针执行方式
ProbeKind = Literal["shell", "http", "python"]

# 数据回传路径（LLM 自行决定如何处置 probe.run 的返回值；interval 探针后台推送到 wm）
ProbeDataBack = Literal["none", "wm"]


@dataclass
class ProbeConfig:
    """一个已安装探针的完整配置。

    Attributes
    ----------
    name:
        探针唯一名称（人类可读）。
    kind:
        执行方式：
        - "shell"  — 执行 shell 命令，stdout 作为结果
        - "http"   — GET 请求指定 URL，响应体作为结果
        - "python" — 执行 Python 代码片段，stdout 作为结果
    spec:
        对应 kind 的内容：命令字符串 / URL / Python 代码。
    trigger:
        调度方式：
        - "interval:<seconds>" — 每隔 N 秒执行一次，如 "interval:60"
        - "manual"             — 仅手动触发（probe.run 工具）
    data_back:
        结果回传路径。
    alert_expr:
        Python 布尔表达式，变量 ``output`` 为结果字符串。
        表达式为 True 时触发告警，如 ``float(output.strip()) > 35.0``
    alert_message:
        告警时发出的人类可读消息。支持 ``{output}`` 占位符。
    chat_id:
        data_back="chat" 时发往哪个会话。为空时使用最近活跃会话。
    enabled:
        False 时探针被暂停，不自动执行。
    """

    name: str
    kind: ProbeKind
    spec: str
    trigger: str
    data_back: ProbeDataBack = "wm"
    alert_expr: str | None = None
    alert_message: str | None = None
    enabled: bool = True

    # 以下字段由 store 填充，不由用户设置
    id: int = 0
    created_at: str = ""
    last_run_at: str | None = None
    last_result: str | None = None
    last_error: str | None = None


@dataclass
class ProbeResult:
    """单次探针执行结果。"""

    probe_name: str
    output: str
    error: str | None
    triggered_at: str
    duration_ms: int
    alerted: bool = False
    alert_detail: str | None = None
