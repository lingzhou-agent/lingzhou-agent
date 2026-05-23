"""tools/browser.py — 浏览器自动化工具。

基于 agent-browser CLI（headless Chromium）。
无需图形界面，在 Linux 服务器上零成本运行。

工具：
  browser.navigate  — 导航到 URL
  browser.snapshot  — 获取页面可访问性树/文本快照
  browser.click     — 点击元素 (@e1, @e2 等 ref)
  browser.type      — 在输入框中输入文字
  browser.scroll    — 滚动页面
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from tools.registry import tool, ToolManifest, ToolResult, ToolParam, ToolContext

# ── 常量 ─────────────────────────────────────────────────────────────────────
BROWSER_CMD = "agent-browser"
BROWSER_ARGS = []
BROWSER_TIMEOUT = 30  # 浏览器操作超时（秒）
_NAVIGATE_NETWORK_PATTERNS = (
    "err_name_not_resolved",
    "err_internet_disconnected",
    "err_connection_refused",
    "err_connection_reset",
    "err_connection_timed_out",
    "err_address_unreachable",
    "econnrefused",
    "enetunreach",
    "dns",
    "network is unreachable",
    "could not resolve host",
)
_NAVIGATE_BLOCKED_PATTERNS = (
    "blocked by upstream",
    "access denied",
    "forbidden",
    "captcha",
    "bot verification",
    "rate limited",
    "temporarily blocked",
    "intercepted",
)
_NAVIGATE_DEPENDENCY_PATTERNS = (
    "failed to launch browser process",
    "executable doesn't exist",
    "browser binary",
    "install chromium",
    "please run the following command to download",
    "shared object file",
    "libnss3",
    "libatk",
    "libx11",
    "no usable sandbox",
)


def _find_browser() -> Optional[str]:
    """检查 agent-browser 是否可用。"""
    if shutil.which("agent-browser"):
        return "agent-browser"
    # npx 后备
    try:
        r = subprocess.run(
            ["npx", "agent-browser", "--version"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if r.stdout.strip():
            return "npx agent-browser"
    except Exception:
        pass
    return None


async def _browser_run(*args: str, timeout: int = BROWSER_TIMEOUT) -> tuple[int, str, str]:
    """异步运行 agent-browser 命令。"""
    cmd = BROWSER_ARGS + list(args)
    env = os.environ.copy()
    proc = await asyncio.create_subprocess_exec(
        BROWSER_CMD, *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        rc = proc.returncode
        if rc is None:
            rc = 0 if stdout.strip() else -1
        return rc, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
    except asyncio.TimeoutError:
        proc.kill()
        return -1, "", "操作超时"


def _make_snapshot_summary(text: str, max_lines: int = 60) -> str:
    """将快照文本压缩为摘要。"""
    lines = text.strip().splitlines()
    if len(lines) <= max_lines:
        return text.strip()
    head = "\n".join(lines[:max_lines // 2])
    tail = "\n".join(lines[-max_lines // 2:])
    return f"{head}\n...({len(lines) - max_lines} 行省略)...\n{tail}"


def _browser_failure_summary(action: str, code: int, stdout: str, stderr: str) -> str:
    detail = (stderr or stdout or "").strip()
    prefix = f"{action}失败(exit={code})"
    if not detail:
        return prefix
    return f"{prefix}: {detail[:200]}"


def _classify_navigate_failure(code: int, stdout: str, stderr: str) -> tuple[str, str]:
    detail = (stderr or stdout or "").strip()
    lowered = detail.lower()
    if "超时" in detail or "timeout" in lowered:
        return "NavigateTimeout", "超时"
    if any(marker in lowered for marker in _NAVIGATE_DEPENDENCY_PATTERNS):
        return "NavigateDependencyMissing", "浏览器依赖缺失"
    if any(marker in lowered for marker in _NAVIGATE_NETWORK_PATTERNS):
        return "NavigateNetworkUnreachable", "网络不可达"
    if any(marker in lowered for marker in _NAVIGATE_BLOCKED_PATTERNS):
        return "NavigateTargetBlocked", "目标拦截"
    return "NavigateError", "未知导航错误"


def _snapshot_looks_blank(snapshot: str) -> bool:
    stripped = snapshot.strip()
    if not stripped:
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return True
    meaningful_lines = [
        line for line in lines
        if re.search(r"[A-Za-z0-9\u4e00-\u9fff]", line)
    ]
    if not meaningful_lines:
        return True
    generic_tokens = {"document", "webarea", "rootwebarea", "main", "generic", "application"}
    normalized = [re.sub(r"[^a-z]+", "", line.lower()) for line in meaningful_lines[:4]]
    normalized = [item for item in normalized if item]
    return len(lines) <= 3 and bool(normalized) and all(item in generic_tokens for item in normalized)


# ── browser.navigate ─────────────────────────────────────────────────────────


@tool(ToolManifest(
    name="browser.navigate",
    description="在无头浏览器中打开 URL。返回页面可访问性快照。",
    progress_category="io",
    params=[
        ToolParam("url", "string", "要打开的 URL", required=True),
    ],
))
async def browser_navigate(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    url = (params.get("url") or "").strip()
    if not url:
        return ToolResult(summary="URL 不能为空", error="EmptyUrl", skipped=True)
    if not url.startswith("http"):
        url = "https://" + url

    browser = _find_browser()
    if not browser:
        return ToolResult(
            summary="agent-browser 未安装。运行: npm install -g agent-browser && agent-browser install",
            error="BrowserNotInstalled", skipped=True,
        )

    try:
        code, stdout, stderr = await _browser_run("navigate", url, "--snapshot")
        if code != 0:
            detail = (stderr or stdout or "").strip()
            err, label = _classify_navigate_failure(code, stdout, stderr)
            return ToolResult(
                summary=f"导航失败[{label}](exit={code}): {(detail or '无详细输出')[:200]}",
                error=err,
                evidence=detail[:500],
                metadata={"url": url, "exit_code": code, "failure_kind": label},
            )
        if _snapshot_looks_blank(stdout):
            return ToolResult(
                summary=f"导航失败[页面空白](exit={code}): 页面已打开，但快照为空或只有空白骨架",
                error="NavigateBlankPage",
                evidence=stdout[:500],
                metadata={"url": url, "exit_code": code, "failure_kind": "页面空白"},
            )
        return ToolResult(
            summary=f"已打开: {url}\n{_make_snapshot_summary(stdout)}",
            resource_key=url,
            evidence=stdout[:500],
            metadata={"url": url, "snapshot_chars": len(stdout)},
            state_delta={"page": url},
        )
    except Exception as e:
        return ToolResult(summary=f"导航异常: {e}", error="BrowserError")


# ── browser.snapshot ─────────────────────────────────────────────────────────


@tool(ToolManifest(
    name="browser.snapshot",
    description="获取当前页面的文本快照（可访问性树）。显示页面结构和可交互元素。",
    progress_category="info",
    params=[],
))
async def browser_snapshot(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    try:
        code, stdout, stderr = await _browser_run("snapshot")
        if code != 0:
            return ToolResult(summary=f"快照失败: {stderr[:200]}", error="SnapshotError")
        return ToolResult(
            summary=_make_snapshot_summary(stdout),
            evidence=stdout[:500],
            metadata={"snapshot_chars": len(stdout)},
        )
    except Exception as e:
        return ToolResult(summary=f"快照异常: {e}", error="BrowserError")


# ── browser.click ────────────────────────────────────────────────────────────


@tool(ToolManifest(
    name="browser.click",
    description="点击页面元素。使用快照中的 ref（如 @e5）定位元素。",
    progress_category="mutation",
    params=[
        ToolParam("ref", "string", "元素引用，如 @e5、@e12", required=True),
    ],
))
async def browser_click(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    ref = (params.get("ref") or "").strip()
    if not ref:
        return ToolResult(summary="ref 不能为空", error="EmptyRef", skipped=True)

    try:
        code, stdout, stderr = await _browser_run("click", ref)
        if code != 0:
            return ToolResult(summary=f"点击失败: {stderr[:200]}", error="ClickError")
        return ToolResult(
            summary=f"已点击 {ref}\n{_make_snapshot_summary(stdout)}",
            evidence=stdout[:300],
            metadata={"ref": ref},
            state_delta={"clicked": ref},
        )
    except Exception as e:
        return ToolResult(summary=f"点击异常: {e}", error="BrowserError")


# ── browser.type ─────────────────────────────────────────────────────────────


@tool(ToolManifest(
    name="browser.type",
    description="在当前焦点元素中输入文字。",
    progress_category="mutation",
    params=[
        ToolParam("text", "string", "要输入的文字", required=True),
        ToolParam("ref", "string", "目标输入框 ref（可选，自动使用焦点元素）", required=False),
    ],
))
async def browser_type(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    text = (params.get("text") or "")
    ref = (params.get("ref") or "").strip()

    try:
        args = ["type"]
        if ref:
            args.extend(["--ref", ref])
        args.append(text)
        code, stdout, stderr = await _browser_run(*args)
        if code != 0:
            return ToolResult(summary=f"输入失败: {stderr[:200]}", error="TypeError")
        return ToolResult(
            summary=f"已输入: {text[:50]}",
            metadata={"text": text[:100], "ref": ref or "focus"},
        )
    except Exception as e:
        return ToolResult(summary=f"输入异常: {e}", error="BrowserError")


# ── browser.scroll ───────────────────────────────────────────────────────────


@tool(ToolManifest(
    name="browser.scroll",
    description="滚动页面。",
    progress_category="io",
    params=[
        ToolParam("direction", "string", "滚动方向: up / down", required=False),
        ToolParam("amount", "number", "滚动像素数（默认 500）", required=False),
    ],
))
async def browser_scroll(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    direction = (params.get("direction") or "down").strip()
    try:
        amount = int(params.get("amount", 500))
    except (ValueError, TypeError):
        amount = 500

    try:
        args = ["scroll"]
        if direction == "up":
            args.extend(["up", str(amount)])
        else:
            args.extend(["down", str(amount)])
        code, stdout, stderr = await _browser_run(*args)
        if code != 0:
            return ToolResult(summary=f"滚动失败: {stderr[:200]}", error="ScrollError")
        return ToolResult(
            summary=f"已滚动 {direction} {amount}px\n{_make_snapshot_summary(stdout, 30)}",
            evidence=stdout[:200],
        )
    except Exception as e:
        return ToolResult(summary=f"滚动异常: {e}", error="BrowserError")
