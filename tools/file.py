"""tools/file.py — 文件读写工具。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool

_log = logging.getLogger("lingzhou.tools.file")


@tool(ToolManifest(
    name="file.read",
    description="读取文件内容，支持按下标区间读取。不指定任何参数时读取全部内容。",
    params=[
        ToolParam("path", "string", "文件路径", required=True),
        ToolParam("start", "number", "起始下标（含），默认 0", required=False),
        ToolParam("end", "number", "结束下标（不含），默认到文件末尾", required=False),
        ToolParam("max_chars", "number", "最大字符数；不传则读取全部内容", required=False),
    ],
))
async def file_read(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = Path(params.get("path") or "").expanduser()
    max_chars_raw = params.get("max_chars")  # None = 不限制，读全部
    max_chars: int | None = int(max_chars_raw) if max_chars_raw is not None else None
    has_range = ("start" in params) or ("end" in params)
    if not path.exists():
        return ToolResult(summary=f"文件不存在: {path}", error="FileNotFound")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        total = len(text)

        if has_range:
            start = int(params.get("start") or 0)
            end_raw = params.get("end")
            end = int(end_raw) if end_raw is not None else total

            # 区间归一化：允许越界输入，最终收敛到 [0, total]
            start = max(0, start)
            end = max(0, end)
            start = min(start, total)
            end = min(end, total)
            if end < start:
                end = start

            sliced = text[start:end]
            completed = (start == 0 and end == total)
            _cont_hint = f"\n⚠️ 未读完，使用 start={end} 续读剩余 {total - end} 字符" if not completed else ""
            return ToolResult(
                summary=(
                    f"[已读取 {path}  区间[{start}:{end})/{total}  completed={str(completed).lower()}]"
                    f"{_cont_hint}\n{sliced}"
                ),
                evidence=(
                    f"path={path} mode=range range={start}:{end} chars={len(sliced)}/{total} "
                    f"completed={str(completed).lower()}"
                ),
                priority=0.6,
            )

        if max_chars is not None:
            content = text[:max_chars]
            completed = (total <= max_chars)
            mode_label = f"max_chars={max_chars}"
        else:
            content = text
            completed = True
            mode_label = "full"
        _cont_hint = f"\n⚠️ 未读完，使用 start={len(content)} 续读剩余 {total - len(content)} 字符" if not completed else ""
        return ToolResult(
            summary=(
                f"[已读取 {path}  {total}字符  {mode_label}  "
                f"range=[0:{len(content)})  completed={str(completed).lower()}]"
                f"{_cont_hint}\n{content}"
            ),
            evidence=(
                f"path={path} mode={mode_label} range=0:{len(content)} "
                f"chars={len(content)}/{total} completed={str(completed).lower()}"
            ),
            priority=0.6,   # 文件内容不需长期捤占 WM
        )
    except Exception as exc:
        _log.error("[file.read] failed on %s: %s", path, exc, exc_info=True)
        return ToolResult(summary=f"读取失败: {exc}", error=str(exc))


@tool(ToolManifest(
    name="file.write",
    description="将文本写入文件（覆盖）",
    params=[
        ToolParam("path", "string", "文件路径", required=True),
        ToolParam("content", "string", "写入内容", required=True),
    ],
))
async def file_write(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if ctx.dry_run:
        return ToolResult(
            summary=f"[dry-run] file.write: {params.get('path')}",
            skipped=True,
        )
    path = Path(params.get("path") or "").expanduser()
    content = params.get("content") or ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(
            summary=f"已写入 {path}（{len(content)} 字符）",
            evidence=f"path={path} chars={len(content)}",
            priority=0.5,   # 写入结果是过渡信息
        )
    except Exception as exc:
        return ToolResult(summary=f"写入失败: {exc}", error=str(exc))


@tool(ToolManifest(
    name="file.list",
    description="列出目录内容",
    params=[
        ToolParam("path", "string", "目录路径", required=True),
        ToolParam("pattern", "string", "glob 模式，默认 *", required=False),
    ],
))
async def file_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw_path = params.get("path") or "."
    path = Path(raw_path).expanduser()

    if not path.exists():
        alt_path = Path.cwd() / raw_path
        if alt_path.exists():
            path = alt_path
        else:
            # ENOENT 写入语义记忆：WM 被清空后该证据仍然存在，阻断错误学习循环
            import hashlib as _hash
            from memory.semantic import MemoryNode as _MN
            _pid = f"enoent_{_hash.md5(raw_path.encode()).hexdigest()[:8]}"
            ctx.semantic.upsert(_MN(
                id=_pid,
                kind="path_not_exist",
                title=f"路径不存在: {raw_path}",
                body=(
                    f"已通过 file.list 确认：`{raw_path}` 在文件系统中不存在。"
                    "不要再尝试此路径。如果之前的记忆说它存在，那是幻觉或旧记忆错误。"
                ),
                activation=0.95,
                valence=0.0,
                tags=["enoent", "path_not_exist"],
            ))
            return ToolResult(
                summary=f"[ENOENT] 路径不存在: {raw_path}  ——已记入长期记忆，禁止再次尝试",
                error="PathNotFound",
                priority=0.3,
            )

    pattern = params.get("pattern") or "*"

    if not path.is_dir():
        return ToolResult(
            summary=f"[NOT_DIR] 路径不是目录: {path}  ——无法列出，请检查路径是否为文件",
            error="NotADirectory",
            priority=0.3,
        )

    try:
        items = sorted(path.glob(pattern))
        lines = [str(p.relative_to(path)) + ("/" if p.is_dir() else "") for p in items[:100]]
        listing = "\n".join(lines) if lines else "（空目录）"
        # summary 以路径+数量开头，让 LLM 在 WM 中区分「已列过」就0资料
        return ToolResult(
            summary=f"[已列导 {path}  共 {len(items)} 项]\n{listing}",
            evidence=f"path={path} count={len(items)}",
            priority=0.5,   # 目录列表是过渡信息，不应长期占据 WM
        )
    except Exception as exc:
        return ToolResult(summary=f"列出目录失败: {exc}", error=str(exc))