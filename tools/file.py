"""tools/file.py — 文件读写和编辑工具。"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool

_log = logging.getLogger("lingzhou.tools.file")


def _tail_after_anchor(path: Path, anchor: str) -> Path | None:
    parts = path.parts
    if anchor not in parts:
        return None
    idx = len(parts) - 1 - parts[::-1].index(anchor)
    tail = parts[idx + 1 :]
    if not tail:
        return None
    return Path(*tail)


def _resolve_read_path(path: Path) -> Path:
    if path.exists():
        return path

    cwd = Path.cwd()
    home = Path.home()

    bases: list[Path] = [cwd, *cwd.parents, home, home / ".openclaw"]
    rels: list[Path] = []

    if not path.is_absolute():
        rels.append(path)

    for anchor in ("workspace", "lingzhou", ".openclaw"):
        rel = _tail_after_anchor(path, anchor)
        if rel is not None:
            rels.append(rel)
            if rel.parts and rel.parts[0] == "workspace" and len(rel.parts) > 1:
                rels.append(Path(*rel.parts[1:]))

    if path.name:
        rels.append(Path(path.name))

    seen_rel: set[str] = set()
    uniq_rels: list[Path] = []
    for rel in rels:
        key = str(rel)
        if key not in seen_rel and key not in ("", "."):
            seen_rel.add(key)
            uniq_rels.append(rel)

    seen_candidates: set[str] = set()
    for rel in uniq_rels:
        for base in bases:
            candidates = [base / rel]
            if rel.parts and rel.parts[0] != "workspace":
                candidates.append(base / "workspace" / rel)
            for candidate in candidates:
                key = str(candidate)
                if key in seen_candidates:
                    continue
                seen_candidates.add(key)
                if candidate.exists():
                    return candidate

    return path


@tool(ToolManifest(
    name="file.list",
    description="列出目录内容。支持 shallow list，用于替代 shell.run 的 ls/find 场景。",
    params=[
        ToolParam("path", "string", "目录路径", required=True),
        ToolParam("limit", "number", "最多返回多少项，默认 200", required=False),
        ToolParam("include_hidden", "boolean", "是否包含隐藏文件，默认 false", required=False),
    ],
))
async def file_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = _resolve_read_path(Path(params.get("path") or "").expanduser())
    limit = int(params.get("limit") or 200)
    include_hidden = bool(params.get("include_hidden", False))

    if not path.exists():
        return ToolResult(summary=f"路径不存在: {path}", error="FileNotFound")
    if not path.is_dir():
        return ToolResult(summary=f"不是目录: {path}", error="NotADirectory")

    try:
        entries = []
        for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
            if not include_hidden and child.name.startswith('.'):
                continue
            suffix = "/" if child.is_dir() else ""
            entries.append(child.name + suffix)
        clipped = entries[:max(0, limit)]
        remaining = max(0, len(entries) - len(clipped))
        body = "\n".join(clipped)
        if remaining:
            body += f"\n... (+{remaining} more)"
        payload = {"path": str(path), "count": len(entries), "returned": len(clipped)}
        return ToolResult(
            summary=body or "（空目录）",
            evidence=json.dumps(payload, ensure_ascii=False),
            resource_key=str(path),
            fingerprint=f"list:{hashlib.md5((body or '（空目录）').encode()).hexdigest()[:12]}",
            artifact_paths=[str(path)],
            metadata=payload,
        )
    except Exception as e:
        _log.exception("列出目录失败: %s", path)
        return ToolResult(summary=f"列出失败: {path}", error=type(e).__name__)


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
    raw_path = str(params.get("path") or "").strip()
    if not raw_path:
        return ToolResult(summary="path 不能为空", error="EmptyPath", skipped=True)

    path = _resolve_read_path(Path(raw_path).expanduser())
    max_chars_raw = params.get("max_chars")
    max_chars: int | None = int(max_chars_raw) if max_chars_raw is not None else None
    has_range = ("start" in params) or ("end" in params)

    if not path.exists():
        return ToolResult(summary=f"文件不存在: {path}", error="FileNotFound")
    if not path.is_file():
        return ToolResult(summary=f"不是文件: {path}", error="NotAFile", skipped=True)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        total = len(text)

        if has_range:
            start = int(params.get("start") or 0)
            end_raw = params.get("end")
            end = int(end_raw) if end_raw is not None else total
            text = text[start:end]

        if max_chars is not None:
            text = text[:max(0, max_chars)]

        return ToolResult(
            summary=text,
            resource_key=str(path),
            fingerprint=f"read:{hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()[:12]}",
            artifact_paths=[str(path)],
            metadata={"path": str(path), "chars": len(text), "has_range": has_range, "max_chars": max_chars},
        )
    except Exception as e:
        _log.exception("读取文件失败: %s", path)
        return ToolResult(summary=f"读取失败: {path}", error=type(e).__name__)


@tool(ToolManifest(
    name="file.write",
    description="写入文件内容。如果文件已存在则覆盖全部内容。创建新文件时自动创建父目录。",
    params=[
        ToolParam("path", "string", "文件路径", required=True),
        ToolParam("content", "string", "要写入的内容", required=True),
    ],
))
async def file_write(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = Path(params.get("path") or "").expanduser()
    content = params.get("content")

    if content is None:
        return ToolResult(summary="写入内容为空", error="EmptyContent", skipped=True)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = str(content)
        path.write_text(text, encoding="utf-8")
        return ToolResult(
            summary=f"写入成功: {path} ({len(text)} 字符)",
            resource_key=str(path),
            fingerprint=f"write:{hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()[:12]}",
            artifact_paths=[str(path)],
            state_delta={"file": "written", "chars": len(text)},
            metadata={"path": str(path), "chars": len(text)},
        )
    except Exception as e:
        _log.exception("写入文件失败: %s", path)
        return ToolResult(summary=f"写入失败: {path}", error=type(e).__name__)


@tool(ToolManifest(
    name="file.edit",
    description=(
        "对文件进行精确文本替换。支持单处或多处替换（edit 列表）。"
        "每个 edit 包含 oldText（原文本）和 newText（新文本），oldText 必须在文件中唯一匹配。"
        "这是修改文件的首选工具——相比全量覆盖的 file.write，edit 只改需要改的部分，安全且节省 token。"
    ),
    params=[
        ToolParam("path", "string", "文件路径", required=True),
        ToolParam("edits", "object",
                  "替换操作列表，每项包含 oldText（要替换的原文）和 newText（替换后的内容）。"
                  "例: [{\"oldText\": \"foo\", \"newText\": \"bar\"}, {\"oldText\": \"baz\", \"newText\": \"qux\"}]",
                  required=True),
    ],
))
async def file_edit(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = Path(params.get("path") or "").expanduser()
    edits_raw = params.get("edits")

    if not path.exists():
        return ToolResult(summary=f"文件不存在: {path}（edit 只能修改已存在的文件，新文件请用 file.write）", error="FileNotFound")

    if not edits_raw:
        return ToolResult(summary="edits 参数为空，请提供至少一个 {oldText, newText} 替换操作", error="EmptyEdits", skipped=True)

    # 支持 list 或 JSON 字符串
    if isinstance(edits_raw, str):
        try:
            edits = json.loads(edits_raw)
        except json.JSONDecodeError:
            return ToolResult(summary="edits 不是合法的 JSON 数组", error="InvalidJSON")
    elif isinstance(edits_raw, list):
        edits = edits_raw
    else:
        return ToolResult(summary="edits 必须是数组或 JSON 字符串", error="InvalidType", skipped=True)

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        original = content
        changes_made = 0
        applied = []

        for i, edit in enumerate(edits):
            old_text = edit.get("oldText", "") if isinstance(edit, dict) else ""
            new_text = edit.get("newText", "") if isinstance(edit, dict) else ""

            if not old_text:
                return ToolResult(summary=f"edits[{i}]: oldText 不能为空", error="EmptyOldText", skipped=True)

            # 检查唯一性
            first_idx = content.find(old_text)
            if first_idx == -1:
                return ToolResult(
                    summary=f"edits[{i}]: oldText 在文件中未找到。请先用 file.read 确认当前内容。",
                    error="OldTextNotFound",
                    skipped=True,
                )

            second_idx = content.find(old_text, first_idx + len(old_text))
            if second_idx != -1:
                return ToolResult(
                    summary=(
                        f"edits[{i}]: oldText 在文件中出现 {content.count(old_text)} 次，不够唯一。"
                        f"请扩大 oldText 范围使其唯一，或拆分为多次 edit 调用。"
                    ),
                    error="NonUniqueOldText",
                    skipped=True,
                )

            content = content.replace(old_text, new_text, 1)
            changes_made += 1
            applied.append({
                "index": i,
                "old_preview": old_text[:60] + ("..." if len(old_text) > 60 else ""),
                "new_preview": new_text[:60] + ("..." if len(new_text) > 60 else ""),
            })

        path.write_text(content, encoding="utf-8")
        applied_summary = "\n".join(
            f"  [{a['index']}] {a['old_preview']} → {a['new_preview']}"
            for a in applied
        )
        payload = {"path": str(path), "changes": changes_made, "applied": applied}
        return ToolResult(
            summary=f"编辑成功: {path}（{changes_made} 处替换）\n{applied_summary}",
            evidence=json.dumps(payload, ensure_ascii=False),
            resource_key=str(path),
            fingerprint=f"edit:{hashlib.md5(content.encode('utf-8', errors='replace')).hexdigest()[:12]}",
            artifact_paths=[str(path)],
            state_delta={"file": "edited", "changes": changes_made},
            metadata=payload,
        )
    except Exception as e:
        _log.exception("编辑文件失败: %s", path)
        return ToolResult(summary=f"编辑失败: {path}", error=type(e).__name__)