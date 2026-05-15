"""tools/image.py — 图片分析工具。"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

from provider import create_provider
from provider.base import Message
from tools.file import _resolve_read_path
from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool


def _collect_image_sources(params: dict[str, Any]) -> list[str]:
    sources: list[str] = []

    for key in ("path", "url"):
        raw = str(params.get(key) or "").strip()
        if raw:
            sources.append(raw)

    for key in ("paths", "urls", "images"):
        raw = params.get(key)
        if raw is None:
            continue

        if isinstance(raw, list):
            sources.extend(str(item).strip() for item in raw if str(item).strip())
            continue

        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                continue
            if text.startswith("["):
                parsed = json.loads(text)
                if not isinstance(parsed, list):
                    raise ValueError(f"{key} 必须是字符串数组")
                sources.extend(str(item).strip() for item in parsed if str(item).strip())
            else:
                sources.extend(part.strip() for part in text.split(",") if part.strip())
            continue

        raise ValueError(f"{key} 必须是数组或 JSON 字符串")

    deduped: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if source not in seen:
            seen.add(source)
            deduped.append(source)
    return deduped


def _image_part_from_source(source: str, detail: str) -> dict[str, Any]:
    raw = source.strip()
    if raw.startswith(("http://", "https://", "data:image/")):
        return {
            "type": "image_url",
            "image_url": {"url": raw, "detail": detail},
        }

    path = _resolve_read_path(Path(raw).expanduser())
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"图片路径不是文件: {path}")

    mime, _ = mimetypes.guess_type(path.name)
    if not mime or not mime.startswith("image/"):
        raise ValueError(f"不支持的图片类型: {path}")

    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{data}", "detail": detail},
    }


@tool(ToolManifest(
    name="image.analyze",
    description="分析一张或多张图片，支持本地文件、远程 URL 或 data URL。",
    params=[
        ToolParam("prompt", "string", "分析提示词；不传则使用默认描述请求", required=False),
        ToolParam("path", "string", "单张本地图片路径或远程图片 URL", required=False),
        ToolParam("paths", "object", "多张图片路径/URL 列表，支持数组或 JSON 字符串", required=False),
        ToolParam("detail", "string", "图片细节等级：low / high / auto，默认 auto", required=False),
    ],
))
async def image_analyze(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    prompt = str(params.get("prompt") or "请分析这些图片里的关键信息，并给出简洁结论。").strip()
    detail = str(params.get("detail") or "auto").strip() or "auto"

    try:
        sources = _collect_image_sources(params)
    except Exception as exc:
        return ToolResult(summary=f"图片参数解析失败: {exc}", error=type(exc).__name__, skipped=True)

    if not sources:
        return ToolResult(summary="至少需要提供 path 或 paths", error="EmptyImageSources", skipped=True)

    try:
        content = [{"type": "text", "text": prompt}]
        content.extend(_image_part_from_source(source, detail) for source in sources)
    except Exception as exc:
        return ToolResult(summary=f"图片读取失败: {exc}", error=type(exc).__name__, skipped=True)

    provider = create_provider(ctx.config)
    try:
        raw = await provider.chat(
            [
                Message(
                    role="system",
                    content="你是灵舟的多模态感知模块。只基于可见内容回答，不要臆造看不见的细节。",
                ),
                Message(role="user", content=content),
            ],
            thinking_override="low",
        )
    except Exception as exc:
        return ToolResult(summary=f"图片分析失败: {exc}", error=type(exc).__name__)
    finally:
        await provider.close()

    summary = raw.strip() or "（空回复）"
    digest = hashlib.md5((prompt + "\n" + "\n".join(sources)).encode("utf-8", errors="replace")).hexdigest()[:12]
    return ToolResult(
        summary=summary,
        resource_key=sources[0],
        fingerprint=f"image:{digest}",
        artifact_paths=[s for s in sources if not s.startswith(("http://", "https://", "data:image/"))],
        metadata={"image_count": len(sources), "detail": detail, "sources": sources},
    )