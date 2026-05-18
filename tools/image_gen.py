"""tools/image_gen.py — image.generate 工具（文字生成图片）。

支持后端：
- DashScope (wanx) — 通过 DASHSCOPE_API_KEY
- OpenAI 兼容 (DALL-E) — 通过标准 API
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from core.paths import generated_dir
from typing import Any

import httpx

from tools.registry import tool, ToolManifest, ToolResult, ToolParam, ToolContext

# ── 选项 ─────────────────────────────────────────────────────────────────────
GEN_TIMEOUT = 120


async def _dashscope_generate(prompt: str, size: str, api_key: str) -> tuple[str, str]:
    """DashScope wanx 图片生成。返回 (url, error)。"""
    size_map = {"1024x1024": "1024*1024", "768x1024": "768*1024", "512x512": "512*512"}
    dash_size = size_map.get(size, "1024*1024")

    async with httpx.AsyncClient(timeout=GEN_TIMEOUT) as client:
        # 提交任务
        resp = await client.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            },
            json={
                "model": "wanx2.0-t2i-turbo",
                "input": {"prompt": prompt},
                "parameters": {"size": dash_size, "n": 1},
            },
        )
        data = resp.json()
        if resp.status_code != 200:
            return "", data.get("message", str(data))

        task_id = data.get("output", {}).get("task_id", "")
        if not task_id:
            return "", f"无 task_id: {data}"

        # 轮询结果
        for _ in range(20):
            await asyncio_sleep(3)
            resp2 = await client.get(
                f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            result = resp2.json()
            status = result.get("output", {}).get("task_status", "")
            if status == "SUCCEEDED":
                url = result["output"]["results"][0]["url"]
                return url, ""
            elif status == "FAILED":
                return "", result.get("output", {}).get("message", "未知错误")

        return "", "图片生成超时"


async def _openai_generate(prompt: str, size: str, api_key: str, base_url: str) -> tuple[str, str]:
    """OpenAI DALL-E 兼容图片生成。"""
    async with httpx.AsyncClient(timeout=GEN_TIMEOUT) as client:
        resp = await client.post(
            f"{base_url}/images/generations",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,
                "size": size,
            },
        )
        data = resp.json()
        if resp.status_code != 200:
            return "", data.get("error", {}).get("message", str(data))
        url = data.get("data", [{}])[0].get("url", "")
        return url, "" if url else "无返回 URL"


async def asyncio_sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)


@tool(ToolManifest(
    name="image.generate",
    description=(
        "文字生成图片。使用 DashScope wanx 或 OpenAI DALL-E。\n"
        "生成的图片保存到 ~/.lingzhou/generated/ 目录。"
    ),
    progress_category="io",
    params=[
        ToolParam("prompt", "string", "图片描述（英文效果最好）", required=True),
        ToolParam("size", "string", "尺寸: 1024x1024 / 768x1024 / 512x512", required=False),
        ToolParam("provider", "string", "后端: dashscope / openai（默认 dashscope）", required=False),
    ],
))
async def image_generate(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    prompt = (params.get("prompt") or "").strip()
    if not prompt:
        return ToolResult(summary="prompt 不能为空", error="EmptyPrompt", skipped=True)

    size = (params.get("size") or "1024x1024").strip()
    backend = (params.get("provider") or "dashscope").strip().lower()

    url = ""
    error = ""

    if backend == "dashscope":
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            return ToolResult(summary="DASHSCOPE_API_KEY 未设置", error="NoApiKey", skipped=True)
        url, error = await _dashscope_generate(prompt, size, api_key)

    elif backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            return ToolResult(summary="OPENAI_API_KEY 未设置", error="NoApiKey", skipped=True)
        url, error = await _openai_generate(prompt, size, api_key, base_url)

    else:
        return ToolResult(summary=f"未知后端: {backend}，支持: dashscope, openai", error="BadProvider", skipped=True)

    if error:
        return ToolResult(summary=f"生成失败: {error}", error="GenerateError")

    # 保存到本地
    out_dir = generated_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            img_data = resp.content
        filename = f"img_{hashlib.md5(prompt.encode()).hexdigest()[:8]}_{int(time.time())}.png"
        out_path = out_dir / filename
        out_path.write_bytes(img_data)
        return ToolResult(
            summary=f"✅ 图片已生成: {out_path} ({len(img_data)} bytes)\nPrompt: {prompt}",
            resource_key=str(out_path),
            evidence=f"file://{out_path}",
            artifact_paths=[str(out_path)],
            metadata={"prompt": prompt, "size": size, "backend": backend, "file": str(out_path)},
            state_delta={"generated": str(out_path)},
        )
    except Exception as e:
        return ToolResult(
            summary=f"生成成功但下载失败: {e}\nURL: {url}",
            error="DownloadError",
            evidence=url,
        )
