"""tools/tts.py — tts.speak 工具（文字转语音）。

后端优先级：
  1. edge-tts — 免费，无需 API key（默认）
  2. DashScope — 需要 DASHSCOPE_API_KEY
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from core.paths import generated_dir
from typing import Any

from tools.registry import tool, ToolManifest, ToolResult, ToolParam, ToolContext

OUT_DIR = generated_dir()


@tool(ToolManifest(
    name="tts.speak",
    description="文字转语音，生成音频文件。",
    progress_category="io",
    params=[
        ToolParam("text", "string", "要转换的文字", required=True),
        ToolParam("voice", "string", "语音名称（默认 zh-CN-XiaoxiaoNeural）", required=False),
    ],
))
async def tts_speak(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    text = (params.get("text") or "").strip()
    if not text:
        return ToolResult(summary="text 不能为空", error="EmptyText", skipped=True)

    voice = (params.get("voice") or "zh-CN-XiaoxiaoNeural").strip()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 尝试 edge-tts
    try:
        import asyncio
        proc = await asyncio.create_subprocess_exec(
            "edge-tts", "--voice", voice, "--text", text, "--write-media", "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode == 0 and stdout:
            import hashlib, time
            filename = f"tts_{hashlib.md5(text.encode()).hexdigest()[:8]}_{int(time.time())}.mp3"
            out_path = OUT_DIR / filename
            out_path.write_bytes(stdout)
            return ToolResult(
                summary=f"✅ 语音已生成: {out_path} ({len(stdout)} bytes)",
                resource_key=str(out_path),
                artifact_paths=[str(out_path)],
                metadata={"voice": voice, "backend": "edge-tts", "file": str(out_path)},
            )
        # edge-tts 未安装或失败 → 尝试 DashScope
    except (FileNotFoundError, Exception):
        pass

    # DashScope 后备
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if api_key:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "qwen-tts",
                        "input": {"text": text},
                        "parameters": {"voice": "zhixiaobai", "format": "mp3"},
                    },
                )
                data = resp.json()
                url = (data.get("output", {}) or {}).get("audio_url", "")
                if url:
                    resp2 = await client.get(url)
                    import hashlib, time
                    filename = f"tts_ds_{hashlib.md5(text.encode()).hexdigest()[:8]}_{int(time.time())}.mp3"
                    out_path = OUT_DIR / filename
                    out_path.write_bytes(resp2.content)
                    return ToolResult(
                        summary=f"✅ 语音已生成: {out_path} ({len(resp2.content)} bytes)",
                        resource_key=str(out_path),
                        artifact_paths=[str(out_path)],
                        metadata={"voice": voice, "backend": "dashscope", "file": str(out_path)},
                    )
                return ToolResult(summary=f"DashScope TTS 失败: {data.get('message', data)}", error="TTSError")
        except Exception as e:
            return ToolResult(summary=f"TTS 失败: {e}", error="TTSError")

    return ToolResult(
        summary="edge-tts 未安装，DASHSCOPE_API_KEY 也未设置。安装: pip install edge-tts",
        error="NoTTSBackend",
        skipped=True,
    )
