"""provider/openai_compat.py — OpenAI 兼容接口实现（百炼/qwen/openai/copilot 等）。"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

from provider.base import Message
from provider.catalog import lookup_model

if TYPE_CHECKING:
    from core.config import Config

# embed 输入字符上限（DashScope text-embedding-v3 单次最大约 6000 tokens，保守按字符计）
_EMBED_MAX_CHARS: int = 6000

# thinking level → budget_max 的比例（与 OpenClaw 对齐）
_LEVEL_FRACS: dict[str, float] = {
    "minimal": 0.05,
    "low":     0.15,
    "medium":  0.40,
    "high":    1.00,
}


class OpenAICompatProvider:
    def __init__(self, cfg: "Config") -> None:
        provider = cfg.active_provider
        self._model = cfg.active_model_id
        self._temperature = cfg.temperature
        self._thinking_level = cfg.thinking          # "off" | "minimal" | "low" | "medium" | "high"
        self._provider_mode = provider.mode          # "openai" | "copilot"
        self._extra_body: dict[str, Any] = dict(provider.extra_body)  # escape hatch，浅拷贝防污染
        self._base_url = provider.base_url.rstrip("/")
        self._api_key = provider.api_key
        self._embed_model: str | None = cfg.memory.embedding_model
        # 同步嵌入复用连接（避免 embed() 每次新建 Client）
        self._sync_client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=cfg.timeout,
        )
        # Copilot 短期 token 缓存（仅 mode=copilot 时使用）
        self._copilot_gh_pat: str = self._api_key   # 原始 GitHub PAT
        self._copilot_token: str | None = None
        self._copilot_token_expires: float = 0.0    # Unix timestamp

    # ── Copilot token 刷新 ─────────────────────────────────────────────────

    async def _ensure_copilot_token(self) -> str:
        """获取或刷新 GitHub Copilot 短期 token（TTL ~30 分钟，提前 5 分钟刷新）。

        流程：GitHub PAT → api.github.com/copilot_internal/v2/token → 短期 token
        若 token exchange 端点返回 4xx（账号未启用内部端点），直接使用原始 PAT。
        """
        if self._copilot_token and time.time() < self._copilot_token_expires - 300:
            return self._copilot_token

        try:
            async with httpx.AsyncClient(timeout=15.0) as tmp:
                resp = await tmp.get(
                    "https://api.github.com/copilot_internal/v2/token",
                    headers={
                        "Authorization": f"token {self._copilot_gh_pat}",
                        "Accept": "application/json",
                        "User-Agent": "lingzhou/1.0",
                    },
                )
            if resp.status_code >= 400:
                # 账号不支持内部 token exchange（如普通 Copilot 订阅）→ 直接使用 PAT
                self._copilot_token = self._copilot_gh_pat
                self._copilot_token_expires = time.time() + 3600
                return self._copilot_token
            data = resp.json()
            self._copilot_token = data["token"]
            expires_str: str = data.get("expires_at", "")
            if expires_str:
                dt = datetime.fromisoformat(expires_str.replace("Z", "+00:00"))
                self._copilot_token_expires = dt.timestamp()
            else:
                self._copilot_token_expires = time.time() + 1800  # 默认 30 分钟
        except Exception:
            # 网络错误等异常 → 回退使用原始 PAT
            self._copilot_token = self._copilot_gh_pat
            self._copilot_token_expires = time.time() + 3600
        return self._copilot_token  # type: ignore[return-value]

    # ── thinking 注入 ──────────────────────────────────────────────────────

    def _inject_thinking(self, payload: dict[str, Any]) -> None:
        """按 provider.mode 和 cfg.thinking 向 payload 注入对应的 thinking 参数。

        openai（百炼/Qwen）体系：
          - 支持 thinking 的模型：off → enable_thinking=False；其余 → enable_thinking=True + budget_tokens
          - 不支持 thinking 的模型：不注入任何 thinking 参数
          - 目录未收录的模型：不注入（由 extra_body escape hatch 兜底）

        copilot（GPT-5/o-series）体系：
          - reasoning=true 且 level != off → reasoning_effort 字符串 + temperature=1
          - off 或非推理模型 → 不注入
        """
        level = self._thinking_level
        spec = lookup_model(self._model)  # 可能为 None（目录未收录）

        if self._provider_mode == "openai":
            thinking_spec = spec.get("thinking") if spec else None
            if thinking_spec is None:
                return  # 目录未收录或模型不支持 thinking，不注入
            if level == "off":
                payload["enable_thinking"] = False
            else:
                budget = self._compute_budget(thinking_spec, level)
                payload["enable_thinking"] = True
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}

        elif self._provider_mode == "copilot":
            is_reasoning = bool(spec and spec.get("reasoning")) if spec else False
            if is_reasoning and level != "off":
                # "minimal" → "low"（copilot 不支持 minimal 档）
                effort = "low" if level == "minimal" else level
                payload["reasoning_effort"] = effort
                payload["temperature"] = 1  # OpenAI reasoning 模型要求 temperature=1

    @staticmethod
    def _compute_budget(thinking_spec: dict[str, Any], level: str) -> int:
        """按等级比例计算 budget_tokens（不低于 budget_min）。"""
        frac = _LEVEL_FRACS.get(level, 0.0)
        budget_max = thinking_spec.get("budget_max", 4096)
        budget_min = thinking_spec.get("budget_min", 1024)
        return max(int(budget_max * frac), budget_min)

    # ── chat ───────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else self._temperature,
        }
        # gpt-5.x / o-series 不支持 max_tokens，须用 max_completion_tokens
        if self._provider_mode == "copilot":
            payload["max_completion_tokens"] = 16384
        # 1. 按 mode 注入 thinking 参数
        self._inject_thinking(payload)
        # 2. extra_body 最后合并（escape hatch，可覆盖上面任意字段）
        if self._extra_body:
            payload.update(self._extra_body)
        # 3. copilot 模式：动态刷新短期 token
        req_headers: dict[str, str] = {}
        if self._provider_mode == "copilot":
            token = await self._ensure_copilot_token()
            req_headers["Authorization"] = f"Bearer {token}"
        resp = await self._client.post(
            "/chat/completions",
            content=json.dumps(payload),
            headers=req_headers if req_headers else None,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def close(self) -> None:
        await self._client.aclose()
        self._sync_client.close()

    def embed(self, text: str) -> list[float]:
        """同步文本嵌入（用于 SemanticMemory 批量计算；使用独立 httpx 同步客户端）。

        仅在 cfg.memory.embedding_model 非 None 时可用，调用方应先检查。
        降级：API 失败时抛出异常，SemanticMemory 会静默跳过。
        """
        if not self._embed_model:
            raise RuntimeError("embedding_model not configured")
        resp = self._sync_client.post(
            "/embeddings",
            content=json.dumps({
                "model": self._embed_model,
                "input": [text[:_EMBED_MAX_CHARS]],
            }),
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
