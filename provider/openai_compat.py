"""provider/openai_compat.py — OpenAI 兼容接口实现（百炼/qwen/openai/copilot 等）。"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

from auth_store import (
    load_copilot_token_cache,
    resolve_copilot_token,
    save_copilot_token_cache,
)
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

COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
COPILOT_EDITOR_VERSION = "vscode/1.96.2"
COPILOT_USER_AGENT = "GitHubCopilotChat/0.26.7"
COPILOT_EDITOR_PLUGIN_VERSION = "copilot-chat/0.35.0"
COPILOT_GITHUB_API_VERSION = "2025-04-01"
DEFAULT_COPILOT_API_BASE_URL = "https://api.individual.githubcopilot.com"

# gpt-5.x / o-series 在该兼容端点上不接受传统 max_tokens，
# 需要改走 max_completion_tokens。
_MAX_COMPLETION_TOKENS_MODELS = ("gpt-5", "o1", "o3", "o4")
_MAX_COMPLETION_TOKENS_DEFAULT = 16384


def _build_copilot_ide_headers(*, include_api_version: bool = False) -> dict[str, str]:
    headers = {
        "Editor-Version": COPILOT_EDITOR_VERSION,
        "Editor-Plugin-Version": COPILOT_EDITOR_PLUGIN_VERSION,
        "User-Agent": COPILOT_USER_AGENT,
    }
    if include_api_version:
        headers["X-Github-Api-Version"] = COPILOT_GITHUB_API_VERSION
    return headers


def _resolve_copilot_proxy_host(proxy_ep: str) -> str | None:
    trimmed = proxy_ep.strip()
    if not trimmed:
        return None
    url_text = trimmed if trimmed.startswith(("http://", "https://")) else f"https://{trimmed}"
    try:
        parsed = httpx.URL(url_text)
    except Exception:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    host = (parsed.host or "").strip().lower()
    return host or None


def _derive_copilot_api_base_url_from_token(token: str) -> str | None:
    trimmed = token.strip()
    if not trimmed:
        return None
    marker = "proxy-ep="
    for part in trimmed.split(";"):
        part = part.strip()
        if part.lower().startswith(marker):
            host = _resolve_copilot_proxy_host(part[len(marker):])
            if not host:
                return None
            return f"https://{host.replace('proxy.', 'api.', 1)}"
    return None


def _normalize_copilot_api_base_url(raw: str) -> str:
    trimmed = raw.strip().rstrip("/")
    if not trimmed or trimmed == "https://api.githubcopilot.com":
        return DEFAULT_COPILOT_API_BASE_URL
    return trimmed


class OpenAICompatProvider:
    def __init__(self, cfg: "Config") -> None:
        provider = cfg.active_provider
        self._model = cfg.active_model_id
        self._temperature = cfg.temperature
        self._thinking_level = cfg.thinking          # "off" | "minimal" | "low" | "medium" | "high"
        self._provider_mode = provider.mode          # "openai" | "copilot"
        self._extra_body: dict[str, Any] = dict(provider.extra_body)  # escape hatch，浅拷贝防污染
        self._base_url = provider.base_url.rstrip("/")
        self._embed_model: str | None = cfg.memory.embedding_model

        if self._provider_mode == "copilot":
            resolved = resolve_copilot_token(provider.api_key_env)
            if not resolved:
                raise EnvironmentError(
                    "未找到 Copilot 的 GitHub token。\n"
                    "Lingzhou 使用：GitHub token → Copilot token exchange → Copilot API\n"
                    "请执行以下任一操作：\n"
                    "  lingzhou auth login-copilot\n"
                    "  export COPILOT_GITHUB_TOKEN=your_token\n"
                    "  export GH_TOKEN=your_token\n"
                    "  export GITHUB_TOKEN=your_token"
                )
            self._api_key = resolved.token
            self._copilot_api_base_url = _normalize_copilot_api_base_url(self._base_url)
            self._sync_client = httpx.Client(
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            )
            self._client = httpx.AsyncClient(
                headers={"Content-Type": "application/json"},
                timeout=cfg.timeout,
            )
            # Copilot 短期 token 缓存（仅 mode=copilot 时使用）
            self._copilot_gh_token: str = self._api_key
            self._copilot_token: str | None = None
            self._copilot_token_expires: float = 0.0    # Unix timestamp
        else:
            self._api_key = provider.api_key
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
            self._copilot_api_base_url = ""
            self._copilot_gh_token = ""
            self._copilot_token = None
            self._copilot_token_expires = 0.0

    # ── Copilot token 刷新 ─────────────────────────────────────────────────

    def _copilot_url(self, path: str) -> str:
        return f"{self._copilot_api_base_url}{path}"

    def _copilot_request_headers(self, token: str) -> dict[str, str]:
        headers = _build_copilot_ide_headers()
        headers["Authorization"] = f"Bearer {token}"
        headers["Content-Type"] = "application/json"
        return headers

    async def _ensure_copilot_token(self) -> str:
        """获取或刷新 GitHub Copilot 短期 token（TTL ~30 分钟，提前 5 分钟刷新）。

        主链路固定为：
          GitHub token → Copilot token exchange → Copilot API
        """
        if self._copilot_token and time.time() < self._copilot_token_expires - 300:
            return self._copilot_token

        cache = load_copilot_token_cache()
        if cache and (time.time() * 1000) < cache.expires_at_ms - 300_000:
            self._copilot_token = cache.token
            self._copilot_token_expires = cache.expires_at_ms / 1000
            self._copilot_api_base_url = (
                _derive_copilot_api_base_url_from_token(cache.token)
                or DEFAULT_COPILOT_API_BASE_URL
            )
            return self._copilot_token

        try:
            async with httpx.AsyncClient(timeout=15.0) as tmp:
                resp = await tmp.get(
                    COPILOT_TOKEN_URL,
                    headers={
                        "Authorization": f"Bearer {self._copilot_gh_token}",
                        "Accept": "application/json",
                        **_build_copilot_ide_headers(include_api_version=True),
                    },
                )
            resp.raise_for_status()
            data = resp.json()
            token = str(data.get("token", "")).strip()
            if not token:
                raise RuntimeError("Copilot token exchange succeeded but returned empty token")

            expires_str = str(data.get("expires_at", "")).strip()
            if expires_str:
                if expires_str.isdigit():
                    expires_at_ms = int(expires_str) * 1000
                else:
                    dt = datetime.fromisoformat(expires_str.replace("Z", "+00:00"))
                    expires_at_ms = int(dt.timestamp() * 1000)
            else:
                expires_at_ms = int((time.time() + 1800) * 1000)

            self._copilot_token = token
            self._copilot_token_expires = expires_at_ms / 1000
            self._copilot_api_base_url = (
                _derive_copilot_api_base_url_from_token(token)
                or DEFAULT_COPILOT_API_BASE_URL
            )
            save_copilot_token_cache(token, expires_at_ms=expires_at_ms)
            return self._copilot_token

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403, 404):
                raise RuntimeError(
                    "GitHub token 无法完成 Copilot token exchange。\n"
                    "请重新执行 `lingzhou auth login-copilot`，并提供可访问\n"
                    "https://api.github.com/copilot_internal/v2/token 的 GitHub token。"
                ) from exc
            raise

    # ── thinking 注入 ──────────────────────────────────────────────────────

    def _inject_thinking(self, payload: dict[str, Any]) -> None:
        """按 provider.mode 和 cfg.thinking 向 payload 注入对应的 thinking 参数。"""
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

    def _inject_completion_limits(self, payload: dict[str, Any]) -> None:
        if self._provider_mode != "copilot":
            return
        if "max_completion_tokens" in payload or "max_tokens" in payload:
            return
        if self._model.startswith(_MAX_COMPLETION_TOKENS_MODELS):
            payload["max_completion_tokens"] = _MAX_COMPLETION_TOKENS_DEFAULT

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
        self._inject_thinking(payload)
        self._inject_completion_limits(payload)
        if self._extra_body:
            payload.update(self._extra_body)

        req_headers: dict[str, str] = {}
        if self._provider_mode == "copilot":
            token = await self._ensure_copilot_token()
            req_headers = self._copilot_request_headers(token)

        target_url = "/chat/completions"
        if self._provider_mode == "copilot":
            target_url = self._copilot_url("/chat/completions")

        resp = await self._client.post(
            target_url,
            content=json.dumps(payload),
            headers=req_headers if req_headers else None,
        )
        if self._provider_mode == "copilot" and resp.status_code == 400:
            body = resp.text
            if "Personal Access Tokens are not supported for this endpoint" in body:
                raise RuntimeError(
                    "当前 GitHub token 没有成功走完 Copilot token exchange，或换到的 token 不可用。\n"
                    "请重新执行 `lingzhou auth login-copilot`，并提供可访问 copilot_internal 的 GitHub token。"
                )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def close(self) -> None:
        await self._client.aclose()
        self._sync_client.close()

    def embed(self, text: str) -> list[float]:
        """同步文本嵌入（用于 SemanticMemory 批量计算）。"""
        if not self._embed_model:
            raise RuntimeError("embedding_model not configured")

        headers: dict[str, str] | None = None
        if self._provider_mode == "copilot":
            cache = load_copilot_token_cache()
            if not cache or (time.time() * 1000) >= cache.expires_at_ms - 300_000:
                raise RuntimeError(
                    "Copilot embeddings 需要先完成 GitHub token → Copilot token exchange。\n"
                    "请先执行一次 chat 请求，或关闭 embedding_model。"
                )
            headers = self._copilot_request_headers(cache.token)

        target_url = "/embeddings"
        if self._provider_mode == "copilot":
            target_url = self._copilot_url("/embeddings")

        resp = self._sync_client.post(
            target_url,
            content=json.dumps({
                "model": self._embed_model,
                "input": [text[:_EMBED_MAX_CHARS]],
            }),
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
