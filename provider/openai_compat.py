"""provider/openai_compat.py — OpenAI 兼容接口实现（百炼/qwen/openai/copilot 等）。"""
from __future__ import annotations

import json
import logging
import os
import re as _re_mod
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

from store.auth import (
    load_copilot_token_cache,
    resolve_copilot_token,
    save_copilot_token_cache,
)
from provider.base import Message
from provider.catalog import lookup_model

if TYPE_CHECKING:
    from core.config import Config

_log = logging.getLogger("lingzhou.provider.openai_compat")

# embed 输入字符上限（DashScope text-embedding-v3 单次最大约 6000 tokens，保守按字符计）
_EMBED_MAX_CHARS: int = 6000

# thinking level → budget_max 的比例
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

_MAX_COMPLETION_TOKENS_DEFAULT = 16384


def _copilot_reasoning_effort(level: str) -> str:
    return "low" if level == "minimal" else level


def _raise_for_status_with_body(resp: httpx.Response) -> None:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = (resp.text or "").strip().replace("\n", " ")
        if not body:
            raise
        raise httpx.HTTPStatusError(
            f"{exc} body={body[:400]}",
            request=exc.request,
            response=exc.response,
        ) from exc


def _extract_responses_text(data: dict[str, Any]) -> str:
    """从 responses API 的返回中提取文本。"""
    if isinstance(data.get("output_text"), str) and data.get("output_text"):
        return str(data["output_text"])
    output = data.get("output") or []
    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
    return "\n".join(text_parts).strip()


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


# ═══════════════════════════════════════════════════════════════════════════════
# 模式适配器：用数据封装 openai / copilot 的差异，消除 if/elif 堆砌
# ═══════════════════════════════════════════════════════════════════════════════

class _ModeAdapter:
    """模式差异的抽象基类。每个具体模式只需覆写差异方法。"""

    def __init__(self, base_url: str, api_key: str, timeout: float):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    def build_sync_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=30.0,
        )

    def build_async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=20),
        )

    def resolve_url(self, path: str) -> str:
        return path  # openai 模式：base_url 已设在 client 上

    async def request_headers(self) -> dict[str, str]:
        return {}  # openai 模式：Authorization 已在 client headers

    async def handle_auth_error(self, resp: httpx.Response) -> httpx.Response | None:
        return None  # openai 模式：不需要 token 刷新

    def apply_completion_limits(self, payload: dict[str, Any]) -> None:
        pass  # openai 模式：不需要 completion_limit 参数

    def uses_responses_api(self, model_spec: dict[str, Any]) -> bool:
        return False

    def build_chat_payload(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        thinking_level: str,
        thinking_override: str | None,
        extra_body: dict[str, Any],
        model_spec: dict[str, Any],
    ) -> dict[str, Any]:
        """构建 chat/completions 请求 payload。"""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        return payload

    def embedding_url(self) -> str:
        return "/embeddings"


class _OpenAIMode(_ModeAdapter):
    """标准 OpenAI 兼容模式：百炼、DeepSeek 等。"""
    pass


class _CopilotMode(_ModeAdapter):
    """GitHub Copilot 模式：token exchange + IDE headers + responses API。"""

    def __init__(self, base_url: str, api_key: str, timeout: float):
        super().__init__(base_url, api_key, timeout)
        self._copilot_api_base_url = _normalize_copilot_api_base_url(base_url)
        self._copilot_gh_token: str = api_key
        self._copilot_token: str | None = None
        self._copilot_token_expires: float = 0.0

    def build_sync_client(self) -> httpx.Client:
        return httpx.Client(headers={"Content-Type": "application/json"}, timeout=30.0)

    def build_async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=20),
        )

    def resolve_url(self, path: str) -> str:
        return f"{self._copilot_api_base_url}{path}"

    async def request_headers(self) -> dict[str, str]:
        token = await self._ensure_copilot_token()
        return _build_copilot_ide_headers() | {"Authorization": f"Bearer {token}"}

    async def handle_auth_error(self, resp: httpx.Response) -> httpx.Response | None:
        body = resp.text
        if "Personal Access Tokens are not supported" in body:
            raise RuntimeError(
                "当前 GitHub token 没有成功走完 Copilot token exchange。\n"
                "请重新执行 `lingzhou auth login-copilot`。"
            )
        return None  # 实际重试逻辑在 chat() 中处理

    def apply_completion_limits(self, payload: dict[str, Any]) -> None:
        if "max_completion_tokens" in payload or "max_tokens" in payload:
            return
        # 从模型规格读取 limit
        param_name = "max_completion_tokens"  # copilot 默认用这个
        payload[param_name] = _MAX_COMPLETION_TOKENS_DEFAULT

    def uses_responses_api(self, model_spec: dict[str, Any]) -> bool:
        return model_spec.get("api") == "responses"

    def build_chat_payload(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        thinking_level: str,
        thinking_override: str | None,
        extra_body: dict[str, Any],
        model_spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Copilot responses API 格式。"""
        level = thinking_override if thinking_override is not None else thinking_level
        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str) and m.content.strip():
                    instructions_parts.append(m.content)
                continue
            input_items.append({"role": m.role, "content": m.content})

        payload: dict[str, Any] = {
            "model": model,
            "input": input_items or [{"role": "user", "content": ""}],
        }
        payload["temperature"] = temperature
        if instructions_parts:
            payload["instructions"] = "\n\n".join(instructions_parts)
        if model_spec.get("reasoning") and level != "off":
            payload["reasoning"] = {"effort": _copilot_reasoning_effort(level)}
        if extra_body:
            payload.update(extra_body)
        return payload

    # ── Copilot 内部方法 ─────────────────────────────────────────────────

    def _copilot_url(self, path: str) -> str:
        return f"{self._copilot_api_base_url}{path}"

    async def _ensure_copilot_token(self, *, force_refresh: bool = False) -> str:
        if (not force_refresh) and self._copilot_token and time.time() < self._copilot_token_expires - 300:
            return self._copilot_token

        cache = load_copilot_token_cache()
        if (not force_refresh) and cache and (time.time() * 1000) < cache.expires_at_ms - 300_000:
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
                        "Authorization": f"token {self._copilot_gh_token}",
                        "Accept": "application/json",
                        **_build_copilot_ide_headers(include_api_version=False),
                    },
                )
            _raise_for_status_with_body(resp)
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
                    "请重新执行 `lingzhou auth login-copilot`。"
                ) from exc
            raise


def _build_mode_adapter(
    *,
    mode: str,
    base_url: str,
    api_key_env: str,
    timeout: float,
) -> _ModeAdapter:
    """根据 provider.mode 创建对应的适配器。

    这是唯一需要 if 的地方——模式选择在构造时完成一次，
    之后所有模式差异通过多态分发。
    """
    if mode == "copilot":
        resolved = resolve_copilot_token(api_key_env)
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
        return _CopilotMode(base_url, resolved.token, timeout)

    # openai 模式（百炼、DeepSeek 等标准 OpenAI 兼容）
    api_key = os.environ.get(api_key_env, "")
    return _OpenAIMode(base_url, api_key, timeout)


# ═══════════════════════════════════════════════════════════════════════════════
# Provider 主体
# ═══════════════════════════════════════════════════════════════════════════════


class OpenAICompatProvider:
    """OpenAI 兼容接口。

    模式差异通过 _ModeAdapter 封装，避免 if/elif 堆砌：
    - mode=openai: 标准 OpenAI 兼容（百炼/DeepSeek），Bearer token + base_url
    - mode=copilot: GitHub Copilot，token exchange + IDE headers + responses API
    """

    def __init__(self, cfg: "Config") -> None:
        provider = cfg.active_provider
        self._model = cfg.active_model_id
        self._temperature = cfg.temperature
        self._thinking_level = cfg.thinking
        self._extra_body: dict[str, Any] = dict(provider.extra_body)
        self._base_url = provider.base_url.rstrip("/")
        self._embed_model: str | None = cfg.memory.embedding_model
        self._provider_mode = provider.mode

        # 模式适配器：封装 openai / copilot 的差异行为
        self._mode = _build_mode_adapter(
            mode=provider.mode,
            base_url=self._base_url,
            api_key_env=provider.api_key_env,
            timeout=cfg.timeout,
        )

        self._sync_client = self._mode.build_sync_client()
        self._client = self._mode.build_async_client()
        self.last_usage: dict[str, int] = {}  # 最近一次 API 调用的 token 用量

    def _resolve_url(self, path: str) -> str:
        _m = self.__dict__.get("_mode")
        if _m is not None:
            return _m.resolve_url(path)
        # Fallback for test injection: _copilot_url overridden as instance attr
        _cu = self.__dict__.get("_copilot_url")
        if callable(_cu):
            return _cu(path)
        base = self._copilot_api_base_url or self.__dict__.get("_base_url", "")
        return f"{str(base).rstrip('/')}{path}"

    async def _request_headers(self) -> dict[str, str]:
        _m = self.__dict__.get("_mode")
        if _m is not None:
            return await _m.request_headers()
        # Fallback for test injection
        if callable(getattr(self, "_ensure_copilot_token", None)) and callable(getattr(self, "_copilot_request_headers", None)):
            token = await self._ensure_copilot_token()
            return self._copilot_request_headers(token)
        return {}

    async def _handle_auth_error(self, resp: httpx.Response) -> httpx.Response | None:
        _m = self.__dict__.get("_mode")
        if _m is not None:
            return await _m.handle_auth_error(resp)
        return None

    async def _copilot_refreshed_headers(self) -> dict[str, str]:
        """Force-refresh Copilot token and return new request headers."""
        _m = self.__dict__.get("_mode")
        if _m is not None and hasattr(_m, "_ensure_copilot_token"):
            token = await _m._ensure_copilot_token(force_refresh=True)  # type: ignore[union-attr]
            return _build_copilot_ide_headers() | {"Authorization": f"Bearer {token}"}
        if callable(getattr(self, "_ensure_copilot_token", None)) and callable(getattr(self, "_copilot_request_headers", None)):
            token = await self._ensure_copilot_token(force_refresh=True)
            return self._copilot_request_headers(token)
        return await self._request_headers()

    # ── 向后兼容：copilot 方法透传（测试用）───────────────────────────

    @property
    def _copilot_api_base_url(self) -> str:
        backing = self.__dict__.get("_copilot_api_base_url_backing")
        if backing is not None:
            return backing
        return getattr(self.__dict__.get("_mode"), "_copilot_api_base_url", "")

    @_copilot_api_base_url.setter
    def _copilot_api_base_url(self, value: str) -> None:
        self.__dict__["_copilot_api_base_url_backing"] = value
        _m = self.__dict__.get("_mode")
        if _m is not None:
            _m._copilot_api_base_url = value

    async def _ensure_copilot_token(self, *, force_refresh: bool = False) -> str:
        if hasattr(self._mode, "_ensure_copilot_token"):
            return await self._mode._ensure_copilot_token(force_refresh=force_refresh)  # type: ignore[union-attr]
        raise RuntimeError("Not in copilot mode")

    def _uses_responses_api(self) -> bool:
        return self._model_api() == "responses"

    def _inject_completion_limits(self, payload: dict[str, Any]) -> None:
        """Inject completion limit param based on model spec (no _mode dependency)."""
        if "max_completion_tokens" in payload or "max_tokens" in payload:
            return
        spec = self._model_spec()
        req_params = spec.get("request_params") if spec else {}
        if not isinstance(req_params, dict):
            req_params = {}
        param_name = req_params.get("completion_limit_param")
        if not param_name:
            return
        max_tokens = spec.get("max_tokens")
        payload[param_name] = int(max_tokens) if max_tokens else _MAX_COMPLETION_TOKENS_DEFAULT

    def _build_responses_payload(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        thinking_override: str | None = None,
    ) -> dict[str, Any]:
        """Build responses API payload (no _mode dependency)."""
        spec = self._model_spec()
        level = thinking_override if thinking_override is not None else self._thinking_level
        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str) and m.content.strip():
                    instructions_parts.append(m.content)
                continue
            input_items.append({"role": m.role, "content": m.content})
        payload: dict[str, Any] = {
            "model": self._model,
            "input": input_items or [{"role": "user", "content": ""}],
        }
        t = temperature if temperature is not None else self._temperature
        unsupported = self._unsupported_request_params()
        if "temperature" not in unsupported:
            payload["temperature"] = t
        if instructions_parts:
            payload["instructions"] = "\n\n".join(instructions_parts)
        if spec.get("reasoning") and level != "off":
            payload["reasoning"] = {"effort": _copilot_reasoning_effort(level)}
        if self._extra_body:
            payload.update(self._extra_body)
        return payload

    def _copilot_compat_fallback_payload(self, *, base_payload: dict[str, Any], payload: dict[str, Any]) -> dict | None:
        # 去除 reasoning 字段：返回干净的 base_payload（无思考/无限制参数）
        if "reasoning_effort" in payload or "max_completion_tokens" in payload:
            return dict(base_payload)
        return None

    # ── thinking 注入 ──────────────────────────────────────────────────────

    def _inject_thinking(self, payload: dict[str, Any], level_override: str | None = None) -> None:
        """按 provider.mode 和 cfg.thinking 向 payload 注入 thinking 参数。"""
        level = level_override if level_override is not None else self._thinking_level
        spec = self._model_spec()

        if self._provider_mode == "openai":
            thinking_spec = spec.get("thinking") if spec else None
            if thinking_spec is None or level == "off":
                if level == "off":
                    payload["enable_thinking"] = False
                return
            budget = self._compute_budget(thinking_spec, level)
            payload["enable_thinking"] = True
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}

        elif self._provider_mode == "copilot":
            is_reasoning = bool(spec and spec.get("reasoning")) if spec else False
            if is_reasoning and level != "off":
                effort = "low" if level == "minimal" else level
                if self._model_api() == "chat_completions":
                    payload["reasoning_effort"] = effort
                    if "temperature" not in self._unsupported_request_params():
                        payload["temperature"] = 1

    @staticmethod
    def _compute_budget(thinking_spec: dict[str, Any], level: str) -> int:
        frac = _LEVEL_FRACS.get(level, 0.0)
        budget_max = thinking_spec.get("budget_max", 4096)
        budget_min = thinking_spec.get("budget_min", 1024)
        return max(int(budget_max * frac), budget_min)

    def _model_spec(self) -> dict[str, Any]:
        spec = lookup_model(self._model)
        return spec if isinstance(spec, dict) else {}

    def _record_usage(self, usage: dict | None) -> None:
        if not isinstance(usage, dict):
            return
        self.last_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    def _model_api(self) -> str:
        api = self._model_spec().get("api")
        return str(api) if isinstance(api, str) and api else "chat_completions"

    def _request_params_spec(self) -> dict[str, Any]:
        params = self._model_spec().get("request_params")
        return params if isinstance(params, dict) else {}

    def _unsupported_request_params(self) -> set[str]:
        raw = self._request_params_spec().get("unsupported")
        return {str(item) for item in raw} if isinstance(raw, list) else set()

    # ── chat ───────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        thinking_override: str | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self._temperature
        level = thinking_override if thinking_override is not None else self._thinking_level

        # responses API 路径（copilot 专属）
        if self._uses_responses_api():
            payload = self._build_responses_payload(
                messages, temperature=temp, thinking_override=thinking_override,
            )
            req_timeout = max(float(self._client.timeout.read or 60.0), 300.0) if level not in (None, "off") else None
            target = self._resolve_url("/responses")

            headers = await self._request_headers()
            resp = await self._client.post(target, content=json.dumps(payload),
                                           headers=headers or None, timeout=req_timeout)
            # Copilot responses: on 400, refresh token and retry
            if resp.status_code == 400 and self._provider_mode == "copilot":
                headers = await self._copilot_refreshed_headers()
                resp = await self._client.post(target, content=json.dumps(payload),
                                               headers=headers or None, timeout=req_timeout)
            _raise_for_status_with_body(resp)
            data = resp.json()
            self._record_usage(data.get("usage"))
            return _extract_responses_text(data)

        # chat/completions 路径（openai + copilot 通用）
        base_payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temp,
        }
        payload = dict(base_payload)
        self._inject_thinking(payload, level_override=thinking_override)
        self._inject_completion_limits(payload)
        if self._extra_body:
            payload.update(self._extra_body)

        req_timeout = max(float(self._client.timeout.read or 60.0), 300.0) if level not in (None, "off") else None
        target = self._resolve_url("/chat/completions")

        headers = await self._request_headers()
        resp = await self._client.post(target, content=json.dumps(payload),
                                       headers=headers or None, timeout=req_timeout)

        # Copilot: on 400, refresh token and retry; still 400 → fallback without reasoning
        if resp.status_code == 400 and self._provider_mode == "copilot":
            headers = await self._copilot_refreshed_headers()
            resp = await self._client.post(target, content=json.dumps(payload),
                                           headers=headers or None, timeout=req_timeout)
            if resp.status_code == 400:
                fallback = self._copilot_compat_fallback_payload(base_payload=base_payload, payload=payload)
                if fallback is not None:
                    resp = await self._client.post(target, content=json.dumps(fallback),
                                                   headers=headers or None, timeout=req_timeout)

        _raise_for_status_with_body(resp)
        data = resp.json()
        self._record_usage(data.get("usage"))
        choices = data.get("choices") or []
        if not choices:
            finish = (data.get("choices") or [{}])[0].get("finish_reason") if data.get("choices") else None
            raise RuntimeError(
                f"API 返回空 choices（可能触发内容过滤或限流）"
                + (f"，finish_reason={finish}" if finish else "")
            )
        msg = choices[0]["message"]
        content: str = msg.get("content") or ""
        if msg.get("reasoning_content"):
            return content
        import re as _re
        content = _re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
        return content

    async def close(self) -> None:
        await self._client.aclose()
        self._sync_client.close()

    def embed(self, text: str) -> list[float]:
        if not self._embed_model:
            raise RuntimeError("embedding_model not configured")

        headers = {}
        if self._provider_mode == "copilot":
            cache = load_copilot_token_cache()
            if not cache or (time.time() * 1000) >= cache.expires_at_ms - 300_000:
                raise RuntimeError(
                    "Copilot embeddings 需要先完成 GitHub token → Copilot token exchange。\n"
                    "请先执行一次 chat 请求，或关闭 embedding_model。"
                )
            headers = _build_copilot_ide_headers() | {"Authorization": f"Bearer {cache.token}"}

        target = self._resolve_url(self._mode.embedding_url())
        resp = self._sync_client.post(
            target,
            content=json.dumps({
                "model": self._embed_model,
                "input": [text[:_EMBED_MAX_CHARS]],
            }),
            headers=headers or None,
        )
        _raise_for_status_with_body(resp)
        return resp.json()["data"][0]["embedding"]
