"""auth_store.py — 认证资料存储与解析。

目标：参考 OpenClaw 的 auth profile store 形状，给 Lingzhou 提供最小但正式的
Copilot 凭证管理能力。

当前范围：
- 规范化 auth-profiles.json 读写
- Copilot token 解析（env → auth profile → legacy credentials）
- Copilot 短期 token cache 读写
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUTH_PROFILES_PATH = Path("~/.lingzhou/auth-profiles.json").expanduser()
LEGACY_CREDENTIALS_PATH = Path("~/.lingzhou/credentials.json").expanduser()
COPILOT_TOKEN_CACHE_PATH = Path("~/.lingzhou/credentials/github-copilot.token.json").expanduser()
GITHUB_DEVICE_AUTH_PATH = Path("~/.lingzhou/auth/github-device.json").expanduser()

COPILOT_PROFILE_ID = "copilot:default"
COPILOT_ENV_ORDER = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
# TODO: 这里应替换成 Lingzhou 自己注册的 GitHub OAuth App client_id。
# 目前保持空字符串，避免误把别家的 client_id 当作 Lingzhou 的正式方案。
BUILTIN_GITHUB_DEVICE_CLIENT_ID = ""


@dataclass(frozen=True)
class TokenResolution:
    token: str
    source: str
    profile_id: str | None = None


@dataclass(frozen=True)
class CopilotTokenCache:
    token: str
    expires_at_ms: int
    updated_at_ms: int


def mask_secret(secret: str) -> str:
    if len(secret) <= 12:
        return "*" * len(secret)
    return f"{secret[:8]}...{secret[-4:]}"


def load_auth_profiles(path: Path | None = None) -> dict[str, Any]:
    path = path or AUTH_PROFILES_PATH
    if not path.exists():
        return {"version": 1, "profiles": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "profiles": {}}
    if not isinstance(data, dict):
        return {"version": 1, "profiles": {}}
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    return {"version": int(data.get("version", 1)), "profiles": profiles}


def save_auth_profiles(data: dict[str, Any], path: Path | None = None) -> None:
    path = path or AUTH_PROFILES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    path.chmod(0o600)


def get_auth_profile(profile_id: str, path: Path | None = None) -> dict[str, Any] | None:
    return load_auth_profiles(path).get("profiles", {}).get(profile_id)


def set_token_profile(
    *,
    profile_id: str = COPILOT_PROFILE_ID,
    provider: str,
    token: str,
    path: Path | None = None,
) -> None:
    data = load_auth_profiles(path)
    profiles = data.setdefault("profiles", {})
    profiles[profile_id] = {
        "type": "token",
        "provider": provider,
        "token": token,
    }
    save_auth_profiles(data, path)


def load_legacy_credentials(path: Path | None = None) -> dict[str, Any]:
    path = path or LEGACY_CREDENTIALS_PATH
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_legacy_credentials(data: dict[str, Any], path: Path | None = None) -> None:
    path = path or LEGACY_CREDENTIALS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    path.chmod(0o600)


def load_github_device_client_id(path: Path | None = None) -> str:
    path = path or GITHUB_DEVICE_AUTH_PATH
    env_value = os.environ.get("LINGZHOU_GITHUB_CLIENT_ID", "").strip()
    if env_value:
        return env_value

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                client_id = str(data.get("client_id", "")).strip()
                if client_id:
                    return client_id
        except Exception:
            pass

    return BUILTIN_GITHUB_DEVICE_CLIENT_ID.strip()


def resolve_copilot_token(api_key_env: str = "GITHUB_TOKEN") -> TokenResolution | None:
    seen: set[str] = set()
    ordered_envs: list[str] = []
    for name in (*COPILOT_ENV_ORDER, api_key_env):
        if name and name not in seen:
            ordered_envs.append(name)
            seen.add(name)

    # 对齐 OpenClaw：显式登录写入的 auth profile 比环境变量优先。
    profile = get_auth_profile(COPILOT_PROFILE_ID)
    if profile and isinstance(profile, dict):
        token = str(profile.get("token", "")).strip()
        if token:
            return TokenResolution(token=token, source="auth-profile", profile_id=COPILOT_PROFILE_ID)

    for name in ordered_envs:
        token = os.environ.get(name, "").strip()
        if token:
            return TokenResolution(token=token, source=f"env:{name}")

    legacy = load_legacy_credentials()
    for name in ordered_envs:
        token = str(legacy.get(name, "")).strip()
        if token:
            return TokenResolution(token=token, source=f"legacy-credentials:{name}")

    return None


def load_copilot_token_cache(path: Path | None = None) -> CopilotTokenCache | None:
    path = path or COPILOT_TOKEN_CACHE_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    token = str(data.get("token", "")).strip()
    expires = int(data.get("expiresAt", 0) or 0)
    updated = int(data.get("updatedAt", 0) or 0)
    if not token or expires <= 0:
        return None
    return CopilotTokenCache(token=token, expires_at_ms=expires, updated_at_ms=updated)


def save_copilot_token_cache(
    token: str,
    *,
    expires_at_ms: int,
    path: Path | None = None,
) -> None:
    path = path or COPILOT_TOKEN_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "token": token,
        "expiresAt": int(expires_at_ms),
        "updatedAt": int(time.time() * 1000),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    path.chmod(0o600)
