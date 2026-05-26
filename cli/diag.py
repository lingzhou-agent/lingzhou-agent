"""cli/diag.py — version / doctor 命令（诊断与版本信息）。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from cli._common import console, load_cfg, PROJECT_ROOT, resolve_config_path, DEFAULT_CONFIG_PATH
from store.auth import (
    AUTH_PROFILES_PATH,
    get_auth_profile,
    load_legacy_credentials,
    resolve_copilot_token,
)


_ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _mask_token(token: str) -> str:
    if len(token) <= 12:
        return "*" * len(token)
    return f"{token[:6]}...{token[-3:]}"


def _resolve_openai_provider_api_key(provider: object) -> tuple[str | None, str | None]:
    api_key_ref = str(getattr(provider, "api_key_env", "") or "").strip()
    if not api_key_ref:
        return None, None
    if not _ENV_VAR_NAME_RE.fullmatch(api_key_ref):
        return api_key_ref, "literal"

    env_token = os.environ.get(api_key_ref, "").strip()
    if env_token:
        return env_token, f"env:{api_key_ref}"

    legacy = load_legacy_credentials()
    legacy_token = str(legacy.get(api_key_ref, "")).strip()
    if legacy_token:
        return legacy_token, f"legacy-credentials:{api_key_ref}"

    profile_id = str(getattr(provider, "auth_profile_id", "") or "").strip()
    if profile_id:
        profile = get_auth_profile(profile_id)
        if isinstance(profile, dict):
            profile_token = str(profile.get("token", "")).strip()
            if profile_token:
                return profile_token, f"auth-profile:{profile_id}"

    return None, None


def version() -> None:
    """显示版本信息。"""
    import sys
    from core.version import __version__, __codename__, __min_python__
    console.print(f"[bold]lingzhou[/bold] v{__version__}  代号: {__codename__}")
    console.print(f"  Python {sys.version.split()[0]}  (要求 ≥ {'.'.join(str(x) for x in __min_python__)})")


def doctor(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
) -> None:
    """自检：诊断运行环境、配置、API key 和数据库状态。"""
    import sys
    import json as _json
    import importlib
    from core.version import __version__, __min_python__

    ok_mark  = "[bold green]✓[/bold green]"
    fail_mark = "[bold red]✗[/bold red]"
    warn_mark = "[bold yellow]![/bold yellow]"
    issues: list[str] = []

    console.print(Panel(
        f"[bold]lingzhou doctor[/bold]  v{__version__}",
        border_style="cyan",
    ))

    # ── 1. Python 版本 ─────────────────────────────────────────────────
    py = sys.version_info
    py_str = f"{py.major}.{py.minor}.{py.micro}"
    if py[:3] >= __min_python__:
        console.print(f"  {ok_mark} Python {py_str}")
    else:
        need = ".".join(str(x) for x in __min_python__)
        console.print(f"  {fail_mark} Python {py_str}  (需要 ≥ {need})")
        issues.append(f"Python 版本过低: {py_str}")

    # ── 2. 必要依赖 ────────────────────────────────────────────────────
    _DEPS = ["pydantic", "httpx", "aiosqlite", "typer", "rich"]
    for dep in _DEPS:
        try:
            importlib.import_module(dep)
            console.print(f"  {ok_mark} {dep}")
        except ImportError:
            console.print(f"  {fail_mark} {dep}  [dim]未安装[/dim]")
            issues.append(f"缺少依赖: {dep}")

    resolved_config = resolve_config_path(config)

    # ── 3. 配置文件 ────────────────────────────────────────────────────
    if resolved_config.exists():
        try:
            _json.loads(resolved_config.read_text(encoding="utf-8"))
            console.print(f"  {ok_mark} 配置文件: {resolved_config}")
        except Exception as e:
            console.print(f"  {fail_mark} 配置文件解析失败: {e}")
            issues.append(f"配置文件无效: {e}")
    else:
        console.print(f"  {warn_mark} 配置文件不存在: {resolved_config}  [dim]运行 lingzhou onboard[/dim]")
        issues.append(f"配置文件缺失: {resolved_config}")

    # ── 4. API Key ──────────────────────────────────────────────────────
    try:
        cfg = load_cfg(config) if resolved_config.exists() else None
    except Exception:
        cfg = None

    if cfg is not None:
        _api_key_env: str | None = None
        _provider_name: str | None = None
        try:
            _provider_name = cfg.active_provider_name
            _api_key_env = cfg.active_provider.api_key_env
        except Exception:
            pass

        if _api_key_env:
            if getattr(cfg.active_provider, 'mode', '') == 'copilot':
                resolved = resolve_copilot_token(_api_key_env)
                if resolved:
                    if resolved.source.startswith('env:'):
                        env_name = resolved.source.split(':', 1)[1]
                        masked = _mask_token(resolved.token)
                        console.print(f"  {ok_mark} Copilot token ({env_name}): {masked}")
                    elif resolved.source == 'auth-profile':
                        console.print(f"  {ok_mark} Copilot token: 来自 auth profile store  [dim]{AUTH_PROFILES_PATH}[/dim]")
                    else:
                        console.print(f"  {ok_mark} Copilot token: 来自 legacy credentials 文件")
                else:
                    console.print(f"  {fail_mark} Copilot token: 未设置")
                    issues.append("Copilot token 未配置: lingzhou auth login-copilot")
            else:
                _resolved_key, _resolved_source = _resolve_openai_provider_api_key(cfg.active_provider)
                if _resolved_key:
                    if _resolved_source == "literal":
                        console.print(f"  {ok_mark} API key (literal): {_mask_token(_resolved_key)}")
                    elif _resolved_source and _resolved_source.startswith("env:"):
                        env_name = _resolved_source.split(":", 1)[1]
                        console.print(f"  {ok_mark} API key ({env_name}): {_mask_token(_resolved_key)}")
                    elif _resolved_source and _resolved_source.startswith("legacy-credentials:"):
                        env_name = _resolved_source.split(":", 1)[1]
                        console.print(f"  {ok_mark} API key ({env_name}): 来自 credentials 文件")
                    elif _resolved_source and _resolved_source.startswith("auth-profile:"):
                        profile_id = _resolved_source.split(":", 1)[1]
                        console.print(f"  {ok_mark} API key ({_api_key_env}): 来自 auth profile  [dim]{profile_id}[/dim]")
                    else:
                        console.print(f"  {ok_mark} API key ({_api_key_env}): 已解析")
                else:
                    console.print(f"  {fail_mark} API key ({_api_key_env}): 未设置")
                    if _ENV_VAR_NAME_RE.fullmatch(_api_key_env):
                        _provider_hint = _provider_name or "<provider>"
                        issues.append(
                            f"API key 未配置: export {_api_key_env}=your_key 或 lingzhou auth set-token --provider {_provider_hint}"
                        )
                    else:
                        issues.append("API key 未配置: lingzhou auth set-token --provider <provider>")
        else:
            console.print(f"  {warn_mark} API key: 跳过（配置文件不可用）")

    # ── 5. 数据库 ──────────────────────────────────────────────────────
    if cfg is not None:
        db_path = cfg.db_path
        if db_path.exists():
            try:
                from store.task.ingress import IngressStore
                tables = IngressStore(db_path).list_tables()
                console.print(f"  {ok_mark} 数据库: {db_path}  [dim]表: {', '.join(tables) or '(空)'}[/dim]")
            except Exception as e:
                console.print(f"  {fail_mark} 数据库异常: {e}")
                issues.append(f"DB 异常: {e}")
        else:
            console.print(f"  {warn_mark} 数据库未初始化: {db_path}  [dim]运行 lingzhou onboard[/dim]")
    else:
        console.print(f"  {warn_mark} 数据库: 跳过（配置文件不可用）")

    # ── 6 & 7. Config schema 兼容性（ThresholdsConfig + MemoryConfig）──────────
    # 复用 core/loop/startup.py 的 patch 定义与逻辑，与运行时启动保持一致。
    try:
        from core.loop.startup import (
            _THRESHOLDS_FIELD_PATCHES,
            _MEMORY_FIELD_PATCHES,
            _patch_config_classes,
        )
        _config_py = PROJECT_ROOT / "core" / "config.py"
        _all_patched = _patch_config_classes(_config_py, {
            "ThresholdsConfig": _THRESHOLDS_FIELD_PATCHES,
            "MemoryConfig":     _MEMORY_FIELD_PATCHES,
        })
        for _cls_name, _patches in [
            ("ThresholdsConfig", _THRESHOLDS_FIELD_PATCHES),
            ("MemoryConfig",     _MEMORY_FIELD_PATCHES),
        ]:
            _injected = _all_patched.get(_cls_name)
            if _injected:
                console.print(f"  {warn_mark} {_cls_name} 缺少字段，已自动注入: {_injected}")
            else:
                # 注入未执行（字段已存在或注入失败），验证实际字段
                try:
                    import importlib as _il
                    _mod = _il.import_module("core.config")
                    _cls = getattr(_mod, _cls_name)
                    _inst = _cls()
                    _still_missing = [f for f in _patches if not hasattr(_inst, f)]
                    if _still_missing:
                        console.print(f"  {fail_mark} {_cls_name} 缺少字段: {_still_missing}  [dim](自动注入失败，请手动 git pull)[/dim]")
                        issues.append(f"core/config.py 版本过旧，{_cls_name} 缺少: {_still_missing}")
                    else:
                        console.print(f"  {ok_mark} {_cls_name} schema 兼容  [dim]({len(_patches)} 个关键字段均存在)[/dim]")
                except Exception as _e:
                    console.print(f"  {fail_mark} {_cls_name} schema 检查失败: {_e}")
                    issues.append(f"{_cls_name} 无法导入: {_e}")
    except Exception as e:
        console.print(f"  {fail_mark} Config schema 检查失败: {e}")
        issues.append(f"Config schema 检查异常: {e}")

    # ── 8. 目录读写权限 ────────────────────────────────────────────────
    if cfg is not None:
        _dirs_to_check = [
            ("memory_dir", cfg.memory_dir),
            ("workspace_dir", cfg.workspace_dir),
            ("db_parent", Path(cfg.db_path).parent),
        ]
        for _label, _dpath in _dirs_to_check:
            _dpath = Path(_dpath).expanduser()
            if not _dpath.exists():
                try:
                    _dpath.mkdir(parents=True, exist_ok=True)
                    console.print(f"  {warn_mark} {_label}: {_dpath}  [dim]不存在，已创建[/dim]")
                except Exception as _e:
                    console.print(f"  {fail_mark} {_label}: {_dpath}  无法创建: {_e}")
                    issues.append(f"{_label} 无法创建: {_e}")
            else:
                import tempfile
                try:
                    with tempfile.NamedTemporaryFile(dir=_dpath, delete=True):
                        pass
                    console.print(f"  {ok_mark} {_label}: {_dpath}  [dim]可写[/dim]")
                except Exception:
                    console.print(f"  {fail_mark} {_label}: {_dpath}  [dim]无写权限[/dim]")
                    issues.append(f"{_label} 目录无写权限: {_dpath}")
    else:
        console.print(f"  {warn_mark} 目录权限: 跳过（配置文件不可用）")

    # ── 9. DB schema 完整性 ────────────────────────────────────────────
    _REQUIRED_TABLES = {
        "tasks",
        "failures",
        "facts",
        "signals",
        "chat_messages",
        "runs",
        "meta_reflections",
    }
    if cfg is not None and Path(cfg.db_path).exists():
        try:
            from store.task.ingress import IngressStore
            _tables = set(IngressStore(cfg.db_path).list_tables())
            _missing_tables = _REQUIRED_TABLES - _tables
            if _missing_tables:
                console.print(f"  {warn_mark} DB schema: 缺少表 {sorted(_missing_tables)}  [dim]lingzhou run 首次启动时会自动建表[/dim]")
            else:
                console.print(f"  {ok_mark} DB schema: 关键表均存在  [dim]{sorted(_REQUIRED_TABLES)}[/dim]")
        except Exception as _e:
            console.print(f"  {fail_mark} DB schema 检查失败: {_e}")
            issues.append(f"DB schema 异常: {_e}")
    else:
        console.print(f"  {warn_mark} DB schema: 跳过（数据库未初始化）")

    # ── 10. 插件状态 ───────────────────────────────────────────────────
    try:
        from core.plugin import PluginManager
        _plugins_dir = PROJECT_ROOT / "plugins"
        _pm = PluginManager(Path(_plugins_dir).expanduser())
        _pm.discover()
        _loaded = _pm.list_plugins()
        if _loaded:
            console.print(f"  {ok_mark} 插件: {len(_loaded)} 个已发现  [dim]{', '.join(p['name'] for p in _loaded[:4])}{'...' if len(_loaded) > 4 else ''}[/dim]")
        else:
            console.print(f"  {ok_mark} 插件: 无已安装插件  [dim](plugins/ 目录为空)[/dim]")
    except Exception as _e:
        console.print(f"  {warn_mark} 插件检查失败: {_e}")

    # ── 11. 工具注册 ───────────────────────────────────────────────────
    try:
        from tools.registry import ToolRegistry
        reg = ToolRegistry()
        tools_dir = PROJECT_ROOT / "tools"
        reg.discover(tools_dir)
        manifests = reg.list_manifests()
        tool_ids = [m.name for m in manifests]
        console.print(f"  {ok_mark} 工具注册: {len(tool_ids)} 个  [dim]{', '.join(tool_ids[:6])}{'...' if len(tool_ids) > 6 else ''}[/dim]")
    except Exception as e:
        console.print(f"  {fail_mark} 工具注册失败: {e}")
        issues.append(f"工具注册异常: {e}")

    # ── 12. 模型连通性探针 ─────────────────────────────────────────────
    if cfg is not None:
        try:
            import time as _time
            import httpx as _httpx

            _prov = cfg.active_provider
            _mode = getattr(_prov, "mode", "openai")
            _base = _prov.base_url.rstrip("/")
            _model_id = cfg.active_model_id

            # 解析 API key
            _ping_key: str | None = None
            if _mode == "copilot":
                _tok = resolve_copilot_token(_prov.api_key_env)
                _ping_key = _tok.token if _tok else None
            else:
                _ping_key, _ = _resolve_openai_provider_api_key(_prov)

            if not _ping_key:
                console.print(f"  {warn_mark} 模型探针: 跳过（无 API key）")
            else:
                _headers = {
                    "Authorization": f"Bearer {_ping_key}",
                    "Content-Type": "application/json",
                }
                _payload = {
                    "model": _model_id,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                }
                _t0 = _time.monotonic()
                try:
                    with _httpx.Client(timeout=8.0) as _hc:
                        _resp = _hc.post(f"{_base}/chat/completions", headers=_headers, json=_payload)
                    _ms = int((_time.monotonic() - _t0) * 1000)
                    if _resp.status_code in (200, 201):
                        console.print(f"  {ok_mark} 模型探针: {_model_id} 响应 {_ms}ms  [dim](HTTP {_resp.status_code})[/dim]")
                    elif _resp.status_code in (401, 403):
                        console.print(f"  {fail_mark} 模型探针: {_model_id} 认证失败 (HTTP {_resp.status_code})")
                        issues.append(f"API key 认证失败: HTTP {_resp.status_code}")
                    else:
                        console.print(f"  {warn_mark} 模型探针: {_model_id} HTTP {_resp.status_code} ({_ms}ms)")
                except _httpx.TimeoutException:
                    _ms = int((_time.monotonic() - _t0) * 1000)
                    console.print(f"  {warn_mark} 模型探针: {_model_id} 超时 ({_ms}ms)  [dim]网络或服务可能不可达[/dim]")
                except Exception as _pe:
                    console.print(f"  {warn_mark} 模型探针: 请求失败: {_pe}")
        except Exception as _e:
            console.print(f"  {warn_mark} 模型探针: 跳过 ({_e})")
    else:
        console.print(f"  {warn_mark} 模型探针: 跳过（配置文件不可用）")

    # ── 汇总 ────────────────────────────────────────────────────────────
    console.print("")
    if not issues:
        console.print(f"[bold green]所有检查通过。[/bold green] 可以运行 [bold]lingzhou run[/bold]")
    else:
        console.print(f"[bold red]发现 {len(issues)} 个问题：[/bold red]")
        for i, issue in enumerate(issues, 1):
            console.print(f"  {i}. {issue}")
        raise typer.Exit(1)
