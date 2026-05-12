"""cli/diag.py — version / doctor 命令（诊断与版本信息）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from cli._common import console, load_cfg, PROJECT_ROOT, resolve_config_path
from auth_store import resolve_copilot_token, AUTH_PROFILES_PATH


def version() -> None:
    """显示版本信息。"""
    import sys
    from core.version import __version__, __codename__, __min_python__
    console.print(f"[bold]lingzhou[/bold] v{__version__}  代号: {__codename__}")
    console.print(f"  Python {sys.version.split()[0]}  (要求 ≥ {'.'.join(str(x) for x in __min_python__)})")


def doctor(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("lingzhou.json"),
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
        console.print(f"  {warn_mark} 配置文件不存在: {resolved_config}  [dim]运行 lingzhou setup 生成[/dim]")
        issues.append(f"配置文件缺失: {resolved_config}")

    # ── 4. API Key ──────────────────────────────────────────────────────
    try:
        cfg = load_cfg(config) if resolved_config.exists() else None
    except Exception:
        cfg = None

    if cfg is not None:
        _api_key_env: str | None = None
        try:
            pname = cfg.model.split("/")[0] if "/" in cfg.model else None
            if pname and pname in cfg.providers:
                _api_key_env = cfg.providers[pname].api_key_env
        except Exception:
            pass

        if _api_key_env:
            if getattr(cfg.active_provider, 'mode', '') == 'copilot':
                resolved = resolve_copilot_token(_api_key_env)
                if resolved:
                    if resolved.source.startswith('env:'):
                        env_name = resolved.source.split(':', 1)[1]
                        masked = (resolved.token[:6] + '...' + resolved.token[-3:])
                        console.print(f"  {ok_mark} Copilot token ({env_name}): {masked}")
                    elif resolved.source == 'auth-profile':
                        console.print(f"  {ok_mark} Copilot token: 来自 auth profile store  [dim]{AUTH_PROFILES_PATH}[/dim]")
                    else:
                        console.print(f"  {ok_mark} Copilot token: 来自 legacy credentials 文件")
                else:
                    console.print(f"  {fail_mark} Copilot token: 未设置")
                    issues.append("Copilot token 未配置: lingzhou auth login-copilot")
            elif os.environ.get(_api_key_env):
                masked = (os.environ[_api_key_env][:6] + "..." + os.environ[_api_key_env][-3:])
                console.print(f"  {ok_mark} API key ({_api_key_env}): {masked}")
            else:
                # 检查 credentials 文件
                cred = Path("~/.lingzhou/credentials.json").expanduser()
                if cred.exists():
                    try:
                        saved = _json.loads(cred.read_text(encoding="utf-8"))
                        if saved.get(_api_key_env):
                            console.print(f"  {ok_mark} API key ({_api_key_env}): 来自 credentials 文件")
                        else:
                            console.print(f"  {fail_mark} API key ({_api_key_env}): 未设置")
                            issues.append(f"API key 未配置: export {_api_key_env}=your_key")
                    except Exception:
                        console.print(f"  {warn_mark} API key ({_api_key_env}): credentials 文件读取失败")
                else:
                    console.print(f"  {fail_mark} API key ({_api_key_env}): 未设置")
                    issues.append(f"API key 未配置: export {_api_key_env}=your_key")
        else:
            console.print(f"  {warn_mark} API key: 跳过（配置文件不可用）")

    # ── 5. 数据库 ──────────────────────────────────────────────────────
    if cfg is not None:
        db_path = cfg.db_path
        if db_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                try:
                    tables = [r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()]
                finally:
                    conn.close()
                console.print(f"  {ok_mark} 数据库: {db_path}  [dim]表: {', '.join(tables) or '(空)'}[/dim]")
            except Exception as e:
                console.print(f"  {fail_mark} 数据库异常: {e}")
                issues.append(f"DB 异常: {e}")
        else:
            console.print(f"  {warn_mark} 数据库未初始化: {db_path}  [dim]运行 lingzhou init[/dim]")
    else:
        console.print(f"  {warn_mark} 数据库: 跳过（配置文件不可用）")

    # ── 6. 工具注册 ────────────────────────────────────────────────────
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

    # ── 汇总 ────────────────────────────────────────────────────────────
    console.print("")
    if not issues:
        console.print(f"[bold green]所有检查通过。[/bold green] 可以运行 [bold]lingzhou run[/bold]")
    else:
        console.print(f"[bold red]发现 {len(issues)} 个问题：[/bold red]")
        for i, issue in enumerate(issues, 1):
            console.print(f"  {i}. {issue}")
        raise typer.Exit(1)
