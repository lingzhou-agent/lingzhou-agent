"""cli/diag.py — version / doctor 命令（诊断与版本信息）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from cli._common import console, load_cfg, PROJECT_ROOT, resolve_config_path, DEFAULT_CONFIG_PATH
from store.auth import resolve_copilot_token, AUTH_PROFILES_PATH


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

    # ── 6. 配置 schema 兼容性（类比 openclaw _reconcile_columns）──────────
    # 检测 ThresholdsConfig 是否包含 runtime 依赖的关键字段；缺失时自动注入默认定义。
    _THRESHOLDS_FIELD_PATCHES: dict[str, str] = {
        "skill_max_inject": (
            "    skill_max_inject: int = Field(\n"
            "        default=3, ge=1, le=8,\n"
            "        description=\"单次 tick 最多注入技能数；压力大时可通过配置增加护栏覆盖\"\n"
            "    )\n"
        ),
        "skill_failure_threshold": (
            "    skill_failure_threshold: int = Field(\n"
            "        default=3, ge=1,\n"
            "        description=\"连续评分函数的失败次数基准点；达到此值时 failure.reflection 技能得分达到峰值\"\n"
            "    )\n"
        ),
        "skill_wm_pressure_threshold": (
            "    skill_wm_pressure_threshold: float = Field(\n"
            "        default=0.4, ge=0.0, le=1.0,\n"
            "        description=\"WM 压力连续评分基准点；达到此值时 evidence-first-change 技能得分达到峰值\"\n"
            "    )\n"
        ),
        "skill_min_budget_tokens": (
            "    skill_min_budget_tokens: int = Field(\n"
            "        default=80, ge=0,\n"
            "        description=\"上下文预算裁剪时 skills_section 保留的最小 token 数；0=可完全裁掉\"\n"
            "    )\n"
        ),
    }
    try:
        from core.config import ThresholdsConfig as _ThresholdsConfig
        _t_instance = _ThresholdsConfig()
        _missing_fields = [f for f in _THRESHOLDS_FIELD_PATCHES if not hasattr(_t_instance, f)]
        if _missing_fields:
            # 自动注入缺失字段到 core/config.py（类比 ALTER TABLE ADD COLUMN）
            _config_py = PROJECT_ROOT / "core" / "config.py"
            _patched: list[str] = []
            if _config_py.exists():
                _lines = _config_py.read_text(encoding="utf-8").splitlines(keepends=True)
                _in_thresholds = False
                _insert_at = len(_lines)
                for _i, _line in enumerate(_lines):
                    if _line.startswith("class ThresholdsConfig"):
                        _in_thresholds = True
                    elif _in_thresholds and _line.startswith("class "):
                        _insert_at = _i
                        break
                if _in_thresholds:
                    # 删除末尾空行，保持整洁
                    while _insert_at > 0 and not _lines[_insert_at - 1].strip():
                        _insert_at -= 1
                    _inject = ["\n"]
                    for _f in _missing_fields:
                        if _f in _THRESHOLDS_FIELD_PATCHES:
                            _inject.append(_THRESHOLDS_FIELD_PATCHES[_f])
                            _patched.append(_f)
                    if _patched:
                        _lines[_insert_at:_insert_at] = _inject
                        _config_py.write_text("".join(_lines), encoding="utf-8")
            if _patched:
                console.print(f"  {warn_mark} ThresholdsConfig 缺少字段，已自动注入: {_patched}")
            else:
                console.print(f"  {fail_mark} ThresholdsConfig 缺少字段: {_missing_fields}  [dim](自动注入失败，请手动 git pull)[/dim]")
                issues.append(f"core/config.py 版本过旧，缺少: {_missing_fields}")
        else:
            console.print(f"  {ok_mark} ThresholdsConfig schema 兼容  [dim]({len(_THRESHOLDS_FIELD_PATCHES)} 个关键字段均存在)[/dim]")
    except Exception as e:
        console.print(f"  {fail_mark} ThresholdsConfig schema 检查失败: {e}")
        issues.append(f"ThresholdsConfig 无法导入: {e}")

    # ── 7. MemoryConfig schema 兼容性（类比 ThresholdsConfig patch）──────
    _MEMORY_FIELD_PATCHES: dict[str, str] = {
        "wm_item_max_tokens": (
            "    wm_item_max_tokens: int = Field(\n"
            "        default=300, ge=0,\n"
            "        description=(\n"
            "            \"工作记忆单条 content token 上限（估算）；超出时自动截断并追加省略提示。\"\n"
            "            \"0 = 不限制。调优请在 lingzhou.json 的 memory 区块覆盖，不要修改此处 default 值。\"\n"
            "        ),\n"
            "    )\n"
        ),
    }
    try:
        from core.config import MemoryConfig as _MemoryConfig
        _m_instance = _MemoryConfig()
        _missing_mem = [f for f in _MEMORY_FIELD_PATCHES if not hasattr(_m_instance, f)]
        if _missing_mem:
            _config_py = PROJECT_ROOT / "core" / "config.py"
            _patched_mem: list[str] = []
            if _config_py.exists():
                _lines = _config_py.read_text(encoding="utf-8").splitlines(keepends=True)
                _in_mem = False
                _insert_at = len(_lines)
                for _i, _line in enumerate(_lines):
                    if _line.startswith("class MemoryConfig"):
                        _in_mem = True
                    elif _in_mem and _line.startswith("class "):
                        _insert_at = _i
                        break
                if _in_mem:
                    while _insert_at > 0 and not _lines[_insert_at - 1].strip():
                        _insert_at -= 1
                    _inject = ["\n"]
                    for _f in _missing_mem:
                        _inject.append(_MEMORY_FIELD_PATCHES[_f])
                        _patched_mem.append(_f)
                    if _patched_mem:
                        _lines[_insert_at:_insert_at] = _inject
                        _config_py.write_text("".join(_lines), encoding="utf-8")
            if _patched_mem:
                console.print(f"  {warn_mark} MemoryConfig 缺少字段，已自动注入: {_patched_mem}")
            else:
                console.print(f"  {fail_mark} MemoryConfig 缺少字段: {_missing_mem}  [dim](自动注入失败，请手动 git pull)[/dim]")
                issues.append(f"core/config.py 版本过旧，MemoryConfig 缺少: {_missing_mem}")
        else:
            console.print(f"  {ok_mark} MemoryConfig schema 兼容  [dim]({len(_MEMORY_FIELD_PATCHES)} 个关键字段均存在)[/dim]")
    except Exception as e:
        console.print(f"  {fail_mark} MemoryConfig schema 检查失败: {e}")
        issues.append(f"MemoryConfig 无法导入: {e}")

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
    _REQUIRED_TABLES = {"tasks", "facts", "failures", "task_events", "runs"}
    if cfg is not None and Path(cfg.db_path).exists():
        try:
            import sqlite3 as _sqlite3
            _conn = _sqlite3.connect(str(cfg.db_path))
            try:
                _tables = {r[0] for r in _conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}
            finally:
                _conn.close()
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

    # ── 汇总 ────────────────────────────────────────────────────────────
    console.print("")
    if not issues:
        console.print(f"[bold green]所有检查通过。[/bold green] 可以运行 [bold]lingzhou run[/bold]")
    else:
        console.print(f"[bold red]发现 {len(issues)} 个问题：[/bold red]")
        for i, issue in enumerate(issues, 1):
            console.print(f"  {i}. {issue}")
        raise typer.Exit(1)
