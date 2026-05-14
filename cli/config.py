"""cli/config.py — config get / config set 命令组。"""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Annotated, Any

import typer

from cli._common import console, resolve_config_path, DEFAULT_CONFIG_PATH

config_app = typer.Typer(name="config", help="配置文件管理", no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})


@config_app.command("get")
def config_get(
    key: Annotated[str, typer.Argument(help="配置键（支持点号路径，如 loop.debug）")],
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
) -> None:
    """读取配置文件中某个键的值。"""
    config = resolve_config_path(config)
    if not config.exists():
        console.print(f"[red]配置文件不存在: {config}[/red]")
        raise typer.Exit(1)

    cfg_data = _json.loads(config.read_text(encoding="utf-8"))
    parts = key.split(".")
    val = cfg_data
    try:
        for p in parts:
            val = val[p]
        console.print(f"{key} = [cyan]{_json.dumps(val, ensure_ascii=False)}[/cyan]")
    except (KeyError, TypeError):
        console.print(f"[yellow]键不存在: {key}[/yellow]")
        raise typer.Exit(1)


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="配置键（支持点号路径，如 loop.debug）")],
    value: Annotated[str, typer.Argument(help="新值（true/false/数字/字符串自动推断类型）")],
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
) -> None:
    """修改配置文件中某个键的值（支持点号嵌套路径）。"""
    config = resolve_config_path(config)
    if not config.exists():
        console.print(f"[red]配置文件不存在: {config}[/red]")
        raise typer.Exit(1)

    cfg_data = _json.loads(config.read_text(encoding="utf-8"))

    # 类型推断
    parsed: Any
    if value.lower() == "true":
        parsed = True
    elif value.lower() == "false":
        parsed = False
    else:
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value

    parts = key.split(".")
    node = cfg_data
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    old = node.get(parts[-1], "(未设置)")
    node[parts[-1]] = parsed

    config.write_text(_json.dumps(cfg_data, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(
        f"[green]✓ {key}[/green]: "
        f"{_json.dumps(old, ensure_ascii=False)} → "
        f"[cyan]{_json.dumps(parsed, ensure_ascii=False)}[/cyan]"
    )
