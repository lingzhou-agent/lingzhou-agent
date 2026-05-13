"""lingzhou.py — CLI 入口（纯注册层，不含业务逻辑）。"""
from __future__ import annotations

from typing import Annotated, Optional

import typer

from core.version import __version__, __codename__
from cli._common import console

# ── 命令实现导入 ──────────────────────────────────────────────────────────────
from cli.task import task_app
from cli.bootstrap import setup, init
from cli.chat import chat
from cli.dev import dev_app
from cli.auth import auth_app
from cli.config import config_app
from cli.gateway import gateway_app, gateway_start, run, stop


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"lingzhou v{__version__} ({__codename__})")
        raise typer.Exit()


app = typer.Typer(
    name="lingzhou",
    help="自编程自进化认知 agent 种子",
    no_args_is_help=False,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def app_callback(
    ctx: typer.Context,
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True, help="显示版本号"),
    ] = None,
) -> None:
    """自编程自进化认知 agent 种子。不带子命令时直接启动认知循环。"""
    if ctx.invoked_subcommand is None:
        gateway_start(channel="local", daemon=False)


# ── 子命令注册 ────────────────────────────────────────────────────────────────

app.add_typer(auth_app)
app.add_typer(config_app)
app.add_typer(gateway_app)
app.add_typer(task_app)
app.add_typer(dev_app)

app.command()(run)
app.command()(stop)
app.command()(chat)

app.command()(setup)
app.command()(init)


@app.command(name="help", hidden=True)
def _help(ctx: typer.Context) -> None:
    """显示帮助信息（等同于 --help）。"""
    import subprocess, sys
    subprocess.run([sys.argv[0], "--help"])


def _normalize_help_args() -> None:
    """将 -help / --h 等非标准 help 变体规范化为 --help。"""
    import sys
    _HELP_ALIASES = {"-help", "--h", "-?", "/?"}
    sys.argv = ["--help" if a in _HELP_ALIASES else a for a in sys.argv]


def main() -> None:
    """CLI 入口（pyproject.toml 中的 scripts 指向此函数）。"""
    _normalize_help_args()
    app()


if __name__ == "__main__":
    main()
