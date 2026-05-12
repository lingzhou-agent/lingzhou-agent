"""lingzhou.py — CLI 入口（纯注册层，不含业务逻辑）。

所有命令实现位于 cli/ 子包，按领域分模块：
  cli/task.py     — task-add / task-list
  cli/bootstrap.py — setup / init
  cli/interact.py — interact
  cli/dev.py      — evolve / tools / model / update
  cli/diag.py     — version / doctor
  cli/auth.py     — auth copilot
  cli/config.py   — config get / set
  cli/gateway.py  — run / gateway channels|setup|start
"""
from __future__ import annotations

from typing import Annotated, Optional

import typer

from core.version import __version__, __codename__
from cli._common import console

# ── 命令实现导入 ──────────────────────────────────────────────────────────────
from cli.task import task_add, task_list
from cli.bootstrap import setup, init
from cli.interact import interact
from cli.dev import evolve, tools, model, update
from cli.diag import version, doctor
from cli.auth import auth_app
from cli.config import config_app
from cli.gateway import gateway_app, gateway_start, run


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"lingzhou v{__version__} ({__codename__})")
        raise typer.Exit()


app = typer.Typer(
    name="lingzhou",
    help="自编程自进化认知 agent 种子",
    no_args_is_help=False,
    invoke_without_command=True,
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
        gateway_start(channel="local")


# ── 子命令注册 ────────────────────────────────────────────────────────────────

# 分组命令（sub-typer）
app.add_typer(auth_app)
app.add_typer(config_app)
app.add_typer(gateway_app)

# 顶层命令
app.command()(task_add)
app.command()(task_list)
app.command()(setup)
app.command()(init)
app.command()(interact)
app.command()(evolve)
app.command(name="tools")(tools)
app.command()(model)
app.command()(update)
app.command()(version)
app.command()(doctor)
app.command()(run)


if __name__ == "__main__":
    app()
