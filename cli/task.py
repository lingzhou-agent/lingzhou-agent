"""cli/task.py — task 子命令组：add / list。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer

from cli._common import console, load_cfg

task_app = typer.Typer(
    name="task",
    help="任务管理",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@task_app.command("add")
def task_add(
    title: Annotated[str, typer.Argument(help="任务标题")],
    goal: Annotated[str, typer.Option("--goal", "-g", help="任务目标")] = "",
    priority: Annotated[str, typer.Option("--priority", "-p")] = "normal",
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("lingzhou.json"),
) -> None:
    """向任务队列添加一个任务。"""
    cfg = load_cfg(config)

    async def _run() -> None:
        from memory.task_store import TaskStore
        store = TaskStore(cfg.db_path)
        await store.open()
        try:
            task_id = await store.add_task(title, goal, priority, source="external")
        finally:
            await store.close()
        console.print(f"[green]任务已创建: [{task_id}] {title}[/green]")

    asyncio.run(_run())


@task_app.command("list")
def task_list(
    status: Annotated[Optional[str], typer.Option("--status", "-s", help="状态过滤（pending / running / done / failed）")] = None,
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("lingzhou.json"),
) -> None:
    """列出任务。"""
    cfg = load_cfg(config)

    async def _run() -> None:
        from memory.task_store import TaskStore
        store = TaskStore(cfg.db_path)
        await store.open()
        try:
            tasks = await store.list_tasks(status=status, limit=50)
        finally:
            await store.close()
        if not tasks:
            console.print("（没有任务）")
            return
        for t in tasks:
            goal_hint = f"  目标: {t.goal[:60]}" if t.goal else ""
            console.print(f"[{t.id}] [{t.status}] [{t.priority}] {t.title}{goal_hint}")

    asyncio.run(_run())
