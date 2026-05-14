"""cli/_common.py — CLI 公共工具：console、_load_cfg、PROJECT_ROOT。"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

if TYPE_CHECKING:
    from core.config import Config

# cli/ 的上一级就是项目根目录
PROJECT_ROOT: Path = Path(__file__).parent.parent

console = Console()

DEFAULT_CONFIG_PATH: Path = Path.home() / ".lingzhou" / "lingzhou.json"

# 全局配置文件搜索顺序（当 --config 指向的路径不存在时依次尝试）
# 用户目录优先于当前工作目录，避免源码目录的 lingzhou.json 被误加载。
_CONFIG_SEARCH_PATHS: list[Path] = [
    DEFAULT_CONFIG_PATH,
    Path.home() / ".config" / "lingzhou" / "lingzhou.json",
    Path("lingzhou.json"),
]


def find_config(hint: Path) -> Path:
    """返回可用的配置文件路径；若 hint 存在则直接使用，否则按预设顺序搜索。
    若均不存在，自动启动 setup 向导，向导完成后返回生成的路径。
    默认配置归属为 ~/.lingzhou/lingzhou.json，而不是仓库根目录。
    """
    if hint.exists():
        return hint
    for candidate in _CONFIG_SEARCH_PATHS:
        if candidate.exists():
            return candidate
    # 未找到配置 — 自动引导 setup
    console.print("[yellow]未找到 lingzhou.json，自动启动初始化向导…[/yellow]\n")
    from cli.bootstrap import setup as _setup
    default_out = DEFAULT_CONFIG_PATH
    _setup(output=default_out)
    if default_out.exists():
        return default_out
    # 兜底：再搜一遍
    for candidate in _CONFIG_SEARCH_PATHS:
        if candidate.exists():
            return candidate
    raise typer.Exit(1)


def resolve_config_path(config: Path) -> Path:
    """解析 CLI 默认配置路径。

    规则：
    1. 显式存在的路径直接使用
    2. 若用户传的是默认名 `lingzhou.json` 且当前目录不存在，回退到 `~/.lingzhou/lingzhou.json`
    3. 否则保持原样，由上层给出缺失错误
    """
    candidate = config.expanduser()
    if candidate.exists():
        return candidate
    if candidate.name == "lingzhou.json" and not candidate.is_absolute():
        state_cfg = DEFAULT_CONFIG_PATH
        if state_cfg.exists():
            return state_cfg
    return candidate


def load_cfg(config: Path) -> "Config":
    from core.config import Config
    return Config.load(find_config(config))
