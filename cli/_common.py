"""cli/_common.py — CLI 公共工具：console、_load_cfg、PROJECT_ROOT。"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from core.config import Config

# cli/ 的上一级就是项目根目录
PROJECT_ROOT: Path = Path(__file__).parent.parent

console = Console()


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
        state_cfg = Path("~/.lingzhou/lingzhou.json").expanduser()
        if state_cfg.exists():
            return state_cfg
    return candidate


def load_cfg(config: Path) -> "Config":
    from core.config import Config
    return Config.load(resolve_config_path(config))
