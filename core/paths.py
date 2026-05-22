"""core/paths.py — 灵舟路径工具。统一 workspace 和 data 目录引用。"""

import os
from pathlib import Path

_DATA_DIR = Path("~/.lingzhou").expanduser()
_PROJECT_DIR = Path(__file__).resolve().parent.parent
_GENERATED_DIR = _DATA_DIR / "generated"


def _ensure_dir_writable(p: Path, name: str) -> None:
    """确保目录存在且可写，否则抛出明确异常。"""
    if not p.exists():
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"无法创建{name}目录 {p}: {e}") from e
    if not os.access(p, os.W_OK):
        raise RuntimeError(f"{name}目录 {p} 不可写")


def project_root() -> Path:
    """项目根目录。"""
    if not _PROJECT_DIR.exists():
        raise RuntimeError(f"项目根目录不存在: {_PROJECT_DIR}")
    return _PROJECT_DIR


def data_dir() -> Path:
    """数据目录 ~/.lingzhou。"""
    _ensure_dir_writable(_DATA_DIR, "数据")
    return _DATA_DIR


def generated_dir() -> Path:
    """生成文件目录 ~/.lingzhou/generated（运行期数据目录）。"""
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    return _GENERATED_DIR
