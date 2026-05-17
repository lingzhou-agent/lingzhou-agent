"""core/paths.py — 灵舟路径工具。统一 workspace 和 data 目录引用。"""

from pathlib import Path

_DATA_DIR = Path("~/.lingzhou").expanduser()
_PROJECT_DIR = Path(__file__).resolve().parent.parent
_GENERATED_DIR = _PROJECT_DIR / "generated"


def project_root() -> Path:
    """项目根目录 /root/lingzhou。"""
    return _PROJECT_DIR


def data_dir() -> Path:
    """数据目录 ~/.lingzhou（logs/state/gateway）。"""
    return _DATA_DIR


def generated_dir() -> Path:
    """生成文件目录 /root/lingzhou/generated。"""
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    return _GENERATED_DIR
