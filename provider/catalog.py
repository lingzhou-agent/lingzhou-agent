"""provider/catalog.py — 模型目录查询（运行时路径优先）。

设计思路：
  - 内置模板（provider/models.json）：随源码发布，记录 context_window / max_tokens /
    thinking 等静态参数，作为种子文件。
  - 运行时文件（workspace_dir/models.json）：由 provider.models_gen.ensure_models_json()
    在每次启动时生成；融入 lingzhou.json 中的 provider 连接参数，用指纹
    决定 skip / noop / write。
  - catalog 函数通过 set_runtime_path() 切换到运行时路径，lru_cache 同步清除。
"""
from __future__ import annotations

import json
import functools
from pathlib import Path
from typing import Any

# 内置模板路径（随包发布，只读种子）
BUILTIN_CATALOG_PATH: Path = Path(__file__).parent / "models.json"

# 运行时路径（由 SoulManager 在 init_files() 后通过 set_runtime_path() 注入）
_runtime_path: Path = BUILTIN_CATALOG_PATH


def set_runtime_path(path: Path) -> None:
    """将 catalog 切换到运行时路径（workspace_dir/models.json）。

    必须在读取任何 catalog 函数之前调用（通常在 SoulManager.init_files() 末尾）。
    调用后自动清除 lru_cache，下次查询时重新从新路径加载。
    """
    global _runtime_path
    if _runtime_path == path:
        return
    _runtime_path = path
    _load.cache_clear()


@functools.lru_cache(maxsize=1)
def _load() -> dict[str, dict[str, Any]]:
    """读取当前 catalog 文件（运行时路径优先，回退内置模板）。"""
    src = _runtime_path if _runtime_path.exists() else BUILTIN_CATALOG_PATH
    raw: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def lookup_model(model_id: str) -> dict[str, Any] | None:
    """在所有 provider 里查找 model_id，返回模型元数据字典；未找到返回 None。"""
    for provider_data in _load().values():
        for m in provider_data.get("models", []):
            if m.get("id") == model_id:
                return m
    return None


def list_providers() -> list[str]:
    """返回 models.json 中所有 provider 名称（过滤掉 _doc 等非 provider 键）。"""
    return [k for k, v in _load().items() if "models" in v]


def list_provider_models(provider_name: str) -> list[dict[str, Any]]:
    """返回指定 provider 的模型列表； provider 不存在时返回空列表。"""
    return _load().get(provider_name, {}).get("models", [])


def is_reasoning_model(model_id: str) -> bool:
    """返回该模型是否具备推理能力（reasoning=true 元数据标记）。

    copilot 的 gpt-5.4 / o3 等模型由顶层 thinking 自动映射到 reasoning_effort，
    bailian 的 Qwen3 模型通过 thinking.budget_tokens 控制。
    两套机制不同，此函数仅供展示/路由层判断，不影响 payload 构造逻辑。
    """
    m = lookup_model(model_id)
    return bool(m and m.get("reasoning"))


def resolve_context_window(model_id: str, override: int | None) -> int | None:
    """返回模型的上下文窗口大小（tokens）。

    优先级：
      1. lingzhou.json 中的 context_window_tokens（显式覆盖）
      2. models.json 内置目录中的 context_window
      3. None（由调用方决定 fallback）
    """
    if override is not None:
        return override
    m = lookup_model(model_id)
    if m:
        return m.get("context_window")
    return None
