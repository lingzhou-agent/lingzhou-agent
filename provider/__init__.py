"""provider/__init__.py — Provider 工厂。
新增 provider 类型：在此处 match 分支里注册即可，其余代码零改动。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from provider.base import Provider
from provider.openai_compat import OpenAICompatProvider

if TYPE_CHECKING:
    from core.config import Config


def create_provider(cfg: "Config") -> Provider:
    provider_def = cfg.active_provider
    match provider_def.type:
        case "openai_compat":
            return OpenAICompatProvider(cfg)
        case _:
            raise ValueError(
                f"未知 provider 类型: {provider_def.type!r}。"
                f"已支持: openai_compat"
            )


def create_provider_with_model(cfg: "Config", model_ref: str) -> Provider:
    """用指定 model_ref 替换 cfg.model 创建 provider（routing 路由用）。"""
    routing_cfg = cfg.model_copy(update={"model": model_ref})
    routing_cfg._base_dir = cfg._base_dir
    p = create_provider(routing_cfg)
    # 存储完整 model_ref，供 _find_or_create_provider 缓存比对（p._model 只存短 ID）
    p._model_ref = model_ref  # type: ignore[attr-defined]
    return p
