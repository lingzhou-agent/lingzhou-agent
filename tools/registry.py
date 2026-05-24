"""tools/registry.py — 工具注册系统。

设计原则：
- 工具模块在 import 时自动注册（@registry.tool 装饰器）
- ToolRegistry.discover() 扫描 tools/ 目录自动导入所有工具
- EvolutionEngine 生成新工具后只需写文件 + importlib.reload，无需重启
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Awaitable

if TYPE_CHECKING:
    from core.config import Config
    from memory.working import WorkingMemory
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore
    from core.perception import EmotionState


# ── 数据模型 ───────────────────────────────────────────────────────────────────

# 高频 capabilities 组合常量，避免在每个 @tool 里重复字符串
CAPS_EXEMPT: tuple[str, ...] = ("plan_bootstrap_exempt", "plan_alignment_exempt")
CAPS_RUN_SPAWN: tuple[str, ...] = ("run_spawn",)


def _is_discoverable_tool_file(mod_file: Path) -> bool:
    stem = mod_file.stem
    if mod_file.suffix != ".py":
        return False
    if stem == "registry" or stem.startswith(("_", ".")):
        return False
    return stem.isidentifier()

@dataclass
class ToolParam:
    name: str
    dtype: str                   # "string" | "number" | "boolean" | "object"
    description: str
    required: bool = True


@dataclass
class ToolManifest:
    name: str                   # 唯一 ID，如 "shell.run"
    description: str
    params: list[ToolParam] = field(default_factory=list[ToolParam])
    prefer_tier: str = ""       # 推荐 tier: "reader" | "reasoner" | ""=自动推断
    progress_category: str = "" # 进展类别: "mutation" | "info" | "io" | ""=自动推断
    capabilities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "prefer_tier": self.prefer_tier,
            "progress_category": self.progress_category,
            "capabilities": list(self.capabilities),
            "params": [
                {"name": p.name, "type": p.dtype, "description": p.description, "required": p.required}
                for p in self.params
            ],
        }


@dataclass
class ToolResult:
    summary: str
    evidence: str = ""
    skipped: bool = False
    error: str | None = None
    kind: str = "execute_result"   # WM 条目类型
    priority: float = 0.9          # WM 注入优先级
    resource_key: str = ""        # 结果所对应的主要资源（path / command / task_id 等）
    fingerprint: str = ""         # 结果指纹（用于 novelty / 去重 / 结果感知）
    artifact_paths: list[str] = field(default_factory=list)
    state_delta: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "evidence": self.evidence,
            "skipped": self.skipped,
            "error": self.error,
            "resource_key": self.resource_key,
            "fingerprint": self.fingerprint,
            "artifact_paths": list(self.artifact_paths),
            "state_delta": dict(self.state_delta),
            "metadata": dict(self.metadata),
        }


@dataclass
class ToolContext:
    """工具运行时上下文，工具通过 ctx 访问所有记忆层，无需直接依赖具体类。"""
    config: "Config"
    wm: "WorkingMemory"
    task_store: "TaskStore"
    episodic: "EpisodicMemory"
    semantic: "SemanticMemory"
    emotion: "EmotionState"
    probe_manager: Any = None  # ProbeManager，由 CognitionLoop._make_ctx() 注入
    judgment: Any = None       # JudgmentLayer，由 CognitionLoop._make_ctx() 注入
    execution: Any = None      # ExecutionLayer，由 CognitionLoop._make_ctx() 注入
    registry: Any = None       # ToolRegistry，由 CognitionLoop._make_ctx() 注入

    @property
    def dry_run(self) -> bool:
        return not self.config.loop.act


# ── ToolHandler 类型别名 ───────────────────────────────────────────────────────

ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass
class ToolEntry:
    manifest: ToolManifest
    handler: ToolHandler


# ── 全局注册表（模块级单例）────────────────────────────────────────────────────

_registry: dict[str, ToolEntry] = {}


@lru_cache(maxsize=1)
def default_tool_registry() -> "ToolRegistry":
    reg = ToolRegistry()
    reg.discover(Path(__file__).resolve().parent)
    return reg


def manifest_has_capability(manifest: ToolManifest | None, capability: str) -> bool:
    return bool(manifest and capability and capability in manifest.capabilities)


def tool_has_capability(registry: "ToolRegistry | None", tool_name: str, capability: str) -> bool:
    if not tool_name or not capability:
        return False
    effective_registry = registry or default_tool_registry()
    entry = effective_registry.get(tool_name)
    return manifest_has_capability(entry.manifest if entry else None, capability)


def tool_name_has_capability(tool_name: str, capability: str) -> bool:
    return tool_has_capability(default_tool_registry(), tool_name, capability)


def tool(manifest: ToolManifest) -> Callable[[ToolHandler], ToolHandler]:
    """装饰器：将函数注册为工具。

    用法：
        @tool(ToolManifest(name="shell.run", description="..."))
        async def shell_run(params, ctx): ...
    """
    def decorator(func: ToolHandler) -> ToolHandler:
        _registry[manifest.name] = ToolEntry(manifest=manifest, handler=func)
        return func
    return decorator


class ToolRegistry:
    """工具注册表的对外接口。"""

    def discover(self, tools_dir: Path) -> None:
        """扫描 tools_dir，import 所有非 _ 开头的 .py 文件，触发 @tool 装饰器注册。"""
        for mod_file in sorted(tools_dir.glob("*.py")):
            if not _is_discoverable_tool_file(mod_file):
                continue
            stem = mod_file.stem
            module_name = f"tools.{stem}"
            if module_name in sys.modules:
                continue
            spec = importlib.util.spec_from_file_location(module_name, mod_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    def reload_tool(self, tool_name: str, tools_dir: Path) -> bool:
        """热重载单个工具模块（EvolutionEngine 调用）。"""
        if not tool_name.isidentifier() or tool_name.startswith(("_", ".")) or tool_name == "registry":
            return False
        module_name = f"tools.{tool_name}"
        mod_file = tools_dir / f"{tool_name}.py"
        if not mod_file.exists():
            return False
        spec = importlib.util.spec_from_file_location(module_name, mod_file)
        if not spec or not spec.loader:
            return False
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return True

    def get(self, name: str) -> ToolEntry | None:
        return _registry.get(name)

    def has_capability(self, name: str, capability: str) -> bool:
        return tool_has_capability(self, name, capability)

    def list_manifests(self) -> list[ToolManifest]:
        return [e.manifest for e in _registry.values()]

    def list_manifests_as_dict(self) -> list[dict[str, Any]]:
        return [e.manifest.to_dict() for e in _registry.values()]
