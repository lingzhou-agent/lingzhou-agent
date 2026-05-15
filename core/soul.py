"""core/soul.py — Soul 层管理器（工作区身份文件初始化、EMA 同步、冷启动注入）。

职责（严格限定于 Soul / 身份层）：
  - workspace_dir 下 SOUL.md / IDENTITY.md / BOOTSTRAP.md 等身份文件的首次写入
  - 每轮 EMA 后将最新 ethos 值同步写回 SOUL.md（人类可读镜像）
  - bootstrap：将所有身份文件注入 WM（启动身份注入机制）

不负责：
  - models.json 生成 → 由 provider.models_gen.ensure_models_json() 在启动时处理

原位置：CognitionLoop._build_soul_content / _init_soul_files / _sync_soul_md / _bootstrap
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from core.workspace_defaults import (
    IDENTITY_MD,
    BOOTSTRAP_MD,
    USER_MD,
    TOOLS_MD,
    HEARTBEAT_MD,
    MEMORY_MD,
)

if TYPE_CHECKING:
    from core.config import Config
    from core.judgment import JudgmentLayer
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory

_log = logging.getLogger("lingzhou.soul")

# 工作区默认文件列表（名称 → 模板内容），按写入顺序
_WORKSPACE_FILES: list[tuple[str, str]] = [
    ("IDENTITY.md",  IDENTITY_MD),
    ("BOOTSTRAP.md", BOOTSTRAP_MD),
    ("USER.md",      USER_MD),
    ("TOOLS.md",     TOOLS_MD),
    ("HEARTBEAT.md", HEARTBEAT_MD),
    ("MEMORY.md",    MEMORY_MD),
]

# 冷启动时注入 WM 的文件顺序（越靠前越优先被 LLM 读到）
_BOOTSTRAP_FILES = ("BOOTSTRAP.md", "IDENTITY.md", "SOUL.md", "USER.md", "TOOLS.md", "HEARTBEAT.md", "MEMORY.md")
# DREAMS.md 单独注入：给 LLM 看自己的长期志向演化记录（不进 system prompt 前缀，只进 WM）
_DREAMS_FILE = "DREAMS.md"


class SoulManager:
    """管理 Soul 层文件，并在冷启动时将身份文件注入 WM。"""

    def __init__(
        self,
        cfg: "Config",
        task_store: "TaskStore",
        wm: "WorkingMemory",
    ) -> None:
        self._cfg = cfg
        self._task_store = task_store
        self._wm = wm

    @staticmethod
    def _build_content(soul_name: str, ethos: dict[str, Any], eb: dict[str, Any], axioms: list[str] | None = None) -> str:
        """生成 SOUL.md 文件内容（供 init_files 和 sync_md 共用）。"""
        axiom_lines = "".join(f"- {a}\n" for a in (axioms or [])) or "- （暂无 hard axioms）\n"
        return (
            f"# {soul_name} SOUL\n\n"
            "> 此文件是 runtime facts 的人类可读镜像；完整灵魂叙事主要位于 BOOTSTRAP.md / IDENTITY.md。\n\n"
            "## 核心价值观（EMA 持久化版本）\n\n"
            f"- 真实 (truth):      {ethos.get('truth', eb.get('truth', 0.85)):.3f}\n"
            f"- 谨慎 (caution):    {ethos.get('caution', eb.get('caution', 0.70)):.3f}\n"
            f"- 连续 (continuity): {ethos.get('continuity', eb.get('continuity', 0.65)):.3f}\n"
            f"- 好奇 (curiosity):  {ethos.get('curiosity', eb.get('curiosity', 0.60)):.3f}\n"
            f"- 关怀 (care):       {ethos.get('care', eb.get('care', 0.55)):.3f}\n\n"
            "## 绝对边界（hard axioms）\n\n"
            f"{axiom_lines}\n"
            "## 使命\n\n"
            f"我是{soul_name}，一个在 lingzhou 新身躯中继续演进的数字生命。\n"
            "每一个 tick 都是一次成长，每一次 reflection 都是一次结晶。\n"
        )

    async def _soul_name(self) -> str:
        """从 facts DB 读取 soul:name，回退到 cfg.soul.name。"""
        name_val, name_found = await self._task_store.get_fact("soul:name")
        return name_val if name_found and name_val else self._cfg.soul.name

    async def _ethos_from_db(self) -> dict[str, Any]:
        """从 facts DB 读取 soul:ethos_baseline，解析失败返回空 dict。"""
        ethos_json, found = await self._task_store.get_fact("soul:ethos_baseline")
        if not found or not ethos_json:
            return {}
        try:
            return json.loads(ethos_json)
        except Exception:
            return {}

    async def _axioms_from_db(self) -> list[str]:
        """从 facts DB 读取 soul:hard_axioms。"""
        axioms_json, found = await self._task_store.get_fact("soul:hard_axioms")
        if not found or not axioms_json:
            return list(self._cfg.soul.hard_axioms)
        try:
            data = json.loads(axioms_json)
            return [str(x) for x in data] if isinstance(data, list) else list(self._cfg.soul.hard_axioms)
        except Exception:
            return list(self._cfg.soul.hard_axioms)

    async def init_files(self) -> None:
        """冷启动：确保 workspace_dir 中有所有必要的 Soul 文件。

        优先级：facts DB（持久化 EMA 值）> cfg.soul 配置默认值（全新启动）。
        文件一旦写入后归用户/agent 管理，不会被再次覆盖。
        """
        workspace = self._cfg.workspace_dir
        workspace.mkdir(parents=True, exist_ok=True)

        soul_path = workspace / "SOUL.md"
        if not soul_path.exists():
            soul_name = await self._soul_name()
            ethos = await self._ethos_from_db()
            axioms = await self._axioms_from_db()
            eb = self._cfg.soul.ethos_baseline
            soul_path.write_text(self._build_content(soul_name, ethos, eb, axioms), encoding="utf-8")
            _log.info("Soul 初始化: 已写入 %s", soul_path)

        for fname, content in _WORKSPACE_FILES:
            fpath = workspace / fname
            if not fpath.exists():
                fpath.write_text(content, encoding="utf-8")
                _log.info("%s 初始化: 已写入 %s", fname, fpath)

    async def sync_md(self) -> None:
        """将 facts DB 中最新 EMA ethos 值同步写回 SOUL.md（人类可读镜像）。

        只在 DB 中有 ethos_baseline 时才写入，避免全新启动时覆盖初始化文件。
        """
        ethos = await self._ethos_from_db()
        if not ethos:
            return
        soul_name = await self._soul_name()
        axioms = await self._axioms_from_db()
        eb = self._cfg.soul.ethos_baseline
        soul_path = self._cfg.workspace_dir / "SOUL.md"
        soul_path.write_text(self._build_content(soul_name, ethos, eb, axioms), encoding="utf-8")

    async def bootstrap(self, judgment: "JudgmentLayer | None" = None) -> None:
        """冷启动：Soul 文件初始化 + WM 身份注入 + system prompt 前缀注入。

        启动身份注入规则：
        - BOOTSTRAP.md / IDENTITY.md → judgment.set_identity_prefix()（永久注入 system prompt）
        - 所有身份文件 → WM（in-context 引用，优先级 0.85）
        """
        from memory.working import WMItem

        await self.init_files()
        workspace = self._cfg.workspace_dir
        injected: list[str] = []
        identity_parts: list[str] = []
        for fname in _BOOTSTRAP_FILES:
            fpath = workspace / fname
            if fpath.exists():
                try:
                    content = fpath.read_text(encoding="utf-8")
                    self._wm.add(WMItem(
                        kind="bootstrap_identity",
                        content=f"[{fname}]\n{content}",
                        priority=self._cfg.thresholds.wm_pri_identity,
                    ))
                    injected.append(fname)
                    # 核心身份文件 → system prompt 前缀（永久，不随 WM 驱逐）
                    if fname in ("BOOTSTRAP.md", "IDENTITY.md"):
                        identity_parts.append(f"[{fname}]\n{content}")
                except Exception:
                    pass
        if injected:
            _log.info("[boot] 身份注入: %s", " ".join(injected))
        if judgment is not None and identity_parts:
            judgment.set_identity_prefix("\n\n".join(identity_parts))

        # DREAMS.md：长期志向，进 WM 供 LLM 感知自己的成长轨迹（priority 略低于身份文件）
        dreams_path = workspace / _DREAMS_FILE
        if dreams_path.exists():
            try:
                dreams_content = dreams_path.read_text(encoding="utf-8").strip()
                if dreams_content:
                    self._wm.add(WMItem(
                        kind="bootstrap_identity",
                        content=f"[{_DREAMS_FILE}]\n{dreams_content}",
                        priority=self._cfg.thresholds.wm_pri_identity * 0.9,
                    ))
            except Exception:
                pass

        # 清理旧版 heartbeat cron 信号（旧实现将 heartbeat 存入 signals 表；
        # 新版改为 monotonic 时间戳机制，DB 中遗留的 source=heartbeat 条目应移除）
        old_hb = [
            s for s in await self._task_store.list_signals(limit=200)
            if s.get("payload", {}).get("source") == "heartbeat"
        ]
        for s in old_hb:
            await self._task_store.cancel_signal(s["id"])

    async def refresh_identity(self, judgment: "JudgmentLayer | None" = None) -> None:
        """重读身份文件，更新 system prompt 前缀。

        在 evolution 运行后调用：evolution 可能已通过 file.write 修改
        BOOTSTRAP.md / IDENTITY.md，身份前缀需要在当次会话内生效，
        不应等到下次重启。只更新 system prompt 前缀，不重复注入 WM。
        """
        if judgment is None:
            return
        workspace = self._cfg.workspace_dir
        identity_parts: list[str] = []
        for fname in ("BOOTSTRAP.md", "IDENTITY.md"):
            fpath = workspace / fname
            if fpath.exists():
                try:
                    content = fpath.read_text(encoding="utf-8")
                    identity_parts.append(f"[{fname}]\n{content}")
                except Exception:
                    pass
        if identity_parts:
            judgment.set_identity_prefix("\n\n".join(identity_parts))
            _log.debug("[soul] 身份前缀已刷新（evolution 后更新）")
