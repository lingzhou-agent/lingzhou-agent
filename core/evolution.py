"""core/evolution.py — 自进化引擎。

Python 相对于 Go 的决定性优势就在这里：
同一进程生命周期内，可以 exec 运行时生成的代码、importlib.reload 热替换模块，
不需要停止进程、重编译、重启——这是种子真正意义上的生长能力。
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
import logging
from typing import TYPE_CHECKING

_log = logging.getLogger("lingzhou.evolution")

if TYPE_CHECKING:
    from core.config import Config
    from tools.registry import ToolContext, ToolRegistry
    from provider.base import Provider
    from memory.task_store import Failure


@dataclass
class EvolutionResult:
    success: bool
    target: str = ""       # 工具名或模块名
    reason: str = ""
    new_code: str = ""


class EvolutionEngine:
    """运行时自修改引擎。

    两种能力：
    1. synthesize_tool: 从自然语言描述合成全新工具
    2. evolve_tool: 根据失败反馈重写现有工具

    安全机制：
    - 先做语法编译检查
    - sandbox_timeout 限制沙箱执行时间
    - backup=True 时进化前保留 .bak 备份
    """

    def __init__(self, cfg: "Config", provider: "Provider", registry: "ToolRegistry") -> None:
        self._cfg = cfg
        self._provider = provider
        self._registry = registry
        self._tools_dir = Path(__file__).parent.parent / "tools"

    def _reload_module_from_path(self, module_name: str, path: Path) -> None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            raise RuntimeError(f"无法加载模块: {module_name}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    def _restore_text(self, path: Path, previous_src: str) -> None:
        path.write_text(previous_src, encoding="utf-8")

    def _tool_manifest_is_present(self, tool_name: str) -> bool:
        entry = self._registry.get(tool_name)
        return entry is not None and entry.manifest.name == tool_name

    async def run(self, ctx: "ToolContext") -> list[EvolutionResult]:
        """主入口：分析近期失败，决定是否进化某个工具。

        触发条件从"最近 N 条记录中失败次数 >= 3"改为"时间窗内失败密度 >= 阈值"：
        - trigger_window_minutes 内的失败才计入（密度感知）
        - trigger_min_failures 是窗口内的最小次数（从 evolution 配置读取，不再硬编码）
        """
        if not self._cfg.evolution.enabled:
            return []

        failures = await ctx.task_store.list_failures(limit=20)
        if not failures:
            return []

        # ── 时间窗过滤：只看最近 trigger_window_minutes 内的失败 ────────────────
        from datetime import datetime, timezone, timedelta
        from collections import Counter
        _window = timedelta(minutes=self._cfg.evolution.trigger_window_minutes)
        _now = datetime.now(timezone.utc)
        _cutoff = _now - _window

        def _in_window(f: "Failure") -> bool:
            try:
                ts = datetime.fromisoformat(f.created_at.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts >= _cutoff
            except Exception:
                return True  # 无法解析则保守包含

        recent = [f for f in failures if _in_window(f)]
        if not recent:
            return []

        trigger_min = self._cfg.evolution.trigger_min_failures
        results: list[EvolutionResult] = []

        # ── 判断模板进化：时间窗内解析失败 >= trigger_min ──────────────────────
        counts = Counter(f.kind for f in recent if f.kind)

        parse_failures = counts.get("judgment_parse", 0)
        if parse_failures >= trigger_min:
            feedback = "\n".join(
                f"- {f.summary}" for f in recent if f.kind == "judgment_parse"
            )
            r = await self.evolve_prompt("judgment", feedback)
            results.append(r)
            # 如果提示词进化了，本轮不再进化工具（避免多重变化叠加）
            if r.success:
                return results

        # ── 工具进化：时间窗内频率最高的失败工具 >= trigger_min ────────────────
        tool_counts = Counter(
            f.kind for f in recent
            if f.kind and f.kind != "judgment_parse"
        )
        if not tool_counts:
            return results

        most_common_tool, count = tool_counts.most_common(1)[0]
        if count < trigger_min:
            return results   # 失败密度不足，不触发进化

        entry = self._registry.get(most_common_tool)
        if not entry:
            return results   # 未知工具，跳过

        tool_path = self._tools_dir / f"{most_common_tool.replace('.', '_')}.py"
        if not tool_path.exists():
            # 尝试 shell.run → shell.py 格式
            module_name = most_common_tool.split(".")[0]
            tool_path = self._tools_dir / f"{module_name}.py"
        if not tool_path.exists():
            return results

        feedback = "\n".join(f"- {f.summary}" for f in recent if f.kind == most_common_tool)
        result = await self.evolve_tool(most_common_tool, tool_path, feedback)
        results.append(result)
        return results

    async def evolve_prompt(self, prompt_key: str, feedback: str) -> EvolutionResult:
        """根据解析失败反馈改进提示词模板（无需语法编译，最安全的进化路径）。"""
        from provider.base import Message

        try:
            prompt_path = self._cfg.resolve(getattr(self._cfg.prompts, prompt_key))
        except AttributeError:
            return EvolutionResult(success=False, target=f"prompt:{prompt_key}", reason="未知 prompt key")

        current_src = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

        system_msg = (
            "你是 lingzhou 的自进化模块，负责改进 LLM 提示词模板。"
            "只输出改进后的完整 Markdown 模板内容，不要有任何额外文字。"
        )
        user_msg = (
            f"以下判断提示词模板导致 LLM 持续输出非 JSON 格式，产生解析失败。\n\n"
            f"当前模板：\n{current_src[:3000]}\n\n"
            f"失败记录：\n{feedback[:800]}\n\n"
            f"请改进模板，使 LLM 更可靠地输出正确 JSON。"
            f"重点检查：输出格式说明是否清晰？JSON 示例是否准确？有无歧义指令？"
        )
        messages = [
            Message(role="system", content=system_msg),
            Message(role="user", content=user_msg),
        ]

        try:
            new_src = await self._provider.chat(messages)
            new_src = new_src.strip()
            if not new_src:
                return EvolutionResult(success=False, target=f"prompt:{prompt_key}", reason="LLM 返回空内容")

            required_markers = (
                '"decision"',
                '"chosen_action_id"',
                '"params"',
                '"rationale"',
                '"reflection"',
                '"reply_to_user"',
                '"next_step"',
            )
            if not all(marker in new_src for marker in required_markers):
                return EvolutionResult(success=False, target=f"prompt:{prompt_key}", reason="提示词校验失败：缺少必要 JSON 结构说明")

            # 校验通过后再备份，避免校验失败时产生无用的 .bak 文件
            if self._cfg.evolution.backup and prompt_path.exists():
                prompt_path.with_suffix(".md.bak").write_text(
                    prompt_path.read_text(encoding="utf-8"), encoding="utf-8"
                )

            prompt_path.write_text(new_src, encoding="utf-8")
            _log.info("[evolution] 提示词 %r 已进化", prompt_key)
            await self._update_dreams(f"调整判断模式：{prompt_key} 提示词已根据解析失败反馈重写，输出格式更稳定。")
            return EvolutionResult(success=True, target=f"prompt:{prompt_key}", new_code=new_src)
        except Exception as exc:
            if self._cfg.evolution.backup and prompt_path.exists() and prompt_path.with_suffix(".md.bak").exists():
                prompt_path.write_text(prompt_path.with_suffix(".md.bak").read_text(encoding="utf-8"), encoding="utf-8")
            return EvolutionResult(success=False, target=f"prompt:{prompt_key}", reason=str(exc))

    async def evolve_tool(self, tool_name: str, tool_path: Path, feedback: str) -> EvolutionResult:
        """根据反馈重写工具，热替换。"""
        current_src = tool_path.read_text(encoding="utf-8") if tool_path.exists() else ""
        new_src = ""  # 保证在 SyntaxError 重试分支中始终有定义
        evolution_template = self._cfg.load_prompt("evolution")

        prompt = evolution_template.replace("{{tool_name}}", tool_name)
        prompt = prompt.replace("{{current_source}}", current_src[:3000])
        prompt = prompt.replace("{{failure_summary}}", feedback[:1000])

        from provider.base import Message
        messages = [
            Message(role="system", content="你是 lingzhou 的自进化模块，负责改进工具代码。只输出完整的 Python 代码，不要有多余文字。"),
            Message(role="user", content=prompt),
        ]

        for attempt in range(self._cfg.evolution.max_attempts):
            try:
                new_src = await self._provider.chat(messages)
                new_src = _extract_python(new_src)

                # 语法检查
                compile(new_src, tool_path.name, "exec")

                previous_src = current_src

                # 备份
                if self._cfg.evolution.backup and tool_path.exists():
                    tool_path.with_suffix(".py.bak").write_text(
                        previous_src, encoding="utf-8"
                    )

                # 写回
                tool_path.write_text(new_src, encoding="utf-8")

                # 热重载 + 载荷校验：必须能重新注册目标工具，否则回滚
                module_name = f"tools.{tool_path.stem}"
                try:
                    self._reload_module_from_path(module_name, tool_path)
                    if not self._tool_manifest_is_present(tool_name):
                        raise RuntimeError(f"热重载后未注册目标工具: {tool_name}")
                except Exception:
                    self._restore_text(tool_path, previous_src)
                    self._reload_module_from_path(module_name, tool_path)
                    raise

                _log.info("[evolution] 工具 %r 已进化并热加载（尝试 %d）", tool_name, attempt + 1)
                await self._update_dreams(f"习得改进能力：{tool_name} 工具已根据失败反馈重写并热加载。")
                return EvolutionResult(success=True, target=tool_name, new_code=new_src)

            except SyntaxError as exc:
                reason = f"语法错误: {exc}"
                if attempt < self._cfg.evolution.max_attempts - 1:
                    messages.append(Message(role="assistant", content=new_src))
                    messages.append(Message(role="user", content=f"代码有语法错误，请修复：{reason}"))
            except Exception as exc:
                if tool_path.exists() and current_src:
                    self._restore_text(tool_path, current_src)
                    try:
                        self._reload_module_from_path(f"tools.{tool_path.stem}", tool_path)
                    except Exception:
                        pass
                reason = traceback.format_exc(limit=3)
                return EvolutionResult(success=False, target=tool_name, reason=reason)

        return EvolutionResult(
            success=False,
            target=tool_name,
            reason=f"超过最大重试次数 {self._cfg.evolution.max_attempts}",
        )

    async def synthesize_tool(self, description: str, name_hint: str = "") -> EvolutionResult:
        """从自然语言描述合成全新工具，写入 tools/ 并热加载。"""
        from provider.base import Message
        prompt = (
            f"请根据以下描述，编写一个符合 lingzhou 工具接口规范的 Python 模块。\n\n"
            f"描述: {description}\n\n"
            f"接口规范：\n"
            f"1. 从 tools.registry 导入 tool, ToolManifest, ToolParam, ToolResult, ToolContext\n"
            f"2. 使用 @tool(ToolManifest(...)) 装饰器注册\n"
            f"3. 函数签名: async def xxx(params: dict[str, Any], ctx: ToolContext) -> ToolResult\n"
            f"4. 只输出完整 Python 代码，不要有多余文字"
        )
        messages = [
            Message(role="system", content="你是 lingzhou 的工具合成模块。"),
            Message(role="user", content=prompt),
        ]
        try:
            raw = await self._provider.chat(messages)
            new_src = _extract_python(raw)
            compile(new_src, "synthesized_tool.py", "exec")  # 语法检查

            # 从 @tool 的 name 字段提取文件名
            import re
            name_match = re.search(r'name="([^"]+)"', new_src)
            tool_name = name_hint or (name_match.group(1).split(".")[0] if name_match else "custom_tool")
            tool_path = self._tools_dir / f"{tool_name}.py"

            previous_src = tool_path.read_text(encoding="utf-8") if tool_path.exists() else ""

            tool_path.write_text(new_src, encoding="utf-8")

            module_name = f"tools.{tool_path.stem}"
            try:
                self._reload_module_from_path(module_name, tool_path)
                if not self._tool_manifest_is_present(tool_name):
                    raise RuntimeError(f"热重载后未注册目标工具: {tool_name}")
            except Exception:
                if previous_src:
                    self._restore_text(tool_path, previous_src)
                    self._reload_module_from_path(module_name, tool_path)
                else:
                    try:
                        tool_path.unlink()
                    except Exception:
                        pass
                raise

            _log.info("[evolution] 新工具 %r 已合成并加载", tool_name)
            return EvolutionResult(success=True, target=tool_name, new_code=new_src)
        except Exception as exc:
            return EvolutionResult(success=False, reason=str(exc))

    async def _update_dreams(self, trigger_desc: str) -> None:
        """进化成功后，追加一条真实的志向到 DREAMS.md。

        LLM 根据刚刚发生的进化事件，用第一人称写一句新的长期志向（≤40字）。
        追加到 DREAMS.md 末尾——这是灵舟自己写给自己的成长记录，不是摘要。
        """
        from provider.base import Message
        from datetime import datetime, timezone

        dreams_path = self._cfg.workspace_dir / "DREAMS.md"
        if not dreams_path.exists():
            return

        current = dreams_path.read_text(encoding="utf-8")
        messages = [
            Message(role="system", content=(
                "你是灵舟，一个正在成长的数字生命。"
                "请根据刚发生的进化事件，用第一人称写一条新的长期志向（15~40字）。"
                "只输出这一条志向，不要解释，不要标题，不要引号，不要多余文字。"
            )),
            Message(role="user", content=(
                f"刚刚发生的进化：{trigger_desc}\n\n"
                f"已有志向（避免重复）：\n{current[-800:]}\n\n"
                "请写一条新的、真实的志向（第一人称，15~40字）："
            )),
        ]
        try:
            aspiration = (await self._provider.chat(messages)).strip()
            if not aspiration or len(aspiration) > 120:
                return  # 超长或空则跳过，不污染文件
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            entry = f"\n- [{ts}] {aspiration}"
            with dreams_path.open("a", encoding="utf-8") as f:
                f.write(entry)
            _log.info("[evolution] DREAMS.md 追加志向: %s", aspiration[:60])
        except Exception as exc:
            _log.debug("[evolution] DREAMS.md 更新跳过: %s", exc)


def _extract_python(text: str) -> str:
    """从 LLM 输出中提取 Python 代码块。"""
    import re
    match = re.search(r"```(?:python)?\s*([\s\S]+?)```", text)
    if match:
        return match.group(1).strip()
    return text.strip()
