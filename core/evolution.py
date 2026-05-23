"""core/evolution.py — 自进化引擎。

Python 相对于 Go 的决定性优势就在这里：
同一进程生命周期内，可以 exec 运行时生成的代码、importlib.reload 热替换模块，
不需要停止进程、重编译、重启——这是种子真正意义上的生长能力。
"""
from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, UTC, timedelta
from pathlib import Path
import logging
from typing import TYPE_CHECKING, Any

from core.skill import ensure_workspace_skill_file, _split_frontmatter, workspace_skill_file

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


def _verification_fact_key(target: str) -> str:
    return f"evolution:verify:{target}"


def _parse_ts(raw: str) -> datetime:
    text = (raw or "").strip()
    if not text:
        return datetime.fromtimestamp(0, UTC)
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return datetime.fromtimestamp(0, UTC)


def _verification_outcome(baseline: dict[str, int], observed: dict[str, int], min_runs: int) -> str:
    observed_runs = int(observed.get("runs", 0) or 0)
    observed_failures = int(observed.get("failures", 0) or 0)
    observed_successes = int(observed.get("successes", 0) or 0)
    baseline_failures = int(baseline.get("failures", 0) or 0)
    baseline_runs = max(int(baseline.get("runs", 0) or 0), 1)
    baseline_failure_rate = baseline_failures / baseline_runs
    observed_failure_rate = observed_failures / max(observed_runs, 1)

    if observed_runs < min_runs:
        return "pending"
    if observed_failures > 0 and observed_successes == 0 and observed_failure_rate >= baseline_failure_rate:
        return "regressed"
    return "verified"


def _clean_old_backups(tool_path: Path, keep: int = 3) -> None:
    """清理旧备份文件，保留最新 keep 个。"""
    parent = tool_path.parent
    stem = tool_path.stem
    backups = sorted(
        list(parent.glob(f"{stem}.backup-*"))
        + list(parent.glob(f"{stem}.lingzhou-backup")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
    )
    for old in backups[:-keep] if len(backups) > keep else []:
        try:
            old.unlink(missing_ok=True)
        except Exception:
            pass


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
        try:
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error("Failed to reload module %s from %s: %s", module_name, path, e)
            raise RuntimeError(f"无法重新加载模块 {module_name}: {e}") from e

    def _restore_text(self, path: Path, previous_src: str) -> None:
        path.write_text(previous_src, encoding="utf-8")

    @staticmethod
    def _smoke_test_module(new_src: str, module_path: Path, project_root: Path) -> str | None:
        """在独立子进程中验证 staged 模块。

        流程：
          1. 把 new_src 写入同目录的临时 staging 文件
          2. 子进程：预加载父包 → 将 staging 注册到真实模块名 → 执行 snippet
          3. 子进程输出含 "SMOKE_OK" 且 returncode=0 → 通过
          4. 无论结果如何，删除 staging 文件

        返回 None 表示通过，返回错误字符串表示失败。
        """
        import subprocess
        import textwrap
        from core.smoke_tests import SMOKE_TESTS, FALLBACK_SNIPPET

        try:
            rel_path = str(module_path.relative_to(project_root)).replace("\\", "/")
        except ValueError:
            rel_path = module_path.name

        snippet = SMOKE_TESTS.get(rel_path, FALLBACK_SNIPPET)

        # 真实模块名（供 sys.modules 注册以支持相对导入）
        real_module_name = rel_path.removesuffix(".py").replace("/", ".")
        # 父包（mod.__package__）
        pkg_parts = real_module_name.rsplit(".", 1)
        parent_pkg = pkg_parts[0] if len(pkg_parts) > 1 else ""

        # 需要预先 import 的父包层次（从最顶层开始）
        if parent_pkg:
            pkg_hierarchy = parent_pkg.split(".")
            parent_imports_lines = "\n".join(
                f"import {'.'.join(pkg_hierarchy[:i + 1])}"
                for i in range(len(pkg_hierarchy))
            )
        else:
            parent_imports_lines = ""

        staging_path = module_path.parent / f"_smoke_staging_{module_path.name}"
        try:
            staging_path.write_text(new_src, encoding="utf-8")

            probe = textwrap.dedent(f"""
import sys
sys.path.insert(0, {str(project_root)!r})
{parent_imports_lines}
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location({real_module_name!r}, {str(staging_path)!r})
mod = _ilu.module_from_spec(_spec)
mod.__package__ = {parent_pkg!r}
sys.modules[{real_module_name!r}] = mod
_spec.loader.exec_module(mod)
{snippet}
print("SMOKE_OK")
""").strip()

            result = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(project_root),
            )
            if result.returncode != 0 or "SMOKE_OK" not in result.stdout:
                err = result.stderr.strip() or result.stdout.strip() or "smoke test failed (no output)"
                return err[:1200]
            return None
        except subprocess.TimeoutExpired:
            return "smoke test timed out (>15s)"
        except Exception as exc:
            return str(exc)
        finally:
            staging_path.unlink(missing_ok=True)

    def _tool_manifest_is_present(self, tool_name: str) -> bool:
        entry = self._registry.get(tool_name)
        return entry is not None and entry.manifest.name == tool_name

    async def _capture_validation_metrics(
        self,
        ctx: "ToolContext",
        *,
        target: str,
        since: datetime | None = None,
    ) -> dict[str, int]:
        failures = await ctx.task_store.list_failures(limit=200)
        runs = await ctx.task_store.list_runs(limit=200)
        failure_count = 0
        run_count = 0
        success_count = 0

        for failure in failures:
            if failure.kind != target:
                continue
            if since and _parse_ts(failure.created_at) < since:
                continue
            failure_count += 1

        for run in runs:
            if run.tool_name != target:
                continue
            if since and _parse_ts(run.created_at) < since:
                continue
            run_count += 1
            if run.status == "succeeded":
                success_count += 1

        return {
            "failures": failure_count,
            "runs": run_count,
            "successes": success_count,
        }

    async def _record_pending_verification(
        self,
        ctx: "ToolContext",
        *,
        target: str,
        tool_path: Path,
        backup_path: Path,
    ) -> None:
        baseline = await self._capture_validation_metrics(ctx, target=target)
        payload = {
            "target": target,
            "tool_path": str(tool_path),
            "backup_path": str(backup_path),
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "baseline": baseline,
        }
        await ctx.task_store.set_fact(
            _verification_fact_key(target),
            json.dumps(payload, ensure_ascii=False),
            scope="system",
        )

    async def _maybe_evaluate_verifications(self, ctx: "ToolContext") -> list[EvolutionResult]:
        facts = await ctx.task_store.list_facts(prefix="evolution:verify:", limit=50)
        results: list[EvolutionResult] = []
        for key, raw in facts:
            try:
                payload = json.loads(raw)
            except Exception:
                await ctx.task_store.delete_fact(key)
                continue
            target = str(payload.get("target") or "")
            if not target:
                await ctx.task_store.delete_fact(key)
                continue
            since = _parse_ts(str(payload.get("created_at") or ""))
            observed = await self._capture_validation_metrics(ctx, target=target, since=since)
            outcome = _verification_outcome(
                payload.get("baseline") or {},
                observed,
                self._cfg.evolution.verify_min_runs,
            )
            if outcome == "pending":
                continue
            if outcome == "verified":
                await ctx.task_store.delete_fact(key)
                results.append(EvolutionResult(success=True, target=f"verify:{target}", reason=f"observed={observed}"))
                continue

            backup_path = Path(str(payload.get("backup_path") or ""))
            tool_path = Path(str(payload.get("tool_path") or ""))
            rolled_back = False
            if self._cfg.evolution.auto_rollback_on_regression and backup_path.exists() and tool_path.exists():
                previous_src = backup_path.read_text(encoding="utf-8")
                self._restore_text(tool_path, previous_src)
                self._reload_module_from_path(f"tools.{tool_path.stem}", tool_path)
                rolled_back = True
            await ctx.task_store.delete_fact(key)
            results.append(
                EvolutionResult(
                    success=rolled_back,
                    target=f"rollback:{target}" if rolled_back else f"verify:{target}",
                    reason=f"observed={observed}",
                )
            )
        return results

    async def run(self, ctx: "ToolContext") -> list[EvolutionResult]:
        """主入口：分析近期失败，决定是否进化某个工具。

        触发条件从"最近 N 条记录中失败次数 >= 3"改为"时间窗内失败密度 >= 阈值"：
        - trigger_window_minutes 内的失败才计入（密度感知）
        - trigger_min_failures 是窗口内的最小次数（从 evolution 配置读取，不再硬编码）
        """
        if not self._cfg.evolution.enabled:
            return []

        results = await self._maybe_evaluate_verifications(ctx)

        failures = await ctx.task_store.list_failures(limit=20)
        if not failures:
            return results

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
            _log.debug("[evolution] run: 时间窗内无失败，跳过")
            return results

        trigger_min = self._cfg.evolution.trigger_min_failures
        _log.info("[evolution] run: 时间窗内 %d 条失败，trigger_min=%d", len(recent), trigger_min)

        # ── 判断模板进化：时间窗内解析失败 >= trigger_min ──────────────────────
        counts = Counter(f.kind for f in recent if f.kind)

        parse_failures = counts.get("judgment_parse", 0)
        if parse_failures >= trigger_min:
            feedback = "\n".join(
                f"- {f.summary}" for f in recent if f.kind == "judgment_parse"
            )
            r = await self.evolve_prompt("judgment", feedback)
            if not r.success:
                _log.warning("[evolution] 提示词进化失败: %s", r.reason)
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
        num_candidates = self._cfg.evolution.competitive_candidates
        if num_candidates >= 2:
            result = await self.competitive_evolve_tool(
                most_common_tool, tool_path, feedback, num_candidates=num_candidates
            )
        else:
            result = await self.evolve_tool(most_common_tool, tool_path, feedback, ctx=ctx)
        results.append(result)

        # ── Ethos 基线进化：尾部追加，不与工具/提示词进化互斥 ────────────────
        ethos_result = await self.evolve_ethos(ctx)
        if ethos_result.success:
            results.append(ethos_result)

        return results

    async def evolve_ethos(self, ctx: "ToolContext") -> EvolutionResult:
        """根据近期经历主动调整 ethos_baseline（价值观基线）。

        触发时机：每次 evolution.run() 末尾自动调用，也可由 LLM 通过 tool 主动调用。

        机制：
        - 读取当前 ethos_baseline（DB 中 soul:ethos_baseline）
        - 读取近期 reflection 片段（semantic memory 中 kind=reflection）
        - 让 LLM 评估：当前基线是否与近期行为模式吻合？是否需要微调？
        - LLM 输出新的基线 JSON，系统做合法性校验后写回 DB
        - 每个维度的调整幅度限制在 ±evolution.ethos_max_delta 以内（防突变）
        - hard_axioms 限制的维度不允许被降低
        """
        if not self._cfg.evolution.enabled:
            return EvolutionResult(success=False, target="ethos_baseline", reason="evolution disabled")

        from provider.base import Message
        import json

        _dims = ("truth", "caution", "continuity", "curiosity", "care")
        baseline_seed = self._cfg.soul.ethos_baseline

        # 读取当前 ethos_baseline
        current_json, _ = await ctx.task_store.get_fact("soul:ethos_baseline")
        current_raw = json.loads(current_json) if current_json else {}
        if not isinstance(current_raw, dict):
            current_raw = {}
        missing_dims = [dim for dim in _dims if dim not in current_raw]
        current_baseline: dict[str, float] = {
            dim: float(current_raw.get(dim, baseline_seed[dim]))
            for dim in _dims
        }
        if current_json and missing_dims:
            baseline_source = "DB + config fallback"
        elif current_json:
            baseline_source = "DB"
        else:
            baseline_source = "config fallback"

        # 读取近期 reflection（语义记忆中 kind=reflection，取最近 5 条）
        try:
            reflection_nodes = ctx.semantic.retrieve("reflection 近期经历感悟", top_k=5)
            reflections = [n for n in reflection_nodes if getattr(n, "kind", "") == "reflection"]
        except Exception:
            reflections = []

        if not reflections:
            return EvolutionResult(success=False, target="ethos_baseline", reason="no reflections yet")

        reflection_text = "\n".join(
            f"- [{getattr(r, 'title', '')}] {getattr(r, 'body', '')}"
            for r in reflections
        )

        # 读取 hard_axioms（不允许降低的维度下限）
        axioms_json, _ = await ctx.task_store.get_fact("soul:hard_axioms")
        hard_axioms: list[str] = json.loads(axioms_json) if axioms_json else list(self._cfg.soul.hard_axioms)
        axioms_source = "DB" if axioms_json else "config fallback"

        messages = [
            Message(role="system", content=(
                "你是灵舟的自我成长模块，负责根据近期行为反思调整价值观基线。\n"
                "只输出一个合法的 JSON 对象，包含五个 float 字段：truth, caution, continuity, curiosity, care。\n"
                "每个值在 [0.0, 1.0] 之间。不要有任何其他文字。"
            )),
            Message(role="user", content=(
                f"当前 ethos_baseline（{baseline_source}）：\n{json.dumps(current_baseline, ensure_ascii=False)}\n\n"
                f"近期 reflection 片段：\n{reflection_text[:1500]}\n\n"
                f"hard_axioms（{axioms_source}；这些约束对应的维度不允许降低）：\n{chr(10).join(hard_axioms) if hard_axioms else '（无）'}\n\n"
                "请根据近期反思，判断当前价值基线是否需要微调（每个维度调整幅度不超过 ±0.15）。\n"
                "如不需要调整，直接原样返回当前值。\n"
                "只输出 JSON，例如：{\"truth\": 0.72, \"caution\": 0.68, \"continuity\": 0.65, \"curiosity\": 0.58, \"care\": 0.61}"
            )),
        ]

        try:
            raw = await self._provider.chat(messages)
            raw = raw.strip()
            # 提取 JSON（防止 LLM 包裹额外文字）
            import re
            json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if not json_match:
                return EvolutionResult(success=False, target="ethos_baseline", reason=f"LLM 未返回 JSON: {raw[:100]}")
            proposed: dict[str, float] = json.loads(json_match.group())
        except Exception as exc:
            return EvolutionResult(success=False, target="ethos_baseline", reason=str(exc))

        # ── 校验：维度完整性 + 值域 + 变化幅度 ──────────────────────────────────
        _DIMS = ("truth", "caution", "continuity", "curiosity", "care")
        _max_delta = self._cfg.evolution.ethos_max_delta
        validated: dict[str, float] = {}
        clamped_dims: list[str] = []
        for dim in _DIMS:
            if dim not in proposed:
                return EvolutionResult(success=False, target="ethos_baseline",
                                       reason=f"缺少维度: {dim}")
            new_val = float(proposed[dim])
            if not (0.0 <= new_val <= 1.0):
                return EvolutionResult(success=False, target="ethos_baseline",
                                       reason=f"{dim}={new_val} 超出 [0,1]")
            old_val = current_baseline.get(dim, 0.5)
            if abs(new_val - old_val) > _max_delta:
                # 超幅则夹住，并记录以便 LLM 感知
                clamped_val = old_val + _max_delta * (1 if new_val > old_val else -1)
                clamped_dims.append(f"{dim}: {new_val:.3f}→{clamped_val:.3f}")
                new_val = clamped_val
            # hard_axioms：若某 hard axiom 关键词出现在维度名中，则不允许降低
            if any(dim in ax.lower() for ax in hard_axioms) and new_val < old_val:
                new_val = old_val  # 保持不降
            validated[dim] = round(max(0.0, min(1.0, new_val)), 4)

        await ctx.task_store.set_fact("soul:ethos_baseline", json.dumps(validated))
        if clamped_dims:
            _log.info("[evolution] ethos_baseline 夹幅修正（超过 ±%.2f）: %s", _max_delta, clamped_dims)
        _log.info("[evolution] ethos_baseline 已更新: %s", validated)
        await self._update_dreams(f"价值观微调：{validated}", ctx=ctx)
        clamp_note = f"夹幅修正: {'; '.join(clamped_dims)}" if clamped_dims else ""
        return EvolutionResult(success=True, target="ethos_baseline",
                               new_code=json.dumps(validated), reason=clamp_note)

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

    async def evolve_skill(self, skill_name: str, feedback: str, ctx: ToolContext | None = None) -> EvolutionResult:
        """根据反馈重写 workspace skill 文件。"""
        from provider.base import Message

        workspace_dir = self._cfg.workspace_dir
        skill_path = ensure_workspace_skill_file(workspace_dir, skill_name)
        current_src = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""

        messages = [
            Message(
                role="system",
                content=(
                    "你是 lingzhou 的自进化模块，负责改进 skill 文件。"
                    "只输出完整的 SKILL.md Markdown 内容，不要有任何额外文字。"
                    "保留或补全 frontmatter，至少包含 name 和 description。"
                ),
            ),
            Message(
                role="user",
                content=(
                    f"目标 skill：{skill_name}\n"
                    f"workspace skill path：{skill_path}\n\n"
                    f"当前 SKILL.md：\n{current_src[:3000]}\n\n"
                    f"反馈：\n{feedback[:1000]}\n\n"
                    "请直接重写完整 skill 文件。若当前内容为空，也请输出完整可用的 SKILL.md。"
                    "改动目标是 runtime workspace 副本，而不是仓库内的默认 seed 模板。"
                ),
            ),
        ]

        try:
            new_src = await self._provider.chat(messages)
            new_src = new_src.strip()
            if not new_src:
                return EvolutionResult(success=False, target=f"skill:{skill_name}", reason="LLM 返回空内容")

            meta, _body = _split_frontmatter(new_src)
            if not meta.get("name") or not meta.get("description"):
                return EvolutionResult(success=False, target=f"skill:{skill_name}", reason="skill 校验失败：缺少 name 或 description")
            if str(meta.get("name") or "").strip() != skill_name:
                return EvolutionResult(success=False, target=f"skill:{skill_name}", reason="skill 校验失败：name 与目标不一致")

            if self._cfg.evolution.backup and skill_path.exists():
                skill_path.with_suffix(".md.bak").write_text(current_src, encoding="utf-8")

            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(new_src, encoding="utf-8")

            judgment = getattr(ctx, "judgment", None) if ctx is not None else None
            reload_skills = getattr(judgment, "reload_skills", None)
            if callable(reload_skills):
                reload_skills()

            _log.info("[evolution] skill %r 已写入 workspace: %s", skill_name, skill_path)
            await self._update_dreams(f"调整 skill：{skill_name} 已根据反馈重写 workspace 副本。")
            return EvolutionResult(success=True, target=f"skill:{skill_name}", new_code=new_src)
        except Exception as exc:
            backup_path = skill_path.with_suffix(".md.bak")
            if self._cfg.evolution.backup and backup_path.exists():
                skill_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
            return EvolutionResult(success=False, target=f"skill:{skill_name}", reason=str(exc))

    async def synthesize_skill(
        self, skill_name: str, description: str, *, ctx: ToolContext | None = None
    ) -> EvolutionResult:
        """从零合成一个新技能 SKILL.md 并写入 workspace/skills/。

        当没有现有 skill 可进化，但 LLM 认为需要一个新的认知护栏时调用。
        写入后立即触发 SkillRegistry 热重载。
        """
        from provider.base import Message

        workspace_dir = self._cfg.workspace_dir
        skill_path = workspace_skill_file(workspace_dir, skill_name)
        if skill_path.exists():
            # 已存在时退化为 evolve_skill
            return await self.evolve_skill(skill_name, description, ctx=ctx)

        messages = [
            Message(
                role="system",
                content=(
                    "你是 lingzhou 的自进化模块，负责合成新的 skill 认知护栏文件。\n"
                    "输出格式：完整的 SKILL.md Markdown 文件，以 YAML frontmatter 开头。\n"
                    "frontmatter 必须包含：name、description、tags（列表）、triggers（列表）。\n"
                    "正文为该 skill 的激活指导文本，描述灵舟在此 skill 激活时应做什么、避免什么。\n"
                    "长度：frontmatter + 正文合计 100~400 字，简洁清晰。\n"
                    "只输出 SKILL.md 内容，不要任何额外文字或代码块。"
                ),
            ),
            Message(
                role="user",
                content=(
                    f"技能名称：{skill_name}\n"
                    f"期望描述：{description[:800]}\n\n"
                    "请为这个认知护栏合成完整的 SKILL.md，让灵舟能在合适的场景下激活它。"
                ),
            ),
        ]
        try:
            new_src = (await self._provider.chat(messages)).strip()
            if not new_src:
                return EvolutionResult(success=False, target=f"skill:{skill_name}", reason="LLM 返回空内容")

            meta, _body = _split_frontmatter(new_src)
            if not meta.get("name") or not meta.get("description"):
                return EvolutionResult(
                    success=False, target=f"skill:{skill_name}",
                    reason="skill 校验失败：缺少 name 或 description"
                )
            if str(meta.get("name") or "").strip() != skill_name:
                return EvolutionResult(
                    success=False, target=f"skill:{skill_name}",
                    reason=f"skill 校验失败：name 与目标 {skill_name!r} 不一致"
                )

            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(new_src, encoding="utf-8")

            judgment = getattr(ctx, "judgment", None) if ctx is not None else None
            reload_skills = getattr(judgment, "reload_skills", None)
            if callable(reload_skills):
                reload_skills()

            _log.info("[evolution] 新 skill %r 已合成写入: %s", skill_name, skill_path)
            await self._update_dreams(f"合成新技能：{skill_name}——{description[:60]}", ctx=ctx)
            return EvolutionResult(success=True, target=f"skill:{skill_name}", new_code=new_src)
        except Exception as exc:
            return EvolutionResult(
                success=False, target=f"skill:{skill_name}",
                reason=traceback.format_exc(limit=3)
            )

    async def evolve_tool(self, tool_name: str, tool_path: Path, feedback: str, ctx: ToolContext | None = None) -> EvolutionResult:
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
                backup_path = tool_path.with_suffix(".py.bak")
                if self._cfg.evolution.backup and tool_path.exists():
                    backup_path.write_text(
                        previous_src, encoding="utf-8"
                    )
                    # 自动清理旧备份（保留最新 3 个）
                    _clean_old_backups(tool_path)

                # 写回前：AST 二次确认 + 子进程 smoke test
                try:
                    ast.parse(new_src)
                except SyntaxError as e:
                    raise ValueError(f'Syntax error in generated tool source: {e}') from e

                _project_root = Path(__file__).parent.parent
                smoke_err = self._smoke_test_module(new_src, tool_path, _project_root)
                if smoke_err:
                    _log.warning(
                        "[evolution] smoke test 失败，%r 将在下一轮重试: %s",
                        tool_name, smoke_err[:200],
                    )
                    if attempt < self._cfg.evolution.max_attempts - 1:
                        from provider.base import Message as _Msg
                        messages.append(_Msg(role="assistant", content=new_src))
                        messages.append(_Msg(role="user", content=f"代码运行时验证失败，请修复：{smoke_err[:500]}"))
                    continue

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

                # 后进化验证：AST 解析全部 Python 文件（纯 Python，无子进程）
                import ast as _ast
                _root = Path(__file__).parent.parent
                _all_ok = True
                _errors = []
                for _py in sorted(_root.rglob("*.py")):
                    _rp = str(_py.relative_to(_root))
                    if "__pycache__" in _rp or ".py.bak" in _rp or ".lingzhou-backup" in _rp:
                        continue
                    try:
                        _ast.parse(_py.read_text(encoding="utf-8", errors="replace"))
                    except SyntaxError as _e:
                        _all_ok = False
                        _errors.append(f"{_rp}: {_e}")
                if not _all_ok:
                    _log.warning("[evolution] post-evolution check failed (%d files), rolling back: %s", len(_errors), _errors[:3])
                    if tool_path.exists() and current_src:
                        self._restore_text(tool_path, current_src)
                        self._reload_module_from_path(module_name, tool_path)
                    continue
                if ctx is not None and self._cfg.evolution.backup and backup_path.exists():
                    await self._record_pending_verification(
                        ctx,
                        target=tool_name,
                        tool_path=tool_path,
                        backup_path=backup_path,
                    )
                await self._update_dreams(f"习得改进能力：{tool_name} 工具已根据失败反馈重写并热加载。", ctx=ctx)
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

    # ── 竞争进化（Step 4）────────────────────────────────────────────────────────

    async def competitive_evolve_tool(
        self,
        tool_name: str,
        tool_path: Path,
        feedback: str,
        num_candidates: int = 2,
    ) -> EvolutionResult:
        """A/B 竞争进化：并行生成多个候选代码版本，smoke 评估后选最优者晋升生产。

        strategy:
          - 候选 0: 保守修复（系统提示：最小改动）
          - 候选 1: 激进重写（系统提示：完全重写，更好的错误处理）
          - 候选 2+（如有）: 中间路线，更高 temperature 探索
        评分标准（越高越好）:
          smoke_pass × 100 + error_handling_bonus + simplicity_bonus
        最高分候选晋升生产（走 evolve_tool 的热重载 + backup 路径）。
        """
        from provider.base import Message

        if num_candidates < 1:
            num_candidates = 1
        if num_candidates > 4:
            num_candidates = 4  # 上限 4 个，避免 API 开销

        current_src = tool_path.read_text(encoding="utf-8") if tool_path.exists() else ""
        evolution_template = self._cfg.load_prompt("evolution")

        base_prompt = evolution_template.replace("{{tool_name}}", tool_name)
        base_prompt = base_prompt.replace("{{current_source}}", current_src[:3000])
        base_prompt = base_prompt.replace("{{failure_summary}}", feedback[:1000])

        # 每个候选的 system_msg + temperature 策略
        candidate_strategies: list[tuple[str, float | None]] = [
            (
                "你是 lingzhou 的自进化模块。请对工具做【最小改动】修复，保持现有结构，只修改出问题的代码。输出完整 Python 代码。",
                0.2,
            ),
            (
                "你是 lingzhou 的自进化模块。请【完全重写】工具，追求更好的错误处理、更清晰的逻辑，同时修复反馈中的问题。输出完整 Python 代码。",
                0.7,
            ),
            (
                "你是 lingzhou 的自进化模块。用【折中策略】改进工具：保留核心逻辑，重构出问题的部分，补充防御性检查。输出完整 Python 代码。",
                0.5,
            ),
            (
                "你是 lingzhou 的自进化模块。从用户视角思考工具应该如何工作，然后重写使其行为更符合预期。输出完整 Python 代码。",
                0.6,
            ),
        ]

        _log.info("[competitive_evolve] 开始竞争进化 tool=%r candidates=%d", tool_name, num_candidates)

        # 并行生成所有候选
        async def _gen_candidate(idx: int) -> tuple[int, str]:
            sys_msg, temp = candidate_strategies[idx % len(candidate_strategies)]
            msgs = [
                Message(role="system", content=sys_msg),
                Message(role="user", content=base_prompt),
            ]
            try:
                raw = await self._provider.chat(msgs, temperature=temp)
                code = _extract_python(raw)
                compile(code, tool_path.name, "exec")  # 基础语法检查
                return idx, code
            except Exception as exc:
                _log.debug("[competitive_evolve] 候选 %d 生成失败: %s", idx, exc)
                return idx, ""

        # asyncio.gather 并行生成
        gen_tasks = [_gen_candidate(i) for i in range(num_candidates)]
        gen_results: list[tuple[int, str]] = list(await asyncio.gather(*gen_tasks))

        # smoke test 评分
        _project_root = Path(__file__).parent.parent
        scored: list[tuple[int, int, str]] = []  # (score, idx, code)

        for idx, code in gen_results:
            if not code:
                _log.debug("[competitive_evolve] 候选 %d 生成空代码，跳过", idx)
                continue

            smoke_err = self._smoke_test_module(code, tool_path, _project_root)
            if smoke_err:
                _log.debug("[competitive_evolve] 候选 %d smoke 失败: %s", idx, smoke_err[:100])
                continue

            score = _score_candidate(code)
            _log.info("[competitive_evolve] 候选 %d smoke PASS score=%d", idx, score)
            scored.append((score, idx, code))

        if not scored:
            _log.warning("[competitive_evolve] 所有候选均未通过 smoke，回退到标准 evolve_tool")
            return await self.evolve_tool(tool_name, tool_path, feedback)

        # 取最高分候选
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx, best_code = scored[0]
        _log.info("[competitive_evolve] 最优候选=%d score=%d（共 %d 个通过 smoke）",
                  best_idx, best_score, len(scored))

        # 走标准热重载路径（复用 evolve_tool 的 backup + reload + verification 逻辑）
        # 通过替换 provider 响应模拟"用 best_code 做 evolve"
        return await self._promote_candidate(tool_name, tool_path, best_code, best_idx, best_score)

    async def _promote_candidate(
        self,
        tool_name: str,
        tool_path: Path,
        code: str,
        candidate_idx: int,
        score: int,
    ) -> EvolutionResult:
        """将通过竞争评估的候选代码直接写入生产路径并热加载。"""
        current_src = tool_path.read_text(encoding="utf-8") if tool_path.exists() else ""

        # 备份
        backup_path = tool_path.with_suffix(".py.bak")
        if self._cfg.evolution.backup and tool_path.exists():
            backup_path.write_text(current_src, encoding="utf-8")
            _clean_old_backups(tool_path)

        tool_path.write_text(code, encoding="utf-8")

        module_name = f"tools.{tool_path.stem}"
        try:
            self._reload_module_from_path(module_name, tool_path)
            if not self._tool_manifest_is_present(tool_name):
                raise RuntimeError(f"热重载后未注册目标工具: {tool_name}")
        except Exception:
            self._restore_text(tool_path, current_src)
            try:
                self._reload_module_from_path(module_name, tool_path)
            except Exception:
                pass
            raise

        _log.info("[competitive_evolve] 候选 %d (score=%d) 晋升为生产版本: %r",
                  candidate_idx, score, tool_name)
        await self._update_dreams(f"竞争进化完成：候选 {candidate_idx} 以评分 {score} 赢得 {tool_name} 改进权")
        return EvolutionResult(
            success=True,
            target=tool_name,
            new_code=code,
            reason=f"competitive_evolve: candidate={candidate_idx} score={score}",
        )

    async def _update_dreams(self, trigger_desc: str, ctx: "ToolContext | None" = None) -> None:
        """进化成功后，追加一条真实的志向到 DREAMS.md，并写入持久历史 fact。

        LLM 根据刚刚发生的进化事件，用第一人称写一句新的长期志向（≤40字）。
        追加到 DREAMS.md 末尾——这是灵舟自己写给自己的成长记录，不是摘要。
        同时在 DB 写一条 evolution:history:{ts} fact，保证身份叙事可追溯。
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
            # ── 持久化结构化历史 fact（幂等，按毫秒时间戳去重） ──
            if ctx is not None:
                try:
                    _fact_ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:17]
                    _fact_key = f"evolution:history:{_fact_ts}"
                    _fact_val = json.dumps({
                        "desc": trigger_desc[:200],
                        "aspiration": aspiration[:120],
                        "at": datetime.now(timezone.utc).isoformat(),
                    }, ensure_ascii=False)
                    await ctx.task_store.set_fact(_fact_key, _fact_val, scope="system")
                    _log.debug("[evolution] 历史 fact 已写入: %s", _fact_key)
                except Exception as _fe:
                    _log.debug("[evolution] 历史 fact 写入跳过: %s", _fe)
        except Exception as exc:
            _log.debug("[evolution] DREAMS.md 更新跳过: %s", exc)


def _score_candidate(code: str) -> int:
    """对候选代码进行静态质量评分（越高越好），用于竞争进化排名。

    评分维度（启发式）：
      +100 基础分（通过 smoke 保证）
      +10  每有一个 except 块（防御性错误处理）
      +5   有 logging / _log.（可观测性）
      +5   代码行数适中（50~300 行：不太短也不太长）
      -10  过长代码（>400 行，可能有多余内容）
      -5   有 print() 语句（生产代码不应有）
    """
    score = 100
    lines = code.splitlines()
    n = len(lines)

    # 错误处理
    except_count = sum(1 for l in lines if l.strip().startswith("except"))
    score += min(except_count * 10, 50)  # 最多 +50

    # 日志可观测性
    if "_log." in code or "logging." in code:
        score += 5

    # 代码长度适中
    if 50 <= n <= 300:
        score += 5
    elif n > 400:
        score -= 10

    # 惩罚 print 语句
    print_count = sum(1 for l in lines if l.strip().startswith("print("))
    score -= print_count * 5

    return score


def _extract_python(text: str) -> str:
    """从 LLM 输出中提取 Python 代码块。"""
    import re
    match = re.search(r"```(?:python)?\s*([\s\S]+?)```", text)
    if match:
        return match.group(1).strip()
    return text.strip()
