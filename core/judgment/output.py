"""core/judgment/output.py — JudgmentOutput 数据模型及工具分层常量。

从 runtime.py 分离，避免 runtime.py 超长。
依赖：无 core.* 循环（仅用 tools.registry.tool_has_capability）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tools.registry import tool_has_capability

if TYPE_CHECKING:
    from tools.registry import ToolRegistry

# ── 工具分层常量 ─────────────────────────────────────────────────────────────

# 低成本读取/枚举类工具 → reader tier（也被 loop.py 用于自动 tier 推断）
_READER_TOOLS = frozenset({
    "file.list", "file.read",
    "memory.get_fact", "memory.search",
    "schedule.list", "schedule.ack", "schedule.cancel",
    "shell.capabilities",
    "task.list",
    "failure.dismiss",
    # 探针：只读列举，不触发执行
    "probe.list",
})
READER_TOOLS = _READER_TOOLS  # 公开别名，供外部模块引用

# 写入/推理/高风险工具 → reasoner tier
_REASONER_TOOLS = frozenset({
    "shell.run",
    "file.write",
    "task.add", "task.update", "task.advance", "task.complete", "task.fail",
    "memory.add_wm", "memory.add_semantic", "memory.set_fact", "memory.snapshot",
    "reflect.structural",
    "schedule.add",
    # 探针：安装/拆除/执行/启停均有副作用
    "probe.install", "probe.remove", "probe.run", "probe.enable", "probe.disable",
})

_TOOL_TIER_MAPPING = {
    "reader": sorted(_READER_TOOLS),
    "reasoner": sorted(_REASONER_TOOLS),
    "repair": [],
}

# ── 续判前置改写：证据预算门 ────────────────────────────────────────────────────

_ASK_EVIDENCE_BUDGET = 2  # task.ask 前须有至少 2 次有效证据工具调用


def _rewrite_task_ask_to_evidence(
    action: "JudgmentOutput",
    *,
    user_message: str = "",
    tool_history: "list[dict[str, Any]] | None" = None,
    registry: "Any | None" = None,
) -> "JudgmentOutput":
    """若 task.ask 前证据不足，改写为先调用 task.list 收集证据。"""
    hits = 0
    for item in tool_history or []:
        tool_id = str(item.get("tool") or "")
        result = str(item.get("result") or "").strip()
        if not result or result.startswith("ERROR["):
            continue
        if registry is not None and tool_has_capability(registry, tool_id, "ask_evidence"):
            hits += 1
    if hits >= _ASK_EVIDENCE_BUDGET:
        return action
    return JudgmentOutput(
        decision="act",
        chosen_action_id="task.list",
        params={"status": "all", "limit": 10},
        rationale=f"证据预算 {hits}/{_ASK_EVIDENCE_BUDGET}，先收集任务列表作为佐证再提问。",
        reflection=action.reflection,
        reply_to_user="",
        next_step=action.next_step,
        model_strategy={"next_phase_tier": "reasoner"},
        applied_skills=list(action.applied_skills or []),
    )


# ── 续判前置改写：复杂 mutation → task.plan ─────────────────────────────────────

def _rewrite_complex_act_to_task_plan(
    action: "JudgmentOutput",
    *,
    user_message: str = "",
    active_task: "Any | None" = None,
    registry: "Any | None" = None,
) -> "JudgmentOutput":
    """若 LLM 对复杂请求直接输出 mutation 且有 next_step，先拆为 task.plan。

    只对非读取类工具生效；file.read / memory.search 等读取工具直接透传。
    任务管理类工具（task.complete/advance/update/fail/add）是状态转换动作，不改写。
    """
    tool_id = action.chosen_action_id or ""
    if tool_id in _READER_TOOLS:
        return action
    if tool_id.startswith("task."):
        return action
    if active_task is None:
        return action
    next_step = (action.next_step or "").strip()
    if not next_step:
        return action
    # 豁免：任务已有 plan 且存在 in_progress 步骤 → LLM 正在执行计划中的步骤，无需再次 plan
    existing_plan = (getattr(active_task, "extras", None) or {}).get("plan") or []
    if isinstance(existing_plan, list) and any(
        isinstance(s, dict) and s.get("status") == "in_progress"
        for s in existing_plan
    ):
        return action
    step1_desc = f"执行 {tool_id}"
    params = action.params or {}
    for key in ("command", "path", "title", "query"):
        if key in params:
            snippet = str(params[key])[:40]
            step1_desc = f"执行 {tool_id}（{snippet}）"
            break
    plan = [
        {"step": step1_desc, "status": "in_progress"},
        {"step": next_step, "status": "pending"},
    ]
    return JudgmentOutput(
        decision="act",
        chosen_action_id="task.plan",
        params={"task_id": active_task.id, "plan": plan},
        rationale=action.rationale,
        reflection=action.reflection,
        reply_to_user="",
        next_step="",
        model_strategy=dict(action.model_strategy or {}),
        applied_skills=list(action.applied_skills or []),
    )


def _structured_tool_history_window(tool_history: list[dict[str, Any]]) -> tuple[str, str]:
    history_parts: list[str] = []
    structured_window: list[dict[str, Any]] = []
    start_index = max(0, len(tool_history) - 6)
    for index, item in enumerate(tool_history[start_index:], start=start_index + 1):
        raw_params = item.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        status = str(item.get("status") or "").strip() or (
            "error" if str(item.get("error") or "").strip() else ("skipped" if item.get("skipped") else "ok")
        )
        summary = str(item.get("summary") or item.get("result") or "").strip()
        error = str(item.get("error") or "").strip()
        state_delta = item.get("state_delta") if isinstance(item.get("state_delta"), dict) else {}
        key = (
            params.get("path")
            or params.get("name")
            or params.get("title")
            or params.get("key")
            or str(params.get("id") or "")
            or params.get("command")
            or params.get("query")
            or ""
        )
        structured_window.append({
            "index": index,
            "tool": str(item.get("tool") or ""),
            "status": status,
            "key": str(key),
            "summary": summary[:400],
            "error": error[:240],
            "error_category": str(item.get("error_category") or ""),
            "state_delta": state_delta,
        })
        parts = [f"[{index}] tool={item.get('tool', '')} status={status}"]
        if key:
            parts.append(f"key={key}")
        if summary:
            parts.append(f"summary={summary[:240]}")
        if error:
            parts.append(f"error={error[:180]}")
        if state_delta:
            try:
                state_text = json.dumps(state_delta, ensure_ascii=False, sort_keys=True)
            except Exception:
                state_text = str(state_delta)
            parts.append(f"state_delta={state_text[:200]}")
        history_parts.append(" | ".join(parts))
    return (
        json.dumps(structured_window, ensure_ascii=False, indent=2),
        "\n".join(history_parts),
    )


def _build_team_view_from_cfg(cfg: Any) -> str:
    """从配置构建思考模型的团队视图（不依赖运行时 model list）。"""
    routing = getattr(cfg, "routing", {}) or {}
    lines = ["## 团队架构（你作为思考模型，统筹以下资源）"]
    for tier in ("reader", "reasoner", "repair"):
        model = routing.get(tier, cfg.model)
        role = {
            "reader": "操作层 — 快速/低成本，适合 file.read/task.list/memory.search 等轻量查询",
            "reasoner": "思考层 — 你本人，深度推理，适合复杂决策/代码修改/用户交互",
            "repair": "修复层 — 格式修复/小修小补",
        }.get(tier, tier)
        lines.append(f"- {tier}: {model}")
        lines.append(f"  {role}")
    lines.append("")
    lines.append("调度规则:")
    lines.append("- 你是 reasoner（思考层），负责复杂决策。简单操作应委派给 reader")
    lines.append("- 委派方式: 在 model_strategy 中设置 next_phase_tier='reader'")
    lines.append("- reader 的操作层模型成本低速度快，不要在 reasoner 上做纯查询")
    lines.append("- 关键决策、代码修改、用户交互必须由你亲自处理")
    return "\n".join(lines)


def tool_tier(tool_id: str, registry: "ToolRegistry | None" = None) -> str:
    """判断工具应该用哪个 tier。

    优先级：manifest.prefer_tier → 硬编码集合 → 默认 reasoner。
    数据驱动的工具可以声明 prefer_tier，无需改此处。
    """
    if registry is not None and tool_id:
        entry = registry.get(tool_id)
        if entry and entry.manifest.prefer_tier:
            return entry.manifest.prefer_tier
    if tool_id in _READER_TOOLS:
        return "reader"
    if tool_id in _REASONER_TOOLS:
        return "reasoner"
    return "reasoner"


# ── 路由/健康数据类 ──────────────────────────────────────────────────────────

@dataclass
class ModelSelection:
    phase: str
    tier: str
    model_ref: str
    thinking: str


@dataclass
class ModelHealth:
    cooldown_until: float = 0.0
    failure_streak: int = 0
    last_error: str = ""
    last_code: str = ""


# ── 判断输出 ───────────────────────────────────────────────────────────────────

@dataclass
class JudgmentOutput:
    decision: str = "wait"              # act | pause | wait
    chosen_action_id: str = ""          # 工具名称
    params: dict[str, Any] = field(default_factory=lambda: {})  # type: ignore[assignment]
    rationale: str = ""                 # 内部推理过程（内部独白）
    reflection: str = ""                # 对最近经历的后验反思（写入语义记忆）
    reply_to_user: str = ""             # 对人类的外部回复（与 rationale 明确分离）
    next_step: str = ""
    model_strategy: dict[str, Any] = field(default_factory=dict)
    applied_skills: list[str] = field(default_factory=list)  # LLM 实际应用的技能名单
    parallel_actions: list[dict[str, Any]] = field(default_factory=list)
    delegate_tasks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def wait(cls, reason: str = "") -> "JudgmentOutput":
        return cls(decision="wait", rationale=reason, reply_to_user="")

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @classmethod
    def from_llm(cls, text: str) -> "JudgmentOutput":
        """从 LLM 输出文本解析 JudgmentOutput，容错处理。"""
        original = text.strip()
        text = original
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        if not text.startswith("{") and not text.startswith("```"):
            stripped = re.sub(
                r"(?:^|\n)(?:[\w./_-]+/\s*\n)?(?:[│├└──+|\\]+[ \t]+[\w.+/_-]+[ \t]*(?:#.*)?\n)+",
                "\n",
                text,
            )
            if stripped != text:
                text = stripped.strip()
        if not text or ("{" not in text and "decision" not in text):
            return cls(decision="pause", rationale=f"LLM 输出解析失败（非JSON）: {original[:120]}")
        _CODE_PREFIXES = ("#!/", "```bash", "```python", "```sh", "```shell", "# -*-")
        _is_raw_code = any(text.lstrip().startswith(p) for p in _CODE_PREFIXES)
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if match:
            text = match.group(1).strip()
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            fixed = text.replace("'", '"')
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            try:
                data = json.loads(fixed)
            except json.JSONDecodeError:
                if _is_raw_code:
                    return cls(
                        decision="pause",
                        chosen_action_id="",
                        params={},
                        rationale="[auto-wrap] LLM 输出了裸代码，已封装为 reply_to_user",
                        reflection="格式错误：代码应放入 reply_to_user 或 params 字段，不能直接输出",
                        reply_to_user=original,
                        next_step="",
                        model_strategy={},
                    )
                return cls(decision="pause", rationale=f"LLM 输出解析失败: {text}")

        return cls(
            decision=cls._coerce_text(data.get("decision", "wait")).lower(),
            chosen_action_id=cls._coerce_text(data.get("chosen_action_id", "")),
            params=dict(data.get("params") or {}),
            rationale=cls._coerce_text(data.get("rationale", "")),
            reflection=cls._coerce_text(data.get("reflection", "")),
            reply_to_user=cls._coerce_text(data.get("reply_to_user", "")),
            next_step=cls._coerce_text(data.get("next_step", "")),
            model_strategy=dict(data.get("model_strategy") or {}),
            applied_skills=[str(s) for s in (data.get("applied_skills") or []) if s],
            parallel_actions=[
                item for item in (data.get("parallel_actions") or [])
                if isinstance(item, dict) and isinstance(item.get("action_id"), str) and item["action_id"]
            ],
            delegate_tasks=[
                item for item in (data.get("delegate_tasks") or [])
                if isinstance(item, dict) and item.get("id") and item.get("goal")
            ],
        )
