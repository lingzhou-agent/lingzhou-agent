"""tools/probe_ops.py — 探针系统工具集。

提供 LLM 可直接调用的工具，用于安装、移除、列出、立即执行探针。

探针让灵舟可以像人布置传感器一样自由感知外部世界：
- 服务器温度、负载、磁盘、网络延迟
- 股票/汇率/天气等 HTTP 数据流
- 自定义 Python 计算结果
- 任何 shell 命令的周期输出

参考模型：
- Prometheus Exporter 模式（具名采集器 + 定时抓取）
- Brooks (1986) Subsumption Architecture（独立感知子系统）
- Weiser (1991) Ubiquitous Computing（环境传感器提供上下文）
"""
from __future__ import annotations

import logging
from typing import Any

from tools.registry import ToolContext, ToolManifest, ToolParam, ToolResult, tool
from core.probe.types import ProbeConfig

_log = logging.getLogger("lingzhou.probe")

# ── 工具实现 ──────────────────────────────────────────────────────────────────


@tool(ToolManifest(
    name="probe.install",
    description=(
        "安装（或更新）一个探针传感器。\n\n"
        "interval 探针会在后台周期运行，结果写入工作记忆（wm）下一轮可见；\n"
        "manual 探针仅在你主动调用 probe.run 时执行，结果直接返回。\n\n"
        "示例（每 60 秒监控 CPU 温度）：\n"
        "  kind=shell  spec='cat /sys/class/thermal/thermal_zone0/temp'\n"
        "  trigger='interval:60'  data_back='wm'\n"
        "  alert_expr='int(output.strip()) > 75000'"
    ),
    params=[
        ToolParam("name", "string", "探针唯一名称（字母数字下划线横线）", required=True),
        ToolParam("purpose", "string", "部署目的/原因，如“监控磁盘使用率，防止消耗过快导致任务失败”", required=True),
        ToolParam("kind", "string", "执行方式：shell | http | python", required=True),
        ToolParam("spec", "string", "命令字符串 / URL / Python 代码（对应 kind）", required=True),
        ToolParam("trigger", "string", "调度触发器：interval:<秒> 或 manual", required=True),
        ToolParam("data_back", "string", "interval 探针结果回传：wm（写入工作记忆）| none（仅日志），默认 wm", required=False),
        ToolParam("alert_expr", "string", "告警表达式（Python bool，变量 output 为结果字符串）", required=False),
        ToolParam("alert_message", "string", "告警消息文本，支持 {output} 占位符", required=False),
    ],
    prefer_tier="reasoner",
    progress_category="mutation",
))
async def probe_install(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    probe_mgr = _get_probe_manager(ctx)
    if probe_mgr is None:
        return ToolResult(summary="探针系统未初始化", error="ProbeManagerNotFound", skipped=True)

    name = str(params.get("name") or "").strip()
    if not name:
        return ToolResult(summary="name 不能为空", error="InvalidParam", skipped=True)

    kind = str(params.get("kind") or "").strip().lower()
    if kind not in ("shell", "http", "python"):
        return ToolResult(summary=f"kind 无效: {kind}，可选 shell / http / python", error="InvalidParam", skipped=True)

    spec = str(params.get("spec") or "").strip()
    if not spec:
        return ToolResult(summary="spec 不能为空", error="InvalidParam", skipped=True)

    trigger = str(params.get("trigger") or "").strip()
    if not trigger:
        return ToolResult(summary="trigger 不能为空，格式：interval:<秒> 或 manual", error="InvalidParam", skipped=True)

    data_back_raw = str(params.get("data_back") or "wm").strip().lower()
    if data_back_raw not in ("none", "wm"):
        data_back_raw = "wm"

    cfg = ProbeConfig(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        spec=spec,
        trigger=trigger,
        purpose=str(params.get("purpose") or "").strip(),
        data_back=data_back_raw,  # type: ignore[arg-type]
        alert_expr=str(params.get("alert_expr") or "") or None,
        alert_message=str(params.get("alert_message") or "") or None,
        enabled=True,
    )

    try:
        saved = await probe_mgr.install(cfg)
    except Exception as exc:
        _log.exception("[probe.install] 失败: %s", exc)
        return ToolResult(summary=f"安装探针失败: {exc}", error=type(exc).__name__)

    first_run_hint = (
        "  ℹ 立即用 probe.run 获取首次读数（interval 探针首次执行需等满一个间隔）"
        if saved.trigger.startswith("interval:")
        else ""
    )
    return ToolResult(
        summary=(
            f"探针已安装: {saved.name}\n"
            f"  purpose={saved.purpose or '（未填写）'}\n"
            f"  kind={saved.kind}  trigger={saved.trigger}  data_back={saved.data_back}\n"
            f"  spec={saved.spec[:80]}"
            + ("\n" + first_run_hint if first_run_hint else "")
        ),
        state_delta={"probe": "installed", "name": saved.name},
    )


@tool(ToolManifest(
    name="probe.remove",
    description="移除（收回）一个已安装的探针，立即停止其调度。",
    params=[
        ToolParam("name", "string", "要移除的探针名称", required=True),
    ],
    prefer_tier="reasoner",
    progress_category="mutation",
))
async def probe_remove(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    probe_mgr = _get_probe_manager(ctx)
    if probe_mgr is None:
        return ToolResult(summary="探针系统未初始化", error="ProbeManagerNotFound", skipped=True)

    name = str(params.get("name") or "").strip()
    if not name:
        return ToolResult(summary="name 不能为空", error="InvalidParam", skipped=True)

    deleted = await probe_mgr.remove(name)
    if not deleted:
        return ToolResult(summary=f"探针不存在: {name}", error="NotFound", skipped=True)

    return ToolResult(
        summary=f"探针已移除: {name}",
        state_delta={"probe": "removed", "name": name},
    )


@tool(ToolManifest(
    name="probe.run",
    description=(
        "立即执行指定探针（无论 trigger 配置），获取当前数据快照。\n"
        "结果始终直接返回，不受 data_back 设置影响。"
    ),
    params=[
        ToolParam("name", "string", "探针名称", required=True),
    ],
    prefer_tier="reasoner",
    progress_category="info",
))
async def probe_run(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    probe_mgr = _get_probe_manager(ctx)
    if probe_mgr is None:
        return ToolResult(summary="探针系统未初始化", error="ProbeManagerNotFound", skipped=True)

    name = str(params.get("name") or "").strip()
    result = await probe_mgr.run_now(name)
    if result is None:
        return ToolResult(summary=f"探针不存在: {name}", error="NotFound", skipped=True)

    if result.error:
        return ToolResult(
            summary=f"[探针 {name}] ❌ 错误: {result.error}",
            error=result.error,
        )

    lines = [f"[探针 {name}] {result.triggered_at} ({result.duration_ms}ms)"]
    if result.output:
        lines.append(result.output[:2000])
    if result.alerted and result.alert_detail:
        lines.append(f"🔔 告警: {result.alert_detail}")

    return ToolResult(
        summary="\n".join(lines),
        evidence=result.output,
        state_delta={"probe_name": name, "alerted": result.alerted},
    )


@tool(ToolManifest(
    name="probe.list",
    description="列出所有已安装的探针及其最近执行状态。",
    params=[],
    prefer_tier="reader",
    progress_category="info",
))
async def probe_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    probe_mgr = _get_probe_manager(ctx)
    if probe_mgr is None:
        return ToolResult(summary="探针系统未初始化", error="ProbeManagerNotFound", skipped=True)

    probes = await probe_mgr.list_probes()
    if not probes:
        return ToolResult(summary="当前没有已安装的探针。使用 probe.install 安装第一个探针。")

    task_statuses = probe_mgr.runner_status()
    lines: list[str] = [f"共 {len(probes)} 个探针：\n"]
    for p in probes:
        status = "⏸ 禁用" if not p.enabled else task_statuses.get(p.name, "✅ 运行中")
        last = p.last_run_at or "从未运行"
        result_preview = ""
        if p.last_error:
            result_preview = f"  最近错误: {p.last_error[:80]}"
        elif p.last_result:
            result_preview = f"  最近结果: {p.last_result[:80]}"
        purpose_line = f"\n  目的: {p.purpose}" if getattr(p, "purpose", "") else ""
        lines.append(
            f"• {p.name} [{status}]{purpose_line}\n"
            f"  kind={p.kind}  trigger={p.trigger}  data_back={p.data_back}\n"
            f"  最近运行: {last}{result_preview}"
        )

    return ToolResult(summary="\n".join(lines))


@tool(ToolManifest(
    name="probe.disable",
    description="暂停（禁用）一个探针：停止其调度但保留配置，可用 probe.enable 恢复。",
    params=[
        ToolParam("name", "string", "要禁用的探针名称", required=True),
    ],
    prefer_tier="reasoner",
    progress_category="mutation",
))
async def probe_disable(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    probe_mgr = _get_probe_manager(ctx)
    if probe_mgr is None:
        return ToolResult(summary="探针系统未初始化", error="ProbeManagerNotFound", skipped=True)
    name = str(params.get("name") or "").strip()
    if not name:
        return ToolResult(summary="name 不能为空", error="InvalidParam", skipped=True)
    ok = await probe_mgr.set_enabled(name, False)
    if not ok:
        return ToolResult(summary=f"探针不存在: {name}", error="NotFound", skipped=True)
    return ToolResult(summary=f"探针已暂停: {name}（配置保留，用 probe.enable 恢复）",
                      state_delta={"probe": "disabled", "name": name})


@tool(ToolManifest(
    name="probe.enable",
    description="恢复（启用）一个已暂停的探针，立即重启调度。",
    params=[
        ToolParam("name", "string", "要启用的探针名称", required=True),
    ],
    prefer_tier="reasoner",
    progress_category="mutation",
))
async def probe_enable(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    probe_mgr = _get_probe_manager(ctx)
    if probe_mgr is None:
        return ToolResult(summary="探针系统未初始化", error="ProbeManagerNotFound", skipped=True)
    name = str(params.get("name") or "").strip()
    if not name:
        return ToolResult(summary="name 不能为空", error="InvalidParam", skipped=True)
    ok = await probe_mgr.set_enabled(name, True)
    if not ok:
        return ToolResult(summary=f"探针不存在: {name}", error="NotFound", skipped=True)
    return ToolResult(summary=f"探针已恢复运行: {name}",
                      state_delta={"probe": "enabled", "name": name})


# ── 内部工具函数 ────────────────────────────────────────────────────────────────

def _get_probe_manager(ctx: ToolContext) -> Any:
    """从 ToolContext.probe_manager 获取 ProbeManager。"""
    return ctx.probe_manager
