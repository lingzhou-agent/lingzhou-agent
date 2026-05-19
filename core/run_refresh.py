from __future__ import annotations

import json
import logging
from typing import Any, cast

from core.execution import build_meta_reflection, record_meta_reflection
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory, MemoryNode
from memory.task_store import TaskStore, Run
from tools.registry import ToolResult

_log = logging.getLogger("lingzhou.loop")

_RUN_PROGRESS_CRYSTAL_CHARS = 120


def _run_monitor_config(run: Run) -> dict[str, Any] | None:
    candidates = [
        run.extras.get("run_monitor"),
        (run.output_json.get("state_delta") or {}).get("run_monitor") if isinstance(run.output_json, dict) else None,
        (run.output_json.get("metadata") or {}).get("run_monitor") if isinstance(run.output_json, dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        kind = str(candidate.get("kind") or "").strip()
        if kind == "fact" and str(candidate.get("key") or "").strip():
            return candidate
        if kind == "process" and str(candidate.get("session_id") or "").strip():
            return candidate
    if run.session_id:
        return {"kind": "process", "session_id": run.session_id}
    return None


def _parse_run_monitor_snapshot(raw: str, monitor: dict[str, Any]) -> tuple[str, str, Any]:
    payload: Any = raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = raw

    status_field = str(monitor.get("status_field") or "status")
    progress_field = str(monitor.get("progress_field") or "progress")
    success_values = {str(v).strip().lower() for v in (monitor.get("success_values") or ["succeeded", "success", "done", "completed"])}
    failed_values = {str(v).strip().lower() for v in (monitor.get("failed_values") or ["failed", "error"])}
    cancelled_values = {str(v).strip().lower() for v in (monitor.get("cancelled_values") or ["cancelled", "canceled"])}

    status_value: Any = raw
    progress_value: Any = ""
    if isinstance(payload, dict):
        status_value = payload.get(status_field, payload.get("status", raw))
        progress_value = payload.get(progress_field, payload.get("progress", ""))

    normalized = str(status_value or "").strip().lower()
    if normalized in success_values:
        status = "succeeded"
    elif normalized in failed_values:
        status = "failed"
    elif normalized in cancelled_values:
        status = "cancelled"
    else:
        status = "running"
    progress = str(progress_value or "").strip()[:2000]
    return status, progress, payload


def _record_refreshed_run_outcome(
    episodic: EpisodicMemory | None,
    semantic: SemanticMemory | None,
    *,
    run: Run,
    status: str,
    progress: str,
    summary: str,
    error: str,
) -> None:
    if episodic is not None:
        episodic.record_event(
            "run_failed" if status == "failed" or error else "run_completed",
            {
                "run_id": run.id,
                "task_id": run.task_id,
                "tool_name": run.tool_name,
                "worker_type": run.worker_type,
                "status": status,
                "summary": summary[:800],
                "error": error[:400],
            },
        )
    if semantic is None:
        return
    semantic.upsert(MemoryNode(
        id=f"run-result-{run.id}",
        kind="run_result",
        title=f"[run#{run.id}] {run.tool_name or 'unknown'} {status}",
        body=(
            f"status={status}\n"
            f"tool={run.tool_name or 'unknown'}\n"
            f"progress={progress}\n"
            f"summary={summary}\n"
            f"error={error}"
        )[:4000],
        activation=0.82 if status == "failed" or error else 0.72,
        valence=0.35 if status == "failed" or error else 0.65,
        tags=[x for x in [status, run.tool_name, run.worker_type, f"task:{run.task_id}" if run.task_id else ""] if x],
    ))


async def _finalize_refreshed_run_learning(
    task_store: TaskStore,
    *,
    run: Run,
    status: str,
    summary: str,
    error: str,
    evidence: str,
    episodic: EpisodicMemory | None = None,
    semantic: SemanticMemory | None = None,
) -> None:
    if error:
        _log.debug("[run-refresh] record failure for run=%s tool=%s error=%s", run.id, run.tool_name, error)
        await task_store.record_failure(
            kind=run.tool_name or run.worker_type or "run",
            summary=summary[:300],
            context=evidence[:200],
            task_id=str(run.task_id) if run.task_id else "",
        )

    result = ToolResult(
        summary=summary,
        evidence=evidence,
        error=error,
        skipped=(status == "cancelled"),
        kind="execute_result",
        metadata={
            "tool_name": run.tool_name,
            "worker_type": run.worker_type,
            "session_id": run.session_id,
        },
    )
    meta = build_meta_reflection(
        run_id=run.id,
        task_id=run.task_id,
        tool_name=run.tool_name,
        result=result,
    )
    if not meta:
        return
    _log.debug(
        "[run-refresh] meta reflection run=%s target=%s decision=%s",
        run.id,
        meta.get("target_kind"),
        meta.get("decision"),
    )
    await task_store.add_meta_reflection(
        reflection_id=str(meta["reflection_id"]),
        target_kind=str(meta["target_kind"]),
        trigger=str(meta["trigger"]),
        loop_level=str(meta["loop_level"]),
        diagnosis=str(meta["diagnosis"]),
        proposal=str(meta["proposal"]),
        verification_plan=str(meta["verification_plan"]),
        decision=str(meta["decision"]),
        task_id=int(meta["task_id"]),
        run_id=int(meta["run_id"]),
        tool_name=str(meta["tool_name"]),
    )
    if episodic is None and semantic is None:
        return

    class _ReflectionCtx:
        def __init__(self) -> None:
            self.episodic = episodic
            self.semantic = semantic

    record_meta_reflection(cast(Any, _ReflectionCtx()), meta)


async def _refresh_run_via_fact_monitor(
    task_store: TaskStore,
    run: Run,
    monitor: dict[str, Any],
    *,
    episodic: EpisodicMemory | None = None,
    semantic: SemanticMemory | None = None,
) -> dict[str, Any]:
    key = str(monitor.get("key") or "").strip()
    raw, found = await task_store.get_fact(key)
    if not found:
        return {"run_id": run.id, "task_id": run.task_id, "status": run.status}

    status, progress, payload = _parse_run_monitor_snapshot(raw, monitor)
    crystal = progress if progress and progress != run.progress else ""
    output_json = dict(run.output_json)
    output_json["monitor_snapshot"] = payload
    output_json["monitor_key"] = key
    await task_store.update_run(
        run.id,
        status=status,
        output_json=output_json,
        log_text=(progress or str(payload or raw))[:4000],
        error_text=(str(payload.get("error") or "") if isinstance(payload, dict) else (raw[:400] if status == "failed" else "")),
        progress=progress,
    )
    if run.task_id and crystal:
        await task_store.set_fact(f"task:{run.task_id}:progress", crystal[:800], scope="task")
    if status in {"succeeded", "failed", "cancelled"}:
        summary = progress or (str(payload.get("summary") or "") if isinstance(payload, dict) else raw) or f"run {status}"
        error = str(payload.get("error") or "") if isinstance(payload, dict) else (raw if status == "failed" else "")
        if run.task_id:
            await task_store.update_task_result(
                run.task_id,
                {
                    "last_run_id": run.id,
                    "last_run_status": status,
                    "worker_type": run.worker_type,
                    "tool_name": run.tool_name,
                    "session_id": run.session_id,
                    "summary": summary,
                    "error": error or None,
                },
            )
        _record_refreshed_run_outcome(
            episodic,
            semantic,
            run=run,
            status=status,
            progress=progress,
            summary=summary,
            error=error,
        )
        await _finalize_refreshed_run_learning(
            task_store,
            run=run,
            status=status,
            summary=summary,
            error=error,
            evidence=str(payload if isinstance(payload, dict) else raw)[:1200],
            episodic=episodic,
            semantic=semantic,
        )
    return {
        "run_id": run.id,
        "task_id": run.task_id,
        "status": status,
        "session_id": run.session_id,
        "crystal": crystal[:400],
    }


async def _refresh_run_via_process_monitor(
    task_store: TaskStore,
    run: Run,
    monitor: dict[str, Any],
    *,
    manager: Any,
    episodic: EpisodicMemory | None = None,
    semantic: SemanticMemory | None = None,
) -> dict[str, Any]:
    session_id = str(monitor.get("session_id") or run.session_id or "").strip()
    if not session_id or manager is None:
        return {"run_id": run.id, "task_id": run.task_id, "status": run.status}

    info = manager.get(session_id)
    if info is None:
        return {"run_id": run.id, "task_id": run.task_id, "status": run.status}
    if not info.finished:
        _log.debug("[run-monitor] process session=%s run=%s still running", session_id, run.id)
        stdout_text = info.stdout or ""
        last_crystal_chars = int(run.extras.get("last_crystal_chars", 0) or 0)
        crystal_excerpt = ""
        if len(stdout_text) - last_crystal_chars >= _RUN_PROGRESS_CRYSTAL_CHARS:
            crystal_excerpt = stdout_text[last_crystal_chars:][-400:].strip()
            await task_store.update_run(
                run.id,
                status="running",
                output_json={**run.output_json, "progress_excerpt": crystal_excerpt},
                log_text=stdout_text[-4000:],
                session_id=session_id,
                progress=crystal_excerpt[:2000],
                extras={**run.extras, "last_crystal_chars": len(stdout_text), "run_monitor": monitor},
            )
            if run.task_id and crystal_excerpt:
                await task_store.set_fact(f"task:{run.task_id}:progress", crystal_excerpt[:800], scope="task")
        return {
            "run_id": run.id,
            "task_id": run.task_id,
            "status": "running",
            "session_id": session_id,
            "crystal": crystal_excerpt,
        }

    status = "succeeded" if (info.return_code in (0, None) and not info.timed_out and not info.error) else "failed"
    _log.info("[run-monitor] process session=%s run=%s finished status=%s", session_id, run.id, status)
    output_json = dict(run.output_json)
    output_json.update({
        "session_id": session_id,
        "return_code": info.return_code,
        "timed_out": info.timed_out,
        "stdout": info.stdout[-4000:],
        "stderr": info.stderr[-2000:],
        "error": info.error,
    })
    await task_store.update_run(
        run.id,
        status=status,
        output_json=output_json,
        log_text=info.stdout[-4000:],
        error_text=info.error or ("timed_out" if info.timed_out else (f"exit_code={info.return_code}" if status == "failed" else "")),
        session_id=session_id,
        progress=(info.stdout or info.stderr or info.error or status)[-800:].strip(),
        extras={
            **run.extras,
            "return_code": info.return_code,
            "timed_out": info.timed_out,
            "background": info.background,
            "run_monitor": monitor,
        },
    )
    if run.task_id:
        await task_store.update_task_result(
            run.task_id,
            {
                "last_run_id": run.id,
                "last_run_status": status,
                "worker_type": run.worker_type,
                "tool_name": run.tool_name,
                "session_id": session_id,
                "summary": output_json.get("stdout", "")[:200] or f"process {status}",
                "error": output_json.get("error"),
            },
        )
    _record_refreshed_run_outcome(
        episodic,
        semantic,
        run=run,
        status=status,
        progress=(info.stdout or info.stderr or info.error or status)[-800:].strip(),
        summary=output_json.get("stdout", "")[:200] or f"process {status}",
        error=str(output_json.get("error") or ""),
    )
    await _finalize_refreshed_run_learning(
        task_store,
        run=run,
        status=status,
        summary=output_json.get("stdout", "")[:200] or f"process {status}",
        error=str(output_json.get("error") or ""),
        evidence=((info.stderr or "") + "\n" + (info.error or "")).strip()[:1200],
        episodic=episodic,
        semantic=semantic,
    )
    return {"run_id": run.id, "task_id": run.task_id, "status": status, "session_id": session_id}


async def refresh_running_runs(
    task_store: TaskStore,
    *,
    episodic: EpisodicMemory | None = None,
    semantic: SemanticMemory | None = None,
) -> list[dict[str, Any]]:
    """刷新所有 running runs，优先走内建 exec 监控，其次走通用 fact-backed run_monitor 协议。"""
    try:
        from tools.exec import _MANAGER
    except Exception:
        _MANAGER = None

    updates: list[dict[str, Any]] = []
    for run in await task_store.list_runs(status="running", limit=20):
        monitor = _run_monitor_config(run)
        if monitor is not None:
            if str(monitor.get("kind") or "") == "fact":
                updates.append(await _refresh_run_via_fact_monitor(task_store, run, monitor, episodic=episodic, semantic=semantic))
            elif str(monitor.get("kind") or "") == "process":
                updates.append(await _refresh_run_via_process_monitor(task_store, run, monitor, manager=_MANAGER, episodic=episodic, semantic=semantic))
            else:
                updates.append({"run_id": run.id, "task_id": run.task_id, "status": run.status})
            continue
        if run.worker_type != "exec-worker" or not run.session_id or _MANAGER is None:
            updates.append({"run_id": run.id, "task_id": run.task_id, "status": run.status})
            continue
        updates.append(await _refresh_run_via_process_monitor(task_store, run, {"kind": "process", "session_id": run.session_id}, manager=_MANAGER, episodic=episodic, semantic=semantic))
    return updates