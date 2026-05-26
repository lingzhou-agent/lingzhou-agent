"""store/task/models.py — Task / Failure / Run / MetaReflection 数据对象。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_TASK_CORE_DATA_KEYS = frozenset({
    "goal",
    "source",
    "next_step",
    "chain_id",
    "parent_task_id",
    "current_step",
    "wait_kind",
    "wait_key",
    "state_json",
    "wait_json",
    "result_json",
    "async_job_id",
    "model_tier",
})


@dataclass
class Task:
    id: int
    title: str
    status: str
    priority: str
    created_at: str
    # 核心 data 字段（data JSON 的常用键）
    goal: str = ""
    source: str = "external"
    next_step: str = ""
    chain_id: str = ""
    parent_task_id: str = ""
    current_step: str = ""
    wait_kind: str = ""
    wait_key: str = ""
    state_json: dict[str, Any] = field(default_factory=dict[str, Any])
    wait_json: dict[str, Any] = field(default_factory=dict[str, Any])
    result_json: dict[str, Any] = field(default_factory=dict[str, Any])
    async_job_id: str = ""
    model_tier: str = ""
    # 其余 data 键，动态扩展无需代码变动
    extras: dict[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def from_row(cls, row: Any) -> "Task":
        """row = (id, title, status, priority, created_at, data_json)"""
        rid, title, status, priority, created_at, data_raw = row
        try:
            data: dict[str, Any] = json.loads(data_raw or "{}")
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        goal = data.pop("goal", "")
        source = data.pop("source", "external")
        next_step = data.pop("next_step", "")
        chain_id = data.pop("chain_id", "")
        parent_task_id = data.pop("parent_task_id", "")
        current_step = data.pop("current_step", "")
        wait_kind = data.pop("wait_kind", "")
        wait_key = data.pop("wait_key", "")
        state_json = data.pop("state_json", {}) or {}
        wait_json = data.pop("wait_json", {}) or {}
        result_json = data.pop("result_json", {}) or {}
        async_job_id = data.pop("async_job_id", "")
        model_tier = data.pop("model_tier", "")
        return cls(
            id=rid,
            title=title,
            status=status,
            priority=priority,
            created_at=created_at,
            goal=goal,
            source=source,
            next_step=next_step,
            chain_id=chain_id,
            parent_task_id=parent_task_id,
            current_step=current_step,
            wait_kind=wait_kind,
            wait_key=wait_key,
            state_json=state_json,
            wait_json=wait_json,
            result_json=result_json,
            async_job_id=async_job_id,
            model_tier=model_tier,
            extras=data,
        )

    def to_data_json(self) -> str:
        d = {
            "goal": self.goal,
            "source": self.source,
            "next_step": self.next_step,
            "chain_id": self.chain_id,
            "parent_task_id": self.parent_task_id,
            "current_step": self.current_step,
            "wait_kind": self.wait_kind,
            "wait_key": self.wait_key,
            "state_json": self.state_json,
            "wait_json": self.wait_json,
            "result_json": self.result_json,
            "async_job_id": self.async_job_id,
            "model_tier": self.model_tier,
        }
        d.update({k: v for k, v in self.extras.items() if k not in _TASK_CORE_DATA_KEYS})
        return json.dumps(d, ensure_ascii=False)


@dataclass
class Failure:
    id: int
    kind: str
    dismissed: bool
    created_at: str
    # 核心 data 字段
    summary: str = ""
    context: str = ""
    task_id: str = ""
    extras: dict[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def from_row(cls, row: Any) -> "Failure":
        """row = (id, kind, dismissed, created_at, data_json)"""
        rid, kind, dismissed, created_at, data_raw = row
        try:
            data: dict[str, Any] = json.loads(data_raw or "{}")
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        summary = data.pop("summary", "")
        context = data.pop("context", "")
        task_id = data.pop("task_id", "")
        return cls(
            id=rid,
            kind=kind,
            dismissed=bool(dismissed),
            created_at=created_at,
            summary=summary,
            context=context,
            task_id=task_id,
            extras=data,
        )


@dataclass
class Run:
    id: int
    task_id: int
    run_type: str
    worker_type: str
    status: str
    created_at: str
    started_at: str = ""
    completed_at: str = ""
    input_json: dict[str, Any] = field(default_factory=dict[str, Any])
    output_json: dict[str, Any] = field(default_factory=dict[str, Any])
    log_text: str = ""
    error_text: str = ""
    tool_name: str = ""
    session_id: str = ""
    model_tier: str = ""
    progress: str = ""
    extras: dict[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def from_row(cls, row: Any) -> "Run":
        rid, task_id, run_type, worker_type, status, created_at, started_at, completed_at, data_raw = row
        try:
            data: dict[str, Any] = json.loads(data_raw or "{}")
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        input_json = data.pop("input_json", {}) or {}
        output_json = data.pop("output_json", {}) or {}
        log_text = data.pop("log_text", "")
        error_text = data.pop("error_text", "")
        tool_name = data.pop("tool_name", "")
        session_id = data.pop("session_id", "")
        model_tier = data.pop("model_tier", "")
        progress = data.pop("progress", "")
        return cls(
            id=rid,
            task_id=task_id,
            run_type=run_type,
            worker_type=worker_type,
            status=status,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            input_json=input_json,
            output_json=output_json,
            log_text=log_text,
            error_text=error_text,
            tool_name=tool_name,
            session_id=session_id,
            model_tier=model_tier,
            progress=progress,
            extras=data,
        )

    def to_data_json(self) -> str:
        data = {
            "input_json": self.input_json,
            "output_json": self.output_json,
            "log_text": self.log_text,
            "error_text": self.error_text,
            "tool_name": self.tool_name,
            "session_id": self.session_id,
            "model_tier": self.model_tier,
            "progress": self.progress,
        }
        data.update(self.extras)
        return json.dumps(data, ensure_ascii=False)


@dataclass
class MetaReflection:
    id: str
    target_kind: str
    trigger: str
    loop_level: str
    diagnosis: str
    proposal: str
    verification_plan: str
    decision: str
    created_at: str
    task_id: int = 0
    run_id: int = 0
    tool_name: str = ""
    extras: dict[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def from_row(cls, row: Any) -> "MetaReflection":
        rid, target_kind, trigger, loop_level, diagnosis, proposal, verification_plan, decision, created_at, data_raw = row
        try:
            data: dict[str, Any] = json.loads(data_raw or "{}")
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        task_id = int(data.pop("task_id", 0) or 0)
        run_id = int(data.pop("run_id", 0) or 0)
        tool_name = str(data.pop("tool_name", "") or "")
        return cls(
            id=str(rid),
            target_kind=target_kind,
            trigger=trigger,
            loop_level=loop_level,
            diagnosis=diagnosis,
            proposal=proposal,
            verification_plan=verification_plan,
            decision=decision,
            created_at=created_at,
            task_id=task_id,
            run_id=run_id,
            tool_name=tool_name,
            extras=data,
        )

    def to_data_json(self) -> str:
        data = {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "tool_name": self.tool_name,
        }
        data.update(self.extras)
        return json.dumps(data, ensure_ascii=False)
