"""core/behavior_tracker.py — 行为模式追踪与反循环门控。

检测 agent 是否陷入重复行为，并生成 WM 自我感知条目：
  1. action streak：连续 3 次完全相同的 (tool, key_param) 对
  2. read streak：连续 3 次 file.read 读取相同文件的相同内容（按内容 MD5 去重）
  3. explore budget：同一任务下文件探索（file.list / file.read）次数超阈值

原位置：CognitionLoop.__init__ 里的 10 个状态变量 + _tick 里约 100 行追踪逻辑
       + _apply_execution_gate 方法。
"""
from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.judgment import JudgmentOutput
    from memory.working import WMItem

_log = logging.getLogger("lingzhou.behavior_tracker")

_EXPLORE_TOOLS = frozenset(("file.list", "file.read"))
_EXPLORE_THRESHOLDS = (8, 12, 16)


class BehaviorTracker:
    """行为模式追踪器：检测循环 + 确定性兜底门控。"""

    def __init__(self) -> None:
        self._recent_actions: deque[tuple[str, str]] = deque(maxlen=3)
        self._recent_read_fps: deque[tuple[str, int, str]] = deque(maxlen=3)
        self._action_streak_sig: tuple[str, str] | None = None
        self._action_streak_count: int = 0
        self._read_streak_fp: tuple[str, int, str] | None = None
        self._read_streak_count: int = 0
        self._loop_probe_version: int = 0
        self._explore_count: int = 0
        self._explore_task_id: str | None = None
        self._explore_warned: set[int] = set()
        self._wait_streak: int = 0          # 连续 wait/pause 决策次数
        self._wait_streak_warned: set[int] = set()  # 已触发提示的阈值

    # ── 状态接口 ──────────────────────────────────────────────────────────────

    def apply_cognitive_probe(self, signals: Any) -> None:
        """将当前循环探针状态写入 cognitive_signals 对象（原地修改）。"""
        signals.repeat_action_count = self._action_streak_count
        signals.repeat_action_tool = (self._action_streak_sig or ("", ""))[0]
        signals.repeat_action_key = (self._action_streak_sig or ("", ""))[1]
        signals.repeat_read_count = self._read_streak_count
        signals.repeat_read_path = (self._read_streak_fp or ("", 0, ""))[0]
        signals.loop_probe_version = self._loop_probe_version

    def snapshot(self) -> dict[str, Any]:
        """返回追踪器状态快照，供 state_snapshot 使用。"""
        _streak_tool, _streak_key = self._action_streak_sig or ("", "")
        _read_path = (self._read_streak_fp or ("", 0, ""))[0]
        return {
            "action_streak": {
                "tool": _streak_tool,
                "key": _streak_key,
                "count": self._action_streak_count,
            },
            "read_streak": {
                "path": _read_path,
                "count": self._read_streak_count,
            },
            "loop_probe_version": self._loop_probe_version,
        }

    # ── 追踪方法 ──────────────────────────────────────────────────────────────

    def on_act(
        self,
        tool_id: str,
        key_param: str,
        task_id: str | None,
    ) -> list["WMItem"]:
        """追踪 act 行为（file.read 和非 file.read 均需调用）。

        - 探索预算感知：file.list / file.read 累计次数超阈值时返回 WMItem
        - action streak：非 file.read 工具连续相同时返回 WMItem
        （file.read 的内容去重由 on_read 处理）

        返回需注入 WM 的条目列表（通常为空或 1 项）。
        """
        from memory.working import WMItem

        items: list[WMItem] = []

        # 探索预算感知（file.list 和 file.read 均计入）
        if task_id != self._explore_task_id:
            self._explore_task_id = task_id
            self._explore_count = 0
            self._explore_warned = set()
        if tool_id in _EXPLORE_TOOLS:
            self._explore_count += 1
        for thresh in _EXPLORE_THRESHOLDS:
            if self._explore_count >= thresh and thresh not in self._explore_warned:
                self._explore_warned.add(thresh)
                _log.warning("[explore-awareness] 任务 %s 已探索 %d 次", task_id, self._explore_count)
                items.append(WMItem(
                    kind="self_awareness",
                    content=(
                        f"[自我感知] 当前任务已执行 {self._explore_count} 次文件探索，"
                        "请评估是否已有足够信息推进或完成任务。继续探索的边际收益正在递减。"
                    ),
                    priority=0.92,
                ))
                break  # 每次 tick 最多触发一个梯度

        if tool_id == "file.read":
            return items  # file.read streak 由 on_read 处理

        # action streak 检测（非 file.read）
        _sig = (tool_id, key_param)
        self._recent_actions.append(_sig)
        if self._action_streak_sig == _sig:
            self._action_streak_count += 1
        else:
            self._action_streak_sig = _sig
            self._action_streak_count = 1
        self._loop_probe_version += 1

        if (
            len(self._recent_actions) == 3
            and len(set(self._recent_actions)) == 1
            and tool_id
        ):
            _log.warning("[self-awareness] 连续 3 次相同行为: %s %s", tool_id, key_param)
            items.append(WMItem(
                kind="self_awareness",
                content=(
                    f"[自我感知] 我已连续 3 次执行 ({tool_id}, {key_param or '相同参数'})，"
                    "这是行为死循环的信号。必须在 reflection 中诊断原因，并立刻改变策略。"
                ),
                priority=0.95,
            ))
        return items

    def on_read(self, path: str, max_chars: int, result_summary: str) -> list["WMItem"]:
        """追踪 file.read 相同内容重复读取。

        按内容 MD5 去重（同路径但内容不同时不触发）。
        返回需注入 WM 的条目列表（通常为空或 1 项）。
        """
        from memory.working import WMItem

        _body = result_summary.split("\n", 1)[1] if "\n" in result_summary else result_summary
        _digest = hashlib.md5(_body.encode("utf-8", errors="replace")).hexdigest()[:12]
        _fp = (path, max_chars, _digest)

        self._recent_read_fps.append(_fp)
        if self._read_streak_fp == _fp:
            self._read_streak_count += 1
        else:
            self._read_streak_fp = _fp
            self._read_streak_count = 1
        self._loop_probe_version += 1

        items: list[WMItem] = []
        if len(self._recent_read_fps) == 3 and len(set(self._recent_read_fps)) == 1:
            _log.warning("[self-awareness] 连续 3 次读取相同内容: %s", path)
            items.append(WMItem(
                kind="self_awareness",
                content=(
                    f"[自我感知] 我已连续 3 次读取相同内容 ({path})，"
                    "这不是必要的代码审计增量。下一步必须切换文件或转为总结/修改。"
                ),
                priority=0.95,
            ))
        return items

    def on_wait(self, decision: str, has_active_task: bool) -> list["WMItem"]:
        """追踪连续 wait/pause 决策，注入自我感知提示。

        阈值（3, 6）：分别对应"轻提示"和"强提示"，不阻断，由 LLM 自主决策。
        act 决策后重置计数。
        """
        from memory.working import WMItem

        if decision not in ("wait", "pause"):
            self._wait_streak = 0
            self._wait_streak_warned = set()
            return []

        self._wait_streak += 1
        items: list[WMItem] = []

        _thresholds = (3, 6)
        for thresh in _thresholds:
            if self._wait_streak >= thresh and thresh not in self._wait_streak_warned:
                self._wait_streak_warned.add(thresh)
                if thresh == 3:
                    _msg = (
                        f"[自我感知] 我已连续 {self._wait_streak} 轮选择 wait/pause。"
                        "请重新评估：当前等待的'信号'是否有明确的到达条件？"
                        "若任务 next_step 不再准确，请用 task.advance 更新它，而非持续等待。"
                    )
                    _priority = 0.88
                else:
                    _msg = (
                        f"[自我感知] 我已连续 {self._wait_streak} 轮 wait/pause，"
                        "这可能是认知停滞的信号。"
                        "必须采取行动：或用 task.advance 更新 next_step，"
                        "或用 task.complete/task.fail 结束任务，或直接执行下一步工具调用。"
                    )
                    _priority = 0.95
                _log.warning("[self-awareness] 连续 %d 轮 wait/pause", self._wait_streak)
                items.append(WMItem(kind="self_awareness", content=_msg, priority=_priority))
                break
        return items

    def apply_execution_gate(
        self,
        action: "JudgmentOutput",
        cognitive_signals: Any | None,
    ) -> "JudgmentOutput":
        """执行前门控（当前无硬拦截）。

        所有重复行为预警均通过 WMItem 注入工作记忆，由 LLM 根据执行结果自主判断是否改变策略。
        保留此方法作为未来扩展点（如需针对特定系统级死锁场景补充兜底）。
        """
        return action
