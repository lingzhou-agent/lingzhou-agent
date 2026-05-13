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

    def apply_execution_gate(
        self,
        action: "JudgmentOutput",
        cognitive_signals: Any | None,
    ) -> "JudgmentOutput":
        """执行前的确定性尺底门控：重复循环时强制 wait，减少对 prompt 遵守的单点依赖。

        repeat_action_count 语义：从 1 开始计（第 1 次出现）。
          count=3 = 上一轮 tick 已连续 3 次该工具，WMItem 已注入预警。
          此处 >= 3 意味 LLM 看到预警后仍重复（第 4 次及以后），强制 wait 兼底。
        """
        from core.judgment import JudgmentOutput

        if not cognitive_signals:
            return action

        tool_id = action.chosen_action_id or ""
        repeat_action_count = int(getattr(cognitive_signals, "repeat_action_count", 0) or 0)
        repeat_action_tool = str(getattr(cognitive_signals, "repeat_action_tool", "") or "")
        repeat_read_count = int(getattr(cognitive_signals, "repeat_read_count", 0) or 0)
        repeat_read_path = str(getattr(cognitive_signals, "repeat_read_path", "") or "")

        if (
            repeat_action_count >= 3
            and repeat_action_tool in {"task.advance", "task.update"}
            and tool_id in {"task.advance", "task.update"}
        ):
            return JudgmentOutput.wait(
                reason=(
                    f"[反循环门控] 已连续 {repeat_action_count} 次调用 {repeat_action_tool}，"
                    "本轮强制进入 wait 状态。\n"
                    "必须改变策略：先通过 task.list 确认任务当前真实状态，"
                    "再决定下一步行动；不要直接再次调用 task.advance/update。"
                )
            )

        # 补强硬门控：连续重复的 shell.run / file.list 直接止损，避免高成本空转。
        if (
            repeat_action_count >= 3
            and repeat_action_tool in {"shell.run", "file.list"}
            and tool_id == repeat_action_tool
        ):
            return JudgmentOutput.wait(
                reason=(
                    f"[反循环门控] 已连续 {repeat_action_count} 次调用 {repeat_action_tool}，"
                    "本轮强制进入 wait 状态。\n"
                    "必须改变策略：你已经收集了足够信息，停止继续探索，"
                    "转为根据已有信息做出决策或总结结论。"
                )
            )

        if repeat_read_count >= 3 and tool_id == "file.read":
            return JudgmentOutput.wait(
                reason=(
                    f"[反循环门控] 已连续 {repeat_read_count} 次读取相同文件内容 ({repeat_read_path})，"
                    "本轮强制进入 wait 状态。\n"
                    "必须改变策略：该文件内容你已知晓，切换到其他文件、"
                    "修改该文件、或基于已有内容直接作出决策。"
                )
            )

        return action
