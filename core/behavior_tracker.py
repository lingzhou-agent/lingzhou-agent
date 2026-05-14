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

    def __init__(self, wait_streak_notify: list[int] | None = None) -> None:
        # wait-streak 通知阈值（升序，来自配置；None → 使用默认 [3, 6]）
        self._wait_notify_thresholds: list[int] = sorted(wait_streak_notify) if wait_streak_notify else [3, 6]
        self._recent_actions: deque[tuple[str, str]] = deque(maxlen=3)
        self._recent_read_fps: deque[tuple[str, int, str]] = deque(maxlen=3)
        self._recent_list_fps: deque[tuple[str, str]] = deque(maxlen=3)
        self._action_streak_sig: tuple[str, str] | None = None
        self._action_streak_count: int = 0
        self._read_streak_fp: tuple[str, int, str] | None = None
        self._read_streak_count: int = 0
        self._list_streak_fp: tuple[str, str] | None = None
        self._list_streak_count: int = 0
        self._loop_probe_version: int = 0
        self._explore_count: int = 0
        self._explore_task_id: str | None = None
        self._explore_warned: set[int] = set()
        self._wait_streak: int = 0          # 连续 wait/pause 决策次数
        self._wait_streak_warned: set[int] = set()   # 已触发通知的阈值

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
        _list_path = (self._list_streak_fp or ("", ""))[0]
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
            "list_streak": {
                "path": _list_path,
                "count": self._list_streak_count,
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
        - action streak：非 file.read / file.list 工具连续相同时返回 WMItem
        （file.read 的内容去重由 on_read 处理，file.list 的结果去重由 on_list 处理）

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

        if tool_id in {"file.read", "file.list"}:
            return items  # file.read / file.list streak 由结果感知处理

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
                    f"[行为信号] 过去 3 次均执行了 ({tool_id}, {key_param or '相同参数'})。"
                    " 这可能是重复，也可能是必要的重试——请你自行判断："
                    " (1) 这 3 次的目的相同吗？(2) 是否已获得足够信息？"
                    " (3) 如果是循环，原因是什么？你可以继续执行，也可以改变策略。"
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
                    f"[行为信号] 过去 3 次均读取了相同内容 ({path})，MD5 一致。"
                    " 请你判断：(1) 这是必要的确认还是无效重复？"
                    " (2) 是否已从该文件获得了所需信息？"
                    " 你可以继续读取，也可以切换到下一步。"
                ),
                priority=0.95,
            ))
        return items

    def on_list(self, path: str, result_summary: str) -> list["WMItem"]:
        """追踪 file.list 相同目录结果的重复枚举。

        只在“同一路径 + 同样列表结果”连续出现时报警；
        同一路径但目录内容发生变化，不视为无效重复。
        """
        from memory.working import WMItem

        digest = hashlib.md5(result_summary.encode("utf-8", errors="replace")).hexdigest()[:12]
        fp = (path, digest)

        self._recent_list_fps.append(fp)
        if self._list_streak_fp == fp:
            self._list_streak_count += 1
        else:
            self._list_streak_fp = fp
            self._list_streak_count = 1
        self._loop_probe_version += 1

        items: list[WMItem] = []
        if len(self._recent_list_fps) == 3 and len(set(self._recent_list_fps)) == 1:
            _log.warning("[self-awareness] 连续 3 次列出相同目录结果: %s", path)
            items.append(WMItem(
                kind="self_awareness",
                content=(
                    f"[行为信号] 过去 3 次均列出了相同目录结果 ({path})，结果指纹一致。"
                    " 这次不是仅路径相同，而是返回内容也没有变化。"
                    " 请判断：这是必要确认，还是应切换到读取/写入/总结等下一步？"
                ),
                priority=0.95,
            ))
        return items

    def on_wait(self, decision: str, has_active_task: bool) -> list["WMItem"]:
        """追踪连续 wait/pause 决策，向 WM 注入状态汇报。

        原则：只汇报事实，由 LLM 自主决定是否继续等待。不做硬阻断。
        阈值来自配置（wait_streak_notify），每个阈值首次触达时发出一条通知。
        act 后重置计数。
        """
        from memory.working import WMItem

        if decision not in ("wait", "pause"):
            self._wait_streak = 0
            self._wait_streak_warned = set()
            return []

        self._wait_streak += 1
        items: list[WMItem] = []

        # 找到本轮应触发的最高阈值（每个阈值只触发一次）
        for thresh in self._wait_notify_thresholds:
            if self._wait_streak >= thresh and thresh not in self._wait_streak_warned:
                self._wait_streak_warned.add(thresh)
                # 阶段数：当前是第几个阈值（影响 priority，越晚越高）
                _stage = self._wait_notify_thresholds.index(thresh)
                _max_stage = max(len(self._wait_notify_thresholds) - 1, 1)
                _priority = round(0.85 + 0.1 * (_stage / _max_stage), 3)
                _msg = (
                    f"[行为汇报] 当前已连续 {self._wait_streak} 轮决策为 {decision}。"
                    f" 任务存在：{'是' if has_active_task else '否'}。"
                    f" 这是第 {_stage + 1}/{len(self._wait_notify_thresholds)} 级通知。"
                    " 以下信息供参考，由你决定下一步："
                    " (1) 当前等待条件是否仍然成立？"
                    " (2) next_step 描述是否仍然准确？"
                    " (3) 是否有可以立即执行的小动作推进任务？"
                    " (4) 是否需要向用户说明当前状态？"
                    " 你可以继续 wait，也可以行动——这只是一条状态通知。"
                )
                _log.info("[behavior] wait-streak=%d, thresh=%d, priority=%.3f",
                          self._wait_streak, thresh, _priority)
                items.append(WMItem(kind="self_awareness", content=_msg, priority=_priority))
                break  # 每轮最多触发一条通知
        return items

    def apply_execution_gate(
        self,
        action: "JudgmentOutput",
        cognitive_signals: Any | None,
    ) -> "JudgmentOutput":
        """透传：不做任何硬拦截，行为决策权完全归 LLM。

        重复行为信号已在 on_act / on_read 中以 WMItem 形式注入工作记忆，
        LLM 在下一轮 judgment 时自主看到并决定是否改变策略。
        """
        return action
