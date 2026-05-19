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
_EXPLORE_THRESHOLDS = (5, 9, 13, 15)  # 递增梯度：5=提醒 9=警告 13=强警告 15=严重
_EXPLORE_WARN_LEVELS = {
    5: ("提醒", 0.85, "请评估是否已有足够信息推进或完成任务。继续探索的边际收益正在递减。"),
    9: ("警告", 0.90, "已重复探索9次，强烈建议换策略：不要继续读文件，直接基于已有信息产出结论或完成任务。"),
    13: ("严重", 0.93, "已严重超预算(13次)！几乎不可能通过继续探索获得更多价值。必须立即停止读取，切换为写入/产出模式。"),
    15: ("紧急", 0.96, "探索已彻底失控(15次+)！这很可能是陷入死循环。请诚实评估：当前方向是否还有意义？如果无意义，立即 task.complete 并记录失败教训。"),
}
# 同文件连续窗口探测阈值：同一文件被分窗口读取超过此次数时触发特定警告
_SEQ_WINDOW_WARN_AT = 3
# 窗口连续性判断：相邻读的 start 与上一次 end 的差距在此比例内视为连续
_SEQ_WINDOW_GAP_RATIO = 0.25  # 25% 窗口大小内视作连续
class BehaviorTracker:
    """行为模式追踪器：检测循环并把信号交给 LLM。"""

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
        # 同文件顺序窗口探测追踪
        self._seq_window_path: str | None = None
        self._seq_window_count: int = 0
        self._seq_window_last_end: int = 0
        self._seq_window_warned: bool = False
        self._wait_streak: int = 0          # 连续 wait/pause 决策次数
        self._wait_streak_warned: set[int] = set()   # 已触发通知的阈值

    @property
    def wait_streak(self) -> int:
        """公开接口：连续 wait/pause 决策次数。"""
        return self._wait_streak

    # ── 状态接口 ──────────────────────────────────────────────────────────────

    def apply_cognitive_probe(self, signals: Any) -> None:
        """将当前循环探针状态写入 cognitive_signals 对象（原地修改）。"""
        signals.repeat_action_count = self._action_streak_count
        signals.repeat_action_tool = (self._action_streak_sig or ("", ""))[0]
        signals.repeat_action_key = (self._action_streak_sig or ("", ""))[1]
        signals.repeat_read_count = self._read_streak_count
        signals.repeat_read_path = (self._read_streak_fp or ("", 0, ""))[0]
        signals.repeat_list_count = self._list_streak_count
        signals.repeat_list_path = (self._list_streak_fp or ("", ""))[0]
        signals.loop_probe_version = self._loop_probe_version
        # 探索预算感知
        signals.explore_count = self._explore_count
        signals.explore_threshold = _EXPLORE_THRESHOLDS[-1] if _EXPLORE_THRESHOLDS else 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "action_streak_sig": self._action_streak_sig,
            "action_streak_count": self._action_streak_count,
            "wait_streak": self._wait_streak,
            "explore_task_id": self._explore_task_id,
            "explore_count": self._explore_count,
        }

    @property
    def is_explore_stuck(self) -> bool:
        """探索卡住：超过硬上限需要强制终止当前任务。"""
        return (
            self._explore_task_id is not None
            and self._explore_count >= _EXPLORE_HARD_LIMIT
        )

    @property
    def stuck_task_id(self) -> str | None:
        """返回卡住的任务 ID（当 is_explore_stuck 时）。"""
        if self.is_explore_stuck:
            return self._explore_task_id
        return None
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
        if not task_id:
            self._explore_task_id = None
            self._explore_count = 0
            self._explore_warned = set()
        else:
            if task_id != self._explore_task_id:
                self._explore_task_id = task_id
                self._explore_count = 0
                self._explore_warned = set()
            if tool_id in _EXPLORE_TOOLS:
                self._explore_count += 1
            for thresh in _EXPLORE_THRESHOLDS:
                if self._explore_count >= thresh and thresh not in self._explore_warned:
                    self._explore_warned.add(thresh)
                    level_name, priority, msg = _EXPLORE_WARN_LEVELS.get(
                        thresh,
                        ("超限", 0.96, f"探索次数已达{thresh}次，严重超预算。请立即切换模式。"),
                    )
                    _log.warning("[explore-awareness] 任务 %s 已探索 %d 次 [%s]", task_id, self._explore_count, level_name)
                    items.append(WMItem(
                        kind="self_awareness",
                        content=f"[{level_name}] 当前任务已执行 {self._explore_count} 次文件探索（{level_name}阈值）。{msg}",
                        priority=priority,
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

    def on_read(
        self,
        path: str,
        max_chars: int,
        result_summary: str,
        *,
        start: int = 0,
        end: int = 0,
    ) -> list["WMItem"]:
        """追踪 file.read 重复读取。

        两层检测：
        1. 内容 MD5 去重 — 同一文件同一窗口读多次
        2. 同文件顺序窗口探测 — 同一文件被分窗口连续读取（不同窗口内容不同但模式重复）

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

        # 层 1：同内容重复
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

        # 层 2：同文件顺序窗口探测（不同窗口但模式为连续扫描）
        if start >= 0 and end > start:
            window_size = end - start
            if path == self._seq_window_path:
                gap = abs(start - self._seq_window_last_end)
                gap_ratio = gap / max(window_size, 1)
                # 窗口连续（与上次 end 接近）→ 递增计数
                if gap_ratio <= _SEQ_WINDOW_GAP_RATIO or gap <= max(window_size, 100):
                    self._seq_window_count += 1
                else:
                    # 跳到了不连续区域 → 重置
                    self._seq_window_count = 1
            else:
                # 换文件了 → 重置
                self._seq_window_path = path
                self._seq_window_count = 1
                self._seq_window_warned = False
            self._seq_window_last_end = end

            if (
                self._seq_window_count >= _SEQ_WINDOW_WARN_AT
                and not self._seq_window_warned
            ):
                self._seq_window_warned = True
                _log.warning(
                    "[self-awareness] 同文件连续 %d 次窗口探测: %s",
                    self._seq_window_count, path,
                )
                items.append(WMItem(
                    kind="self_awareness",
                    content=(
                        f"[行为信号] 已连续 {self._seq_window_count} 次按窗口分段读取 ({path})。"
                        " 这种逐段扫描模式通常是探索而非验证。"
                        " 请判断：(1) 是否已定位到目标代码区域？"
                        " (2) 是否可以直接用 grep/sed 定位关键符号？"
                        " (3) 是否已有足够信息进行下一步修改或验证？"
                        " 建议：切换到 shell.run(grep) 或直接 file.edit 目标位置。"
                    ),
                    priority=0.93,
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
        """透传：不替 LLM 做决定，只把行为探针信号写进上下文与日志。"""
        if action.decision != "act" or cognitive_signals is None:
            return action

        tool_id = action.chosen_action_id or ""
        repeat_action_count = int(getattr(cognitive_signals, "repeat_action_count", 0) or 0)
        repeat_action_tool = str(getattr(cognitive_signals, "repeat_action_tool", "") or "")
        repeat_read_count = int(getattr(cognitive_signals, "repeat_read_count", 0) or 0)
        repeat_read_path = str(getattr(cognitive_signals, "repeat_read_path", "") or "")
        repeat_list_count = int(getattr(cognitive_signals, "repeat_list_count", 0) or 0)
        repeat_list_path = str(getattr(cognitive_signals, "repeat_list_path", "") or "")

        if repeat_action_count >= 3 and repeat_action_tool == tool_id:
            _log.info("[behavior-sense] repeated action delegated to llm: tool=%s count=%d key=%s", tool_id, repeat_action_count, getattr(cognitive_signals, "repeat_action_key", "") or "")
        if tool_id == "file.read" and repeat_read_count >= 3:
            _log.info("[behavior-sense] repeated read delegated to llm: path=%s count=%d", repeat_read_path, repeat_read_count)
        if tool_id == "file.list" and repeat_list_count >= 3:
            _log.info("[behavior-sense] repeated list delegated to llm: path=%s count=%d", repeat_list_path, repeat_list_count)
        return action

    def on_edit_failure(self, error: str) -> list:
        """追踪 file.edit 失败，连续失败时返回感知信号。
        
        LLM 可感知信号，自行决定是否切换策略（如用 file.write 或 shell.run sed 代替 file.edit）。
        """
        from memory.working import WMItem
        if "OldTextNotFound" in (error or ""):
            self._edit_fail_count = getattr(self, '_edit_fail_count', 0) + 1
            if self._edit_fail_count >= 2:
                return [WMItem(
                    kind="behavior_sense",
                    content=(
                        f"[感知] file.edit 已连续 {self._edit_fail_count} 次因 oldText 不匹配而失败。"
                        f"考虑换策略：① 用 shell.run sed/python 做精确插入"
                        f"② 用 file.read 读更大范围（≥500字符）获取完整上下文后重试"
                        f"③ 用 file.write 全量覆写（先 file.read 全文）"
                    ),
                    priority=0.80,
                )]
        else:
            self._edit_fail_count = 0
        return []
