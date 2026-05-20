"""core/behavior_tracker.py — 行为模式追踪与认知信号生成。

检测 agent 是否陷入重复行为，并生成 WM 自我感知条目供 LLM 自主判断：
  1. action streak：连续 3 次完全相同的 (tool, key_param) 对
  2. read streak：连续 3 次 file.read 读取相同文件的相同内容（按内容 MD5 去重）
  3. rationale 指纹：连续相同推理结论超阈值时触发信念固化警告

所有检测结果均以 WMItem 形式注入工作记忆，由 LLM 决定如何响应——
不做硬阻断，不替 LLM 做决定。
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

# 同文件连续窗口探测阈值：同一文件被分窗口读取超过此次数时触发特定警告
_SEQ_WINDOW_WARN_AT = 3
# 窗口连续性判断：相邻读的 start 与上一次 end 的差距在此比例内视为连续
_SEQ_WINDOW_GAP_RATIO = 0.25  # 25% 窗口大小内视作连续

# rationale 指纹：连续相同结论超过此阈值时触发"信念固化"警告
_BELIEF_STALE_THRESHOLD = 4
# rationale 指纹窗口大小（deque maxlen）
_BELIEF_WINDOW = 8
# rationale 前缀截取长度（指纹只取前 N 字符，减少随机微小措辞差异的影响）
_BELIEF_HASH_PREFIX = 120


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
        # 同文件顺序窗口探测追踪
        self._seq_window_path: str | None = None
        self._seq_window_count: int = 0
        self._seq_window_last_end: int = 0
        self._seq_window_warned: bool = False
        self._wait_streak: int = 0          # 连续 wait/pause 决策次数
        self._wait_streak_warned: set[int] = set()   # 已触发通知的阈值
        # rationale 指纹追踪（信念固化检测）
        self._rationale_hashes: deque[str] = deque(maxlen=_BELIEF_WINDOW)
        self._belief_stale_hash: str | None = None
        self._belief_stale_count: int = 0
        self._belief_stale_warned: bool = False        # 上次 act 结果指纹（用于 on_act_result 折回调用）
        self._last_act_result_fp: str = ""
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

    def snapshot(self) -> dict[str, Any]:
        return {
            "action_streak_sig": self._action_streak_sig,
            "action_streak_count": self._action_streak_count,
            "wait_streak": self._wait_streak,
        }

    # ── 追踪方法 ──────────────────────────────────────────────────────────────

    def on_act(
        self,
        tool_id: str,
        key_param: str,
        task_id: str | None,
        params: dict | None = None,
    ) -> list["WMItem"]:
        """追踪 act 行为（file.read 和非 file.read 均需调用）。

        - action streak：非 file.read / file.list 工具连续相同时返回 WMItem
          对 file.edit / file.write 使用内容指纹尺化 key，避免同文件不同内容的导致假阳性。

        返回需注入 WM 的条目列表（通常为空或 1 项）。
        """
        from memory.working import WMItem

        items: list[WMItem] = []

        if tool_id in {"file.read", "file.list"}:
            return items  # file.read / file.list streak 由结果感知处理

        # 对编辑类工具，把内容指纹混入 key，避免同文件不同内容的连续误判为循环
        _effective_key = key_param
        if tool_id in {"file.edit", "file.write"} and params:
            _p = params or {}
            _content_sig = str(_p.get("old_text") or _p.get("content") or "")[:80]
            if _content_sig:
                _fp = hashlib.md5(_content_sig.encode("utf-8", errors="replace")).hexdigest()[:8]
                _effective_key = f"{key_param}#{_fp}"

        # action streak 检测（非 file.read）
        _sig = (tool_id, _effective_key)
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
            _log.info("[self-awareness] 连续 3 次相同行为: %s %s", tool_id, key_param)
            items.append(WMItem(
                kind="self_awareness",
                content=f"[行为信号] 过去 3 次均执行了 ({tool_id}, {key_param or '相同参数'})。",
                priority=0.95,
            ))
        return items

    def on_act_result(self, tool_id: str, result_summary: str) -> None:
        """act 执行后修正 streak：如果本次结果与上次不同，说明有实质进展，将 streak 折回 1。

        未防止属于 file.read / file.list （它们由各自的 on_read / on_list 处理）。
        """
        if tool_id in {"file.read", "file.list"}:
            return
        fp = hashlib.md5((result_summary or "").encode("utf-8", errors="replace")).hexdigest()[:12]
        if fp and fp != self._last_act_result_fp:
            # 结果发生变化 → 实际有进展，将 streak 计数折回 1
            self._action_streak_count = 1
        self._last_act_result_fp = fp

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
            _log.info("[self-awareness] 连续 3 次读取相同内容: %s", path)
            items.append(WMItem(
                kind="self_awareness",
                content=f"[行为信号] 过去 3 次均读取了相同内容 ({path})，MD5 一致。",
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
                _log.info(
                    "[self-awareness] 同文件连续 %d 次窗口探测: %s",
                    self._seq_window_count, path,
                )
                items.append(WMItem(
                    kind="self_awareness",
                    content=f"[行为信号] 已连续 {self._seq_window_count} 次按窗口分段读取 ({path})。",
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
            _log.info("[self-awareness] 连续 3 次列出相同目录结果: %s", path)
            items.append(WMItem(
                kind="self_awareness",
                content=f"[行为信号] 过去 3 次均列出了相同目录结果 ({path})，结果指纹一致。",
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
                    f"[行为汇报] 已连续 {self._wait_streak} 轮 {decision}。"
                    f" 任务存在：{'是' if has_active_task else '否'}。"
                    f" 通知 {_stage + 1}/{len(self._wait_notify_thresholds)}。"
                )
                _log.info("[behavior] wait-streak=%d, thresh=%d, priority=%.3f",
                          self._wait_streak, thresh, _priority)
                items.append(WMItem(kind="self_awareness", content=_msg, priority=_priority))
                break  # 每轮最多触发一条通知
        return items

    def on_judgment(self, rationale: str) -> list["WMItem"]:
        """追踪 LLM rationale 指纹，检测"信念固化"（连续相同结论）。

        将 rationale 前 _BELIEF_HASH_PREFIX 字符规范化后计算 MD5 指纹。
        若同一指纹连续出现 >= _BELIEF_STALE_THRESHOLD 次，注入 WM 警告。
        警告仅触发一次（_belief_stale_warned），直到结论真正改变后重置。

        返回需注入 WM 的条目列表（通常为空或 1 项）。
        """
        from memory.working import WMItem

        items: list[WMItem] = []
        if not rationale or not rationale.strip():
            return items

        # 规范化：去首尾空白、折叠空白、取前 N 字符、转小写
        normalized = " ".join(rationale.strip().split())[:_BELIEF_HASH_PREFIX].lower()
        fp = hashlib.md5(normalized.encode("utf-8", errors="replace")).hexdigest()[:12]

        self._rationale_hashes.append(fp)
        if self._belief_stale_hash == fp:
            self._belief_stale_count += 1
        else:
            # 结论改变 → 重置警告状态
            self._belief_stale_hash = fp
            self._belief_stale_count = 1
            self._belief_stale_warned = False

        if (
            self._belief_stale_count >= _BELIEF_STALE_THRESHOLD
            and not self._belief_stale_warned
        ):
            self._belief_stale_warned = True
            _log.info(
                "[self-awareness] rationale 指纹连续 %d 次相同 (fp=%s)，可能存在信念固化",
                self._belief_stale_count, fp,
            )
            items.append(WMItem(
                kind="self_awareness",
                content=f"[认知信号] 推理结论已连续 {self._belief_stale_count} 轮基本相同（指纹 {fp}）。",
                priority=0.96,
            ))
        return items

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
                    content=f"[感知] file.edit 已连续 {self._edit_fail_count} 次因 oldText 不匹配而失败。",
                    priority=0.80,
                )]
        else:
            self._edit_fail_count = 0
        return []
