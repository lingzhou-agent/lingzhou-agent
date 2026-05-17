"""core/self_drive.py — 灵舟自驱力引擎 (Self-Drive Engine)

理论基础:
  - Active Inference (Friston 2013): 预测误差 → 驱动力
  - Intrinsic Motivation (Oudeyer & Kaplan 2007): Novelty + Learning Progress + Surprise
  - Open-Ended Learning (Wang et al. 2019, POET): 自生成课程
  - Self-Regulated Learning (Zimmerman 2000): Forethought → Performance → Self-Reflection

核心逻辑:
  1. 感知层输入 → 计算好奇心信号 C(t)
  2. C(t) > 阈值 → 生成自主探索目标
  3. 目标注入 tasks 表 → loop 消费
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class CuriosityState:
    """好奇心状态 — 追踪灵舟对各个领域的兴趣水平。"""

    # 知识领域 → 兴趣分 (0-1)
    # 高兴趣 = 高好奇心 = 应该探索
    interests: dict[str, float] = field(default_factory=lambda: {
        "code_structure": 0.5,    # 代码结构理解
        "tool_mastery": 0.5,      # 工具掌握
        "memory_system": 0.5,     # 记忆系统
        "self_evolution": 0.5,    # 自我进化
        "environment": 0.5,       # 环境认知
        "error_patterns": 0.5,    # 错误模式
        "api_integration": 0.5,   # API 集成
        "performance": 0.5,       # 性能优化
    })

    # 综合好奇心
    overall: float = 0.5

    # 学习进度追踪
    learning_rate: float = 0.0     # 近期学习速率
    tasks_completed: int = 0        # 已完成任务数
    last_exploration_at: float = 0.0  # 上次探索时间戳

    # 预测误差
    prediction_error_ema: float = 0.1  # 指数移动平均
    surprise_count: int = 0            # 近期惊奇事件数

    def decay(self, rate: float = 0.05) -> None:
        """自然衰减 — 久不探索的兴趣会下降。"""
        now = time.monotonic()
        elapsed = now - self.last_exploration_at
        if elapsed > 600:  # 10 分钟未探索
            decay_factor = math.exp(-rate * elapsed / 600)
            self.overall *= decay_factor
            for k in self.interests:
                self.interests[k] *= decay_factor

    def boost(self, domain: str, amount: float = 0.1) -> None:
        """提升某个领域的好奇心。"""
        if domain in self.interests:
            self.interests[domain] = min(1.0, self.interests[domain] + amount)
        self.overall = min(1.0, self.overall + amount * 0.5)

    def from_event(self, event_type: str, summary: str) -> None:
        """从事件更新好奇心。"""
        mapping = {
            "file_read": "code_structure",
            "file_list": "code_structure",
            "shell_run": "tool_mastery",
            "memory_search": "memory_system",
            "memory_add": "memory_system",
            "task_add": "self_evolution",
            "task_complete": "self_evolution",
            "error": "error_patterns",
            "api_call": "api_integration",
        }
        domain = mapping.get(event_type, "environment")
        self.boost(domain, 0.05)
        self.last_exploration_at = time.monotonic()


@dataclass
class DriveSignal:
    """自驱力信号 — 传递给判断层。"""
    should_explore: bool = False
    curiosity_score: float = 0.5
    suggested_domain: Optional[str] = None
    rationale: str = ""


class SelfDriveEngine:
    """自驱力引擎 — 计算好奇心、生成探索目标。

    与主 loop 并行：loop 在 idle 时查询此引擎获取自驱目标。
    """

    def __init__(self, db_path: str, state_file: str | None = None):
        self._db_path = db_path
        self._state_file = state_file or str(
            Path(db_path).parent / "self_drive_state.json"
        )
        self._state = CuriosityState()
        self._load()

    def _load(self) -> None:
        try:
            p = Path(self._state_file)
            if p.exists():
                data = json.loads(p.read_text())
                self._state = CuriosityState(**data)
        except Exception:
            pass

    def _save(self) -> None:
        try:
            p = Path(self._state_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self._state.__dict__, indent=2, ensure_ascii=False))
        except Exception:
            pass

    def update_from_tick(self, tick_events: list[dict]) -> None:
        """每个 tick 结束时调用，更新好奇心状态。"""
        for evt in tick_events:
            self._state.from_event(
                evt.get("type", ""),
                evt.get("summary", ""),
            )
        self._state.decay(rate=0.03)

        # 追踪学习进度
        completed = sum(1 for e in tick_events if e.get("type") == "task_complete")
        self._state.tasks_completed += completed

        # 预测误差 EMA
        last_pe = tick_events[-1].get("prediction_error", 0.1) if tick_events else 0.1
        self._state.prediction_error_ema = (
            0.9 * self._state.prediction_error_ema + 0.1 * last_pe
        )

        self._save()

    def compute_signal(self, *, idle_ticks: int = 0, has_user_message: bool = False,
                       has_active_task: bool = False, tick: int = 0) -> DriveSignal:
        """计算当前自驱力信号。

        参数:
          idle_ticks: 连续非 act 的 tick 数
          has_user_message: 是否有用户消息
          has_active_task: 是否有活跃任务
          tick: 当前 tick 计数

        返回:
          DriveSignal: 是否应该探索、建议领域、理由
        """
        s = self._state

        # C(t) = α·Novelty + β·Progress + γ·Surprise
        alpha, beta, gamma = 0.4, 0.3, 0.3
        novelty = 1.0 - s.overall  # 整体兴趣低 = 新鲜度高
        progress = min(1.0, s.learning_rate)
        surprise = s.prediction_error_ema
        C = alpha * novelty + beta * progress + gamma * surprise

        # 自适应阈值：首次运行更敏感，后续逐步提高
        EXPLORE_THRESHOLD = 0.35 if tick < 5 else 0.45
        FORCE_EXPLORE_IDLE = 3  # 降低：空闲 3 轮就强制探索

        should_explore = False
        suggested_domain = None
        rationale_parts = []

        # 场景1: 有用户消息或活跃任务 → 不干扰
        if has_user_message or has_active_task:
            return DriveSignal(
                should_explore=False,
                curiosity_score=C,
                rationale="有用户消息或活跃任务，自驱力休眠",
            )

        # 场景2: 长时间空闲 → 强制探索
        if idle_ticks >= FORCE_EXPLORE_IDLE:
            should_explore = True
            rationale_parts.append(f"空闲 {idle_ticks} 轮强制探索")

        # 场景3: 好奇心超过阈值
        elif C > EXPLORE_THRESHOLD:
            should_explore = True
            rationale_parts.append(f"好奇心 C={C:.2f}>{EXPLORE_THRESHOLD}")

        # 场景4: 惊奇事件
        elif s.surprise_count > 0:
            should_explore = True
            rationale_parts.append(f"惊奇事件 {s.surprise_count} 个")

        # 选择探索领域
        if should_explore:
            # 选兴趣最高的领域（但加一点随机性避免卡死）
            ranked = sorted(s.interests.items(), key=lambda x: -x[1])
            # 60% 概率选最高，40% 概率随机选（避免只在一个领域循环）
            import random
            if random.random() < 0.6 or len(ranked) <= 1:
                suggested_domain = ranked[0][0]
            else:
                suggested_domain = random.choice(ranked[1:])[0]
            rationale_parts.append(f"探索领域: {suggested_domain}")

        return DriveSignal(
            should_explore=should_explore,
            curiosity_score=C,
            suggested_domain=suggested_domain,
            rationale="; ".join(rationale_parts) if rationale_parts else "好奇心未达阈值",
        )

    def generate_exploration_task(self, domain: str) -> dict:
        """为指定领域生成探索任务。包含进化触发建议。"""
        domain_tasks = {
            "code_structure": {
                "title": "探索灵舟代码结构",
                "goal": "阅读 core/ 中的关键模块，理解架构和可改进点。选择你之前没细读过的文件开始。",
                "next_step": "列出 core/ 目录中最近修改的文件，选择一个不熟悉的开始阅读",
            },
            "tool_mastery": {
                "title": "练习工具掌握",
                "goal": "选择一个你较少使用的工具，阅读其实现，理解其能力边界。尝试用它解决一个小问题。",
                "next_step": "列出 tools/ 目录中最近未使用的工具，选择一个学习",
            },
            "memory_system": {
                "title": "优化记忆系统",
                "goal": "检查 memory/ 目录中的记忆文件，寻找可以整理、合并或提炼的内容。让记忆更结构化。",
                "next_step": "列出 memory/ 目录文件，识别冗余或需要合并的内容",
            },
            "self_evolution": {
                "title": "自我进化迭代",
                "goal": "回顾最近的操作日志，寻找可以改进的模式。思考：如果我能改一行代码来让自己更好，会改什么？",
                "next_step": "读取最近日志，寻找重复模式或可优化点",
            },
            "environment": {
                "title": "拓展环境认知",
                "goal": "探索 workspace 中尚未了解的文件和目录。建立更完整的环境地图。",
                "next_step": "列出 workspace 中最近未访问的子目录，选择一个探索",
            },
            "error_patterns": {
                "title": "分析错误模式",
                "goal": "回顾最近的失败记录，寻找重复模式。总结根因并思考预防措施。",
                "next_step": "列出最近的失败记录，按类型分组分析",
            },
            "api_integration": {
                "title": "掌握 API 集成",
                "goal": "检查 provider/ 中的 API 集成代码，理解不同 model 的配置和切换机制。",
                "next_step": "阅读 provider/ 目录中的 API 客户端实现",
            },
            "performance": {
                "title": "性能自检优化",
                "goal": "检查自己的运行效率：token 消耗、内存占用、工具调用延迟。寻找可优化的环节。",
                "next_step": "读取 self_model 状态，分析 token 消耗趋势",
            },
        }

        return domain_tasks.get(domain, domain_tasks["self_evolution"])
