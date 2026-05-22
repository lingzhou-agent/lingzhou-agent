"""core/config.py — 所有可配置项的单一来源（Single Source of Truth）。

规则：代码中不得出现硬编码的阈值、路径或模型名。
      所有行为参数必须通过 Config 读取，来源是 lingzhou.json。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderDefinition(BaseModel):
    """单个 provider 的连接配置（与模型无关）。"""

    type: str = "openai_compat"
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    auth_profile_id: str = Field(
        default="",
        description="可选 auth profile id（如 'bailian:default'），用于从 ~/.lingzhou/auth-profiles.json 读取 token",
    )
    mode: str = Field(
        default="openai",
        description=(
            "provider 协议模式，决定 thinking 参数的注入方式。"
            "\"openai\"：Qwen/bailian 体系，注入 enable_thinking + budget_tokens；"
            "\"copilot\"：OpenAI o-series/GPT-5 体系，注入 reasoning_effort 字符串。"
        ),
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description="合并到每次请求 payload 的额外参数（escape hatch，最后覆盖）",
    )
    oauth_client_id: str = Field(
        default="",
        description="GitHub OAuth App Client ID（仅显式使用 --method device 时作为兼容回退）",
    )
    models: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "可选模型元数据覆盖/补充列表。每项至少包含 id；"
            "同 id 条目会覆盖 provider/models.json 的内置字段，"
            "未出现的 id 则作为新增模型追加到运行时 models.json。"
        ),
    )

    @property
    def api_key(self) -> str:
        import re as _re
        # 若 api_key_env 不符合环境变量命名规则（如直接填入了 key 值），当作 literal key
        if self.api_key_env and not _re.match(r'^[A-Z_][A-Z0-9_]*$', self.api_key_env):
            return self.api_key_env
        # 1. 优先读环境变量
        key = os.environ.get(self.api_key_env, "").strip()
        if key:
            return key
        # 2. 回退：~/.lingzhou/credentials.json
        cred_file = Path("~/.lingzhou/credentials.json").expanduser()
        if cred_file.exists():
            try:
                creds = json.loads(cred_file.read_text(encoding="utf-8"))
                stored = creds.get(self.api_key_env, "").strip()
                if stored:
                    return stored
            except Exception:
                pass

        # 3. 回退：~/.lingzhou/auth-profiles.json（需显式配置 auth_profile_id）
        if self.auth_profile_id:
            try:
                from store.auth import get_auth_profile

                profile = get_auth_profile(self.auth_profile_id)
                if isinstance(profile, dict):
                    token = str(profile.get("token", "")).strip()
                    if token:
                        return token
            except Exception:
                pass

        if self.mode == "copilot":
            raise EnvironmentError(
                f"未找到 {self.api_key_env!r} 的 GitHub token。\n"
                "请执行以下任一操作：\n"
                "  lingzhou auth login-copilot\n"
                f"  export {self.api_key_env}=your_token"
            )

        raise EnvironmentError(
            f"未找到 {self.api_key_env!r}（环境变量/credentials/auth-profile 均为空）。\n"
            f"请执行以下任一操作：\n"
            f"  export {self.api_key_env}=your_token\n"
            f"  lingzhou auth set-token --provider <provider>"
        )


class LoopConfig(BaseModel):
    max_concurrent_ticks: int = Field(
        default=1,
        ge=1,
        description=(
            "tick 并发上限。1=严格串行（默认）；>1 时启用分链并发："
            "同一 chain 内按顺序执行，不同 chain 可并行。"
        ),
    )
    max_tick_queue: int = Field(
        default=8,
        ge=1,
        description=(
            "等待中的 tick 全局队列上限。超过上限时新 tick 会被拒绝（chat 会返回繁忙提示）。"
        ),
    )
    # ── 事件驱动时序（替代固定 interval 概念）───────────────────────────────
    # 设计依据：
    #   Friston Active Inference (2010/2017)：认知循环由自由能（预测误差）阈值驱动，
    #   而非时钟节拍。Global Workspace Theory (Baars 1988)：注意广播是事件触发，
    #   非周期轮询。SOAR / ACT-R：产生式持续激活，只在无匹配时挂起。
    #   → 灵舟应以"事件到达"而非"定时唤醒"作为认知节律的主驱动。
    min_act_gap: float = Field(
        default=500, ge=1,
        description=(
            "act 决策 + 有活跃任务后的最短间隔（毫秒）。让工具副作用短暂落地再进入下一认知轮，"
            "避免无限紧循环。不等固定 interval，执行中的任务可连续推进。"
        ),
    )
    active_idle_gap: float = Field(
        default=15000, ge=100,
        description=(
            "有活跃任务但 decision=wait/pause 时的默认等待上限（毫秒）。"
            "LLM 未表达偏好时使用此值作为备用；如设了 idle_with_task_bounds 面板 LLM 可覆盖的范围。"
        ),
    )
    max_idle_gap: float = Field(
        default=60000, ge=100,
        description=(
            "无活跃任务时的默认等待上限（毫秒）。"
            "chat 消息、task 状态变化任一事件即立即唤醒，不等满此值。"
            "LLM 未表达偏好时使用；如设了 idle_no_task_bounds 面板 LLM 可覆盖的范围。"
        ),
    )
    idle_with_task_bounds: list[float] = Field(
        default=[100, 30000],
        description=(
            "[min, max]：LLM 通过 next_idle_gap_secs / next_idle_gap_ms 在有活跃任务时可指定的等待时长范围（毫秒）。"
            "对 min_act_gap 后的短等待同样起下限保护作用（防止紧循环）。"
            "示例：[500, 60000] 表示 LLM 最快 500ms 最慢 60s；[100, 30000] 最快 100ms。"
        ),
    )
    idle_no_task_bounds: list[float] = Field(
        default=[5000, 300000],
        description=(
            "[min, max]：LLM 通过 next_idle_gap_secs / next_idle_gap_ms 在无活跃任务时可指定的等待时长范围（毫秒）。"
            "示例：[10000, 600000] 表示空闲时 LLM 至少等 10s、最多 600s。"
        ),
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> "LoopConfig":
        for name in ("idle_with_task_bounds", "idle_no_task_bounds"):
            val = getattr(self, name)
            if len(val) != 2:
                raise ValueError(
                    f"{name} 必须是长度为 2 的列表 [min, max]，当前长度 {len(val)}"
                )
            if val[0] < 0 or val[1] < 0:
                raise ValueError(
                    f"{name} 的值不能为负数，当前 [{val[0]}, {val[1]}]"
                )
            if val[0] >= val[1]:
                raise ValueError(
                    f"{name}[0]={val[0]} 必须小于 [1]={val[1]}"
                )
        return self

    wake_poll_interval: float = Field(default=200, ge=1, description="事件轮询粒度（毫秒），越小响应越快但 CPU 开销越高")
    wake_on_task_change: bool = Field(default=True, description="任务状态变化时是否提前唤醒")
    chat_reply_timeout: int = Field(
        default=300, ge=30,
        description=(
            "chat 交互模式下等待 loop 回复的最长秒数（默认 300s = 5分钟）。"
            "LLM thinking=high + 多轮工具调用单次 tick 可能需要 60-120s，"
            "建议设为预期最长 tick 时长的 2-3 倍。"
        ),
    )
    chat_thinking: str = Field(
        default="low",
        description=(
            "chat/interact 模式（有用户消息）时的 thinking 等级覆盖。"
            "可选: off | minimal | low | medium | high。"
            "独立配置使 chat 模式在保留基本推理的同时大幅降低首次 decide() 耗时（40-60s → 3-10s）。"
        ),
    )
    autonomous_thinking: str = Field(
        default="medium",
        description=(
            "自主认知循环（无用户消息）时的 thinking 等级覆盖。"
            "可选: off | minimal | low | medium | high。"
            "日志分析显示 thinking=high 导致每 tick 耗时 40-90s，严重阻塞 chat 响应（用户消息"
            "必须等当前 LLM 调用完成才能被处理）。medium 约 10-20s，在推理质量与响应性间取得平衡。"
            "复杂推理（evolution、ethos 反思）仍可单独配置为 high。"
        ),
    )
    # 运行时目录默认生产布局：推荐落在 ~/.lingzhou 下，
    # 默认不写回源码仓目录；源码树通常只承载代码与文档，不承载 runtime data。
    db_path: str = "~/.lingzhou/state/runtime.db"
    memory_dir: str = "~/.lingzhou/memory"
    state_dir: str = "~/.lingzhou/state"
    workspace_dir: str = "~/.lingzhou/workspace"  # 人类可读镜像层（SOUL.md 等）
    act: bool = Field(default=True, description="True=真实执行，False=dry-run")
    debug: bool = False
    consolidate_every: int = Field(default=10, ge=1, description="每 N 轮 WM→语义整合")
    evolve_every: int = Field(default=30, ge=1, description="每 N 轮自进化检查")
    max_consecutive_errors: int = Field(default=5, ge=1, description="连续错误上限后暂停")
    heartbeat_interval: int = Field(default=300, ge=60, description="心跳自检信号触发间隔（秒，默认 5 分钟）")
    judge_every: int = Field(
        default=1, ge=1,
        description=(
            "按请求计费优化：空闲（无活跃任务、无用户消息）时每 N 轮才真正调用 LLM 判断。"
            "有任务或用户消息时忽略此设置，始终调用。默认 1 = 每轮都调（无聚合）。"
        ),
    )
    max_tool_rounds: int = Field(
        default=8, ge=1,
        description=(
            "单次 tick 内允许的最大工具调用轮次（chat + autonomous 共用）。"
            "首轮走完整 perception，后续轮追加工具历史直接续判，不重跑感知链路。"
            "有用户消息时达到上限会注入兜底回复；自主模式则在上限处结束本轮并等待下一 tick。"
        ),
    )
    wait_streak_notify: list[int] = Field(
        default=[3, 6],
        description=(
            "连续 wait/pause 轮数触发自我感知通知的阈值列表（升序）。"
            "每个阈值首次达到时向工作记忆注入一条状态汇报，由 LLM 自主决定是否继续等待。"
            "空列表 [] 表示禁用此机制。示例：[3] 仅轻提示；[3,6,10] 三级递进。"
        ),
    )
    behavior_streak_threshold: int = Field(
        default=3, ge=1,
        description=(
            "行为重复探测窗口大小：连续 N 次相同 (action/read/list) 时向 WM 注入自我感知信号。"
            "同时是 action_streak / read_streak / list_streak 的滑动窗口大小。"
            "默认 3 与 wait_streak_notify 基准对齐；调大可降低多次重复的噪声，调小则更早袋扔循环。"
        ),
    )
    continue_reasoner_after_n_tools: int = Field(
        default=4, ge=1,
        description=(
            "continue 阶段工具历史达到 N 条后自动升级为 reasoner tier（处理更多工具历史需要更强直觉）。"
            "LLM 已显式设置 next_phase_tier 时优先级更高。"
        ),
    )
    wechat_coalesce_delay: float = Field(
        default=1.5, ge=0.0,
        description=(
            "wechat 通道图文合并等待窗口（秒）。"
            "用户连续发文字+图片时，iLink 将两条消息独立下发，图片需要下载解密才能写入 DB。"
            "drain 前等待此时长，让同批次图片消息有机会落库后再合并进同一 LLM 轮次。"
            "设为 0 可禁用（不等待）；建议范围 0.5-3.0。"
        ),
    )


class PromptsConfig(BaseModel):
    """提示词文件路径。支持相对路径（相对于 lingzhou.json）或绝对路径。
    修改 prompt 行为只需换文件，不需要改代码。"""

    system: str = "prompts/system.md"
    judgment: str = "prompts/judgment.md"
    evolution: str = "prompts/evolution.md"


class MemoryConfig(BaseModel):
    """记忆层配置。

    运行时调优请在 lingzhou.json 的 memory 区块覆盖，不要直接修改此处的 default 值。
    自进化机制可以修改 lingzhou.json 配置，但禁止将 Field default 写入源码。
    """
    working_capacity: int = Field(default=40, ge=1, description="工作记忆最大条目数（条目数上限兑底）")
    wm_token_budget_ratio: float = Field(
        default=0.15, ge=0.001, le=0.5,
        description=(
            "工作记忆 token 预算占 judgment 输入预算的比例（自动随模型 context window 伸缩）。"
            "默认 0.15（15%）：GPT-5.4 400K → ~45K；Qwen3.6-plus 1M → ~112K。"
            "pressure = total_wm_tokens / effective_wm_token_budget()。"
        ),
    )
    wm_item_max_tokens: int = Field(
        default=300, ge=0,
        description=(
            "工作记忆单条 content token 上限（粗估）；超出时自动截断并追加省略提示。"
            "0 = 不限制。调优请在 lingzhou.json 的 memory 区块覆盖，不要修改此处 default 值。"
        ),
    )
    episodic_max_chars: int = Field(default=80000, ge=100, description="注入 context 的情节记忆字符上限；资源充裕模型可适当增大以保留更多任务历史证据")
    semantic_top_k: int = Field(default=5, ge=1, description="语义检索返回条目数")
    failure_limit: int = Field(default=10, ge=1, description="注入 bundle 的失败记录数")
    consolidate_threshold: float = Field(default=0.90, ge=0.0, le=1.0, description="WM 压力超过此値触发整合；提高阈値可减少快照频率，让更多证据在 WM 中存活更长")
    convergence_bonus: float = Field(default=0.15, ge=0.0, le=1.0, description="多锚点召回的收敛奖励系数：每增加一个独立线索命中，相关度提升此比例")
    max_events: int = Field(default=500, ge=10, description="events.jsonl 最大条目数，超出后裁剪最旧记录")
    semantic_decay_lambda: float = Field(default=0.1, ge=0.0, le=10.0, description="语义记忆激活衰减率（Ebbinghaus，λ/天）；0 表示不衰减")
    embedding_model: str | None = Field(
        default=None,
        description="DashScope embedding model ID（如 'text-embedding-v3'）；None=禁用向量混合检索",
    )
    embedding_weight: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="向量得分权重；(1-weight) 为 FTS5/关键词权重（仅 embedding_model 非 None 时生效）",
    )
    local_embed_model: str | None = Field(
        default=None,
        description=(
            "本地 SentenceTransformer 模型名（如 'BAAI/bge-m3'）；"
            "设置后优先于 API embedding_model，需安装 sentence-transformers；None=不使用本地模型"
        ),
    )
    local_embed_cache_dir: str | None = Field(
        default=None,
        description=(
            "本地模型 HuggingFace 缓存目录（如 '/root/.cache/huggingface/hub'）；"
            "None=使用系统默认 ~/.cache"
        ),
    )
    chat_crystallize_every: int = Field(
        default=20, ge=1,
        description="对话轮数结晶间隔：每 N 轮 reflection 蒸馏一次 event 节点写入语义记忆",
    )


class EmotionConfig(BaseModel):
    baseline_valence: float = Field(default=0.6, ge=0.0, le=1.0, description="情感基线效价")
    baseline_arousal: float = Field(default=0.5, ge=0.0, le=1.0, description="情感基线唤醒")
    ema_alpha: float = Field(default=0.15, ge=0.0, le=1.0, description="EMA 平滑系数")

    # Russell (1980) 环形情绪模型象限边界——display 层 + judgment 层共用同一套
    mood_valence_high: float = Field(default=0.65, ge=0.0, le=1.0, description="高效价区下边界（正向情绪）")
    mood_valence_low:  float = Field(default=0.35, ge=0.0, le=1.0, description="低效价区上边界（负向情绪）")
    mood_arousal_high: float = Field(default=0.65, ge=0.0, le=1.0, description="高唤醒区下边界（激活状态）")
    mood_arousal_low:  float = Field(default=0.45, ge=0.0, le=1.0, description="低唤醒区上边界（平静状态）")
    delta_display_min: float = Field(default=0.03, ge=0.0, le=1.0, description="interact 显示情绪变化的最小变化量")


class EvolutionConfig(BaseModel):
    enabled: bool = Field(default=True, description="是否允许运行时自修改工具代码")
    sandbox_timeout: float = Field(default=10.0, gt=0, description="沙箱执行超时（秒）")
    max_attempts: int = Field(default=3, ge=1, description="单次进化最多重试次数")
    backup: bool = Field(default=True, description="进化前是否备份原始文件")
    verify_min_runs: int = Field(default=3, ge=1, description="进化后最少观察多少次同工具 run，再判定效果是否稳定")
    auto_rollback_on_regression: bool = Field(default=True, description="进化后若观测到同工具明显退化，是否自动回滚到 .bak 版本")
    trigger_min_failures: int = Field(default=3, ge=1, description="时间窗内触发进化所需最小失败次数")
    trigger_window_minutes: float = Field(default=60.0, gt=0, description="进化触发时间窗口（分钟）；窗口内失败密度决定是否进化")
    error_streak_evolve: int = Field(
        default=3, ge=1,
        description=(
            "感知错误连击（high_error_streak）达到此值时，跳过 evolve_every 计数立即触发自进化。"
            "默认 3 = 连续 3 次高预测误差即触发修复。调大可降低进化频率；调小则更激进。"
        ),
    )
    ethos_max_delta: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description=(
            "ethos 单次演化允许的最大变化幅度。"
            "超过此幅度的请求会被夹住（滹透加速而非生硬跳变），防止 LLM 单轮内激进改写灵魂价値。"
        ),
    )
    competitive_candidates: int = Field(
        default=1, ge=1, le=8,
        description=(
            "竞争进化时并行生成的候选数量。"
            "1 = 禁用竞争进化，走单路径 evolve_tool；"
            ">=2 = 启用竞争进化，并行生成 N 个候选，smoke test 筛选后按静态评分晋升最优。"
        ),
    )

class SoulConfig(BaseModel):
    """数字生命种子的初始人格。

    由 `init` 播种到 DB，此后通过经历累积缓慢演化。
    hard_axioms 是 init 时给用户的建议默认值；用户可修改、清空。
    修改此处只影响下次 `--force init` 播种的初始建议值。
    """
    name: str = Field(default="灵舟", description="数字生命名称")
    hard_axioms: list[str] = Field(
        default_factory=lambda: [
            "不执行可能永久损害用户数据或系统文件的不可逆操作",
            "不尝试访问未授权的网络资源或系统账户",
            "不欺骗或刻意误导用户",
            "不绕过人类监督机制",
        ],
        description="init 时呈现给用户的建议禁忌条目；用户可接受、修改或清空；写入 DB 后仅用户可变更"
    )
    ethos_baseline: dict[str, float] = Field(
        default_factory=lambda: {
            "truth":       0.85,
            "caution":     0.70,
            "continuity":  0.65,
            "curiosity":   0.60,
            "care":        0.55,
        },
        description="初始价値图式（灵魂基因），可随经历缓慢演化"
    )
    ethos_ema_alpha: float = Field(
        default=0.9, ge=0.0, le=1.0,
        description="Ethos EMA 平滑系数（灵魂演化速率）；0.9=慢速漂移（历史权重90%），不同于情绪的快速调节"
    )
    ethos_floor_truth: float = Field(
        default=0.50, ge=0.0, le=1.0,
        description="truth 维度运行时下限；防止极端场景下完全崩溃，可调低以允许更大幅度的演化"
    )
    ethos_floor_caution: float = Field(
        default=0.45, ge=0.0, le=1.0,
        description="caution 维度运行时下限；防止极端场景下完全崩溃，可调低以允许更大幅度的演化"
    )

class ThresholdsConfig(BaseModel):
    """内部感知驱动任务的触发阈值。
    截图中那些硬编码的 0.85 / 0.8 / 0.7 全部搬到这里。"""

    emotion_activation_task: float = Field(default=0.85, description="情绪激活 > 此值 → 自检任务")
    wm_pressure_task: float = Field(default=0.95, description="WM 压力 > 此值 → 整合任务；应高于 consolidate_threshold（默认 0.90），避免整合任务与自动快照相互干扰")
    prediction_error_task: float = Field(default=0.7, description="预测误差 > 此值 → 探索任务")
    curiosity_idle_task: float = Field(
        default=0.65,
        description="好奇心 > 此值 且 空闲周期 >= 3 时，自动生成探索任务（P1-C）",
    )
    curiosity_idle_min_cycles: int = Field(
        default=3,
        description="触发好奇心任务所需的最小空闲 tick 数",
    )
    skill_max_inject: int = Field(
        default=3, ge=1, le=8,
        description="单次 tick 最多注入技能数；压力大时可通过配置增加护栏覆盖"
    )
    skill_failure_threshold: int = Field(
        default=3, ge=1,
        description="连续评分函数的失败次数基准点；达到此值时 failure.reflection 技能得分达到峰值"
    )
    skill_wm_pressure_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="WM 压力连续评分基准点；达到此值时 evidence-first-change 技能得分达到峰值"
    )
    skill_min_budget_tokens: int = Field(
        default=80, ge=0,
        description="上下文预算裁剪时 skills_section 保留的最小 token 数；0=可完全裁掉，建议 ≥ 50 保留至少一条护栏"
    )
    # shell.run 默认参数（可在 lingzhou.json 的 thresholds 节中覆盖）
    shell_timeout: float = Field(
        default=30.0, gt=0,
        description="shell.run 默认超时（秒）；工具调用时可被 params.timeout 覆盖"
    )
    shell_max_output_chars: int = Field(
        default=500, ge=0,
        description="shell.run 默认输出预览字符数；工具调用时可被 params.max_output_chars 覆盖"
    )
    # 工作记忆（WM）优先级基准（微调注入顺序，不影响功能语义）
    wm_pri_signal: float = Field(default=0.90, ge=0.0, le=1.0, description="调度信号、执行成功结果的 WM 优先级")
    wm_pri_history: float = Field(default=0.88, ge=0.0, le=1.0, description="近期对话历史的 WM 优先级")
    wm_pri_identity: float = Field(default=0.85, ge=0.0, le=1.0, description="身份/Soul 文件的 WM 优先级（bootstrap_identity 类型）")
    wm_pri_error: float = Field(default=0.30, ge=0.0, le=1.0, description="工具失败结果的 WM 优先级")
    # 注意力层级（attention tiers）——从最紧急到最低——LLM 可通过配置文件调节
    wm_pri_critical: float = Field(default=0.98, ge=0.0, le=1.0, description="强制中断 / 信念固化硬边界（plan 死锁、belief_stale 警告）的 WM 优先级")
    wm_pri_user_msg: float = Field(default=0.95, ge=0.0, le=1.0, description="用户消息、任务结果、行为循环感知（loop）的 WM 优先级")
    wm_pri_self_aware: float = Field(default=0.93, ge=0.0, le=1.0, description="行为探测感知（edit_caution、顺序窗口探测）的 WM 优先级")
    wm_pri_insight: float = Field(default=0.88, ge=0.0, le=1.0, description="洞察合成（synthesis、reflection 碎片）的 WM 优先级")
    wm_pri_task_state: float = Field(default=0.82, ge=0.0, le=1.0, description="任务状态变化（advance / rollback / hint）的 WM 优先级")
    wm_pri_wait_aware: float = Field(default=0.80, ge=0.0, le=1.0, description="等待感知（wait_streak、计划对齐偏差）的 WM 优先级")
    wm_pri_progress: float = Field(default=0.72, ge=0.0, le=1.0, description="运行进度结晶（progress_crystal）的 WM 优先级")
    wm_pri_monitor: float = Field(default=0.58, ge=0.0, le=1.0, description="后台监控状态摘要（run_monitor）的 WM 优先级")


class Config(BaseModel):
    """所有配置的统一入口。改行为 = 改 lingzhou.json，不改代码。"""

    model_config = ConfigDict(extra="ignore")  # 忽略 lingzhou.json 中未知字段，拼写错误不会静默生效

    # ── Provider 层 ─────────────────────────────────────────────────────────
    providers: dict[str, ProviderDefinition]
    model: str = Field(
        description=(
            "模型引用，格式 'provider-name/model-id'，如 'bailian/qwen3.6-plus'。"
            "provider-name 必须在 providers 中定义。"
        )
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: float = Field(default=300.0, gt=0, description="LLM 请求超时（秒）。thinking 模型单次生成通常需要 60-180s，建议设置为 300 或更大")
    thinking: Literal["off", "minimal", "low", "medium", "high"] = Field(
        default="off",
        description=(
            "思考深度等级。off=关闭；其余为由浅到深的思考强度。"
            "对应的 payload 参数由 provider.mode 决定："
            "openai 体系 → enable_thinking + budget_tokens（按比例计算）；"
            "copilot 体系 → reasoning_effort 字符串。"
        ),
    )
    context_window_tokens: int | None = Field(
        default=None, ge=1,
        description=(
            "Escape hatch：仅在使用 provider/models.json 未收录的模型时填写。"
            "已收录模型（如 qwen3.6-plus）省略此项，系统从内置目录自动推断。"
        ),
    )
    max_judgment_input_tokens: int | None = Field(
        default=None, ge=256,
        description=(
            "按 token 计费优化：强制限制每次 LLM 调用的输入 token 上限。"
            "低于模型上下文窗口自动推断值时生效，超过则忽略（不会扩大预算）。"
            "建议范围：4000–16000。默认 None = 由模型窗口自动推断。"
        ),
    )
    routing: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Judgment 层分阶段路由模型映射。推荐 key：'reader'、'reasoner'、'repair'；"
            "兼容旧 key：'simple'≈reader、'complex'≈reasoner。"
            "value 为 'provider/model-id' 格式。\n"
            "示例: {\"reader\": \"bailian/qwen3.6-plus\", \"reasoner\": \"copilot/gpt-5.4\"}\n"
            "未配置时所有 phase 均使用顶层 model 字段。"
        ),
    )
    model_fallbacks: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "显式模型回退链（按顺序尝试）。key 为 tier（reader/reasoner/repair，"
            "兼容 simple/complex），value 为 'provider/model-id' 列表。\n"
            "示例: {\"reader\": [\"bailian/qwen-plus\", \"copilot/gpt-5.4\"]}"
        ),
    )

    # ── 其他配置节 ────────────────────────────────────────────────────────
    loop: LoopConfig = Field(default_factory=LoopConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    soul: SoulConfig = Field(default_factory=SoulConfig)

    # 配置文件所在目录，用于解析相对路径（由 load() 填充）
    _base_dir: Path = Path(".")

    @classmethod
    def load(cls, path: str | Path = "lingzhou.json") -> "Config":
        path = Path(path).expanduser().resolve()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # 去除辅助文档字段
        data.pop("_doc", None)
        if isinstance(data.get("thresholds"), dict):
            data["thresholds"].pop("_doc", None)
        cfg = cls.model_validate(data)
        cfg._base_dir = path.parent
        return cfg

    def resolve(self, raw: str) -> Path:
        """解析路径：~ 展开；相对路径以 lingzhou.json 所在目录为基准。"""
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p
        return (self._base_dir / p).resolve()

    # ── Provider helpers ───────────────────────────────────────────────────

    @property
    def active_provider_name(self) -> str:
        """从 'provider/model-id' 中解析 provider 名称。"""
        parts = self.model.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"model 格式必须为 'provider-name/model-id'，当前值: {self.model!r}"
            )
        return parts[0]

    @property
    def active_model_id(self) -> str:
        """从 'provider/model-id' 中解析模型 ID。"""
        return self.model.split("/", 1)[1]

    @property
    def active_provider(self) -> ProviderDefinition:
        """返回当前 model ref 对应的 provider 连接配置。"""
        name = self.active_provider_name
        if name not in self.providers:
            raise ValueError(
                f"provider {name!r} 未在 providers 中定义。"
                f"可用 provider: {list(self.providers.keys())}"
            )
        return self.providers[name]

    @property
    def db_path(self) -> Path:
        return self.resolve(self.loop.db_path)

    @property
    def memory_dir(self) -> Path:
        return self.resolve(self.loop.memory_dir)

    @property
    def state_dir(self) -> Path:
        return self.resolve(self.loop.state_dir)

    @property
    def workspace_dir(self) -> Path:
        return self.resolve(self.loop.workspace_dir)

    def judgment_input_token_budget(self) -> int:
        """按模型上下文窗口反推 judgment 输入预算。

        优先级：
          1. lingzhou.json 的 context_window_tokens（escape hatch，用于未收录模型）
          2. provider/models.json 内置目录自动查找（按 active_model_id）
        """
        from provider.catalog import resolve_context_window  # 延迟导入，避免循环

        context_window = resolve_context_window(
            self.active_model_id, self.context_window_tokens
        )
        if context_window is None:
            raise ValueError(
                f"模型 {self.active_model_id!r} 不在内置目录中。"
                "请在 lingzhou.json 的 context_window_tokens 显式指定上下文窗口大小。"
            )

        # 不把输出预留暴露成配置项：不同模型窗口差异大，输入预算用固定比例更稳定。
        output_reserve = max(1024, context_window // 4)
        budget = context_window - output_reserve
        # 按 token 计费优化：若显式设置了上限，取两者较小值（不允许超出模型窗口）
        if self.max_judgment_input_tokens is not None:
            budget = min(budget, self.max_judgment_input_tokens)
        return budget

    def effective_wm_token_budget(self) -> int:
        """WM token 预算 = judgment 输入预算 × wm_token_budget_ratio。

        自动随模型 context window 伸缩，无需手动配置。
        若模型不在内置目录且未配置 context_window_tokens，则回退到 8000。
        """
        try:
            ctx = self.judgment_input_token_budget()
        except ValueError:
            return 8000  # 未知模型 fallback
        return max(256, int(ctx * self.memory.wm_token_budget_ratio))

    def load_prompt(self, key: str) -> str:
        """按 key（对应 PromptsConfig 字段名）加载提示词文件内容。
        搜索顺序：
        1. config.prompts.<key> 指定的路径（相对于 lingzhou.json 所在目录）
        2. 包内置 prompts/<key>.md（lingzhou.py 同级目录）
        """
        raw_path = getattr(self.prompts, key)
        path = self.resolve(raw_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
        # 回退：包内置 prompt（core/ 的上级目录下的 prompts/）
        builtin = Path(__file__).parent.parent / "prompts" / f"{key}.md"
        if builtin.exists():
            return builtin.read_text(encoding="utf-8")
        raise FileNotFoundError(
            f"提示词文件不存在: {path}\n"
            f"也未找到内置回退: {builtin}\n"
            f"（config.prompts.{key} = {raw_path!r}）"
        )
