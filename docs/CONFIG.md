# 配置参考

所有配置在 `~/.lingzhou/lingzhou.json` 中。LLM 可通过 `config.get` / `config.set` 工具运行
时读写，修改后自动热重载。

运行时目录是硬规则：`db_path`、`memory_dir`、`state_dir`、`workspace_dir` 以及日志、临时产物都必须位于 `~/.lingzhou/` 下；源码仓目录只承载代码、样例配置和文档，不承载 runtime data。

## 模型

```jsonc
{
  "model": "deepseek/deepseek-v4-flash",  // 主模型
  "routing": {
    "reader": "deepseek/deepseek-v4-flash",   // 信息浏览
    "reasoner": "deepseek/deepseek-v4-pro",    // 深度推理
    "repair": "deepseek/deepseek-v4-flash"     // 错误修复
  }
}
```

Provider 在 providers 段定义，API key 通过环境变量读取（不写入配置文件）。

## 循环参数

| 键 | 默认 | 说明 |
|----|------|------|
| `loop.max_idle_gap` | 60 | 无任务时最大空闲等待(秒)（LLM 可通过 next_idle_gap_secs 自主选择 5-300） |
| `loop.active_idle_gap` | 15 | 有任务时等待间隔(秒) |
| `loop.min_act_gap` | 2 | 连续 act 间最小间隔(秒) |
| `loop.chat_reply_timeout` | 300 | 聊天回复超时(秒) |
| `loop.max_tool_rounds` | 8 | 单 tick 内最多工具调用轮数 |
| `loop.max_consecutive_errors` | 5 | 连续错误阈值 |
| `loop.evolve_every` | 30 | 自进化检查频率(tick 数) |
| `loop.default_idle_gap` | — | LLM 未指定 next_idle_gap_secs 时的兜底值（同 max_idle_gap） |

## 记忆

| 键 | 默认 | 说明 |
|----|------|------|
| `memory.working_capacity` | 40 | 工作记忆容量 |
| `memory.max_events` | 200 | 情节记忆最大事件数 |
| `memory.semantic_decay_lambda` | 0.001 | 语义记忆衰减率 |
| `memory.embedding_weight` | 0.4 | 向量搜索权重 |

## 进化

| 键 | 默认 | 说明 |
|----|------|------|
| `evolution.enabled` | true | 是否启用自进化 |
| `evolution.trigger_min_failures` | 3 | 时间窗内触发进化的最小失败 |
| `evolution.trigger_window_minutes` | 60 | 进化触发时间窗口 |
| `evolution.error_streak_evolve` | 5 | 错误连击立即触发 |
| `evolution.max_attempts` | 3 | 单次进化最大重试 |
| `evolution.backup` | true | 进化前备份原文件 |

## 网关

| 键 | 默认 | 说明 |
|----|------|------|
| `gateway.default_channel` | "local" | 默认消息渠道 (local/wechat/webhook) |

## 环境变量 (`.env`)

```bash
DASHSCOPE_API_KEY=sk-...    # 百炼/通义 API
DEEPSEEK_API_KEY=sk-...     # DeepSeek API
COPILOT_GITHUB_TOKEN=gho_... # GitHub Copilot
```
