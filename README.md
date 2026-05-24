# 灵舟 (lingzhou) — 自编程自进化认知 Agent

[中文](README.md) | [English](README.en.md)

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

灵舟是一个运行在 Linux 主机上的自编程、自进化认知 Agent。它不是聊天壳，而是一套事件驱动的认知运行时：能够通过微信或本地终端与用户交互，持续积累记忆、自主探索，并在失败中改进自身行为。

## 它是什么

灵舟的核心是一个持续循环的认知系统：

- 感知 → 自驱力 → 判断 → 执行 → 反思
- 工作记忆、情节记忆、语义记忆、任务存储四层持久化
- reader / reasoner / repair 多模型分层路由
- 工具、提示词和运行时行为支持热更新与演进
- 当前内置工具端点，覆盖文件、Shell、任务、记忆、网页、浏览器、探针和媒体

## 快速开始

推荐安装路径（一条命令）：

```bash
curl -fsSL https://raw.githubusercontent.com/suuugeee/lingzhou-agent/main/scripts/install.sh | bash
lingzhou
```

如果你偏好 `pipx`：

```bash
pipx install --python python3.12 git+https://github.com/suuugeee/lingzhou-agent.git
lingzhou
```

如果你需要源码检出、本地开发或提交流程，见 [CONTRIBUTING.md](CONTRIBUTING.md)。

首次运行 `lingzhou` 会自动进入 `onboard`，完成 provider 配置、初始化数据库与 workspace，并把运行时目录自动准备到 `~/.lingzhou/`。

如果要接入微信等外部渠道：

```bash
lingzhou gateway setup --channel wechat
lingzhou gateway start --channel wechat -d
```

运行时数据默认写入 `~/.lingzhou/`，包括 `state/`、`memory/`、`workspace/`、日志与临时产物。生产环境建议保持这一布局；源码仓默认只承载代码、样例配置和文档，不承载 runtime data。

### 系统服务

```bash
sudo cp scripts/lingzhou.service /etc/systemd/system/
sudo systemctl enable --now lingzhou
```

## 架构总览

```text
感知层  ->  自驱力
      |          |
      v          v
判断层  <-  模型路由
      |
      v
执行层  ->  内置工具
      |
      v
反思层  ->  进化引擎
```

## 文档目录

每个文档页顶部都带有中英文切换入口。

- [架构设计与当前差距](docs/ARCHITECTURE.md)
- [自驱力与自主探索](docs/SELF_DRIVE.md)
- [工具目录](docs/TOOLS.md)
- [配置参考](docs/CONFIG.md)
- [探针说明](docs/PROBE.md)
- [插件开发指南](docs/PLUGIN.md)

## 内置工具面

当前内置工具端点，主要分为：

- 文件与配置：`file.*`、`config.*`
- Shell 与进程：`shell.run`、`shell.capabilities`、`exec`、`process.*`
- 任务、计划与调度：`task.*`、`task.ask`、`task.plan`、`schedule.*`
- 记忆与反思：`memory.*`、`reflect.structural`、`failure.dismiss`
- 网页、浏览器与媒体：`web.*`、`browser.*`、`image.*`、`tts.speak`
- 技能、探针与通知：`skill.*`、`probe.*`、`wechat.send`

完整清单见 [docs/TOOLS.md](docs/TOOLS.md)。

## 配置

```jsonc
// ~/.lingzhou/lingzhou.json
{
     "model": "bailian/qwen3.6-plus",
     "routing": {
          "reader": "bailian/qwen-plus",
          "reasoner": "copilot/gpt-5.4"
     },
      "loop": { "act": true, "max_idle_gap": 60000 },
     "gateway": { "default_channel": "wechat" }
}
```

运行时可通过 `config.get` 和 `config.set` 工具读取或调整配置。详见 [docs/CONFIG.md](docs/CONFIG.md)。

## 仓库结构

```text
lingzhou-agent/
├── channels/   # 外部通道，如 wechat
├── cli/        # chat、gateway、auth、logs、bootstrap
├── core/       # 认知循环、判断、执行、进化
├── docs/       # 设计与运维文档
├── memory/     # 记忆系统 facade
├── plugins/    # 插件工作区
├── provider/   # 模型 provider
├── store/      # 持久化 helper
├── tests/      # 冒烟与行为测试
└── tools/      # 内置工具实现
```

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

MIT
