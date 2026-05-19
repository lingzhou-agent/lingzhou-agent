# lingzhou — Self-Evolving Cognitive Agent

[English](README.md) | [中文](README.zh.md)

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

lingzhou is a self-programming, self-evolving cognitive agent designed to run autonomously on a Linux host, interact through WeChat or local chat, and improve its own behavior over time.

## What It Is

lingzhou is not a chat wrapper. It is an event-driven runtime with:

- a perception → self-drive → judgment → execution → reflection loop
- persistent memory across working, episodic, semantic, and task storage
- multi-model routing for reader / reasoner / repair roles
- hot-reload evolution for tools, prompts, and runtime behavior
- 56 built-in tools for files, shell, tasks, memory, web, browser, probes, and media

## Quick Start

```bash
git clone https://github.com/suuugeee/lingzhou-agent.git
cd lingzhou-agent
pip install -e .

mkdir -p ~/.lingzhou
cp lingzhou.json.example ~/.lingzhou/lingzhou.json
# edit ~/.lingzhou/lingzhou.json
# create ~/.lingzhou/.env with provider keys

lingzhou gateway start -d
lingzhou gateway start --channel local
```

Runtime data is always stored under `~/.lingzhou/`, including `state/`, `memory/`, `workspace/`, logs, and temporary artifacts. The repository itself only stores source code, sample config, and documentation.

### System Service

```bash
sudo cp scripts/lingzhou.service /etc/systemd/system/
sudo systemctl enable --now lingzhou
```

## Architecture

```text
Perception  ->  Self-Drive
     |              |
     v              v
Judgment   <-  Model Routing
     |
     v
Execution  ->  Built-in Tools
     |
     v
Reflection ->  Evolution
```

## Documentation

Most in-depth design documents are currently written in Chinese.

- [Architecture](docs/ARCHITECTURE.md)
- [Self-Drive](docs/SELF_DRIVE.md)
- [Tool Catalog](docs/TOOLS.md)
- [Configuration Reference](docs/CONFIG.md)
- [Probe Guide](docs/PROBE.md)
- [Plugin Guide](docs/PLUGIN.md)
- [Deviation Review](docs/DEVIATION_REVIEW.md)

## Built-In Tool Surface

lingzhou currently ships 56 built-in tool endpoints across these groups:

- file and config: `file.*` + `config.*`
- shell and process control: `shell.run`, `shell.capabilities`, `exec`, `process.*`
- tasks, planning, and scheduling: `task.*`, `task.ask`, `task.plan`, `schedule.*`
- memory and reflection: `memory.*`, `reflect.structural`, `failure.dismiss`
- web, browser, and media: `web.*`, `browser.*`, `image.*`, `tts.speak`
- skills, probes, and notifications: `skill.*`, `probe.*`, `wechat.send`

See [docs/TOOLS.md](docs/TOOLS.md) for the grouped catalog and capability tags.

## Configuration

```jsonc
// ~/.lingzhou/lingzhou.json
{
  "model": "bailian/qwen3.6-plus",
  "routing": {
    "reader": "bailian/qwen-plus",
    "reasoner": "copilot/gpt-5.4"
  },
  "loop": { "act": true, "max_idle_gap": 60 },
  "gateway": { "default_channel": "wechat" }
}
```

The runtime can inspect and adjust config through `config.get` and `config.set`. Full details are in [docs/CONFIG.md](docs/CONFIG.md).

## Repository Layout

```text
lingzhou-agent/
├── channels/   # external channels such as wechat
├── cli/        # chat, gateway, auth, logs, bootstrap
├── core/       # cognition loop, judgment, execution, evolution
├── docs/       # design and operator docs
├── memory/     # memory system facade
├── plugins/    # plugin workspace
├── provider/   # model providers
├── store/      # persistence helpers
├── tests/      # smoke and behavior tests
└── tools/      # built-in tool implementations
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
