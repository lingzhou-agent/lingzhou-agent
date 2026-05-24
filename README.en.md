# lingzhou — Self-Evolving Cognitive Agent

[中文](README.md) | [English](README.en.md)

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

lingzhou is a self-programming, self-evolving cognitive agent designed to run autonomously on a Linux host, interact through WeChat or local chat, and improve its own behavior over time.

## What It Is

lingzhou is not a chat wrapper. It is an event-driven runtime with:

- a perception → self-drive → judgment → execution → reflection loop
- persistent memory across working, episodic, semantic, and task storage
- multi-model routing for reader / reasoner / repair roles
- hot-reload evolution for tools, prompts, and runtime behavior
- built-in tools for files, shell, tasks, memory, web, browser, probes, and media

## Quick Start

Recommended install path (one command):

```bash
curl -fsSL https://raw.githubusercontent.com/suuugeee/lingzhou-agent/main/scripts/install.sh | bash
lingzhou
```

If you prefer `pipx`:

```bash
pipx install --python python3.12 git+https://github.com/suuugeee/lingzhou-agent.git
lingzhou
```

For source checkout, local development, or contribution workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).

The first `lingzhou` run automatically enters `onboard`, walks through provider setup, seeds the runtime database, and prepares the workspace under `~/.lingzhou/`.

To connect an external channel such as WeChat:

```bash
lingzhou gateway setup --channel wechat
lingzhou gateway start --channel wechat -d
```

Runtime data is stored under `~/.lingzhou/` by default, including `state/`, `memory/`, `workspace/`, logs, and temporary artifacts. Production setups should keep this layout; the repository itself is intended to store source code, sample config, and documentation.

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

Each document page now includes a language switch at the top.

- [Architecture and Current Gaps](docs/ARCHITECTURE.en.md)
- [Self-Drive](docs/SELF_DRIVE.en.md)
- [Tool Catalog](docs/TOOLS.en.md)
- [Configuration Reference](docs/CONFIG.en.md)
- [Probe Guide](docs/PROBE.en.md)
- [Plugin Guide](docs/PLUGIN.en.md)

## Built-In Tool Surface

lingzhou currently ships built-in tool endpoints across these groups:

- file and config: `file.*` + `config.*`
- shell and process control: `shell.run`, `shell.capabilities`, `exec`, `process.*`
- tasks, planning, and scheduling: `task.*`, `task.ask`, `task.plan`, `schedule.*`
- memory and reflection: `memory.*`, `reflect.structural`, `failure.dismiss`
- web, browser, and media: `web.*`, `browser.*`, `image.*`, `tts.speak`
- skills, probes, and notifications: `skill.*`, `probe.*`, `wechat.send`

See [docs/TOOLS.en.md](docs/TOOLS.en.md) for the grouped catalog and capability tags.

## Configuration

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

The runtime can inspect and adjust config through `config.get` and `config.set`. Full details are in [docs/CONFIG.en.md](docs/CONFIG.en.md).

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