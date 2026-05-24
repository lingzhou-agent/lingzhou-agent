# 贡献指南

普通安装与运行入口见 [README.md](README.md)。本页只保留源码检出、开发环境与提交流程。

## 开发环境

```bash
git clone https://github.com/suuugeee/lingzhou-agent.git
cd lingzhou-agent
./setup-lingzhou.sh
```

手动路径（等价于上面的脚本）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.12
UV_PROJECT_ENVIRONMENT="$(pwd)/.venv" uv pip install -e ".[test]"
ln -sf "$(pwd)/.venv/bin/lingzhou" ~/.local/bin/lingzhou
```

## 代码风格

- Python 3.12+，类型注解
- 4 空格缩进
- `from __future__ import annotations`

## 提交规范

```
feat: xxx      # 新功能
fix: xxx       # 修复
docs: xxx      # 文档
chore: xxx     # 杂项
refactor: xxx  # 重构
```

## 工具开发

1. 在 `tools/my_tool.py` 中定义工具
2. 使用 `@tool(ToolManifest(...))` 装饰器
3. 自动发现，无需注册

## 插件开发

见 [docs/PLUGIN.md](docs/PLUGIN.md)

## 测试

```bash
python -m pytest tests/
```

## 认知循环

灵舟的核心循环：`感知 → 判断 → 执行 → 反思`。
所有机制遵循"LLM 感知优先"原则：信号以叙事注入 WM，不机械阻塞。

## 设计原则

1. **LLM 感知优先** — 所有信号以叙事注入 WM，不机械命令
2. **可配不硬编码** — 阈值、窗宽通过 lingzhou.json 调整
3. **自驱非指令** — 好奇心以"内心感知"呈现，LLM 自主选择
