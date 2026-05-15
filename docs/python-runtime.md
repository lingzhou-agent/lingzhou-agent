# Python Runtime 优势

> 为什么 lingzhou 选择 Python 而非 Go 作为数字生命的运行时。

---

## 1. 核心问题

数字生命需要**在运行时修改自己**。Go 是编译语言，这在根本上排除了该能力。

---

## 2. Python 的决定性优势

### 2.1 运行时代码生成与热替换

```python
# core/evolution.py — 真正的自我修改
code = await llm.generate(f"编写工具 {name}，用于 {description}")
compile(code, "<generated>", "exec")   # 语法检查
tool_path.write_text(code)             # 写入磁盘
importlib.reload(module)               # 热替换，零重启
```

Go 无法在不重启进程的情况下加载新编译的代码。  
Python 可以在毫秒内替换一个工具模块的实现。

### 2.2 `@tool` 装饰器 + 全局注册表

```python
from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool

@tool(ToolManifest(
    name="file.read",
    description="读取文件内容",
    params=[
        ToolParam("path", "string", "文件路径", required=True),
    ],
))
async def file_read(params: dict, ctx: ToolContext) -> ToolResult:
    ...
```

装饰器在导入时自动注册，`discover(tools_dir)` 扫描目录即可发现所有工具。  
新工具无需修改注册代码，只需新建 `.py` 文件。

### 2.3 动态 import + reload

```python
# tools/registry.py
async def reload_tool(name: str, tools_dir: Path):
    spec = importlib.util.spec_from_file_location(name, tools_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # 工具在模块导入时通过 @tool 自动重新注册
```

进化后的工具立即生效，不需要重启 loop。

---

## 3. Go 的局限

| 能力 | Python | Go |
|---|---|---|
| 运行时代码生成 | ✅ `compile()` + `exec()` | ❌ 只能调用 plugin（CGO） |
| 热替换模块 | ✅ `importlib.reload()` | ❌ 需要重新编译和重启 |
| 装饰器注册模式 | ✅ 原生支持 | ❌ 需要代码生成或反射 |
| 动态 duck typing | ✅ 天然 | ❌ 需要 interface 声明 |
| AI/ML 生态 | ✅ 主场（torch, transformers, etc.） | ⚠️ 有限 |

**Go 的优势**（静态类型、低延迟、并发原语）在数字生命的场景中不是决定性因素：  
- LLM 调用本身延迟在秒级，Go 的微秒级优势被淹没
- 数字生命的复杂性在于认知架构，不在于并发吞吐

---

## 4. 进化机制（evolution.py）

### synthesize_tool：从零生成工具
```python
async def synthesize_tool(name: str, description: str, cfg: Config):
    prompt = f"""
    编写一个 Python 异步工具函数，使用 @tool 装饰器。
    工具名：{name}
    用途：{description}
    """
    code = await provider.chat(prompt)
    compile(code, "<synthesized>", "exec")  # 编译检查
    (tools_dir / f"{name}.py").write_text(code)
    await registry.reload_tool(name, tools_dir)
```

### evolve_tool：修复已有工具
```python
async def evolve_tool(name: str, failure_summary: str, cfg: Config):
    current_code = (tools_dir / f"{name}.py").read_text()
    backup = (tools_dir / f"{name}.bak").open("w")
    backup.write(current_code)           # 保留备份
    
    prompt = f"""
    这个工具持续失败：
    {failure_summary}
    
    当前实现：
    {current_code}
    
    请修复它。
    """
    new_code = await provider.chat(prompt)
    compile(new_code, "<evolved>", "exec")
    (tools_dir / f"{name}.py").write_text(new_code)
    await registry.reload_tool(name, tools_dir)
```

---

## 5. 进化触发条件

```python
# core/loop.py
failures_same_kind = [f for f in failures if f.kind == last_tool_kind]
if len(failures_same_kind) >= cfg.evolution.failure_threshold:  # 默认 3
    await evolution.evolve_tool(last_tool_kind, failures_summary, cfg)
```

连续失败 3 次 → 触发进化 → 工具热替换 → 下一 tick 用新实现。

---

## 6. 进化边界

进化的范围：**除 hard_axioms 之外的一切**。

| 可进化 | 不可进化（自编程） |
|---|---|
| 工具实现（tools/*.py） | `hard_axioms`（已写入 DB 的禁忌条目，只有人类可改） |
| 价值观基线（ethos_baseline） | — |
| 技能定义（skills/*.py） | — |
| 判断模板（prompts/judgment.md） | — |
| 感知逻辑（core/perception.py） | — |
| 记忆检索算法（memory/*.py） | — |

> **注**：hard_axioms 在 `lingzhou init` 时由用户可选配置，不是硬编码在代码里的；  
> 但一旦写入 DB，在运行期间 evolution 不能修改它，只有用户通过 `soul edit` 命令变更。

进化本身的安全机制：

1. **语法检查**：`compile()` 失败 → 拒绝写入，保留原版
2. **备份机制**：写入前先 `.bak`，人工可回滚
3. **轻量安全门**（后置实现）：当前优先级低于 run/worker/double-loop；后续先做命令/路径级 guard，而不是一上来做重型沙箱
4. **禁忌守卫**：evolution 提示词中明确注明不得生成修改 `facts["soul:hard_axioms"]` 的代码

---

## 7. 当前技术选择的含义

**lingzhou** 选择 Python，不是因为“简单”，而是因为：

- 能在运行时生成代码
- 能在同一进程中热替换模块
- 能把进化从“下次重启再说”变成“当下即可生效”

这是 lingzhou 作为“数字生命种子”的决定性技术选择。

---

## 8. 设计原则

1. **自我修改是第一等能力**——不是 nice-to-have，是数字生命的本质
2. **Python 是选择，不是妥协**——动态性 > 性能，在这个场景里
3. **进化有安全门**——compile + backup + axioms，不是盲目自改
4. **工具是可进化的，Soul 不是**——hard_axioms 保持不变
5. **热替换是关键**——不重启、不中断、无感知地升级自身
