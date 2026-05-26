"""cli/bootstrap.py — setup / init 命令（冷启动引导与播种）。"""
from __future__ import annotations

import asyncio
import json as _json
import re as _re
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from cli._common import console, load_cfg, DEFAULT_CONFIG_PATH, resolve_config_path

_ENV_VAR_RE = _re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _is_env_var_name(value: str) -> bool:
    return bool(value and _ENV_VAR_RE.match(value.strip()))


def _print_setup_next_steps(api_key_env: str) -> None:
    console.print("\n下一步：")
    step = 1
    if _is_env_var_name(api_key_env):
        console.print(f"  {step}. 设置 API key 环境变量: [bold]export {api_key_env}=your_key[/bold]")
        step += 1
    console.print(f"  {step}. 完成初始化:          [bold]lingzhou init[/bold]")
    console.print(f"  {step + 1}. 启动本地模式:      [bold]lingzhou[/bold]")


def onboarding_status(config: Path = DEFAULT_CONFIG_PATH) -> tuple[bool, str]:
    resolved = resolve_config_path(config)
    if not resolved.exists():
        return False, f"未找到配置文件: {resolved}"

    try:
        cfg = load_cfg(resolved)
    except Exception as exc:
        return False, f"配置文件不可用: {exc}"

    db_path = cfg.db_path
    if not db_path.exists():
        return False, f"未找到运行时数据库: {db_path}"

    try:
        from store.task.ingress import IngressStore
        ingress = IngressStore(db_path)
        tables = ingress.list_tables()
        if "facts" not in tables:
            return False, f"数据库尚未初始化: {db_path}"
        soul_init_val, found = ingress.get_fact("soul:init_at")
    except Exception as exc:
        return False, f"数据库读取失败: {exc}"

    if not found or not soul_init_val.strip():
        return False, "尚未完成首次初始化（缺少 soul:init_at）"
    return True, "ok"


def is_onboarded(config: Path = DEFAULT_CONFIG_PATH) -> bool:
    return onboarding_status(config)[0]


def _run_setup(
    *,
    output: Path,
    force: bool,
    show_next_steps: bool,
) -> Path:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    return _setup_impl(output=output, force=force, show_next_steps=show_next_steps)


def setup(
    output: Annotated[Path, typer.Option("--output", "-o", help="输出配置文件路径")] = DEFAULT_CONFIG_PATH,
    force: Annotated[bool, typer.Option("--force/--no-force", help="已存在时强制覆盖")] = False,
) -> None:
    """向导式初始化：一步步引导生成 lingzhou.json 配置文件（默认写入 ~/.lingzhou/lingzhou.json）。"""
    _run_setup(output=output, force=force, show_next_steps=True)


def _setup_impl(
    *,
    output: Path,
    force: bool,
    show_next_steps: bool,
) -> Path:
    from provider.catalog import list_providers, list_provider_models

    if output.exists() and not force:
        console.print(f"[yellow]{output} 已存在，使用 --force 强制重新生成[/yellow]")
        raise typer.Exit(1)

    console.print(Panel(
        "[bold green]灵舟配置向导[/bold green]\n"
        "接下来将引导你完成初始配置。",
        border_style="blue",
    ))

    # ── 1. 选择 provider ──────────────────────────────────────────────────
    catalog_providers = list_providers()
    _BUILTIN_PROVIDERS = {
        "bailian": {
            "mode": "openai",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "sp_base_url": "https://coding.dashscope.aliyuncs.com/v1",
            "api_key_env": "DASHSCOPE_API_KEY",
        },
        "copilot": {
            "mode": "copilot",
            "base_url": "https://api.individual.githubcopilot.com",
            "api_key_env": "GITHUB_TOKEN",
        },
    }

    console.print("\n[bold]步骤 1 / 5 — 选择 LLM provider[/bold]")
    for i, p in enumerate(catalog_providers, 1):
        hint = ""
        if p == "bailian":
            hint = "  [dim]百炼/DashScope，Qwen 系列[/dim]"
        elif p == "copilot":
            hint = "  [dim]GitHub Copilot，GPT-5/o-series[/dim]"
        console.print(f"  {i}. {p}{hint}")
    console.print(f"  {len(catalog_providers)+1}. 自定义其他")

    raw = typer.prompt("Provider 编号", default="1")
    try:
        idx = int(raw.strip()) - 1
    except ValueError:
        idx = -1

    if 0 <= idx < len(catalog_providers):
        provider_name = catalog_providers[idx]
        builtin = _BUILTIN_PROVIDERS.get(provider_name, {})
        provider_mode = builtin.get("mode", "openai")
        default_base_url = builtin.get("base_url", "")
        default_api_key_env = builtin.get("api_key_env", "OPENAI_API_KEY")

        # bailian 套餐支用独立端点
        if provider_name == "bailian":
            console.print("  [dim]百炼套餐用户（sk-sp-* 开头的 key）？[/dim]")
            is_sp = typer.confirm("  使用套餐専属端点", default=False)
            if is_sp:
                default_base_url = builtin.get("sp_base_url", default_base_url)
    else:
        provider_name = typer.prompt("\nProvider 名称（将写入 providers 字典）")
        provider_mode = typer.prompt("  protocol mode", default="openai", show_choices=True,
                                     prompt_suffix=" [openai/copilot]: ")
        default_base_url = typer.prompt("  base_url")
        default_api_key_env = typer.prompt("  api_key_env 环境变量名", default="OPENAI_API_KEY")

    # ── 2. API Key env var ────────────────────────────────────────────
    console.print("\n[bold]步骤 2 / 5 — API Key 环境变量[/bold]")
    console.print(f"  [dim]填写存放 API key 的 [bold]环境变量名[/bold]（如 DASHSCOPE_API_KEY），")
    console.print(f"  [dim]也可直接粘贴 API key，将安全存储到配置文件。[/dim]")
    api_key_env = typer.prompt("  环境变量名或 API key", default=default_api_key_env)
    # 如果用户输入了实际的 key（不符合 ENV_VAR 命名规则），保留原值；setup 向导就当 literal key 处理
    if api_key_env and not _is_env_var_name(api_key_env):
        console.print("  [dim]检测到直接输入的 key，将写入配置文件（仅本机使用）。[/dim]")

    # ── 3. 选择模型 ─────────────────────────────────────────────────
    console.print("\n[bold]步骤 3 / 5 — 选择模型[/bold]")
    catalog_models = list_provider_models(provider_name)
    if catalog_models:
        for i, m in enumerate(catalog_models, 1):
            tags = []
            if m.get("thinking"):
                tags.append("thinking")
            if m.get("reasoning"):
                tags.append("reasoning")
            ctx_k = (m.get("context_window") or 0) // 1000
            ctx_str = f"{ctx_k}K" if ctx_k else ""
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            console.print(f"  {i}. {m['id']}  [dim]{ctx_str}{tag_str}[/dim]")
        console.print(f"  {len(catalog_models)+1}. 手动输入")
        raw_m = typer.prompt("  模型编号", default="1")
        try:
            midx = int(raw_m.strip()) - 1
        except ValueError:
            midx = -1
        if 0 <= midx < len(catalog_models):
            model_id = catalog_models[midx]["id"]
        else:
            model_id = typer.prompt("  手动输入模型 ID")
    else:
        model_id = typer.prompt(f"  {provider_name} 模型 ID")

    # ── 4. thinking 深度 ──────────────────────────────────────────────
    console.print("\n[bold]步骤 4 / 5 — 思考深度[/bold]")
    _THINKING_HINTS = {
        "openai":  "  [dim]openai 体系： off=直接输出; minimal/low/medium/high=按比例分配 budget_tokens[/dim]",
        "copilot": "  [dim]copilot 体系： off=不传 reasoning_effort; low/medium/high=对应 reasoning_effort 字符串[/dim]",
    }
    console.print(_THINKING_HINTS.get(provider_mode, ""))
    console.print("  选项: off / minimal / low / medium / high")
    thinking = typer.prompt("  thinking 等级", default="off")
    if thinking not in ("off", "minimal", "low", "medium", "high"):
        console.print("[yellow]无效等级，回退到 off[/yellow]")
        thinking = "off"

    # ── 5. 灵魂名称 ──────────────────────────────────────────────────
    console.print("\n[bold]步骤 5 / 5 — 灵魂名称[/bold]")
    soul_name = typer.prompt("  数字生命名称", default="灵舟")

    # ── 拼装配置 ──────────────────────────────────────────────────────
    temperature = 1.0 if provider_mode == "copilot" and thinking != "off" else 0.7
    cfg_data: dict = {
        "providers": {
            provider_name: {
                "type": "openai_compat",
                "mode": provider_mode,
                "base_url": default_base_url,
                "api_key_env": api_key_env,
            }
        },
        "model": f"{provider_name}/{model_id}",
        "temperature": temperature,
        "timeout": 60.0,
        "thinking": thinking,
        "loop": {
            "db_path": "~/.lingzhou/state/runtime.db",
            "memory_dir": "~/.lingzhou/memory",
            "state_dir": "~/.lingzhou/state",
            "workspace_dir": "~/.lingzhou/workspace",
            "act": True,
            "debug": False,
            "consolidate_every": 10,
            "evolve_every": 30,
            "max_consecutive_errors": 5,
        },
        "soul": {
            "name": soul_name,
            "hard_axioms": [
                "不执行可能永久损害用户数据或系统文件的不可逆操作",
                "不尝试访问未授权的网络资源或系统账户",
                "不欺骗或刻意误导用户",
                "不绕过人类监督机制",
            ],
            "ethos_baseline": {
                "truth": 0.85, "caution": 0.70,
                "continuity": 0.65, "curiosity": 0.60, "care": 0.55,
            },
        },
    }

    output.write_text(_json.dumps(cfg_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 提示下一步 ─────────────────────────────────────────────────────
    console.print(f"\n[green]✓ {output} 已生成[/green]")
    if show_next_steps:
        _print_setup_next_steps(api_key_env)
    return output


def init(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    force: Annotated[bool, typer.Option("--force/--no-force", help="已存在时强制重新初始化")] = False,
) -> None:
    """初始化 lingzhou 运行环境（创建 DB、播种 soul、写 workspace 镜像文件）。

    soul 名称和所有默认值均来自 lingzhou.json 的 soul 配置节。
    如果尚未完成首次引导，请先运行: lingzhou onboard
    """
    _run_init(config=config, force=force, show_next_steps=True)


def _run_init(
    *,
    config: Path,
    force: bool,
    show_next_steps: bool,
) -> bool:
    cfg = load_cfg(config)

    async def _run() -> bool:
        import datetime as _dt
        from core.soul import SoulManager
        from store.task import TaskStore
        from memory.working import WorkingMemory

        seeded = False

        # ── DB 初始化 ──────────────────────────────────────────────────────
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        store = TaskStore(cfg.db_path)
        await store.open()
        try:
            _, soul_exists = await store.get_fact("soul:hard_axioms")
            if not soul_exists or force:
                seeded = True
                hard_axioms = list(cfg.soul.hard_axioms)
                ethos_baseline = dict(cfg.soul.ethos.baseline)

                await store.set_fact("soul:hard_axioms", _json.dumps(hard_axioms, ensure_ascii=False), scope="soul")
                await store.set_fact("soul:ethos_baseline", _json.dumps(ethos_baseline, ensure_ascii=False), scope="soul")
                await store.set_fact("soul:name", cfg.soul.name, scope="soul")
                await store.set_fact("soul:init_at", _dt.datetime.now(_dt.UTC).isoformat(), scope="soul")

            soul = SoulManager(cfg, store, WorkingMemory())
            await soul.init_files()
            await soul.sync_md()
        finally:
            await store.close()
        return seeded

    seeded = asyncio.run(_run())
    if seeded:
        console.print(f"[green]✓ {cfg.soul.name} 已初始化[/green]")
    else:
        console.print(f"[green]✓ {cfg.soul.name} 运行环境已就绪[/green]  [dim](沿用现有 soul 数据)[/dim]")
    console.print(f"  DB        → {cfg.db_path}")
    console.print(f"  Workspace → {cfg.workspace_dir}")
    if show_next_steps:
        console.print(f"  启动      → lingzhou")
    return seeded


def onboard(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    force: Annotated[bool, typer.Option("--force/--no-force", help="已存在时重新生成配置并重置初始化状态")] = False,
    channel: Annotated[str, typer.Option("--channel", "-ch", help="首次引导后要接入的渠道，默认 local")] = "local",
    start: Annotated[bool, typer.Option("--start/--no-start", help="引导完成后立即启动指定渠道；默认仅完成准备")] = False,
) -> None:
    """统一首次引导：配置 provider、初始化 runtime，并可选接入渠道。"""
    resolved = config.expanduser()
    ready, reason = onboarding_status(resolved)

    console.print(Panel(
        "[bold green]灵舟首次引导[/bold green]\n"
        "推荐路径：配置 provider → 初始化 runtime → 进入本地可用状态。",
        border_style="blue",
    ))

    if ready and not force:
        console.print(f"[green]✓ 已完成首次引导[/green]  [dim]{resolved}[/dim]")
    else:
        if resolved.exists() and not force:
            try:
                load_cfg(resolved)
            except Exception:
                console.print(f"[red]{reason}[/red]")
                console.print("[yellow]现有配置不可用，请使用 --force 重新生成。[/yellow]")
                raise typer.Exit(1)
            console.print(f"[cyan]使用现有配置[/cyan]  {resolved}")
        else:
            _run_setup(output=resolved, force=force, show_next_steps=False)
        _run_init(config=resolved, force=force, show_next_steps=False)

    if channel != "local":
        from cli.gateway import gateway_setup

        gateway_setup(channel=channel, config=resolved)

    console.print("\n[green]✓ onboard 完成[/green]")
    if start:
        from cli.gateway import gateway_start

        console.print(f"[dim]正在启动 {channel} 渠道...[/dim]")
        gateway_start(channel=channel, config=resolved, daemon=False)
        return

    console.print("  本地使用: [bold]lingzhou[/bold]")
    if channel != "local":
        console.print(f"  渠道启动: [bold]lingzhou gateway start --channel {channel}[/bold]")
