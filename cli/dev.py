"""cli/dev.py — dev 子命令组：evolve / tools / skills / model / update / version / doctor。"""
from __future__ import annotations

import asyncio
import json as _json
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Optional

import typer

from cli._common import console, load_cfg, PROJECT_ROOT, DEFAULT_CONFIG_PATH
from core.version import __version__, __codename__

dev_app = typer.Typer(
    name="dev",
    help="开发者工具：evolve / tools / skills / model / update / version / doctor",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _sync_routing_models_on_primary_switch(
    cfg_data: dict,
    *,
    old_model: str,
    new_model: str,
) -> list[str]:
    """同步那些原本跟随旧主模型的 routing 项。

    `lingzhou dev model` 的用户心智是“把当前运行模型切过去”。
    若 routing.reasoner/reader 仍精确指向旧主模型，运行时会继续命中旧路由，
    造成“顶层 model 已切，但真实判断仍像没切”的错觉。
    """
    if not old_model or not new_model or old_model == new_model:
        return []
    routing = cfg_data.get("routing")
    if not isinstance(routing, dict):
        return []
    changed: list[str] = []
    for tier, model_ref in routing.items():
        if model_ref == old_model:
            routing[tier] = new_model
            changed.append(str(tier))
    return changed


def _preferred_model_index(catalog_models: list[dict], current_model_id: str = "") -> int:
    """优先当前模型；否则优先 reasoning/thinking 模型；都没有再退回列表首项。"""
    if not catalog_models:
        return -1
    if current_model_id:
        for idx, model in enumerate(catalog_models):
            if str(model.get("id") or "") == current_model_id:
                return idx
    for idx, model in enumerate(catalog_models):
        if model.get("reasoning") or model.get("thinking"):
            return idx
    return 0


@dev_app.command("evolve")
def evolve(
    description: Annotated[str, typer.Argument(help="新工具的自然语言描述")],
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
) -> None:
    """合成并热加载一个新工具（自进化）。"""
    cfg = load_cfg(config)

    async def _run() -> None:
        from provider import create_provider
        from tools.registry import ToolRegistry
        from core.evolution import EvolutionEngine

        provider = create_provider(cfg)
        registry = ToolRegistry()
        engine = EvolutionEngine(cfg, provider, registry)
        result = await engine.synthesize_tool(description)
        await provider.close()
        if result.success:
            console.print(f"[green]工具 {result.target!r} 已合成[/green]")
        else:
            console.print(f"[red]合成失败: {result.reason}[/red]")

    asyncio.run(_run())


@dev_app.command("tools")
def tools(
    search: Annotated[Optional[str], typer.Argument(help="关键词过滤")] = None,
) -> None:
    """列出所有已注册的工具（支持关键词过滤）。"""
    from tools.registry import ToolRegistry

    reg = ToolRegistry()
    tools_dir = PROJECT_ROOT / "tools"
    reg.discover(tools_dir)
    manifests = reg.list_manifests()

    if search:
        kw = search.lower()
        manifests = [
            m for m in manifests
            if kw in m.name.lower() or kw in (m.description or "").lower()
        ]

    if not manifests:
        console.print("（没有匹配的工具）")
        return

    console.print(f"[bold]已注册工具[/bold]  ({len(manifests)} 个)\n")
    for m in sorted(manifests, key=lambda x: x.name):
        console.print(f"  [cyan]{m.name:<26}[/cyan] {m.description or ''}")


@dev_app.command("skills")
def skills(
    search: Annotated[Optional[str], typer.Argument(help="关键词过滤")] = None,
    disabled: Annotated[bool, typer.Option("--disabled", help="显示已禁用 skills，而不是 active skills")] = False,
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
) -> None:
    """列出当前 workspace 中可被运行态加载的 skills。"""
    from core.skill import SkillRegistry

    cfg = load_cfg(config)
    skills_dir = Path(cfg.loop.workspace_dir).expanduser() / ("skills-disabled" if disabled else "skills")
    reg = SkillRegistry(skills_dir=skills_dir)
    items = [s for s in reg.all_skills() if bool(s.source_path)]

    if search:
        kw = search.lower()
        items = [
            s for s in items
            if kw in s.name.lower()
            or kw in (s.description or "").lower()
            or kw in " ".join(s.triggers).lower()
        ]

    state = "disabled" if disabled else "active"
    if not items:
        console.print(f"（没有匹配的 {state} skills）")
        return

    console.print(f"[bold]{state} skills[/bold]  ({len(items)} 个)  [dim]{skills_dir}[/dim]\n")
    for s in sorted(items, key=lambda x: x.name):
        trig = f"  [dim]triggers: {', '.join(s.triggers[:6])}[/dim]" if s.triggers else ""
        console.print(f"  [cyan]{s.name:<24}[/cyan] {s.description}{trig}")


@dev_app.command("model")
def model(
    set_model: Annotated[Optional[str], typer.Argument(help="要切换的模型 ID，如 bailian/qwen-plus")] = None,
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    list_all: Annotated[bool, typer.Option("--list", "-l", help="列出所有可用模型")] = False,
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="交互式选择 provider 和模型")] = False,
) -> None:
    """查看或切换当前使用的 LLM provider / 模型。"""
    from provider.catalog import list_providers, list_provider_models

    if list_all:
        for pname in list_providers():
            models_list = list_provider_models(pname)
            console.print(f"\n[bold]{pname}[/bold]")
            for m in models_list:
                ctx_k = (m.get("context_window") or 0) // 1000
                tags = []
                if m.get("thinking"):
                    tags.append("thinking")
                if m.get("reasoning"):
                    tags.append("reasoning")
                tag_str = f"  [dim][{', '.join(tags)}][/dim]" if tags else ""
                ctx_str = f"  [dim]{ctx_k}K[/dim]" if ctx_k else ""
                console.print(f"  {m['id']}{ctx_str}{tag_str}")
        return

    cfg_path = config if config.exists() else None
    # 尝试在搜索路径中找到配置
    if cfg_path is None:
        from cli._common import find_config
        try:
            cfg_path = find_config(config)
        except SystemExit:
            cfg_path = None

    if cfg_path is None or not cfg_path.exists():
        console.print(f"[red]配置文件不存在: {config}，请先运行 lingzhou setup[/red]")
        raise typer.Exit(1)

    cfg_data = _json.loads(cfg_path.read_text(encoding="utf-8"))
    current = cfg_data.get("model", "(未设置)")
    chosen_thinking = cfg_data.get("thinking", "off")
    current_provider, _, current_model_id = str(current).partition("/")

    # ── 交互式选择 ─────────────────────────────────────────────────────────
    if interactive or (not set_model):
        console.print(f"当前模型: [bold cyan]{current}[/bold cyan]")
        if not interactive:
            console.print(f"[dim]切换模型: lingzhou model <provider/model-id>[/dim]")
            console.print(f"[dim]交互切换: lingzhou model -i[/dim]")
            console.print(f"[dim]查看全部: lingzhou model --list[/dim]")
            return

        # 交互式：先选 provider
        configured_providers = list(cfg_data.get("providers", {}).keys())
        all_catalog = list_providers()
        # 配置了的 provider 排在前面
        ordered = configured_providers + [p for p in all_catalog if p not in configured_providers]

        console.print("\n[bold]选择 provider[/bold]")
        for i, p in enumerate(ordered, 1):
            mark = "[green]✓[/green]" if p in configured_providers else "[dim]  [/dim]"
            console.print(f"  {i}. {mark} {p}")

        raw_p = typer.prompt("Provider 编号", default="1")
        try:
            pidx = int(raw_p.strip()) - 1
        except ValueError:
            pidx = 0
        if not (0 <= pidx < len(ordered)):
            console.print("[red]无效编号[/red]")
            raise typer.Exit(1)
        chosen_provider = ordered[pidx]

        # 如果选了未配置的 provider，引导用户补充配置并写入 lingzhou.json
        if chosen_provider not in configured_providers:
            _BUILTIN_PROVIDER_DEFAULTS: dict[str, dict] = {
                "bailian": {
                    "type": "openai_compat",
                    "mode": "openai",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key_env": "DASHSCOPE_API_KEY",
                },
                "copilot": {
                    "type": "openai_compat",
                    "mode": "copilot",
                    "base_url": "https://api.individual.githubcopilot.com",
                    "api_key_env": "GITHUB_TOKEN",
                },
            }
            defaults = _BUILTIN_PROVIDER_DEFAULTS.get(chosen_provider, {
                "type": "openai_compat",
                "mode": "openai",
                "base_url": "",
                "api_key_env": "OPENAI_API_KEY",
            })

            new_provider_cfg: dict = {
                "type": defaults["type"],
                "mode": defaults["mode"],
                "base_url": defaults["base_url"],
                "api_key_env": defaults["api_key_env"],
                "auth_profile_id": f"{chosen_provider}:default",
            }

            # copilot 走 auth login 的 token exchange 链，不需要手动填 key
            if chosen_provider == "copilot":
                from auth_store import get_auth_profile, COPILOT_PROFILE_ID
                existing_auth = get_auth_profile(COPILOT_PROFILE_ID)
                if existing_auth and existing_auth.get("token"):
                    console.print(f"\n[green]✓ 已检测到 Copilot 登录凭证[/green]  [dim](lingzhou auth login 已完成)[/dim]")
                else:
                    console.print(f"\n[yellow]Copilot 尚未登录[/yellow]")
                    console.print(f"  请在切换后运行: [bold]lingzhou auth login[/bold]")
            else:
                # 其他 provider 需要手动输入 API key 或环境变量名
                import re as _re
                console.print(f"\n[yellow]{chosen_provider} 未在配置中，现在为你补充配置。[/yellow]")
                api_key_input = typer.prompt(
                    "  环境变量名或直接粘贴 API key",
                    default=defaults["api_key_env"],
                )
                new_provider_cfg["api_key_env"] = api_key_input
                # 如果输入的不是 ENV_VAR 格式（直接贴了 key），存 credentials.json
                if api_key_input and not _re.match(r'^[A-Z_][A-Z0-9_]*$', api_key_input.strip()):
                    cred_file = Path.home() / ".lingzhou" / "credentials.json"
                    cred_file.parent.mkdir(parents=True, exist_ok=True)
                    creds: dict = {}
                    if cred_file.exists():
                        try:
                            creds = _json.loads(cred_file.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                    cred_key = f"{chosen_provider.upper()}_API_KEY"
                    creds[cred_key] = api_key_input.strip()
                    cred_file.write_text(_json.dumps(creds, ensure_ascii=False, indent=2), encoding="utf-8")
                    cred_file.chmod(0o600)
                    new_provider_cfg["api_key_env"] = cred_key
                    console.print(f"  [dim]key 已安全存入 {cred_file}，配置中使用 {cred_key}[/dim]")

            if "providers" not in cfg_data:
                cfg_data["providers"] = {}
            cfg_data["providers"][chosen_provider] = new_provider_cfg
            configured_providers.append(chosen_provider)
            console.print(f"[green]✓ {chosen_provider} 已添加到配置[/green]")

        # 选模型
        catalog_models = list_provider_models(chosen_provider)
        console.print(f"\n[bold]选择模型[/bold]  [dim](provider={chosen_provider})[/dim]")
        if catalog_models:
            preferred_index = _preferred_model_index(
                catalog_models,
                current_model_id=current_model_id if chosen_provider == current_provider else "",
            )
            for i, m in enumerate(catalog_models, 1):
                ctx_k = (m.get("context_window") or 0) // 1000
                tags = []
                if m.get("thinking"):
                    tags.append("thinking")
                if m.get("reasoning"):
                    tags.append("reasoning")
                ctx_str = f"  [dim]{ctx_k}K[/dim]" if ctx_k else ""
                tag_str = f"  [dim][{', '.join(tags)}][/dim]" if tags else ""
                mark = " [bold cyan]← 默认[/bold cyan]" if i - 1 == preferred_index else ""
                console.print(f"  {i}. {m['id']}{ctx_str}{tag_str}{mark}")
            console.print(f"  {len(catalog_models)+1}. 手动输入")
            raw_m = typer.prompt("  模型编号", default=str(preferred_index + 1))
            try:
                midx = int(raw_m.strip()) - 1
            except ValueError:
                midx = -1
            if 0 <= midx < len(catalog_models):
                chosen_model_id = catalog_models[midx]["id"]
            else:
                chosen_model_id = typer.prompt("  手动输入模型 ID")
        else:
            chosen_model_id = typer.prompt(f"  {chosen_provider} 模型 ID")

        set_model = f"{chosen_provider}/{chosen_model_id}"

        # 选思考等级
        _THINKING_LEVELS = ["off", "minimal", "low", "medium", "high"]
        _THINKING_DESC = {
            "off":     "关闭思考，速度最快，省 token",
            "minimal": "极浅思考，轻量推理",
            "low":     "低强度思考，例行决策",
            "medium":  "中等思考，常规判断（推荐日常）",
            "high":    "深度思考，复杂推理/代码生成",
        }
        current_thinking = cfg_data.get("thinking", "off")
        console.print(f"\n[bold]选择思考等级[/bold]  [dim](当前: {current_thinking})[/dim]")
        for i, lvl in enumerate(_THINKING_LEVELS, 1):
            mark = "[bold cyan]●[/bold cyan]" if lvl == current_thinking else " "
            console.print(f"  {i}. {mark} {lvl:<8} [dim]{_THINKING_DESC[lvl]}[/dim]")
        cur_default = str(_THINKING_LEVELS.index(current_thinking) + 1) if current_thinking in _THINKING_LEVELS else "1"
        raw_t = typer.prompt("  等级编号", default=cur_default)
        try:
            tidx = int(raw_t.strip()) - 1
        except ValueError:
            tidx = -1
        chosen_thinking = _THINKING_LEVELS[tidx] if 0 <= tidx < len(_THINKING_LEVELS) else current_thinking

    # ── 写入配置 ───────────────────────────────────────────────────────────
    cfg_data["model"] = set_model
    synced_routing = _sync_routing_models_on_primary_switch(
        cfg_data,
        old_model=current,
        new_model=set_model,
    )
    if not interactive:
        chosen_thinking = cfg_data.get("thinking", "off")  # 非交互模式保持原值

    old_thinking = cfg_data.get("thinking", "off")
    cfg_data["thinking"] = chosen_thinking
    cfg_path.write_text(_json.dumps(cfg_data, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]✓ 模型已切换:[/green] {current} → [bold cyan]{set_model}[/bold cyan]")
    if synced_routing:
        console.print(
            f"[green]✓ 已同步 routing:[/green] {', '.join(synced_routing)} → [bold cyan]{set_model}[/bold cyan]"
        )
    if chosen_thinking != old_thinking:
        console.print(f"[green]✓ 思考等级已更新:[/green] {old_thinking} → [bold cyan]{chosen_thinking}[/bold cyan]")
    console.print("[dim]lingzhou 运行中时将在下一轮自动生效（配置热重载）[/dim]")


@dev_app.command("update")
def update() -> None:
    """更新 lingzhou 到最新版本（git pull + 重新安装依赖）。"""
    console.print(f"当前版本: [bold]v{__version__}[/bold]  代号: {__codename__}")

    repo_dir = PROJECT_ROOT
    if not (repo_dir / ".git").exists():
        console.print("[yellow]当前目录不是 git 工作区，请手动拉取最新代码后重新安装：[/yellow]")
        console.print("  git pull && uv pip install -e .")
        raise typer.Exit(1)

    console.print("[dim]执行 git pull...[/dim]")
    result = subprocess.run(["git", "pull"], cwd=repo_dir, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]git pull 失败:[/red]\n{result.stderr.strip()}")
        raise typer.Exit(1)
    console.print(f"[green]{result.stdout.strip() or 'Already up to date.'}[/green]")

    uv = shutil.which("uv")
    pip_cmd = [uv, "pip", "install", "-e", "."] if uv else [
        shutil.which("pip") or "pip", "install", "-e", "."
    ]
    console.print(f"[dim]重装依赖: {' '.join(pip_cmd)}[/dim]")
    result = subprocess.run(pip_cmd, cwd=repo_dir, capture_output=True, text=True)
    if result.returncode == 0:
        console.print("[green]✓ 更新完成，重启 lingzhou 生效[/green]")
    else:
        console.print(f"[red]依赖安装失败:[/red]\n{result.stderr.strip()}")
        raise typer.Exit(1)


# 注册诊断命令
from cli.diag import version, doctor  # noqa: E402

dev_app.command("version")(version)
dev_app.command("doctor")(doctor)
