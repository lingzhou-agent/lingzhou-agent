"""cli/dev.py — dev 子命令组：evolve / tools / skills / model / update / version / doctor。"""
from __future__ import annotations

import asyncio
import json as _json
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from cli._common import console, load_cfg, PROJECT_ROOT, DEFAULT_CONFIG_PATH
from core.version import __version__, __codename__

dev_app = typer.Typer(
    name="dev",
    help="开发者工具：evolve / tools / skills / model / update / version / doctor",
    context_settings={"help_option_names": ["-h", "--help"]},
)

_RUNTIME_ROUTING_TIERS = frozenset({"reader", "reasoner", "repair"})
_MODEL_TARGET_ALIASES = {
    "": "primary",
    "model": "primary",
    "main": "primary",
    "primary": "primary",
    "thinking": "reasoner",
    "reasoner": "reasoner",
    "complex": "reasoner",
    "reader": "reader",
    "simple": "reader",
    "repair": "repair",
}


def _provider_name(model_ref: str) -> str:
    provider, _, _ = model_ref.partition("/")
    return provider


def _normalize_model_target(target: str) -> str:
    return _MODEL_TARGET_ALIASES.get((target or "").strip().lower(), (target or "").strip())


def _effective_target_model(cfg_data: dict[str, Any], target: str) -> str:
    normalized = _normalize_model_target(target)
    if normalized == "primary":
        return str(cfg_data.get("model") or "")
    routing = cfg_data.get("routing")
    if isinstance(routing, dict):
        model_ref = routing.get(normalized)
        if isinstance(model_ref, str) and model_ref:
            return model_ref
    return str(cfg_data.get("model") or "")


def _apply_model_target_selection(
    cfg_data: dict[str, Any],
    *,
    current_model: str,
    new_model: str,
    target: str,
) -> dict[str, Any]:
    normalized = _normalize_model_target(target)
    if normalized == "primary":
        previous = str(cfg_data.get("model") or current_model)
        cfg_data["model"] = new_model
        return {
            "target": "primary",
            "previous": previous,
            "routing_changed": _sync_routing_models_on_primary_switch(
                cfg_data,
                old_model=current_model,
                new_model=new_model,
            ),
            "runtime_override_tier": None,
        }

    routing = cfg_data.get("routing")
    if not isinstance(routing, dict):
        routing = {}
        cfg_data["routing"] = routing

    previous = _effective_target_model(cfg_data, normalized)
    routing[normalized] = new_model
    return {
        "target": normalized,
        "previous": previous,
        "routing_changed": [normalized],
        "runtime_override_tier": normalized if normalized in _RUNTIME_ROUTING_TIERS else None,
    }


def _merge_runtime_routing_override(overrides: dict[str, str], *, tier: str, model_ref: str) -> dict[str, str]:
    merged = {
        key: value
        for key, value in overrides.items()
        if key in _RUNTIME_ROUTING_TIERS and isinstance(value, str) and value
    }
    merged[tier] = model_ref
    return merged


def _set_db_routing_override(cfg_path: Path, *, tier: str, model_ref: str) -> None:
    if tier not in _RUNTIME_ROUTING_TIERS or not model_ref:
        return

    import sqlite3 as _sqlite3

    try:
        from core.config import Config as _Config

        cfg = _Config.load(cfg_path)
        db_path = cfg.db_path
        if not db_path.exists():
            return

        conn = _sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT value FROM facts WHERE key='pref:routing_overrides'"
            ).fetchone()
            overrides: dict[str, str] = {}
            if row and row[0]:
                payload = _json.loads(row[0])
                if isinstance(payload, dict):
                    overrides = {
                        key: value
                        for key, value in payload.items()
                        if isinstance(key, str) and isinstance(value, str)
                    }

            merged = _merge_runtime_routing_override(overrides, tier=tier, model_ref=model_ref)
            serialized = _json.dumps(merged, ensure_ascii=False)
            if row:
                conn.execute(
                    "UPDATE facts SET value=?, scope='system', updated_at=datetime('now') WHERE key='pref:routing_overrides'",
                    (serialized,),
                )
            else:
                conn.execute(
                    "INSERT INTO facts (key, value, scope, updated_at) VALUES ('pref:routing_overrides', ?, 'system', datetime('now'))",
                    (serialized,),
                )
            conn.commit()
        finally:
            conn.close()

        console.print(
            f"[green]✓ 运行时 {tier} override 已同步:[/green] [bold cyan]{model_ref}[/bold cyan]"
        )
    except Exception as exc:
        console.print(f"[yellow]⚠ 运行时 {tier} override 同步失败（非致命）: {exc}[/yellow]")


def _sync_routing_models_on_primary_switch(
    cfg_data: dict,
    *,
    old_model: str,
    new_model: str,
) -> list[str]:
    """切换主模型时，仅同步跟随主模型的 routing 条目。

    规则：
    - 常规切换时，仅同步精确指向旧主模型的 routing 条目。
    - 若用户重选当前主模型，则修复仍停留在同 provider 旧模型上的残留 routing 条目。

    这样既能让主推理链路跟随当前主模型，又不会覆盖 reader 等明确分流到其他 provider 的配置。
    """
    if not old_model or not new_model:
        return []
    routing = cfg_data.get("routing")
    if not isinstance(routing, dict):
        return []

    repair_same_provider = old_model == new_model
    new_provider = _provider_name(new_model)
    changed: list[str] = []
    for tier, model_ref in routing.items():
        if not isinstance(model_ref, str) or model_ref == new_model:
            continue
        if model_ref == old_model or (
            repair_same_provider
            and new_provider
            and _provider_name(model_ref) == new_provider
        ):
            routing[tier] = new_model
            changed.append(str(tier))
    return changed


def _sync_db_routing_overrides(cfg_path: Path, *, old_model: str, new_model: str) -> None:
    """将 DB pref:routing_overrides 里精确指向 old_model 的条目更新为 new_model。

    DB routing_overrides 优先级高于 lingzhou.json；若不同步，切换主模型后
    重启仍会从 DB 恢复到旧模型，导致 `dev model` 看似不生效。
    只替换精确等于 old_model 的条目，保留用户有意设置的差异（如 reader: bailian）。
    """
    import sqlite3 as _sqlite3
    if not old_model or not new_model or old_model == new_model:
        return
    try:
        from core.config import Config as _Config
        cfg = _Config.load(cfg_path)
        db_path = cfg.db_path
        if not db_path.exists():
            return
        conn = _sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT value FROM facts WHERE key='pref:routing_overrides'"
        ).fetchone()
        if not row:
            conn.close()
            return
        overrides = _json.loads(row[0])
        changed = [tier for tier, model in overrides.items() if model == old_model]
        for tier in changed:
            overrides[tier] = new_model
        if changed:
            conn.execute(
                "UPDATE facts SET value=? WHERE key='pref:routing_overrides'",
                (_json.dumps(overrides, ensure_ascii=False),),
            )
            conn.commit()
            console.print(
                f"[green]✓ DB routing_overrides 已同步:[/green]"
                f" {', '.join(changed)} → [bold cyan]{new_model}[/bold cyan]"
            )
        conn.close()
    except Exception as exc:
        console.print(f"[yellow]⚠ DB routing_overrides 同步失败（非致命）: {exc}[/yellow]")
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
    items = [s for s in reg.all_skills() if getattr(s, "origin", "builtin") == "workspace"]

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
    model_target = "primary"
    current_target_model = str(current)
    current_provider, _, current_model_id = current_target_model.partition("/")

    # ── 交互式选择 ─────────────────────────────────────────────────────────
    if interactive or (not set_model):
        console.print(f"当前模型: [bold cyan]{current}[/bold cyan]")
        if not interactive:
            console.print(f"[dim]切换模型: lingzhou model <provider/model-id>[/dim]")
            console.print(f"[dim]交互切换: lingzhou dev model -i[/dim]")
            console.print(f"[dim]查看全部: lingzhou model --list[/dim]")
            return

        console.print("\n[bold]选择要设置的模型槽位[/bold]")
        target_options = [
            ("primary", f"主模型 (model)  [dim]当前: {current}[/dim]"),
            ("reasoner", f"思考层 (reasoner)  [dim]当前: {_effective_target_model(cfg_data, 'reasoner')}[/dim]"),
            ("reader", f"Reader 层 (reader)  [dim]当前: {_effective_target_model(cfg_data, 'reader')}[/dim]"),
            ("repair", f"Repair 层 (repair)  [dim]当前: {_effective_target_model(cfg_data, 'repair')}[/dim]"),
            ("other", "其他 routing 键（手动输入）"),
        ]
        for i, (_, label) in enumerate(target_options, 1):
            console.print(f"  {i}. {label}")

        raw_target = typer.prompt("模型槽位编号", default="1")
        try:
            target_idx = int(raw_target.strip()) - 1
        except ValueError:
            target_idx = 0
        if not (0 <= target_idx < len(target_options)):
            console.print("[red]无效编号[/red]")
            raise typer.Exit(1)

        selected_target = target_options[target_idx][0]
        if selected_target == "other":
            entered_key = typer.prompt("  输入 routing 键", default="reasoner")
            model_target = _normalize_model_target(entered_key)
        else:
            model_target = selected_target

        current_target_model = _effective_target_model(cfg_data, model_target)
        current_provider, _, current_model_id = current_target_model.partition("/")
        console.print(
            f"[dim]本次将设置 {_normalize_model_target(model_target)} 模型槽位，当前值: {current_target_model or current}[/dim]"
        )

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
                "deepseek": {
                    "type": "openai_compat",
                    "mode": "openai",
                    "base_url": "https://api.deepseek.com",
                    "api_key_env": "DEEPSEEK_API_KEY",
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
                from store.auth import get_auth_profile, COPILOT_PROFILE_ID
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

        if _normalize_model_target(model_target) == "primary":
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
    selection = _apply_model_target_selection(
        cfg_data,
        current_model=str(current),
        new_model=str(set_model),
        target=model_target,
    )
    synced_routing = selection["routing_changed"]
    if not interactive or selection["target"] != "primary":
        chosen_thinking = cfg_data.get("thinking", "off")  # 非交互模式保持原值

    old_thinking = cfg_data.get("thinking", "off")
    cfg_data["thinking"] = chosen_thinking
    cfg_path.write_text(_json.dumps(cfg_data, ensure_ascii=False, indent=2), encoding="utf-8")
    if selection["target"] == "primary":
        # 同步 DB routing_overrides：避免重启后从 DB 恢复到旧模型
        _sync_db_routing_overrides(cfg_path, old_model=str(current), new_model=str(set_model))
        console.print(f"[green]✓ 模型已切换:[/green] {current} → [bold cyan]{set_model}[/bold cyan]")
    else:
        console.print(
            f"[green]✓ {selection['target']} 模型已更新:[/green]"
            f" {selection['previous'] or current} → [bold cyan]{set_model}[/bold cyan]"
        )
        runtime_tier = selection["runtime_override_tier"]
        if runtime_tier:
            _set_db_routing_override(cfg_path, tier=runtime_tier, model_ref=str(set_model))
        else:
            console.print(
                f"[yellow]⚠ routing 键 {selection['target']} 不是标准运行时 tier；仅已写入 config routing。[/yellow]"
            )
    if synced_routing and selection["target"] == "primary":
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
