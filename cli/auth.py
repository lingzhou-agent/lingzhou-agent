"""cli/auth.py — auth 命令组（参考 OpenClaw 的 auth profile store）。"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.panel import Panel

from cli._common import console, load_cfg
from auth_store import (
    AUTH_PROFILES_PATH,
    COPILOT_PROFILE_ID,
    GITHUB_DEVICE_AUTH_PATH,
    LEGACY_CREDENTIALS_PATH,
    get_auth_profile,
    load_github_device_client_id,
    load_legacy_credentials,
    mask_secret,
    save_legacy_credentials,
    set_token_profile,
)

auth_app = typer.Typer(name="auth", help="凭证授权管理", context_settings={"help_option_names": ["-h", "--help"]})

# GitHub Copilot VS Code 插件的 OAuth App client_id（公开）
# 用该 client_id 做 device flow 得到的 token 才能访问 copilot_internal API
COPILOT_VSCODE_CLIENT_ID = "Iv1.b507a08c87ecfe98"


def _load_copilot_client_id(config: Path) -> str:
    client_id = load_github_device_client_id()
    if client_id:
        return client_id

    try:
        cfg = load_cfg(config)
        pdef = cfg.providers.get("copilot")
        if pdef:
            return getattr(pdef, "oauth_client_id", "") or ""
    except Exception:
        pass
    return ""


def _store_copilot_token(token: str) -> None:
    # Canonical store: OpenClaw 风格 auth profile store
    set_token_profile(profile_id=COPILOT_PROFILE_ID, provider="copilot", token=token)

    # Legacy compatibility: 保留 credentials.json 回退读取能力
    legacy = load_legacy_credentials()
    legacy["GITHUB_TOKEN"] = token
    save_legacy_credentials(legacy)


def _login_copilot_impl(
    config: Path,
    force: bool,
    method: Literal["auto", "gh", "device", "token"] = "auto",
    oauth_client_id: str = "",
) -> None:
    """交互式授权 GitHub Copilot（优先 GitHub token → Copilot token exchange）。"""
    import httpx

    existing = get_auth_profile(COPILOT_PROFILE_ID)
    if existing and not force:
        token = str(existing.get("token", "")).strip()
        if token:
            console.print("[yellow]已存在 Copilot 登录（使用 --force 重新授权）[/yellow]")
            console.print(f"  profile: [dim]{COPILOT_PROFILE_ID}[/dim]")
            console.print(f"  token:   [dim]{mask_secret(token)}[/dim]")
            raise typer.Exit(0)

    token: str = ""

    # ── 路径 1：gh CLI（仅 --method gh 显式指定时使用） ───────────────────────
    if not token and method == "gh":
        console.print("\n[bold]gh CLI[/bold]")
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                token = result.stdout.strip()
                if token:
                    console.print("[green]✓ 通过 gh CLI 获取 GitHub token[/green]")
            else:
                console.print(f"[red]gh CLI 返回非 0：{result.returncode}[/red]")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            console.print("  gh CLI 未找到")

    if not token and method in ("auto", "device"):
        client_id = oauth_client_id.strip() or _load_copilot_client_id(config) or COPILOT_VSCODE_CLIENT_ID
        console.print(f"\n[bold]路径 1/2：GitHub Device Flow[/bold]  [dim]client_id={client_id}[/dim]")
        try:
            resp = httpx.post(
                "https://github.com/login/device/code",
                headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
                content=f"client_id={client_id}&scope=read%3Auser",
                timeout=15.0,
            )
            resp.raise_for_status()
            dc = resp.json()
            if "error" in dc:
                raise RuntimeError(dc.get("error_description", dc["error"]))

            user_code = dc["user_code"]
            device_code = dc["device_code"]
            verification_uri = dc["verification_uri"]
            expires_in = int(dc["expires_in"])
            interval_s = max(5, int(dc.get("interval", 5)))

            console.print(Panel(
                f"[bold]访问以下网址并输入验证码：[/bold]\n\n"
                f"  网址: [link]{verification_uri}[/link]\n"
                f"  验证码: [bold yellow]{user_code}[/bold yellow]\n\n"
                f"  [dim]（{expires_in}s 内有效）[/dim]",
                border_style="cyan",
                title="GitHub Copilot 授权",
            ))

            try:
                import webbrowser
                webbrowser.open(verification_uri)
            except Exception:
                pass

            expires_at = time.time() + expires_in
            console.print("[dim]等待 GitHub 授权...[/dim]")
            while time.time() < expires_at:
                time.sleep(interval_s)
                poll = httpx.post(
                    "https://github.com/login/oauth/access_token",
                    headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
                    content=(
                        f"client_id={client_id}"
                        f"&device_code={device_code}"
                        f"&grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code"
                    ),
                    timeout=15.0,
                )
                poll.raise_for_status()
                pdata = poll.json()
                if "access_token" in pdata:
                    token = pdata["access_token"]
                    console.print("[green]✓ GitHub Device Flow 授权成功[/green]")
                    break
                err = pdata.get("error", "")
                if err == "authorization_pending":
                    console.print("  [dim]等待确认...[/dim]", end="\r")
                    continue
                if err == "slow_down":
                    interval_s += 5
                    continue
                if err in ("access_denied", "expired_token"):
                    console.print(f"[red]授权失败: {err}[/red]")
                    break
                console.print(f"[red]未知错误: {pdata}[/red]")
                break
            else:
                console.print("[yellow]授权超时，请重试[/yellow]")
        except Exception as exc:
            console.print(f"  Device Flow 失败: {exc}")

    # ── 路径 2：手动粘贴 GitHub token ─────────────────────────────────
    if not token and method in ("auto", "token"):
        console.print("\n[bold]路径 2/2：手动输入 GitHub token[/bold]")
        console.print(
            "  [dim]Lingzhou 的 Copilot 主链路是：GitHub token → Copilot token exchange → Copilot API。\n"
            "  优先使用 gh auth token、GitHub OAuth token，或其他可成功访问\n"
            "  https://api.github.com/copilot_internal/v2/token 的 GitHub token。[/dim]"
        )
        token = typer.prompt("  粘贴 GitHub token", hide_input=True).strip()

    if not token:
        console.print("[red]未获取到 token，授权失败[/red]")
        raise typer.Exit(1)

    # ── 立即验证：做一次 Copilot token exchange，确保 GitHub token 可用 ──
    console.print("\n[dim]正在验证 token（Copilot token exchange）…[/dim]")
    try:
        from provider.openai_compat import COPILOT_TOKEN_URL, _build_copilot_ide_headers

        # 先检查 token 有效性和 scope
        diag = httpx.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {token}", "Accept": "application/json"},
            timeout=10.0,
        )
        if diag.status_code != 200:
            console.print(f"[red]✗ GitHub token 无效（/user 返回 {diag.status_code}）[/red]")
            raise typer.Exit(1)
        scopes = diag.headers.get("X-OAuth-Scopes", "")
        login = diag.json().get("login", "?")
        console.print(f"  GitHub 用户: [bold]{login}[/bold]  scopes: [dim]{scopes or '(none)'}[/dim]")

        resp = httpx.get(
            COPILOT_TOKEN_URL,
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/json",
                **_build_copilot_ide_headers(include_api_version=False),
            },
            timeout=15.0,
        )
        if resp.status_code == 200:
            copilot_token = resp.json().get("token", "")
            if copilot_token:
                # 缓存 Copilot token（避免 provider 首次启动再 exchange 一次）
                from auth_store import save_copilot_token_cache
                expires_str = str(resp.json().get("expires_at", "")).strip()
                if expires_str and expires_str.isdigit():
                    exp_ms = int(expires_str) * 1000
                else:
                    import time as _time
                    exp_ms = int((_time.time() + 1800) * 1000)
                save_copilot_token_cache(copilot_token, expires_at_ms=exp_ms)
                console.print("[green]✓ Copilot token exchange 成功[/green]")
            else:
                console.print("[yellow]exchange 返回空 token，可能稍后重试[/yellow]")
        elif resp.status_code in (401, 403):
            console.print(
                f"[red]✗ token 验证失败（{resp.status_code}）：GitHub token 权限不足。[/red]\n"
                "  请确认此 GitHub 账户已订阅 GitHub Copilot（Individual / Business）。\n"
                "  也可以尝试：[bold]gh auth refresh --scopes copilot[/bold]"
            )
            raise typer.Exit(1)
        elif resp.status_code == 404:
            console.print(
                f"[red]✗ token 验证失败（404）[/red]  body: [dim]{resp.text[:200]}[/dim]\n"
                "  GitHub 账户可能未开通 Copilot 订阅，或 token 缺少权限。\n"
                "  请登录 https://github.com/settings/copilot 确认订阅状态。"
            )
            raise typer.Exit(1)
        else:
            console.print(f"[yellow]exchange 返回异常状态 {resp.status_code}，请确认 Copilot 订阅状态[/yellow]")
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[yellow]验证请求失败（{exc}），仍将保存 token，运行时可能报错[/yellow]")

    _store_copilot_token(token)
    console.print(
        f"\n[green]✓ Copilot 登录信息已保存[/green]\n"
        f"  auth profiles: [dim]{AUTH_PROFILES_PATH}[/dim]\n"
        f"  legacy compat: [dim]{LEGACY_CREDENTIALS_PATH}[/dim]\n"
        f"  profile:       [dim]{COPILOT_PROFILE_ID}[/dim]\n"
        f"  token:         [dim]{mask_secret(token)}[/dim]"
    )


@auth_app.command("login-copilot")
def auth_login_copilot(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("lingzhou.json"),
    force: Annotated[bool, typer.Option("--force/--no-force", help="已有 token 时强制重新授权")] = False,
    method: Annotated[
        Literal["auto", "gh", "device", "token"],
        typer.Option("--method", help="授权方式：auto | gh | device | token"),
    ] = "auto",
    oauth_client_id: Annotated[
        str,
        typer.Option("--oauth-client-id", help="GitHub OAuth App Client ID（仅 --method device 时使用）"),
    ] = "",
) -> None:
    """专用 Copilot 登录命令（默认走 GitHub token → Copilot token exchange）。"""
    _login_copilot_impl(config, force, method=method, oauth_client_id=oauth_client_id)


@auth_app.command("copilot")
def auth_copilot(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("lingzhou.json"),
    force: Annotated[bool, typer.Option("--force/--no-force", help="已有 token 时强制重新授权")] = False,
    method: Annotated[
        Literal["auto", "gh", "device", "token"],
        typer.Option("--method", help="授权方式：auto | gh | device | token"),
    ] = "auto",
    oauth_client_id: Annotated[
        str,
        typer.Option("--oauth-client-id", help="GitHub OAuth App Client ID（仅 --method device 时使用）"),
    ] = "",
) -> None:
    """Copilot 登录的兼容别名。"""
    _login_copilot_impl(config, force, method=method, oauth_client_id=oauth_client_id)


@auth_app.command("set-token")
def auth_set_token(
    provider: Annotated[str, typer.Option("--provider", "-p", help="provider 名称，如 bailian / copilot")],
    token: Annotated[str, typer.Option("--token", help="访问 token", prompt=True, hide_input=True)],
    env_name: Annotated[str, typer.Option("--env", help="写入 credentials.json 的键名；空则按 provider 自动推断")] = "",
    profile_id: Annotated[str, typer.Option("--profile-id", help="auth profile id；空则自动使用 <provider>:default")] = "",
) -> None:
    """写入通用 provider token（auth-profiles + legacy credentials）。

    适用于 bailian/openai_compat 等 provider。
    """
    provider_name = provider.strip().lower()
    if not provider_name:
        console.print("[red]provider 不能为空[/red]")
        raise typer.Exit(1)

    profile = profile_id.strip() or (
        COPILOT_PROFILE_ID if provider_name == "copilot" else f"{provider_name}:default"
    )
    env_key = env_name.strip()
    if not env_key:
        env_key = {
            "bailian": "DASHSCOPE_API_KEY",
            "copilot": "GITHUB_TOKEN",
        }.get(provider_name, f"{provider_name.upper()}_API_KEY")

    set_token_profile(profile_id=profile, provider=provider_name, token=token)

    legacy = load_legacy_credentials()
    legacy[env_key] = token
    save_legacy_credentials(legacy)

    console.print(
        f"[green]✓ token 已保存[/green]\n"
        f"  provider: [dim]{provider_name}[/dim]\n"
        f"  profile:  [dim]{profile}[/dim]\n"
        f"  env key:  [dim]{env_key}[/dim]\n"
        f"  auth:     [dim]{AUTH_PROFILES_PATH}[/dim]\n"
        f"  legacy:   [dim]{LEGACY_CREDENTIALS_PATH}[/dim]\n"
        "  [dim]提示：若要优先使用 auth profile，请在对应 provider 配置中设置 auth_profile_id。[/dim]"
    )


@auth_app.command("bailian")
def auth_bailian(
    token: Annotated[str, typer.Option("--token", help="百炼 API Key", prompt=True, hide_input=True)],
    profile_id: Annotated[str, typer.Option("--profile-id", help="auth profile id，默认 bailian:default")] = "bailian:default",
    env_name: Annotated[str, typer.Option("--env", help="写入 credentials.json 的键名，默认 DASHSCOPE_API_KEY")] = "DASHSCOPE_API_KEY",
) -> None:
    """快捷配置百炼 token：等价于 lingzhou auth set-token --provider bailian。"""
    auth_set_token(provider="bailian", token=token, env_name=env_name, profile_id=profile_id)
