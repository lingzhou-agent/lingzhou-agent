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

auth_app = typer.Typer(name="auth", help="凭证授权管理")


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

    # ── 路径 1：gh CLI ──────────────────────────────────────────────────
    if method in ("auto", "gh"):
        console.print("\n[bold]尝试路径 1/2：gh CLI[/bold]")
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                token = result.stdout.strip()
                if token:
                    console.print("[green]✓ 通过 gh CLI 获取 GitHub token[/green]")
            elif method == "gh":
                console.print(f"[red]gh CLI 返回非 0：{result.returncode}[/red]")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            console.print("  gh CLI 未找到，跳过")

    # ── 可选路径：GitHub OAuth Device Flow（仅显式指定时使用） ───────────
    if not token and method == "device":
        console.print("\n[bold]可选路径：GitHub OAuth Device Flow[/bold]")
        client_id = oauth_client_id.strip() or _load_copilot_client_id(config)
        if not client_id:
            console.print(
                "  [yellow]当前构建还没有内置 Lingzhou 自己的 GitHub OAuth App client_id，"
                "因此 Device Flow 的交互链已就位，但还差 app registration 这一步。[/yellow]\n"
                f"  [dim]临时覆盖方式：\n"
                f"  1. --oauth-client-id Iv1.xxxx\n"
                f"  2. export LINGZHOU_GITHUB_CLIENT_ID=Iv1.xxxx\n"
                f"  3. 写入 {GITHUB_DEVICE_AUTH_PATH}：{{\"client_id\": \"Iv1.xxxx\"}}\n"
                f"  4. 兼容旧配置：providers.copilot.oauth_client_id[/dim]"
            )
        else:
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
