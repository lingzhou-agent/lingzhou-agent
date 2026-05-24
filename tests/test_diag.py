"""doctor 命令回归测试。"""

import json
from pathlib import Path

from typer.testing import CliRunner


def _write_doctor_config(
    path: Path,
    *,
    api_key_env: str,
    auth_profile_id: str = "",
) -> None:
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "deepseek": {
                        "type": "openai_compat",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": api_key_env,
                        "auth_profile_id": auth_profile_id,
                    }
                },
                "model": "deepseek/deepseek-chat",
                "loop": {
                    "db_path": str(path.parent / "state" / "runtime.db"),
                    "memory_dir": str(path.parent / "memory"),
                    "state_dir": str(path.parent / "state"),
                    "workspace_dir": str(path.parent / "workspace"),
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class _DoctorProbeResponse:
    status_code = 200


class _DoctorProbeClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def post(self, *args, **kwargs):
        return _DoctorProbeResponse()


def test_dev_doctor_accepts_auth_profile_api_key(monkeypatch, tmp_path):
    import httpx
    import store.auth as auth_mod

    from cli.main import app
    from store.auth import set_token_profile

    auth_path = tmp_path / "auth-profiles.json"
    cfg_path = tmp_path / "lingzhou.json"
    _write_doctor_config(
        cfg_path,
        api_key_env="DEEPSEEK_API_KEY",
        auth_profile_id="deepseek:default",
    )
    set_token_profile(
        profile_id="deepseek:default",
        provider="deepseek",
        token="sk-profile-token-123456",
        path=auth_path,
    )

    monkeypatch.setattr(auth_mod, "AUTH_PROFILES_PATH", auth_path)
    monkeypatch.setattr(httpx, "Client", _DoctorProbeClient)

    runner = CliRunner()
    result = runner.invoke(app, ["dev", "doctor", "--config", str(cfg_path)])

    assert result.exit_code == 0
    assert "API key (DEEPSEEK_API_KEY): 来自 auth profile" in result.stdout
    assert "API key (DEEPSEEK_API_KEY): 未设置" not in result.stdout


def test_dev_doctor_accepts_literal_api_key(monkeypatch, tmp_path):
    import httpx

    from cli.main import app

    cfg_path = tmp_path / "lingzhou.json"
    _write_doctor_config(cfg_path, api_key_env="sk-direct-literal-key-123456")

    monkeypatch.setattr(httpx, "Client", _DoctorProbeClient)

    runner = CliRunner()
    result = runner.invoke(app, ["dev", "doctor", "--config", str(cfg_path)])

    assert result.exit_code == 0
    assert "API key (literal):" in result.stdout
    assert "API key (sk-direct-literal-key-123456): 未设置" not in result.stdout


def test_dev_doctor_db_schema_matches_current_runtime_tables(monkeypatch, tmp_path):
    import httpx
    import sqlite3

    from cli.main import app

    cfg_path = tmp_path / "lingzhou.json"
    db_path = tmp_path / "state" / "runtime.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _write_doctor_config(cfg_path, api_key_env="sk-direct-literal-key-123456")

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT);
            CREATE TABLE failures (id INTEGER PRIMARY KEY AUTOINCREMENT);
            CREATE TABLE facts (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE signals (id INTEGER PRIMARY KEY AUTOINCREMENT);
            CREATE TABLE chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT);
            CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT);
            CREATE TABLE meta_reflections (id TEXT PRIMARY KEY);
            """
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(httpx, "Client", _DoctorProbeClient)

    runner = CliRunner()
    result = runner.invoke(app, ["dev", "doctor", "--config", str(cfg_path)])

    assert result.exit_code == 0
    assert "DB schema: 缺少表" not in result.stdout
    assert "DB schema: 关键表均存在" in result.stdout