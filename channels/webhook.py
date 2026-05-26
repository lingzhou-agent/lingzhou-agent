from __future__ import annotations

import json as _json
import logging
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from store.task.ingress import IngressStore

log = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765


@dataclass
class WebhookConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    secret: str = ""


class WebhookChannel:
    def __init__(self, cfg: WebhookConfig, db_path: str | Path) -> None:
        self._cfg = cfg
        self._ingress = IngressStore(db_path)
        self._server: HTTPServer | None = None

    def start(self) -> None:
        cfg = self._cfg
        ingress = self._ingress

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

            def do_POST(self) -> None:
                if self.path != "/message":
                    self.send_response(404)
                    self.end_headers()
                    return
                if cfg.secret and self.headers.get("Authorization", "") != f"Bearer {cfg.secret}":
                    self.send_response(401)
                    self.end_headers()
                    return
                try:
                    length = min(int(self.headers.get("Content-Length", 0)), 65536)
                except (ValueError, TypeError):
                    self.send_response(400)
                    self.end_headers()
                    return
                body = self.rfile.read(length)
                try:
                    payload = _json.loads(body)
                    msg = str(payload.get("message", "")).strip()
                    priority = str(payload.get("priority", "high"))
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                if not msg:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error":"empty message"}')
                    return

                short = msg.replace("\n", " ")[:28] + ("..." if len(msg) > 28 else "")
                try:
                    task_id = _enqueue_webhook_task(ingress, msg, priority)
                    resp = _json.dumps({"ok": True, "task_id": task_id}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(resp)
                    log.info("[webhook] 注入任务 id=%d: %s", task_id, short)
                except Exception as exc:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(_json.dumps({"error": str(exc)}).encode())

        self._server = HTTPServer((cfg.host, cfg.port), _Handler)
        threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="webhook-gateway",
        ).start()
        log.info("[webhook] 通道已启动 host=%s port=%d", cfg.host, cfg.port)

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        log.info("[webhook] 通道已停止")


def describe_webhook_channel(wc_cfg: dict[str, Any]) -> str:
    host = wc_cfg.get("host", DEFAULT_HOST)
    port = int(wc_cfg.get("port", DEFAULT_PORT))
    secret = wc_cfg.get("secret")
    return (
        f"Webhook 监听: http://{host}:{port}/message"
        f"{'  (Bearer token)' if secret else '  (无鉴权)'}"
    )


def _enqueue_webhook_task(ingress: IngressStore, message: str, priority: str) -> int:
    short = message.replace("\n", " ")[:28] + ("..." if len(message) > 28 else "")
    return ingress.add_task(
        f"webhook: {short}",
        goal=message,
        priority=priority,
        source="gateway:webhook",
    )


def start_webhook_channel(wc_cfg: dict[str, Any], db_path: str | Path) -> WebhookChannel:
    config = WebhookConfig(
        host=str(wc_cfg.get("host", DEFAULT_HOST)),
        port=int(wc_cfg.get("port", DEFAULT_PORT)),
        secret=str(wc_cfg.get("secret", "") or ""),
    )
    channel = WebhookChannel(config, db_path)
    channel.start()
    return channel