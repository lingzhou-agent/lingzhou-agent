"""core/wechat_channel.py — lingzhou 独立的微信 bot 通道 (iLink long-poll)。

独立于 OpenClaw 和 hermes，lingzhou 自己的微信入口。
工作方式：
  - iLink long-poll 拉取消息 → 写入 chat_messages 表（user/pending）
  - Loop 自动消费 pending chat message → 产生 assistant 回复
  - 回复监控轮询 chat_messages → 通过 iLink sendMessage 发送
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests  # type: ignore[import]

log = logging.getLogger("lingzhou.wechat")

# ── iLink 常量 ──────────────────────────────────────────────────────────────
T = 1
VO = 3
ILINK_VER = "2.1.7"
ILINK_CV = "65547"
DEFAULT_POLL_SEC = 35
DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
DEFAULT_REPLY_POLL = 3
MAX_REPLY_RETRIES = 3


@dataclass
class WechatConfig:
    base_url: str = DEFAULT_BASE_URL
    token: str = ""
    poll_sec: int = DEFAULT_POLL_SEC
    reply_poll_sec: int = DEFAULT_REPLY_POLL


# ── iLink HTTP helpers ─────────────────────────────────────────────────────


def _hdrs(token: str, body: str = "") -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "Content-Length": str(len(body.encode())),
        "iLink-App-Id": "",
        "iLink-App-ClientVersion": ILINK_CV,
        "Authorization": "Bearer " + token if token else "",
    }


def _ilink_post(base_url: str, ep: str, bd: dict, token: str, timeout: int = 30) -> dict:
    url = base_url.rstrip("/") + "/" + ep.lstrip("/")
    bs = json.dumps(bd)
    r = requests.post(url, headers=_hdrs(token, bs), data=bs.encode(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_updates(base_url: str, token: str, buf: str = "", timeout: int | None = None) -> dict:
    if timeout is None:
        timeout = DEFAULT_POLL_SEC
    try:
        return _ilink_post(
            base_url,
            "ilink/bot/getupdates",
            {"get_updates_buf": buf, "base_info": {"channel_version": ILINK_VER}},
            token,
            timeout + 5,
        )
    except requests.exceptions.Timeout:
        return {"ret": 0, "msgs": [], "get_updates_buf": buf}
    except Exception as e:
        log.warning("getUpdates error: %s", e)
        return {"ret": -1, "msgs": [], "get_updates_buf": buf}


def send_text(base_url: str, token: str, to_user: str, text: str, ctx: str | None = None) -> dict:
    m = {
        "from_user_id": "",
        "to_user_id": to_user,
        "client_id": "lingzhou-" + secrets.token_hex(8),
        "message_type": 2,
        "message_state": 2,
        "item_list": [{"type": T, "text_item": {"text": text}}],
    }
    if ctx:
        m["context_token"] = ctx
    return _ilink_post(
        base_url,
        "ilink/bot/sendmessage",
        {"msg": m, "base_info": {"channel_version": ILINK_VER}},
        token,
    )


# ── 消息提取 ────────────────────────────────────────────────────────────────


def extract_text(items: list[dict]) -> str:
    parts = []
    for it in items:
        tp = it.get("type", 0)
        if tp == T:
            x = it.get("text_item", {}).get("text", "")
            if x:
                parts.append(x)
        elif tp == VO:
            x = it.get("voice_item", {}).get("text", "")
            if x:
                parts.append(f'[语音消息: "{x}"]')
        elif tp == 2:
            img = it.get("image_item", {})
            aeskey = img.get("aeskey", "")
            media = img.get("media", {})
            encrypt_param = media.get("encrypt_query_param", "")
            full_url = media.get("full_url", "")
            if aeskey and (encrypt_param or full_url):
                import json as _json
                img_data = _json.dumps({"aeskey": aeskey, "encrypt_query_param": encrypt_param, "full_url": full_url})
                parts.append(f"[图片消息] {img_data}")
            elif full_url:
                parts.append(f"[图片消息] {full_url}")
            else:
                parts.append("[图片消息]（无 URL）")
    return "\n".join(parts).strip()


# ── WechatChannel ──────────────────────────────────────────────────────────


class WechatChannel:
    """lingzhou 微信通道 — daemon 线程，与主 loop 并行运行。

    poll_loop:    iLink long-poll → chat_messages (user/pending)
    reply_loop:   轮询 chat_messages → iLink sendMessage
    """

    def __init__(self, wc_cfg: WechatConfig, db_path: str):
        self._cfg = wc_cfg
        self._db_path = db_path
        self._stop = threading.Event()
        self._replied: set[int] = set()  # 已回复的 chat_message id
        self._user_msg_ids: dict[str, int] = {}  # from_user → 最近的 user msg id
        self._conn = None  # shared sqlite connection

    def _get_db(self):
        """Shared connection, thread-safe."""
        import sqlite3
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        return self._conn

    # ── poll: iLink → chat_messages ─────────────────────────────────────

    def run_poll(self) -> None:
        log.info("[wechat] poll 启动 base_url=%s", self._cfg.base_url)
        buf = ""
        fails = 0

        while not self._stop.is_set():
            try:
                resp = get_updates(self._cfg.base_url, self._cfg.token, buf, self._cfg.poll_sec)
            except Exception as e:
                log.error("[wechat] getUpdates 异常: %s", e)
                fails += 1
                time.sleep(min(2 ** fails, 30))
                continue

            fails = 0
            if resp.get("get_updates_buf"):
                buf = resp["get_updates_buf"]

            for msg in resp.get("msgs", []):
                try:
                    self._handle_inbound(msg)
                except Exception as e:
                    log.error("[wechat] 处理入站消息异常: %s", e, exc_info=True)

    def _handle_inbound(self, msg: dict) -> None:
        from_user = msg.get("from_user_id", "")
        items = msg.get("item_list", [])
        items = self._download_images(items, from_user)
        text = extract_text(items)
        if not text:
            return

        ctx_token = msg.get("context_token", "")
        short = text.replace("\n", " ")[:50]

        import sqlite3
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO chat_messages (role, content, session_id, status, created_at) "
                "VALUES (?,?,?,?,datetime('now'))",
                ("user", text, f"wechat:{from_user}", "pending"),
            )
            conn.commit()
            msg_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            # 记录 context_token 到 meta
            if ctx_token:
                conn.execute(
                    "INSERT OR REPLACE INTO facts (key, value) VALUES (?,?)",
                    (f"wechat:ctx:{from_user}", ctx_token),
                )
                conn.commit()
            self._user_msg_ids[from_user] = msg_id
            log.info("[wechat] chat_msg id=%d from=%s: %s", msg_id, from_user[:16], short)
        finally:
            conn.close()

    # ── reply: chat_messages → iLink ────────────────────────────────────

    def run_reply(self) -> None:
        log.info("[wechat] reply 监控启动 interval=%ds", self._cfg.reply_poll_sec)

        while not self._stop.is_set():
            try:
                self._check_and_reply()
            except Exception as e:
                log.error("[wechat] reply 检查异常: %s", e, exc_info=True)
            self._stop.wait(self._cfg.reply_poll_sec)

    def _check_and_reply(self) -> None:
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            # 查找 wechat session 中未回复的 assistant 消息
            rows = conn.execute(
                "SELECT id, content, session_id, created_at FROM chat_messages "
                "WHERE role = 'assistant' AND session_id LIKE 'wechat:%' "
                "AND status IN ('pending', 'processed') "
                "ORDER BY id DESC LIMIT 20"
            ).fetchall()
        finally:
            conn.close()

        for row in rows:
            mid = row["id"]
            if mid in self._replied:
                continue

            content = row["content"]
            session_id = row["session_id"] or ""
            from_user = session_id.replace("wechat:", "", 1)

            if not from_user or not content:
                continue

            # 获取 context_token
            ctx_token = self._get_ctx_token(from_user)

            log.info("[wechat] → iLink msg=%d to=%s len=%d", mid, from_user[:16], len(content))

            for attempt in range(MAX_REPLY_RETRIES):
                try:
                    send_text(self._cfg.base_url, self._cfg.token, from_user, content, ctx_token)
                    self._replied.add(mid)
                    # 标记为已送达
                    self._mark_delivered(mid)
                    log.info("[wechat] 回复成功 msg=%d", mid)
                    break
                except Exception as e:
                    log.warning("[wechat] 回复失败 msg=%d attempt=%d: %s", mid, attempt + 1, e)
                    time.sleep(1)

    def _get_ctx_token(self, from_user: str) -> str:
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT value FROM facts WHERE key = ?",
                (f"wechat:ctx:{from_user}",),
            ).fetchone()
            return row[0] if row else ""
        finally:
            conn.close()

    def _mark_delivered(self, msg_id: int) -> None:
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "UPDATE chat_messages SET status = 'delivered' WHERE id = ?",
                (msg_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def _download_images(self, items: list[dict], from_user: str) -> list[dict]:
        """Download and decrypt iLink images (AES-ECB)."""
        import base64, hashlib
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes as cipher_modes  # type: ignore[import-untyped]
        except ImportError:
            return items
        
        img_dir = Path.home() / ".lingzhou" / "wechat_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        new_items = []
        for it in items:
            tp = it.get("type", 0)
            if tp != 2:
                new_items.append(it)
                continue
            img = it.get("image_item", {})
            aeskey_hex = img.get("aeskey", "")
            media = img.get("media", {})
            encrypt_param = media.get("encrypt_query_param", "")
            full_url = media.get("full_url", "")
            if not aeskey_hex or (not encrypt_param and not full_url):
                new_items.append(it)
                continue
            try:
                aes_key = bytes.fromhex(aeskey_hex)
                cipher = Cipher(algorithms.AES(aes_key[:16]), cipher_modes.ECB())
                decryptor = cipher.decryptor()
                if encrypt_param:
                    from urllib.parse import quote
                    cdn_url = f"https://novac2c.cdn.weixin.qq.com/c2c/download?encrypted_query_param={quote(encrypt_param, safe='')}"
                else:
                    cdn_url = full_url
                resp = requests.get(cdn_url, timeout=30)
                if resp.status_code != 200:
                    new_items.append(it)
                    continue
                decrypted = decryptor.update(resp.content) + decryptor.finalize()
                if decrypted:
                    pad_len = decrypted[-1]
                    if 1 <= pad_len <= 16:
                        decrypted = decrypted[:-pad_len]
                fhash = hashlib.md5(decrypted).hexdigest()[:12]
                ext = ".jpg"
                if decrypted[:4] == b"\x89PNG": ext = ".png"
                elif decrypted[:4] == b"GIF8": ext = ".gif"
                fname = img_dir / f"{from_user[:16]}_{fhash}{ext}"
                fname.write_bytes(decrypted)
                new_items.append({"type": 1, "text_item": {"text": f"[图片消息，已保存] {fname} ({len(decrypted)} bytes)"}})
                log.info("[wechat] image downloaded: %s (%d bytes)", fname, len(decrypted))
            except Exception as e:
                log.error("[wechat] image download failed: %s", e)
                new_items.append(it)
        return new_items

    def start(self) -> None:
        t1 = threading.Thread(target=self.run_poll, daemon=True, name="wechat-poll")
        t1.start()
        t2 = threading.Thread(target=self.run_reply, daemon=True, name="wechat-reply")
        t2.start()
        log.info("[wechat] 通道已启动")

    def stop(self) -> None:
        self._stop.set()
        log.info("[wechat] 通道已停止")


def start_wechat_channel(wc_cfg: dict, db_path: str) -> WechatChannel:
    config = WechatConfig(
        base_url=wc_cfg.get("base_url", DEFAULT_BASE_URL),
        token=wc_cfg.get("token", ""),
        poll_sec=int(wc_cfg.get("poll_sec", DEFAULT_POLL_SEC)),
        reply_poll_sec=int(wc_cfg.get("reply_poll_sec", DEFAULT_REPLY_POLL)),
    )
    channel = WechatChannel(config, db_path)
    channel.start()
    return channel
