"""memory/semantic.py — 语义记忆（SemanticMemory）。

双层存储设计：
    1. nodes/{id}.json  — 运行期语义节点源数据（首先写入，可重建）
  2. semantic.db      — 搜索索引层（由 json 派生，可完全重建，删除后无数据丢失）

恼复路径： semantic.db 损坏 → 启动时自动检测 → 删除并重建 → 从 nodes/*.json 重导入。
检索：FTS5 关键词评分（标准库，零依赖），接口稳定，内部可替换。

说明：这里的 nodes/ 位于配置的 memory_dir 下（默认 ~/.lingzhou/memory/nodes），
不是 workspace_dir，也不应该作为源码仓库的一部分提交。
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable

import logging as _log_sem
from .quality_checker import evaluate_retrieval_quality

_log = _log_sem.getLogger("lingzhou.memory.semantic")

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    activation  REAL NOT NULL DEFAULT 0.5,
    valence     REAL NOT NULL DEFAULT 0.5,
    tags        TEXT NOT NULL DEFAULT '[]',
    source      TEXT NOT NULL DEFAULT '',
    embedding   TEXT,
    created_at  TEXT NOT NULL
);
"""


def _parse_table_cols(ddl: str) -> dict[str, str]:
    """从 CREATE TABLE DDL 中提取列名 → 定义的映射，用于 schema reconciler。"""
    m = re.search(r'CREATE TABLE[^(]*\((.*?)\);', ddl, re.DOTALL | re.IGNORECASE)
    if not m:
        return {}
    cols: dict[str, str] = {}
    for line in m.group(1).split(','):
        line = line.strip()
        if not line:
            continue
        # 跳过表级约束（UNIQUE、CHECK、FOREIGN KEY 等；PRIMARY KEY 内联列上）
        if re.match(r'(PRIMARY\s+KEY|UNIQUE|CHECK|FOREIGN\s+KEY)\b', line, re.IGNORECASE):
            continue
        col_m = re.match(r'^(\w+)\s+(.*)', line)
        if col_m:
            cols[col_m.group(1)] = col_m.group(2).strip()
    return cols

# FTS5 虚拟表：用于 retrieve/retrieve_multi_anchor 的候选集预过滤（O(log n) 代替 O(n)）
# 独立表（非 content table），通过 _db_upsert 手动同步
_DDL_FTS5 = """
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    id UNINDEXED,
    title,
    body,
    tags,
    tokenize='unicode61'
);
"""


@dataclass
class MemoryNode:
    id: str
    kind: str
    title: str
    body: str
    activation: float = 0.5
    valence: float = 0.5
    tags: list[str] = field(default_factory=list[str])
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryNode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def effective_activation(node: "MemoryNode", decay_lambda: float) -> float:
    """Ebbinghaus forgetting curve: activation x exp(-lambda x days_since_created)."""
    if decay_lambda <= 0:
        return node.activation
    try:
        created = datetime.fromisoformat(node.created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        days = max(0.0, (datetime.now(UTC) - created).total_seconds() / 86400)
        return node.activation * math.exp(-decay_lambda * days)
    except Exception:
        return node.activation


_effective_activation = effective_activation


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticMemory:
    def __init__(
        self,
        memory_dir: Path,
        decay_lambda: float = 0.1,
        db_path: Path | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
        embedding_weight: float = 0.3,
    ) -> None:
        self._dir = memory_dir / "nodes"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._decay_lambda = decay_lambda
        self._db_path = db_path or (memory_dir / "semantic.db")
        self._fts5_ok: bool = False  # P1-A: FTS5 可用标志
        # 向量混合检索（实验性功能，opt-in）：
        #   默认不启用（embed_fn=None）。只有在 lingzhou.json 配置 embedding_model 后，
        #   loop.py 会构造 embed_fn 并传入。当前 embed 调用为同步：
        #   如果 embedding_model 已配置，每次 upsert 都会同步调用 LLM embed 接口。
        #   拳说：需要大量语义记忆写入时才建议开启；最终应改为异步批量嵌入。
        if embed_fn is not None:
            _log.info("[semantic] 向量混合检索已启用（实验性，embedding_weight=%.2f）", embedding_weight)
        self._embed_fn = embed_fn
        self._embedding_weight = embedding_weight
        self._conn = self._open_db()
        self._migrate()             # 迁移机制：幂等 ALTER TABLE，补齐新列
        # On startup: sync any json nodes missing from DB (idempotent)
        self._sync_from_files()

    # --- DB init & recovery ---------------------------------------------------

    def _open_db(self) -> sqlite3.Connection:
        """Open DB; auto-delete and recreate if corrupted (_sync_from_files will restore data)."""
        try:
            conn = self._connect()
            conn.executescript(_DDL)
            conn.commit()
            self._setup_fts5(conn)
            return conn
        except sqlite3.DatabaseError:
            # Corrupt DB file: delete and rebuild; _sync_from_files re-imports from json
            self._db_path.unlink(missing_ok=True)
            conn = self._connect()
            conn.executescript(_DDL)
            conn.commit()
            self._setup_fts5(conn)
            return conn

    def _migrate(self) -> None:
        """幂等 schema 迁移：以 _DDL 为权威 schema，ADD COLUMN 补齐老 DB 缺失列，永不 DROP。

        新增列只需在 _DDL 中声明，无需手动修改此方法。
        """
        try:
            desired = _parse_table_cols(_DDL)
            existing = {row[1] for row in self._conn.execute("PRAGMA table_info(nodes)")}
            changed = False
            for col, definition in desired.items():
                if col not in existing:
                    self._conn.execute(f"ALTER TABLE nodes ADD COLUMN {col} {definition}")
                    changed = True
            if changed:
                self._conn.commit()
        except Exception:
            pass  # 迁移失败静默跳过，不影响现有功能
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind)"
            )
            self._conn.commit()
        except Exception:
            pass

    def _setup_fts5(self, conn: sqlite3.Connection) -> None:
        """Create FTS5 table (if absent) and populate from existing nodes; set self._fts5_ok."""
        try:
            conn.executescript(_DDL_FTS5)
            # Populate FTS5 for nodes already in DB but missing from FTS5 (idempotent)
            conn.execute("""
                INSERT INTO nodes_fts(id, title, body, tags)
                SELECT id, title, body, tags FROM nodes
                WHERE id NOT IN (SELECT id FROM nodes_fts)
            """)
            conn.commit()
            self._fts5_ok = True
        except Exception as exc:
            _log.warning("[semantic] FTS5 初始化失败，降级为全表扫描：%s", exc)
            self._fts5_ok = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _sync_from_files(self) -> None:
        """Import nodes from nodes/*.json that are missing in DB (idempotent, runs at startup)."""
        try:
            existing_ids: set[str] = {
                row[0] for row in self._conn.execute("SELECT id FROM nodes")
            }
            for p in self._dir.glob("*.json"):
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    if d.get("id") not in existing_ids:
                        self._db_upsert(MemoryNode.from_dict(d))
                except Exception as exc:
                    _log.warning("[semantic] 跳过损坏的节点文件 %s: %s", p.name, exc)
            self._conn.commit()
        except Exception as exc:
            _log.warning("[semantic] _sync_from_files 失败，回退到文件扫描: %s", exc)

    def rebuild_index(self) -> None:
        """从 nodes/*.json 全量重建数据库索引（手动恢复；也可直接删除 semantic.db 触发自动重建）。"""
        self._conn.execute("DELETE FROM nodes")
        if self._fts5_ok:
            try:
                self._conn.execute("DELETE FROM nodes_fts")
            except Exception:
                pass
        self._conn.commit()
        for p in self._dir.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                self._db_upsert(MemoryNode.from_dict(d))
                # 保留 JSON 中已有的 embedding（避免重建时丢失已计算向量）
                emb = d.get("embedding")
                if emb is not None:
                    emb_json = json.dumps(emb) if not isinstance(emb, str) else emb
                    self._conn.execute(
                        "UPDATE nodes SET embedding = ? WHERE id = ?",
                        (emb_json, d.get("id")),
                    )
            except Exception:
                pass
        self._conn.commit()

    def _db_upsert(self, node: MemoryNode) -> None:
        tags_json = json.dumps(node.tags, ensure_ascii=False)
        # 取已有 embedding（若 embed_fn 可用则在 upsert() 上层写入）
        self._conn.execute(
            """INSERT OR REPLACE INTO nodes
               (id, kind, title, body, activation, valence, tags, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node.id, node.kind, node.title, node.body,
                node.activation, node.valence,
                tags_json,
                getattr(node, 'source', ''),
                node.created_at,
            ),
        )
        # P1-A: 同步 FTS5 索引（DELETE+INSERT 模式，保证幂等）
        if self._fts5_ok:
            try:
                self._conn.execute("DELETE FROM nodes_fts WHERE id = ?", (node.id,))
                self._conn.execute(
                    "INSERT INTO nodes_fts(id, title, body, tags) VALUES (?, ?, ?, ?)",
                    (node.id, node.title, node.body, tags_json),
                )
            except Exception:
                pass  # FTS5 失败不阻断主写入

    # --- Public interface (signatures identical to original) ------------------

    @property
    def fts5_ok(self) -> bool:
        """Whether the FTS5 index is available."""
        return self._fts5_ok

    @property
    def decay_lambda(self) -> float:
        """Decay coefficient used by Ebbinghaus-style activation scoring."""
        return self._decay_lambda

    def stats(self) -> dict[str, Any]:
        """Return lightweight health stats for prompt/context diagnostics."""
        total_nodes = 0
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
            total_nodes = int(row[0] or 0) if row else 0
        except Exception:
            total_nodes = 0
        return {
            "nodes": total_nodes,
            "fts5_ok": bool(self._fts5_ok),
            "decay_lambda": float(self._decay_lambda),
            "embedding_enabled": bool(self._embed_fn is not None),
            "db_path": str(self._db_path),
            "nodes_dir": str(self._dir),
        }

    def upsert(self, node: MemoryNode) -> None:
        """Write or overwrite a memory node. json written first (disaster recovery), then DB (search index)."""
        # 1. Disaster recovery layer: write json first (safe even if DB fails)
        path = self._dir / f"{node.id}.json"
        path.write_text(json.dumps(node.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        # 2. Search index layer: write to DB first so the row exists before embedding UPDATE
        try:
            self._db_upsert(node)
            self._conn.commit()
        except Exception:
            pass
        # 3. 可选：计算并存储 embedding（行已存在，UPDATE 必然生效）
        if self._embed_fn is not None:
            try:
                vec = self._embed_fn(node.title + " " + node.body)
                self._conn.execute(
                    "UPDATE nodes SET embedding = ? WHERE id = ?",
                    (json.dumps(vec), node.id),
                )
                self._conn.commit()
            except Exception:
                pass  # embedding 失败不阻断主写入

    def get(self, node_id: str) -> MemoryNode | None:
        # DB first (O(1) index); fall back to json if DB unavailable
        try:
            row = self._conn.execute(
                "SELECT * FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if row:
                return self._row_to_node(row)
        except Exception:
            pass
        path = self._dir / f"{node_id}.json"
        if path.exists():
            return MemoryNode.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        kind: str | None = None,
        tag: str | None = None,
        source: str | None = None,
        task_id: str | int | None = None,
        path_prefix: str | None = None,
        id_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        """单锤点语义检索：返回与查询最相关的 top_k 节点。

        FTS5 预过滤把候选集从 O(n) 压缩到 O(log n)；
        FTS5 不可用或无命中时降级为全扫描。
        embed_fn 可用时，计算 query_vec 混合 cosine 相似度评分。
        """
        query_vec: list[float] | None = None
        if self._embed_fn is not None:
            try:
                query_vec = self._embed_fn(query)
            except Exception:
                pass
        candidate_ids = self._fts_candidates(query, limit=100 if any((kind, tag, task_id, path_prefix, id_prefix)) else 50)
        nodes = self._load_by_ids(candidate_ids) if candidate_ids else self._load_all()
        if any((kind, tag, source, task_id, path_prefix, id_prefix)):
            nodes = [
                node for node in nodes
                if self._matches_filters(
                    node,
                    kind=kind,
                    tag=tag,
                    source=source,
                    task_id=task_id,
                    path_prefix=path_prefix,
                    id_prefix=id_prefix,
                )
            ]
            if candidate_ids and not nodes:
                nodes = [
                    node for node in self._load_all()
                    if self._matches_filters(
                        node,
                        kind=kind,
                        tag=tag,
                        source=source,
                        task_id=task_id,
                        path_prefix=path_prefix,
                        id_prefix=id_prefix,
                    )
                ]
        if not nodes:
            return []
        scored = [(self._score(query, n, query_vec=query_vec), n) for n in nodes]
        scored.sort(key=lambda x: x[0], reverse=True)
        retrieved = []
        for score, node in scored[:top_k]:
            item = node.to_dict()
            item["score"] = round(float(score), 4)
            retrieved.append(item)

        if _log.isEnabledFor(_log_sem.DEBUG):
            qm = evaluate_retrieval_quality(query, retrieved, self._decay_lambda)
            _log.debug("[semantic.retrieve] quality=%s", qm.get("overall_score", 0))
        return retrieved

    @staticmethod
    def _matches_filters(
        node: MemoryNode,
        *,
        kind: str | None = None,
        tag: str | None = None,
        source: str | None = None,
        task_id: str | int | None = None,
        path_prefix: str | None = None,
        id_prefix: str | None = None,
    ) -> bool:
        if kind and node.kind != str(kind).strip():
            return False
        if tag:
            normalized_tag = str(tag).strip()
            if normalized_tag and normalized_tag not in node.tags:
                return False
        if source:
            normalized_source = str(source).strip()
            node_source = str(getattr(node, 'source', '')).strip()
            if normalized_source and normalized_source != node_source:
                return False
        if task_id is not None:
            expected_task_tag = f"task:{str(task_id).strip()}"
            if expected_task_tag not in node.tags:
                return False
        if id_prefix:
            normalized_id_prefix = str(id_prefix).strip()
            if normalized_id_prefix and not node.id.startswith(normalized_id_prefix):
                return False
        if path_prefix:
            normalized_path = str(path_prefix).strip().replace("\\", "/")
            if normalized_path:
                haystack = [
                    node.title.replace("\\", "/"),
                    node.body.replace("\\", "/"),
                    *(tag_item.replace("\\", "/") for tag_item in node.tags),
                ]
                if not any(normalized_path in item for item in haystack):
                    return False
        return True

    def retrieve_multi_anchor(
        self, anchors: list[str], top_k: int = 5, convergence_bonus: float = 0.15, source: str | None = None
    ) -> list[dict[str, Any]]:
        """多锤点情境召回（Anderson 1983 ACT-R 收敛激活原理）。

        合并各锤点的 FTS5 候选集，去重后精排；
        多个镔点命中同一节点 → convergence_bonus 加分（越多镄点命中相关度越高）。
        """
        valid_anchors = [a for a in anchors if a and a.strip()]
        if not valid_anchors:
            return []
        # 合并各锚点 FTS5 候选（去重）
        all_ids: list[str] = []
        seen: set[str] = set()
        for anchor in valid_anchors:
            for nid in self._fts_candidates(anchor, limit=30):
                if nid not in seen:
                    seen.add(nid)
                    all_ids.append(nid)
        nodes = self._load_by_ids(all_ids) if all_ids else self._load_all()
        if source:
            nodes = [n for n in nodes if n.get("source") == source]
        if not nodes:
            return []

        best_score: dict[str, float] = {}
        hit_count: dict[str, int] = {}
        for anchor in valid_anchors:
            # 有 embed_fn 时计算锚点向量，传入 _score 启用混合评分（与 retrieve() 对齐）
            query_vec: list[float] | None = None
            if self._embed_fn is not None:
                try:
                    query_vec = self._embed_fn(anchor)
                except Exception:
                    pass
            for node in nodes:
                s = self._score(anchor, node, query_vec=query_vec)
                if s > 0:
                    if node.id not in best_score or s > best_score[node.id]:
                        best_score[node.id] = s
                    hit_count[node.id] = hit_count.get(node.id, 0) + 1

        if not best_score:
            return []

        node_map = {n.id: n for n in nodes}
        final: list[tuple[float, MemoryNode]] = []
        for nid, base in best_score.items():
            hits = hit_count.get(nid, 1)
            score = base * (1.0 + convergence_bonus * (hits - 1))
            final.append((score, node_map[nid]))

        final.sort(key=lambda x: x[0], reverse=True)
        retrieved = []
        for score, node in final[:top_k]:
            item = node.to_dict()
            item["score"] = round(float(score), 4)
            retrieved.append(item)

        if _log.isEnabledFor(_log_sem.DEBUG):
            combined_query = " ".join(valid_anchors)
            qm = evaluate_retrieval_quality(combined_query, retrieved, self._decay_lambda)
            _log.debug("[semantic.multi_anchor] quality=%s", qm.get("overall_score", 0))
        return retrieved

    def store_reflection(self, kind: str, insight: str, valence: float = 0.5) -> str:
        """Write a reflection insight as a learned_insight node; return node_id."""
        node_id = f"reflection-{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
        self.upsert(MemoryNode(
            id=node_id,
            kind="learned_insight",
            title=f"[{kind}]",
            body=insight.strip(),
            activation=0.8,
            valence=valence,
            tags=[kind],
        ))
        return node_id

    def list_reflections(self, limit: int = 10) -> list[MemoryNode]:
        """Return the most recent `limit` learned_insight nodes (newest first)."""
        nodes = [n for n in self._load_all() if n.kind == "learned_insight"]
        nodes.sort(key=lambda n: n.created_at, reverse=True)
        return nodes[:limit]

    # --- Internal helpers -----------------------------------------------------

    def _fts_candidates(self, query: str, limit: int = 50) -> list[str]:
        """用 FTS5 从 query 中提取 token，返回命中节点 ID 列表；不可用时返回空列表。"""
        if not self._fts5_ok:
            return []
        # 只保留 unicode 字词字符，去掉 FTS5 特殊符号
        safe = re.sub(r'[^\w\s]', ' ', query, flags=re.UNICODE)
        # ASCII 词 ≥5 字符（过滤 "core" "loop" 等常见短词，防止 OR 泛命中）；
        # 非 ASCII ≥2；若严格过滤后为空则回退原行为。
        _strict = [t for t in safe.split() if len(t) >= 2 and not (t.isascii() and len(t) < 5)]
        terms = _strict if _strict else [t for t in safe.split() if len(t) > 1]
        if not terms:
            return []
        fts_query = " OR ".join(terms)
        try:
            rows = self._conn.execute(
                "SELECT id FROM nodes_fts WHERE nodes_fts MATCH ? LIMIT ?",
                (fts_query, limit),
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    def _load_by_ids(self, ids: list[str]) -> list[MemoryNode]:
        """按 ID 列表从 DB 加载节点；DB 不可用时返回空列表（调用方可降级到 _load_all）。"""
        if not ids:
            return []
        try:
            placeholders = ",".join("?" * len(ids))
            rows = self._conn.execute(
                f"SELECT * FROM nodes WHERE id IN ({placeholders})", ids
            ).fetchall()
            return [self._row_to_node(r) for r in rows]
        except Exception:
            return []

    def _load_all(self) -> list[MemoryNode]:
        """Load all nodes from DB; fall back to nodes/*.json file scan if DB unavailable."""
        try:
            rows = self._conn.execute("SELECT * FROM nodes").fetchall()
            return [self._row_to_node(r) for r in rows]
        except Exception:
            pass
        # Fallback: DB unavailable, scan json files (full recovery)
        nodes: list[MemoryNode] = []
        for p in self._dir.glob("*.json"):
            try:
                nodes.append(MemoryNode.from_dict(json.loads(p.read_text(encoding="utf-8"))))
            except Exception:
                pass
        return nodes

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> MemoryNode:
        d: dict[str, Any] = dict(row)
        d["tags"] = json.loads(d.get("tags") or "[]")
        node = MemoryNode.from_dict(d)
        # embedding 不在 dataclass 字段内，附加为实例属性供 _score 使用
        emb = d.get("embedding")
        if emb is not None:
            node.__dict__["embedding"] = emb  # type: ignore[index]
        return node

    def _score(
        self,
        query: str,
        node: MemoryNode,
        query_vec: "list[float] | None" = None,
    ) -> float:
        """Keyword overlap score + Ebbinghaus-decayed activation weighting.

        如果 query_vec 可用且节点有 embedding，
        使用 cosine similarity 混合评分（embedding_weight 加权）。
        """
        eff_act = _effective_activation(node, self._decay_lambda)
        q_tokens = set(re.findall(r"\w+", query.lower()))
        n_tokens = set(re.findall(r"\w+", (node.title + " " + node.body).lower()))
        if not q_tokens or not n_tokens:
            kw_score = 0.1
        else:
            kw_score = len(q_tokens & n_tokens) / len(q_tokens | n_tokens)

        # 向量混合评分（embed_fn 已配置且节点有 embedding 时生效）
        node_emb_raw = getattr(node, "embedding", None)
        if query_vec is not None and node_emb_raw is not None:
            try:
                node_vec: list[float] = (
                    json.loads(node_emb_raw)
                    if isinstance(node_emb_raw, str)
                    else node_emb_raw
                )
                cos_sim = _cosine(query_vec, node_vec)
                w = self._embedding_weight
                text_score = kw_score * 0.7 + eff_act * 0.3
                return (1 - w) * text_score + w * cos_sim
            except Exception:
                pass

        return kw_score * 0.7 + eff_act * 0.3

    # ── 向量嵌入工具方法（供 CognitionLoop 批量异步计算）─────────────────────

    def get_unembedded(self, limit: int = 20) -> list[tuple[str, str]]:
        """返回尚未计算 embedding 的节点 (id, text)，供调用方批量嵌入后回填。

        text = title + ' ' + body，与 upsert() 中计算 embedding 的输入一致。
        """
        try:
            rows = self._conn.execute(
                "SELECT id, title, body FROM nodes WHERE embedding IS NULL LIMIT ?",
                (limit,),
            ).fetchall()
            return [(r[0], (r[1] or "") + " " + (r[2] or "")) for r in rows]
        except Exception:
            return []

    def set_embedding(self, node_id: str, vec: list[float]) -> None:
        """将外部计算好的向量写入指定节点的 embedding 列，同时同步 json 文件。"""
        try:
            vec_json = json.dumps(vec)
            self._conn.execute(
                "UPDATE nodes SET embedding = ? WHERE id = ?",
                (vec_json, node_id),
            )
            self._conn.commit()
            # 同步 json 文件（灾难恢复层）
            json_path = self._dir / f"{node_id}.json"
            if json_path.exists():
                try:
                    d = json.loads(json_path.read_text(encoding="utf-8"))
                    d["embedding"] = vec
                    json_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
        except Exception:
            pass
