"""跨任务记忆：SQLite 事务 + FTS5/BM25 + float32 向量 + RRF/MMR。

向量搜索目前为精确扫描，适合原型规模；不声称已实现 ANN 索引。
curated 表示经过记忆整理流程选入，不能作为外部事实核验标记。
"""

import hashlib
import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def terms(text: str) -> list[str]:
    """英文词 + 中文字/双字片段，不依赖系统分词器。"""
    result = re.findall(r"[a-z0-9_]+", text.lower())
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        result.extend(segment)
        result.extend(segment[i:i + 2] for i in range(len(segment) - 1))
    return list(dict.fromkeys(result))


def vector(values) -> np.ndarray:
    arr = np.asarray(values, dtype="<f4")
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError("记忆向量必须是一维有限数值数组")
    return arr


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or a.shape != b.shape:
        return 0.0
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm else 0.0


class SQLiteMemoryStore:
    def __init__(self, path: Path, namespace: str, session_id: str,
                 embedding_model: str, search_limit: int = 5):
        if not namespace.strip():
            raise ValueError("记忆 namespace 不能为空")
        if search_limit <= 0:
            raise ValueError("search_limit 必须为正数")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.session_id = session_id
        self.embedding_model = embedding_model
        self.search_limit = search_limit
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS memory (
                    id TEXT PRIMARY KEY, namespace TEXT NOT NULL,
                    session_id TEXT NOT NULL, trace_id TEXT NOT NULL,
                    source_agent TEXT NOT NULL, created_at REAL NOT NULL,
                    task_topic TEXT NOT NULL, summary TEXT NOT NULL,
                    content TEXT NOT NULL, tags TEXT NOT NULL,
                    kind TEXT NOT NULL, status TEXT NOT NULL,
                    source_ids TEXT NOT NULL, expires_at REAL,
                    embedding_model TEXT NOT NULL, dim INTEGER NOT NULL,
                    embedding BLOB NOT NULL, content_hash TEXT NOT NULL,
                    UNIQUE(namespace, content_hash)
                );
                CREATE INDEX IF NOT EXISTS memory_scope ON memory(namespace, status, trace_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(id UNINDEXED, words);
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    model TEXT NOT NULL, hash TEXT NOT NULL, dim INTEGER NOT NULL,
                    embedding BLOB NOT NULL, PRIMARY KEY(model, hash)
                );
                CREATE TABLE IF NOT EXISTS memory_revision (
                    old_id TEXT NOT NULL, new_id TEXT NOT NULL,
                    changed_at REAL NOT NULL, PRIMARY KEY(old_id, new_id)
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, *, role: str, topic: str, summary: str, content: dict,
            embedding, trace_id: str, kind: str = "episode", status: str = "pending",
            tags=(), source_ids=(), expires_at: float | None = None) -> str:
        if status not in {"pending", "curated", "quarantined"}:
            raise ValueError("未知记忆状态")
        arr = vector(embedding)
        # 原始轨迹按任务保留；整理后的相同内容跨任务去重。
        digest = hashlib.sha256(compact([
            self.embedding_model, status, kind, content,
            trace_id if status != "curated" else "",
        ]).encode()).hexdigest()
        ident = "mem-" + uuid.uuid4().hex
        with self.connect() as db:
            existing = db.execute(
                "SELECT id FROM memory WHERE namespace=? AND content_hash=?",
                (self.namespace, digest),
            ).fetchone()
            if existing:
                # 再次产生相同结论不自动刷新过期时间，须显式重新核验/建新版本。
                return existing["id"]
            inserted = db.execute("INSERT OR IGNORE INTO memory VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                ident, self.namespace, self.session_id, trace_id, role, time.time(),
                topic, summary, compact(content), compact(list(tags)), kind, status,
                compact(list(source_ids)), expires_at, self.embedding_model, arr.size,
                arr.tobytes(), digest,
            ))
            if not inserted.rowcount:
                return db.execute("SELECT id FROM memory WHERE namespace=? AND content_hash=?",
                                  (self.namespace, digest)).fetchone()["id"]
            words = " ".join(terms(topic + " " + summary + " " + compact(content)))
            db.execute("INSERT INTO memory_fts(id, words) VALUES (?,?)", (ident, words))
        return ident

    def add(self, role, content, sentence_vec, *, trace_id="", task_topic="",
            tags=(), **kwargs):
        """与旧存储写入接口兼容；原始模型输出不会直接进入跨任务检索。"""
        if kwargs.get("memory_scope"):
            content = {**content, "_memory_scope": dict(kwargs["memory_scope"])}
        return self.put(role=role, topic=task_topic, summary=compact(content)[:400],
                        content=content, embedding=sentence_vec, trace_id=trace_id,
                        status="quarantined" if content.get("_fallback") else "pending",
                        tags=tags)

    @staticmethod
    def unpack(row):
        item = dict(row)
        for key in ("content", "tags", "source_ids"):
            item[key] = json.loads(item[key])
        item["vector"] = np.frombuffer(item.pop("embedding"), dtype="<f4")
        item["role"] = item["source_agent"]
        return item

    def _visible(self, trace_id):
        return ("namespace=? AND (expires_at IS NULL OR expires_at>?) "
                "AND (status='curated' OR (status='pending' AND trace_id=?))",
                [self.namespace, time.time(), trace_id])

    def trace_records(self, trace_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM memory WHERE namespace=? AND trace_id=? "
                              "AND status='pending' ORDER BY created_at, id",
                              (self.namespace, trace_id)).fetchall()
        return [self.unpack(row) for row in rows]

    def get(self, ident: str, trace_id: str = "") -> dict | None:
        clause, params = self._visible(trace_id)
        with self.connect() as db:
            row = db.execute(f"SELECT * FROM memory WHERE {clause} AND id=?",
                             [*params, ident]).fetchone()
        return self.unpack(row) if row else None

    def provenance(self, ident: str, trace_id: str = "") -> list[dict]:
        """只展开一个可见记忆的直接来源，不绕过 namespace 边界。"""
        item = self.get(ident, trace_id)
        if item is None:
            return []
        result = []
        with self.connect() as db:
            for source_id in item["source_ids"][:8]:
                row = db.execute("SELECT id, source_agent, trace_id, content, status FROM memory "
                                 "WHERE namespace=? AND id=?", (self.namespace, source_id)).fetchone()
                if row:
                    source = dict(row)
                    source["content"] = json.loads(source["content"])
                    result.append(source)
        return result

    def supersede(self, old_id: str, new_id: str) -> None:
        """显式用已整理的新版本替代旧版本；不会根据相似度自动推断冲突。"""
        if old_id == new_id:
            raise ValueError("新旧版本不能是同一条记忆")
        with self.connect() as db:
            rows = db.execute("SELECT id, status FROM memory WHERE namespace=? AND id IN (?,?)",
                              (self.namespace, old_id, new_id)).fetchall()
            states = {row["id"]: row["status"] for row in rows}
            if len(states) != 2 or states.get(new_id) != "curated":
                raise ValueError("新旧记忆必须属于当前 namespace，且新版本已经整理")
            db.execute("UPDATE memory SET status='quarantined' WHERE id=?", (old_id,))
            db.execute("INSERT OR IGNORE INTO memory_revision VALUES (?,?,?)", (old_id, new_id, time.time()))

    def search(self, query: str, query_vectors: list[np.ndarray], *, trace_id: str,
               role: str, limit: int | None = None, strategy: str = "hybrid",
               tags: list[str] | None = None, min_similarity: float = 0.35,
               historical_only: bool = False, context: dict | None = None) -> list[dict]:
        """候选召回 -> RRF 融合 -> 角色偏好 -> MMR 去冗余。

        query_vectors 由调用者直接传入，检索过程不会重新调用 embedding 模型。
        """
        limit = self.search_limit if limit is None else limit
        if limit <= 0:
            return []
        if strategy not in {"hybrid", "dense", "keyword", "tag"}:
            raise ValueError("未知检索策略")
        queries = [vector(q) for q in query_vectors if len(q)]
        clause, params = self._visible(trace_id)
        if historical_only:
            clause += " AND status='curated' AND trace_id<>?"
            params.append(trace_id)
        with self.connect() as db:
            rows = db.execute(f"SELECT * FROM memory WHERE {clause} ORDER BY created_at, id", params).fetchall()
            pool = {row["id"]: self.unpack(row) for row in rows}
            if context:
                pool = {k: v for k, v in pool.items() if self.scope_compatible(v, context)}
            if tags:
                pool = {k: v for k, v in pool.items() if set(tags) & set(v["tags"])}
            rankings = []
            if strategy in {"hybrid", "keyword"}:
                tokens = terms(query)[:80]
                if tokens:
                    match = " OR ".join('"' + t + '"' for t in tokens)
                    # 先在 SQL 内过滤 namespace/可见性，再限制候选数。
                    lexical = db.execute(
                        f"SELECT memory.id FROM memory_fts JOIN memory ON memory.id=memory_fts.id "
                        f"WHERE memory_fts MATCH ? AND {clause} "
                        "ORDER BY bm25(memory_fts), memory.id LIMIT 80", [match, *params],
                    ).fetchall()
                    rankings.append([r["id"] for r in lexical if r["id"] in pool])
        similarities = {}
        if strategy in {"hybrid", "dense"}:
            for q in queries:
                scored = []
                for ident, item in pool.items():
                    if item["embedding_model"] != self.embedding_model or item["dim"] != q.size:
                        continue
                    score = cosine(q, item["vector"])
                    similarities[ident] = max(similarities.get(ident, -1.0), score)
                    if score >= min_similarity:
                        scored.append((score, ident))
                rankings.append([ident for _, ident in sorted(scored, key=lambda x: (-x[0], x[1]))[:80]])
        if strategy == "tag":
            rankings.append(list(pool) if tags else [])
        if strategy == "dense":
            ordered = sorted(similarities, key=lambda k: (-similarities[k], k))
            result = []
            for ident in ordered:
                if similarities[ident] < min_similarity:
                    continue
                pool[ident]["score"] = similarities[ident]
                pool[ident]["similarity"] = similarities[ident]
                result.append(pool[ident])
            return result[:limit]
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, ident in enumerate(ranking, 1):
                fused[ident] = fused.get(ident, 0) + 1 / (60 + rank)
        preferred = {"任务分配": {"strategy", "conclusion"}, "资料搜索": {"evidence"},
                     "任务执行": {"evidence", "strategy"}, "总结验证": {"evidence", "conclusion"}}
        peak = max(fused.values(), default=1.0)
        for ident, score in fused.items():
            pool[ident]["score"] = score / peak
            pool[ident]["similarity"] = similarities.get(ident, 0.0)
        remaining = sorted(fused, key=lambda k: (-fused[k], k))[:40]
        selected = []
        while remaining and len(selected) < limit:
            def utility(ident):
                item = pool[ident]
                redundancy = max((max(0.0, cosine(item["vector"], other["vector"]))
                                  for other in selected
                                  if item["embedding_model"] == other["embedding_model"]), default=0.0)
                bonus = 0.04 if item["kind"] in preferred.get(role, set()) else 0.0
                return 0.8 * item["score"] - 0.2 * redundancy + bonus
            ident = max(remaining, key=utility)
            selected.append(pool[ident])
            remaining.remove(ident)
        return selected

    @staticmethod
    def scope_compatible(item: dict, context: dict) -> bool:
        """Conservative metadata check, not semantic or factual verification."""
        saved = item["content"].get("scope") or item["content"].get("_memory_scope") or {}
        if not isinstance(saved, dict):
            return False
        if not saved and item["kind"] in {"evidence", "conclusion"}:
            # An explicit current scope cannot validate an unscoped old claim.
            return False
        stable = ("application", "device", "platform", "python", "performance_version", "manifest_version")
        if any(k in saved and k in context and saved[k] != context[k] for k in stable):
            return False
        if item["kind"] == "conclusion":
            decision = ("exact_min", "semantic_min", "p95_max_ms", "ram_max_gb")
            if any(k in context and saved.get(k) != context[k] for k in decision):
                return False
        return True

    def cache_get(self, model: str, text: str):
        digest = hashlib.sha256(text.encode()).hexdigest()
        with self.connect() as db:
            row = db.execute("SELECT embedding FROM embedding_cache WHERE model=? AND hash=?",
                             (model, digest)).fetchone()
        return np.frombuffer(row[0], dtype="<f4").astype(float).tolist() if row else None

    def cache_put(self, model: str, text: str, values):
        arr = vector(values)
        digest = hashlib.sha256(text.encode()).hexdigest()
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO embedding_cache VALUES (?,?,?,?)",
                       (model, digest, arr.size, arr.tobytes()))

    @property
    def items(self):
        """兼容旧 UI；这里只返回当前会话轨迹及当前 namespace 的长期记忆。"""
        with self.connect() as db:
            rows = db.execute("SELECT * FROM memory WHERE namespace=? AND "
                              "(status='curated' OR session_id=?) ORDER BY created_at",
                              (self.namespace, self.session_id)).fetchall()
        result = []
        for row in rows:
            item = self.unpack(row)
            item.pop("vector")
            result.append(item)
        return result

    def summaries(self):
        return [{k: item[k] for k in ("id", "source_agent", "role", "created_at",
                "task_topic", "summary", "tags", "kind", "status", "source_ids")}
                for item in self.items]
