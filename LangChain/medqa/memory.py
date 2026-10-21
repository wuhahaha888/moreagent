"""共享记忆服务：纯余弦召回，float32 通过进程间 Pipe 原样传输。"""
import json
import hashlib
import multiprocessing as mp
import time
import threading
from pathlib import Path

import numpy as np

from ..memory.advanced import SQLiteMemoryStore, cosine, vector
from .memory_units import normalize_unit, KINDS

MEMORY_EMBEDDING_VERSION = "medqa-task-evidence-v2-four-checks"


def memory_embedding_text(document, topic, conditions, task_group, excerpts):
    """对可复用记忆单元编码，而非把教材整段当成历史任务记忆。"""
    return "\n".join([task_group, topic, *conditions, document["title"], *excerpts])


def _worker(connection, path, namespace, model):
    try:
        store = SQLiteMemoryStore(Path(path), namespace, namespace, model)
        connection.send({"ready": True})
        while True:
            request = json.loads(connection.recv())
            if request["action"] == "close":
                break
            # 非文本状态：接收原始 float32 bytes，没有 JSON 数值列表编解码。
            raw_values = connection.recv_bytes()
            values = vector(np.frombuffer(raw_values, dtype="<f4")) if request["action"] in {"recall", "put", "put_unit"} else None
            try:
                if request["action"] == "recall":
                    clause, params = store._visible(request["trace"])
                    with store.connect() as db:
                        rows = db.execute(
                            f"SELECT * FROM memory WHERE {clause} AND status='curated' AND trace_id<>?",
                            [*params, request["trace"]]).fetchall()
                    candidates = []
                    kinds = request.get("kinds") or ["evidence"]
                    for row in rows:
                        item = store.unpack(row)
                        kind = "evidence" if item["kind"] == "textbook_extract" else item["kind"]
                        if kind not in kinds:
                            continue
                        if (item["embedding_model"] != model or item["dim"] != values.size
                                or item["content"].get("corpus_version") != request["version"]
                                or (request.get("task_group") and item["content"].get("task_group") != request["task_group"])):
                            continue
                        score = cosine(values, item.pop("vector"))
                        if score >= request["threshold"]:
                            item["similarity"] = score
                            # 接收方会从唯一教材回读原文，IPC 不再重复发送完整摘录。
                            item["content"] = {k: v for k, v in item["content"].items() if k != "text"}
                            if kind in {"strategy", "episode"}:
                                item["content"] = {k:v for k,v in item["content"].items()
                                                   if k in {"corpus_version","task_group","validation","memory_type","task_intent",
                                                            "applicability","conditions","conditions_role"}}
                            candidates.append(item)
                    selected, seen_sources = [], set()
                    ordered = sorted(candidates, key=lambda x: (-x["similarity"], x["id"]))
                    if request.get("kinds") and request.get('diversify',True):
                        first, rest, seen_kinds = [], [], set()
                        for candidate in ordered:
                            kind = "evidence" if candidate["kind"] == "textbook_extract" else candidate["kind"]
                            (rest if kind in seen_kinds else first).append(candidate)
                            seen_kinds.add(kind)
                        ordered = first + rest
                    for candidate in ordered:
                        source = ((candidate["content"]["id"], candidate["content"]["sha256"])
                                  if candidate["kind"] in {"evidence","textbook_extract"} and not request.get('preserve_fragments')
                                  else (candidate["kind"],candidate["id"]))
                        if source in seen_sources:
                            continue
                        seen_sources.add(source)
                        selected.append(candidate)
                        if len(selected) >= request["limit"]:
                            break
                    response = {"hits": selected}
                elif request["action"] == "resolve":
                    clause, params = store._visible(request["trace"])
                    ids = request["ids"]
                    with store.connect() as db:
                        rows = db.execute(
                            f"SELECT * FROM memory WHERE {clause} AND status='curated' AND trace_id<>? "
                            f"AND id IN ({','.join('?' for _ in ids)})",
                            [*params, request["trace"], *ids]).fetchall()
                    resolved = []
                    for row in rows:
                        item = store.unpack(row)
                        content = item["content"]
                        if (item["embedding_model"] != model or content.get("corpus_version") != request["version"]
                                or (request.get("task_group") and content.get("task_group") != request["task_group"])):
                            continue
                        item.pop("vector", None)
                        item["content"] = {k: v for k, v in content.items() if k != "text"}
                        resolved.append(item)
                    response = {"records": resolved}
                elif request["action"] == "put_unit":
                    unit = normalize_unit(request["unit"],request["trace"])
                    content = unit["content"]
                    identity = None
                    if (unit['kind']=='evidence' and content.get('index_schema')=='need-memory-v1'
                            and content.get('conditions_role')=='origin_context_only; recheck_for_each_task'):
                        # Case provenance is not part of the reusable knowledge identity.
                        # Keep different goals, source versions and excerpts separate.
                        identity = {k:v for k,v in content.items() if k not in {'origin_question_id','conditions'}}
                        identity['excerpts'] = sorted(set(content['excerpts']))
                    ident = store.put(role=unit["source_agent"],topic=unit["topic"],summary=unit["summary"],
                        content=content,embedding=values,trace_id=request["trace"],kind=unit["kind"],
                        status="curated",tags=[content["corpus_version"],unit["kind"]],
                        source_ids=[content["id"]] if unit["kind"]=="evidence" else content.get("source_ids",[]),
                        expires_at=time.time()+30*86400,identity_content=identity)
                    if identity is not None:
                        origin = json.dumps({k:content[k] for k in ('origin_question_id','conditions','need_goal')},
                                            ensure_ascii=False,sort_keys=True,separators=(',',':'))
                        with store.connect() as db:
                            db.execute('INSERT OR IGNORE INTO memory_origin VALUES (?,?,?,?,?)',
                                (ident,request['trace'],hashlib.sha256(origin.encode()).hexdigest(),time.time(),origin))
                    response = {"memory_id":ident}
                elif request["action"] == "put":
                    doc = request["document"]
                    metadata = request.get("metadata", {})
                    content = {k: doc[k] for k in ("id", "title", "text", "sha256", "corpus_version")}
                    content.update({k: metadata[k] for k in ("task_group", "conditions", "retrieval_queries", "execution_feedback", "excerpts", "review_policy") if k in metadata})
                    excerpts = metadata.get("excerpts", [])
                    if (not isinstance(excerpts, list) or any(not isinstance(t, str) or not t or t not in doc["text"] for t in excerpts)):
                        raise ValueError("memory excerpts must be exact source spans")
                    summary = "\n".join(excerpts)[:600] if excerpts else doc["text"][:300]
                    content["memory_context_hash"] = hashlib.sha256(json.dumps(
                        [metadata.get("topic", doc["title"]), metadata.get("conditions", []), excerpts],
                        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
                    content["validation"] = "source_checked; applicability_requires_current_task_review"
                    # 相同来源和记忆上下文去重；新上下文形成新单位，不覆盖旧条件或向量。
                    with store.connect() as db:
                        existing = db.execute(
                            "SELECT id FROM memory WHERE namespace=? AND status='curated' "
                            "AND (expires_at IS NULL OR expires_at>?) "
                            "AND json_extract(content,'$.id')=? AND json_extract(content,'$.sha256')=? "
                            "AND json_extract(content,'$.corpus_version')=? "
                            "AND coalesce(json_extract(content,'$.task_group'),'')=? "
                            "AND json_extract(content,'$.memory_context_hash')=? AND embedding_model=? LIMIT 1",
                            (namespace, time.time(), doc["id"], doc["sha256"], doc["corpus_version"], metadata.get("task_group", ""),
                             content["memory_context_hash"], model)).fetchone()
                    ident = existing["id"] if existing else store.put(role=metadata.get("source_agent", "Investigator"),
                                      topic=metadata.get("topic", doc["title"]), summary=summary,
                                      content=content,
                                      embedding=values, trace_id=request["trace"], kind="textbook_extract",
                                      status="curated", tags=[doc["corpus_version"]],
                                      source_ids=[doc["id"]], expires_at=time.time() + 30 * 86400)
                    response = {"memory_id": ident}
                else:
                    raise ValueError("不支持的记忆动作")
            except Exception as exc:
                response = {"error": type(exc).__name__}
            connection.send(response)
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class SharedMemory:
    """独立的基础设施服务；不调用LLM，不计入三个模型Agent。"""
    def __init__(self, path, namespace, model, timeout=30):
        self.timeout = timeout
        if timeout <= 0:
            raise ValueError("记忆服务超时必须大于0")
        self._lock, self._closed = threading.RLock(), False
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, str(path), namespace, model), daemon=True)
        self._process.start()
        child.close()
        self.transfers = self.bytes = self.lookups = self.vector_transfers = 0
        try:
            if not self._receive().get("ready"):
                raise RuntimeError("记忆服务握手失败")
        except Exception:
            self.close()
            raise

    def _receive(self):
        if not self._connection.poll(self.timeout):
            raise TimeoutError("记忆服务超时")
        response = self._connection.recv()
        if "error" in response:
            raise RuntimeError("记忆服务错误: " + response["error"])
        return response

    def _request(self, request, values=None):
        raw = vector(values).tobytes() if values is not None else b""
        if values is not None and (not raw or len(raw) > 8192 * 4):
            raise ValueError("向量需为1至8192维")
        with self._lock:
            if self._closed:
                raise RuntimeError("记忆通道已关闭，不能消费迟到响应")
            try:
                self._connection.send(json.dumps(request, ensure_ascii=False))
                self._connection.send_bytes(raw)
                self.transfers += 1
                self.vector_transfers += bool(raw)
                self.bytes += len(raw)
                return self._receive()
            except (TimeoutError, EOFError, BrokenPipeError, OSError):
                # 超时后禁止复用旧通道，以免下一任务收到上一任务的迟到结果。
                self.close()
                raise

    def recall(self, values, trace, version, threshold=.65, limit=3, *, task_group=None, kinds=None, diversify=True,
               preserve_fragments=False):
        if not np.isfinite(threshold) or not -1 <= threshold <= 1 or not 1 <= limit <= 100:
            raise ValueError("记忆召回阈值/条数无效")
        if kinds is not None and (not isinstance(kinds,list) or not kinds or any(k not in KINDS for k in kinds)):
            raise ValueError("记忆检索类型无效")
        self.lookups += 1
        return self._request({"action": "recall", "trace": trace, "version": version,
                              "threshold": threshold, "limit": limit, "task_group": task_group, "kinds":kinds,
                              "diversify":bool(diversify),"preserve_fragments":bool(preserve_fragments)}, values)["hits"]

    def put_unit(self, unit, values, trace):
        normalize_unit(unit,trace)
        return self._request({"action":"put_unit","unit":unit,"trace":trace},values)["memory_id"]

    def put(self, document, values, trace, **metadata):
        if document.get("sha256") != hashlib.sha256(document["text"].encode()).hexdigest():
            raise ValueError("记忆原文与hash不一致")
        return self._request({"action": "put", "trace": trace, "document": document, "metadata": metadata}, values)["memory_id"]

    def resolve(self, memory_ids, trace, version, *, task_group=None):
        """Resolve IDs inside the shared service; full textbook text stays at its source."""
        if (not isinstance(memory_ids, list) or not 1 <= len(memory_ids) <= 10
                or any(not isinstance(i, str) or not i for i in memory_ids)
                or len(set(memory_ids)) != len(memory_ids)):
            raise ValueError("记忆引用须为1至10个不重复ID")
        return self._request({"action": "resolve", "ids": memory_ids, "trace": trace,
                              "version": version, "task_group": task_group})["records"]

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._process.is_alive():
                try:
                    self._connection.send(json.dumps({"action": "close"}))
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self._process.join(timeout=1)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=1)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=1)
            self._connection.close()
