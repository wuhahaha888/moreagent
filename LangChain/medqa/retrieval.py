"""只读 Textbooks：BM25 召回、MedCPT 重排、原文回读与版本指纹。"""
import hashlib
import json
import os
import re
import selectors
import subprocess
import time
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Textbooks:
    def __init__(self, root: Path, python: Path, java: Path, ranker: Path,
                 candidates=20, top_k=4, timeout=60):
        self.root, self.python, self.java, self.ranker = map(Path, (root, python, java, ranker))
        # Conda Java 的 bin/java 可能指向 lib/jvm/bin/java，需使用真实 JDK 根目录。
        self.java = (self.java / "bin/java").resolve().parent.parent
        if not 1 <= top_k <= candidates <= 100:
            raise ValueError("要求 1 <= top_k <= candidates <= 100")
        self.candidates, self.top_k, self.timeout = candidates, top_k, timeout
        files = sorted((self.root / "chunk").glob("*.jsonl"))
        if not files or not (self.root / "index/bm25").is_dir():
            raise ValueError("需要已有 Textbooks chunk 与 BM25 索引，不自动构建")
        fingerprint = hashlib.sha256()
        for file in files:
            fingerprint.update(file.name.encode())
            with file.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    fingerprint.update(block)
        self.version = fingerprint.hexdigest()
        retrieval_fingerprint = hashlib.sha256(b"bm25-medcpt-maxlen512-v1")
        for base in (self.root / "index/bm25", self.ranker):
            for file in sorted(base.rglob("*")):
                if file.is_file():
                    retrieval_fingerprint.update(str(file.relative_to(base)).encode())
                    with file.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            retrieval_fingerprint.update(block)
        self.operation_version = retrieval_fingerprint.hexdigest()
        self._file_stats = {file.stem: (file.stat().st_size, file.stat().st_mtime_ns) for file in files}
        self._offsets, self._process, self._selector, self._reranker = {}, None, None, None
        self.stats = {"searches": 0, "source_reads": 0, "reranks": 0}

    def read(self, ident):
        match = re.fullmatch(r"([A-Za-z0-9_-]+)_(\d+)", ident)
        if not match:
            raise ValueError("非法教材 ID")
        stem, index = match.group(1), int(match.group(2))
        file = self.root / "chunk" / (stem + ".jsonl")
        stat = file.stat()
        if (stat.st_size, stat.st_mtime_ns) != self._file_stats.get(stem):
            raise ValueError("教材在运行中已更改，须重新启动并冻结语料版本")
        if stem not in self._offsets:
            offsets = []
            with file.open("rb") as handle:
                while True:
                    pos = handle.tell()
                    if not handle.readline():
                        break
                    offsets.append(pos)
            self._offsets[stem] = offsets
        with file.open("rb") as handle:
            handle.seek(self._offsets[stem][index])
            row = json.loads(handle.readline())
        if row["id"] != ident:
            raise ValueError("教材 ID 与原文不一致")
        self.stats["source_reads"] += 1
        text = row["content"]
        return {"id": ident, "title": row["title"], "text": text,
                "sha256": digest(text), "corpus_version": self.version}

    def _response(self):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            # 二进制无缓冲 readline，避免 TextIO 的预读与 select 不一致。
            events = self._selector.select(max(0, deadline - time.monotonic()))
            if not events:
                break
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("BM25 worker 已退出")
            if line.startswith(b"STRUCKFLOW:"):
                response = json.loads(line[len(b"STRUCKFLOW:"):])
                if "error" in response:
                    raise RuntimeError("BM25 worker 错误: " + response["error"])
                return response
        raise TimeoutError("BM25 worker 响应超时")

    def _start(self):
        if self._process is not None and self._process.poll() is not None:
            self.close()
        if self._process is not None:
            return
        env = dict(os.environ)
        env["JAVA_HOME"] = str(self.java)
        env["PATH"] = str(self.java / "bin") + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = str(self.python.parent.parent / "lib")
        worker = Path(__file__).with_name("bm25_worker.py")
        self._process = subprocess.Popen([str(self.python), "-B", str(worker),
                                          str(self.root / "index/bm25")],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, env=env, bufsize=0)
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        if not self._response().get("ready"):
            raise RuntimeError("BM25 worker 握手失败")

    def rankings(self, query):
        """相同 BM25 候选池的原排序与 MedCPT 排序，用于独立检索评测。"""
        self.stats["searches"] += 1
        try:
            self._start()
            payload = json.dumps({"query": query, "limit": self.candidates}) + "\n"
            self._process.stdin.write(payload.encode())
            hits = self._response()["hits"]
        except Exception:
            self.close()  # 超时后重建 worker，避免下一查询误收到前一查询的迟到结果。
            raise
        documents = [{**self.read(hit["id"]), "bm25": hit["bm25"]} for hit in hits]
        if not documents:
            return {"bm25": [], "medcpt": []}
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(str(self.ranker), device="cpu", max_length=512,
                                         local_files_only=True)
        scores = self._reranker.predict([(query, d["text"]) for d in documents],
                                        batch_size=8, show_progress_bar=False)
        self.stats["reranks"] += 1
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        return {"bm25": documents, "medcpt": sorted(documents, key=lambda d: (-d["rerank_score"], d["id"]))}

    def search(self, query):
        return self.rankings(query)["medcpt"][:self.top_k]

    def warmup(self):
        """统一预热，避免把仅第一条件的模型/Java 启动时间算作协议差异。"""
        self.search("clinical evidence")

    def close(self):
        if self._selector:
            self._selector.close()
        if self._process:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process.stdin.close()
            self._process.stdout.close()
        self._process, self._selector = None, None
