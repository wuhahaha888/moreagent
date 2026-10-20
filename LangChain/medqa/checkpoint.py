"""Durable batch ledger. A remote request and local disk cannot commit atomically."""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import uuid


FORMAT = 2


class CheckpointError(RuntimeError):
    """Safe diagnostic with no SDK exception, URL or credentials."""


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_text(path, text):
    """An interrupted export keeps either the previous file or the new whole file."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def immutable_manifest(manifest):
    result = deepcopy(manifest)
    # Command location/authorization flags are not experimental treatments.
    for key in ("resume", "retry_unknown_calls", "config"):
        result["settings"].pop(key, None)
    return result


class Checkpoint:
    def __init__(self, directory, *, resume=False):
        self.directory = Path(directory)
        self.lock = (self.directory / ".run.lock").open("a+b")
        self.db = None
        self.exported = {}
        try:
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise CheckpointError("该结果目录已有进程运行，不能同时执行或续跑") from None
            path = self.directory / "checkpoint.sqlite3"
            if resume and not path.is_file():
                raise CheckpointError("该目录没有新版检查点；旧实验不能自动续跑")
            if resume:
                with sqlite3.connect("file:" + str(path.resolve()) + "?mode=ro", uri=True) as probe:
                    row = probe.execute("SELECT value FROM metadata WHERE key='format'").fetchone()
                    if not row or json.loads(row[0]) != FORMAT:
                        raise CheckpointError("检查点版本不同；v5记录保留不迁移，请使用v6新目录")
            self.db = sqlite3.connect(path, timeout=30)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS questions (
                    condition TEXT, position INTEGER, question_id TEXT, trace TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'running', result TEXT, writes TEXT,
                    PRIMARY KEY(condition,position), UNIQUE(condition,question_id));
                CREATE TABLE IF NOT EXISTS calls (
                    condition TEXT, position INTEGER, ordinal INTEGER, attempt INTEGER,
                    request_hash TEXT NOT NULL, role TEXT NOT NULL, state TEXT NOT NULL,
                    raw TEXT, usage TEXT, seconds REAL,
                    PRIMARY KEY(condition,position,ordinal,attempt));
                CREATE TABLE IF NOT EXISTS operations (
                    condition TEXT, position INTEGER, operation_id TEXT, kind TEXT, request TEXT,
                    state TEXT, result TEXT, error TEXT, attempts INTEGER, seconds REAL DEFAULT 0,
                    PRIMARY KEY(condition,position,operation_id));
                CREATE TABLE IF NOT EXISTS control_states (
                    condition TEXT, position INTEGER, ordinal INTEGER, trigger TEXT, snapshot TEXT,
                    PRIMARY KEY(condition,position,ordinal));
            """)
            if resume and self.meta("format") != FORMAT:
                raise CheckpointError("检查点格式不支持，不能恢复")
            if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise CheckpointError("检查点数据库损坏，停止恢复")
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        self.lock.close()

    def meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, encode(value)))

    def initialize(self, manifest, conditions, *, resume):
        if resume:
            if (immutable_manifest(manifest) != immutable_manifest(self.meta("manifest"))
                    or conditions != self.meta("conditions")):
                raise CheckpointError("续跑配置、题序、模型、数据或源码版本已变化；请恢复原配置或使用新的结果目录")
        with self.db:
            if not resume:
                self.set_meta("format", FORMAT)
                self.set_meta("manifest", manifest)
                self.set_meta("conditions", conditions)
            else:
                self.set_meta("resume_count", self.meta("resume_count", 0) + 1)
        sync_directory(self.directory)

    def begin(self, condition, position, question_id):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO questions(condition,position,question_id,trace) VALUES (?,?,?,?)",
                            (condition, position, question_id, uuid.uuid4().hex))
        row = self.question(condition, position)
        if row["question_id"] != question_id:
            raise CheckpointError("检查点题目顺序不一致")
        return row

    def question(self, condition, position):
        return self.db.execute("SELECT * FROM questions WHERE condition=? AND position=?",
                               (condition, position)).fetchone()

    def started(self, condition):
        return self.db.execute("SELECT count(*) FROM questions WHERE condition=?", (condition,)).fetchone()[0] > 0

    def results(self, condition):
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT result FROM questions WHERE condition=? AND state='complete' ORDER BY position", (condition,))]

    def save_result(self, condition, position, result, writes):
        # One transaction binds the graded result and the ungraded memory outbox.
        with self.db:
            self.db.execute("UPDATE questions SET state='ready',result=?,writes=? WHERE condition=? AND position=?",
                            (encode(result), encode(writes), condition, position))

    def apply_memory(self, condition, memory):
        for row in self.db.execute("SELECT * FROM questions WHERE condition=? AND state='ready' ORDER BY position",
                                   (condition,)).fetchall():
            writes = json.loads(row["writes"])
            if writes and memory is None:
                raise CheckpointError("检查点有待写入记忆，但记忆服务未启用")
            try:
                for item in writes:
                    if "unit" in item:
                        memory.put_unit(item["unit"],item["values"],item["trace"])
                    else:
                        memory.put(item["document"], item["values"], item["trace"], **item["metadata"])
            except Exception:
                raise CheckpointError("已保存答案，但记忆提交尚未完成；修复记忆服务后用 --resume 重试，不会重调模型") from None
            # SharedMemory.put deduplicates the original source/context. Reapplying
            # after a crash between put and this transaction does not add a row.
            with self.db:
                self.db.execute("UPDATE questions SET state='complete' WHERE condition=? AND position=?",
                                (condition, row["position"]))

    def unknown_calls(self, condition, position=None):
        where, params = "condition=? AND state IN ('pending','abandoned')", [condition]
        if position is not None:
            where += " AND position=?"
            params.append(position)
        return self.db.execute("SELECT count(*) FROM calls WHERE " + where, params).fetchone()[0]

    def ensure_retry_allowed(self, retry):
        row = self.db.execute("SELECT count(*) FROM calls c JOIN questions q "
            "ON q.condition=c.condition AND q.position=c.position WHERE c.state='pending' AND q.state='running'").fetchone()
        if row[0] and not retry:
            raise CheckpointError("有已发出但未保存响应的请求，服务端用量未知；未自动重发。核对账单后，可显式加 --retry-unknown-calls 承担重复调用费用")

    def export_results(self, condition):
        # Rebuild once on process start, then append only newly committed rows.
        # A crash during append is repaired from SQLite at the next start.
        previous = self.exported.get(condition)
        path = self.directory / (condition + ".jsonl")
        if previous is None:
            rows = self.results(condition)
            atomic_text(path, "".join(encode(result) + "\n" for result in rows))
            self.exported[condition] = len(rows)
        else:
            rows = self.db.execute("SELECT result FROM questions WHERE condition=? AND state='complete' "
                                   "ORDER BY position LIMIT -1 OFFSET ?", (condition, previous)).fetchall()
            if rows:
                with path.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(row[0] + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self.exported[condition] += len(rows)


class StoredToolError(RuntimeError):
    def __init__(self, name):
        self.tool_error_type = name
        super().__init__(name)


class DurableTools:
    """缓存完整只读结果和处理视图。未知计算窗口可重算，不宣称物理exactly-once。"""
    def __init__(self, store, condition, position, *, max_searches=6, max_reads=4):
        self.store, self.condition, self.position = store, condition, position
        self.replayed = self.executed = self.unknown_recomputations = self.ordinal = 0
        self.fatal_error = None
        self.max_searches, self.max_reads = max_searches, max_reads
        self.historical_seconds = store.db.execute(
            "SELECT coalesce(sum(seconds),0) FROM operations WHERE condition=? AND position=?",
            (condition, position)).fetchone()[0]

    def execute(self, kind, request, callback):
        try:
            return self._execute(kind, request, callback)
        except (sqlite3.Error, OSError) as exc:
            self.fatal_error = CheckpointError("工具检查点写入失败，停止以保留恢复数据")
            raise self.fatal_error from None
        except CheckpointError as exc:
            self.fatal_error = exc
            raise

    def _execute(self, kind, request, callback):
        operation_id = hashlib.sha256(encode(["medqa-v6", kind, request]).encode()).hexdigest()
        key = self.condition, self.position, operation_id
        row = self.store.db.execute("SELECT * FROM operations WHERE condition=? AND position=? AND operation_id=?", key).fetchone()
        if row and row["state"] == "complete":
            self.replayed += 1
            return json.loads(row["result"])
        if row and row["state"] == "error":
            self.replayed += 1
            raise StoredToolError(row["error"])
        attempt = row["attempts"] + 1 if row else 1
        if attempt > 3:
            raise CheckpointError("同一只读工具连续未知中断已达保护上限，保留检查点")
        fuse = {"search_textbooks": self.max_searches, "read_source": self.max_reads}.get(kind)
        attempts_used = self.store.db.execute("SELECT coalesce(sum(attempts),0) FROM operations WHERE condition=? AND position=? AND kind=?",
                                             (self.condition, self.position, kind)).fetchone()[0]
        if fuse is not None and attempts_used >= fuse:
            raise StoredToolError("physical_tool_attempt_budget_exhausted")
        if row:
            self.unknown_recomputations += 1
        with self.store.db:
            self.store.db.execute("INSERT OR REPLACE INTO operations VALUES (?,?,?,?,?,'pending',NULL,NULL,?,0)",
                                 (*key, kind, encode(request), attempt))
        started = time.monotonic()
        self.executed += 1
        try:
            result = callback()
        except Exception as exc:
            with self.store.db:
                self.store.db.execute("UPDATE operations SET state='error',error=?,seconds=? WHERE condition=? AND position=? AND operation_id=?",
                                     (type(exc).__name__, time.monotonic() - started, *key))
            raise StoredToolError(type(exc).__name__) from None
        # Commit full post-rerank docs / raw-view mapping before returning them to Runtime.
        with self.store.db:
            self.store.db.execute("UPDATE operations SET state='complete',result=?,seconds=? WHERE condition=? AND position=? AND operation_id=?",
                                 (encode(result), time.monotonic() - started, *key))
        return result

    def attempts(self, kind):
        return self.store.db.execute("SELECT coalesce(sum(attempts),0) FROM operations WHERE condition=? AND position=? AND kind=?",
                                     (self.condition, self.position, kind)).fetchone()[0]

    def snapshot(self, trigger, state):
        self.ordinal += 1
        key = self.condition, self.position, self.ordinal
        try:
            previous = self.store.db.execute("SELECT trigger,snapshot FROM control_states WHERE condition=? AND position=? AND ordinal=?", key).fetchone()
            if previous and (previous["trigger"] != trigger or previous["snapshot"] != encode(state)):
                raise CheckpointError("控制状态恢复不一致，保留账本，禁止静默改写决策历史")
            with self.store.db:
                self.store.db.execute("INSERT OR IGNORE INTO control_states VALUES (?,?,?,?,?)", (*key, trigger, encode(state)))
        except (sqlite3.Error, OSError, CheckpointError):
            self.fatal_error = CheckpointError("控制状态无法可靠保存或重放，停止并保留账本")
            raise self.fatal_error from None


class PendingMemory:
    """Stage task knowledge until its result and writes are durable together."""
    def __init__(self, memory):
        self.memory, self.writes = memory, []

    @property
    def transfers(self):
        return self.memory.transfers + len(self.writes)

    @property
    def bytes(self):
        return self.memory.bytes + sum(len(item["values"]) * 4 for item in self.writes)

    @property
    def vector_transfers(self):
        return getattr(self.memory,'vector_transfers',0) + len(self.writes)

    def recall(self, *args, **kwargs):
        return self.memory.recall(*args, **kwargs)

    def resolve(self, *args, **kwargs):
        return self.memory.resolve(*args, **kwargs)

    def put(self, document, values, trace, **metadata):
        from ..memory.advanced import vector
        values = vector(values)
        if not 1 <= values.size <= 8192:
            raise ValueError("记忆向量维度无效")
        if hashlib.sha256(document["text"].encode()).hexdigest() != document["sha256"]:
            raise ValueError("记忆原文校验失败")
        self.writes.append(deepcopy({"document": document, "values": values.tolist(),
                                     "trace": trace, "metadata": metadata}))
        return "pending:" + str(len(self.writes))

    def put_unit(self, unit, values, trace):
        from .memory_units import normalize_unit
        from ..memory.advanced import vector
        normalize_unit(unit,trace)
        values = vector(values)
        if not 1 <= values.size <= 8192:
            raise ValueError("记忆向量维度无效")
        self.writes.append(deepcopy({"unit":unit,"values":values.tolist(),"trace":trace}))
        return "pending:" + str(len(self.writes))


class DurableCalls:
    def __init__(self, agents, checkpoint, condition, position, *, retry_unknown=False, max_calls=8):
        from ..agents.medqa_prompts import MEDQA_ROLE_PROMPTS
        self.agents, self.store = agents, checkpoint
        self.condition, self.position = condition, position
        self.prompts = getattr(agents, "prompts", MEDQA_ROLE_PROMPTS)
        self.ordinal = self.replayed = 0
        self.retry_unknown, self.fatal_error = retry_unknown, None
        self.max_calls = max_calls
        self.historical_seconds = checkpoint.db.execute(
            "SELECT coalesce(sum(seconds),0) FROM calls WHERE condition=? AND position=? AND state='complete'",
            (condition, position)).fetchone()[0]

    def replay_pending(self):
        return bool(self.store.db.execute("SELECT 1 FROM calls WHERE condition=? AND position=? AND ordinal>?",
                                         (self.condition, self.position, self.ordinal)).fetchone())

    def request_attempts(self):
        return self.store.db.execute("SELECT count(*) FROM calls WHERE condition=? AND position=?",
                                     (self.condition, self.position)).fetchone()[0]

    def call_with_budget(self, role, payload, mode, **budget):
        try:
            return self._call(role, payload, mode, **budget)
        except CheckpointError as exc:
            # Runtime catches model exceptions; the batch must not commit that
            # sanitized error as an answer and lose the original resumable state.
            self.fatal_error = exc
            raise
        except (sqlite3.Error, OSError):
            self.fatal_error = CheckpointError("调用检查点写入失败，停止以保留可恢复状态；请检查磁盘后续跑")
            raise self.fatal_error from None

    def validate_replay(self):
        maximum = self.store.db.execute("SELECT max(ordinal) FROM calls WHERE condition=? AND position=?",
                                        (self.condition, self.position)).fetchone()[0] or 0
        if maximum > self.ordinal:
            raise CheckpointError("本次运行未重放完已记录的调用，不能丢弃历史用量；停止并保留检查点")

    def _call(self, role, payload, mode, **budget):
        self.ordinal += 1
        fingerprint = hashlib.sha256(encode([role, mode, self.prompts[role], payload]).encode()).hexdigest()
        key = self.condition, self.position, self.ordinal
        row = self.store.db.execute("SELECT * FROM calls WHERE condition=? AND position=? AND ordinal=? "
                                    "ORDER BY attempt DESC LIMIT 1", key).fetchone()
        if row:
            if row["request_hash"] != fingerprint or row["role"] != role:
                raise CheckpointError("恢复重放时角色输入已变化（可能是来源、记忆或检索结果变化），停止以避免复用错误响应")
            if row["state"] == "complete":
                self.replayed += 1
                return json.loads(row["raw"]), json.loads(row["usage"])
            if not self.retry_unknown:
                raise CheckpointError("该请求响应未落盘，用量未知；需要显式 --retry-unknown-calls 才会重发")
        attempt = row["attempt"] + 1 if row else 1
        if self.request_attempts() >= self.max_calls:
            raise CheckpointError("该题实际请求次数（包括响应未知的请求）已达上限，停止重发并保留检查点")
        with self.store.db:
            if row:
                self.store.db.execute("UPDATE calls SET state='abandoned' WHERE condition=? AND position=? AND ordinal=? AND attempt=?",
                                      (*key, row["attempt"]))
            self.store.db.execute("INSERT INTO calls(condition,position,ordinal,attempt,request_hash,role,state) "
                                  "VALUES (?,?,?,?,?,?,'pending')", (*key, attempt, fingerprint, role))
        start = time.monotonic()
        if hasattr(self.agents, "call_with_budget"):
            raw, usage = self.agents.call_with_budget(role, payload, mode, **budget)
        else:
            raw, usage = self.agents.call(role, payload, mode)
        # Commit immediately, before parsing/validation or any further tool work.
        with self.store.db:
            self.store.db.execute("UPDATE calls SET state='complete',raw=?,usage=?,seconds=? "
                "WHERE condition=? AND position=? AND ordinal=? AND attempt=?",
                (encode(raw), encode(usage), time.monotonic() - start, *key, attempt))
        return raw, usage
