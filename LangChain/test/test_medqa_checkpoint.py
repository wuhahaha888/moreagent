"""Crash/restart tests use synthetic providers and local SQLite, never a live API."""
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .test_medqa import Agents, Books, Embeddings
from ..medqa import cli
from ..medqa.checkpoint import Checkpoint, CheckpointError, DurableCalls, DurableTools
from ..medqa.memory import SharedMemory
from ..medqa.retrieval import digest


class PowerCut(BaseException):
    pass


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.benchmark = self.root / "benchmark.json"
        self.benchmark.write_text(json.dumps({"medqa": {
            "q" + str(i): {"question": "fixture question", "options": {k: k for k in "ABCD"}, "answer": "A"}
            for i in range(3)}}))
        self.out = self.root / "run"

    def run_cli(self, *, agents=None, books=None, resume=False, extra=(), memory="off", limit=2, model=None, command="run"):
        agents = agents or Agents()
        books = books or Books()
        books.warmup = books.close = lambda: None
        embedding = SimpleNamespace(_model=object(), cache_key="fixture-model", cache_store=None,
                                    embed_text=Embeddings().embed_text)
        # These legacy checkpoint cases intentionally exercise the strict compatibility path.
        argv = ["medqa", command, "--output", str(self.out)]
        if resume:
            argv += ["--resume"]
        else:
            argv += ["--output-policy", "strict", "--benchmark", str(self.benchmark), "--split", "all", "--limit", str(limit),
                     "--memory", memory, "--max-total-tokens", "10000"]
        argv += list(extra)
        with patch("sys.argv", argv), patch.object(cli, "books", return_value=books), \
             patch.object(cli, "create_model", return_value=(agents, model or {"model": "fixture"})), \
             patch("LangChain.memory.embedding.QwenEmbeddingService", return_value=embedding), patch("builtins.print"):
            cli.main()
        return agents

    def results(self, condition="structured_off"):
        return [json.loads(line) for line in (self.out / (condition + ".jsonl")).read_text().splitlines()]

    def count_memory(self):
        with sqlite3.connect(self.out / "structured_on.sqlite3") as db:
            return db.execute("SELECT count(*) FROM memory").fetchone()[0]

    def test_completed_questions_skip_and_truncated_exports_are_rebuilt(self):
        original = self.run_cli()
        self.assertEqual(len(original.calls), 6)
        before = self.results()
        (self.out / "structured_off.jsonl").write_text('{"truncated":')
        (self.out / "report.json").write_text("broken export")
        (self.out / "manifest.json").write_text("broken export")
        after = self.run_cli(resume=True)
        self.assertEqual(after.calls, [])
        self.assertEqual(self.results(), before)
        report = json.loads((self.out / "report.json").read_text())["structured_off"]
        self.assertEqual(report["total_tokens"], 90)
        self.assertEqual(report["resume_count"], 1)
        self.assertEqual(json.loads((self.out / "manifest.json").read_text())["settings"]["limit"], 2)

    def test_known_responses_replay_after_tool_crash_without_rebilling(self):
        books = Books()
        books.search = lambda query: (_ for _ in ()).throw(PowerCut())
        original = Agents()
        with self.assertRaises(PowerCut):
            self.run_cli(agents=original, books=books)
        self.assertEqual(len(original.calls), 2)
        after = self.run_cli(resume=True)
        self.assertEqual(len(after.calls), 4)  # first question's last call + second question
        rows = self.results()
        self.assertEqual(sum(r["metrics"]["total_tokens"] for r in rows), 90)
        self.assertEqual(rows[0]["metrics"]["checkpoint_replayed_calls"], 2)
        self.assertEqual(rows[0]["metrics"]["model_calls"], 3)
        self.assertEqual(rows[0]["status"], "supported")

    def test_crash_after_search_result_commit_does_not_execute_search_again(self):
        original_execute = DurableTools.execute
        def cut_after_commit(journal, kind, request, callback):
            result = original_execute(journal, kind, request, callback)
            if kind == "search_textbooks":
                raise PowerCut()
            return result
        first_books = Books()
        with patch.object(DurableTools, "execute", cut_after_commit):
            with self.assertRaises(PowerCut):
                self.run_cli(books=first_books, limit=1)
        self.assertEqual(first_books.calls, 1)
        new_books = Books()
        after = self.run_cli(books=new_books, resume=True)
        self.assertEqual(new_books.calls, 0)
        self.assertEqual(len(after.calls), 1)
        result = self.results()[0]
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["retrieval_count"], 1)
        self.assertEqual(result["metrics"]["checkpoint_search_attempts"], 1)

    def test_changed_persisted_view_input_is_rejected_before_reusing_response(self):
        with patch.object(Checkpoint, "save_result", side_effect=PowerCut()):
            with self.assertRaises(PowerCut):
                self.run_cli(limit=1)
        # Corrupting a cached result changes the next paid request fingerprint.
        with sqlite3.connect(self.out / "checkpoint.sqlite3") as db:
            row = db.execute("SELECT operation_id,result FROM operations WHERE kind='evidence_view' AND result LIKE '%Fixture_0%'").fetchone()
            result = json.loads(row[1])
            result["views"][0]["text"] = "tampered visible evidence"
            db.execute("UPDATE operations SET result=? WHERE operation_id=?", (json.dumps(result), row[0]))
        with self.assertRaises(CheckpointError):
            self.run_cli(resume=True)

    def test_unknown_request_requires_explicit_retry_and_never_counts_as_free(self):
        class Interrupted(Agents):
            def call(self, *args):
                raise PowerCut()
        with self.assertRaises(PowerCut):
            self.run_cli(agents=Interrupted())
        with self.assertRaisesRegex(CheckpointError, "未自动重发"):
            self.run_cli(resume=True)
        after = self.run_cli(resume=True, extra=["--retry-unknown-calls"])
        self.assertEqual(len(after.calls), 3)
        self.assertEqual(len(self.results()), 1)  # stop further tasks with unknown bill
        report = json.loads((self.out / "report.json").read_text())["structured_off"]
        self.assertEqual(report["checkpoint_unknown_calls"], 1)
        self.assertEqual(report["usage_missing_calls"], 1)
        self.assertFalse(report["token_usage_complete"])
        with self.assertRaisesRegex(CheckpointError, "用量缺失"):
            self.run_cli(resume=True)

    def test_memory_not_published_before_result_commit(self):
        with patch.object(Checkpoint, "save_result", side_effect=PowerCut()):
            with self.assertRaises(PowerCut):
                self.run_cli(memory="on", limit=1)
        self.assertEqual(self.count_memory(), 0)
        after = self.run_cli(resume=True)
        self.assertEqual(after.calls, [])
        self.assertEqual(self.count_memory(), 1)
        row = self.results("structured_on")[0]
        self.assertEqual(row["metrics"]["checkpoint_replayed_calls"], 3)
        self.assertEqual(row["metrics"]["memory_written"], 1)
        self.assertNotIn("correct", self.memory_content())
        self.assertNotIn("answer", self.memory_content())

    def memory_content(self):
        with sqlite3.connect(self.out / "structured_on.sqlite3") as db:
            return json.loads(db.execute("SELECT content FROM memory LIMIT 1").fetchone()[0])

    def test_memory_commit_replayed_idempotently_after_result_is_durable(self):
        real_put = SharedMemory.put
        def cut_after_put(service, *args, **kwargs):
            real_put(service, *args, **kwargs)
            raise PowerCut()
        with patch.object(SharedMemory, "put", cut_after_put):
            with self.assertRaises(PowerCut):
                self.run_cli(memory="on", limit=1)
        self.assertEqual(self.count_memory(), 1)
        with sqlite3.connect(self.out / "checkpoint.sqlite3") as db:
            self.assertEqual(db.execute("SELECT state FROM questions").fetchone()[0], "ready")
        after = self.run_cli(resume=True)
        self.assertEqual(after.calls, [])
        self.assertEqual(self.count_memory(), 1)
        self.assertEqual(len(self.results("structured_on")), 1)

    def test_budgets_cannot_be_reset_or_increased_by_resume(self):
        self.run_cli(extra=["--max-total-tokens", "45"])
        self.assertEqual(len(self.results()), 1)
        after = self.run_cli(resume=True)
        self.assertEqual(after.calls, [])
        self.assertEqual(len(self.results()), 1)
        with self.assertRaisesRegex(CheckpointError, "已变化"):
            self.run_cli(resume=True, extra=["--max-total-tokens", "10000"])

    def test_model_and_code_changes_are_rejected(self):
        self.run_cli(limit=1)
        with self.assertRaisesRegex(CheckpointError, "已变化"):
            self.run_cli(resume=True, model={"model": "different-model"})
        with patch.object(cli, "source_hashes", return_value={"changed": "sha"}):
            with self.assertRaisesRegex(CheckpointError, "已变化"):
                self.run_cli(resume=True)

    def test_missing_condition_memory_is_not_recreated(self):
        with patch.object(Checkpoint, "save_result", side_effect=PowerCut()):
            with self.assertRaises(PowerCut):
                self.run_cli(memory="on", limit=1)
        (self.out / "structured_on.sqlite3").unlink()
        with self.assertRaisesRegex(CheckpointError, "数据库缺失"):
            self.run_cli(resume=True)

    def test_persisted_tool_result_reused_even_if_live_tool_changes(self):
        with patch.object(Checkpoint, "save_result", side_effect=PowerCut()):
            with self.assertRaises(PowerCut):
                self.run_cli(limit=1)
        books = Books()
        books.doc["text"] = "changed fixture evidence"
        books.doc["sha256"] = digest(books.doc["text"])
        after = self.run_cli(resume=True, books=books)
        self.assertEqual(after.calls, [])
        self.assertEqual(books.calls, 0)
        self.assertEqual(self.results()[0]["status"], "supported")

    def test_split_products_resume_without_repeating_answer_or_search(self):
        class BadAudit(Agents):
            def call(self, role, p, mode):
                output, usage = super().call(role, p, mode)
                if role == "VerifierReasoner":
                    output["claims"][0]["question_quotes"] = [p["question"]]
                return output, usage
        with patch.object(Checkpoint, "save_result", side_effect=PowerCut()):
            with self.assertRaises(PowerCut):
                self.run_cli(agents=BadAudit(), limit=1,
                             extra=["--output-policy", "decoupled", "--max-calls", "3"])
        books = Books()
        after = self.run_cli(agents=BadAudit(), books=books, resume=True)
        self.assertEqual(after.calls, [])
        self.assertEqual(books.calls, 0)
        result = self.results()[0]
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["protocol_result"]["status"], "issues")
        self.assertEqual(result["metrics"]["total_tokens"], 45)

    def test_split_products_keep_exactly_the_existing_four_condition_matrix(self):
        class BadAudit(Agents):
            def call(self, role, p, mode):
                output, usage = super().call(role, p, mode)
                if role == "VerifierReasoner":
                    output["claims"][0]["question_quotes"] = [p["question"]]
                return output, usage
        agents = self.run_cli(agents=BadAudit(), command="compare", limit=1,
                             extra=["--output-policy", "decoupled", "--max-calls", "3", "--memory-write", "off"])
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(set(report), {"text_off", "structured_off", "text_on", "structured_on"})
        self.assertEqual(len(agents.calls), 12)
        for condition, stats in report.items():
            self.assertEqual(stats["answer_completion_rate"], 1)
            self.assertEqual(stats["model_calls"], 3)
            self.assertEqual(stats["runtime_failed_questions"], 0)
            self.assertEqual(stats["protocol_status_counts"], {"issues": 1})
            self.assertEqual(self.results(condition)[0]["answer"], "A")

    def test_lock_prevents_two_writers_and_is_released(self):
        self.out.mkdir()
        with Checkpoint(self.out):
            with self.assertRaisesRegex(CheckpointError, "已有进程"):
                with Checkpoint(self.out):
                    self.fail("second writer acquired lock")
        with Checkpoint(self.out):
            pass

    def test_real_process_exit_keeps_saved_response_and_releases_lock(self):
        self.out.mkdir()
        program = '''
import os, sys
from pathlib import Path
from LangChain.medqa.checkpoint import Checkpoint, DurableCalls
class Agent:
    prompts = {"Planner": "fixture"}
    def call(self, role, payload, mode):
        return "saved response", {"total_tokens": 15}
store = Checkpoint(Path(sys.argv[1]))
store.initialize({"settings": {}}, [["structured", "off"]], resume=False)
store.begin("structured_off", 0, "q")
DurableCalls(Agent(), store, "structured_off", 0).call_with_budget("Planner", {"q": "fixture"}, "structured")
os._exit(23)
'''
        result = subprocess.run([sys.executable, "-B", "-c", program, str(self.out)],
                                cwd=cli.PROJECT, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 23, result.stderr.decode())
        class NoCall:
            prompts = {"Planner": "fixture"}
            def call(self, *args):
                raise AssertionError("must reuse saved response")
        with Checkpoint(self.out, resume=True) as store:
            calls = DurableCalls(NoCall(), store, "structured_off", 0)
            raw, usage = calls.call_with_budget("Planner", {"q": "fixture"}, "structured")
            self.assertEqual(raw, "saved response")
            self.assertEqual(usage["total_tokens"], 15)
            self.assertEqual(calls.replayed, 1)

    def test_resume_flags_cannot_be_enabled_by_config(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"resume": True}))
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            cli.parse_args(["run", "--config", str(config), "--limit", "1"])

    def test_four_conditions_resume_preserves_order_and_disables_timing_claim(self):
        class Reported(Agents):
            def call(self, *args):
                output, usage = super().call(*args)
                usage["source"] = "model"  # synthetic provider fixture, not an API
                return output, usage
        books, original = Books(), Reported()
        search = books.search
        def cut_third_search(query):
            if books.calls == 2:
                raise PowerCut()
            return search(query)
        books.search = cut_third_search
        with self.assertRaises(PowerCut):
            self.run_cli(agents=original, books=books, command="compare", limit=1)
        order = json.loads((self.out / "condition-order.json").read_text())
        self.assertEqual(len(original.calls), 8)
        resumed = Reported()
        self.run_cli(agents=resumed, command="compare", resume=True)
        self.assertEqual(len(resumed.calls), 4)
        self.assertEqual(json.loads((self.out / "condition-order.json").read_text()), order)
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(sum(r["total_tokens"] for r in report.values()), 180)
        comparison = json.loads((self.out / "comparison.json").read_text())
        self.assertTrue(comparison["matched_complete"])
        self.assertTrue(comparison["token_comparison_valid"])
        self.assertFalse(comparison["latency_comparison_valid"])
        self.assertIsNone(comparison["memory_structured"]["latency_reduction"])

    def test_unknown_request_in_ready_result_still_stops_after_outbox_recovery(self):
        class Failed(Agents):
            def call(self, *args):
                raise RuntimeError("synthetic provider error")
        real_apply = Checkpoint.apply_memory
        def interrupt_ready(store, *args):
            if store.db.execute("SELECT count(*) FROM questions WHERE state='ready'").fetchone()[0]:
                raise PowerCut()
            return real_apply(store, *args)
        with patch.object(Checkpoint, "apply_memory", interrupt_ready):
            with self.assertRaises(PowerCut):
                self.run_cli(agents=Failed())
        after = self.run_cli(resume=True)
        self.assertEqual(after.calls, [])
        self.assertEqual(len(self.results()), 1)
        report = json.loads((self.out / "report.json").read_text())["structured_off"]
        self.assertEqual(report["usage_missing_calls"], 1)
        self.assertFalse(report["token_usage_complete"])

    def test_unknown_retries_count_against_actual_request_limit(self):
        self.out.mkdir()
        class Cut:
            prompts = {"Planner": "fixture"}
            def call(self, *args):
                raise PowerCut()
        with Checkpoint(self.out) as store:
            store.initialize({"settings": {}}, [], resume=False)
            store.begin("structured_off", 0, "q")
            with self.assertRaises(PowerCut):
                DurableCalls(Cut(), store, "structured_off", 0, max_calls=1).call_with_budget("Planner", {}, "structured")
        with Checkpoint(self.out, resume=True) as store:
            calls = DurableCalls(Cut(), store, "structured_off", 0, max_calls=1, retry_unknown=True)
            with self.assertRaisesRegex(CheckpointError, "已达上限"):
                calls.call_with_budget("Planner", {}, "structured")
            self.assertEqual(calls.request_attempts(), 1)

    def test_resume_with_same_config_and_saved_limit(self):
        config = self.root / "settings.json"
        config.write_text(json.dumps({"benchmark": str(self.benchmark), "split": "all", "memory": "off", "max_total_tokens": 10000}))
        self.run_cli(extra=["--config", str(config)], limit=1)
        self.assertEqual(self.run_cli(resume=True, extra=["--config", str(config)]).calls, [])
        settings = json.loads(config.read_text())
        settings["memory_threshold"] = .9
        config.write_text(json.dumps(settings))
        with self.assertRaisesRegex(CheckpointError, "已变化"):
            self.run_cli(resume=True, extra=["--config", str(config)])


if __name__ == "__main__":
    unittest.main()
