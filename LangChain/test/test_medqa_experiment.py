"""实验统计与交接审计；全部使用替身，不报告实际API收益。"""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .test_medqa import Agents, Books, Embeddings
from ..agents.medqa_prompts import INVESTIGATOR, REASONER
from ..medqa.data import Question
from ..medqa.runtime import Runtime, ChatRoles
from ..medqa.usage import normalize_usage
from ..medqa import cli
from ..medqa.experiment import validate_seed


class ExperimentTest(unittest.TestCase):
    def test_partial_or_invalid_usage_is_never_billed_as_complete(self):
        for raw in ({}, {"total_tokens": 15}, {"prompt_tokens": 10},
                    {"prompt_tokens": True, "completion_tokens": 2},
                    {"prompt_tokens": -1, "completion_tokens": 2},
                    {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 100}):
            reply = SimpleNamespace(response_metadata={"token_usage": raw}, usage_metadata={})
            usage = normalize_usage(reply, "prompt", "response")
            self.assertEqual(usage["source"], "estimated")
            self.assertFalse(usage["usage_complete"])

    def test_authoritative_usage_handles_metadata_empty_completion_and_derived_total(self):
        reply = SimpleNamespace(response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 0}}, usage_metadata={})
        usage = normalize_usage(reply, "not a tokenizer", "")
        self.assertEqual(usage["total_tokens"], 10)
        self.assertTrue(usage["total_derived"])
        self.assertEqual(usage["source"], "model")
        reply = SimpleNamespace(response_metadata={}, usage_metadata={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15})
        self.assertEqual(normalize_usage(reply, "", "")["total_tokens"], 15)

    def test_cached_and_reasoning_tokens_are_subsets_not_added_again(self):
        class Model:
            def invoke_bounded(self, messages, **kwargs):
                return SimpleNamespace(content='{}', response_metadata={"model": "fixture", "token_usage": {
                    "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                    "prompt_tokens_details": {"cached_tokens": 70},
                    "completion_tokens_details": {"reasoning_tokens": 10}}})
        _, usage = ChatRoles(Model()).call(REASONER, {}, "structured")
        self.assertEqual(usage["total_tokens"], 120)
        self.assertEqual(usage["cached_prompt_tokens"], 70)
        self.assertEqual(usage["reasoning_tokens"], 10)

    def test_followup_preserves_basis_question_and_conflicts_without_unrelated_raw_text(self):
        books = Books()
        documents = [{**books.doc, "id": "Fixture_" + str(i)} for i in range(3)]
        books.search = lambda query: documents
        class Review(Agents):
            def call(self, role, p, mode):
                output, usage = super().call(role, p, mode)
                if role == REASONER and output.get("decision") == "retrieve":
                    output["followup_request"]["evidence_ids"] = ["Fixture_0"]
                    output["gaps"] = ["missing A", "missing B"]
                    output["conflicts"] = [{"evidence_ids": ["Fixture_1"], "status": "resolved", "reason": "synthetic context retained"}]
                return output, usage
        agents = Review(verify=["retrieve", "answer"])
        q = Question("q", "Original question including NOT and time", {k: k for k in "ABCD"})
        result = Runtime(agents, books).run(q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["model_calls"], 5)
        initial, followup = [p for role, p, _ in agents.calls if role == INVESTIGATOR]
        self.assertEqual(initial["evidence_pool"], [])
        self.assertEqual(followup["question"], q.question)
        self.assertEqual(followup["options"], q.options)
        self.assertTrue(followup["feedback"]["basis"])
        self.assertEqual({i for d in followup["evidence_pool"] for i in [d["id"], *d.get("aliases", [])]}, {"Fixture_0", "Fixture_1"})
        self.assertEqual(len(followup["evidence_index"]), 3)
        self.assertIn("missing A", followup["gaps"])
        self.assertIn("missing B", followup["gaps"])
        self.assertEqual(len(followup["conflicts"]), 1)

    def test_three_call_budget_and_system_bytes_are_accounted(self):
        agents = Agents()
        q = Question("q", "fixture", {k: k for k in "ABCD"})
        result = Runtime(agents, Books(), max_calls=3).run(q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["model_calls"], 3)
        self.assertEqual(result["metrics"]["request_bytes"], sum(len(cli_payload(p, mode).encode()) for _, p, mode in agents.calls))
        self.assertGreater(result["metrics"]["system_bytes"], 0)

    def test_reasoner_can_unlock_a_dependent_followup_without_investigator_summary(self):
        class Dependent(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == "Planner":
                    data["knowledge_tasks"].append({"id": "K2", "goal": "dependent fixture need", "condition_ids": ["P1"], "depends_on": ["K1"]})
                elif role == INVESTIGATOR and p["feedback"].get("gap"):
                    for query in data["search"]["queries"]:
                        query["knowledge_task_ids"] = ["K2"]
                elif role == REASONER and data["decision"] == "retrieve":
                    data["followup_request"]["knowledge_task_ids"] = ["K2"]
                    data["findings"] = [{"id": "F1", "claim": "observed fixture prerequisite",
                        "knowledge_task_ids": ["K1"], "evidence_ids": [p["evidence_pool"][0]["id"]]}]
                elif role == REASONER and data["decision"] == "answer":
                    data["claims"][0]["knowledge_task_ids"] = ["K1", "K2"]
                return data, usage
        q = Question("q", "fixture", {k: k for k in "ABCD"})
        result = Runtime(Dependent(verify=["retrieve", "answer"]), Books()).run(q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["model_calls"], 5)

    def test_unknown_usage_on_final_answer_does_not_create_memory(self):
        class Memory:
            called = False
            def recall(self, *args, **kwargs):
                return []
            def put(self, *args, **kwargs):
                self.called = True
        class EstimatedAnswer(Agents):
            def call(self, role, *args):
                data, usage = super().call(role, *args)
                if role == REASONER:
                    usage["source"] = "estimated"
                return data, usage
        memory = Memory()
        q = Question("q", "fixture", {k: k for k in "ABCD"})
        result = Runtime(EstimatedAnswer(), Books(), memory=memory, embedder=Embeddings()).run(q)
        self.assertEqual(result["status"], "usage_unavailable")
        self.assertFalse(memory.called)

    def test_four_conditions_share_frozen_seed_and_reject_overlapping_questions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark.json"
            benchmark.write_text(json.dumps({"medqa": {key: {"question": "fixture question", "options": {k: k for k in "ABCD"}, "answer": "A"}
                                                       for key in ("warm1", "warm2", "eval1", "eval2")}}))
            sha = cli.Dataset(benchmark).sha256
            for stage, ids in (("warm", ["warm1", "warm2"]), ("eval", ["eval1", "eval2"])):
                (root / (stage + ".json")).write_text(json.dumps({"dataset_sha256": sha, "ids": ids}))
            class Reported(Agents):
                def call(self, *args):
                    data, usage = super().call(*args)
                    usage["source"] = "model"  # synthetic provider fixture; no real API call
                    return data, usage
            def run(argv):
                books = Books()
                books.warmup = books.close = lambda: None
                embedding = SimpleNamespace(_model=object(), cache_key="fixture-model", cache_store=None, embed_text=Embeddings().embed_text)
                with patch("sys.argv", ["medqa", *argv, "--output-policy", "strict"]), patch.object(cli, "books", return_value=books), \
                     patch.object(cli, "create_model", return_value=(Reported(), {"model": "fixture"})), \
                     patch("LangChain.memory.embedding.QwenEmbeddingService", return_value=embedding), patch("builtins.print"):
                    cli.main()
            common = ["--benchmark", str(benchmark), "--split", "all", "--memory-namespace", "shared", "--all", "--max-total-tokens", "10000"]
            warm, evaluation = root / "warm", root / "evaluation"
            run(["run", *common, "--ids", str(root / "warm.json"), "--output", str(warm)])
            source = warm / "structured_on.sqlite3"
            before = source.read_bytes()
            run(["compare", *common, "--ids", str(root / "eval.json"), "--output", str(evaluation),
                 "--memory-source", str(source), "--memory-write", "off"])
            report = json.loads((evaluation / "report.json").read_text())
            comparison = json.loads((evaluation / "comparison.json").read_text())
            self.assertEqual(len(report), 4)
            self.assertTrue(comparison["matched_complete"])
            self.assertTrue(comparison["token_comparison_valid"])
            self.assertIn("memory_text", comparison)
            for key, stats in report.items():
                self.assertEqual(stats["memory_written"], 0)
                self.assertEqual(stats["questions_completed"], 2)
                if key.endswith("_on"):
                    self.assertEqual(stats["memory_used"], 2)
                    with sqlite3.connect(evaluation / (key + ".sqlite3")) as db:
                        self.assertEqual(db.execute("SELECT count(*) FROM memory").fetchone()[0], 1)
                else:
                    self.assertEqual(stats["memory_candidates"], 0)
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "重叠"):
                validate_seed(source, dataset_sha256=sha, question_ids=["warm1"], namespace="shared",
                    task_group="medqa", corpus_version=Books.version, protocol=cli.PROTOCOL, source_hashes=cli.source_hashes())


def cli_payload(payload, mode):
    from ..medqa.runtime import compact, render_text
    return compact(payload) if mode == "structured" else render_text(payload)
