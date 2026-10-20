"""三Agent机制集成测试；替身输出不用于报告医学准确率或真实Token收益。"""
import json
import tempfile
import unittest
from pathlib import Path

from ..agents.medqa_prompts import PLANNER, INVESTIGATOR, REASONER, ROLES
from ..medqa.data import Dataset, Question
from ..medqa.memory import SharedMemory
from ..medqa.retrieval import digest, Textbooks
from ..medqa.runtime import Runtime


class Embeddings:
    def embed_text(self, text):
        return [1., 0., 0., 0.]


class Books:
    version = "fixture-v1"

    def __init__(self, empty_until=0, errors=False):
        self.calls, self.empty_until, self.errors = 0, empty_until, errors
        self.stats = {"searches": 0, "source_reads": 0, "reranks": 0}
        self.doc = {"id": "Fixture_0", "title": "Fixture", "text": "Fictional fixture evidence, not a medical fact.",
                    "corpus_version": self.version}
        self.doc["sha256"] = digest(self.doc["text"])

    def search(self, query):
        self.calls += 1
        self.stats["searches"] += 1
        if self.errors:
            raise RuntimeError("fixture")
        return [] if self.calls <= self.empty_until else [dict(self.doc)]

    def read(self, ident):
        self.stats["source_reads"] += 1
        if ident != self.doc["id"]:
            raise ValueError("unknown")
        return dict(self.doc)


class Agents:
    """只根据输入的真实工具观测行动，不访问评分答案。"""
    def __init__(self, verify=None, illegal=False):
        self.verify, self.illegal, self.calls = list(verify or []), illegal, []

    def call(self, role, p, mode):
        self.calls.append((role, p, mode))
        data = {"question_id": p["question_id"]}
        if role == PLANNER:
            data.update(key_conditions=[{"id": "P1", "text": "synthetic condition"}],
                        knowledge_tasks=[{"id": "K1", "goal": "check fixture", "condition_ids": ["P1"], "depends_on": []}],
                        criteria=["source support"], memory_assessments=[{
                            "memory_id": m["id"], "decision": "read", "reason": "check source before using", "knowledge_task_ids": ["K1"]}
                            for m in p["memory_hits"]])
        elif role == INVESTIGATOR:
            evidence = p["evidence_pool"]
            unread = [m["id"] for m in p["memory_hits"] if m["status"] == "candidate_needs_source_read"]
            if not evidence and unread and p["budget"]["document_reads_remaining"]:
                data.update(action="read_memory", read_memory={"memory_ids": unread[:p["budget"]["document_reads_remaining"]]})
            elif p["budget"]["search_calls_remaining"] and (not evidence or p["feedback"].get("gap")):
                followup = bool(p["attempted_queries"])
                data.update(action="search", search={"gap": "synthetic critical gap", "queries": [{
                    "query": "fixture query " + str(len(p["attempted_queries"]) + i),
                    "knowledge_task_ids": ["K1"], "expected_information": "synthetic support"}
                    for i in range(2 if followup else 1)]})
            else:
                data.update(action="finish", findings=[{"id": "F1", "claim": "synthetic observed finding",
                    "evidence_ids": [evidence[0]["id"]], "knowledge_task_ids": ["K1"]}] if evidence else [],
                    gaps=[] if evidence else ["synthetic support missing"])
        elif role == REASONER:
            decision = self.verify.pop(0) if self.verify else ("answer" if p["evidence_pool"] else "retrieve" if p["budget"]["can_investigate"] else "answer")
            if decision == "pass":
                decision = "answer"
            data["decision"] = decision
            if decision == "retrieve":
                data.update(followup_request={"gap": "synthetic missing decisive condition", "expected_information": "new supporting passage",
                    "knowledge_task_ids": ["K1"], "basis": "synthetic evidence is missing or incomplete",
                    "evidence_ids": [d["id"] for d in p["evidence_pool"]], "queries": ["synthetic followup"]}, final_answer=None)
            elif decision == "answer":
                doc = p["evidence_pool"][0] if p["evidence_pool"] else None
                ident = ("INVENTED" if self.illegal else doc["id"]) if doc else None
                memory_sources = {m["source_id"] for m in p["memory_hits"] if m["status"] == "source_checked_not_semantically_verified"}
                basis = (["memory"] if ident in memory_sources else ["textbook"]) if doc else ["model_knowledge"]
                sources = [ident] if doc else []
                data.update(claims=[{"id": "C1", "claim": "synthetic claim", "knowledge_task_ids": ["K1"],
                    "basis": basis, "evidence_ids": sources,
                    "support_quotes": [{"evidence_id": ident, "quote": doc["text"]}] if doc else [], "question_quotes": []}],
                    checks=[{"claim_id": "C1", "verdict": "supported" if doc else "unsupported", "reason": "synthetic support", "evidence_ids": sources}],
                    condition_checks=[{"condition_id": "P1", "claim_ids": ["C1"], "verdict": "matched" if doc else "unclear", "reason": "synthetic applicable condition"}],
                    option_comparison={"alternative": "B", "claim_ids": ["C1"], "verdict": "supported" if doc else "unsupported", "reason": "synthetic option distinction"},
                    critical_gaps=[] if doc else ["synthetic external support missing"],
                    conflicts=[], final_answer={"answer": "A", "evidence_status": "supported" if doc else "insufficient",
                        "answer_basis": basis, "summary": "fixture only", "evidence_ids": sources})
            else:
                data.update(gaps=["synthetic input ambiguous"], abstain_reason="synthetic candidates cannot be compared", final_answer=None)

        else:
            raise AssertionError("unexpected fourth role")
        return data, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                      "source": "fixture", "response_model": "fixture"}


class MedQATest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.q = Question("q1", "A synthetic mechanism fixture?", {k: k for k in "ABCD"})

    def memory(self, namespace="a"):
        service = SharedMemory(Path(self.temp.name) / "memory.sqlite3", namespace, "fixture-model")
        self.addCleanup(service.close)
        return service

    def test_three_roles_and_one_initial_search(self):
        agents, books = Agents(), Books()
        result = Runtime(agents, books).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["visited_agents"], sorted(ROLES))
        self.assertEqual([r for r, _, _ in agents.calls], [PLANNER, INVESTIGATOR, REASONER])
        self.assertEqual(books.calls, 1)
        self.assertEqual(result["metrics"]["total_tokens"], 45)
        self.assertEqual(result["metrics"]["followup_rounds"], 0)
        self.assertEqual(result["verification"]["review_type"], "same_agent_reasoning_and_verification")

    def test_one_followup_batch_two_queries_and_no_fourth_search(self):
        books = Books(empty_until=100)
        result = Runtime(Agents(), books, max_searches=3, max_stalled_rounds=3).run(self.q)
        self.assertEqual(books.calls, 3)
        self.assertEqual(result["metrics"]["followup_rounds"], 1)
        self.assertEqual(result["metrics"]["search_empty"], 3)
        self.assertEqual(result["status"], "answered_unsupported")
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["evidence_status"], "insufficient")

    def test_failed_searches_consume_same_budget(self):
        result = Runtime(Agents(), Books(errors=True), max_searches=3, max_stalled_rounds=3).run(self.q)
        self.assertEqual(result["metrics"]["retrieval_count"], 3)
        self.assertEqual(result["metrics"]["search_errors"], 3)
        self.assertEqual(result["metrics"]["followup_rounds"], 1)

    def test_third_search_can_still_answer(self):
        result = Runtime(Agents(), Books(empty_until=2), max_stalled_rounds=3).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["retrieval_count"], 3)

    def test_reasoner_can_request_multiple_followups_without_losing_answer(self):
        agents, books = Agents(verify=["retrieve", "answer"]), Books()
        result = Runtime(agents, books).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["followup_rounds"], 1)
        self.assertEqual(result["metrics"]["model_calls"], 5)
        self.assertEqual(books.calls, 3)
        again = Runtime(Agents(verify=["retrieve", "retrieve"]), Books(), max_stalled_rounds=4).run(self.q)
        self.assertEqual(again["status"], "supported")
        self.assertEqual(again["metrics"]["retrieval_count"], 5)
        self.assertEqual(again["metrics"]["followup_rounds"], 2)

    def test_fabricated_evidence_fails_closed(self):
        result = Runtime(Agents(illegal=True), Books()).run(self.q)
        self.assertEqual(result["status"], "protocol_error")
        self.assertIsNone(result["answer"])

    def test_prefetch_precedes_planner_and_raw_memory_requires_read(self):
        memory, books = self.memory(), Books()
        runtime = Runtime(Agents(), books, memory=memory, embedder=Embeddings())
        runtime.run(self.q)
        second = runtime.run(Question("q2", self.q.question, self.q.options))
        self.assertEqual(second["events"][0]["tool"], "memory_prefetch")
        planner = next(e for e in second["events"] if e.get("agent") == PLANNER)
        self.assertEqual(planner["input"]["memory_hits"][0]["status"], "candidate_needs_source_read")
        self.assertNotIn("text", planner["input"]["memory_hits"][0])
        self.assertEqual(second["metrics"]["memory_used"], 1)
        self.assertEqual(second["metrics"]["retrieval_count"], 0)
        self.assertEqual(second["metrics"]["document_reads"], 1)
        self.assertEqual(second["metrics"]["state_bytes"], 32)
        self.assertEqual(second["metrics"]["state_transfers"], 2)

    def test_memory_scope_version_group_and_similarity(self):
        memory, books = self.memory(), Books()
        memory.put(books.doc, [1., 0., 0., 0.], "old", task_group="g1")
        self.assertEqual(memory.recall([1., 0., 0., 0.], "old", books.version), [])
        self.assertEqual(memory.recall([1., 0., 0., 0.], "new", "changed"), [])
        self.assertEqual(memory.recall([0., 1., 0., 0.], "new", books.version), [])
        self.assertEqual(memory.recall([1., 0.], "new", books.version), [])
        self.assertEqual(memory.recall([1., 0., 0., 0.], "new", books.version, task_group="g2"), [])
        self.assertEqual(len(memory.recall([1., 0., 0., 0.], "new", books.version, task_group="g1")), 1)
        other = self.memory("other")
        self.assertEqual(other.recall([1., 0., 0., 0.], "new", books.version), [])

    def test_memory_write_preserves_conditions_and_deduplicates_identical_context(self):
        memory, books = self.memory(), Books()
        first = memory.put(books.doc, [1., 0., 0., 0.], "old", task_group="g", conditions=["original scope"], retrieval_queries=["query1"])
        second = memory.put(books.doc, [1., 0., 0., 0.], "later", task_group="g", conditions=["original scope"], retrieval_queries=["query2"])
        self.assertEqual(first, second)
        hit = memory.recall([1., 0., 0., 0.], "now", books.version, task_group="g")[0]
        self.assertEqual(hit["content"]["conditions"], ["original scope"])
        self.assertNotIn("text", hit["content"])
        self.assertIn("applicability_requires", hit["content"]["validation"])

    def test_modified_source_cannot_be_reused(self):
        memory, books = self.memory(), Books()
        memory.put(books.doc, [1., 0., 0., 0.], "old", task_group="medqa")
        books.doc["text"] = "A new synthetic source version for hash mismatch testing."
        books.doc["sha256"] = digest(books.doc["text"])
        result = Runtime(Agents(), books, memory=memory, embedder=Embeddings()).run(self.q)
        self.assertEqual(result["metrics"]["memory_candidates"], 1)
        self.assertEqual(result["metrics"]["memory_source_valid"], 0)
        self.assertEqual(result["metrics"]["memory_used"], 0)

    def test_dataset_gold_is_only_available_to_grader(self):
        path = Path(self.temp.name) / "benchmark.json"
        path.write_text(json.dumps({"medqa": {"q1": {"question": self.q.question, "options": self.q.options, "answer": "D"}}}))
        dataset, agents = Dataset(path), Agents()
        result = Runtime(agents, Books()).run(dataset.questions[0])
        self.assertFalse(dataset.grade(result))
        self.assertNotIn("answer", dataset.questions[0].public())
        self.assertTrue(all("gold" not in p and "answer" not in p for _, p, _ in agents.calls))

    def test_two_groups_ten_rounds_both_modes_same_semantics(self):
        for group in ("g1", "g2"):
            for mode in ("text", "structured"):
                memory, books = self.memory(group + mode), Books()
                runtime = Runtime(Agents(), books, memory=memory, embedder=Embeddings(), mode=mode, task_group=group)
                results = [runtime.run(Question(str(i), self.q.question, self.q.options)) for i in range(10)]
                self.assertTrue(all(r["status"] == "supported" for r in results))
                self.assertEqual(books.calls, 1)
                self.assertEqual(sum(r["metrics"]["memory_used"] for r in results), 9)

    def test_hyphenated_source_ids_and_path_traversal(self):
        root = Path(self.temp.name)
        (root / "chunk").mkdir()
        (root / "index/bm25").mkdir(parents=True)
        row = {"id": "Psichiatry_DSM-5_0", "title": "fixture", "content": "fixture"}
        (root / "chunk/Psichiatry_DSM-5.jsonl").write_text(json.dumps(row) + "\n")
        books = Textbooks(root, root / "python", root / "java", root / "ranker")
        self.assertEqual(books.read(row["id"])["text"], "fixture")
        with self.assertRaises(ValueError):
            books.read("../outside_0")

    def test_provider_respects_output_limit_deadline_and_thinking(self):
        from types import SimpleNamespace
        from langchain_core.messages import HumanMessage
        from ..medqa.model import CompatibleChat
        sent = []
        def create(**kwargs):
            sent.append(kwargs)
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        for base in ("https://api.deepseek.com/v1", "http://localhost:8000/v1"):
            chat = CompatibleChat(model="fixture", base_url=base, api_key="fixture", max_tokens=321, client=client)
            chat.invoke_bounded([HumanMessage(content="fixture")], remaining_seconds=7, remaining_tokens=100)
        self.assertEqual(sent[0]["max_tokens"], 100)
        self.assertEqual(sent[0]["timeout"], 7)
        self.assertNotIn("max_completion_tokens", sent[0])
        self.assertEqual(sent[0]["extra_body"]["thinking"]["type"], "disabled")
        self.assertNotIn("extra_body", sent[1])

    def test_cli_comparison_requires_same_completed_stream(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from ..medqa import cli
        path = Path(self.temp.name) / "benchmark.json"
        path.write_text(json.dumps({"medqa": {str(i): {"question": self.q.question, "options": self.q.options, "answer": "A"} for i in range(2)}}))
        for budget, expected in ((10000, True), (100, False)):
            out, books = Path(self.temp.name) / str(budget), Books()
            books.warmup = books.close = lambda: None
            embedding = SimpleNamespace(_model=object(), cache_key="fixture-model", cache_store=None, embed_text=Embeddings().embed_text)
            argv = ["medqa", "compare", "--benchmark", str(path), "--output", str(out), "--limit", "2", "--max-total-tokens", str(budget)]
            with patch("sys.argv", argv), patch.object(cli, "books", return_value=books), \
                 patch.object(cli, "create_model", return_value=(Agents(), {"model": "fixture"})), \
                 patch("LangChain.memory.embedding.QwenEmbeddingService", return_value=embedding), patch("builtins.print"):
                cli.main()
            comparison = json.loads((out / "comparison.json").read_text())
            self.assertEqual(comparison["matched_complete"], expected)
            self.assertEqual("protocol_memory_off" in comparison, expected)


if __name__ == "__main__":
    unittest.main()
