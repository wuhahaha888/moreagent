"""Bounded-path regression cases. Synthetic providers only; no medical/API claims."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from .test_medqa import Agents, Books, Embeddings
from .test_medqa_checkpoint import PowerCut
from ..agents.medqa_prompts import PLANNER, INVESTIGATOR, REASONER
from ..medqa.runtime import Runtime, ChatRoles
from ..medqa.data import Question
from ..medqa.retrieval import digest
from ..medqa.memory import SharedMemory
from ..medqa.checkpoint import Checkpoint, DurableCalls, DurableTools, PendingMemory
from ..medqa.results import split_result
from ..medqa.experiment import validate_seed
from ..medqa.cli import copy_memory_snapshot, summarize


def query(text, ident="Q1", **extra):
    return {"id": ident, "query": text, "knowledge_task_ids": ["K1"],
            "expected_information": "synthetic relation", **extra}


class PlanAgents(Agents):
    def __init__(self, queries=None, followup=False, break_update=None):
        super().__init__(verify=["answer", "answer"])
        self.queries = [query("initial fixture")] if queries is None else queries
        self.followup, self.break_update, self.reviews = followup, break_update, 0

    def call(self, role, payload, mode):
        data, usage = super().call(role, payload, mode)
        if role == INVESTIGATOR:
            data = {"question_id": payload["question_id"], "action": "tool_plan",
                    "search": {"gap": "synthetic missing relation", "queries": deepcopy(self.queries)}}
        elif role == REASONER:
            self.reviews += 1
            if self.reviews > 1:
                if self.break_update == "exception":
                    raise RuntimeError("synthetic provider failure")
                if self.break_update == "json":
                    return '{"broken":', usage
                if self.break_update == "option":
                    data["final_answer"]["answer"] = "INVALID"
            if self.followup:
                data["followup_request"] = {"gap": "synthetic decisive missing relation", "basis": "could change the choice",
                    "may_change_answer": True, "expected_information": "a distinct relation", "knowledge_task_ids": ["K1"],
                    "evidence_ids": [], "queries": ["new distinct fixture query"]}
        return data, usage


class NewBooks(Books):
    def search(self, text):
        docs = super().search(text)
        if docs and self.calls > 1:
            doc = docs[0]
            doc.update(id="New_" + str(self.calls), text="Distinct synthetic excerpt number " + str(self.calls) + ".")
            doc["sha256"] = digest(doc["text"])
        return docs


class GapMemory:
    def __init__(self, document, *, initially=False):
        self.document, self.initially = document, initially
        self.recalls, self.resolutions, self.writes = 0, [], []
        self.hit = {"id": "mem-gap", "summary": "A synthetic memory clue", "similarity": .9,
                    "content": {k: v for k, v in document.items() if k != "text"}}

    def recall(self, *args, **kwargs):
        self.recalls += 1
        return [self.hit] if self.initially or self.recalls > 1 else []

    def resolve(self, ids, *args, **kwargs):
        self.resolutions.append(list(ids))
        return [self.hit] if self.hit["id"] in ids else []

    def put(self, *args, **kwargs):
        self.writes.append((args, kwargs))


class BoundedPathTest(unittest.TestCase):
    def setUp(self):
        self.q = Question("q", "Synthetic NOT condition and time limit?", {k: k for k in "ABCD"})

    def run_case(self, agents=None, books=None, **kwargs):
        kwargs.setdefault("max_calls", 4)
        trace = kwargs.pop("trace_id", None)
        return Runtime(agents or PlanAgents(), books or Books(), output_policy="decoupled", **kwargs).run(self.q, trace_id=trace)

    def test_memory_id_is_resolved_before_investigator_and_does_not_replace_search(self):
        books, agents = Books(), PlanAgents()
        memory = GapMemory(books.doc, initially=True)
        result = self.run_case(agents, books, memory=memory, embedder=Embeddings(), memory_write=False)
        self.assertEqual([r for r, _, _ in agents.calls], [PLANNER, INVESTIGATOR, REASONER])
        investigator = agents.calls[1][1]
        self.assertEqual(memory.resolutions, [["mem-gap"]])
        self.assertEqual(investigator["memory_hits"], [{"id": "mem-gap", "source_id": "Fixture_0",
            "status": "source_checked_not_semantically_verified"}])
        self.assertTrue(investigator["evidence_pool"])
        self.assertEqual(books.calls, 1)
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["metrics"]["contract_repairs"], 0)

    def test_missing_memory_id_still_searches_without_extra_agent_call(self):
        books = Books()
        memory = GapMemory(books.doc, initially=True)
        memory.resolve = lambda *a, **k: []
        result = self.run_case(memory=memory, embedder=Embeddings(), memory_write=False)
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["metrics"]["model_calls"], 3)
        self.assertEqual(result["metrics"]["retrieval_count"], 1)
        self.assertEqual(result["metrics"]["source_read_errors"], 1)

    def test_empty_initial_query_runs_backup_without_another_agent(self):
        books = Books(empty_until=1)
        agents = PlanAgents([query("first"), query("backup", "Q2", run_if="empty_or_duplicate", after="Q1")])
        result = self.run_case(agents, books)
        self.assertEqual((books.calls, result["metrics"]["model_calls"]), (2, 3))
        self.assertEqual(result["answer"], "A")

    def test_successful_query_skips_conditional_backup(self):
        books = Books()
        agents = PlanAgents([query("first"), query("backup", "Q2", run_if="empty_or_duplicate", after="Q1")])
        result = self.run_case(agents, books)
        self.assertEqual(books.calls, 1)
        self.assertEqual(result["metrics"]["queries_filtered"], 1)

    def test_distinct_need_queries_can_run_in_one_tool_plan(self):
        books = Books()
        result = self.run_case(PlanAgents([query("first"), query("complementary", "Q2")]), books)
        self.assertEqual((books.calls, result["metrics"]["model_calls"]), (2, 3))

    def test_exact_duplicate_query_reuses_results(self):
        books = Books()
        result = self.run_case(PlanAgents([query("first"), query(" FIRST ", "Q2")]), books)
        self.assertEqual(books.calls, 1)
        self.assertEqual(result["metrics"]["queries_reused"], 1)

    def test_one_optional_followup_uses_four_calls_without_investigator_reentry(self):
        agents = PlanAgents(followup=True)
        result = self.run_case(agents, NewBooks())
        self.assertEqual([r for r, _, _ in agents.calls], [PLANNER, INVESTIGATOR, REASONER, REASONER])
        self.assertEqual(result["followup_result"]["status"], "updated")
        self.assertEqual(result["metrics"]["followup_rounds"], 1)
        self.assertEqual(result["metrics"]["total_tokens"], 60)

    def test_followup_budget_does_not_erase_current_answer(self):
        result = self.run_case(PlanAgents(followup=True), max_calls=3)
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["metrics"]["model_calls"], 3)
        self.assertEqual(result["followup_result"]["reason"], "answer_update_budget_unavailable")

    def test_empty_or_duplicate_followup_keeps_answer_without_update_call(self):
        for books in (Books(), Books(empty_until=99)):
            result = self.run_case(PlanAgents(followup=True), books)
            self.assertEqual(result["answer"], "A")
            self.assertEqual(result["metrics"]["model_calls"], 3)
            self.assertEqual(result["followup_result"]["status"], "answer_preserved")

    def test_new_id_with_identical_content_does_not_trigger_another_reasoner_call(self):
        class AliasBooks(Books):
            def search(self, text):
                doc = super().search(text)[0]
                doc["id"] = "Alias_" + str(self.calls)
                return [doc]
        result = self.run_case(PlanAgents(followup=True), AliasBooks())
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["metrics"]["model_calls"], 3)

    def test_broken_followup_update_keeps_answer_and_all_known_costs(self):
        for error in ("json", "option", "exception"):
            result = self.run_case(PlanAgents(followup=True, break_update=error), NewBooks())
            self.assertEqual(result["answer"], "A", error)
            self.assertTrue(result["answer_result"]["complete"])
            self.assertEqual(result["metrics"]["model_calls"], 4)
            self.assertEqual(result["metrics"]["total_tokens"], 45 if error == "exception" else 60)
            self.assertEqual(result["metrics"]["usage_missing_calls"], int(error == "exception"))
            self.assertFalse(result["protocol_result"]["blocks_answer"])

    def test_followup_requeries_memory_for_gap_before_search(self):
        books = NewBooks()
        memory_doc = {**books.doc, "id": "Memory_source", "text": "A distinct synthetic gap memory excerpt."}
        memory_doc["sha256"] = digest(memory_doc["text"])
        memory = GapMemory(memory_doc)
        original_read = books.read
        books.read = lambda ident: dict(memory_doc) if ident == memory_doc["id"] else original_read(ident)
        agents = PlanAgents(followup=True)
        result = self.run_case(agents, books, memory=memory, embedder=Embeddings(), memory_write=False)
        self.assertEqual(memory.recalls, 2)
        self.assertEqual(memory.resolutions, [["mem-gap"]])
        gap_event = next(i for i, e in enumerate(result["events"]) if e.get("tool") == "memory_prefetch" and e.get("stage") == "followup")
        followup_search = max(i for i, e in enumerate(result["events"]) if e.get("tool") == "search_textbooks")
        self.assertLess(gap_event, followup_search)
        self.assertTrue(any(h["id"] == "mem-gap" for h in agents.calls[-1][1]["memory_hits"]))
        self.assertEqual(result["metrics"]["model_calls"], 4)

    def test_memory_only_followup_is_available_when_search_budget_is_spent(self):
        books = Books()
        memory_doc = {**books.doc, "id": "Memory_source", "text": "Another synthetic memory relation."}
        memory_doc["sha256"] = digest(memory_doc["text"])
        memory = GapMemory(memory_doc)
        books.read = lambda ident: dict(memory_doc)
        result = self.run_case(PlanAgents(followup=True), books, memory=memory, embedder=Embeddings(), max_searches=1, memory_write=False)
        self.assertEqual(memory.recalls, 2)
        self.assertEqual(result["metrics"]["model_calls"], 4)
        self.assertEqual(books.calls, 1)

    def test_unverified_and_insufficient_are_separate_attributes(self):
        output = {"final_answer": {"answer": "A", "evidence_status": "insufficient"}}
        result = split_result(output, self.q, {}, [], {}, {})
        self.assertEqual(result["evidence_result"]["verification_status"], "unverified")
        self.assertEqual(result["evidence_result"]["support_assessment"]["status"], "insufficient")
        self.assertFalse(result["evidence_result"]["memory_eligible"])

    def test_processor_switch_changes_views_without_changing_roles(self):
        class DuplicateBooks(Books):
            def search(self, text):
                doc = super().search(text)[0]
                return [doc, {**doc, "id": "Alias"}]
        for enabled, expected in ((True, 1), (False, 2)):
            agents = PlanAgents()
            result = self.run_case(agents, DuplicateBooks(), evidence_processor=enabled)
            self.assertEqual(len(agents.calls[-1][1]["evidence_pool"]), expected)
            self.assertEqual(result["metrics"]["model_calls"], 3)

    def test_full_accumulation_prompts_keep_audit_without_old_retrieve_contract(self):
        roles = ChatRoles(None, profile="full", output_policy="decoupled")
        self.assertIn("condition_checks", roles.prompts[REASONER])
        self.assertNotIn("retrieve时final_answer=null", roles.prompts[REASONER])
        self.assertIn("tool_plan", roles.prompts[INVESTIGATOR])

    def test_cli_defaults_to_decoupled_path(self):
        from ..medqa.cli import make_parser
        self.assertEqual(make_parser().parse_args(["audit"]).output_policy, "decoupled")

    def test_durable_replay_after_plan_search_commit_does_not_repeat_search(self):
        with tempfile.TemporaryDirectory() as root:
            with Checkpoint(Path(root)) as cp:
                books, agents = NewBooks(), PlanAgents()
                original_execute = DurableTools.execute
                def cut_after_commit(journal, kind, request, callback):
                    result = original_execute(journal, kind, request, callback)
                    if kind == "search_textbooks":
                        raise PowerCut()
                    return result
                with patch.object(DurableTools, "execute", cut_after_commit):
                    with self.assertRaises(PowerCut):
                        self.run_case(DurableCalls(agents, cp, "structured_off", 0), books,
                                      journal=DurableTools(cp, "structured_off", 0), trace_id="fixed")
                after_books, after_agents = NewBooks(), PlanAgents()
                result = self.run_case(DurableCalls(after_agents, cp, "structured_off", 0), after_books,
                                      journal=DurableTools(cp, "structured_off", 0), trace_id="fixed")
                self.assertEqual(after_books.calls, 0)
                self.assertEqual(len(after_agents.calls), 1)
                self.assertEqual(result["metrics"]["total_tokens"], 45)


class SharedReferenceTest(unittest.TestCase):
    def test_id_resolution_is_scoped_and_pending_wrapper_forwards_it(self):
        with tempfile.TemporaryDirectory() as root:
            with_service = SharedMemory(Path(root) / "memory.sqlite3", "group", "fixture-model")
            try:
                ident = with_service.put(Books().doc, [1., 0.], "old", task_group="g")
                records = PendingMemory(with_service).resolve([ident], "new", Books.version, task_group="g")
                self.assertEqual(records[0]["id"], ident)
                self.assertNotIn("text", records[0]["content"])
                self.assertEqual(with_service.resolve([ident], "new", "changed", task_group="g"), [])
                self.assertEqual(with_service.resolve([ident], "old", Books.version, task_group="g"), [])
                self.assertEqual(with_service.resolve([ident], "new", Books.version, task_group="other"), [])
                self.assertEqual(with_service.resolve(["unknown-id"], "new", Books.version, task_group="g"), [])
                with sqlite3.connect(Path(root) / "memory.sqlite3") as db:
                    db.execute("UPDATE memory SET expires_at=1 WHERE id=?", (ident,))
                self.assertEqual(with_service.resolve([ident], "new", Books.version, task_group="g"), [])
            finally:
                with_service.close()


class PartialSeedTest(unittest.TestCase):
    def make_seed(self, root):
        from ..memory.advanced import SQLiteMemoryStore
        manifest = {"dataset_sha256": "dataset", "textbooks_version": Books.version,
                    "runtime_protocol": "moreagent.medqa/baseline-2", "source_sha256": {"test": "fixture"},
                    "question_ids": ["warm-good", "warm-pending"],
                    "settings": {"task_group": "g", "memory_namespace": "shared"}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "report.json").write_text(json.dumps({"structured_on": {
            "questions_completed": 1, "runtime_failed_questions": 0, "memory_degraded_questions": 0,
            "token_usage_complete": True, "total_tokens": 45}}))
        store = SQLiteMemoryStore(root / "structured_on.sqlite3", "shared", "shared", "fixture")
        ids = []
        for trace, text in (("good-trace", "Synthetic good source."), ("pending-trace", "Synthetic pending source.")):
            ids.append(store.put(role="VerifierReasoner", topic="fixture", summary=text,
                content={"id": trace, "text": text, "corpus_version": Books.version, "task_group": "g"},
                embedding=[1., 0.], trace_id=trace, kind="textbook_extract", status="curated"))
        with Checkpoint(root) as cp:
            good = cp.begin("structured_on", 0, "warm-good")
            result = {"answer": "A", "evidence_status": "supported", "metrics": {"usage_sources": ["model"]}}
            cp.db.execute("UPDATE questions SET trace=? WHERE position=0", ("good-trace",))
            cp.db.commit()
            cp.save_result("structured_on", 0, result, [])
            cp.apply_memory("structured_on", None)
            cp.begin("structured_on", 1, "warm-pending")
            cp.db.execute("UPDATE questions SET trace=? WHERE position=1", ("pending-trace",))
            cp.db.execute("INSERT INTO calls(condition,position,ordinal,attempt,request_hash,role,state) "
                          "VALUES ('structured_on',1,1,1,'fixture','Planner','pending')")
            cp.db.commit()
        return root / "structured_on.sqlite3", ids

    def test_partial_seed_filters_pending_provenance_and_reports_unknown_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, ids = self.make_seed(root)
            before = source.read_bytes()
            args = dict(dataset_sha256="dataset", question_ids=["eval"], namespace="shared", task_group="g",
                        corpus_version=Books.version, protocol="moreagent.medqa/baseline-2", source_hashes={"test": "fixture"})
            with self.assertRaises(ValueError):
                validate_seed(source, **args)
            seed = validate_seed(source, **args, allow_partial=True)
            self.assertEqual(seed["memory_ids"], ids[:1])
            self.assertTrue(seed["partial_source"])
            self.assertEqual(seed["excluded_memory_records"], 1)
            self.assertEqual(seed["accumulation_unknown_calls"], 1)
            self.assertFalse(seed["accumulation_cost_complete"])
            target = root / "frozen.sqlite3"
            copy_memory_snapshot(source, target, memory_ids=seed["memory_ids"])
            with sqlite3.connect(target) as db:
                self.assertEqual(db.execute("SELECT id FROM memory").fetchall(), [(ids[0],)])
                self.assertEqual(db.execute("SELECT id FROM memory_fts").fetchall(), [(ids[0],)])
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "重叠"):
                validate_seed(source, **{**args, "question_ids": ["warm-pending"]}, allow_partial=True)

    def test_partial_seed_cannot_promote_unverified_whole_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.make_seed(root)
            with sqlite3.connect(root / "checkpoint.sqlite3") as db:
                db.execute("UPDATE questions SET result=? WHERE position=0", (json.dumps({
                    "answer": "A", "evidence_status": "unverified", "metrics": {"usage_sources": ["model"]}}),))
            with self.assertRaisesRegex(ValueError, "没有未过期"):
                validate_seed(source, dataset_sha256="dataset", question_ids=["eval"], namespace="shared", task_group="g",
                    corpus_version=Books.version, protocol="moreagent.medqa/baseline-2", source_hashes={"test": "fixture"}, allow_partial=True)

