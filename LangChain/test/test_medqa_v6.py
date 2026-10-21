"""v6的机制反例和恢复边界；合成材料不代表医学准确率。"""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from .test_medqa import Agents, Books
from .test_medqa_checkpoint import PowerCut
from ..agents.medqa_prompts import PLANNER, INVESTIGATOR, REASONER
from ..medqa.data import Question
from ..medqa.runtime import Runtime, compact, render_text
from ..medqa.evidence_processor import EvidenceProcessor, validate_visible_quotes, validate_view
from ..medqa.grounding import ContractError
from ..medqa.protocol import normalize_response
from ..medqa.task_state import TaskStates
from ..medqa.retrieval_policy import choose
from ..medqa.budget import SearchBudget
from ..medqa.retrieval import digest
from ..medqa.checkpoint import Checkpoint, CheckpointError, DurableTools, DurableCalls
from ..protocol.g2cp import build_control, build_precise_for_agent


def document(ident, text):
    return {"id": ident, "text": text, "title": ident, "sha256": digest(text), "corpus_version": Books.version}


class ProcessorTest(unittest.TestCase):
    def test_cross_id_exact_dedup_preserves_aliases_targets_and_originals(self):
        docs = {i: document(i, "Same extract with exact bytes, dose 10 mg.") for i in ("A", "B")}
        before = deepcopy(docs)
        result = EvidenceProcessor().process(docs, query="dose", targets={"A": ["K1"], "B": ["K2"]})
        self.assertEqual(docs, before)
        self.assertEqual(len(result["views"]), 1)
        self.assertEqual(result["views"][0]["aliases"], ["B"])
        self.assertEqual(result["candidate_coverage"], {"A": ["K1"], "B": ["K2"]})
        self.assertEqual(result["visible"]["A"], result["visible"]["B"])
        self.assertNotIn("coverage", result["views"][0])

    def test_negation_numbers_units_population_and_direction_are_retained(self):
        texts = ["Drug X is recommended in adults at 10 mg with food daily.",
                 "Drug X is not recommended in adults at 10 mg with food daily.",
                 "Drug X is recommended in children at 100 mg with food daily.",
                 "Drug X is recommended in adults at 10 mcg with food daily.",
                 "A increases B in adults at 10 mg with food daily.",
                 "B increases A in adults at 10 mg with food daily."]
        docs = {str(i): document(str(i), t) for i, t in enumerate(texts)}
        result = EvidenceProcessor().process(docs, query="Drug dose")
        self.assertEqual([v["text"] for v in result["views"]], texts)
        self.assertTrue(result["similar_groups"])
        self.assertTrue(all(g["action"] == "retained_both" for g in result["similar_groups"]))

    def test_long_document_extractive_offsets_stable_and_expandable(self):
        text = "Background sentence unrelated to target. " * 90 + "Target relation must be verified. Except in children. " + "Other context. " * 90
        docs = {"A": document("A", text)}
        ep = EvidenceProcessor(span_chars=500)
        first = ep.process(docs, query="target relation")
        second = ep.process(docs, query="target relation")
        self.assertEqual(first["views"], second["views"])
        self.assertTrue(first["views"][0]["has_more"])
        self.assertIn("Except in children.", first["views"][0]["text"])
        for span in first["visible"]["A"]:
            self.assertEqual(span["text"], text[span["start"]:span["end"]])
        full = ep.process(docs, query="target relation", expanded=["A"])
        self.assertEqual(full["views"][0]["text"], text)
        self.assertFalse(full["views"][0].get("has_more", False))

    def test_hidden_raw_quote_rejected_even_if_it_exists_in_full_source(self):
        text = "Target relation. " * 100 + "Hidden unrelated statement to check visibility."
        pack = EvidenceProcessor(span_chars=180).process({"A": document("A", text)}, query="target relation")
        output = {"claims": [{"support_quotes": [{"evidence_id": "A", "quote": "Hidden unrelated statement to check visibility."}]}]}
        with self.assertRaisesRegex(ContractError, "未在当前请求"):
            validate_visible_quotes(output, pack["visible"])

    def test_protected_quote_and_neighbor_kept_after_relevance_changes(self):
        quote = "Important exception is present in children only."
        text = "Unrelated material. " * 120 + quote + " Adjacent restriction must remain. " + "Target relation. " * 120
        pack = EvidenceProcessor(span_chars=300).process({"A": document("A", text)}, query="target relation", protected={"A": [quote]})
        self.assertIn(quote, pack["views"][0]["text"])
        self.assertIn("Adjacent restriction must remain.", pack["views"][0]["text"])

    def test_disabled_processor_preserves_full_content_and_distinct_sources(self):
        docs = {i: document(i, "Duplicate but full baseline evidence.") for i in ("A", "B")}
        result = EvidenceProcessor(enabled=False).process(docs, query="x")
        self.assertEqual(len(result["views"]), 2)
        self.assertEqual(result["view_chars"], result["raw_chars"])

    def test_integrity_mismatch_rejected_and_context_omission_explicit(self):
        docs = {"A": document("A", "first evidence"), "B": document("B", "second evidence")}
        result = EvidenceProcessor(max_documents=1).process(docs, query="x")
        self.assertEqual(result["omitted"], ["B"])
        docs["A"]["text"] = "modified bytes"
        with self.assertRaises(ValueError):
            EvidenceProcessor().process(docs, query="x")

    def test_restored_model_view_must_match_original_visible_spans(self):
        docs = {"A": document("A", "An immutable source excerpt to verify.")}
        pack = EvidenceProcessor().process(docs, query="source")
        validate_view(pack, docs)
        pack["views"][0]["text"] = "silently changed model input"
        with self.assertRaises(ContractError):
            validate_view(pack, docs)


class DynamicRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.q = Question("q", "Synthetic question with NOT, adult, 10 mg and time limit.", {i: i for i in "ABCD"})

    def test_actual_input_uses_precise_only_and_public_legacy_still_works(self):
        agents = Agents()
        result = Runtime(agents, Books()).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["model_calls"], 3)
        for role, payload, mode in agents.calls:
            self.assertFalse(set(payload) & {"control", "semantic", "operation_id", "task_vec", "trace_id"})
            self.assertEqual(payload["question"], self.q.question)
            self.assertEqual(payload["options"], self.q.options)
            self.assertTrue(all(not set(s) & {"can_start", "can_complete", "can_close"}
                                for s in payload.get("task_status", {}).values()))
            event = next(e for e in result["events"] if e.get("agent") == role)
            self.assertEqual(event['handoff_mode'],'ACP-1')
            self.assertNotIn('control',event)
        self.assertGreater(result['metrics']['acp_operations'],0)
        self.assertEqual(build_precise_for_agent("任务分配", question="legacy")["question"], "legacy")
        self.assertIn("资料搜索", build_control(trace_id="t", src="system", dst="任务分配")["allowed_next"])

    def test_knowledge_dependency_does_not_block_complete_query(self):
        class Dependent(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == PLANNER:
                    data["knowledge_tasks"].append({"id": "K2", "goal": "related knowledge", "condition_ids": ["P1"], "depends_on": ["K1"]})
                elif role == INVESTIGATOR:
                    data["search"]["queries"][0]["knowledge_task_ids"] = ["K1", "K2"]
                else:
                    data["claims"][0]["knowledge_task_ids"] = ["K1", "K2"]
                return data, usage
        books = Books()
        result = Runtime(Dependent(), books).run(self.q)
        self.assertEqual(books.calls, 1)
        self.assertEqual(result["status"], "supported")
        self.assertTrue(all(t["can_complete"] for t in result["task_results"].values()))

    def test_missing_real_input_does_not_block_another_valid_proposal(self):
        class Mixed(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == INVESTIGATOR:
                    blocked = deepcopy(data["search"]["queries"][0])
                    blocked.update(query="missing entity", required_inputs=["F_missing"])
                    data["search"]["queries"].insert(0, blocked)
                return data, usage
        books = Books()
        result = Runtime(Mixed(), books).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(books.calls, 1)
        self.assertTrue(any(e.get("decision") == "need_input" for e in result["events"]))

    def test_two_distinct_initial_queries_use_same_dynamic_policy(self):
        class Batch(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == INVESTIGATOR:
                    second = deepcopy(data["search"]["queries"][0])
                    second["query"] = "second specific relation"
                    data["search"]["queries"].append(second)
                return data, usage
        books = Books()
        result = Runtime(Batch(), books).run(self.q)
        self.assertEqual(books.calls, 2)
        self.assertEqual(result["metrics"]["model_calls"], 3)

    def test_existing_reasoner_review_filters_covered_target_without_extra_judge(self):
        class Reviewed(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == PLANNER:
                    data["knowledge_tasks"].append({"id": "K2", "goal": "other relation", "condition_ids": ["P1"], "depends_on": []})
                elif role == REASONER and data["decision"] == "retrieve":
                    doc = p["evidence_pool"][0]
                    data["need_reviews"] = [{"task_id": "K1", "verdict": "supported", "condition_verdict": "matched",
                        "support_quotes": [{"evidence_id": doc["id"], "quote": doc["text"]}], "reason": "synthetic supported relation"}]
                    data["followup_request"]["knowledge_task_ids"] = ["K2"]
                elif role == INVESTIGATOR and p["feedback"].get("gap"):
                    data["search"]["queries"][1]["knowledge_task_ids"] = ["K2"]
                elif role == REASONER and data["decision"] == "answer":
                    data["claims"][0]["knowledge_task_ids"] = ["K1", "K2"]
                return data, usage
        books = Books()
        result = Runtime(Reviewed(verify=["retrieve", "answer"]), books).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(books.calls, 2)
        self.assertEqual(result["metrics"]["queries_filtered"], 1)
        self.assertEqual(result["metrics"]["model_calls"], 5)

    def test_empty_searches_close_incomplete_needs_but_keep_best_effort_answer(self):
        result = Runtime(Agents(), Books(empty_until=99), max_stalled_rounds=2).run(self.q)
        self.assertEqual(result["status"], "answered_unsupported")
        self.assertEqual(result["task_results"]["K1"]["lifecycle"], "closed_incomplete")
        self.assertFalse(result["task_results"]["K1"]["can_complete"])
        self.assertEqual(result["metrics"]["retrieval_count"], 2)
        self.assertEqual(result["metrics"]["memory_written"], 0)

    def test_expand_then_answer_does_not_retrieve_again(self):
        books = Books()
        books.doc = document("Fixture_0", "Target evidence. " * 250)
        class Expand(Agents):
            def __init__(self):
                super().__init__()
                self.reviewed = 0
            def call(self, role, p, mode):
                if role == REASONER:
                    self.reviewed += 1
                    if self.reviewed == 1:
                        self.calls.append((role, p, mode))
                        return {"question_id": p["question_id"], "decision": "expand_evidence",
                                "expand_evidence": {"evidence_ids": ["Fixture_0"], "reason": "inspect omitted context"}}, {"source": "fixture", "total_tokens": 15}
                data, usage = super().call(role, p, mode)
                if role == REASONER:
                    data["claims"][0]["support_quotes"][0]["quote"] = p["evidence_pool"][0]["text"][:100]
                return data, usage
        result = Runtime(Expand(), books, evidence_span_chars=300).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["expansions"], 1)
        self.assertEqual(result["metrics"]["model_calls"], 4)
        self.assertEqual(books.calls, 1)

    def test_question_background_is_not_automatically_promoted_to_support(self):
        class Ambiguous(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == REASONER:
                    data["claims"][0]["question_quotes"] = [p["question"]]
                return data, usage
        result = Runtime(Ambiguous(), Books()).run(self.q)
        self.assertEqual(result["status"], "protocol_error")
        self.assertEqual(result["metrics"]["semantic_reviews"], 1)
        self.assertEqual(result["metrics"]["memory_written"], 0)
        self.assertTrue(any(e.get("category") == "attribution" for e in result["events"]))

    def test_partial_evidence_does_not_disable_local_field_repair(self):
        class Partial(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == REASONER:
                    data["checks"][0]["verdict"] = "partial"
                    data["option_comparison"]["verdict"] = "partial"
                    data["critical_gaps"] = ["unresolved decisive relation"]
                    data["final_answer"]["evidence_status"] = "insufficient"
                    if not p["feedback"].get("contract_error"):
                        del data["final_answer"]["summary"]
                return data, usage
        result = Runtime(Partial(), Books()).run(self.q)
        self.assertEqual(result["status"], "answered_unsupported")
        self.assertEqual(result["metrics"]["contract_repairs"], 1)
        self.assertEqual(result["metrics"]["recovered_action_errors"], 1)

    def test_unresolved_need_can_be_explicit_without_fabricating_supported_claim(self):
        class Unresolved(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == PLANNER:
                    data["knowledge_tasks"].append({"id": "K2", "goal": "unresolved relation", "condition_ids": ["P1"], "depends_on": []})
                elif role == REASONER:
                    data["unresolved_tasks"] = [{"task_id": "K2", "reason": "no applicable evidence"}]
                    data["critical_gaps"] = ["unresolved relation"]
                    data["final_answer"]["evidence_status"] = "partial"
                return data, usage
        result = Runtime(Unresolved(), Books()).run(self.q)
        self.assertEqual(result["status"], "answered_partial")
        self.assertEqual(result["task_results"]["K2"]["lifecycle"], "closed_incomplete")

    def test_normalizer_maps_only_unambiguous_fields_never_support_tags(self):
        raw = {"action": "search", "search": {"findings": [], "gaps": ["missing"], "used_memory_ids": []}}
        normalized, maps = normalize_response(INVESTIGATOR, raw)
        self.assertEqual(normalized["gaps"], ["missing"])
        self.assertEqual(len(maps), 3)
        self.assertNotIn("gaps", raw)
        raw["gaps"] = ["different"]
        with self.assertRaises(ContractError):
            normalize_response(INVESTIGATOR, raw)

    def test_modes_keep_business_observations_but_use_different_handoffs(self):
        a, b = Agents(), Agents()
        structured=Runtime(a, Books(), mode="structured").run(self.q)
        text=Runtime(b, Books(), mode="text").run(self.q)
        for (ra, pa, _), (rb, pb, _) in zip(a.calls, b.calls):
            self.assertEqual(ra, rb)
            self.assertEqual(pa,{k:v for k,v in pb.items() if k!='_natural_handoffs'})
        self.assertEqual(text['metrics']['acp_frames'],0)
        self.assertGreater(structured['metrics']['acp_frames'],0)
        inputs=[e['model_input'] for e in text['events'] if 'model_input' in e]
        self.assertTrue(all('key_conditions' not in text and 'evidence_pool' not in text for text in inputs))


class PolicyAndJournalTest(unittest.TestCase):
    def test_review_not_candidates_controls_coverage_and_can_reopen(self):
        tasks = TaskStates([{"id": "K1"}])
        tasks.candidate(["K1"], ["A"])
        self.assertFalse(tasks.items["K1"]["can_complete"])
        doc = document("A", "Synthetic applicable source excerpt.")
        reviews = [{"task_id": "K1", "verdict": "supported", "condition_verdict": "matched",
                    "support_quotes": [{"evidence_id": "A", "quote": doc["text"]}], "reason": "supports relation"}]
        tasks.review_needs(reviews, {"A": doc}, {"A": [{"text": doc["text"]}]})
        self.assertTrue(tasks.items["K1"]["can_complete"])
        tasks.candidate(["K1"], ["B"])
        self.assertFalse(tasks.items["K1"]["can_complete"])

    def test_policy_covered_target_waits_but_conflict_can_reopen(self):
        item = {"query": "new query", "knowledge_task_ids": ["K1"]}
        args = dict(history=[], tasks={"K1": {"support": "supported"}}, available_inputs=[],
                    remaining=5, stalled=0, limits=SearchBudget())
        self.assertEqual(choose(item, **args)[0], "finalize")
        self.assertEqual(choose(item, **args, reopen=True)[0], "execute")

    def test_tool_errors_can_retry_but_empty_success_reuses(self):
        item = {"query": "A   Query", "knowledge_task_ids": ["K1"]}
        args = dict(tasks={"K1": {"support": "unknown"}}, available_inputs=[], remaining=5, stalled=0, limits=SearchBudget())
        self.assertEqual(choose(item, history=[{"query": "a query", "error": "Timeout"}], **args)[0], "execute")
        self.assertEqual(choose(item, history=[{"query": "a query", "evidence_ids": []}], **args)[0], "reuse")

    def test_result_saved_before_return_is_reused_after_interruption(self):
        with tempfile.TemporaryDirectory() as directory, Checkpoint(directory) as store:
            journal = DurableTools(store, "structured_off", 0)
            result = journal.execute("search", {"query": "q", "corpus": "v"}, lambda: [document("A", "full source text")])
            restarted = DurableTools(store, "structured_off", 0)
            restored = restarted.execute("search", {"query": "q", "corpus": "v"}, lambda: self.fail("duplicate physical execution"))
            self.assertEqual(result, restored)
            self.assertEqual(restarted.replayed, 1)
            isolated = DurableTools(store, "text_off", 0)
            self.assertEqual(isolated.execute("search", {"query": "q", "corpus": "v"}, lambda: "separate condition"), "separate condition")

    def test_unknown_readonly_computation_can_recompute_and_is_counted(self):
        with tempfile.TemporaryDirectory() as directory, Checkpoint(directory) as store:
            journal = DurableTools(store, "s", 0)
            with self.assertRaises(PowerCut):
                journal.execute("search", {"query": "q"}, lambda: (_ for _ in ()).throw(PowerCut()))
            restarted = DurableTools(store, "s", 0)
            self.assertEqual(restarted.execute("search", {"query": "q"}, lambda: "result"), "result")
            self.assertEqual(restarted.unknown_recomputations, 1)
            self.assertEqual(store.db.execute("SELECT attempts FROM operations").fetchone()[0], 2)

    def test_control_state_replay_does_not_silently_overwrite_history(self):
        with tempfile.TemporaryDirectory() as directory, Checkpoint(directory) as store:
            DurableTools(store, "s", 0).snapshot("applied", {"queries": ["old"], "count": 1})
            replay = DurableTools(store, "s", 0)
            with self.assertRaises(CheckpointError):
                replay.snapshot("applied", {"queries": ["new"], "count": 0})

    def test_unknown_tool_attempt_does_not_get_a_fresh_search_budget(self):
        from ..medqa.checkpoint import StoredToolError
        with tempfile.TemporaryDirectory() as directory, Checkpoint(directory) as store:
            journal = DurableTools(store, "s", 0, max_searches=1)
            with self.assertRaises(PowerCut):
                journal.execute("search_textbooks", {"query": "q"}, lambda: (_ for _ in ()).throw(PowerCut()))
            restarted = DurableTools(store, "s", 0, max_searches=1)
            with self.assertRaisesRegex(StoredToolError, "budget_exhausted"):
                restarted.execute("search_textbooks", {"query": "q"}, lambda: self.fail("budget reset"))
            self.assertEqual(restarted.attempts("search_textbooks"), 1)

    def test_v5_checkpoint_is_rejected_without_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT)")
                db.execute("INSERT INTO metadata VALUES ('format','1')")
            before = path.read_bytes()
            with self.assertRaisesRegex(CheckpointError, "v5记录保留"):
                Checkpoint(directory, resume=True)
            self.assertEqual(path.read_bytes(), before)
