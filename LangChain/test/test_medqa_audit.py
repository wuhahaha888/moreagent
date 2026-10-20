"""三Agent反例审查：预算、记忆适用性、证据与评测；无医学准确率声明。"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from .test_medqa import Agents, Books, Embeddings
from . import test_medqa as fixtures
from ..medqa.retrieval import digest
from ..agents.medqa_prompts import PLANNER, INVESTIGATOR, REASONER
from ..medqa.runtime import Runtime


class AuditTest(unittest.TestCase):
    setUp = fixtures.MedQATest.setUp
    memory = fixtures.MedQATest.memory

    def test_frozen_memory_can_be_used_without_learning_evaluation_question(self):
        memory, books = self.memory(), Books()
        existing = memory.put(books.doc, [1., 0., 0., 0.], "accumulation", task_group="medqa")
        runtime = Runtime(Agents(), books, memory=memory, embedder=Embeddings(), memory_write=False)
        result = runtime.run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["memory_used"], 1)
        self.assertEqual(result["metrics"]["memory_written"], 0)
        self.assertFalse(runtime.handshake()["memory_service"]["write_enabled"])
        hits = memory.recall([1., 0., 0., 0.], "after-evaluation", books.version)
        self.assertEqual([hit["id"] for hit in hits], [existing])
        self.assertEqual(hits[0]["trace_id"], "accumulation")

    def test_memory_snapshot_includes_wal_and_keeps_original_unchanged(self):
        import sqlite3
        from ..medqa.cli import copy_memory_snapshot
        source = Path(self.temp.name) / "source.sqlite3"
        target = Path(self.temp.name) / "evaluation.sqlite3"
        with sqlite3.connect(source) as original:
            original.execute("PRAGMA journal_mode=WAL")
            original.execute("CREATE TABLE evidence (id TEXT)")
            original.execute("INSERT INTO evidence VALUES ('accumulated')")
            original.commit()
            copy_memory_snapshot(source, target)
            with sqlite3.connect(target) as evaluation:
                self.assertEqual(evaluation.execute("SELECT id FROM evidence").fetchall(), [("accumulated",)])
                evaluation.execute("INSERT INTO evidence VALUES ('evaluation-only')")
            self.assertEqual(original.execute("SELECT id FROM evidence").fetchall(), [("accumulated",)])

    def test_summary_only_memory_cannot_be_marked_usable(self):
        memory, books = self.memory(), Books()
        memory.put(books.doc, [1., 0., 0., 0.], "old", task_group="medqa")
        class Overconfident(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == PLANNER:
                    data["memory_assessments"][0]["decision"] = "use"
                return data, usage
        result = Runtime(Overconfident(), books, memory=memory, embedder=Embeddings()).run(self.q)
        self.assertEqual(result["status"], "protocol_error")
        self.assertEqual(result["metrics"]["model_calls"], 2)
        self.assertEqual(books.calls, 0)
        self.assertEqual(result["metrics"]["memory_written"], 0)

    def test_planner_cannot_invent_candidate_memory_ids(self):
        class Invented(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == PLANNER:
                    data["memory_assessments"] = [{"memory_id": "fake", "decision": "read", "reason": "fake", "knowledge_task_ids": ["K1"]}]
                return data, usage
        self.assertEqual(Runtime(Invented(), Books()).run(self.q)["status"], "protocol_error")

    def test_no_hidden_answer_from_memory_without_source_read(self):
        memory, books = self.memory(), Books()
        memory.put(books.doc, [1., 0., 0., 0.], "old", task_group="medqa")
        class SkipRead(Agents):
            def call(self, role, p, mode):
                if role == INVESTIGATOR:
                    return {"question_id": p["question_id"], "action": "finish", "findings": [{"id": "F1", "claim": "summary is enough",
                        "knowledge_task_ids": ["K1"], "evidence_ids": ["Fixture_0"]}]}, {"total_tokens": 15, "source": "fixture"}
                return super().call(role, p, mode)
        result = Runtime(SkipRead(), books, memory=memory, embedder=Embeddings()).run(self.q)
        self.assertEqual(result["status"], "protocol_error")
        self.assertEqual(result["metrics"]["memory_source_valid"], 0)

    def test_missing_checks_can_be_repaired_once(self):
        class Repairable(Agents):
            count = 0
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == REASONER:
                    self.count += 1
                    if self.count == 1:
                        data["checks"] = []
                    else:
                        self.feedback_received = "contract_error" in p["feedback"]
                return data, usage
        agents = Repairable()
        result = Runtime(agents, Books()).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["model_calls"], 4)
        self.assertEqual(result["metrics"]["contract_repairs"], 1)
        self.assertTrue(agents.feedback_received)

    def test_unchecked_answer_never_passes(self):
        class Unchecked(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == REASONER:
                    data.pop("checks")
                return data, usage
        result = Runtime(Unchecked(), Books()).run(self.q)
        self.assertEqual(result["status"], "protocol_error")
        self.assertEqual(result["metrics"]["contract_repairs"], 1)

    def test_fabricated_quote_is_rejected(self):
        class Fabricated(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == REASONER:
                    data["claims"][0]["support_quotes"][0]["quote"] = "Invented quotation that never occurs in this source."
                return data, usage
        self.assertEqual(Runtime(Fabricated(), Books()).run(self.q)["status"], "protocol_error")

    def test_semantic_failure_cannot_be_relabelled_in_format_repair(self):
        for key in ("checks", "condition_checks"):
            class Unsupported(Agents):
                def call(self, role, p, mode):
                    data, usage = super().call(role, p, mode)
                    if role == REASONER:
                        data[key][0]["verdict"] = "unsupported"
                    return data, usage
            result = Runtime(Unsupported(), Books()).run(self.q)
            self.assertEqual(result["status"], "protocol_error")
            self.assertEqual(result["metrics"]["contract_repairs"], 1)

    def test_unresolved_conflict_and_missing_condition_block_answer(self):
        for key in ("conflicts", "condition_checks"):
            class Incomplete(Agents):
                def call(self, role, p, mode):
                    data, usage = super().call(role, p, mode)
                    if role == REASONER:
                        data[key] = ([{"evidence_ids": ["Fixture_0"], "status": "unresolved", "reason": "synthetic conflict"}]
                                     if key == "conflicts" else [])
                    return data, usage
            self.assertEqual(Runtime(Incomplete(), Books()).run(self.q)["status"], "protocol_error")

    def test_initial_duplicate_batch_is_reused_without_aborting(self):
        class TooMany(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == INVESTIGATOR and data.get("action") == "search":
                    data["search"]["queries"] *= 2
                return data, usage
        books = Books()
        result = Runtime(TooMany(), books).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(books.calls, 1)
        self.assertEqual(result["metrics"]["queries_reused"], 1)

    def test_duplicate_query_cannot_retry_outside_budget(self):
        class Repeat(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == INVESTIGATOR and data.get("action") == "search":
                    for query in data["search"]["queries"]:
                        query["query"] = "  FIXTURE   query 0 "
                return data, usage
        books = Books(empty_until=100)
        result = Runtime(Repeat(), books).run(self.q)
        self.assertEqual(result["status"], "answered_unsupported")
        self.assertEqual(books.calls, 1)

    def test_followup_round_is_reserved_before_tool_execution(self):
        result = Runtime(Agents(), Books(errors=True)).run(self.q)
        reserve = next(i for i, e in enumerate(result["events"]) if e.get("tool") == "request_investigation")
        searches = [i for i, e in enumerate(result["events"]) if e.get("tool") == "search_textbooks"]
        self.assertLess(searches[0], reserve)
        self.assertLess(reserve, searches[1])
        self.assertEqual(result["metrics"]["followup_rounds"], 1)

    def test_insufficient_model_budget_does_not_start_unfinishable_search(self):
        agents, books = Agents(), Books()
        result = Runtime(Agents(verify=["retrieve"]), books, max_calls=4).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(books.calls, 1)
        self.assertLessEqual(result["metrics"]["model_calls"], 4)
        with self.assertRaises(ValueError):
            Runtime(agents, books, max_calls=65)

    def test_observed_token_budget_stops_next_call(self):
        result = Runtime(Agents(), Books(), max_question_tokens=15).run(self.q)
        self.assertEqual(result["status"], "token_budget_exhausted")
        self.assertEqual(result["metrics"]["model_calls"], 1)

    def test_deadline_does_not_start_first_model_call(self):
        result = Runtime(Agents(), Books(), deadline_seconds=1e-12).run(self.q)
        self.assertEqual(result["status"], "deadline_exhausted")
        self.assertEqual(result["metrics"]["model_calls"], 0)

    def test_unknown_usage_does_not_continue_calls(self):
        class Estimated(Agents):
            def call(self, *args):
                data, usage = super().call(*args)
                usage["source"] = "estimated"
                return data, usage
        result = Runtime(Estimated(), Books()).run(self.q)
        self.assertEqual(result["status"], "usage_unavailable")
        self.assertEqual(result["metrics"]["model_calls"], 1)

    def test_no_model_call_for_memory_curation_and_no_answer_letter_stored(self):
        memory, books = self.memory(), Books()
        agents = Agents()
        result = Runtime(agents, books, memory=memory, embedder=Embeddings()).run(self.q)
        self.assertEqual(len(agents.calls), 3)
        hit = memory.recall([1., 0., 0., 0.], "later", books.version, task_group="medqa")[0]
        self.assertNotIn("answer", hit["content"])
        self.assertEqual(hit["content"]["conditions"], ["synthetic condition"])
        self.assertEqual(result["metrics"]["memory_written"], 1)

    def test_failed_memory_write_does_not_erase_answer(self):
        class BrokenMemory:
            transfers = bytes = 0
            def recall(self, *args, **kwargs):
                return []
            def put(self, *args, **kwargs):
                raise OSError("fixture")
        result = Runtime(Agents(), Books(), memory=BrokenMemory(), embedder=Embeddings()).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["memory_write_errors"], 1)

    def test_memory_failure_falls_back_to_textbooks(self):
        class BrokenMemory:
            transfers = bytes = 0
            def recall(self, *args, **kwargs):
                raise TimeoutError("fixture")
            def put(self, *args, **kwargs):
                return "stored"
        books = Books()
        result = Runtime(Agents(), books, memory=BrokenMemory(), embedder=Embeddings()).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["memory_read_errors"], 1)
        self.assertEqual(books.calls, 1)

    def test_api_failure_sanitized_and_unknown_usage_counted(self):
        for exception in (RuntimeError, ValueError):
            class Broken(Agents):
                def call(self, *args):
                    raise exception("private-token-must-not-appear")
            result = Runtime(Broken(), Books()).run(self.q)
            self.assertEqual(result["status"], "runtime_error")
            self.assertEqual(result["metrics"]["usage_missing_calls"], 1)
            self.assertNotIn("private-token", str(result))

    def test_actual_response_chars_and_bytes_are_counted(self):
        class Spaced(Agents):
            chars = byte_count = 0
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                data["test_unicode"] = "中文"
                raw = json.dumps(data, ensure_ascii=False, indent=4)
                self.chars += len(raw)
                self.byte_count += len(raw.encode())
                return raw, usage
        agents = Spaced()
        result = Runtime(agents, Books()).run(self.q)
        self.assertEqual(result["metrics"]["response_chars"], agents.chars)
        self.assertEqual(result["metrics"]["response_bytes"], agents.byte_count)
        self.assertGreater(agents.byte_count, agents.chars)

    def test_memory_timeout_invalidates_channel(self):
        service = self.memory()
        with patch.object(service, "_receive", side_effect=TimeoutError("fixture")):
            with self.assertRaises(TimeoutError):
                service.recall([1., 0., 0., 0.], "trace", Books.version)
        with self.assertRaises(RuntimeError):
            service.recall([1., 0., 0., 0.], "new", Books.version)
        self.assertFalse(service._process.is_alive())

    def test_concurrent_requests_remain_paired(self):
        from concurrent.futures import ThreadPoolExecutor
        service, books = self.memory(), Books()
        second = {**books.doc, "id": "Fixture_1", "text": "Second synthetic source for concurrency fixture."}
        second["sha256"] = digest(second["text"])
        service.put(books.doc, [1., 0., 0., 0.], "old")
        service.put(second, [0., 1., 0., 0.], "old")
        def recall(index):
            vector = [1., 0., 0., 0.] if index % 2 == 0 else [0., 1., 0., 0.]
            return service.recall(vector, str(index), books.version, limit=1)[0]["content"]["id"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(recall, range(12))), ["Fixture_" + str(i % 2) for i in range(12)])

    def test_incomplete_usage_cannot_produce_token_saving_claim(self):
        from ..medqa.cli import summarize, compare_reports
        report, ids = {}, {}
        for condition in ("text_off", "structured_off", "text_on", "structured_on"):
            result = Runtime(Agents(), Books()).run(self.q)
            result["correct"] = True
            result["metrics"]["usage_sources"] = ["model"] * result["metrics"]["model_calls"]
            if condition == "structured_on":
                result["metrics"]["usage_missing_calls"] = 1
            report[condition] = summarize([result])
            ids[condition] = [self.q.id]
        comparison = compare_reports(report, ids, [self.q.id])
        self.assertTrue(comparison["matched_complete"])
        self.assertFalse(comparison["token_comparison_valid"])
        self.assertIsNone(comparison["memory_structured"]["token_reduction"])

    def test_independent_retrieval_labels_still_required(self):
        from ..medqa.evaluation import prepare, evaluate, EvaluationError
        out, books = Path(self.temp.name), Books()
        negative = {**books.doc, "id": "Fixture_1", "text": "Unrelated synthetic passage, not a medical fact."}
        negative["sha256"] = digest(negative["text"])
        books.rankings = lambda query: {"bm25": [negative, books.doc], "medcpt": [books.doc, negative]}
        prepare(out, [self.q], books, "fixture-dataset", 42)
        path = out / "judgments-template.json"
        labels = json.loads(path.read_text())
        with self.assertRaises(EvaluationError):
            evaluate(out, path, 1)
        labels["annotator"] = "mechanism-fixture-only"
        for item in labels["cases"][0]["judgments"]:
            item.update(relevance=2 if item["evidence_id"] == books.doc["id"] else 0,
                        rationale="synthetic relation", support_quote=books.doc["text"] if item["evidence_id"] == books.doc["id"] else "")
        path.write_text(json.dumps(labels))
        report = evaluate(out, path, 1)
        self.assertEqual(report["metrics"]["bm25"]["precision_at_k"], 0)
        self.assertEqual(report["metrics"]["medcpt"]["precision_at_k"], 1)

    def test_debugged_question_is_excluded_from_heldout(self):
        from ..medqa.data import Dataset
        path = Path(self.temp.name) / "benchmark.json"
        row = {"question": self.q.question, "options": self.q.options, "answer": "A"}
        path.write_text(json.dumps({"medqa": {"0990": row, "0202": row}}))
        dataset = Dataset(path)
        self.assertEqual([q.id for q in dataset.select()], ["0202"])
        self.assertEqual([q.id for q in dataset.select(split="development")], ["0990"])
        with self.assertRaises(ValueError):
            dataset.select(ids=["0990"])

    def test_event_snapshots_and_request_ids_are_independent(self):
        result = Runtime(Agents(), Books(empty_until=2)).run(self.q)
        requests = [e for e in result["events"] if e.get("agent") == INVESTIGATOR]
        self.assertEqual(requests[0]["input"]["attempted_queries"], [])
        self.assertEqual(len(requests[1]["input"]["attempted_queries"]), 1)
        ids = [e["request_id"] for e in result["events"] if "agent" in e]
        self.assertEqual(len(ids), len(set(ids)))

    def test_whitespace_mapping_cannot_invent_quote(self):
        from ..medqa.protocol import normalize_response
        source = {"S1": {"text": "The absenceof ganglion cells confirms this synthetic fixture."}}
        data = {"claims": [{"support_quotes": [{"evidence_id": "S1", "quote": "absence of ganglion cells confirms this synthetic fixture."}]}]}
        output, mapped = normalize_response(REASONER, data, source)
        self.assertTrue(mapped)
        self.assertIn(output["claims"][0]["support_quotes"][0]["quote"], source["S1"]["text"])
        data["claims"][0]["support_quotes"][0]["quote"] = "presence of ganglion cells confirms this synthetic fixture."
        self.assertEqual(normalize_response(REASONER, data, source)[1], [])

    def test_known_conflict_cannot_disappear_from_final_answer(self):
        class OmittedConflict(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == REASONER and data.get("decision") == "retrieve":
                    data["conflicts"] = [{"evidence_ids": ["Fixture_0"], "status": "unresolved", "reason": "synthetic known conflict"}]
                return data, usage
        result = Runtime(OmittedConflict(verify=["retrieve", "answer"]), Books()).run(self.q)
        self.assertEqual(result["status"], "protocol_error")
        self.assertEqual(result["metrics"]["contract_repairs"], 1)

    def test_circular_task_dependency_is_rejected_before_search(self):
        class Circular(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == PLANNER:
                    data["knowledge_tasks"][0]["depends_on"] = ["K1"]
                return data, usage
        books = Books()
        result = Runtime(Circular(), books).run(self.q)
        self.assertEqual(result["status"], "protocol_error")
        self.assertEqual(books.calls, 0)

    def test_followup_requires_actionable_expected_information(self):
        class Vague(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == REASONER:
                    data = {"question_id": p["question_id"], "decision": "retrieve", "followup_request": {
                        "gap": "low confidence", "expected_information": "", "knowledge_task_ids": ["K1"]}}
                return data, usage
        result = Runtime(Vague(), Books()).run(self.q)
        self.assertEqual(result["status"], "protocol_error")
        self.assertEqual(result["metrics"]["followup_rounds"], 0)

    def test_evidence_budget_retains_higher_ranked_results(self):
        books = Books()
        documents = [{**books.doc, "id": "Fixture_" + str(i)} for i in range(3)]
        books.search = lambda query: documents
        agents = Agents()
        result = Runtime(agents, books, max_evidence=2, evidence_processor=False).run(self.q)
        reasoner = next(p for role, p, _ in agents.calls if role == REASONER)
        self.assertEqual([d["id"] for d in reasoner["evidence_pool"]], ["Fixture_0", "Fixture_1"])
        self.assertEqual(result["evidence_processing"]["omitted"], ["Fixture_2"])
        self.assertEqual(result["status"], "supported")

    def test_failed_answer_does_not_write_long_term_memory(self):
        memory = self.memory()
        result = Runtime(Agents(illegal=True), Books(), memory=memory, embedder=Embeddings()).run(self.q)
        self.assertEqual(result["status"], "protocol_error")
        self.assertEqual(result["metrics"]["memory_written"], 0)
        self.assertEqual(memory.recall([1., 0., 0., 0.], "later", Books.version, task_group="medqa"), [])

    def test_document_read_request_cannot_exceed_two_sources(self):
        memory, books = self.memory(), Books()
        for i in range(3):
            memory.put({**books.doc, "id": "Fixture_" + str(i)}, [1., 0., 0., 0.], "old", task_group="medqa")
        class TooManyReads(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == INVESTIGATOR:
                    data = {"question_id": p["question_id"], "action": "read_memory", "read_memory": {
                        "memory_ids": [m["id"] for m in p["memory_hits"]]}}
                return data, usage
        result = Runtime(TooManyReads(), books, memory=memory, embedder=Embeddings(), max_reads=2).run(self.q)
        self.assertEqual(result["status"], "answered_unsupported")
        self.assertEqual(result["metrics"]["document_reads"], 0)
        self.assertEqual(books.stats["source_reads"], 0)

    def test_planner_repair_feedback_is_not_repeated_to_investigator(self):
        class Repairable(Agents):
            planner_calls = 0
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == PLANNER:
                    self.planner_calls += 1
                    if self.planner_calls == 1:
                        data["criteria"] = {}
                return data, usage
        agents = Repairable()
        result = Runtime(agents, Books()).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["metrics"]["contract_repairs"], 1)
        self.assertEqual(result["metrics"]["model_calls"], 4)
        self.assertTrue(all(p["feedback"] == {} for role, p, _ in agents.calls if role == INVESTIGATOR))

    def test_truncated_response_keeps_actual_bytes_and_usage(self):
        class Truncated(Agents):
            def call(self, *args):
                usage = {"total_tokens": 37, "prompt_tokens": 30, "completion_tokens": 7,
                         "source": "fixture", "finish_reason": "length"}
                return '{"question_id":"q1","fragment":"中文', usage
        result = Runtime(Truncated(), Books()).run(self.q)
        self.assertEqual(result["status"], "protocol_error")
        self.assertEqual(result["metrics"]["total_tokens"], 37)
        response = next(e["response"] for e in result["events"] if e.get("agent"))
        self.assertEqual(result["metrics"]["response_bytes"], len(response.encode()))

    def test_invalid_question_id_response_is_retained_for_diagnosis(self):
        class WrongID(Agents):
            def call(self, *args):
                data, usage = super().call(*args)
                data["question_id"] = "wrong"
                return data, usage
        result = Runtime(WrongID(), Books()).run(self.q)
        self.assertEqual(result["status"], "protocol_error")
        self.assertTrue(any('wrong' in e.get("response", "") for e in result["events"]))

    def test_memory_vector_represents_task_and_exact_evidence(self):
        class Recording(Embeddings):
            texts = []
            def embed_text(self, text):
                self.texts.append(text)
                return super().embed_text(text)
        embedding, books = Recording(), Books()
        result = Runtime(Agents(), books, memory=self.memory(), embedder=embedding).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertIn(self.q.question, embedding.texts[-1])
        self.assertIn("synthetic condition", embedding.texts[-1])
        self.assertIn(books.doc["text"], embedding.texts[-1])

    def test_memory_excerpts_must_be_exact_source_text(self):
        memory, books = self.memory(), Books()
        with self.assertRaises(RuntimeError):
            memory.put(books.doc, [1., 0., 0., 0.], "old", excerpts=["invented unobserved text"])
        self.assertEqual(memory.recall([1., 0., 0., 0.], "later", books.version), [])

    def test_new_memory_context_does_not_overwrite_old_unit(self):
        memory, books = self.memory(), Books()
        first = memory.put(books.doc, [1., 0., 0., 0.], "old", task_group="g", conditions=["scope1"])
        second = memory.put(books.doc, [1., 0., 0., 0.], "new", task_group="g", conditions=["scope2"])
        self.assertNotEqual(first, second)
        # 不把同一来源的多个历史上下文挤成三个重复候选。
        hits = memory.recall([1., 0., 0., 0.], "later", books.version, task_group="g")
        self.assertEqual(len(hits), 1)

    def test_unmapped_condition_survives_planning_and_requires_final_check(self):
        class PartialPlan(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == PLANNER:
                    data["key_conditions"].append({"id": "P2", "text": "additional synthetic condition"})
                return data, usage
        agents = PartialPlan()
        result = Runtime(agents, Books()).run(self.q)
        self.assertEqual(sum(role == PLANNER for role, _, _ in agents.calls), 1)
        reasoner = next(p for role, p, _ in agents.calls if role == REASONER)
        self.assertEqual(reasoner["unmapped_condition_ids"], ["P2"])
        self.assertEqual(len(reasoner["key_conditions"]), 2)
        self.assertEqual(result["status"], "protocol_error")

    def test_read_memory_summary_is_not_repeated_but_scope_and_raw_evidence_remain(self):
        from ..medqa.context import memory_view
        hits = [{"id": "M1", "source_id": "S1", "conditions": ["historical scope"],
                 "summary": "repeated raw excerpt", "similarity": .9,
                 "status": "source_checked_not_semantically_verified"},
                {"id": "M2", "source_id": "S2", "conditions": [], "summary": "unread summary",
                 "status": "candidate_needs_source_read"}]
        planner = memory_view(PLANNER, hits, {"S1"})
        reasoner = memory_view(REASONER, hits, {"S1"})
        self.assertEqual(planner, hits)
        self.assertEqual(len(reasoner), 1)
        self.assertNotIn("summary", reasoner[0])
        self.assertNotIn("similarity", reasoner[0])
        self.assertEqual(reasoner[0]["conditions"], ["historical scope"])
        self.assertIn("summary", hits[0])

    def test_unread_candidate_is_not_counted_as_invalid_source(self):
        from ..medqa.cli import summarize
        result = Runtime(Agents(), Books()).run(self.q)
        result["correct"] = True
        result["metrics"].update(memory_candidates=3, document_reads=2, memory_source_valid=2,
                                 memory_candidates_by_kind={"evidence":3,"strategy":0,"episode":0},memory_source_check_attempts=2)
        report = summarize([result])
        self.assertEqual(report["memory_source_valid_rate"], 1)
        self.assertAlmostEqual(report["memory_source_check_coverage"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
