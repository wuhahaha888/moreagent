"""Independent result products for the same three-Agent/four-condition experiment."""
from copy import deepcopy
import json
import unittest

from .test_medqa import Agents, Books, Embeddings
from ..agents.medqa_prompts import PLANNER, INVESTIGATOR, REASONER
from ..medqa.data import Question
from ..medqa.runtime import Runtime
from ..medqa.results import split_result
from ..medqa.protocol import parse_agent_output
from ..medqa.grounding import ContractError
from ..medqa.cli import summarize


class AgentJSONParsingTest(unittest.TestCase):
    def test_valid_object_and_string_content_are_unchanged(self):
        value = {"text": 'literal } , "critical_gaps": and \\ escapes', "nested": {"v": 1}}
        parsed, changes = parse_agent_output(json.dumps(value), allow_repair=True)
        self.assertEqual(parsed, value)
        self.assertEqual(changes, [])

    def test_premature_root_close_keeps_all_fields_and_suffix(self):
        raw = '  {"answer":"A","nested":{"text":"}"}} , "critical_gaps":["retain me"]}  '
        parsed, changes = parse_agent_output(raw, allow_repair=True)
        self.assertEqual(parsed, {"answer": "A", "nested": {"text": "}"}, "critical_gaps": ["retain me"]})
        self.assertEqual(changes[0]["extra_model_calls"], 0)
        self.assertEqual(changes[0]["added_top_level_fields"], ["critical_gaps"])

    def test_duplicate_fields_are_rejected_in_valid_and_repaired_json(self):
        for raw in ('{"answer":"A","answer":"B"}',
                    '{"answer":"A"},"answer":"B"}',
                    '{"answer":"A","x":{"v":1,"v":2}}'):
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                parse_agent_output(raw, allow_repair=True)

    def test_multiple_objects_prose_and_incomplete_suffix_are_not_discarded(self):
        for raw in ('{"answer":"A"}{"answer":"B"}',
                    '{"answer":"A"} some explanation',
                    '{"answer":"A"},"critical_gaps":[',
                    '{"answer":"A"},"critical_gaps":[],}',
                    '{"answer":"A"},"critical_gaps":[]} extra',
                    '{"answer":"A"'):
            with self.subTest(raw=raw), self.assertRaises(json.JSONDecodeError):
                parse_agent_output(raw, allow_repair=True)

    def test_truncated_and_disabled_repair_remain_errors(self):
        raw = '{"answer":"A"},"critical_gaps":[]}'
        for options in ({"allow_repair": False}, {"allow_repair": True, "truncated": True}):
            with self.subTest(options=options), self.assertRaises(json.JSONDecodeError):
                parse_agent_output(raw, **options)

    def test_nonfinite_values_are_rejected(self):
        for raw in ('{"value":NaN}', '{"value":Infinity}', '{"value":-Infinity}', '{"value":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                parse_agent_output(raw, allow_repair=True)


class Memory:
    def __init__(self):
        self.writes = []
    def recall(self, *args, **kwargs):
        return []
    def put(self, document, values, trace, **metadata):
        self.writes.append((document, metadata))


class ResultProductsTest(unittest.TestCase):
    def setUp(self):
        self.q = Question("q", "Synthetic patient condition with NOT and time limit.", {i: i for i in "ABCD"})

    def run_agents(self, agents, **kwargs):
        return Runtime(agents, Books(), output_policy="decoupled", **kwargs).run(self.q)

    def test_premature_root_close_is_audited_without_model_retry_in_both_modes(self):
        class MalformedAnswer(Agents):
            def call(self, role, payload, mode):
                data, usage = super().call(role, payload, mode)
                if role == REASONER:
                    gaps = data.pop("critical_gaps")
                    data = json.dumps(data) + ',"critical_gaps":' + json.dumps(gaps) + '}'
                return data, usage
        for mode in ("text", "structured"):
            with self.subTest(mode=mode):
                result = self.run_agents(MalformedAnswer(), mode=mode)
                self.assertEqual(result["answer"], "A")
                self.assertEqual(result["metrics"]["model_calls"], 3)
                self.assertEqual(result["protocol_result"]["status"], "issues")
                self.assertTrue(any(e["tool"] == "json_syntax_repair"
                                    for e in result["protocol_result"]["repairs"]))
                event = next(e for e in result["events"] if e.get("agent") == REASONER)
                with self.assertRaises(json.JSONDecodeError):
                    json.loads(event["response"])
                self.assertEqual(event["normalized_output"]["final_answer"]["answer"], "A")

    def test_syntax_repair_cannot_bypass_current_question_identity(self):
        class WrongQuestion(Agents):
            def call(self, role, payload, mode):
                data, usage = super().call(role, payload, mode)
                if role == REASONER:
                    data["question_id"] = "another-question"
                    gaps = data.pop("critical_gaps")
                    data = json.dumps(data) + ',"critical_gaps":' + json.dumps(gaps) + '}'
                return data, usage
        result = self.run_agents(WrongQuestion())
        self.assertIsNone(result["answer"])
        self.assertEqual(result["metrics"]["model_calls"], 3)
        self.assertTrue(result["protocol_result"]["blocks_answer"])

    def test_attribution_error_keeps_answer_and_never_retries_or_writes_memory(self):
        class Ambiguous(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == REASONER:
                    data["claims"][0]["question_quotes"] = [p["question"]]
                return data, usage
        agents, memory = Ambiguous(), Memory()
        result = self.run_agents(agents, memory=memory, embedder=Embeddings())
        self.assertEqual([role for role, _, _ in agents.calls], [PLANNER, INVESTIGATOR, REASONER])
        self.assertEqual(result["answer"], "A")
        self.assertTrue(result["answer_result"]["complete"])
        self.assertEqual(result["protocol_result"]["status"], "issues")
        self.assertEqual(result["evidence_result"]["strict_audit_status"], "failed")
        self.assertEqual(result["evidence_status"], "unverified")
        self.assertFalse(result["evidence_result"]["memory_eligible"])
        self.assertEqual(memory.writes, [])
        self.assertEqual(result["metrics"]["contract_repairs"], 0)
        self.assertEqual(result["metrics"]["total_tokens"], 45)
        self.assertTrue(any(i.get("path") == "claims[0].question_quotes" for i in result["protocol_result"]["issues"]))

    def test_minimal_answer_requires_no_full_audit_table(self):
        class Minimal(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == REASONER:
                    doc = p["evidence_pool"][0]
                    data = {"question_id": p["question_id"], "decision": "answer", "final_answer": {"answer": "A", "summary": "fixture"},
                            "citations": [{"evidence_id": doc["id"], "quote": doc["text"], "support": "partial"}],
                            "critical_gaps": ["unresolved relation"]}
                return data, usage
        result = self.run_agents(Minimal())
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["evidence_result"]["strict_audit_status"], "not_requested")
        cite = result["evidence_result"]["citations"][0]
        self.assertTrue(cite["source_verified"])
        self.assertTrue(cite["visible"])
        self.assertEqual(cite["model_reported_support"], "partial")
        self.assertFalse(cite["eligible_as_reliable_support"])

    def test_invalid_source_quote_does_not_become_reliable_support(self):
        output = {"question_id": self.q.id, "final_answer": {"answer": "A"},
                  "citations": [{"evidence_id": "INVENTED", "quote": "A fabricated source excerpt.", "support": "supported"}]}
        result = split_result(output, self.q, [], [], {}, {})
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["evidence_result"]["source_verified_ids"], [])
        self.assertEqual(result["evidence_result"]["reliable_evidence_ids"], [])
        self.assertFalse(result["evidence_result"]["citations"][0]["eligible_as_reliable_support"])

    def test_visible_raw_match_is_not_independent_proof_of_medical_support(self):
        doc = Books().doc
        output = {"final_answer": {"answer": "A"}, "citations": [{"evidence_id": doc["id"], "quote": doc["text"], "support": "supported"}]}
        result = split_result(output, self.q, [], [], {doc["id"]: doc}, {doc["id"]: [{"text": doc["text"]}]})
        citation = result["evidence_result"]["citations"][0]
        self.assertTrue(citation["source_verified"])
        self.assertFalse(citation["support_audit_passed"])
        self.assertFalse(citation["eligible_as_reliable_support"])

    def test_malformed_optional_audit_fields_cannot_clear_valid_option(self):
        for key, value in (("claims", {}), ("citations", "bad"), ("critical_gaps", {})):
            output = {"final_answer": {"answer": "A"}, key: value}
            original = deepcopy(output)
            result = split_result(output, self.q, [], [], {}, {})
            self.assertEqual(result["answer"], "A")
            self.assertTrue(result["protocol_result"]["issues"])
            self.assertEqual(output, original)

    def test_invalid_option_and_wrong_question_are_real_failures(self):
        for output in ({"final_answer": {"answer": "Z"}}, {"question_id": "another", "final_answer": {"answer": "A"}}):
            result = split_result(output, self.q, [], [], {}, {})
            self.assertIsNone(result["answer"])
            self.assertFalse(result["answer_result"]["complete"])

    def test_full_supported_audit_retains_existing_memory_gate(self):
        memory = Memory()
        result = self.run_agents(Agents(), memory=memory, embedder=Embeddings())
        self.assertEqual(result["answer_result"]["status"], "answered")
        self.assertEqual(result["evidence_result"]["status"], "supported")
        self.assertTrue(result["evidence_result"]["memory_eligible"])
        self.assertEqual(len(memory.writes), 1)
        self.assertNotIn("answer", memory.writes[0][1])

    def test_unsupported_model_knowledge_cannot_become_reliable_memory(self):
        class Overclaim(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == REASONER:
                    data["claims"][0]["basis"].append("model_knowledge")
                    data["final_answer"]["answer_basis"].append("model_knowledge")
                return data, usage
        memory = Memory()
        result = self.run_agents(Overclaim(), memory=memory, embedder=Embeddings())
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["evidence_result"]["strict_audit_status"], "failed")
        self.assertEqual(memory.writes, [])

    def test_unknown_usage_keeps_answer_but_disables_memory_and_cost_claims(self):
        class MissingUsage(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == REASONER:
                    usage = {"source": "unknown"}
                return data, usage
        memory = Memory()
        result = self.run_agents(MissingUsage(), memory=memory, embedder=Embeddings())
        self.assertEqual(result["answer"], "A")
        self.assertEqual(memory.writes, [])
        result["correct"] = True
        report = summarize([result])
        self.assertFalse(report["token_usage_complete"])
        self.assertIsNone(report["tokens_per_answer"])

    def test_planner_schema_fallback_does_not_add_model_calls(self):
        class BrokenPlanner(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == PLANNER:
                    data = {"question_id": p["question_id"]}
                return data, usage
        result = self.run_agents(BrokenPlanner())
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["metrics"]["model_calls"], 3)
        self.assertTrue(any(r.get("tool") == "baseline_fallback" for r in result["protocol_result"]["repairs"]))

    def test_investigator_schema_fallback_still_retrieves_without_model_retry(self):
        class BrokenInvestigator(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == INVESTIGATOR:
                    data = {"question_id": p["question_id"], "action": "unknown"}
                return data, usage
        result = self.run_agents(BrokenInvestigator())
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["metrics"]["model_calls"], 3)
        self.assertEqual(result["metrics"]["retrieval_count"], 1)

    def test_invalid_answer_json_is_not_retried_or_guessed(self):
        class BrokenAnswer(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                return ('{"broken":' if role == REASONER else data), usage
        result = self.run_agents(BrokenAnswer())
        self.assertIsNone(result["answer"])
        self.assertEqual(result["metrics"]["model_calls"], 3)
        self.assertTrue(result["protocol_result"]["blocks_answer"])

    def test_four_conditions_keep_the_same_agents_and_non_memory_payload(self):
        payloads = {}
        for mode in ("text", "structured"):
            for enabled in (False, True):
                agents = Agents()
                kwargs = {"mode": mode}
                if enabled:
                    kwargs.update(memory=Memory(), embedder=Embeddings(), memory_write=False)
                result = self.run_agents(agents, **kwargs)
                self.assertEqual(result["answer"], "A")
                self.assertEqual([r for r, _, _ in agents.calls], [PLANNER, INVESTIGATOR, REASONER])
                self.assertEqual(result["metrics"]["model_calls"], 3)
                payloads[mode, enabled] = [p for _, p, _ in agents.calls]
        self.assertEqual([{k:v for k,v in p.items() if k!='_natural_handoffs'} for p in payloads['text',False]],
                         payloads['structured',False])
        self.assertEqual([{k:v for k,v in p.items() if k!='_natural_handoffs'} for p in payloads['text',True]],
                         payloads['structured',True])

    def test_reporting_separates_completion_protocol_evidence_and_cost(self):
        class BadAudit(Agents):
            def call(self, role, p, mode):
                data, usage = super().call(role, p, mode)
                if role == REASONER:
                    data["claims"][0]["question_quotes"] = [p["question"]]
                return data, usage
        result = self.run_agents(BadAudit())
        result["correct"] = True
        report = summarize([result])
        self.assertEqual(report["accuracy"], 1)
        self.assertEqual(report["answer_completion_rate"], 1)
        self.assertEqual(report["runtime_failed_questions"], 0)
        self.assertEqual(report["protocol_status_counts"], {"issues": 1})
        self.assertEqual(report["strict_evidence_audit_counts"], {"failed": 1})
        self.assertEqual(report["total_tokens"], 45)
