"""中文提示与四项证据审计：替身测试，不验证医学准确率。"""
from copy import deepcopy
import unittest

from .test_medqa import Agents, Books, Embeddings
from . import test_medqa as fixtures
from ..agents.medqa_prompts import PLANNER, INVESTIGATOR, REASONER, OUTPUT_EXAMPLES, MEDQA_ROLE_PROMPTS
from ..agents.medqa_compact_prompts import COMPACT_ROLE_PROMPTS
from ..medqa.runtime import Runtime
from ..medqa.answer_audit import validate_answer
from ..medqa.grounding import ContractError
from ..medqa.cli import summarize


class AnswerPolicyTest(unittest.TestCase):
    setUp = fixtures.MedQATest.setUp
    memory = fixtures.MedQATest.memory

    def output(self):
        p = {"question_id": self.q.id, "evidence_pool": [Books().doc], "memory_hits": []}
        return Agents().call(REASONER, p, "structured")[0]

    def audit(self, output):
        return validate_answer(output, self.q, {"K1"}, [{"id": "P1", "text": "fixture condition"}],
                               {"Fixture_0": Books().doc}, set())

    @staticmethod
    def partial(output):
        output["claims"].append({"id": "C2", "claim": "unsourced decisive rule", "knowledge_task_ids": ["K1"],
            "basis": ["model_knowledge"], "evidence_ids": [], "support_quotes": [], "question_quotes": []})
        output["checks"].append({"claim_id": "C2", "verdict": "unsupported", "evidence_ids": [], "reason": "no external support"})
        output["condition_checks"][0]["claim_ids"].append("C2")
        output["option_comparison"].update(verdict="partial", claim_ids=["C1", "C2"], reason="distinction partly relies on model knowledge")
        output["critical_gaps"] = ["decisive rule still needs external evidence"]
        output["final_answer"].update(evidence_status="partial", answer_basis=sorted({b for c in output["claims"] for b in c["basis"]}))
        return output

    def test_supported_answer_has_all_four_audits_and_can_write_memory(self):
        result = Runtime(Agents(), Books(), memory=self.memory(), embedder=Embeddings()).run(self.q)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["evidence_status"], "supported")
        self.assertEqual(result["answer_basis"], ["textbook"])
        self.assertEqual(result["metrics"]["memory_written"], 1)
        self.assertIn("option_comparison", result["verification"])
        self.assertEqual(result["verification"]["critical_gaps"], [])

    def test_no_evidence_can_still_answer_in_same_three_calls_without_memory_write(self):
        memory = self.memory()
        result = Runtime(Agents(verify=["answer"]), Books(empty_until=100), memory=memory, embedder=Embeddings()).run(self.q)
        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["status"], "answered_unsupported")
        self.assertEqual(result["answer_basis"], ["model_knowledge"])
        self.assertEqual(result["metrics"]["model_calls"], 3)
        self.assertEqual(result["metrics"]["memory_written"], 0)
        self.assertTrue(result["gaps"])
        self.assertEqual(memory.recall([1., 0., 0., 0.], "later", Books.version), [])

    def test_partial_answer_does_not_store_even_its_supported_subset(self):
        class Mixed(Agents):
            def call(self, role, p, mode):
                output, usage = super().call(role, p, mode)
                return (AnswerPolicyTest.partial(output) if role == REASONER else output), usage
        memory = self.memory()
        result = Runtime(Mixed(), Books(), memory=memory, embedder=Embeddings()).run(self.q)
        self.assertEqual(result["status"], "answered_partial")
        self.assertEqual(result["answer"], "A")
        self.assertEqual(set(result["answer_basis"]), {"textbook", "model_knowledge"})
        self.assertEqual(result["metrics"]["memory_written"], 0)
        self.assertEqual(memory.recall([1., 0., 0., 0.], "later", Books.version), [])

    def test_partially_supported_memory_reuse_does_not_teach_unsourced_rule(self):
        memory, books = self.memory(), Books()
        original = memory.put(books.doc, [1., 0., 0., 0.], "old", task_group="medqa")
        class Mixed(Agents):
            def call(self, role, p, mode):
                output, usage = super().call(role, p, mode)
                return (AnswerPolicyTest.partial(output) if role == REASONER else output), usage
        result = Runtime(Mixed(), books, memory=memory, embedder=Embeddings()).run(self.q)
        self.assertEqual(result["status"], "answered_partial")
        self.assertEqual(set(result["answer_basis"]), {"memory", "model_knowledge"})
        self.assertEqual(result["metrics"]["memory_used"], 1)
        self.assertEqual(result["metrics"]["memory_written"], 0)
        self.assertEqual([h["id"] for h in memory.recall([1., 0., 0., 0.], "later", Books.version)], [original])

    def test_own_knowledge_cannot_be_relabelled_supported(self):
        output = self.partial(self.output())
        output["checks"][1]["verdict"] = "supported"
        with self.assertRaises(ContractError):
            self.audit(output)

    def test_partial_assessment_is_downgraded_without_discarding_choice(self):
        output = self.partial(self.output())
        output["final_answer"]["evidence_status"] = "supported"
        audit = self.audit(output)
        self.assertEqual(audit["evidence_status"], "partial")
        self.assertEqual(audit["status"], "answered_partial")
        self.assertEqual(audit["declared_evidence_status"], "supported")
        self.assertTrue(audit["audit_warnings"])

    def test_program_never_upgrades_model_declared_insufficient(self):
        output = self.partial(self.output())
        output["final_answer"]["evidence_status"] = "insufficient"
        self.assertEqual(self.audit(output)["evidence_status"], "insufficient")

    def test_given_question_fact_needs_exact_quote_and_does_not_prove_a_medical_rule(self):
        output = self.partial(self.output())
        claim = output["claims"][1]
        claim.update(basis=["question"], question_quotes=[self.q.question])
        output["checks"][1]["verdict"] = "supported"
        output["final_answer"]["answer_basis"] = ["textbook", "question"]
        self.assertEqual(self.audit(output)["evidence_status"], "partial")
        claim["question_quotes"] = ["invented patient fact"]
        with self.assertRaises(ContractError):
            self.audit(output)

    def test_related_evidence_without_option_distinction_is_only_partial(self):
        output = self.output()
        output["option_comparison"]["verdict"] = "unsupported"
        output["critical_gaps"] = ["retrieval supports topic but not requested decision"]
        output["final_answer"]["evidence_status"] = "partial"
        self.assertEqual(self.audit(output)["status"], "answered_partial")

    def test_mismatched_scope_or_unresolved_conflict_requires_insufficient(self):
        for case in ("scope", "conflict"):
            output = self.output()
            if case == "scope":
                output["condition_checks"][0]["verdict"] = "mismatched"
            else:
                output["conflicts"] = [{"evidence_ids": ["Fixture_0"], "status": "unresolved", "reason": "contradictory context"}]
            output["option_comparison"]["verdict"] = "unsupported"
            output["critical_gaps"] = ["decisive conflict or scope mismatch"]
            output["final_answer"]["evidence_status"] = "insufficient"
            self.assertEqual(self.audit(output)["status"], "answered_unsupported")

    def test_competing_option_must_be_a_different_real_option(self):
        output = self.output()
        for option in ("A", "INVENTED"):
            output["option_comparison"]["alternative"] = option
            with self.assertRaises(ContractError):
                self.audit(output)

    def test_claims_cannot_be_omitted_from_applicability_check(self):
        output = self.partial(self.output())
        output["condition_checks"][0]["claim_ids"] = ["C1"]
        with self.assertRaises(ContractError):
            self.audit(output)

    def test_non_supported_answers_must_disclose_gaps_and_full_basis(self):
        for field in ("gaps", "basis"):
            output = self.partial(self.output())
            if field == "gaps":
                output["critical_gaps"] = []
            else:
                output["final_answer"]["answer_basis"] = ["textbook"]
            with self.assertRaises(ContractError):
                self.audit(output)

    def test_history_basis_requires_real_source_read(self):
        output = self.output()
        output["claims"][0]["basis"] = ["memory"]
        output["final_answer"]["answer_basis"] = ["memory"]
        with self.assertRaises(ContractError):
            self.audit(output)

    def test_reporting_separates_answer_correctness_and_evidence_grade(self):
        supported = Runtime(Agents(), Books()).run(self.q)
        unsupported = Runtime(Agents(verify=["answer"]), Books(empty_until=100)).run(self.q)
        supported["correct"], unsupported["correct"] = False, True
        report = summarize([supported, unsupported])
        self.assertEqual(report["accuracy"], .5)
        self.assertEqual(report["supported_rate"], .5)
        self.assertEqual(report["wrong_supported_questions"], 1)
        self.assertEqual(report["accuracy_by_evidence_status"], {"supported": 0., "insufficient": 1.})

    def test_chinese_example_contract_and_both_prompt_profiles(self):
        output = deepcopy(OUTPUT_EXAMPLES[REASONER])
        output["question_id"] = self.q.id
        output["final_answer"]["answer"] = "A"
        output["option_comparison"]["alternative"] = "B"
        self.assertEqual(self.audit(output)["status"], "answered_unsupported")
        for profile in (MEDQA_ROLE_PROMPTS, COMPACT_ROLE_PROMPTS):
            for role in (PLANNER, INVESTIGATOR, REASONER):
                self.assertIn("model_knowledge", profile[role])
            self.assertIn("四项审核三", profile[REASONER])
            self.assertIn("critical_gaps", profile[REASONER])

