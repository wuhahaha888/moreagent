"""Guard against duplicate context, stale scopes and unnecessary curation calls."""

import tempfile
import unittest
from pathlib import Path

from LangChain.memory.advanced import SQLiteMemoryStore
from LangChain.memory.agent import MemoryAgent


class Embeddings:
    def embed_sentence(self, text):
        return [1, 0, 0, 0]


class NoModelCall:
    def invoke(self, messages):
        raise AssertionError("structured evidence must not need a second model call")


class EfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = SQLiteMemoryStore(Path(self.temp.name) / "mem.sqlite3", "test", "test", "probe", 3)

    def put(self, text, trace="past", status="curated", scope=None, kind="evidence"):
        return self.store.put(role="资料搜索", topic="CPU检索方案", summary=text,
                              content={"text": text, "scope": scope or {}}, embedding=[1, 0, 0, 0],
                              trace_id=trace, status=status, kind=kind)

    def test_current_trace_not_reinjected_even_when_already_curated(self):
        current = self.put("当前任务的重复发现", trace="now", status="pending")
        self.put("当前任务已整理但不应反复注入", trace="now")
        previous = self.put("过去任务的可用发现")
        agent = MemoryAgent(self.store, Embeddings())
        hits, _ = agent.retrieve({"trace_id": "now", "user_question": "CPU检索方案", "task_vec": [1, 0, 0, 0]}, "任务执行")
        self.assertEqual([h["id"] for h in hits], [previous])
        self.assertIsNotNone(self.store.get(current, "now"))

    def test_version_filter_applies_before_candidate_limit(self):
        old = self.put("CPU旧版本v1性能", scope={"performance_version": "v1"})
        fresh = self.put("CPU当前v2性能", scope={"performance_version": "v2"})
        hits = self.store.search("CPU性能", [[1, 0, 0, 0]], trace_id="now", role="资料搜索",
                                 strategy="dense", limit=1, historical_only=True, context={"performance_version": "v2"})
        self.assertEqual([h["id"] for h in hits], [fresh])
        self.assertIsNotNone(self.store.get(old))

    def test_changed_threshold_keeps_evidence_but_rejects_old_conclusion(self):
        evidence = self.put("CPU v1原始数据仍可重新筛选", scope={"performance_version": "v1"})
        self.put("旧条件下推荐dense", scope={"performance_version": "v1", "p95_max_ms": 100}, kind="conclusion")
        hits = self.store.search("CPU推荐", [[1, 0, 0, 0]], trace_id="now", role="任务分配", strategy="dense",
                                 context={"performance_version": "v1", "p95_max_ms": 50})
        self.assertEqual([h["id"] for h in hits], [evidence])

    def test_structured_finding_saved_without_model_and_keeps_source(self):
        claim = "[S02] CPU v1中hybrid p95为62ms，数据为模拟输入。"
        source = self.store.add("资料搜索", {"findings": [{"claim": claim, "url": "evidence://S02"}]},
                                [1, 0, 0, 0], trace_id="now", memory_scope={"performance_version": "v1", "p95_max_ms": 100})
        agent = MemoryAgent(self.store, Embeddings(), NoModelCall(), curation_mode="auto")
        event = agent.consolidate({"trace_id": "now", "verification": {"final_answer": "已完成分析"}, "user_question": "CPU选型"})
        self.assertEqual(event["llm_calls"], 0)
        item = self.store.get(event["memory_ids"][0])
        self.assertEqual(item["content"]["text"], claim)
        self.assertEqual(item["source_ids"], [source])
        self.assertEqual(item["content"]["scope"], {"performance_version": "v1"})
        self.assertEqual(item["content"]["trust"], "extracted_not_fact_checked")


if __name__ == "__main__":
    unittest.main()
