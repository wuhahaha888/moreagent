"""无网络、无 API 费用的记忆/集成回归测试；不代表真实 LLM 质量评测。"""

import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from langchain_core.messages import AIMessage

from LangChain.memory.advanced import SQLiteMemoryStore, compact
from LangChain.memory.agent import MemoryAgent
from LangChain.memory.embedding import QwenEmbeddingConfig, QwenEmbeddingService
from LangChain.memory.sessions import MemorySessionManager
from LangChain.agents.workflow import MultiAgentSystem
from LangChain.tools.web_search import DisabledSearchTool


FINAL = "在 openEuler 24.03-LTS-SP3 部署时，应创建应用专用 Python 环境，保留原有系统环境。"


class FakeEmbeddings:
    model_name = "test-model"
    dim = 4

    def embed_task(self, text):
        return [1., 0., 0., 0.]

    embed_sentence = embed_task


class FakeLLM:
    def __init__(self, hallucinate=False):
        self.calls = 0
        self.hallucinate = hallucinate

    def invoke(self, messages):
        self.calls += 1
        system = messages[0].content
        data = json.loads(messages[1].content)
        if "你是记忆管理 Agent" in system:
            source = next(x for x in data["sources"] if x["role"] == "总结验证")
            result = {"memories": [{"source_id": source["source_id"],
                "text": "来源中没有这句话，不能编造新的事实加入记忆。" if self.hallucinate else source["text"],
                "kind": "conclusion"}]}
        elif "你是任务分配 Agent" in system:
            result = {"goal": "openEuler 环境部署", "tasks": ["规划环境"], "criteria": ["环境隔离"], "need": "research"}
        elif "你是资料搜索 Agent" in system:
            result = {"findings": [], "gaps": ["没有联网"], "need": "execute"}
        elif "你是任务执行 Agent" in system:
            result = {"draft": FINAL, "steps": ["创建独立环境"], "need": "verify"}
        else:
            result = {"final_answer": FINAL, "checks": ["保留原环境"], "need": "end"}
        if "memory_hits" in data:
            result["used_memory_ids"] = [x["id"] for x in data["memory_hits"]]
        return AIMessage(content=compact(result),
                         usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "memory.sqlite3"
        self.store = self.make_store()

    def make_store(self, namespace="project", session="session", model="test-model"):
        return SQLiteMemoryStore(self.path, namespace, session, model, 3)

    def add(self, store=None, **kwargs):
        args = dict(role="总结验证", topic="openEuler 部署", summary=FINAL,
                    content={"text": FINAL}, embedding=[1, 0, 0, 0],
                    trace_id="old", status="curated", kind="conclusion")
        args.update(kwargs)
        return (store or self.store).put(**args)

    def search(self, store=None, **kwargs):
        args = dict(query="openEuler 部署", query_vectors=[np.array([1, 0, 0, 0])],
                    trace_id="new", role="任务分配")
        args.update(kwargs)
        return (store or self.store).search(**args)

    def test_cross_session_and_namespace_isolation(self):
        ident = self.add()
        self.assertEqual(self.search(self.make_store(session="other"))[0]["id"], ident)
        self.assertEqual(self.search(self.make_store(namespace="private")), [])
        self.assertIsNone(self.make_store(namespace="private").get(ident))

    def test_pending_is_task_local_and_expired_filtered(self):
        pending = self.add(status="pending")
        self.add(expires_at=time.time() - 1, content={"text": "expired"})
        self.assertEqual(self.search(), [])
        self.assertEqual(self.search(trace_id="old")[0]["id"], pending)

    def test_provenance_and_explicit_replacement(self):
        source = self.add(status="pending")
        old = self.add(source_ids=[source])
        self.assertEqual(self.store.provenance(old)[0]["id"], source)
        new = self.add(content={"text": FINAL + "新增条件：只采用 CPU 模型。"})
        self.store.supersede(old, new)
        self.assertIsNone(self.store.get(old))
        self.assertEqual(self.search()[0]["id"], new)
        self.assertEqual(self.make_store(namespace="other").provenance(new), [])

    def test_dimension_and_model_mismatch_not_compared(self):
        self.add()
        self.assertEqual(self.search(strategy="dense", query_vectors=[np.array([1, 0])]), [])
        self.assertEqual(self.search(self.make_store(model="other"), strategy="dense"), [])
        # 词法检索独立于向量模型，可返回文本，但不能假装其向量可用。
        self.assertTrue(self.search(self.make_store(model="other"), strategy="keyword"))

    def test_transactional_dedup_under_concurrency(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(lambda _: self.add(), range(12)))
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(len(self.store.items), 1)

    def test_budget_and_real_vector_payload_size(self):
        self.add()
        agent = MemoryAgent(self.store, FakeEmbeddings(), max_chars=1000)
        state = {"trace_id": "new", "user_question": "openEuler 部署", "task_vec": [1, 0, 0, 0],
                 "context_vec": [1, 0, 0, 0]}
        hits, metrics = agent.retrieve(state, "任务分配")
        self.assertTrue(hits)
        self.assertLessEqual(len(compact(hits)), 1000)
        self.assertEqual(metrics["vector_bytes"], 32)
        self.assertEqual(metrics["state_transfer_count"], 1)
        self.assertEqual(metrics["cross_task_count"], 1)

    def test_curator_cannot_invent_text_or_import_foreign_source(self):
        self.add(status="pending", content={"final_answer": FINAL})
        agent = MemoryAgent(self.store, FakeEmbeddings(), FakeLLM(hallucinate=True))
        report = agent.consolidate({"trace_id": "old", "verification": {"final_answer": FINAL}})
        self.assertEqual(report["memory_ids"], [])
        self.assertEqual(report["rejected_count"], 1)
        self.assertEqual(self.search(), [])

    def test_fallback_is_not_promoted(self):
        llm = FakeLLM()
        agent = MemoryAgent(self.store, FakeEmbeddings(), llm)
        event = agent.consolidate({"trace_id": "old", "verification": {"final_answer": FINAL, "_fallback": True}})
        self.assertEqual(llm.calls, 0)
        self.assertIn("skipped", event)

    def test_persistent_embedding_cache_and_version(self):
        class Encoder:
            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 4)) / 2
        encoder = QwenEmbeddingService(QwenEmbeddingConfig("fake-model"))
        encoder._model = Encoder()
        encoder.cache_store = self.store
        first = encoder.embed_text("隔离环境")
        second = encoder.embed_text("隔离环境")
        self.assertEqual(first, second)
        self.assertEqual(encoder.encoded_texts, 1)
        self.assertEqual(encoder.cache_hits, 1)
        self.assertIsNone(self.store.cache_get("different-version", "隔离环境"))

    def test_ten_tasks_five_agents_with_cross_task_reuse(self):
        llm = FakeLLM()
        seen_traces = set()
        for index in range(10):
            store = self.make_store(session=f"session-{index}")
            embeddings = FakeEmbeddings()
            agent = MemoryAgent(store, embeddings, llm)
            system = MultiAgentSystem(llm, embeddings, store, DisabledSearchTool(), agent)
            result = system.run(f"openEuler 部署方案，第 {index + 1} 个连续任务")
            self.assertEqual(result["final_answer"], FINAL)
            self.assertEqual(len(result["precise_stack"]), 5)
            summary = result["metrics"]["summary"]
            self.assertLessEqual(summary["memory_hit_rate"], 1)
            self.assertEqual(summary["token_usage"]["total_tokens"], 75)
            self.assertEqual(summary["memory_query_count"], 4)
            if index:
                self.assertGreater(summary["cross_task_memory_hits"], 0)
                self.assertGreater(summary["cross_task_memory_citations"], 0)
            seen_traces.add(result["trace_id"])
        self.assertEqual(llm.calls, 50)
        self.assertEqual(len(seen_traces), 10)

    def test_sqlite_session_restored_from_chat_log(self):
        root = Path(self.temp.name)
        manager = MemorySessionManager(root / "memories", root / "chats")
        session = manager.new_session()
        session.chat_path.write_text("{}")
        restored = manager.list_sessions()
        self.assertEqual(restored[0].memory_path, session.memory_path)


if __name__ == "__main__":
    unittest.main()
