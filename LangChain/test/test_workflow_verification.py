"""通用工具结果验证与失败记忆隔离；模型替身检查，无 API 调用。"""

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from LangChain.agents.workflow import MultiAgentSystem
from LangChain.memory.advanced import SQLiteMemoryStore
from LangChain.memory.agent import MemoryAgent
from LangChain.tools.web_search import DisabledSearchTool
from LangChain.test.test_memory_v2 import FakeEmbeddings, FakeLLM


class WorkflowVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = SQLiteMemoryStore(Path(self.temp.name) / "memory.sqlite3", "test", "test", "probe", 3)

    def test_verifier_repairs_false_installation_and_preserves_evidence(self):
        class Model(FakeLLM):
            def __init__(self, repair):
                super().__init__()
                self.verifications = 0
                self.repair = repair

            def invoke(self, messages):
                if "你是总结验证 Agent" not in messages[0].content:
                    return super().invoke(messages)
                self.calls += 1
                self.verifications += 1
                result = {"plan": "使用hybrid制定独立环境部署与验收步骤", "actual_install_verified": not (self.repair and self.verifications > 1)}
                return AIMessage(content=json.dumps({"final_answer": {"result": result, "claims": []}, "need": "end"}),
                                 usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

        def handler(state, data):
            result = {"plan": "hybrid", "actual_install_verified": False}
            return {"draft": json.dumps({"result": result, "claims": [{"text": "只制定计划，未安装", "evidence_ids": ["S07"]}]}),
                    "_trusted_tool_result": result, "need": "verify"}

        for repair in (True, False):
            with self.subTest(repair=repair):
                model = Model(repair)
                agent = MemoryAgent(self.store, FakeEmbeddings(), model)
                runtime = MultiAgentSystem(model, FakeEmbeddings(), self.store, DisabledSearchTool(), agent, handler)
                state = runtime.run("制定部署计划")
                self.assertEqual(model.verifications, 2)
                self.assertEqual(state["metrics"]["summary"]["token_usage"]["total_tokens"], 75 if repair else 60)
                answer = json.loads(state["final_answer"])
                self.assertEqual(answer["result"]["plan"], "hybrid")
                self.assertIn("部署", answer["plan_details"])
                self.assertEqual(answer["claims"][0]["evidence_ids"], ["S07"])
                self.assertEqual(bool(state["verification"].get("_verification_failed")), not repair)
                if not repair:
                    self.assertFalse(state["metrics"]["memory_consolidation"]["memory_ids"])


if __name__ == "__main__":
    unittest.main()
