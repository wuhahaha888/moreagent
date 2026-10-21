"""Sequential shared task state and bounded role views; no additional LLM calls.

This is an application-owned workspace, not POSIX shared memory or an LLM's KV
cache. Full source material remains available to the runtime and checkpoint.
"""
from copy import deepcopy
import hashlib
import json
import re

from ..agents.medqa_prompts import PLANNER, INVESTIGATOR, REASONER
from .grounding import ContractError

VERSION = "moreagent.workspace/1"
POLICY = "role_views_v1"


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def weight(value):
    """Deterministic selection heuristic, never reported as API token usage."""
    text = packed(value)
    wide = sum(ord(c) > 127 for c in text)
    return wide + (len(text) - wide) / 4


class Workspace:
    WRITES = {
        PLANNER: {"key_conditions", "knowledge_tasks", "criteria", "memory_assessments"},
        INVESTIGATOR: {"investigation_plan", "need_status"},
        REASONER: {"reasoning_result"},
        "Runtime": {"terminal_result"},
    }

    def __init__(self, task, state):
        self._task = deepcopy(task)
        self.state = state
        self.evidence = {}
        self.revision = 0
        self.patches = []
        self.state.update(deepcopy(task))

    @property
    def task(self):
        return deepcopy(self._task)

    def apply(self, actor, patch):
        if actor not in self.WRITES or not isinstance(patch, dict) or not patch or set(patch) - self.WRITES[actor]:
            raise ContractError("工作空间写入方或变更字段无效")
        # Serialize the whole patch before mutation; invalid values cannot partly
        # update canonical state. Existing business validators run before apply.
        raw = packed(patch)
        values = deepcopy(patch)
        self.state.update(values)
        self.revision += 1
        self.patches.append({"revision": self.revision, "actor": actor,
                            "fields": sorted(values), "sha256": hashlib.sha256(raw.encode()).hexdigest()})

    def snapshot(self):
        if any(self.state.get(k) != v for k, v in self._task.items()):
            raise ContractError("工作空间原题与选项不可修改")
        return {"version": VERSION, "revision": self.revision, "patches": deepcopy(self.patches),
                "task_sha256": hashlib.sha256(packed(self._task).encode()).hexdigest(),
                "sections": {
                    "task": ["question_id", "question", "options"],
                    "planning": ["key_conditions", "knowledge_tasks", "criteria"],
                    "investigation": ["investigation_plan", "attempted_queries"],
                    "evidence": sorted(self.evidence),
                    "memory": [h["id"] for h in self.state.get("memory_hits", [])],
                    "nontext": "ACP vector handles / binary memory-service vectors; not prompt text",
                    "reasoning": ["reasoning_result", "gaps", "conflicts"],
                    "execution": ["task_status", "tool_results", "budget", "feedback"],
                }}


def target_sentence(question):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", question) if s.strip()]
    return sentences[-1] if sentences and "?" in sentences[-1] else ""


def brief_is_eligible(question, conditions, target):
    """Conservative lexical checks, not proof of semantic completeness.

    Source-quoted conditions only; retain all numbers and explicit negations.
    The Reasoner always receives the original task and all options.
    """
    texts = [c.get("text", "") for c in conditions]
    if not target or not texts or any(not t or t not in question for t in texts):
        return False
    brief = " ".join([target, *texts])
    numbers = lambda s: set(re.findall(r"\d+(?:\.\d+)?", s))
    if not numbers(question) <= numbers(brief):
        return False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", question):
        if re.search(r"\b(no|not|without|denies|negative|except|never)\b|无|未|否认|除外", sentence, re.I):
            if sentence.strip() and sentence.strip() not in brief:
                return False
    return True


class RoleViewBuilder:
    def build(self, role, payload):
        original = deepcopy(payload)
        view = {k: deepcopy(payload[k]) for k in ("question_id", "question", "options")}
        decision = "full_task"
        keep = {
            PLANNER: ("task_group", "memory_hits", "feedback"),
            INVESTIGATOR: ("memory_hits", "evidence_pool", "attempted_queries", "gaps", "feedback", "conflicts", "need_status", "active_need_ids"),
            REASONER: ("evidence_pool", "attempted_queries", "gaps", "feedback", "conflicts", "need_status", "active_need_ids"),
        }
        for key in keep[role]:
            if payload.get(key) not in (None, [], {}, ""):
                view[key] = deepcopy(payload[key])
        if role != PLANNER:
            # Relations and support maps remain canonical execution data; these
            # roles need the need IDs/goals, not a repeated copy of every map.
            view["knowledge_tasks"] = [{"id": t["id"], "goal": t["goal"]}
                                       for t in payload.get("knowledge_tasks", [])]
            source_budget = payload.get("budget", {})
            view["budget"] = {k: source_budget[k] for k in ("can_investigate", "search_calls_remaining", "model_calls_remaining")
                              if k in source_budget}
        if role == INVESTIGATOR:
            conditions = payload.get("key_conditions", [])
            target = target_sentence(payload["question"])
            candidate = {"question": target, "key_conditions": conditions}
            if (brief_is_eligible(payload["question"], conditions, target)
                    and weight(candidate) < weight({"question": payload["question"]})):
                view.update(deepcopy(candidate))
                decision = "source_quoted_brief"
            else:
                decision = "full_task_without_duplicate_conditions"
        return view, {"version": VERSION, "role": role, "task_view": decision,
                      "omitted_fields": sorted(set(original) - set(view)),
                      "original_bytes": len(packed(original).encode()), "view_bytes": len(packed(view).encode()),
                      "question_preserved": view["question"] == original["question"],
                      "all_options_preserved": view["options"] == original["options"]}


def role_prompts(prompts):
    result = dict(prompts)
    result[PLANNER] += "\n【共享工作状态】只提交本步骤的规划字段，不重复题目、选项或交接摘要。关键条件优先使用简短原文片段，保留数值、否定和适用限制；不要罗列与检索无关的背景。"
    result[INVESTIGATOR] += "\n【共享工作状态】输入是必要角色视图，knowledge_tasks仅提供ID与目标，关联及执行状态由程序维护。直接提交可执行tool_plan，不复述上游计划或工具结果。原题若为片段，key_conditions为原文条件；全部选项仍保留。"
    result[REASONER] += "\n【共享工作状态】原题和全部选项完整保留。直接根据可见原文回答；规划条件、执行映射和历史记忆正文不重复投递。knowledge_tasks提供补查可用的ID与目标。不要复述工作流程。"
    return result
