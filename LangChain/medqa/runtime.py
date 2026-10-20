"""三 Agent 有界协作：程序预取记忆 → 规划 → 工具调查 → 核验回答 → 程序写记忆。"""
import json
import math
import time
import uuid
from copy import deepcopy

from ..agents.medqa_prompts import (
    PLANNER, INVESTIGATOR, REASONER, ROLES, MEDQA_ROLE_PROMPTS, MEDQA_INPUT_FIELDS,
)
from .usage import normalize_usage
from .grounding import ContractError, identifiers, validate_plan
from .answer_audit import validate_answer
from .protocol import normalize_response
from .retrieval import digest
from .memory import memory_embedding_text
from .memory_units import build_units, embedding_text as unit_embedding_text, POLICY_VERSION as MEMORY_UNIT_POLICY
from .memory_selection import (previews, prefilter, select as select_memories, ReuseTrace,
    VERSION as SELECTION_VERSION, PLANNER_LIMIT, PLANNER_PREVIEW_CHARS)
from .context import memory_view
from .g2cp_profile import message
from .evidence_processor import EvidenceProcessor, validate_visible_quotes, validate_view
from .evidence_handoff import build_handoff, visible_delta, VERSION as HANDOFF_VERSION
from .task_state import TaskStates
from .budget import SearchBudget
from .retrieval_policy import choose, query_key
from .issues import Issue
from .results import split_result, failure_results

PROTOCOL = "moreagent.medqa/6"
ANSWER_POLICY = "best_effort_with_four_checks_v1"


def protocol_for(output_policy):
    return "moreagent.medqa/baseline-2" if output_policy == "decoupled" else PROTOCOL


def answer_policy_for(output_policy):
    return "answer_evidence_protocol_split_v1" if output_policy == "decoupled" else ANSWER_POLICY
CAPABILITIES = {
    PLANNER: ["assess_candidate_memories", "propose_evidence_requirements"],
    INVESTIGATOR: ["search_textbooks", "read_memory", "challenge_followup_direction"],
    REASONER: ["check_conditions", "resolve_conflicts", "answer", "request_followup"],
}


class ModelCallError(RuntimeError):
    """SDK异常不得带着URL或凭据进入轨迹。"""


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def render_text(value, indent=0):
    """同字段可读文本基线；不加入虚构或重复历史。"""
    if isinstance(value, dict):
        return "\n".join(" " * indent + str(k) +
                         (":\n" + render_text(v, indent + 2) if isinstance(v, (dict, list)) and v else ": " + compact(v))
                         for k, v in value.items()) or " " * indent + "{}"
    if isinstance(value, list):
        return "\n".join(" " * indent +
                         ("-\n" + render_text(v, indent + 2) if isinstance(v, (dict, list)) and v else "- " + compact(v))
                         for v in value) or " " * indent + "[]"
    return " " * indent + compact(value)


class ChatRoles:
    """三个独立角色提示；相同模型设置；不为检索或记忆写入另开模型调用。"""
    def __init__(self, model, profile="compact", output_policy="strict", memory_selection="legacy"):
        from ..agents.medqa_compact_prompts import COMPACT_ROLE_PROMPTS
        from ..agents.medqa_baseline_prompts import BASELINE_ROLE_PROMPTS
        if profile not in {"compact", "full"}:
            raise ValueError("未知上下文配置")
        self.model = model
        if output_policy == "decoupled":
            from ..agents.medqa_baseline_prompts import FOLLOWUP_RULES
            self.prompts = dict(BASELINE_ROLE_PROMPTS)
            if profile == "full":
                # Accumulation keeps the existing whole-answer memory admission audit.
                self.prompts[REASONER] = MEDQA_ROLE_PROMPTS[REASONER].split("【动态补查】")[0] + FOLLOWUP_RULES
        else:
            self.prompts = COMPACT_ROLE_PROMPTS if profile == "compact" else MEDQA_ROLE_PROMPTS
        if memory_selection == "needs":
            from ..agents.medqa_memory_prompts import MEMORY_ROLE_PROMPTS
            self.prompts.update({role:MEMORY_ROLE_PROMPTS[role] for role in (PLANNER,INVESTIGATOR)})

    def call(self, role, payload, mode):
        return self.call_with_budget(role, payload, mode)

    def call_with_budget(self, role, payload, mode, *, remaining_seconds=None, remaining_tokens=None):
        from langchain_core.messages import HumanMessage, SystemMessage
        text = compact(payload) if mode == "structured" else render_text(payload)
        messages = [SystemMessage(content=self.prompts[role]), HumanMessage(content=text)]
        if hasattr(self.model, "invoke_bounded"):
            reply = self.model.invoke_bounded(messages, remaining_seconds=remaining_seconds,
                                              remaining_tokens=remaining_tokens)
        else:
            reply = self.model.invoke(messages)
        usage = normalize_usage(reply, self.prompts[role] + text, reply.content)
        details = getattr(reply, "response_metadata", {})
        usage.update(response_model=details.get("model", "unknown"), response_chars=len(reply.content))
        counts = details.get("token_usage", {})
        usage["cached_prompt_tokens"] = (counts.get("prompt_tokens_details") or {}).get("cached_tokens", counts.get("prompt_cache_hit_tokens", 0))
        usage["reasoning_tokens"] = (counts.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        usage["finish_reason"] = details.get("finish_reason")
        return reply.content, usage


class Runtime:
    def __init__(self, agents, textbooks, *, memory=None, embedder=None, mode="structured",
                 max_calls=8, memory_threshold=.65, max_evidence=8, max_evidence_chars=12000,
                 max_prompt_chars=24000, max_contract_repairs=4, context_profile="compact",
                 memory_limit=3, memory_context_chars=2400, max_question_tokens=24000,
                 deadline_seconds=180, task_group="medqa", memory_write=True,
                 max_searches=6, max_reads=4, max_stalled_rounds=2,
                 evidence_processor=True, evidence_span_chars=1800, journal=None,
                 output_policy="strict", memory_policy="strict", memory_selection="legacy", memory_recall_limit=10):
        if output_policy not in {"strict", "decoupled"}:
            raise ValueError("未知输出策略")
        if memory_policy not in {"strict", "basic"}:
            raise ValueError("未知记忆策略")
        if memory_selection not in {"legacy","needs"} or not isinstance(memory_recall_limit,int) or not 3<=memory_recall_limit<=10:
            raise ValueError("未知记忆选择策略或粗召回上限（3至10）")
        if memory_selection=="needs" and (memory_policy!="basic" or output_policy!="decoupled"):
            raise ValueError("需求重排需要basic记忆与decoupled输出")
        if mode not in {"text", "structured"} or context_profile not in {"compact", "full"}:
            raise ValueError("无效通信模式或上下文配置")
        if (isinstance(max_calls, bool) or not isinstance(max_calls, int) or not 3 <= max_calls <= 64
                or not isinstance(max_contract_repairs, int) or not 0 <= max_contract_repairs <= 8 or not 1 <= memory_limit <= 3
                or min(max_evidence, max_evidence_chars, max_prompt_chars, memory_context_chars, max_question_tokens) <= 0):
            raise ValueError("调用保护上限须为3至64，纠错保护上限0至8，候选至多3条，预算须为正数")
        if not math.isfinite(deadline_seconds) or deadline_seconds <= 0 or not math.isfinite(memory_threshold) or not -1 <= memory_threshold <= 1:
            raise ValueError("无效期限或记忆阈值")
        if not isinstance(task_group, str) or not task_group.strip():
            raise ValueError("任务组不能为空")
        if memory is not None and embedder is None:
            raise ValueError("记忆需要embedding服务")
        self.agents, self.textbooks, self.memory, self.embedder = agents, textbooks, memory, embedder
        self.memory_write = memory_write
        self.memory_policy = memory_policy
        self.memory_selection,self.memory_recall_limit=memory_selection,memory_recall_limit
        self.mode, self.context_profile, self.max_calls = mode, context_profile, max_calls
        self.memory_threshold, self.memory_limit = memory_threshold, memory_limit
        self.max_evidence, self.max_evidence_chars, self.max_prompt_chars = max_evidence, max_evidence_chars, max_prompt_chars
        self.max_contract_repairs, self.memory_context_chars = max_contract_repairs, memory_context_chars
        self.max_question_tokens, self.deadline_seconds, self.task_group = max_question_tokens, deadline_seconds, task_group
        self.limits = SearchBudget(max_searches, max_reads, max_stalled_rounds)
        self.processor = EvidenceProcessor(enabled=evidence_processor, span_chars=evidence_span_chars,
            max_chars=max_evidence_chars, max_documents=max_evidence)
        self.journal = journal
        self.output_policy = output_policy
        self.protocol, self.answer_policy = protocol_for(output_policy), answer_policy_for(output_policy)
        if output_policy == "decoupled":
            from ..agents.medqa_baseline_prompts import BASELINE_ROLE_PROMPTS
            self.default_prompts = BASELINE_ROLE_PROMPTS
            self.max_contract_repairs = 0
        else:
            self.default_prompts = MEDQA_ROLE_PROMPTS
        if memory_selection=="needs":
            from ..agents.medqa_memory_prompts import MEMORY_ROLE_PROMPTS
            self.default_prompts = MEMORY_ROLE_PROMPTS

    def handshake(self):
        return {"protocol": self.protocol, "mode": self.mode, "context_profile": self.context_profile, "answer_policy": self.answer_policy,
                "output_policy": self.output_policy,
                "agents": [{"id": role, "capabilities": (["plan_queries", "choose_query_conditions"]
                    if self.output_policy == "decoupled" and role == INVESTIGATOR else CAPABILITIES[role])} for role in ROLES],
                "memory_service": {"is_llm_agent": False, "transport": "multiprocessing.Pipe", "state_format": "float32-le",
                                   "write_enabled": self.memory is not None and self.memory_write,
                                   "policy": self.memory_policy, "unit_schema": MEMORY_UNIT_POLICY if self.memory_policy=="basic" else "legacy_textbook_extract"},
                "memory_selection":{"policy":self.memory_selection,"version":SELECTION_VERSION,
                    "coarse_limit":self.memory_recall_limit if self.memory_selection=="needs" else self.memory_limit,
                    "read_limit":self.memory_limit,
                    "planner_limit":min(PLANNER_LIMIT,self.memory_limit) if self.memory_selection=='needs' else self.memory_limit,
                    "preview_chars":min(PLANNER_PREVIEW_CHARS,self.memory_context_chars) if self.memory_selection=='needs' else self.memory_context_chars,
                    "weights":[.45,.35,.20]},
                "evidence_processor": {"enabled": self.processor.enabled, "span_chars": self.processor.span_chars},
                "execution_path": "program_memory_by_id → Planner → program_read → Investigator → tools → Reasoner",
                "optional_followup_limit": 1 if self.output_policy == "decoupled" else None,
                "followup_handoff": {"version": HANDOFF_VERSION, "allocation": "gap_and_key_citations",
                                     "requires_new_visible_text": True},
                "limits": {"model_calls": self.max_calls, "policy": "adaptive",
                           "search_calls": self.limits.max_searches, "document_reads": self.limits.max_reads,
                           "stalled_rounds": self.limits.max_stalled_rounds,
                           "contract_repairs": self.max_contract_repairs,
                           "observed_token_budget": self.max_question_tokens, "deadline_seconds": self.deadline_seconds},
                "token_budget_note": "用量达到预算后不再调用；当前请求输入计费可能使总量越界，不宣称硬账单上限。"}

    def run(self, question, *, trace_id=None):
        started, trace = time.monotonic(), trace_id or uuid.uuid4().hex
        state = {**question.public(), "trace_id": trace, "task_group": self.task_group, "corpus_version":self.textbooks.version,
                 "key_conditions": [], "knowledge_tasks": [], "criteria": [], "memory_assessments": [],
                 "memory_hits": [], "findings": [], "gaps": [], "feedback": {}, "tool_results": {},
                 "phase": "investigation", "attempted_queries": [], "task_status": {}, "conflicts": [],
                 "unmapped_condition_ids": [], "evidence_index": []}
        metrics = {key: 0 for key in (
            "model_calls", "messages", "communication_chars", "request_chars", "response_chars",
            "request_bytes", "response_bytes", "system_chars", "system_bytes", "protocol_control_chars", "protocol_control_bytes", "protocol_maps",
            "prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens", "reasoning_tokens",
            "usage_missing_calls", "memory_candidates", "memory_source_valid", "memory_used", "memory_written",
            "memory_queries", "memory_read_errors", "memory_write_errors", "memory_rejected",
            "source_read_errors", "document_reads", "search_errors", "search_empty", "retrieval_count",
            "followup_rounds", "contract_repairs", "revision_count", "evidence_dropped_by_budget", "evidence_evicted",
            "state_transfers", "state_bytes")}
        metrics["usage_sources"] = []
        for key in ("queries_reused", "queries_filtered", "search_progress", "stalled_rounds", "expansions",
                    "evidence_raw_chars", "evidence_raw_delivery_chars", "evidence_view_chars", "evidence_exact_duplicates", "evidence_similar_groups",
                    "recovered_action_errors", "semantic_reviews", "deterministic_repairs",
                    "followup_new_visible_chars", "followup_new_visible_sources"):
            metrics[key] = 0
        metrics["evidence_processor_seconds"] = 0.
        metrics["memory_candidates_by_kind"] = {k:0 for k in ("evidence","strategy","episode")}
        metrics["memory_written_by_kind"] = {k:0 for k in ("evidence","strategy","episode")}
        metrics["memory_experience_reads"] = 0
        metrics['memory_source_check_attempts']=0
        reuse_trace=ReuseTrace()
        planner_presented=set()
        events, evidence, memory_candidates, memory_sources, visited = [], {}, {}, {}, set()
        attempted_reads, history, targets, expanded, protected_quotes = set(), [], {}, set(), {}
        task_states, last_view, repair_fingerprints = TaskStates([]), {}, set()
        processing_cache = {}
        pending_repairs = set()
        base_transfers = getattr(self.memory, "transfers", 0)
        base_bytes = getattr(self.memory, "bytes", 0)
        base_vectors=getattr(self.memory,'vector_transfers',0)
        base_stats = dict(getattr(self.textbooks, "stats", {}))
        base_cache_hits = getattr(self.embedder, "cache_hits", 0)
        base_encoded = getattr(self.embedder, "encoded_texts", 0)
        base_embedding_seconds=getattr(self.embedder,'encoded_seconds',0.)
        final = None
        retained_result, retained_output = None, None
        followup_view, followup_previous_view = None, None

        def operation(kind, request, callback):
            return self.journal.execute(kind, request, callback) if self.journal else callback()

        def persist_state(trigger):
            if self.journal:
                self.journal.snapshot(trigger, {"state": state, "task_states": task_states.items,
                    "transitions": task_states.transitions, "history": history, "seen_sources": sorted(evidence),
                    "targets": targets, "expanded": sorted(expanded), "repair_fingerprints": sorted(repair_fingerprints),
                    "resources": {k: metrics[k] for k in ("model_calls", "total_tokens", "retrieval_count", "document_reads", "stalled_rounds")}})

        def stop(status, reason):
            if retained_result is not None:
                result = deepcopy(retained_result)
                events.append({"tool": "answer_preserved", "reason": status, "message": reason})
                return result
            return {"question_id": question.id, "answer": None, "status": status,
                    "summary": reason, "evidence_status": "not_assessed", "answer_basis": [],
                    "evidence_ids": sorted(evidence), "gaps": deepcopy(state["gaps"])}

        def basic_plan():
            # Program fallback contains only the original input and an unresolved goal.
            state.update(key_conditions=[{"id": "P1", "text": question.question}],
                knowledge_tasks=[{"id": "K1", "goal": "核实题目所问关系并比较选项", "condition_ids": ["P1"], "depends_on": []}],
                criteria=["给出选项并披露证据边界"], memory_assessments=[{
                    "memory_id": h["id"], "decision": "read", "reason": "仅供调查角色回读核对", "knowledge_task_ids": ["K1"]}
                    for h in state["memory_hits"]])

        def basic_search():
            return search({"search": {"gap": "核实当前题目", "queries": [{"query": question.question[:2000],
                "knowledge_task_ids": [state["knowledge_tasks"][0]["id"]], "expected_information": "当前问题的教材依据"}]}})

        def out_of_budget():
            replaying = hasattr(self.agents, "replay_pending") and self.agents.replay_pending()
            spent = (time.monotonic() - started + getattr(self.agents, "historical_seconds", 0)
                     + getattr(self.journal, "historical_seconds", 0))
            if spent >= self.deadline_seconds and not replaying:
                return "deadline_exhausted"
            if metrics["total_tokens"] >= self.max_question_tokens:
                return "token_budget_exhausted"
            if metrics["model_calls"] >= self.max_calls:
                return "budget_exhausted"
            return None

        def add_document(doc, protected=()):
            if (not isinstance(doc, dict) or doc.get("corpus_version") != self.textbooks.version
                    or not isinstance(doc.get("text"), str) or doc.get("sha256") != digest(doc["text"])):
                raise ContractError("教材版本或原文校验值不一致")
            ident = doc.get("id")
            if not isinstance(ident, str) or not ident:
                raise ContractError("来源ID无效")
            if ident in evidence:
                if evidence[ident]["sha256"] != doc["sha256"]:
                    raise ContractError("同一来源ID对应不同内容")
                return True
            # 原文保留；上下文容量由独立视图控制，不能淘汰后忘记历史来源。
            evidence[ident] = deepcopy(doc)
            return True

        def budget():
            return {"model_calls_remaining": self.max_calls - metrics["model_calls"],
                    "search_calls_remaining": self.limits.max_searches - metrics["retrieval_count"],
                    "document_reads_remaining": self.limits.max_reads - metrics["document_reads"],
                    "can_investigate": (self.max_calls - metrics["model_calls"] >= (2 if self.output_policy == "decoupled" else 3)
                        and (self.output_policy != "decoupled" or metrics["followup_rounds"] < 1)
                        and ((metrics["retrieval_count"] < self.limits.max_searches
                              and metrics["stalled_rounds"] < self.limits.max_stalled_rounds)
                             or (self.output_policy == "decoupled" and self.memory is not None
                                 and metrics["document_reads"] < self.limits.max_reads))),
                    "evidence_dropped_by_budget": metrics["evidence_dropped_by_budget"],
                    "observed_tokens_remaining": max(0, self.max_question_tokens - metrics["total_tokens"])}

        def planner_memories():
            selected,ranking=prefilter(question.question,memory_candidates,self.memory_limit)
            by_id={h['id']:h for h in state['memory_hits']}
            return previews([by_id[i] for i in selected if i in by_id],self.memory_context_chars,compact),ranking

        def payload_for(role):
            nonlocal last_view
            state["task_status"] = {i: {"support": s["support"], "evidence_ids": s["evidence_ids"],
                **({"waiting_input": True} if s["lifecycle"] == "waiting_input" else {})}
                for i, s in task_states.items.items()}
            state["budget"] = budget()
            protected = deepcopy(protected_quotes)
            for conflict in state["conflicts"]:
                for ident in conflict["evidence_ids"]:
                    protected.setdefault(ident, [])
                    expanded.add(ident)
            docs = evidence
            if role == PLANNER:
                docs = {}
            elif role == INVESTIGATOR:
                relevant = set(state["feedback"].get("evidence_ids", [])) | set(protected)
                if self.output_policy == "decoupled":
                    relevant |= set(memory_sources.values())
                docs = {i: d for i, d in evidence.items() if i in relevant}
            short_memory_view = self.output_policy == "decoupled" and role == INVESTIGATOR
            request = {"documents": docs, "query": question.question + " " + " ".join(t["goal"] for t in state["knowledge_tasks"]),
                       "targets": targets, "protected": protected, "expanded": sorted(expanded),
                       "enabled": self.processor.enabled, "span_chars": self.processor.span_chars,
                       "max_chars": self.processor.max_chars, "max_documents": self.processor.max_documents}
            if short_memory_view:
                request.update(span_chars=min(600, self.processor.span_chars), max_chars=min(2400, self.processor.max_chars))
            cache_key = digest(compact(request))
            if role == REASONER and followup_view is not None:
                last_view = deepcopy(followup_view)
            elif cache_key not in processing_cache:
                processor = EvidenceProcessor(enabled=request["enabled"], span_chars=request["span_chars"],
                    max_chars=request["max_chars"], max_documents=request["max_documents"])
                processing_cache[cache_key] = operation("evidence_view", request, lambda: processor.process(docs,
                    query=request["query"], targets=targets, protected=protected, expanded=expanded))
                metrics["evidence_processor_seconds"] += processing_cache[cache_key]["cpu_seconds"]
                last_view = deepcopy(processing_cache[cache_key])
            else:
                last_view = deepcopy(processing_cache[cache_key])
            validate_view(last_view, docs)
            state["evidence_pool"] = last_view["views"]
            state["evidence_deferred"] = last_view["omitted"]
            state["allowed_actions"] = (["plan"] if role == PLANNER else
                ["answer", "stop", "expand_evidence"] + (["retrieve"] if budget()["can_investigate"] else [])
                if role == REASONER else ["search", "read_memory", "expand_evidence", "finish"])
            if self.output_policy == "decoupled":
                state["allowed_actions"] = (["plan"] if role == PLANNER else
                    ["tool_plan", "finish"] if role == INVESTIGATOR else ["answer"])
            state["evidence_index"] = [{"id": d["id"], "title": d["title"]} for d in evidence.values()]
            if role == REASONER:
                state["evidence_index"] = [d for d in state["evidence_index"] if d["id"] not in last_view["visible"]]
            payload = {key: deepcopy(state[key]) for key in MEDQA_INPUT_FIELDS[role]}
            payload["memory_hits"] = memory_view(role, state["memory_hits"], evidence)
            if role==PLANNER and self.memory_selection=="needs":
                payload["memory_hits"],ranking=planner_memories()
                events.append({'tool':'memory_prefilter','version':SELECTION_VERSION,
                    'coarse_ids':sorted(memory_candidates),'ranking':ranking,
                    'presented_ids':[h['id'] for h in payload['memory_hits']],
                    'scope':'literal_metadata_and_weak_intent; not_clinical_verification'})
            if self.output_policy == "decoupled" and role != PLANNER:
                payload["memory_hits"] = [({k: h[k] for k in ("id", "source_id", "status")}
                    if h["status"] == "source_checked_not_semantically_verified" else h)
                    for h in payload["memory_hits"] if h["status"] in {"source_checked_not_semantically_verified","experience_observation"}]
                # Preserve the question/options and actual evidence; omit execution-only maps/history.
                for key in ("criteria", "memory_assessments", "tool_results", "findings", "unmapped_condition_ids"):
                    payload.pop(key, None)
                if self.memory_selection=='needs':
                    payload['memory_hits']=[({k:h[k] for k in
                        ('id','kind','status','retrieval_queries','execution_feedback','episode_context') if k in h}
                        if h.get('kind') in {'strategy','episode'} else h) for h in payload['memory_hits']]
            if payload.get("feedback", {}).get("contract_agent") not in {None, role}:
                payload["feedback"] = {k: v for k, v in payload["feedback"].items()
                                       if k not in {"contract_error", "contract_agent", "previous_output", "instruction"}}
            # 新结构不重复传candidate与审核原文；所有角色保留其真实需要的材料。
            return payload

        def prefetch(*, focus=None):
            if self.memory is None:
                return []
            metrics["memory_queries"] += 1
            memory_question = focus or question.question
            if self.memory_policy=="basic" and not focus:
                # Short stems often place the useful clinical concepts in the options.
                # Include all options equally; the grading answer is never available here.
                memory_question += "\nOptions:\n" + "\n".join(f"{k}: {v}" for k,v in question.options.items())
            try:
                values = self.embedder.embed_text(self.task_group + "\n" + memory_question)
                kinds = (["evidence"] if focus else ["evidence","strategy","episode"]) if self.memory_policy=="basic" else None
                recall_limit=self.memory_recall_limit if self.memory_selection=="needs" and not focus else self.memory_limit
                hits = operation("memory_prefetch", {"trace": trace, "corpus": self.textbooks.version,
                    "threshold": self.memory_threshold, "limit": recall_limit, "task_group": self.task_group,
                    "question": memory_question, "embedding": getattr(self.embedder, "cache_key", "fixture"), "kinds":kinds,
                    "diversify":self.memory_selection!='needs'},
                    lambda: self.memory.recall(values, trace, self.textbooks.version, self.memory_threshold,
                                              limit=recall_limit, task_group=self.task_group, **({"kinds":kinds} if kinds else {}),
                                              **({'diversify':False} if self.memory_selection=='needs' else {})))
            except Exception as exc:
                if self.journal and self.journal.fatal_error:
                    raise
                metrics["memory_read_errors"] += 1
                events.append({"tool": "memory_prefetch", "stage": "followup" if focus else "initial", "error": type(exc).__name__})
                return []
            selected = []
            for hit in hits:
                content = hit.get("content", {})
                experience = self.memory_policy=="basic" and hit.get("kind") in {"strategy","episode"}
                if (not isinstance(hit.get("id"), str) or content.get("corpus_version") != self.textbooks.version
                        or (not experience and not isinstance(content.get("id"), str))):
                    continue
                if hit["id"] in memory_candidates:
                    selected.append(hit["id"])
                    continue
                candidate = {"id": hit["id"], "summary": str(hit.get("summary", ""))[:240 if self.memory_policy=="basic" else 600],
                             "source_id": hit["id"] if experience else content["id"], "similarity": hit.get("similarity"),
                             "source_agent": hit.get("source_agent", hit.get("role", "")),
                             "created_at": hit.get("created_at"), "kind": hit.get("kind", "textbook_extract"),
                             "conditions": content.get("conditions", []), "status": "experience_candidate" if experience else "candidate_needs_source_read",
                             "validation":content.get("validation","source_read_required"),
                             "retrieval_queries": content.get("retrieval_queries", [])[:2],
                             "execution_feedback": content.get("execution_feedback", [])[:2]}
                if self.memory_selection=="needs":candidate['task_topic']=str(hit.get('task_topic',''))[:240]
                if self.memory_selection=="legacy" and focus is None and len(compact([*state["memory_hits"], candidate])) > self.memory_context_chars:
                    continue
                memory_candidates[hit["id"]] = deepcopy(hit)
                reuse_trace.mark('recalled',hit['id'],hit)
                state["memory_hits"].append(candidate)
                selected.append(hit["id"])
            metrics["memory_candidates"] = len(memory_candidates)
            metrics["memory_candidates_by_kind"] = {k:sum(("evidence" if h.get("kind","textbook_extract")=="textbook_extract" else h.get("kind"))==k
                for h in memory_candidates.values()) for k in ("evidence","strategy","episode")}
            events.append({"tool": "memory_prefetch", "before_agent": REASONER if focus else PLANNER,
                           "stage": "followup" if focus else "initial", "candidate_ids": selected,
                           "context_chars": len(compact(planner_memories()[0]
                            if self.memory_selection=='needs' and not focus else state["memory_hits"]))})
            return selected

        def reserve_followup(request):
            if not isinstance(request, dict) or not all(isinstance(request.get(k), str) and request[k].strip()
                                                       for k in ("gap", "basis", "expected_information")):
                raise ContractError("补查需说明关键缺口、判断依据和预期信息")
            identifiers(request.get("knowledge_task_ids"), task_states.items)
            identifiers(request.get("evidence_ids", []), evidence, empty=True)
            state["feedback"] = deepcopy(request)
            state["gaps"] = list(dict.fromkeys([*state["gaps"], request["gap"]]))
            metrics["followup_rounds"] += 1  # descriptive only, never grants a quota
            events.append({"tool": "request_investigation", "request": deepcopy(request)})
            return True

        def merge_findings(output):
            findings = output.get("findings", [])
            if not isinstance(findings, list):
                raise ContractError("findings必须为列表")
            merged = {f["id"]: f for f in state["findings"]}
            for finding in findings:
                if (not isinstance(finding, dict) or not isinstance(finding.get("id"), str)
                        or not isinstance(finding.get("claim"), str) or not finding["claim"].strip()):
                    raise ContractError("发现需要有效ID和判断")
                identifiers(finding.get("evidence_ids"), evidence)
                identifiers(finding.get("knowledge_task_ids"), {t["id"] for t in state["knowledge_tasks"]})
                identifiers(finding.get("used_memory_ids", []), memory_sources, empty=True)
                if finding["id"] in merged and merged[finding["id"]]["knowledge_task_ids"] != finding["knowledge_task_ids"]:
                    raise ContractError("不能用旧发现ID覆盖另一知识需求")
                merged[finding["id"]] = deepcopy(finding)
            state["findings"] = list(merged.values())

        def merge_conflicts(output):
            if "conflicts" not in output:
                return
            conflicts = output["conflicts"]
            if not isinstance(conflicts, list):
                raise ContractError("冲突须为列表")
            merged = {tuple(sorted(c["evidence_ids"])): c for c in state["conflicts"]}
            for conflict in conflicts:
                if (not isinstance(conflict, dict) or conflict.get("status") not in {"resolved", "unresolved"}
                        or not isinstance(conflict.get("reason"), str) or not conflict["reason"].strip()):
                    raise ContractError("冲突必须有状态及理由")
                ids = identifiers(conflict.get("evidence_ids"), evidence)
                merged[tuple(sorted(ids))] = deepcopy(conflict)
            state["conflicts"] = list(merged.values())

        def search(output):
            request = output.get("search")
            if not isinstance(request, dict) or not isinstance(request.get("gap"), str) or not request["gap"].strip():
                raise ContractError("检索必须说明具体缺口")
            queries = request.get("queries")
            if not isinstance(queries, list) or not queries or len(queries) > 12:
                raise ContractError("检索需1至12个按优先级排序的候选方案")
            batches = []
            executed = 0
            observations = {}
            for index, item in enumerate(queries):
                if not isinstance(item, dict):
                    raise ContractError("查询必须为对象")
                plan_id = item.get("id", "Q" + str(index + 1))
                if not isinstance(plan_id, str) or plan_id in observations:
                    raise ContractError("查询计划ID无效或重复")
                condition = item.get("run_if", "always")
                if condition not in {"always", "empty_or_duplicate", "tool_error"}:
                    raise ContractError("查询执行条件无效")
                if condition != "always":
                    after = item.get("after")
                    previous = observations.get(after) if isinstance(after, str) else None
                    if previous is None:
                        raise ContractError("备用查询必须关联本计划前面已处理的查询")
                    eligible = (bool(previous.get("error")) if condition == "tool_error" else
                        not previous.get("error") and not previous.get("skipped")
                        and previous.get("decision") not in {"need_input", "stop_resource", "stop_no_progress"}
                        and not previous.get("new_content_sources", previous.get("new_sources", previous.get("evidence_ids", []))))
                    if not eligible:
                        observations[plan_id] = {"skipped": True, "condition": condition}
                        batches.append({"query_id": plan_id, "decision": "skip_condition", "reason": condition})
                        metrics["queries_filtered"] += 1
                        continue
                query = item.get("query", "")
                if not isinstance(query, str) or not query.strip() or len(query) > 2000:
                    raise ContractError("空或过长查询")
                if not isinstance(item.get("expected_information"), str) or not item["expected_information"].strip():
                    raise ContractError("查询必须给出预期信息")
                requested = identifiers(item.get("knowledge_task_ids"), task_states.items)
                strategy_ids=item.get('strategy_memory_ids',[])
                available_strategies={h['id'] for h in state['memory_hits']
                    if h.get('kind')=='strategy' and h.get('status')=='experience_observation'}
                if not isinstance(strategy_ids,list) or any(not isinstance(i,str) or i not in available_strategies for i in strategy_ids):
                    events.append({'tool':'memory_attribution_issue','code':'unavailable_strategy_reference',
                                   'query_id':plan_id,'note':'忽略无效关联，保留有效检索计划'})
                    strategy_ids=[]
                strategy_ids=sorted(set(strategy_ids))
                required = item.get("required_inputs", [])
                if not isinstance(required, list) or any(not isinstance(i, str) for i in required):
                    raise ContractError("required_inputs须为已记录产物ID列表")
                query = " ".join(query.split())
                action, reason = choose(item, history=history, tasks=task_states.items,
                    available_inputs={f["id"] for f in state["findings"]},
                    remaining=self.limits.max_searches - metrics["retrieval_count"],
                    stalled=metrics["stalled_rounds"], limits=self.limits,
                    reopen=self.output_policy == "decoupled" or any(c["status"] == "unresolved" for c in state["conflicts"]))
                if out_of_budget():
                    action, reason = "stop_resource", "resource_exhausted"
                if action != "execute":
                    metrics["queries_reused" if action == "reuse" else "queries_filtered"] += 1
                    observation = {"query": query, "decision": action, "reason": reason, "new_sources": [], "new_content_sources": [],
                        "evidence_ids": sorted({i for h in history if query_key(h["query"]) == query_key(query)
                                                for i in h.get("evidence_ids", [])}) if action == "reuse" else []}
                    if action == "need_input":
                        for ident in requested:
                            task_states.change(ident, "waiting_input", reason)
                    batches.append(observation)
                    events.append({"tool": "query_decision", **observation})
                    observations[plan_id] = observation
                    continue
                metrics["retrieval_count"] += 1
                executed += 1
                state["attempted_queries"].append(query)
                before = set(evidence)
                previous_hashes = {d["sha256"] for d in evidence.values()}
                try:
                    # Full reranked raw documents commit before any downstream use.
                    request_key = {"query": query, "corpus": self.textbooks.version,
                        "retriever": getattr(self.textbooks, "operation_version", type(self.textbooks).__name__),
                        "candidates": getattr(self.textbooks, "candidates", None),
                        "top_k": getattr(self.textbooks, "top_k", None),
                        "attempt": sum(query_key(h["query"]) == query_key(query) for h in history)}
                    documents = operation("search_textbooks", request_key, lambda: self.textbooks.search(query))
                    if not documents:
                        metrics["search_empty"] += 1
                    accepted = []
                    for doc in documents:
                        add_document(doc)
                        accepted.append(doc["id"])
                        targets.setdefault(doc["id"], [])
                        targets[doc["id"]] = sorted(set(targets[doc["id"]]) | requested)
                    new_sources = sorted(set(evidence) - before)
                    new_content = [i for i in new_sources if evidence[i]["sha256"] not in previous_hashes]
                    # An Agent may identify a new relation in an old source. Do not equate old ID with no value.
                    new_target = (not any(set(h.get("knowledge_task_ids", [])) >= requested
                        and set(h.get("evidence_ids", [])) >= set(accepted) for h in history)) if accepted else False
                    progress = bool(new_sources or new_target)
                    observation = {"query": query, "evidence_ids": accepted,
                        "lifecycle": "completed", "result": "candidates" if accepted else "empty",
                        "knowledge_task_ids": sorted(requested), "new_sources": new_sources, "candidate_progress": progress,
                        "new_content_sources": new_content,
                        "scores": [{"id": d["id"], "bm25": d.get("bm25"), "rerank_score": d.get("rerank_score")}
                                   for d in documents]}
                    task_states.candidate(requested, accepted)
                    if progress:
                        metrics["search_progress"] += 1
                        metrics["stalled_rounds"] = 0
                    else:
                        metrics["stalled_rounds"] += 1
                except ContractError:
                    raise
                except Exception as exc:
                    if self.journal and self.journal.fatal_error:
                        raise
                    metrics["search_errors"] += 1
                    metrics["stalled_rounds"] += 1
                    observation = {"query": query, "error": getattr(exc, "tool_error_type", type(exc).__name__), "evidence_ids": [],
                                   "lifecycle": "failed", "result": "tool_error",
                                   "knowledge_task_ids": sorted(requested)}
                history.append(deepcopy(observation))
                if strategy_ids:
                    observation['strategy_memory_ids']=strategy_ids
                    history[-1]['strategy_memory_ids']=strategy_ids
                    for ident in strategy_ids:reuse_trace.mark('executed',ident,memory_candidates[ident])
                    events.append({'tool':'memory_strategy_execution','memory_ids':strategy_ids,'query_id':plan_id,
                                   'query':query,'outcome':observation['result'],
                                   'attribution':'agent_declared_reference_checked_and_action_executed'})
                observations[plan_id] = observation
                batches.append(observation)
                events.append({"tool": "search_textbooks", "search_count": metrics["retrieval_count"], **deepcopy(observation)})
                persist_state("search_result_applied")
            if not executed:
                metrics["stalled_rounds"] += 1
            state["tool_results"] = {"search_results": batches}
            return REASONER

        def execute_tool_plan(output):
            """One Investigator decision; local tools never schedule another Agent themselves."""
            action = output.get("action")
            if action in {"finish", "stop"}:
                return REASONER
            if action not in {"tool_plan", "search"}:
                raise ContractError("调查角色应输出tool_plan或finish；记忆已由程序回读")
            request = output.get("search")
            if isinstance(request, dict) and request.get("queries") == []:
                return REASONER
            return search(output)

        def expand_evidence(output):
            request = output.get("expand_evidence") or {}
            chosen = identifiers(request.get("evidence_ids"), evidence)
            if self.max_calls - metrics["model_calls"] < 1:
                return False
            remaining = self.limits.max_reads - metrics["document_reads"]
            new = chosen - expanded
            if not new or len(new) > remaining:
                raise ContractError("展开请求重复或超出回读保护上限")
            expanded.update(new)
            metrics["document_reads"] += len(new)
            metrics["expansions"] += len(new)
            events.append({"tool": "expand_evidence", "evidence_ids": sorted(new)})
            return True

        def read_memory(output, *, automatic=False):
            request = output.get("read_memory") or {}
            chosen = identifiers(request.get("memory_ids"), memory_candidates)
            if chosen & attempted_reads:
                raise ContractError("禁止重复回读同一记忆")
            experience_ids = {i for i in chosen if self.memory_policy=="basic" and memory_candidates[i].get("kind") in {"strategy","episode"}}
            if len(chosen-experience_ids) > self.limits.max_reads - metrics["document_reads"]:
                if not automatic:
                    state["tool_results"] = {"error": "document_read_budget_exhausted"}
                    return REASONER
                # Read as many chosen sources as allowed; leave fresh search independent.
                ordered = [a["memory_id"] for a in state["memory_assessments"] if a["memory_id"] in chosen-experience_ids]
                chosen = experience_ids | set(ordered[:max(0, self.limits.max_reads - metrics["document_reads"])])
            results = []
            resolved = None
            if automatic and chosen and hasattr(self.memory, "resolve"):
                try:
                    records = operation("resolve_memory", {"ids": sorted(chosen), "corpus": self.textbooks.version,
                        "task_group": self.task_group}, lambda: self.memory.resolve(sorted(chosen), trace,
                            self.textbooks.version, task_group=self.task_group))
                    resolved = {r["id"]: r for r in records}
                    events.append({"tool": "resolve_memory", "memory_ids": sorted(chosen), "resolved_ids": sorted(resolved)})
                except Exception:
                    if self.journal and self.journal.fatal_error:
                        raise
                    metrics["memory_read_errors"] += 1
                    resolved = {}
            for ident in sorted(chosen):
                if out_of_budget():
                    break
                assessment = next(a for a in state["memory_assessments"] if a["memory_id"] == ident)
                if assessment["decision"] == "reject" and not str(request.get("reason", "")).strip():
                    raise ContractError("读取Planner拒绝的候选须说明原因")
                attempted_reads.add(ident)
                hit = memory_candidates[ident]
                reuse_trace.mark('selected',ident,hit)
                if ident in experience_ids:
                    record = resolved.get(ident) if resolved is not None else None
                    if (record is None or record.get("content_hash") != hit.get("content_hash")
                            or record.get("kind") != hit.get("kind")):
                        metrics["memory_read_errors"] += 1
                        results.append({"memory_id":ident,"error":"experience_reference_unavailable_or_changed"})
                        continue
                    for candidate in state["memory_hits"]:
                        if candidate["id"]==ident:
                            candidate["status"] = "experience_observation"
                            if record["kind"]=="strategy":
                                observations = record["content"].get("queries",[])[:2]
                                candidate["retrieval_queries"] = [q["query"][:300] for q in observations]
                                candidate["execution_feedback"] = [{k:q[k][:4] if isinstance(q[k],list) else q[k]
                                    for k in ("outcome","source_ids","cited_source_ids") if k in q} for q in observations]
                            else:
                                content=record["content"]
                                candidate["episode_context"] = {
                                    "needs":[n[:80] for n in content.get("needs",[])[:2]],
                                    "gaps":[g[:80] for g in content.get("gaps",[])[:2]],
                                    "outcome":str(content.get("outcome","unknown"))[:40]}
                    metrics["memory_experience_reads"] += 1
                    reuse_trace.mark('resolved',ident,hit)
                    results.append({"memory_id":ident,"kind":record["kind"],"status":"experience_observation"})
                    continue
                metrics["document_reads"] += 1
                metrics['memory_source_check_attempts'] += 1
                try:
                    if resolved is not None:
                        record = resolved.get(ident)
                        if record is None or any(record.get("content", {}).get(k) != hit["content"].get(k)
                                                 for k in ("id", "sha256", "corpus_version")):
                            raise ValueError("memory_reference_unavailable_or_changed")
                        hit = record
                    source = operation("read_source", {"id": hit["content"]["id"], "corpus": self.textbooks.version},
                        lambda: self.textbooks.read(hit["content"]["id"]))
                    if source["sha256"] != hit["content"]["sha256"]:
                        raise ValueError("source_changed")
                    accepted = add_document(source)
                except Exception as exc:
                    if self.journal and self.journal.fatal_error:
                        raise
                    metrics["source_read_errors"] += 1
                    for candidate in state["memory_hits"]:
                        if candidate["id"] == ident:
                            candidate["status"] = "source_unavailable"
                    results.append({"memory_id": ident, "error": getattr(exc, "tool_error_type", type(exc).__name__)})
                    continue
                if accepted:
                    reuse_trace.mark('resolved',ident,hit)
                    memory_sources[ident] = source["id"]
                    targets.setdefault(source["id"], [])
                    targets[source["id"]] = sorted(set(targets[source["id"]]) | set(assessment["knowledge_task_ids"]))
                    for candidate in state["memory_hits"]:
                        if candidate["id"] == ident:
                            candidate["status"] = "source_checked_not_semantically_verified"
                    results.append({"memory_id": ident, "evidence_id": source["id"]})
            metrics["memory_source_valid"] = len(memory_sources)
            state["tool_results"] = {"memory_reads": results}
            events.append({"tool": "read_memory", "automatic": automatic, "results": deepcopy(results)})
            return REASONER

        def prepare_memories():
            if self.output_policy == "decoupled":
                selected = [a["memory_id"] for a in state["memory_assessments"] if a["decision"] == "read"]
                for ident in selected:reuse_trace.mark('planner_accepted',ident,memory_candidates[ident])
                if self.memory_selection=='needs':
                    selected,ranking=select_memories(question.question,memory_candidates,state['memory_assessments'],
                        [t['id'] for t in state['knowledge_tasks']],self.memory_limit,
                        task_goals=[t['goal'] for t in state['knowledge_tasks']])
                    events.append({'tool':'memory_selection','version':SELECTION_VERSION,'selected_ids':selected,
                                   'ranking':ranking,'weights':{'R':.45,'T':.35,'C':.20},
                                   'coverage':'potential_only; not_medical_support'})
                    for assessment in state['memory_assessments']:
                        if assessment['decision']=='read' and assessment['memory_id'] not in selected:
                            assessment.update(decision='reject',reason='runtime_rank_or_budget_not_selected')
                for ident in selected:reuse_trace.mark('selected',ident,memory_candidates[ident])
                if selected:
                    read_memory({"read_memory": {"memory_ids": selected}}, automatic=True)
                    persist_state("selected_memories_prepared")

        def optional_followup(output, usage):
            """Reuse this response's decision. Never spend a judge call or discard its answer."""
            nonlocal followup_view, followup_previous_view
            request = output.get("followup_request")
            if request is None:
                return False
            reason = None
            if retained_result is None:
                reason = "missing_current_answer"
            elif usage.get("source") not in {"model", "fixture"} or usage.get("finish_reason") == "length":
                reason = "usage_unknown_or_truncated"
            elif metrics["followup_rounds"] >= 1:
                reason = "followup_fuse"
            elif self.max_calls - metrics["model_calls"] < 1 or out_of_budget():
                reason = "answer_update_budget_unavailable"
            elif not isinstance(request, dict) or request.get("may_change_answer") is not True:
                reason = "no_decisive_gap_declared"
            if reason is None:
                try:
                    if any(not isinstance(request.get(k), str) or not request[k].strip()
                           for k in ("gap", "basis", "expected_information")):
                        raise ContractError("补查缺口、依据和预期信息不能为空")
                    wanted = identifiers(request.get("knowledge_task_ids"), task_states.items)
                    identifiers(request.get("evidence_ids", []), evidence, empty=True)
                    proposals = request.get("queries")
                    if not isinstance(proposals, list) or len(proposals) > 3:
                        raise ContractError("补查候选查询须为至多3项的列表")
                    queries = []
                    for proposal in proposals:
                        query = proposal if isinstance(proposal, str) else proposal.get("query") if isinstance(proposal, dict) else None
                        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
                            raise ContractError("补查查询无效")
                        item = {"query": query, "knowledge_task_ids": sorted(wanted),
                                "expected_information": request["expected_information"]}
                        action, _ = choose(item, history=history, tasks=task_states.items, available_inputs=set(),
                            remaining=self.limits.max_searches - metrics["retrieval_count"],
                            stalled=metrics["stalled_rounds"], limits=self.limits, reopen=True)
                        if action == "execute" and query_key(query) not in {query_key(q["query"]) for q in queries}:
                            queries.append(item)
                    if not queries and self.memory is None:
                        reason = "no_new_executable_query"
                except ContractError:
                    reason = "invalid_followup_plan"
            if reason is not None:
                events.append({"tool": "followup_decision", "decision": "skip", "reason": reason})
                return False
            metrics["followup_rounds"] += 1
            events.append({"tool": "followup_decision", "decision": "execute", "queries": [q["query"] for q in queries]})
            previous_view = deepcopy(last_view)
            # Re-query memory for the *gap*, not just the initial question. No extra Agent turn.
            fresh = prefetch(focus=question.question + "\n" + request["gap"] + "\n" + request["expected_information"])
            unread = [ident for ident in fresh if ident not in attempted_reads]
            for ident in unread:
                assessment = next((a for a in state["memory_assessments"] if a["memory_id"] == ident), None)
                if assessment is None:
                    state["memory_assessments"].append({"memory_id": ident, "decision": "read",
                        "reason": "按当前关键缺口召回，仅校验来源，适用性由Reasoner判断", "knowledge_task_ids": sorted(wanted)})
                else:
                    assessment["knowledge_task_ids"] = sorted(set(assessment["knowledge_task_ids"]) | wanted)
            if unread:
                read_memory({"read_memory": {"memory_ids": unread, "reason": "当前缺口与初始规划不同，按来源回读核对"}}, automatic=True)
            if queries:
                search({"search": {"gap": request["gap"], "queries": queries}})
            state["gaps"] = list(dict.fromkeys([*state["gaps"], request["gap"]]))
            state["feedback"] = {"followup_completed": True, "gap": request["gap"],
                "previous_answer": retained_result["answer"], "instruction": "按现有证据更新答案并披露缺口，不再补查。"}
            handoff_request = {"version": HANDOFF_VERSION, "documents": evidence,
                "previous": previous_view, "citations": retained_result["evidence_result"]["citations"],
                "request": request, "question": question.question, "targets": targets,
                "conflicts": [*state["conflicts"], *(output.get("conflicts", []) if isinstance(output.get("conflicts", []), list) else [])],
                "enabled": self.processor.enabled, "span_chars": self.processor.span_chars,
                "max_chars": self.processor.max_chars, "max_documents": self.processor.max_documents}
            prepared = operation("evidence_handoff", handoff_request, lambda: build_handoff(self.processor, evidence,
                previous=previous_view, citations=handoff_request["citations"], request=request,
                question=question.question, targets=targets, conflicts=handoff_request["conflicts"]))
            validate_view(prepared, evidence)
            metrics["evidence_processor_seconds"] += prepared["cpu_seconds"]
            delta = visible_delta(prepared, previous_view, evidence)
            events.append({"tool": "evidence_handoff", **deepcopy(prepared["handoff"])})
            # Actual novel source text is required, including unseen spans from an old source.
            if not delta["characters"]:
                events.append({"tool": "answer_preserved", "reason": "followup_no_new_visible_evidence"})
                return False
            followup_view, followup_previous_view = prepared, previous_view
            state["feedback"]["new_visible_evidence_ids"] = delta["evidence_ids"]
            if prepared["handoff"]["omitted_anchor_ids"]:
                state["feedback"]["omitted_prior_citation_ids"] = prepared["handoff"]["omitted_anchor_ids"]
            if prepared["handoff"]["conflict_ids"]:
                state["feedback"]["reported_conflict_evidence_ids"] = prepared["handoff"]["conflict_ids"]
            if prepared["handoff"]["omitted_conflict_ids"]:
                state["feedback"]["omitted_conflict_evidence_ids"] = prepared["handoff"]["omitted_conflict_ids"]
            persist_state("optional_followup_completed")
            return True

        try:
            prefetch()
            role = PLANNER
            while final is None:
                exhausted = out_of_budget()
                if exhausted:
                    final = stop(exhausted, "已达到程序预算，不再开始新调用")
                    break
                # 调查动作后直接核验，不再固定追加调查角色的整理调用。
                if role == INVESTIGATOR and self.max_calls - metrics["model_calls"] < 2:
                    final = stop("budget_exhausted", "剩余调用不足以完成调查与核验")
                    break
                payload = payload_for(role)
                wire = compact(payload) if self.mode == "structured" else render_text(payload)
                prompts = getattr(self.agents, "prompts", self.default_prompts)
                if len(wire) + len(prompts[role]) > self.max_prompt_chars:
                    final = stop("context_budget_exhausted", "角色输入超过统一字符预算，未静默截断原文")
                    break
                if role == REASONER and followup_view is not None:
                    delta = visible_delta(last_view, followup_previous_view, evidence)
                    if not delta["characters"]:
                        final = stop("followup_no_new_visible_evidence", "补查正文未实际进入本次模型输入，保留已有答案")
                        break
                    metrics["followup_new_visible_chars"] += delta["characters"]
                    metrics["followup_new_visible_sources"] += len(delta["evidence_ids"])
                    events.append({"tool": "followup_delivery", "new_visible_ids": delta["evidence_ids"],
                                   "new_visible_chars": delta["characters"]})
                request_id = trace + ":" + str(metrics["model_calls"] + 1)
                envelope = {"protocol": self.protocol, "request_id": request_id, "task_id": question.id,
                            "target": role, "action": "run_role"}
                semantic_object = message(role, payload, trace=trace, operation_id=request_id,
                    step=metrics["model_calls"] + 1, max_steps=self.max_calls,
                    semantic_refs={"memory_ids": [h["id"] for h in state["memory_hits"]],
                                   "evidence_processor": last_view.get("version")})
                payload = semantic_object["precise"]
                control = compact(semantic_object["control"])
                metrics["protocol_control_chars"] += len(control)
                metrics["protocol_control_bytes"] += len(control.encode())
                metrics["model_calls"] += 1
                if role==PLANNER:
                    for hit in payload['memory_hits']:
                        planner_presented.add(hit['id'])
                        reuse_trace.mark('presented',hit['id'],memory_candidates[hit['id']])
                else:
                    visible_sources={d['id'] for d in payload.get('evidence_pool',[])}
                    for hit in payload['memory_hits']:
                        ident=hit['id']
                        if hit.get('status')=='experience_observation' or hit.get('source_id') in visible_sources:
                            reuse_trace.mark('delivered',ident,memory_candidates[ident])
                metrics["evidence_raw_delivery_chars"] += last_view["raw_chars"]
                metrics["messages"] += 1
                metrics["request_chars"] += len(wire)
                metrics["request_bytes"] += len(wire.encode())
                metrics["communication_chars"] += len(wire)
                visited.add(role)
                attempt = {"agent": role, "request_id": request_id, "input": deepcopy(payload),
                           "search_count": metrics["retrieval_count"], "system_chars": len(prompts[role]),
                           "control": semantic_object["control"], "semantic": semantic_object["semantic"]}
                metrics["system_chars"] += len(prompts[role])
                metrics["system_bytes"] += len(prompts[role].encode())
                events.append(attempt)
                model_started = time.monotonic()
                try:
                    if hasattr(self.agents, "call_with_budget"):
                        raw, usage = self.agents.call_with_budget(role, payload, self.mode,
                            remaining_seconds=max(.01, self.deadline_seconds - (time.monotonic() - started)
                                                  - getattr(self.agents, "historical_seconds", 0)
                                                  - getattr(self.journal, "historical_seconds", 0)),
                            remaining_tokens=max(1, self.max_question_tokens - metrics["total_tokens"]))
                    else:
                        raw, usage = self.agents.call(role, payload, self.mode)
                except Exception as exc:
                    attempt["error"] = type(exc).__name__
                    attempt["duration_seconds"] = time.monotonic() - model_started
                    raise ModelCallError() from None
                attempt["duration_seconds"] = time.monotonic() - model_started
                response = raw if isinstance(raw, str) else compact(raw)
                attempt["response"] = response
                metrics["messages"] += 1
                length = int(usage.get("response_chars", len(response)))
                metrics["response_chars"] += length
                metrics["response_bytes"] += len(response.encode())
                metrics["communication_chars"] += length
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens", "reasoning_tokens"):
                    metrics[key] += int(usage.get(key, 0))
                metrics["usage_sources"].append(usage.get("source", "unknown"))
                attempt["usage"] = deepcopy(usage)
                if usage.get("finish_reason") == "length" and self.output_policy == "strict":
                    raise ContractError("模型输出被截断，保留实际响应与用量，不解析为成功")
                if usage.get("source") not in {"model", "fixture"} and not (self.output_policy == "decoupled" and role == REASONER):
                    final = stop("usage_unavailable", "模型用量未知或不完整，不继续调用或写入记忆")
                    break
                output = None
                try:
                    output = json.loads(raw) if isinstance(raw, str) else raw
                    if not isinstance(output, dict) or output.get("_error"):
                        raise ContractError("角色输出不是有效对象或报告生成错误")
                    if self.output_policy == "decoupled" and "question_id" not in output:
                        output["question_id"] = question.id
                        metrics["protocol_maps"] += 1
                        metrics["deterministic_repairs"] += 1
                        events.append({"tool": "protocol_mapping", "agent_role": role,
                                       "mappings": ["missing question_id bound to current request"]})
                    if output.get("question_id") != question.id:
                        raise ContractError("角色输出题目ID不匹配", category="identity")
                    attempt["output"] = deepcopy(output)
                    if self.output_policy == "decoupled" and role == REASONER:
                        reviewed = deepcopy(output)
                        # Known conflicts remain visible to the optional full audit.
                        if state["conflicts"]:
                            reviewed["conflicts"] = deepcopy(state["conflicts"])
                        final = split_result(reviewed, question, task_states.items, state["key_conditions"], evidence,
                                             last_view["visible"], set(memory_sources.values()))
                        if usage.get("source") not in {"model", "fixture"} or usage.get("finish_reason") == "length":
                            final["protocol_result"]["issues"].append({"category": "execution", "code": "usage_unknown_or_truncated",
                                "path": "usage", "message": "答案保留，用量未知或输出截断，不写可靠记忆"})
                            final["protocol_result"]["status"] = "issues"
                            final["evidence_result"]["memory_eligible"] = False
                        if final["evidence_result"]["strict_audit_status"] == "passed":
                            task_states.review(reviewed)
                        if final.get("answer") in question.options:
                            if retained_result is not None:
                                events.append({"tool": "followup_answer_updated"})
                            retained_result, retained_output = deepcopy(final), deepcopy(reviewed)
                        elif retained_result is not None:
                            events.append({"tool": "answer_preserved", "reason": "invalid_followup_answer",
                                           "discarded_protocol": deepcopy(final["protocol_result"])})
                            final = deepcopy(retained_result)
                        persist_state("answer_retained_independently")
                        if optional_followup(reviewed, usage):
                            final, role = None, REASONER
                        continue
                    output, mappings = normalize_response(role, output, evidence)
                    if mappings:
                        metrics["protocol_maps"] += 1
                        metrics["deterministic_repairs"] += len(mappings)
                        attempt["normalized_output"] = deepcopy(output)
                        events.append({"tool": "protocol_mapping", "agent_role": role, "mappings": mappings})
                    identifiers(output.get("used_memory_ids", []), memory_sources, empty=True)
                    if "gaps" in output:
                        if not isinstance(output["gaps"], list) or any(not isinstance(g, str) for g in output["gaps"]):
                            raise ContractError("缺口须为文本列表")
                        # 没有逐项解决协议时不允许空列表悄悄抹掉已有缺口；成功回答时统一关闭。
                        state["gaps"] = list(dict.fromkeys([*state["gaps"], *output["gaps"]]))
                    if role == PLANNER:
                        if self.memory_selection=='needs' and isinstance(output.get('memory_assessments'),list):
                            # Compact Planner choices omit mechanically implied
                            # labels/reasons. Fill internal bookkeeping locally.
                            compact_choices=False
                            for assessment in output['memory_assessments']:
                                if isinstance(assessment,dict):
                                    compact_choices |= 'decision' not in assessment or 'reason' not in assessment
                                    assessment.setdefault('decision','read')
                                    assessment.setdefault('reason','planner_selected_for_current_need' if assessment['decision']=='read'
                                                          else 'planner_rejected_candidate')
                            if compact_choices:
                                attempt['normalized_output']=deepcopy(output)
                                events.append({'tool':'planner_memory_defaults','source':'runtime',
                                    'fields':['decision','reason'],'extra_model_calls':0})
                        unmapped = validate_plan(output, planner_presented if self.memory_selection=='needs' else memory_candidates,
                                                 sparse_memories=self.memory_selection=='needs')
                        for key in ("key_conditions", "knowledge_tasks", "criteria", "memory_assessments"):
                            state[key] = deepcopy(output[key])
                        if self.memory_selection=='needs':
                            assessed={a['memory_id'] for a in state['memory_assessments']}
                            state['memory_assessments'].extend({'memory_id':ident,'decision':'reject',
                                'reason':'not_proposed_by_planner','knowledge_task_ids':[]}
                                for ident in sorted(planner_presented-assessed))
                        state["unmapped_condition_ids"] = unmapped
                        task_states = TaskStates(state["knowledge_tasks"], resource_snapshot=budget)
                        if state["feedback"].get("contract_agent") == PLANNER:
                            state["feedback"] = {}
                        metrics["memory_rejected"] = sum(a["decision"] == "reject" for a in state["memory_assessments"])
                        prepare_memories()
                        role = INVESTIGATOR
                    elif role == INVESTIGATOR:
                        if self.output_policy == "decoupled":
                            role = execute_tool_plan(output)
                            persist_state("investigation_plan_applied")
                            continue
                        merge_findings(output)
                        merge_conflicts(output)
                        action = output.get("action")
                        if action == "search":
                            if self.max_calls - metrics["model_calls"] < 1:
                                final = stop("budget_exhausted", "不再启动无法完成最终核验的工具请求")
                            else:
                                role = search(output)
                        elif action == "read_memory":
                            if self.max_calls - metrics["model_calls"] < 1:
                                final = stop("budget_exhausted", "剩余预算不足以核验回读结果并回答")
                            else:
                                role = read_memory(output)
                        elif action == "expand_evidence":
                            expand_evidence(output)
                            role = REASONER
                        elif action in {"finish", "stop"}:
                            role = REASONER
                        else:
                            raise ContractError("无效调查动作")
                    else:
                        decision = output.get("decision")
                        merge_findings(output)
                        merge_conflicts(output)
                        if "need_reviews" in output:
                            task_states.review_needs(output["need_reviews"], evidence, last_view["visible"])
                        if decision == "answer":
                            proposed = output.get("final_answer") or {}
                            validate_visible_quotes(output, last_view["visible"])
                            audit = validate_answer({**output, "conflicts": state["conflicts"]}, question,
                                {t["id"] for t in state["knowledge_tasks"]}, state["key_conditions"], evidence,
                                set(memory_sources.values()))
                            task_states.review(output)
                            for claim in output["claims"]:
                                for quote in claim["support_quotes"]:
                                    protected_quotes.setdefault(quote["evidence_id"], []).append(quote["quote"])
                            final = {**deepcopy(proposed), "question_id": question.id,
                                     "status": audit["status"], "evidence_status": audit["evidence_status"],
                                     "declared_evidence_status": audit["declared_evidence_status"],
                                     "audit_warnings": audit["audit_warnings"],
                                     "answer_basis": audit["answer_basis"], "gaps": audit["gaps"],
                                     "verification": {"checks": output["checks"], "condition_checks": output["condition_checks"],
                                                      "option_comparison": output["option_comparison"],
                                                      "critical_gaps": output["critical_gaps"],
                                                      "claims": deepcopy(output["claims"]),
                                                      "grounded_claim_ids": audit["grounded_claim_ids"],
                                                      "conflicts": deepcopy(state["conflicts"]),
                                                      "review_type": "same_agent_reasoning_and_verification"}}
                        elif decision == "retrieve":
                            request = output.get("followup_request")
                            if isinstance(request, dict) and isinstance(request.get("gap"), str):
                                state["gaps"] = list(dict.fromkeys([*state["gaps"], request["gap"]]))
                            if (self.max_calls - metrics["model_calls"] < 2
                                    or metrics["retrieval_count"] >= self.limits.max_searches
                                    or metrics["stalled_rounds"] >= self.limits.max_stalled_rounds):
                                # Invalid proposal cannot grant resources. Reserve the remaining call for an answer.
                                if self.max_calls - metrics["model_calls"] >= 1:
                                    state["feedback"] = {"instruction": "无可执行补查预算/进展；本次必须按现有依据尽力回答并披露缺口。"}
                                    role = REASONER
                                else:
                                    final = stop("budget_exhausted", "没有已审核答案；资源不足以继续调查")
                            elif reserve_followup(request):
                                role = INVESTIGATOR
                        elif decision == "expand_evidence":
                            if not expand_evidence(output):
                                final = stop("budget_exhausted", "无剩余调用读取展开结果")
                            role = REASONER
                        elif decision == "stop":
                            reason = output.get("abstain_reason")
                            if not isinstance(reason, str) or not reason.strip():
                                raise ContractError("确实无法比较选项时stop须说明abstain_reason，证据不足应披露后尽力答题")
                            final = stop("abstained", reason)
                        else:
                            raise ContractError("无效核验决策")
                except (ContractError, json.JSONDecodeError) as exc:
                    issue = Issue(getattr(exc, "category", "agent_output"), role, str(exc))
                    events.append({"tool": "issue", **issue.public()})
                    if self.output_policy == "decoupled":
                        if issue.category in {"identity", "integrity"} or role == REASONER:
                            raise
                        events.append({"tool": "baseline_fallback", "agent_role": role,
                                       "reason": str(exc), "llm_call": False})
                        if role == PLANNER:
                            basic_plan()
                            task_states = TaskStates(state["knowledge_tasks"], resource_snapshot=budget)
                            prepare_memories()
                            role = INVESTIGATOR
                        else:
                            if not history:
                                basic_search()
                            role = REASONER
                        continue
                    remaining = self.max_calls - metrics["model_calls"]
                    reserve = 2 if role == PLANNER else 1 if role == INVESTIGATOR else 0
                    # Same error without new input is not eligible for an endless repair loop.
                    if (issue.fingerprint in repair_fingerprints or remaining <= reserve
                            or metrics["contract_repairs"] >= self.max_contract_repairs or out_of_budget()):
                        raise ContractError(str(exc), category=issue.category) from None
                    repair_fingerprints.add(issue.fingerprint)
                    metrics["contract_repairs"] += 1
                    pending_repairs.add(role)
                    metrics["semantic_reviews"] += issue.category in {"attribution", "evidence_judgment"}
                    state["feedback"] = {"contract_agent": role, "contract_error": str(exc),
                        "category": issue.category, "previous_output": deepcopy(output) if isinstance(output, dict) else None,
                        "instruction": "只修复受影响字段。重新核对支持含义；允许partial/insufficient，禁止改标签掩盖缺证。"}
                    continue
                called_role = attempt["agent"]
                if called_role in pending_repairs:
                    metrics["recovered_action_errors"] += 1
                    pending_repairs.remove(called_role)
                if state["feedback"].get("contract_agent") == called_role:
                    state["feedback"] = {}
                persist_state("agent_output_applied")
            if final.get("answer") in question.options:
                if self.output_policy == "decoupled" and retained_output is not None:
                    output = retained_output
                used = set(final["evidence_ids"])
                final["reused_memory_ids"] = sorted(i for i, source in memory_sources.items() if source in used)
                metrics["memory_used"] = len(final["reused_memory_ids"])
                if (self.memory_policy=="strict" and self.memory and self.memory_write and final["evidence_status"] == "supported"
                        and (self.output_policy == "strict" or final.get("evidence_result", {}).get("memory_eligible"))):
                    # 整理已审核来源；不另调LLM，不保存答案字母或生成医学结论。
                    for source_id in sorted(used):
                        try:
                            doc = evidence[source_id]
                            excerpts = list(dict.fromkeys(q["quote"] for c in output["claims"]
                                for q in c["support_quotes"] if q["evidence_id"] == source_id))[:3]
                            conditions = [c["text"] for c in state["key_conditions"]]
                            memory_text = memory_embedding_text(doc, question.question, conditions, self.task_group, excerpts)
                            self.memory.put(doc, self.embedder.embed_text(memory_text), trace,
                                topic=question.question, source_agent=REASONER, task_group=self.task_group,
                                review_policy=self.answer_policy,
                                conditions=conditions, excerpts=excerpts,
                                retrieval_queries=[e["query"] for e in events if e.get("tool") == "search_textbooks"
                                                   and source_id in e.get("evidence_ids", [])],
                                execution_feedback=[{k: e[k] for k in ("query", "evidence_ids", "error") if k in e}
                                                    for e in events if e.get("tool") == "search_textbooks"])
                            metrics["memory_written"] += 1
                            metrics["memory_written_by_kind"]["evidence"] += 1
                        except Exception as exc:
                            metrics["memory_write_errors"] += 1
                            events.append({"tool": "memory_put", "evidence_id": source_id, "error": type(exc).__name__})
        except ContractError as exc:
            final = stop("protocol_error", str(exc))
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            final = stop("protocol_error", type(exc).__name__)
        except Exception as exc:
            final = stop("runtime_error", type(exc).__name__)
        if self.memory_policy=="basic" and self.memory and self.memory_write:
            known_usage = len(metrics["usage_sources"])==metrics["model_calls"] and bool(metrics["usage_sources"]) and all(s in {"model","fixture"} for s in metrics["usage_sources"])
            if known_usage:
                for unit in build_units(question,state,final,evidence,history,metrics,
                                        actionable_strategy=self.memory_selection=='needs'):
                    try:
                        ident = self.memory.put_unit(unit,self.embedder.embed_text(unit_embedding_text(unit)),trace)
                        metrics["memory_written"] += 1
                        metrics["memory_written_by_kind"][unit["kind"]] += 1
                        events.append({"tool":"memory_put_unit","kind":unit["kind"],"reference":ident,
                                       "staged":ident.startswith("pending:")})
                    except Exception as exc:
                        metrics["memory_write_errors"] += 1
                        events.append({"tool":"memory_put_unit","kind":unit["kind"],"error":type(exc).__name__})
            else:
                events.append({"tool":"memory_write_skip","reason":"model_usage_unknown"})
        if self.output_policy == "decoupled":
            if "answer_result" not in final:
                final.update(failure_results(final, events))
            protocol = final["protocol_result"]
            protocol["issues"].extend(deepcopy(e) for e in events if e.get("tool") == "issue")
            protocol["issues"].extend({"category": "optional_followup", "code": e["reason"], "blocks_answer": False}
                for e in events if e.get("tool") == "answer_preserved"
                and e["reason"] in {"runtime_error", "protocol_error", "invalid_followup_answer"})
            protocol["repairs"].extend(deepcopy(e) for e in events if e.get("tool") in {"baseline_fallback", "protocol_mapping"})
            if protocol["issues"] and protocol["status"] == "valid":
                protocol["status"] = "issues"
            decisions = [e for e in events if e.get("tool") == "followup_decision"]
            preserved = [e for e in events if e.get("tool") == "answer_preserved"]
            final["followup_result"] = {"status": "updated" if any(e.get("tool") == "followup_answer_updated" for e in events)
                else "answer_preserved" if preserved else "skipped" if decisions else "not_requested",
                "rounds": metrics["followup_rounds"],
                "reason": preserved[-1]["reason"] if preserved else decisions[-1].get("reason") if decisions else None}
        task_states.close(final.get("status", "unknown"))
        state["task_status"] = deepcopy(task_states.items)
        persist_state("question_closed")
        for event in events:
            if "input" in event:
                pool = event["input"].get("evidence_pool", [])
                metrics["evidence_view_chars"] += sum(len(d["text"]) for d in pool)
        metrics["evidence_raw_chars"] = sum(len(d["text"]) for d in evidence.values())
        metrics["evidence_exact_duplicates"] = sum(r["action"] == "exact_duplicate" for r in last_view.get("records", []))
        metrics["evidence_similar_groups"] = len(last_view.get("similar_groups", []))
        metrics["usage_missing_calls"] = metrics["model_calls"] - len(metrics["usage_sources"])
        metrics["visited_agents"] = sorted(visited)
        metrics["elapsed_seconds"] = time.monotonic() - started
        metrics["embedding_cache_hits"] = getattr(self.embedder, "cache_hits", 0) - base_cache_hits
        metrics["embedding_encoded_texts"] = getattr(self.embedder, "encoded_texts", 0) - base_encoded
        metrics['embedding_seconds']=getattr(self.embedder,'encoded_seconds',0.)-base_embedding_seconds
        metrics["state_transfers"] = getattr(self.memory, "transfers", 0) - base_transfers
        metrics["state_bytes"] = getattr(self.memory, "bytes", 0) - base_bytes
        metrics['state_vector_transfers']=getattr(self.memory,'vector_transfers',0)-base_vectors
        cited_sources={c['evidence_id'] for c in final.get('evidence_result',{}).get('citations',[])
                       if c.get('source_verified') and c.get('visible')}
        for ident,source in memory_sources.items():
            if source in cited_sources:reuse_trace.mark('cited',ident,memory_candidates[ident])
        metrics['memory_funnel']=reuse_trace.export()
        events.append({'tool':'memory_reuse_trace','stages':reuse_trace.references(),
                       'note':'进入请求、执行关联或原文引用不等于因果收益；需与关闭记忆对照。'})
        metrics["tool_counts"] = {k: v - base_stats.get(k, 0) for k, v in getattr(self.textbooks, "stats", {}).items()}
        metrics["quality_note"] = "记忆命中/引用不等于正确复用；同一Reasoner自检不等于独立医学审核。"
        return {**final, "trace_id": trace, "protocol": self.protocol, "answer_policy": self.answer_policy,
                "output_policy": self.output_policy,
                "mode": self.mode, "task_group": self.task_group,
                "metrics": metrics, "tasks": state["knowledge_tasks"], "task_results": state["task_status"], "events": events,
                "task_transitions": task_states.transitions, "evidence_processing": last_view,
                "retrieval_history": history,
                "question_task": {"kind": "answer", "lifecycle": "completed" if final.get("answer") is not None else
                    "closed_incomplete" if final.get("status") == "abstained" else "failed",
                    "can_complete": final.get("answer") is not None, "can_close": True,
                    "result": final.get("status")}}
