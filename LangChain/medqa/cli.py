"""单一实验入口：预检、真实检索探针、题目流、四条件实验与报告。"""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
import math
import os
import re
import random
import statistics
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from .data import Dataset, DEVELOPMENT_IDS
from .retrieval import Textbooks
from .runtime import ChatRoles, Runtime, PROTOCOL, ANSWER_POLICY, protocol_for, answer_policy_for
from .experiment import validate_seed as _validate_seed
from .evaluation import EvaluationError
from .checkpoint import Checkpoint, CheckpointError, DurableCalls, DurableTools, PendingMemory, atomic_text, sync_directory


PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent


class ConfigurationError(ValueError):
    """仅受控本地配置诊断可直接展示，不暴露SDK异常。"""


def validate_seed(*args, **kwargs):
    try:
        return _validate_seed(*args, **kwargs)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from None


def dump(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def make_parser():
    p = argparse.ArgumentParser(description="MedQA/Textbooks 三Agent协作：预取记忆→Planner→Investigator→VerifierReasoner")
    p.add_argument("command", choices=["preflight", "audit", "probe", "streams", "run", "compare",
                                       "prepare-judgments", "evaluate-retrieval"])
    p.add_argument("--config", type=Path, help="JSON实验参数；路径相对于项目根目录，显式命令行参数优先")
    p.add_argument("--benchmark", type=Path, default=WORKSPACE / "MedRGAG/benchmark.json")
    p.add_argument("--textbooks", type=Path, default=WORKSPACE / "medrgag-data/corpus/textbooks")
    p.add_argument("--retrieval-python", type=Path, default=WORKSPACE / "miniconda3/envs/medrgag/bin/python")
    p.add_argument("--java-home", type=Path, default=WORKSPACE / "medrgag-data/runtime/java21")
    p.add_argument("--ranker", type=Path, default=WORKSPACE / "medrgag-data/models/ncbi/MedCPT-Cross-Encoder")
    p.add_argument("--embedding", type=Path, default=PROJECT / "models/Qwen3-Embedding-0.6B")
    p.add_argument('--model-name',help='显式选择模型；优先于MEDQA_MODEL/DEEPSEEK_MODEL')
    p.add_argument('--model-base-url',help='OpenAI兼容地址，例如http://127.0.0.1:8010/v1；不会失败后回退云API')
    p.add_argument('--model-api-key-env',help='读取指定环境变量中的key；本机服务默认使用local-moreagent占位key')
    p.add_argument('--model-timeout-seconds',type=float,default=60,help='单次模型请求超时，仍受当前题剩余期限限制')
    p.add_argument('--model-precision',help='实验记录用的服务精度标签，例如bitsandbytes-int8；须与实际部署一致')
    p.add_argument("--output", type=Path, help="新实验需新目录；续跑需指定原结果目录")
    p.add_argument("--resume", action="store_true", help="恢复原目录的run/compare；校验原配置，重用已保存响应")
    p.add_argument("--retry-unknown-calls", action="store_true", help="仅续跑：允许重发响应未落盘的请求，可能重复计费，用量标为未知")
    p.add_argument("--limit", type=int, help="明确限制本次题数；run/compare 必填，或用 --all")
    p.add_argument("--all", action="store_true", help="明确请求完整所选题目流，可能有较高模型费用")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--condition-seed", type=int, default=42, help="随机化四条件执行顺序，记录种子；重复实验更换该种子")
    p.add_argument("--split", choices=["heldout", "development", "all"], default="heldout",
                   help="默认排除已用于调试的题目；全量流选择all；调试同一题选择development")
    p.add_argument("--ids", type=Path, help="streams 生成的固定题目 ID 顺序文件")
    p.add_argument("--mode", choices=["structured", "text"], default="structured")
    p.add_argument("--output-policy", choices=["strict", "decoupled"], default="decoupled",
                   help="decoupled分别输出答案/证据/协议结果，审计问题不触发完整重答")
    p.add_argument("--context-profile", choices=["compact", "full"], default="compact",
                   help="两者保留最小JSON形状；compact省去重复输入字段列表，full提供完整字段说明")
    p.add_argument('--workspace-policy', choices=['legacy', 'role_views_v1'], default='legacy',
                   help='role_views_v1仅在structured条件启用共享任务状态与按角色展开；Text基线不变')
    p.add_argument('--workflow', choices=['legacy','need_first'], default='legacy',
                   help='need_first先规划，再按需求召回记忆、核验覆盖、仅检索缺口；两个通信模式共用')
    p.add_argument('--max-investigation-rounds', type=int, default=3,
                   help='need_first调查轮数保护上限1至3，通常仍是三次模型调用')
    p.add_argument("--task-group", default="medqa", help="关联任务组/记忆适用域，四条件必须相同")
    p.add_argument("--memory", choices=["on", "off"], default="on")
    p.add_argument("--memory-write", choices=["on", "off"], default="on", help="off时仍可读取历史记忆，但不保存当前题的新记忆")
    p.add_argument("--memory-policy",choices=["strict","basic"],default="strict",
                   help="basic保存分类型来源片段、检索经验和过程记录；strict保留原整题审核准入")
    p.add_argument("--memory-source", type=Path, help="冻结积累库；compare的两个memory-on条件各用独立副本")
    p.add_argument("--memory-namespace", help="积累和评估必须使用相同命名空间；未指定时沿用条件名称")
    p.add_argument("--memory-threshold", type=float, default=.65)
    p.add_argument('--memory-selection',choices=['legacy','needs'],default='legacy',
                   help='needs启用需求选择；need_first按至多3个来源/9条记忆保留片段；legacy流程保留旧路径')
    p.add_argument('--memory-recall-limit',type=int,default=10,help='每次需求召回条数（3至10）；need_first另按唯一来源控制回读')
    p.add_argument("--max-calls", type=int, default=8, help="每题模型请求保护上限3至64；所有角色/纠错共用")
    p.add_argument("--max-searches", type=int, default=6, help="动态查询的失控保护上限，不预分配阶段配额")
    p.add_argument("--max-reads", type=int, default=4)
    p.add_argument("--max-stalled-rounds", type=int, default=2)
    p.add_argument("--max-contract-repairs", type=int, default=4, help="不同可恢复错误共用的资源保护上限；同错不循环")
    p.add_argument("--evidence-processor", choices=["on", "off"], default="on", help="独立于通信格式/记忆的抽取去重开关")
    p.add_argument("--evidence-span-chars", type=int, default=1800)
    p.add_argument("--max-output-tokens", type=int, default=1800, help="四项审计字段输出上限；各实验条件相同")
    p.add_argument("--max-prompt-chars", type=int, default=24000)
    p.add_argument("--max-question-tokens", type=int, default=24000,
                   help="观测用量达到预算后停止新调用；当前请求可能越界，不是硬账单上限")
    p.add_argument("--deadline-seconds", type=float, default=180, help="每题期限，模型请求timeout不超过剩余时间")
    p.add_argument("--max-total-tokens", type=int, default=60000,
                   help="已用 token 达到此值后不再开始下一题；当前题最多 max-calls 次调用")
    p.add_argument("--max-condition-tokens", type=int, default=16000000, help="每个条件相同的题间停止阈值；仍受全局预算约束")
    p.add_argument("--candidates", type=int, default=32)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--query", default="pregnancy urinary tract infection antibiotic allergy")
    p.add_argument("--packet", type=Path, help="prepare-judgments 生成的目录")
    p.add_argument("--judgments", type=Path, help="完成独立人工标注的 JSON 文件")
    p.add_argument("--cutoff", type=int, default=4, help="独立检索评测的 k")
    return p


def parse_args(argv=None):
    p = make_parser()
    preliminary, _ = p.parse_known_args(argv)
    actions = {a.dest: a for a in p._actions}
    saved = None
    if preliminary.resume:
        if preliminary.command not in {"run", "compare"} or not preliminary.output:
            p.error("--resume 仅用于run/compare，且必须指定 --output 原结果目录")
        # The durable ledger is authoritative, even if an export was interrupted.
        path = preliminary.output.resolve() / "checkpoint.sqlite3"
        if not path.is_file():
            p.error("该目录没有新版检查点；旧实验不能自动续跑")
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            row = db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if row is None:
            p.error("检查点尚未完成初始化，不能恢复")
        saved = json.loads(row[0])["settings"]
        # Restore saved defaults before applying an explicit config/CLI override.
        # Otherwise a new CLI default can change omitted fields on resume with the same config.
        settings = {k: v for k, v in saved.items()
                    if k in actions and k not in {"command", "config", "resume", "retry_unknown_calls", "output"}}
        for key, value in settings.items():
            if value is not None and actions[key].type:
                settings[key] = actions[key].type(value)
        p.set_defaults(**settings)
    if preliminary.config:
        settings = json.loads(preliminary.config.read_text(encoding="utf-8"))
        forbidden = {"help", "command", "config", "all", "limit", "resume", "retry_unknown_calls"}
        if not isinstance(settings, dict) or set(settings) - (set(actions) - forbidden):
            p.error("配置只能包含受支持的参数；command、all、limit需显式指定")
        for key, value in settings.items():
            action = actions[key]
            if action.type is Path:
                value = Path(value)
                value = value if value.is_absolute() else PROJECT / value
            elif action.type:
                value = action.type(value)
            if action.choices and value not in action.choices:
                p.error("无效配置项：" + key)
            settings[key] = value
        p.set_defaults(**settings)
        if saved:
            p.set_defaults(all=saved["all"], limit=saved["limit"])
    return p, p.parse_args(argv)


def copy_memory_snapshot(source, target, *, memory_ids=None):
    """SQLite backup包含已提交的WAL内容；不修改积累阶段的数据库。"""
    if not source.is_file():
        raise ValueError("记忆积累数据库尚不存在")
    temporary = target.with_name("." + target.name + ".snapshot")
    try:
        with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as original:
            with closing(sqlite3.connect(temporary)) as destination:
                original.backup(destination)
                if memory_ids is not None:
                    # Filter only the copy. Failed/pending provenance never enters the frozen seed.
                    destination.execute("DELETE FROM memory WHERE id NOT IN (SELECT value FROM json_each(?))",
                                        (json.dumps(memory_ids),))
                    destination.execute("DELETE FROM memory_fts WHERE id NOT IN (SELECT id FROM memory)")
                    destination.execute("DELETE FROM memory_revision WHERE old_id NOT IN (SELECT id FROM memory) "
                                        "OR new_id NOT IN (SELECT id FROM memory)")
                    destination.commit()
                destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                destination.execute("PRAGMA journal_mode=DELETE")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def new_directory(args):
    if args.resume:
        if not args.output.is_dir():
            raise CheckpointError("原结果目录不存在")
        return args.output.resolve()
    path = args.output or PROJECT / "experiments/results" / (time.strftime("%Y%m%d-%H%M%S") + "-" + args.command)
    path.mkdir(parents=True, exist_ok=False)
    sync_directory(path.parent)
    return path.resolve()


def books(args):
    return Textbooks(args.textbooks, args.retrieval_python, args.java_home, args.ranker,
                     args.candidates, args.top_k)


def source_hashes():
    files = [*Path(__file__).parent.glob("*.py"), PROJECT / "LangChain/agents/medqa_prompts.py",
             PROJECT / "LangChain/agents/medqa_compact_prompts.py",
             PROJECT / "LangChain/agents/medqa_baseline_prompts.py",
             PROJECT / "LangChain/agents/medqa_memory_prompts.py",
             PROJECT / "LangChain/agents/medqa_need_first_prompts.py",
             PROJECT / "LangChain/memory/advanced.py", PROJECT / "LangChain/memory/embedding.py",
             PROJECT / "LangChain/protocol/g2cp.py"]
    return {str(f.relative_to(PROJECT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def select_questions(args, data):
    ids = None
    if args.ids:
        order = json.loads(args.ids.read_text())
        if order["dataset_sha256"] != data.sha256:
            raise ValueError("顺序文件对应的数据版本不一致")
        ids = order["ids"]
    return data.select(limit=args.limit, seed=args.seed, ids=ids, split=args.split)


def summarize(results):
    n = len(results)
    metrics = [r["metrics"] for r in results]
    used = [r for r in results if r["metrics"]["memory_used"] > 0]
    candidates = sum(m.get('memory_candidate_sources',0) if r.get('workflow')=='need_first' else
                     m.get("memory_candidates_by_kind", {}).get("evidence", m["memory_candidates"])
                     for r,m in zip(results,metrics))
    source_checks = sum(m.get('memory_source_check_attempts',m["document_reads"]) for m in metrics)
    missing = sum(max(m.get("usage_missing_calls", 0), m.get("checkpoint_unknown_calls", 0)) for m in metrics)
    sources = sorted({source for m in metrics for source in m["usage_sources"]})
    from .memory_selection import STAGES,KINDS
    funnel={stage:{k:sum(m.get('memory_funnel',{}).get(stage,{}).get(k,0) for m in metrics)
                   for k in KINDS} for stage in STAGES}
    presented=sum(funnel['presented'].values())
    delivered=funnel['delivered']
    need_first=any(r.get('workflow')=='need_first' for r in results)
    needs=sum(m.get('knowledge_needs',0) for m in metrics)
    return {"questions_completed": n,
            "workflows":sorted({r.get('workflow','legacy') for r in results}),
            "communication_versions": sorted({r.get('communication_version','legacy_format_only') for r in results}),
            "workspace_versions": sorted({r.get('workspace_version') for r in results if r.get('workspace_version')}),
            "workspace_metrics": {key:sum(m.get(key,0) for m in metrics) for key in (
                'workspace_views','workspace_input_bytes_before','workspace_input_bytes_after',
                'workspace_brief_views','workspace_patch_count')},
            "handoff_metrics": {key:sum(m.get(key,0) for m in metrics) for key in (
                'acp_request_bytes','acp_result_bytes','acp_handshake_bytes','acp_object_registration_bytes',
                'acp_frames','acp_operations','acp_batches','acp_dispatch_seconds','acp_overhead_seconds','acp_handler_seconds',
                'acp_evidence_ref_reads','acp_memory_ref_reads','state_handle_reads','state_handle_bytes',
                'natural_handoff_chars','natural_handoff_rendered')},
            "status_counts": dict(Counter(r["status"] for r in results)),
            "answer_completion_rate": sum(r.get("answer") is not None for r in results) / n if n else None,
            "protocol_status_counts": dict(Counter(r.get("protocol_result", {}).get("status", "legacy") for r in results)),
            "strict_evidence_audit_counts": dict(Counter(r.get("evidence_result", {}).get("strict_audit_status", "legacy") for r in results)),
            "evidence_verification_counts": dict(Counter(r.get("evidence_result", {}).get("verification_status", "legacy") for r in results)),
            "support_assessment_counts": dict(Counter(r.get("evidence_result", {}).get("support_assessment", {}).get("status", "unknown") for r in results)),
            "source_verified_citations": sum(c.get("source_verified", False) and c.get("visible", False)
                                              for r in results for c in r.get("evidence_result", {}).get("citations", [])),
            "reliable_support_citations": sum(c.get("eligible_as_reliable_support", False)
                                               for r in results for c in r.get("evidence_result", {}).get("citations", [])),
            "usage_missing_calls": missing,
            "checkpoint_replayed_calls": sum(m.get("checkpoint_replayed_calls", 0) for m in metrics),
            "checkpoint_unknown_calls": sum(m.get("checkpoint_unknown_calls", 0) for m in metrics),
            "checkpoint_request_attempts": sum(m.get("checkpoint_request_attempts", m["model_calls"]) for m in metrics),
            "resumed_questions": sum(bool(m.get("checkpoint_resumed")) for m in metrics),
            "token_usage_complete": bool(n) and missing == 0 and sources == ["model"],
            "resolved_models": sorted({event["usage"].get("response_model", "unknown")
                                       for r in results for event in r["events"] if "usage" in event}),
            "memory_degraded_questions": sum(m.get("memory_read_errors", 0) + m.get("memory_write_errors", 0) + m.get("source_read_errors", 0) > 0 for m in metrics),
            "retrieval_error_questions": sum(m.get("search_errors", 0) > 0 for m in metrics),
            "retrieval_failed_questions": sum(r["metrics"].get("search_errors", 0) > 0 and r.get("answer") is None for r in results),
            "retrieval_errors_with_answer": sum(r["metrics"].get("search_errors", 0) > 0 and r.get("answer") is not None for r in results),
            "runtime_failed_questions": sum(r["status"] in {"runtime_error", "protocol_error", "no_answer", "context_budget_exhausted", "budget_exhausted",
                                                              "deadline_exhausted", "token_budget_exhausted", "usage_unavailable"} for r in results),
            "accuracy": sum(r["correct"] for r in results) / n if n else None,
            "answered_questions": sum(r.get("answer") is not None for r in results),
            "tokens_per_answer": (sum(m["total_tokens"] for m in metrics) / sum(r.get("answer") is not None for r in results)
                                  if any(r.get("answer") is not None for r in results) and not missing and sources == ["model"] else None),
            "tokens_per_correct": (sum(m["total_tokens"] for m in metrics) / sum(r.get("correct", False) for r in results)
                                   if any(r.get("correct", False) for r in results) and not missing and sources == ["model"] else None),
            "v6_metrics": {key: sum(m.get(key, 0) for m in metrics) for key in (
                "queries_reused", "queries_filtered", "search_progress", "stalled_rounds", "expansions",
                "deterministic_repairs", "semantic_reviews", "recovered_action_errors", "evidence_raw_chars", "evidence_raw_delivery_chars",
                "evidence_view_chars", "evidence_exact_duplicates", "evidence_similar_groups", "evidence_processor_seconds",
                "checkpoint_replayed_operations", "checkpoint_executed_operations", "checkpoint_unknown_recomputations")},
            "supported_rate": sum(r.get("evidence_status") == "supported" for r in results) / n if n else None,
            "evidence_status_counts": dict(Counter(r.get("evidence_status", "not_assessed") for r in results)),
            "answer_basis_counts": dict(Counter(b for r in results for b in r.get("answer_basis", []))),
            "evidence_status_downgraded_questions": sum(bool(r.get("audit_warnings")) for r in results),
            "accuracy_by_evidence_status": {s: sum(r["correct"] for r in results if r.get("evidence_status") == s) / sum(r.get("evidence_status") == s for r in results)
                                          for s in sorted({r.get("evidence_status", "not_assessed") for r in results})
                                          if any(r.get("evidence_status") == s for r in results)},
            "accuracy_on_answered": sum(r["correct"] for r in results if r["answer"] is not None) / sum(r["answer"] is not None for r in results) if any(r["answer"] is not None for r in results) else None,
            "wrong_supported_questions": sum(r.get("evidence_status") == "supported" and not r["correct"] for r in results),
            "followup_memory_queries": sum(e.get("tool") == "memory_prefetch" and e.get("stage") == "followup"
                                           for r in results for e in r["events"]),
            "memory_reference_resolutions": sum(e.get("tool") == "resolve_memory" for r in results for e in r["events"]),
            "preserved_answer_questions": sum(any(e.get("tool") == "answer_preserved" for e in r["events"]) for r in results),
            "followup_outcomes": dict(Counter(r.get("followup_result", {}).get("status", "not_requested") for r in results)),
            "followup_new_visible_chars": sum(m.get("followup_new_visible_chars", 0) for m in metrics),
            "followup_new_visible_sources": sum(m.get("followup_new_visible_sources", 0) for m in metrics),
            "memory_candidates_by_kind": {k:sum(m.get("memory_candidates_by_kind",{}).get(k,0) for m in metrics) for k in ("evidence","strategy","episode")},
            "memory_written_by_kind": {k:sum(m.get("memory_written_by_kind",{}).get(k,0) for m in metrics) for k in ("evidence","strategy","episode")},
            "memory_experience_reads": sum(m.get("memory_experience_reads",0) for m in metrics),
            "state_vector_transfers":sum(m.get('state_vector_transfers',0) for m in metrics),
            "memory_rpc_requests":sum(m['state_transfers'] for m in metrics),
            "embedding_seconds":sum(m.get('embedding_seconds',0.) for m in metrics),
            "memory_funnel":funnel,
            "memory_planner_acceptance_rate":sum(funnel['planner_accepted'].values())/presented if presented and not need_first else None,
            "memory_selection_rate":sum(funnel['selected'].values())/presented if presented and not need_first else None,
            "memory_investigator_acceptance_rate":sum(funnel['investigator_accepted'].values())/presented if presented and need_first else None,
            "memory_need_coverage_rate":sum(m.get('memory_covered_needs',0) for m in metrics)/needs if needs else None,
            "memory_need_support_rate":sum(m.get('memory_supported_needs',0) for m in metrics)/needs if needs else None,
            "memory_zero_search_supported_rate":sum(m.get('memory_only_supported_tasks',0) for m in metrics)/n if n else None,
            "memory_evidence_citation_rate":funnel['cited']['evidence']/delivered['evidence'] if delivered['evidence'] else None,
            "memory_strategy_execution_rate":funnel['executed']['strategy']/delivered['strategy'] if delivered['strategy'] else None,
            "reused_question_rate": len(used) / n if n else None,
            "memory_hit_question_rate": sum(m["memory_candidates"] > 0 for m in metrics) / n if n else None,
            "reused_without_search_questions": sum(r["metrics"]["retrieval_count"] == 0 for r in used),
            "accuracy_on_reused_questions": sum(r["correct"] for r in used) / len(used) if used else None,
            "memory_source_valid_rate": sum(m["memory_source_valid"] for m in metrics) / source_checks if source_checks else None,
            "memory_source_check_coverage": source_checks / candidates if candidates else None,
            "memory_source_check_attempts": source_checks,
            "memory_presented": presented,
            "memory_coarse_candidate_occurrences": sum(len(e.get("candidate_ids", [])) for r in results for e in r["events"] if e.get("tool") == "memory_prefetch"),
            **{key:sum(m.get(key,0) for m in metrics) for key in
               ('investigation_rounds','queries_skipped_covered','memory_only_supported_tasks',
                'memory_evidence_presented','memory_covered_needs','needs_review_errors','knowledge_needs','memory_supported_needs')},
            **{key:sum(m.get(key,0) for m in metrics) for key in
               ('memory_candidate_sources','memory_fragment_valid','memory_source_cache_hits')},
            "mean_seconds": statistics.mean(m["elapsed_seconds"] for m in metrics) if n else None,
            **{key: sum(m[key] for m in metrics) for key in
               ("model_calls", "messages", "communication_chars", "prompt_tokens", "completion_tokens", "total_tokens",
                "request_chars", "response_chars", "memory_read_errors", "memory_write_errors",
                "cached_prompt_tokens", "reasoning_tokens",
                "embedding_cache_hits", "embedding_encoded_texts",
                "state_transfers", "state_bytes", "retrieval_count", "memory_candidates", "memory_used", "search_errors",
                "memory_queries", "memory_source_valid", "memory_written", "memory_rejected", "document_reads",
                "followup_rounds", "contract_repairs", "request_bytes", "response_bytes", "system_chars", "system_bytes", "protocol_control_bytes")},
            "role_usage": {role: {key: sum(e["usage"].get(key, 0) for r in results for e in r["events"]
                                     if e.get("agent") == role and "usage" in e)
                                  for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
                           for role in ("Planner", "Investigator", "VerifierReasoner")},
            "usage_sources": sources,
            "note": "命中/引用不等于正确复用；来源通过率以实际回读为分母，未读候选不算无效；memory_written是成功写入请求数，含幂等去重。检索质量需独立标注。"}


def compare_reports(report, completed_ids, question_ids):
    complete = len(completed_ids) == 4 and all(v == question_ids for v in completed_ids.values())
    known_models = {model for item in report.values() for model in item["resolved_models"] if model != "unknown"}
    same_model = len(known_models) == 1 and all("unknown" not in item["resolved_models"] for item in report.values())
    healthy = all(r["runtime_failed_questions"] == 0 and r["memory_degraded_questions"] == 0 and r.get("retrieval_failed_questions", 0) == 0 for r in report.values())
    token_valid = complete and same_model and healthy and all(r["token_usage_complete"] for r in report.values())
    timing_valid = complete and healthy and not any(r.get("resume_count", 0) for r in report.values())
    comparisons = {"matched_complete": complete, "token_comparison_valid": token_valid,
                   "observed_cost_complete": complete and same_model and all(r["token_usage_complete"] for r in report.values()),
                   "latency_comparison_valid": timing_valid,
                   "resolved_model_consistent": same_model,
                   "runtime_healthy": healthy,
                   "note": "未知/估算 token 不计算 token 节省率；缺失用量不是零成本。"}
    if complete and same_model:
        for name, base, target in [("protocol_memory_off", "text_off", "structured_off"),
                                   ("protocol_memory_on", "text_on", "structured_on"),
                                   ("memory_text", "text_off", "text_on"),
                                   ("memory_structured", "structured_off", "structured_on")]:
            a, b = report[base], report[target]
            pair_healthy = all(r["runtime_failed_questions"] == 0 and r["memory_degraded_questions"] == 0 and r.get("retrieval_failed_questions", 0) == 0 for r in (a, b))
            comparisons[name] = {"token_reduction": 1 - b["total_tokens"] / a["total_tokens"] if token_valid and a["total_tokens"] else None,
                                 "search_reduction": 1 - b["retrieval_count"] / a["retrieval_count"] if pair_healthy and a["retrieval_count"] else None,
                                 "latency_reduction": 1 - b["mean_seconds"] / a["mean_seconds"] if timing_valid and a["mean_seconds"] else None,
                                 "accuracy_delta": b["accuracy"] - a["accuracy"],
                                 "quality_not_worse_on_this_sample": b["accuracy"] >= a["accuracy"],
                                 "memory_degraded_questions": b["memory_degraded_questions"]}
        comparisons["factorial_effects"] = {}
        for metric in ("accuracy", "total_tokens"):
            if metric == "total_tokens" and not token_valid:
                continue
            t0, s0, t1, s1 = (report[k][metric] for k in ("text_off", "structured_off", "text_on", "structured_on"))
            comparisons["factorial_effects"][metric] = {
                "structured_minus_text": ((s0 - t0) + (s1 - t1)) / 2,
                "memory_on_minus_off": ((t1 - t0) + (s1 - s0)) / 2,
                "interaction": (s1 - s0) - (t1 - t0)}
    return comparisons


def model_connection(args):
    """Resolve local/cloud settings without leaking cloud keys to localhost."""
    base=getattr(args,'model_base_url',None) or os.getenv('MEDQA_BASE_URL') or os.getenv('DEEPSEEK_BASE_URL')
    local=urlparse(base or '').hostname in {'127.0.0.1','localhost','::1'}
    model=getattr(args,'model_name',None) or os.getenv('MEDQA_MODEL') or (None if local else os.getenv('DEEPSEEK_MODEL'))
    key_env=getattr(args,'model_api_key_env',None)
    key=os.getenv(key_env) if key_env else 'local-moreagent' if local else os.getenv('MEDQA_API_KEY') or os.getenv('DEEPSEEK_API_KEY')
    timeout=getattr(args,'model_timeout_seconds',60)
    if not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
        raise ConfigurationError('model_timeout_seconds须为正有限数值')
    if not all((model,base,key)):
        raise ConfigurationError('模型配置不完整：本地服务需明确model_name和model_base_url；云API沿用MEDQA_*/DEEPSEEK_*')
    if urlparse(base).scheme not in {'http','https'} or not urlparse(base).hostname:
        raise ConfigurationError('模型地址须为包含主机名的http或https地址')
    return model,base,key,timeout


def create_model(args):
    from ..framework.config import load_env_file
    from .model import CompatibleChat
    load_env_file()
    model,base,key,timeout = model_connection(args)
    chat = CompatibleChat(model=model, base_url=base, api_key=key, max_tokens=args.max_output_tokens,timeout_seconds=timeout)
    return ChatRoles(chat, args.context_profile, args.output_policy,args.memory_selection,
                     workspace_policy=getattr(args,'workspace_policy','legacy'), workflow=getattr(args,'workflow','legacy')), {"model": model, "temperature": 0, "max_output_tokens": args.max_output_tokens,
                            "request_limit_field": "max_tokens",
                            "endpoint_sha256": hashlib.sha256(base.rstrip("/").encode()).hexdigest(),
                            "thinking_disabled": chat.deepseek or chat.qwen3_nonthinking,"timeout_seconds":timeout,
                            "precision":getattr(args,'model_precision',None)}


def execute_batch(args, data, textbooks, agents, out, questions, conditions, checkpoint):
    from .memory import SharedMemory, MEMORY_EMBEDDING_VERSION
    from ..memory.embedding import QwenEmbeddingConfig, QwenEmbeddingService
    from ..memory.advanced import SQLiteMemoryStore

    names = [mode + "_" + memory for mode, memory in conditions]

    def unknown_usage(result):
        m = result["metrics"]
        return (max(m.get("usage_missing_calls", 0), m.get("checkpoint_unknown_calls", 0)) > 0
                or any(s not in {"model", "fixture"} for s in m["usage_sources"]))

    def publish():
        report, completed_ids = {}, {}
        for name in names:
            if not checkpoint.started(name):
                continue
            results = checkpoint.results(name)
            checkpoint.export_results(name)
            report[name] = summarize(results)
            report[name]["resume_count"] = checkpoint.meta("resume_count", 0)
            completed_ids[name] = [r["question_id"] for r in results]
        dump(out / "report.json", report)
        dump(out / "recovery.json", {
            "resume_count": checkpoint.meta("resume_count", 0),
            "unknown_calls": sum(checkpoint.unknown_calls(name) for name in names),
            "latency_comparable": checkpoint.meta("resume_count", 0) == 0,
            "note": "已保存响应重用不重复调用；未知请求用量不当作零成本。续跑时延/IPC不用于性能收益比较。"})
        dump(out / "comparison.json", compare_reports(report, completed_ids, [q.id for q in questions])
             if args.command == "compare" else {})

    # Repair stale/truncated exports before deciding whether another paid request
    # is allowed. Completed rows are never rerun, including failed rows.
    publish()
    checkpoint.ensure_retry_allowed(args.retry_unknown_calls)
    existing = [r for name in names for r in checkpoint.results(name)]
    if any(unknown_usage(r) for r in existing):
        raise CheckpointError("已有已完成题目的用量缺失；已恢复报告，停止后续任务以避免扩大未知费用")
    total_used = sum(r["metrics"]["total_tokens"] for r in existing)
    embedder = QwenEmbeddingService(QwenEmbeddingConfig(str(args.embedding), cache_version=MEMORY_EMBEDDING_VERSION))
    if args.memory_source:
        seed = validate_seed(args.memory_source, dataset_sha256=data.sha256,
            question_ids=[q.id for q in questions], namespace=args.memory_namespace or "structured_on",
            task_group=args.task_group, corpus_version=textbooks.version, protocol=protocol_for(args.output_policy),
            source_hashes=source_hashes(), embedding_model=embedder.cache_key, allow_partial=args.output_policy == "decoupled")
        frozen = out / "frozen-seed.sqlite3"
        if not checkpoint.meta("frozen_ready", False):
            copy_memory_snapshot(args.memory_source, frozen, memory_ids=seed["memory_ids"])
            with checkpoint.db:
                checkpoint.set_meta("frozen_ready", True)
        elif not frozen.is_file():
            raise CheckpointError("冻结记忆快照缺失，不能恢复")
    warmed = False
    halted = False
    for mode, memory_mode in conditions:
        condition = mode + "_" + memory_mode
        memory_path = out / (condition + ".sqlite3")
        if checkpoint.started(condition) and memory_mode == "on" and not memory_path.is_file():
            raise CheckpointError("原条件记忆数据库缺失，不能重建空库后继续")
        prior = checkpoint.results(condition)
        # Respect both the original condition order and cumulative budgets.
        if len(prior) == len(questions):
            continue
        if total_used >= args.max_total_tokens:
            break
        condition_used = sum(r["metrics"]["total_tokens"] for r in prior)
        if condition_used >= args.max_condition_tokens:
            continue
        if not warmed:
            textbooks.warmup()
            warmed = True
        namespace = args.memory_namespace or ("structured_on" if args.memory_source else condition)
        if memory_mode == "on" and args.memory_source and not checkpoint.meta(condition + ":memory_ready", False):
            if checkpoint.started(condition):
                raise CheckpointError("原条件记忆初始化标记缺失，停止恢复")
            # The atomic copy may have completed before the process saved the
            # ready marker. Reuse it, including any SQLite recovery journal;
            # replacing the main file under an old WAL would be unsafe.
            if not memory_path.is_file():
                copy_memory_snapshot(out / "frozen-seed.sqlite3", memory_path)
        memory = SharedMemory(memory_path, namespace, embedder.cache_key) if memory_mode == "on" else None
        try:
            if memory is not None:
                with checkpoint.db:
                    checkpoint.set_meta(condition + ":memory_ready", True)
                embedder.cache_store = SQLiteMemoryStore(memory_path, namespace, namespace, embedder.cache_key)
                if embedder._model is None:
                    from sentence_transformers import SentenceTransformer
                    embedder._model = SentenceTransformer(str(args.embedding), device="cpu", local_files_only=True)
            else:
                embedder.cache_store = None
            # Finish the durable outbox first. No LLM runs during this step.
            before = sum(r["metrics"]["total_tokens"] for r in prior)
            checkpoint.apply_memory(condition, memory)
            prior = checkpoint.results(condition)
            condition_used = sum(r["metrics"]["total_tokens"] for r in prior)
            total_used += condition_used - before
            publish()
            if any(unknown_usage(r) for r in prior):
                halted = True
                print("恢复的题目含未知用量，已保存报告，停止后续任务。", flush=True)
                break
            for position, question in enumerate(questions):
                row = checkpoint.question(condition, position)
                if row and row["state"] == "complete":
                    continue
                if total_used >= args.max_total_tokens or condition_used >= args.max_condition_tokens:
                    break
                resumed = row is not None
                row = checkpoint.begin(condition, position, question.id)
                calls = DurableCalls(agents, checkpoint, condition, position,
                                     retry_unknown=args.retry_unknown_calls, max_calls=args.max_calls)
                staged = PendingMemory(memory) if memory is not None else None
                tools = DurableTools(checkpoint, condition, position, max_searches=args.max_searches, max_reads=args.max_reads)
                runtime = Runtime(calls, textbooks, memory=staged, embedder=embedder if staged else None,
                                  mode=mode, max_calls=args.max_calls, memory_threshold=args.memory_threshold,
                                  max_prompt_chars=args.max_prompt_chars, context_profile=args.context_profile,
                                  max_question_tokens=args.max_question_tokens, deadline_seconds=args.deadline_seconds,
                                  task_group=args.task_group, memory_write=args.memory_write == "on",
                                  max_searches=args.max_searches, max_reads=args.max_reads,
                                  max_stalled_rounds=args.max_stalled_rounds, max_contract_repairs=args.max_contract_repairs,
                                  evidence_processor=args.evidence_processor == "on", evidence_span_chars=args.evidence_span_chars,
                                  journal=tools, output_policy=args.output_policy, memory_policy=args.memory_policy,
                                  memory_selection=args.memory_selection,memory_recall_limit=args.memory_recall_limit,
                                  workspace_policy=getattr(args,'workspace_policy','legacy'),
                                  workflow=getattr(args,'workflow','legacy'),
                                  max_investigation_rounds=getattr(args,'max_investigation_rounds',3))
                result = runtime.run(question, trace_id=row["trace"])
                dump(out / (condition + "-handshake.json"), runtime.handshake())
                if calls.fatal_error:
                    raise calls.fatal_error
                if tools.fatal_error:
                    raise tools.fatal_error
                calls.validate_replay()
                # Gold is used only for grading, never put in the memory outbox.
                result["correct"] = data.grade(result)
                result["metrics"].update(checkpoint_replayed_calls=calls.replayed,
                    checkpoint_replayed_operations=tools.replayed, checkpoint_executed_operations=tools.executed,
                    checkpoint_unknown_recomputations=tools.unknown_recomputations,
                    checkpoint_search_attempts=tools.attempts("search_textbooks"),
                    checkpoint_source_read_attempts=tools.attempts("read_source"),
                    checkpoint_unknown_calls=checkpoint.unknown_calls(condition, position),
                    checkpoint_resumed=resumed, checkpoint_request_attempts=calls.request_attempts())
                checkpoint.save_result(condition, position, result, staged.writes if staged else [])
                checkpoint.apply_memory(condition, memory)
                total_used += result["metrics"]["total_tokens"]
                condition_used += result["metrics"]["total_tokens"]
                publish()
                print(condition, position + 1, question.id, result["status"],
                      "tokens=" + str(result["metrics"]["total_tokens"]), flush=True)
                if unknown_usage(result):
                    halted = True
                    print("用量存在未知项，结果已保存，停止后续任务；已知Token仅为下界。", flush=True)
                    break
        finally:
            if memory:
                memory.close()
        if halted:
            break
    publish()
    print("结果目录:", out)


def main():
    p, args = parse_args()
    if args.retry_unknown_calls and not args.resume:
        p.error("--retry-unknown-calls 必须与 --resume 同时使用")
    if args.workspace_policy == 'role_views_v1' and args.output_policy != 'decoupled':
        p.error('role_views_v1需要--output-policy decoupled')
    if args.workflow == 'need_first' and (args.output_policy!='decoupled' or args.memory_policy!='basic' or args.memory_selection!='needs'):
        p.error('need_first需要decoupled输出、basic记忆与needs选择')
    if not 1 <= args.max_investigation_rounds <= 3:
        p.error('--max-investigation-rounds需为1至3')
    if args.memory_source and args.command == "run" and args.memory == "off":
        p.error("关闭记忆的run不应加载积累库")
    if args.memory_source and args.memory_write != "off":
        p.error("冻结积累库评估必须使用 --memory-write off")
    if args.limit is not None and args.limit < 1:
        p.error("--limit 必须大于 0")
    if args.all and args.limit is not None:
        p.error("--all 与 --limit 不能同时使用")
    if (not 3 <= args.max_calls <= 64 or args.max_output_tokens < 100
            or min(args.max_searches, args.max_reads, args.max_stalled_rounds, args.evidence_span_chars) < 1
            or not 0 <= args.max_contract_repairs <= 8
            or min(args.max_total_tokens, args.max_condition_tokens, args.max_prompt_chars, args.max_question_tokens) < 1
            or not math.isfinite(args.deadline_seconds) or args.deadline_seconds <= 0):
        p.error("调用/token 预算无效")
    if not -1 <= args.memory_threshold <= 1:
        p.error("余弦阈值需在 [-1,1] 内")
    if not args.task_group.strip():
        p.error("--task-group 不能为空")
    if not 3<=args.memory_recall_limit<=10:
        p.error('--memory-recall-limit需为3至10')
    if args.memory_selection=='needs' and (args.memory_policy!='basic' or args.output_policy!='decoupled'):
        p.error('需求重排需要 --memory-policy basic 与 --output-policy decoupled')
    if args.command in {"run", "compare"} and args.limit is None and not args.all:
        p.error("run/compare 必须明确指定 --limit N 或 --all，避免无意启动全量付费实验")
    if args.command == "evaluate-retrieval":
        if not args.packet or not args.judgments:
            p.error("evaluate-retrieval 需要 --packet 和 --judgments")
        from .evaluation import evaluate
        result = evaluate(args.packet, args.judgments, args.cutoff)
        out = new_directory(args)
        dump(out / "retrieval-report.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("结果目录:", out)
        return
    data = Dataset(args.benchmark)
    if args.command == "streams":
        out = new_directory(args)
        # 预先声明的题干规则；不按参考答案、模型表现或记忆命中率选题。
        rules = {"cardiovascular": r"\b(cardiac|coronary|myocardial|heart failure|aortic|mitral|arrhythmia|electrocardiogram)\b",
                 "neurology": r"\b(seizure|epilepsy|parkinson|stroke|cerebral|spinal cord|multiple sclerosis|neurologic)\b"}
        for name, pattern in rules.items():
            ids = [q.id for q in data.select(split=args.split) if re.search(pattern, q.question, re.I)]
            if args.limit:
                ids = ids[:args.limit]
            dump(out / (name + ".json"), {"dataset_sha256": data.sha256, "selection_rule": pattern,
                                          "ids": ids, "note": "题干关键词粗分的关联流，不代表专家主题标注或强任务依赖。"})
            print(name, len(ids))
        print("输出:", out)
        return
    t = books(args)
    try:
        if args.command == "audit":
            from ..framework.config import load_env_file
            from .memory import MEMORY_EMBEDDING_VERSION
            load_env_file()
            questions = select_questions(args, data)
            issues = []
            seed = None
            if args.memory_source:
                try:
                    seed = validate_seed(args.memory_source, dataset_sha256=data.sha256,
                        question_ids=[q.id for q in questions], namespace=args.memory_namespace or "structured_on",
                        task_group=args.task_group, corpus_version=t.version, protocol=protocol_for(args.output_policy),
                        source_hashes=source_hashes(), allow_partial=args.output_policy == "decoupled",
                        embedding_model=f"{args.embedding}|norm=True|{MEMORY_EMBEDDING_VERSION}")
                except (ValueError, KeyError, OSError, sqlite3.Error) as exc:
                    issues.append(str(exc))
            for label, path in {"retrieval_python": args.retrieval_python, "ranker": args.ranker / "config.json",
                                "embedding": args.embedding / "config.json", "java": t.java / "bin/java"}.items():
                if not path.is_file():
                    issues.append(label + "不存在")
            try:
                model_connection(args)
                api_configured=True
            except ConfigurationError as exc:
                api_configured=False
                issues.append(str(exc))
            if args.output and args.output.exists():
                issues.append("结果目录已存在")
            result = {"runtime_protocol": protocol_for(args.output_policy), "answer_policy": answer_policy_for(args.output_policy),
                      "communication_version": "moreagent.handoff/1",
                      "workspace_policy":args.workspace_policy,
                      "workflow":args.workflow,"max_investigation_rounds":args.max_investigation_rounds,
                      "communication_paths": {"text":"natural_language_without_acp", "structured":"ACP-1_binary_dispatch"},
                      "output_policy": args.output_policy,
                      "questions": len(questions), "dataset_sha256": data.sha256,
                      "condition_matrix": ["text_off", "structured_off", "text_on", "structured_on"],
                      "memory_write": args.memory_write, "seed": seed, "ready_for_batch_by_static_checks": not issues,
                      "memory_policy":args.memory_policy,"memory_experiment":"frozen" if args.memory_source else "online" if args.memory_write=="on" else "read_only",
                      "memory_selection":args.memory_selection,"memory_recall_limit":args.memory_recall_limit,
                      "blocking_issues": issues, "api_configured": api_configured, "model_api_called": False,
                      "note": "只检查配置、版本与积累产物；没有验证API可达性、真实usage、医学准确率或系统时延。"}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.command == "prepare-judgments":
            from .evaluation import prepare
            out = new_directory(args)
            questions = data.select(limit=args.limit or 4, seed=args.seed, split=args.split)
            result = prepare(out, questions, t, data.sha256, args.seed)
            print(json.dumps(result, ensure_ascii=False))
            print("标注材料目录:", out)
            return
        if args.command == "preflight":
            print(json.dumps({"medqa_questions": len(data.questions), "dataset_sha256": data.sha256,
                              "heldout_questions": len(data.select(split="heldout")),
                              "development_question_ids": sorted(DEVELOPMENT_IDS),
                              "textbooks_version": t.version, "corpus_files": len(list((args.textbooks / "chunk").glob("*.jsonl"))),
                              "retrieval_python_exists": args.retrieval_python.is_file(),
                              "ranker_exists": (args.ranker / "config.json").is_file(),
                              "embedding_exists": (args.embedding / "config.json").is_file(),
                              "model_api_called": False}, ensure_ascii=False, indent=2))
            return
        if args.command == "probe":
            start = time.monotonic()
            documents = t.search(args.query)
            print(json.dumps({"query": args.query, "documents": documents, "stats": t.stats,
                              "seconds": time.monotonic() - start}, ensure_ascii=False, indent=2))
            return
        questions = select_questions(args, data)
        if not questions:
            raise ValueError("没有可执行的题目")
        if args.memory_source and not args.memory_source.is_file():
            p.error("记忆积累数据库尚不存在，请先完成积累阶段")
        seed_meta = None
        if args.memory_source:
            seed_meta = validate_seed(args.memory_source, dataset_sha256=data.sha256,
                question_ids=[q.id for q in questions], namespace=args.memory_namespace or "structured_on",
                task_group=args.task_group, corpus_version=t.version, protocol=protocol_for(args.output_policy),
                source_hashes=source_hashes(), allow_partial=args.output_policy == "decoupled")
        agents, model_meta = create_model(args)
        out = new_directory(args)
        args.output = out
        settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
        manifest = {"settings": settings, "model": model_meta,
                                     "dataset_sha256": data.sha256, "textbooks_version": t.version,
                                     "retrieval_version": getattr(t, "operation_version", type(t).__name__),
                                     "source_sha256": source_hashes(), "runtime_protocol": protocol_for(args.output_policy),
                                     "answer_policy": answer_policy_for(args.output_policy),
                                     "seed_provenance": seed_meta,
                                     "question_ids": [q.id for q in questions],
                                     "sequence_sha256": hashlib.sha256(json.dumps([q.id for q in questions]).encode()).hexdigest(),
                                     "gold_visible_to_agents": False, "memory_strategy": "exact_cosine",
                                     "memory_embedding_strategy": ("task_group + evidence + need_goal + title + exact_source_excerpts" if args.workflow=='need_first' else
                                         "task_group + memory_type + topic + summary + conditions + source_excerpts_or_queries"
                                         if args.memory_policy == "basic" else "task_group + task_topic + historical_conditions + title + exact_support_quotes"),
                                     "architecture": ("Planner → need_memory_recall/read → Investigator_coverage → gap_tools → Reasoner_need_review → optional_Investigator → need_memory_update"
                                         if args.workflow=='need_first' else "program_prefetch → Planner → Investigator → tools → VerifierReasoner → program_memory_update"),
                                     "retrieval_policy": {"type": "adaptive", "search_fuse": args.max_searches,
                                         "stalled_fuse": args.max_stalled_rounds, "document_reads": args.max_reads,
                                         "model_calls": args.max_calls},
                                     "evidence_processor": args.evidence_processor,
                                     "development_question_ids": sorted(DEVELOPMENT_IDS),
                                     "contains_development_questions": any(q.id in DEVELOPMENT_IDS for q in questions),
                                     "communication_version": "moreagent.handoff/1",
                                     "text_baseline": "自然语言角色交接，不经过ACP解析与派发；本角色工具和结果接口仍为JSON。",
                                     "structured_treatment": "ACP-1二进制位置参数、任务引用、能力协商、批次映射和向量句柄消费；原工具逻辑保留。",
                                     "workspace_policy":args.workspace_policy,
                                     "workflow":args.workflow,"max_investigation_rounds":args.max_investigation_rounds,
                                     "workspace_treatment": ("structured条件启用应用层共享状态、校验后的角色变更和按角色输入视图；"
                                         "Text保持原自然语言交接。收益属于通信与上下文组合方案，不能单独归因于二进制编码。"
                                         if args.workspace_policy=='role_views_v1' else "disabled"),
                                     "measurement_boundary": "API usage与ACP消息字节分开统计；对象登记、握手和自然语言交接成本均保留。"}
        conditions = [(args.mode, args.memory)] if args.command == "run" else [
            ("text", "off"), ("structured", "off"), ("text", "on"), ("structured", "on")]
        if args.command == "compare":
            random.Random(args.condition_seed).shuffle(conditions)
        with Checkpoint(out, resume=args.resume) as checkpoint:
            checkpoint.initialize(manifest, [list(c) for c in conditions], resume=args.resume)
            # Exports are repairable views. SQLite remains authoritative.
            dump(out / "manifest.json", checkpoint.meta("manifest"))
            dump(out / "condition-order.json", [mode + "_" + memory for mode, memory in conditions])
            execute_batch(args, data, t, agents, out, questions, conditions, checkpoint)
    finally:
        t.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # 不将 SDK URL、header 或原始异常输出到实验文件。
        print("MedQA 入口失败:", str(exc) if isinstance(exc, (EvaluationError, CheckpointError, ConfigurationError)) else type(exc).__name__, file=sys.stderr)
        sys.exit(1)
