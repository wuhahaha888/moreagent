"""Offline diagnostics: saved v5 outputs and synthetic evidence, never a model API.

Run from the repository root with the existing struckflow Python environment.
This checks dispatch and payload sizes, not new medical accuracy or API savings.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from LangChain.medqa.data import Question
from LangChain.medqa.evidence_processor import EvidenceProcessor
from LangChain.medqa.retrieval import digest
from LangChain.medqa.runtime import Runtime, compact, PROTOCOL
from LangChain.agents.medqa_prompts import PLANNER, INVESTIGATOR, REASONER
from LangChain.agents.medqa_compact_prompts import COMPACT_ROLE_PROMPTS
from LangChain.test.test_medqa import Books


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, default=ROOT / "experiments/results/medqa-pilot-5-accumulate/structured_on.jsonl")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "offline-effects.json"
    if target.exists():
        parser.error("输出已存在，使用新目录保存新一轮诊断")
    rows = [json.loads(line) for line in args.pilot.read_text().splitlines()]
    checks = []
    for row in rows:
        calls = [e for e in row["events"] if "agent" in e]
        question = Question(row["question_id"], calls[0]["input"]["question"], calls[0]["input"]["options"])

        class SavedDecisions:
            def call(self, role, payload, mode):
                if role in (PLANNER, INVESTIGATOR):
                    saved = next(e for e in calls if e["agent"] == role)
                    output = json.loads(saved["response"])
                else:
                    output = {"question_id": question.id, "decision": "stop", "gaps": [],
                              "abstain_reason": "离线回放只检查原有查询能否派发；不生成新的医学答案"}
                return output, {"source": "fixture", "total_tokens": 0}

        books = Books()
        result = Runtime(SavedDecisions(), books, max_contract_repairs=0).run(question)
        checks.append({"question_id": question.id, "old_status": row["status"], "old_reason": row["summary"],
                       "old_searches": row["metrics"]["retrieval_count"], "v6_dispatched_queries": books.calls,
                       "mappings": result["metrics"]["deterministic_repairs"],
                       "query_dispatch_passed": books.calls > 0,
                       "scope": "原Planner/Investigator响应+合成只读工具；不评价新答案"})

    row = next(r for r in rows if any(e.get("agent") == REASONER for e in r["events"]))
    review = next(e for e in row["events"] if e.get("agent") == REASONER)
    original = review["input"]["evidence_pool"]
    docs = {d["id"]: {**d, "sha256": digest(d["text"]), "corpus_version": "saved-v5"} for d in original}
    query = review["input"]["question"]  # No future answer or gold used for selection.
    ep = EvidenceProcessor()
    pack = ep.process(docs, query=query)
    full = EvidenceProcessor(enabled=False).process(docs, query=query)
    short_comparison = {"question_id": row["question_id"], "documents": len(docs),
        "raw_chars": pack["raw_chars"], "selected_text_chars": pack["view_chars"],
        "v5_pool_json_chars": len(compact(original)), "v6_full_pool_json_chars": len(compact(full["views"])),
        "v6_ep_pool_json_chars": len(compact(pack["views"])),
        "exact_original_text_preserved": all(v["text"] == docs[v["id"]]["text"] for v in pack["views"]),
        "note": "短chunk保留全文；审计追踪映射留在系统侧，本例不声称压缩或Token收益。"}

    duplicate_docs = deepcopy(docs)
    sample = next(iter(docs.values()))
    duplicate_docs["copy_alias"] = {**sample, "id": "copy_alias"}
    full_dup = EvidenceProcessor(enabled=False).process(duplicate_docs, query=query)
    dedup = ep.process(duplicate_docs, query=query)
    result = {"protocol": PROTOCOL, "model_api_called": False, "new_api_tokens": 0,
        "prompt_chars_by_role": {role: {"v5_saved": next(e["system_chars"] for r in rows for e in r["events"] if e.get("agent") == role),
                                       "v6_current": len(prompt)} for role, prompt in COMPACT_ROLE_PROMPTS.items()},
        "saved_pilot_sha256": hashlib.sha256(args.pilot.read_bytes()).hexdigest(),
        "saved_pilot_total_tokens": sum(r["metrics"]["total_tokens"] for r in rows),
        "saved_pilot_model_calls": sum(r["metrics"]["model_calls"] for r in rows),
        "dispatch_checks": checks, "all_saved_query_batches_dispatch": all(c["query_dispatch_passed"] for c in checks),
        "actual_short_evidence_comparison": short_comparison,
        "synthetic_duplicate_ablation": {"same_inputs": True, "inserted_duplicates": 1,
            "full_view_chars": full_dup["view_chars"], "ep_view_chars": dedup["view_chars"],
            "full_payload_chars": len(compact(full_dup["views"])), "ep_payload_chars": len(compact(dedup["views"])),
            "all_raw_sources_preserved": len(duplicate_docs) == len(docs) + 1,
            "note": "合成重复反例，只能说明去重机制；字符差不是实际Token收益。"},
        "limits": ["没有新医学准确率/真实Token/时延结论", "未调用API，不产生新医学记忆", "旧五题属于调试题，不作独立评测集"]}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
