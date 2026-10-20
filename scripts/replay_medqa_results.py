"""Replay saved outputs through independent result handling. No model/tool execution."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from LangChain.medqa.data import Question, Dataset
from LangChain.medqa.results import split_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/results/medqa-v6-smoke-3-20261019-1304")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source / "structured_on.jsonl"
    manifest = json.loads((args.source / "manifest.json").read_text())
    data = Dataset(Path(manifest["settings"]["benchmark"]))
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    results = []
    total, prefix, retries = 0, 0, 0
    with sqlite3.connect((args.source / "checkpoint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        for position, row in enumerate(rows):
            calls = [e for e in row["events"] if "agent" in e]
            review_index = next(i for i, e in enumerate(calls) if e["agent"] == "VerifierReasoner")
            review = calls[review_index]
            payload = review["input"]
            condition = row["mode"] + "_on"
            packs = db.execute("SELECT request,result FROM operations WHERE condition=? AND position=? AND kind='evidence_view' AND state='complete'",
                               (condition, position)).fetchall()
            matching = [(json.loads(req), json.loads(pack)) for req, pack in packs
                        if json.loads(pack)["views"] == payload["evidence_pool"]]
            if not matching:
                raise ValueError("缺少首次回答时的精确原文/可见视图映射")
            request, pack = matching[0]
            question = Question(row["question_id"], payload["question"], payload["options"])
            output = deepcopy(review.get("normalized_output", review.get("output")))
            if not isinstance(output, dict):
                output = json.loads(review["response"])
            result = split_result(output, question, [t["id"] for t in payload["knowledge_tasks"]],
                                  payload["key_conditions"], request["documents"], pack["visible"])
            historical_prefix = sum(e["usage"]["total_tokens"] for e in calls[:review_index + 1])
            historical_total = row["metrics"]["total_tokens"]
            result.update(original_status=row["status"], reference_match_posthoc=data.grade(result),
                          historical_paid_total_tokens=historical_total, historical_prefix_tokens=historical_prefix,
                          historical_later_retry_tokens=historical_total - historical_prefix,
                          new_model_api_calls=0, interpretation="已存响应的结果层回放，不是新提示词的实测成绩/Token节省")
            results.append(result)
            total += historical_total
            prefix += historical_prefix
            retries += len(calls) - review_index - 1
    assert before == hashlib.sha256(source.read_bytes()).hexdigest()
    (args.output / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    report = {"model_api_called": False, "new_api_tokens": 0, "new_model_api_calls": 0,
              "source_sha256": before, "source_unchanged": True,
              "questions": len(results), "retained_answers": sum(r["answer_result"]["complete"] for r in results),
              "protocol_issue_questions": sum(bool(r["protocol_result"]["issues"]) for r in results),
              "strict_audit_failed_questions": sum(r["evidence_result"]["strict_audit_status"] == "failed" for r in results),
              "reliable_evidence_ids": {r["question_id"]: r["evidence_result"]["reliable_evidence_ids"] for r in results},
              "historical_paid_total_tokens": total, "historical_prefix_tokens": prefix,
              "historical_later_retry_tokens": total - prefix, "historical_later_retry_calls": retries,
              "experiment_matrix": ["text_off", "structured_off", "text_on", "structured_on"],
              "limits": ["仅验证结果解耦；不代表新提示真实运行稳定性或医学泛化准确率",
                         "历史重试费用已经发生，不能从账单删除或冒充已实现节省",
                         "四组均为相同三Agent流程；没有RAG/直接回答对照"]}
    (args.output / "replay-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
