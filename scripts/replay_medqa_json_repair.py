"""Audit saved JSON and replay recoverable answers without API or retrieval calls."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from LangChain.medqa.data import Dataset, Question
from LangChain.medqa.protocol import parse_agent_output
from LangChain.medqa.results import split_result


def fingerprint(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    protected = [source / "checkpoint.sqlite3", source / "manifest.json", *source.glob("*.jsonl")]
    before = fingerprint(protected)
    manifest = json.loads((source / "manifest.json").read_text())
    data = Dataset(Path(manifest["settings"]["benchmark"]))
    row_lookup = {}
    for path in source.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row_lookup[(path.stem, row["question_id"])] = row
    recovered, repairs, rejected = [], [], []
    examined = valid = 0
    with sqlite3.connect((source / "checkpoint.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for saved in db.execute("SELECT condition,position,ordinal,role,raw,usage FROM calls WHERE state='complete' ORDER BY condition,position,ordinal"):
            examined += 1
            raw, usage = json.loads(saved["raw"]), json.loads(saved["usage"])
            try:
                output, changes = parse_agent_output(raw, allow_repair=True,
                                                     truncated=usage.get("finish_reason") == "length")
            except ValueError as error:
                rejected.append({"condition": saved["condition"], "position": saved["position"],
                                 "role": saved["role"], "error_type": type(error).__name__})
                continue
            if not changes:
                valid += 1
                continue
            info = {"condition": saved["condition"], "position": saved["position"],
                    "role": saved["role"], "changes": changes}
            repairs.append(info)
            if saved["role"] != "VerifierReasoner":
                continue
            question_id = manifest["question_ids"][saved["position"]]
            row = row_lookup[(saved["condition"], question_id)]
            events = [e for e in row["events"] if e.get("agent") == saved["role"] and e.get("response") == raw]
            if len(events) != 1:
                raise ValueError("Need unique saved model request for offline result replay")
            payload = events[0]["input"]
            if output.get("question_id") != payload["question_id"]:
                raise ValueError("Repaired response question identity mismatch")
            packs = db.execute("SELECT request,result FROM operations WHERE condition=? AND position=? AND kind='evidence_view' AND state='complete'",
                               (saved["condition"], saved["position"])).fetchall()
            matching = [(json.loads(p["request"]), json.loads(p["result"])) for p in packs
                        if json.loads(p["result"])["views"] == payload["evidence_pool"]]
            if len(matching) != 1:
                raise ValueError("Need exact original source/visible evidence mapping")
            request, pack = matching[0]
            question = Question(question_id, payload["question"], payload["options"])
            memory_sources = {m["source_id"] for m in payload.get("memory_hits", [])
                              if m.get("status") == "source_checked_not_semantically_verified"}
            result = split_result(deepcopy(output), question, [t["id"] for t in payload["knowledge_tasks"]],
                                  payload["key_conditions"], request["documents"], pack["visible"], memory_sources)
            result["protocol_result"]["repairs"].append({"tool": "json_syntax_repair", "changes": changes, "extra_model_calls": 0})
            result["protocol_result"]["issues"].append({"category": "json_syntax", "code": "repaired_premature_root_close"})
            result["protocol_result"]["status"] = "issues"
            result.update(original_status=row["status"], condition=saved["condition"],
                          reference_match_after_replay=data.grade(result),
                          original_task_known_tokens=row["metrics"]["total_tokens"],
                          new_model_calls=0, online_experiment_result=False,
                          interpretation="Offline syntax/result replay; original scores, usage and traces unchanged")
            recovered.append(result)
        pending = [dict(r) for r in db.execute("SELECT condition,position,ordinal,role,state FROM calls WHERE state!='complete'")]
        completed = {r["condition"]: r["count"] for r in db.execute("SELECT condition,count(*) AS count FROM questions WHERE state='complete' GROUP BY condition")}
    assert fingerprint(protected) == before, "Original experiment artifacts changed"
    conditions = ["text_off", "structured_off", "text_on", "structured_on"]
    remaining = {c: manifest["question_ids"][completed.get(c, 0):] for c in conditions}
    report = {"saved_calls_examined": examined, "valid_json_unchanged": valid,
              "repaired_saved_calls": repairs, "rejected_saved_calls": rejected,
              "recovered_answers": len(recovered), "pending_calls_without_response": pending,
              "unexecuted_tasks": sum(len(ids) for ids in remaining.values()),
              "remaining_question_ids": remaining, "new_model_calls": 0,
              "source_unchanged": True, "source_fingerprints": before,
              "limits": ["Offline recovery is separate from original online results.",
                         "Missing API responses cannot be replayed or assigned zero cost.",
                         "Changed runtime source cannot bypass the original resume manifest check.",
                         "Replay/resumed wall time is not comparable to uninterrupted online time."]}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output / "recovered-answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recovered))
    print(json.dumps({k: report[k] for k in ("saved_calls_examined", "valid_json_unchanged", "recovered_answers",
                     "unexecuted_tasks", "new_model_calls", "source_unchanged")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
