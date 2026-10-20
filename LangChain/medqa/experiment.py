"""冻结记忆的来源审计；只读SQLite和实验产物，不调用模型。"""
import hashlib
import json
import sqlite3
import time
from contextlib import closing


def validate_seed(path, *, dataset_sha256, question_ids, namespace, task_group,
                  corpus_version, protocol, source_hashes, embedding_model=None, allow_partial=False):
    parent = path.parent
    if not path.is_file():
        raise ValueError("尚未生成记忆积累数据库")
    if not (parent / "manifest.json").is_file() or not (parent / "report.json").is_file():
        raise ValueError("积累库缺少manifest/report，不能确认来源")
    manifest = json.loads((parent / "manifest.json").read_text())
    report = json.loads((parent / "report.json").read_text())
    condition = path.stem
    settings = manifest["settings"]
    source_ids = manifest["question_ids"]
    stats = report.get(condition, {})
    if (manifest["dataset_sha256"] != dataset_sha256 or manifest["textbooks_version"] != corpus_version
            or manifest.get("runtime_protocol") != protocol or manifest["source_sha256"] != source_hashes):
        raise ValueError("积累库的数据、教材或代码版本与当前实验不一致")
    if (settings.get("task_group") != task_group
            or (settings.get("memory_namespace") or condition) != namespace):
        raise ValueError("积累库的任务组或命名空间不一致")
    if set(source_ids) & set(question_ids):
        raise ValueError("积累题目与评估题目重叠")
    if settings.get("ids") and not allow_partial:
        from pathlib import Path
        planned = json.loads(Path(settings["ids"]).read_text())["ids"]
        if source_ids != planned:
            raise ValueError("积累阶段只完成了预定题目的一部分")
    healthy = bool(source_ids) and (stats.get("questions_completed") == len(source_ids)
            and not stats.get("runtime_failed_questions", 1) and not stats.get("memory_degraded_questions", 1)
            and not stats.get("retrieval_failed_questions", 0) and stats.get("token_usage_complete"))
    if not healthy and not allow_partial:
        raise ValueError("积累阶段未完整健康完成，或用量未知")
    ledger = parent / "checkpoint.sqlite3"
    eligible_traces, unknown, completed_count = None, 0, stats.get("questions_completed", 0)
    known_tokens, ledger_usage_complete = stats.get("total_tokens", 0), True
    if ledger.is_file():
        with closing(sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            completed = db.execute("SELECT question_id,state,trace,result,position FROM questions WHERE condition=? ORDER BY position",
                                   (condition,)).fetchall()
            unknown = db.execute("SELECT count(*) FROM calls WHERE condition=? AND state<>'complete'",
                                 (condition,)).fetchone()[0]
            usages = [json.loads(row[0]) if row[0] else {} for row in db.execute(
                "SELECT usage FROM calls WHERE condition=? AND state='complete'", (condition,))]
            known_tokens = max(known_tokens, sum(u.get("total_tokens", 0) for u in usages if u.get("source") == "model"))
            ledger_usage_complete = all(u.get("source") == "model" for u in usages)
            eligible_traces = set()
            for qid, state, trace, raw, position in completed:
                if state != "complete" or qid not in source_ids or not raw:
                    continue
                result = json.loads(raw)
                metrics = result.get("metrics", {})
                # Per-record provenance changes; whole-answer memory admission itself is unchanged.
                valid = (result.get("answer") is not None and result.get("evidence_status") == "supported"
                    and (result.get("output_policy") != "decoupled" or result.get("evidence_result", {}).get("memory_eligible"))
                    and metrics.get("usage_missing_calls", 0) == 0 and metrics.get("checkpoint_unknown_calls", 0) == 0
                    and metrics.get("usage_sources") and set(metrics["usage_sources"]) == {"model"}
                    and not db.execute("SELECT 1 FROM calls WHERE condition=? AND position=? AND state<>'complete'",
                                       (condition, position)).fetchone())
                if valid:
                    eligible_traces.add(trace)
            completed_count = sum(row[1] == "complete" for row in completed)
        if not allow_partial and ([row[0] for row in completed] != source_ids or any(row[1] != "complete" for row in completed) or unknown):
            raise ValueError("积累检查点仍有未完成题目、待提交记忆或未知请求，不能用于评估")
    elif allow_partial:
        raise ValueError("局部积累库必须有检查点，以核对每条记忆的已提交来源")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT id, content_hash, embedding_model, content, trace_id FROM memory "
                          "WHERE namespace=? AND status='curated' AND (expires_at IS NULL OR expires_at>?) ORDER BY id",
                          (namespace, time.time())).fetchall()
    original_count = len(rows)
    if allow_partial:
        rows = [row for row in rows if row[4] in eligible_traces]
    if not rows:
        raise ValueError("积累库没有未过期的可复用记忆")
    if embedding_model and any(row[2] != embedding_model for row in rows):
        raise ValueError("积累库的embedding配置与当前不一致")
    if any(json.loads(row[3]).get("corpus_version") != corpus_version
           or json.loads(row[3]).get("task_group") != task_group for row in rows):
        raise ValueError("记忆条目的教材版本或任务组不一致")
    fingerprint = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()
    return {"source": str(path), "memory_records": len(rows), "memory_content_sha256": fingerprint,
            "memory_ids": [row[0] for row in rows], "excluded_memory_records": original_count - len(rows),
            "partial_source": not healthy or completed_count != len(source_ids) or bool(unknown),
            "accumulation_questions": len(source_ids), "accumulation_completed_questions": completed_count,
            "accumulation_total_tokens": known_tokens,
            "accumulation_cost_complete": bool(stats.get("token_usage_complete")) and ledger_usage_complete and not unknown,
            "accumulation_unknown_calls": unknown,
            "accumulation_runtime_failed_questions": stats.get("runtime_failed_questions", 0),
            "note": "仅冻结已提交且满足原整题准入门槛的记忆；局部失败、未完成及未知成本另行披露，积累成本单列。"}
