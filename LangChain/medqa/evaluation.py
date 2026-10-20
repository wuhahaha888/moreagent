"""盲标注材料与独立检索评测；不把模型分数或 MedQA 答案当作证据标签。"""
import json
import hashlib
import math
import random
from pathlib import Path


class EvaluationError(ValueError):
    """本模块产生的固定错误说明可安全显示给用户。"""


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(out, questions, textbooks, dataset_sha256, seed):
    manifest = {"schema": "struckflow.retrieval-pool/2", "dataset_sha256": dataset_sha256,
                "corpus_version": textbooks.version, "seed": seed,
                "question_ids": [q.id for q in questions],
                "query_policy": "题干原文；同一 BM25 候选池比较原排序与 MedCPT 重排",
                "positive_label": 2, "labels_generated": False,
                "recall_scope": "仅统计池内已判断的相关文档，不是全教材 Recall"}
    write_json(out / "manifest.json", manifest)
    labels = {"annotation_method": "independent_human", "annotator": "",
              "dataset_sha256": dataset_sha256, "corpus_version": textbooks.version, "cases": []}
    with (out / "packet.jsonl").open("w") as packet, (out / "rankings.jsonl").open("w") as runs:
        for index, question in enumerate(questions):
            ranked = textbooks.rankings(question.question)
            documents = [{k: d[k] for k in ("id", "title", "text", "sha256")} for d in ranked["bm25"]]
            random.Random(seed + index).shuffle(documents)
            packet.write(json.dumps({**question.public(), "candidate_documents": documents}, ensure_ascii=False) + "\n")
            runs.write(json.dumps({"question_id": question.id, "query": question.question,
                                  "rankings": {name: [d["id"] for d in docs] for name, docs in ranked.items()}}, ensure_ascii=False) + "\n")
            labels["cases"].append({"question_id": question.id, "judgments": [
                {"evidence_id": d["id"], "relevance": None, "rationale": "", "support_quote": ""} for d in documents]})
    write_json(out / "judgments-template.json", labels)
    for name in ("packet.jsonl", "rankings.jsonl"):
        manifest[name + "_sha256"] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    write_json(out / "manifest.json", manifest)
    return {"questions": len(questions), "candidate_pairs": sum(len(c["judgments"]) for c in labels["cases"]),
            "labels_generated": False}


def _unique(rows, field):
    result = {}
    for row in rows:
        if row[field] in result:
            raise EvaluationError("重复的题目或文档ID")
        result[row[field]] = row
    return result


def evaluate(packet_dir: Path, judgments: Path, k=4):
    if k < 1:
        raise EvaluationError("cutoff必须大于0")
    manifest = json.loads((packet_dir / "manifest.json").read_text())
    if manifest.get("schema") != "struckflow.retrieval-pool/2":
        raise EvaluationError("标注材料版本不兼容，请生成带内容hash的新标注材料")
    labels = json.loads(judgments.read_text())
    for name in ("packet.jsonl", "rankings.jsonl"):
        if hashlib.sha256((packet_dir / name).read_bytes()).hexdigest() != manifest[name + "_sha256"]:
            raise EvaluationError("标注材料或排序文件已改变，不能沿用旧版本标签")
    if labels.get("annotation_method") != "independent_human" or not labels.get("annotator", "").strip():
        raise EvaluationError("需要独立人工标注方法与标注者记录；模型标签不能当作独立标准")
    for key in ("dataset_sha256", "corpus_version"):
        if labels.get(key) != manifest[key]:
            raise EvaluationError("标签与标注材料的数据/语料版本不一致")
    packets = _unique([json.loads(line) for line in (packet_dir / "packet.jsonl").read_text().splitlines()], "question_id")
    rankings = _unique([json.loads(line) for line in (packet_dir / "rankings.jsonl").read_text().splitlines()], "question_id")
    cases = _unique(labels["cases"], "question_id")
    if not cases or set(cases) != set(packets) or set(cases) != set(rankings) or set(cases) != set(manifest["question_ids"]):
        raise EvaluationError("题目集合必须完整一致，不允许删掉难题后评分")
    scores, details = {name: [] for name in ("bm25", "medcpt")}, []
    for qid, case in cases.items():
        documents = _unique(packets[qid]["candidate_documents"], "id")
        if any(hashlib.sha256(doc["text"].encode()).hexdigest() != doc["sha256"] for doc in documents.values()):
            raise EvaluationError("标注材料的原文与hash不一致")
        judgments_by_id = _unique(case["judgments"], "evidence_id")
        if set(documents) != set(judgments_by_id):
            raise EvaluationError("所有池内候选必须标注；不能把未判断文档当作负例")
        relevance = {}
        for ident, label in judgments_by_id.items():
            grade = label.get("relevance")
            if type(grade) is not int or grade not in (0, 1, 2) or not label.get("rationale", "").strip():
                raise EvaluationError("仍有未完成标签：relevance需为0/1/2，并填写判断理由")
            quote = label.get("support_quote", "")
            if grade == 2 and (not isinstance(quote, str) or len(quote.strip()) < 16 or quote not in documents[ident]["text"]):
                raise EvaluationError("决策支持标签必须提供可回查的原文引句")
            relevance[ident] = grade
        positive = {ident for ident, grade in relevance.items() if grade == 2}
        per_query = {"question_id": qid, "pool_positive_documents": len(positive)}
        for name in scores:
            ranked = rankings[qid]["rankings"][name]
            if len(set(ranked)) != len(ranked) or set(ranked) != set(documents):
                raise EvaluationError("两种排序须覆盖同一候选池，不得补入未判断文档")
            top = ranked[:k]
            relevant = sum(i in positive for i in top)
            dcg = sum((2 ** relevance[i] - 1) / math.log2(n + 2) for n, i in enumerate(top))
            ideal = sorted(relevance.values(), reverse=True)[:k]
            idcg = sum((2 ** grade - 1) / math.log2(n + 2) for n, grade in enumerate(ideal))
            score = {"precision_at_k": relevant / k,
                     "pooled_recall_at_k": relevant / len(positive) if positive else None,
                     "mrr_at_k": next((1 / (n + 1) for n, i in enumerate(top) if i in positive), 0),
                     "ndcg_at_k": dcg / idcg if idcg else None}
            scores[name].append(score)
            per_query[name] = score
        details.append(per_query)
    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None
    return {"questions": len(cases), "cutoff": k, "annotation_method": "independent_human",
            "annotator": labels["annotator"], "judgment_coverage": 1.0,
            "queries_without_pool_positive": sum(d["pool_positive_documents"] == 0 for d in details),
            "metrics": {name: {key: mean([row[key] for row in rows]) for key in rows[0]} for name, rows in scores.items()},
            "per_question": details,
            "limitations": "Recall仅限同一BM25候选池；不能检验候选池之外漏检，也不证明人工标注独立性。"}
