"""Answer, evidence and protocol are independent products; no repair model calls."""
from copy import deepcopy
from .answer_audit import validate_answer
from .evidence_processor import validate_visible_quotes
from .grounding import ContractError


def split_result(output, question, tasks, conditions, evidence, visible, memory_sources=()):
    issues, citations = [], []

    def issue(code, path, message, category="protocol"):
        issues.append({"category": category, "code": code, "path": path, "message": message})

    proposed = output.get("final_answer")
    if not isinstance(proposed, dict):
        proposed = {}
        issue("missing_answer_object", "final_answer", "需要明确的答案对象")
    answer = proposed.get("answer")
    valid_answer = isinstance(answer, str) and answer in question.options
    if output.get("question_id", question.id) != question.id:
        issue("question_mismatch", "question_id", "响应属于另一题，不能作为当前答案", "identity")
        valid_answer = False
    if not valid_answer:
        issue("invalid_option", "final_answer.answer", "没有取得当前题目的有效选项")
    summary = proposed.get("summary", "")
    if not isinstance(summary, str):
        issue("invalid_summary", "final_answer.summary", "说明须为文本；保留有效选项")
        summary = ""
    gaps = output.get("critical_gaps", output.get("gaps", []))
    if not isinstance(gaps, list):
        issue("invalid_gaps", "critical_gaps", "未解问题须为列表")
        gaps = []
    elif any(not isinstance(g, str) for g in gaps):
        issue("invalid_gap_item", "critical_gaps", "忽略非文本缺口，记录字段问题")
    gaps = [g for g in gaps if isinstance(g, str)]

    # Validate each citation separately, retaining rejected records for inspection.
    groups = []
    if "citations" in output:
        groups.append(("citations", output["citations"], None, None))
    claims = output.get("claims", [])
    if not isinstance(claims, list):
        issue("invalid_claims", "claims", "判断须为列表")
        claims = []
    checks = output.get("checks", [])
    checks_by_id = {c.get("claim_id"): c for c in checks if isinstance(c, dict) and isinstance(c.get("claim_id"), str)} if isinstance(checks, list) else {}
    question_text = question.question + "\n" + "\n".join(question.options.values())
    for i, claim in enumerate(claims):
        path = "claims[" + str(i) + "]"
        if not isinstance(claim, dict):
            issue("invalid_claim", path, "判断须为对象")
            continue
        ident = claim.get("id")
        check = checks_by_id.get(ident, {}) if isinstance(ident, str) else {}
        groups.append((path + ".support_quotes", claim.get("support_quotes", []), ident, check.get("verdict")))
        basis, quotes = claim.get("basis", []), claim.get("question_quotes", [])
        if not isinstance(basis, list) or not isinstance(quotes, list):
            issue("invalid_attribution_shape", path, "依据与题干引文须为列表", "attribution")
        else:
            if bool(quotes) != ("question" in basis):
                issue("question_attribution_mismatch", path + ".question_quotes", "题干引文角色与basis不一致；不自动补标签", "attribution")
            if any(not isinstance(q, str) or not q.strip() or q not in question_text for q in quotes):
                issue("invalid_question_quote", path + ".question_quotes", "题干引文不是实际连续原文", "attribution")
        declared = claim.get("evidence_ids", [])
        if isinstance(declared, list) and isinstance(claim.get("support_quotes", []), list):
            quoted = {q.get("evidence_id") for q in claim.get("support_quotes", [])
                      if isinstance(q, dict) and isinstance(q.get("evidence_id"), str)}
            for source in declared:
                if isinstance(source, str) and source not in quoted:
                    issue("missing_source_quote", path + ".evidence_ids", "来源缺少对应引句：" + source, "citation")

    for path, items, claim_id, claim_support in groups:
        if not isinstance(items, list):
            issue("invalid_citations", path, "引用须为列表", "citation")
            continue
        for index, item in enumerate(items[:32]):
            loc = path + "[" + str(index) + "]"
            if not isinstance(item, dict):
                issue("invalid_citation", loc, "引用须为对象", "citation")
                continue
            ident, quote = item.get("evidence_id"), item.get("quote")
            support = item.get("support", claim_support or "unknown")
            support = support if support in ("supported", "partial", "unsupported", "contradicted", "unknown") else "unknown"
            source_valid = (isinstance(ident, str) and ident in evidence and isinstance(quote, str)
                            and 16 <= len(quote.strip()) <= 500 and quote in evidence[ident]["text"])
            shown = source_valid and any(quote in s["text"] for s in visible.get(ident, []))
            if not source_valid:
                issue("invalid_source_quote", loc, "来源未知、缺引句或引句不匹配原文", "citation")
            elif not shown:
                issue("quote_not_visible", loc, "引句未在当前请求展示", "citation")
            citations.append({"evidence_id": ident if isinstance(ident, str) else None,
                "quote": quote if isinstance(quote, str) else None, "claim_id": claim_id,
                "source_verified": bool(source_valid), "visible": bool(shown),
                "model_reported_support": support, "support_audit_passed": False,
                "eligible_as_reliable_support": False})
        if len(items) > 32:
            issue("citation_limit", path, "超过32条的引用未审核", "citation")

    # Legacy full audit is optional metadata. Its failure never clears a valid option.
    audit, audit_status = None, "not_requested"
    if claims and all(k in output for k in ("checks", "condition_checks", "option_comparison")):
        try:
            validate_visible_quotes(output, visible)
            audit = validate_answer(output, question, set(tasks), conditions, evidence, set(memory_sources))
            audit_status = "passed"
        except (ContractError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            audit_status = "failed"
            issue("strict_audit_failed", "evidence_audit", str(exc), "evidence_audit")
    if audit and audit["evidence_status"] == "supported":
        grounded = set(audit["grounded_claim_ids"])
        for citation in citations:
            citation["eligible_as_reliable_support"] = bool(citation["source_verified"] and citation["visible"]
                and citation["claim_id"] in grounded and citation["model_reported_support"] == "supported")
            citation["support_audit_passed"] = citation["eligible_as_reliable_support"]
    reliable = sorted({c["evidence_id"] for c in citations if c["eligible_as_reliable_support"]})
    source_ids = sorted({c["evidence_id"] for c in citations if c["source_verified"] and c["visible"]})
    status = audit["evidence_status"] if audit else "unverified" if citations else "not_provided"
    declared_support = proposed.get("evidence_status")
    if declared_support not in ("supported", "partial", "insufficient", "contradicted", "unverified"):
        declared_support = "unknown"
    basis = audit["answer_basis"] if audit else []
    final = {"question_id": question.id, "answer": answer if valid_answer else None,
        "status": "answered" if valid_answer else "no_answer", "summary": summary,
        "evidence_status": status, "answer_basis": basis, "evidence_ids": source_ids, "gaps": gaps,
        "answer_result": {"answer": answer if valid_answer else None, "complete": valid_answer,
                          "status": "answered" if valid_answer else "no_answer", "summary": summary,
                          "unresolved_questions": gaps, "correctness": "not_assessed_by_runtime"},
        "evidence_result": {"status": status, "declared_status": proposed.get("evidence_status"),
                            "verification_status": "same_agent_checked" if audit else "unverified",
                            "support_assessment": {"status": audit["evidence_status"] if audit else declared_support,
                                                   "assessor": "same_agent", "independently_verified": False},
                            "citations": citations, "source_verified_ids": source_ids,
                            "reliable_evidence_ids": reliable, "strict_audit_status": audit_status,
                            "memory_eligible": bool(valid_answer and audit and audit["evidence_status"] == "supported" and reliable),
                            "note": "原文匹配与模型报告的支持关系分别记录；完整自检也不是独立医学认证。"},
        "protocol_result": {"status": "issues" if issues else "valid", "issues": issues,
                            "repairs": [], "llm_repairs": 0, "blocks_answer": not valid_answer}}
    if audit:
        final.update(declared_evidence_status=audit["declared_evidence_status"], audit_warnings=audit["audit_warnings"])
    return final


def failure_results(final, events):
    """Even an execution failure has three outputs; preserve costs in Runtime.metrics."""
    reason = final.get("summary", final.get("status", "unknown"))
    return {"answer_result": {"answer": None, "complete": False, "status": final.get("status"),
                              "summary": reason, "unresolved_questions": final.get("gaps", [])},
            "evidence_result": {"status": "not_assessed", "citations": [], "source_verified_ids": [],
                                "verification_status": "unverified", "support_assessment": {
                                    "status": "unknown", "assessor": "none", "independently_verified": False},
                                "reliable_evidence_ids": [], "strict_audit_status": "not_requested", "memory_eligible": False},
            "protocol_result": {"status": "invalid", "issues": [],
                                "repairs": [], "llm_repairs": 0, "blocks_answer": True}}
