"""分开校验选项和外部证据支持；语义标签仍由模型判断，不是医学正确性证明。"""
from .grounding import ContractError, identifiers


BASES = {"question", "textbook", "memory", "model_knowledge"}
VERDICTS = {"supported", "partial", "unsupported", "contradicted"}
CONDITION_VERDICTS = {"matched", "mismatched", "unclear"}
ANSWER_STATUSES = {"supported": "supported", "partial": "answered_partial",
                   "insufficient": "answered_unsupported"}


def nonempty_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(label + "需要非空说明")


def validate_answer(output, question, tasks, conditions, evidence, memory_source_ids):
    """四项检查与来源一致性：不因缺少教材就拒绝合法的best-effort选项。"""
    final = output.get("final_answer")
    if not isinstance(final, dict) or final.get("answer") not in question.options:
        raise ContractError("最终选项无效")
    nonempty_text(final.get("summary"), "最终答案")
    claims = output.get("claims")
    if not isinstance(claims, list) or not 1 <= len(claims) <= 8:
        raise ContractError("需要1至8项关键判断，通常仅列1至3项")
    known, covered, cited, bases, external = {}, set(), set(), set(), set()
    question_text = question.question + "\n" + "\n".join(question.options.values())
    for claim in claims:
        if not isinstance(claim, dict):
            raise ContractError("关键判断必须为对象")
        ident = claim.get("id")
        if not isinstance(ident, str) or not ident.strip() or ident in known:
            raise ContractError("关键判断ID为空或重复")
        nonempty_text(claim.get("claim"), "关键判断")
        covered |= identifiers(claim.get("knowledge_task_ids"), tasks)
        basis = identifiers(claim.get("basis"), BASES)
        sources = identifiers(claim.get("evidence_ids"), evidence, empty=True)
        has_external = bool(basis & {"textbook", "memory"})
        if bool(sources) != has_external:
            raise ContractError("教材/记忆依据必须对应实际来源；模型知识或题干不能冒充教材")
        if "memory" in basis and not sources & memory_source_ids:
            raise ContractError("记忆依据必须引用本题实际回读的历史来源")
        if sources & memory_source_ids and "memory" not in basis:
            raise ContractError("实际使用历史来源时须标明memory依据")
        if sources - memory_source_ids and "textbook" not in basis:
            raise ContractError("非历史回读的教材来源须标明textbook依据")
        quotes = claim.get("support_quotes")
        if not isinstance(quotes, list) or len(quotes) > 12:
            raise ContractError("support_quotes必须是有限原文列表，可为空")
        quoted = set()
        for quote in quotes:
            if not isinstance(quote, dict) or quote.get("evidence_id") not in sources:
                raise ContractError("引句来源不属于该判断")
            text = quote.get("quote")
            if not isinstance(text, str) or not 16 <= len(text.strip()) <= 500 or text not in evidence[quote["evidence_id"]]["text"]:
                raise ContractError("引句必须为所引教材中16至500字符的连续原文")
            quoted.add(quote["evidence_id"])
        if quoted != sources:
            raise ContractError("每项引用来源都须附真实原文")
        question_quotes = claim.get("question_quotes", [])
        if (not isinstance(question_quotes, list) or bool(question_quotes) != ("question" in basis)
                or any(not isinstance(q, str) or not q.strip() or q not in question_text for q in question_quotes)):
            raise ContractError("question依据须引用题干/选项原文；背景引文请单列context_quotes，不能自动补事实依据", category="attribution")
        context_quotes = claim.get("context_quotes", [])
        if not isinstance(context_quotes, list) or any(not isinstance(q, str) or not q.strip() or q not in question_text for q in context_quotes):
            raise ContractError("context_quotes只能为真实题干背景，不计入支持依据", category="attribution")
        known[ident] = claim
        cited |= sources
        bases |= basis
        if has_external:
            external.add(ident)
    unresolved = output.get("unresolved_tasks", [])
    if not isinstance(unresolved, list):
        raise ContractError("unresolved_tasks须为列表")
    unresolved_ids = set()
    for item in unresolved:
        if not isinstance(item, dict) or item.get("task_id") not in tasks or item["task_id"] in unresolved_ids:
            raise ContractError("未解需求ID无效或重复")
        nonempty_text(item.get("reason"), "未解需求")
        unresolved_ids.add(item["task_id"])
    if covered | unresolved_ids != set(tasks):
        raise ContractError("全部规划需求须被判断关联或明确记入unresolved_tasks，不得捏造已支持判断")

    # 第一项：每个关键判断的支持关系，而不是只检查引用存在。
    checks, by_claim = output.get("checks"), {}
    if not isinstance(checks, list) or len(checks) != len(known):
        raise ContractError("需要逐项审核全部关键判断")
    for check in checks:
        if not isinstance(check, dict) or check.get("claim_id") not in known or check["claim_id"] in by_claim:
            raise ContractError("判断审核ID缺失、未知或重复")
        ident = check["claim_id"]
        if check.get("verdict") not in VERDICTS:
            raise ContractError("无效的判断支持关系")
        nonempty_text(check.get("reason"), "判断审核")
        if identifiers(check.get("evidence_ids"), evidence, empty=True) != set(known[ident]["evidence_ids"]):
            raise ContractError("判断审核须对应其全部引用来源")
        if "model_knowledge" in known[ident]["basis"] and check["verdict"] == "supported":
            raise ContractError("依赖模型自身知识的判断不能标为已有证据充分支持", category="evidence_judgment")
        by_claim[ident] = check

    # 第二项：题干事实不需要教材证明；这里审核医学规则的适用条件。
    expected = {c["id"] for c in conditions}
    condition_checks, by_condition = output.get("condition_checks"), {}
    if not isinstance(condition_checks, list) or len(condition_checks) != len(expected):
        raise ContractError("需要逐项核对全部题干关键条件的适用性")
    invalid_scope, scope_covered = set(), set()
    for check in condition_checks:
        if not isinstance(check, dict) or check.get("condition_id") not in expected or check["condition_id"] in by_condition:
            raise ContractError("条件审核ID未知或重复")
        if check.get("verdict") not in CONDITION_VERDICTS:
            raise ContractError("条件适用性须为matched/mismatched/unclear")
        nonempty_text(check.get("reason"), "条件适用性")
        linked = identifiers(check.get("claim_ids"), known)
        scope_covered |= linked
        if check["verdict"] != "matched":
            invalid_scope |= linked
        by_condition[check["condition_id"]] = check
    if scope_covered != set(known):
        raise ContractError("全部关键判断都须说明题干条件下的适用性")

    # 第三项：对主要竞争选项的区分，不能用术语相关性替代。
    comparison = output.get("option_comparison")
    if (not isinstance(comparison, dict) or comparison.get("alternative") not in question.options
            or comparison["alternative"] == final["answer"] or comparison.get("verdict") not in VERDICTS - {"contradicted"}):
        raise ContractError("须说明所选项与一个有效主要竞争选项的区分")
    nonempty_text(comparison.get("reason"), "选项区分")
    comparison_claims = identifiers(comparison.get("claim_ids"), known)
    if comparison["verdict"] == "supported" and any(by_claim[i]["verdict"] != "supported" or i in invalid_scope for i in comparison_claims):
        raise ContractError("不能用尚未支持的判断宣称选项区分已有支持", category="evidence_judgment")

    # 第四项：仍可能改变选项的缺口与已知冲突。
    gaps = output.get("critical_gaps")
    if not isinstance(gaps, list) or any(not isinstance(g, str) or not g.strip() for g in gaps):
        raise ContractError("critical_gaps须明确列出，可为空列表")
    conflicts = output.get("conflicts", [])
    if not isinstance(conflicts, list):
        raise ContractError("冲突须为列表")
    for conflict in conflicts:
        if not isinstance(conflict, dict) or conflict.get("status") not in {"resolved", "unresolved"}:
            raise ContractError("冲突状态无效")
        identifiers(conflict.get("evidence_ids"), evidence)
        nonempty_text(conflict.get("reason"), "冲突")
    unresolved = any(c["status"] == "unresolved" for c in conflicts)
    grounded = {i for i in external if by_claim[i]["verdict"] == "supported"
                and "model_knowledge" not in known[i]["basis"] and i not in invalid_scope}
    sufficient = (not unresolved_ids and bool(grounded) and all(c["verdict"] == "supported" for c in checks)
                  and all(c["verdict"] == "matched" for c in condition_checks)
                  and "model_knowledge" not in bases and comparison["verdict"] == "supported"
                  and bool(comparison_claims & grounded) and not gaps and not unresolved)
    incompatible = unresolved or any(c["verdict"] == "contradicted" for c in checks) or any(c["verdict"] == "mismatched" for c in condition_checks)
    computed = "supported" if sufficient else "partial" if grounded and not incompatible else "insufficient"
    declared = final.get("evidence_status")
    if declared not in ANSWER_STATUSES:
        raise ContractError("证据状态须为supported/partial/insufficient")
    # 只允许程序下调夸大的充分性，不提升模型声明，也不修改任何判断或原文。
    rank = {"insufficient": 0, "partial": 1, "supported": 2}
    status = min((computed, declared), key=rank.get)
    warnings = ["证据状态由" + declared + "下调为" + status] if rank[declared] > rank[status] else []
    if status != "supported" and not gaps:
        raise ContractError("证据未充分支持时必须说明仍未解决的关键缺口")
    if identifiers(final.get("answer_basis"), BASES) != bases:
        raise ContractError("最终答案依据必须准确汇总各判断的知识来源")
    if identifiers(final.get("evidence_ids"), evidence, empty=True) != cited:
        raise ContractError("最终引用必须与实际审核来源一致")
    if "status" in final and final["status"] != ANSWER_STATUSES[status]:
        raise ContractError("运行状态与证据状态不一致")
    return {"evidence_status": status, "declared_evidence_status": declared, "audit_warnings": warnings,
            "status": ANSWER_STATUSES[status],
            "answer_basis": sorted(bases), "gaps": list(gaps), "grounded_claim_ids": sorted(grounded)}
