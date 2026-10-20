"""需求与动作分开；检索命中仅候选，审核支持才允许完成。"""
from copy import deepcopy


class TaskStates:
    def __init__(self, tasks, *, resource_snapshot=None):
        self.resource_snapshot = resource_snapshot
        self.items = {t["id"]: {"lifecycle": "ready", "support": "unknown", "evidence_ids": [],
            "can_start": True, "can_complete": False, "can_close": True} for t in tasks}
        self.transitions = []

    def change(self, ident, lifecycle, trigger, **values):
        item = self.items[ident]
        before = deepcopy(item)
        item.update(values, lifecycle=lifecycle)
        item["can_start"] = lifecycle not in {"waiting_input", "closed_incomplete", "failed"}
        item["can_complete"] = item["support"] == "supported"
        if before != item:
            self.transitions.append({"task_id": ident, "before": before, "after": deepcopy(item),
                                     "trigger": trigger, "actor": "program",
                                     "resources": self.resource_snapshot() if self.resource_snapshot else {}})

    def candidate(self, ids, sources):
        for ident in sorted(ids):
            item = self.items[ident]
            self.change(ident, "waiting_evidence", "retrieval_candidate",
                        support="unknown", evidence_ids=sorted(set(item["evidence_ids"]) | set(sources)))

    def review(self, output):
        claims = {c["id"]: c for c in output.get("claims", [])}
        for ident in self.items:
            checks = [c for c in output.get("checks", []) if ident in claims[c["claim_id"]]["knowledge_task_ids"]]
            sources = sorted({s for c in checks for s in c["evidence_ids"]})
            supported = bool(checks) and all(c["verdict"] == "supported" for c in checks)
            # Scope/option audit can reopen apparently supported needs.
            relevant = {c["claim_id"] for c in checks}
            applicable = all(c["verdict"] == "matched" for c in output.get("condition_checks", [])
                             if relevant & set(c["claim_ids"]))
            supported &= applicable and not any(c["status"] == "unresolved" for c in output.get("conflicts", []))
            self.change(ident, "completed" if supported else "waiting_evidence", "reasoner_review",
                        support="supported" if supported else "partial" if sources else "insufficient",
                        evidence_ids=sources)

    def review_needs(self, reviews, evidence, visible):
        from .grounding import ContractError
        from .evidence_processor import validate_visible_quotes
        if not isinstance(reviews, list):
            raise ContractError("need_reviews须为列表")
        seen = set()
        for review in reviews:
            if (not isinstance(review, dict) or review.get("task_id") not in self.items
                    or review["task_id"] in seen or review.get("verdict") not in {"supported", "partial", "insufficient", "contradicted"}
                    or not isinstance(review.get("reason"), str) or not review["reason"].strip()):
                raise ContractError("需求审核ID/状态/理由无效")
            seen.add(review["task_id"])
            quotes = review.get("support_quotes", [])
            if not isinstance(quotes, list):
                raise ContractError("需求引句须为列表")
            for quote in quotes:
                if (not isinstance(quote, dict) or quote.get("evidence_id") not in evidence
                        or not isinstance(quote.get("quote"), str) or not 16 <= len(quote["quote"]) <= 500
                        or quote["quote"] not in evidence[quote["evidence_id"]]["text"]):
                    raise ContractError("需求审核必须保留真实来源引句", category="attribution")
            validate_visible_quotes({"claims": [{"support_quotes": quotes}]}, visible)
            supported = review["verdict"] == "supported"
            if supported and (not quotes or review.get("condition_verdict") != "matched"):
                raise ContractError("需求充分支持须有可见原文及适用性审核", category="evidence_judgment")
            self.change(review["task_id"], "completed" if supported else "waiting_evidence", "reasoner_need_review",
                        support=review["verdict"], evidence_ids=sorted({q["evidence_id"] for q in quotes}))

    def close(self, reason):
        for ident, item in self.items.items():
            if not item["can_complete"]:
                self.change(ident, "closed_incomplete", "question_closed", close_reason=reason)
