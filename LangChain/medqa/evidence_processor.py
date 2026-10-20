"""保守抽取视图；原文不变，相似性只标记，不推断医学支持。"""
from copy import deepcopy
import hashlib
import math
import re
import time

VERSION = "extractive-v1"
SEPARATOR = "\n[未展开区间]\n"


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def terms(text):
    return set(re.findall(r"[a-zA-Z]{3,}", text.casefold()))


class EvidenceProcessor:
    def __init__(self, *, enabled=True, span_chars=1800, max_chars=12000, max_documents=8):
        if min(span_chars, max_chars, max_documents) <= 0:
            raise ValueError("证据视图预算须为正")
        self.enabled, self.span_chars = enabled, span_chars
        self.max_chars, self.max_documents = max_chars, max_documents

    def process(self, documents, *, query, targets=None, protected=None, expanded=()):
        start = time.monotonic()
        targets, protected = targets or {}, protected or {}
        views, visible, omitted, records, groups, candidates = [], {}, [], [], {}, {}
        remaining = self.max_chars
        # Prioritize existing citations and explicit expansion over unseen candidates.
        ordered = sorted(documents.values(), key=lambda d: (d["id"] not in protected and d["id"] not in expanded,))
        for doc in ordered:
            ident, text = doc["id"], doc["text"]
            if doc["sha256"] != sha(text):
                raise ValueError("不可变证据校验失败")
            group = groups.get(doc["sha256"]) if self.enabled else None
            candidates[ident] = sorted(set(targets.get(ident, [])))
            if group is not None:
                group.setdefault("aliases", []).append(ident)
                visible[ident] = deepcopy(visible[group["id"]])
                records.append({"source_id": ident, "action": "exact_duplicate", "canonical": group["id"]})
                continue
            limit = len(text) if not self.enabled or ident in expanded else self.span_chars
            ranges = self._ranges(text, query, limit, protected.get(ident, []))
            cost = sum(end - begin for begin, end in ranges) + len(SEPARATOR) * max(0, len(ranges) - 1)
            if len(views) >= self.max_documents or cost > remaining:
                omitted.append(ident)
                records.append({"source_id": ident, "action": "deferred_context_budget"})
                continue
            spans = [{"id": ident + ":" + sha(doc["sha256"] + ":" + str(a) + ":" + str(b))[:12],
                      "start": a, "end": b, "text": text[a:b]} for a, b in ranges]
            # Offset maps and candidate coverage are audit data. Models already receive
            # source IDs/text and task_status; repeating these maps adds no decision value.
            view = {"id": ident, "title": doc.get("title", ident),
                    "text": SEPARATOR.join(s["text"] for s in spans)}
            if ranges != [(0, len(text))]:
                view["has_more"] = True
            views.append(view)
            visible[ident] = deepcopy(spans)
            groups[doc["sha256"]] = view
            remaining -= cost
            records.append({"source_id": ident, "action": "full" if not view.get("has_more") else "extract",
                            "raw_sha256": doc["sha256"], "ranges": ranges})
        similar = []
        # Lexical cosine is a redundancy risk signal, not an embedding or equivalence proof.
        for i, left in enumerate(views):
            a = terms(left["text"])
            for right in views[i + 1:]:
                b = terms(right["text"])
                score = len(a & b) / math.sqrt(len(a) * len(b)) if a and b else 0
                if score >= .8:
                    similar.append({"sources": [left["id"], right["id"]], "lexical_cosine": round(score, 4),
                                    "action": "retained_both"})
        return {"version": VERSION, "views": views, "visible": visible, "omitted": omitted,
                "candidate_coverage": candidates,
                "records": records, "similar_groups": similar,
                "raw_chars": sum(len(d["text"]) for d in documents.values()),
                "view_chars": self.max_chars - remaining, "cpu_seconds": time.monotonic() - start}

    @staticmethod
    def _ranges(text, query, limit, quotes):
        if len(text) <= limit:
            return [(0, len(text))]
        # Contiguous windows retain neighboring context; unexplored text remains available.
        wanted = terms(query)
        windows = [(a, min(len(text), a + limit)) for a in range(0, len(text), max(1, limit // 2))]
        a, b = max(windows, key=lambda r: len(terms(text[r[0]:r[1]]) & wanted))
        # Include adjacent sentence boundaries where possible; truncation remains explicitly expandable.
        left = list(re.finditer(r"[.!?]\s+|\n\s*\n", text[max(0, a - 300):a]))
        a = max(0, a - 300) + left[-1].end() if left else max(0, a - 300)
        right = re.search(r"[.!?](?:\s|$)|\n\s*\n", text[b:min(len(text), b + 300)])
        b = b + right.end() if right else min(len(text), b + 300)
        ranges = [(a, b)]
        for quote in quotes:
            pos = text.find(quote)
            if pos >= 0:
                ranges.append((max(0, pos - 200), min(len(text), pos + len(quote) + 200)))
        merged = []
        for a, b in sorted(ranges):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged


def validate_view(pack, documents):
    """Check restored raw-view mappings before any paid request can consume them."""
    from .grounding import ContractError
    for ident, spans in pack["visible"].items():
        if ident not in documents:
            raise ContractError("证据视图包含未知原文", category="integrity")
        raw = documents[ident]["text"]
        for span in spans:
            if (not isinstance(span.get("start"), int) or not isinstance(span.get("end"), int)
                    or not 0 <= span["start"] <= span["end"] <= len(raw)
                    or raw[span["start"]:span["end"]] != span["text"]):
                raise ContractError("证据片段偏移/原文不一致", category="integrity")
    for view in pack["views"]:
        if (view["id"] not in pack["visible"]
                or view["text"] != SEPARATOR.join(s["text"] for s in pack["visible"][view["id"]])):
            raise ContractError("实际模型正文与已保存片段不一致", category="integrity")
        for alias in view.get("aliases", []):
            if (alias not in documents or documents[alias]["sha256"] != documents[view["id"]]["sha256"]
                    or pack["visible"].get(alias) != pack["visible"][view["id"]]):
                raise ContractError("重复来源别名与原文不一致", category="integrity")


def validate_visible_quotes(output, visible):
    from .grounding import ContractError
    claims = output.get("claims", [])
    if not isinstance(claims, list):
        raise ContractError("claims须为列表")
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("support_quotes", []), list):
            raise ContractError("判断及support_quotes形状无效")
        for citation in claim.get("support_quotes", []):
            if not isinstance(citation, dict):
                raise ContractError("引用须为对象")
            ident, quote = citation.get("evidence_id"), citation.get("quote")
            if not isinstance(quote, str) or not any(quote in s["text"] for s in visible.get(ident, [])):
                raise ContractError("引用未在当前请求的证据片段展示；需展开或改用实际可见引句", category="attribution")
