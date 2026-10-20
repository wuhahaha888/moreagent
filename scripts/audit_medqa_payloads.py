"""Offline attribution of saved requests; never invokes a model or changes results.

Provider usage remains authoritative. Local token offsets provide a reproducible
content attribution, with all provider/local differences kept as a residual.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def field_group(key):
    if key in {"question_id", "question", "options", "task_group"}:
        return "task_input"
    if key in {"evidence_pool", "evidence_index", "evidence_deferred"}:
        return "evidence"
    if key == "memory_hits":
        return "history_memory"
    if key in {"budget", "allowed_actions"}:
        return "runtime_and_framing"
    return "agent_intermediate"


def labeled_prompt(system, payload, mode, compact, render_text):
    # Keep message framing explicit; do not infer an undocumented server template.
    pieces, spans, size = [], [], 0

    def add(text, group):
        nonlocal size
        if text:
            pieces.append(text)
            spans.append((size, size + len(text), group))
            size += len(text)

    add("<｜begin▁of▁sentence｜>", "runtime_and_framing")
    add(system, "fixed_prompt")
    add("<｜User｜>", "runtime_and_framing")
    if mode == "structured":
        add("{", "runtime_and_framing")
    for index, (key, value) in enumerate(payload.items()):
        if index:
            add("," if mode == "structured" else "\n", "runtime_and_framing")
        text = compact({key: value})[1:-1] if mode == "structured" else render_text({key: value})
        add(text, field_group(key))
    if mode == "structured":
        add("}", "runtime_and_framing")
    add("<｜Assistant｜></think>", "runtime_and_framing")
    prompt = "".join(pieces)
    body = compact(payload) if mode == "structured" else render_text(payload)
    assert prompt == "<｜begin▁of▁sentence｜>" + system + "<｜User｜>" + body + "<｜Assistant｜></think>"
    return prompt, spans, body


def attribute(tokenizer, prompt, spans):
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    ends = [end for _, end, _ in spans]
    counts, boundary_tokens = Counter(), 0
    for start, end in encoded.offsets:
        if start == end:
            counts["runtime_and_framing"] += 1
            continue
        scores = Counter()
        i = bisect_right(ends, start)
        while i < len(spans) and spans[i][0] < end:
            left, right, group = spans[i]
            scores[group] += max(0, min(end, right) - max(start, left))
            i += 1
        assert scores, (start, end)
        boundary_tokens += len(scores) > 1
        # The longest character overlap owns a token spanning two categories.
        counts[max(scores, key=scores.get)] += 1
    assert sum(counts.values()) == len(encoded.ids)
    return counts, boundary_tokens


def audit_condition(condition, rows, roles, tokenizer, compact, render_text):
    tokens, chars, by_role = Counter(), Counter(), defaultdict(Counter)
    provider, evidence_roles, overlap, cross_task = Counter(), defaultdict(Counter), Counter(), Counter()
    prior_queries, prior_sources = set(), set()
    memory_observations = Counter()
    records = []
    for row in rows:
        seen_views = {}
        for event in row["events"]:
            if "agent" not in event or "input" not in event:
                continue
            role = event["agent"]
            system = roles.prompts[role]
            assert len(system) == event["system_chars"]
            prompt, spans, body = labeled_prompt(system, event["input"], condition.split("_")[0], compact, render_text)
            counts, boundaries = attribute(tokenizer, prompt, spans)
            usage = event["usage"]
            assert usage["source"] == "model"
            residual = usage["prompt_tokens"] - sum(counts.values())
            counts["provider_residual"] = residual
            counts["model_output"] = usage["completion_tokens"]
            assert sum(counts.values()) == usage["total_tokens"]
            tokens.update(counts)
            by_role[role].update(counts)
            for start, end, group in spans:
                chars[group] += end - start
            chars["model_output"] += len(event["response"])
            output_local = len(tokenizer.encode(event["response"], add_special_tokens=False).ids)
            records.append({"question_id": row["question_id"], "agent": role,
                "provider_prompt_tokens": usage["prompt_tokens"], "provider_completion_tokens": usage["completion_tokens"],
                "provider_minus_local_prompt_tokens": residual,
                "provider_minus_local_output_tokens": usage["completion_tokens"] - output_local,
                "local_attributed_tokens": dict(counts), "cross_category_boundary_tokens": boundaries})
            provider.update({key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")})
            provider["request_chars"] += len(body)
            for doc in event["input"].get("evidence_pool", []):
                text = doc["text"]
                evidence_roles[role].update(chars=len(text), views=1)
                old = seen_views.get(doc["id"])
                if old is not None:
                    overlap["repeated_source_views"] += 1
                    if text == old:
                        overlap["exact_same_view_chars"] += len(text)
                        overlap["exact_same_views"] += 1
                    elif old in text:
                        overlap["earlier_view_contained_chars"] += len(old)
                        overlap["earlier_view_contained_views"] += 1
                seen_views[doc["id"]] = text
        task_queries, task_sources = set(), set()
        for search in row["retrieval_history"]:
            query = " ".join(search["query"].casefold().split())
            cross_task["searches"] += 1
            cross_task["exact_query_seen_in_previous_task"] += query in prior_queries
            cross_task["exact_query_repeated_in_same_task"] += query in task_queries
            task_queries.add(query)
            for source in search.get("evidence_ids", []):
                cross_task["source_results"] += 1
                cross_task["source_result_seen_in_previous_task"] += source in prior_sources
                task_sources.add(source)
        investigator = next(e for e in row["events"] if e.get("agent") == "Investigator")
        memory_sources = {h["source_id"] for h in investigator["input"].get("memory_hits", []) if h.get("source_id")}
        memory_observations["source_task_pairs_delivered_before_search"] += len(memory_sources)
        memory_observations["sources_also_returned_by_current_search"] += len(memory_sources & task_sources)
        memory_observations["tasks_with_memory_sources_and_no_search"] += bool(memory_sources) and not row["retrieval_history"]
        prior_queries.update(task_queries)
        prior_sources.update(task_sources)
    return {"calls": len(records), "provider_usage": dict(provider), "local_attributed_tokens": dict(tokens),
        "local_attributed_tokens_by_role": dict(by_role), "labeled_prompt_characters": dict(chars),
        "provider_prompt_residual_histogram": dict(Counter(x["provider_minus_local_prompt_tokens"] for x in records)),
        "provider_output_residual_histogram": dict(Counter(x["provider_minus_local_output_tokens"] for x in records)),
        "evidence_text_by_role": dict(evidence_roles), "within_task_evidence_overlap": dict(overlap),
        "cross_task_search": dict(cross_task), "memory_source_observations": dict(memory_observations),
        "replaced_search_count": None, "calls_detail": records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from tokenizers import Tokenizer
    from LangChain.medqa.runtime import ChatRoles, compact, render_text
    manifest = json.loads((args.run / "manifest.json").read_text())
    changed = [name for name, sha in manifest["source_sha256"].items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha]
    if changed:
        raise ValueError("当前源码与已保存实验不匹配，不能用新提示重建旧输入: " + str(changed))
    settings = manifest["settings"]
    roles = ChatRoles(None, settings["context_profile"], settings["output_policy"], settings["memory_selection"])
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    report = json.loads((args.run / "report.json").read_text())
    result = {"run": str(args.run.resolve()), "model_calls_during_analysis": 0,
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "attribution_method": "Pinned official local tokenizer; tokens assigned by longest overlap with disjoint field spans. Provider/local prompt residual is never redistributed.",
        "limitations": ["分类Token是可重建的本地归因，API只提供每次调用总量；边界Token归属是统计约定。",
            "provider_residual保留服务端模板或配置差异，不推断为固定提示或证据。",
            "evidence包含正文、引用及目录；history_memory是memory_hits，已核查的记忆正文在evidence中，只计一次。",
            "原文重复不一定冗余：首次接收的新角色可能仍需要它；相同来源ID不等于相同区间或相同信息需求。",
            "相同来源再次出现不代表可直接跳过检索；相似查询也不等于可安全复用。"], "conditions": {}}
    for condition, stats in report.items():
        rows = [json.loads(line) for line in (args.run / (condition + ".jsonl")).read_text().splitlines() if line.strip()]
        item = audit_condition(condition, rows, roles, tokenizer, compact, render_text)
        for key in ("total_tokens", "prompt_tokens", "completion_tokens", "request_chars"):
            assert item["provider_usage"][key] == stats[key], (condition, key)
        result["conditions"][condition] = item
        print(condition, json.dumps({"calls": item["calls"], "tokens": item["local_attributed_tokens"],
            "prompt_residual": item["provider_prompt_residual_histogram"],
            "output_residual": item["provider_output_residual_histogram"]}, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
