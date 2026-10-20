"""API用量只在输入、输出计数完整有效时标记为model；估算不用于成本结论。"""
from ..metrics.recorder import estimate_token_count


def normalize_usage(response, prompt_text, completion_text):
    details = getattr(response, "response_metadata", {}) or {}
    raw = details.get("token_usage", {}) if isinstance(details, dict) else {}
    metadata = getattr(response, "usage_metadata", None)
    issues = []
    for counts in (raw, metadata):
        if not isinstance(counts, dict) or not counts:
            continue
        def read(*keys):
            for key in keys:
                if key in counts:
                    value = counts[key]
                    return value if type(value) is int and value >= 0 else None
            return None
        prompt = read("prompt_tokens", "input_tokens")
        completion = read("completion_tokens", "output_tokens")
        total = read("total_tokens")
        if prompt is None or completion is None or prompt == 0:
            issues.append("missing_or_invalid_input_output")
            continue
        if "total_tokens" in counts and (total is None or total != prompt + completion):
            issues.append("inconsistent_total")
            continue
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": prompt + completion, "source": "model",
                "usage_complete": True, "total_derived": "total_tokens" not in counts}
    return {"prompt_tokens": estimate_token_count(prompt_text),
            "completion_tokens": estimate_token_count(completion_text),
            "total_tokens": estimate_token_count(prompt_text) + estimate_token_count(completion_text),
            "source": "estimated", "usage_complete": False,
            "usage_error": ",".join(issues) or "usage_absent"}
