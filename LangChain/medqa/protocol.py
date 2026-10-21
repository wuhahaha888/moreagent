"""已知字段布局的无损映射；不生成判断、不修正事实或来源ID。"""
from copy import deepcopy
import json
import math
import re
from .grounding import ContractError


def parse_agent_output(raw, *, allow_repair=False, truncated=False):
    """Parse all content; optionally remove one premature root closing brace.

    No suffix is discarded and no value is invented. Duplicate keys, nonfinite
    numbers, multiple objects, prose and incomplete output remain errors.
    """
    if not isinstance(raw, str):
        return deepcopy(raw), []

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError("JSON对象包含重复字段", category="integrity")
            result[key] = value
        return result

    def reject_constant(_):
        raise ContractError("JSON不允许非有限数值", category="integrity")

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ContractError("JSON数值超出有限范围", category="integrity")
        return value

    decoder = json.JSONDecoder(object_pairs_hook=unique_object, parse_constant=reject_constant,
                               parse_float=finite_float)
    try:
        return decoder.decode(raw), []
    except json.JSONDecodeError as original_error:
        if not allow_repair or truncated or original_error.msg != "Extra data":
            raise
        start = len(raw) - len(raw.lstrip())
        try:
            first, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            raise original_error from None
        if (not isinstance(first, dict) or not first or end <= start
                or raw[end - 1] != "}" or not raw[end:].lstrip().startswith(",")):
            raise original_error
        # Deleting precisely the root close must make the entire string valid.
        # The strict decoder also rejects duplicate/conflicting tail fields.
        candidate = raw[:end - 1] + raw[end:]
        try:
            parsed = decoder.decode(candidate)
        except json.JSONDecodeError:
            raise original_error from None
        added = sorted(set(parsed) - set(first)) if isinstance(parsed, dict) else []
        if not added or any(parsed.get(key) != value for key, value in first.items()):
            raise original_error
        return parsed, [{"kind": "remove_premature_root_close", "character_offset": end - 1,
                         "removed_character": "}", "added_top_level_fields": added,
                         "extra_model_calls": 0}]


def normalize_response(role, output, evidence=None):
    result, mapped = deepcopy(output), []
    if role in {"资料搜索", "Investigator"}:
        for container, fields in (("observe", ("findings",)), ("request", ("search_request",)),
                                  ("search", ("findings", "gaps", "used_memory_ids"))):
            nested = result.get(container)
            if not isinstance(nested, dict):
                continue
            for field in fields:
                if field not in nested:
                    continue
                if field in result and result[field] != nested[field]:
                    raise ContractError("顶层与嵌套协议字段冲突")
                result[field] = nested[field]
                mapped.append(container + "." + field + "→" + field)
    if role in {"任务执行", "VerifierReasoner"} and evidence:
        claims = result.get("claims", []) if role == "VerifierReasoner" else (result.get("draft") or {}).get("claims", [])
        if not isinstance(claims, list):
            raise ContractError("claims必须为列表")
        for claim in claims:
            if not isinstance(claim, dict):
                raise ContractError("判断必须为对象")
            for citation in claim.get("support_quotes", []):
                ident, quote = citation.get("evidence_id"), citation.get("quote")
                if ident not in evidence or not isinstance(quote, str) or not quote or re.search(r"\d", quote):
                    continue
                text = evidence[ident]["text"]
                if quote in text:
                    continue
                positions = [i for i, char in enumerate(text) if not char.isspace()]
                plain = "".join(text[i] for i in positions)
                wanted = "".join(char for char in quote if not char.isspace())
                # 只允许非数字引句的空白差异，且唯一匹配；不改变字母、标点或临床判断。
                start = plain.find(wanted) if wanted else -1
                if start < 0 or plain.find(wanted, start + 1) >= 0:
                    continue
                exact = text[positions[start]:positions[start + len(wanted) - 1] + 1]
                if 16 <= len(exact.strip()) <= 500:
                    citation["quote"] = exact
                    mapped.append("引句空白差异→教材原始区间:" + ident)
    return result, mapped
