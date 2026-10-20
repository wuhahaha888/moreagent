"""按角色投递记忆；已回读原文不再重复传摘要、相似度和查询历史。"""
from copy import deepcopy

from ..agents.medqa_prompts import PLANNER, REASONER


def memory_view(role, hits, evidence_ids):
    if role == PLANNER:
        return deepcopy(hits)
    result = []
    for hit in hits:
        if hit.get("kind") in {"strategy","episode"}:
            if role != REASONER and hit["status"]=="experience_observation":
                result.append(deepcopy(hit))
            continue
        source_ready = hit["status"] == "source_checked_not_semantically_verified" and hit["source_id"] in evidence_ids
        if role == REASONER and not source_ready:
            # 未读摘要不能作为最终事实依据；原文仍可由调查角色按预算回读。
            continue
        if source_ready:
            result.append({key: deepcopy(hit[key]) for key in ("id", "source_id", "conditions", "status")})
        else:
            result.append(deepcopy(hit))
    return result
