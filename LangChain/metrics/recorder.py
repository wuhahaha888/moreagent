"""三层指标统计：区分 Control / Semantic / Precise 层开销。

核心变化：
- Control 层：零 token，记录路由决策次数
- Semantic 层：零 token，记录向量传递次数和字节数
- Precise 层：记录 token 开销、字符开销
- 总计：合并三层指标
"""

import math
import time
from typing import Any, Literal


TransferMode = Literal["structured", "text"]


def estimate_token_count(text: str) -> int:
    """按字符数粗略估算 token。"""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _read_int(mapping: dict[str, Any], *names: str) -> int:
    """从不同模型 SDK 的 usage 字段名中读取整数值。"""
    for name in names:
        value = mapping.get(name)
        if isinstance(value, int):
            return value
    return 0


def normalize_token_usage(response: Any, prompt_text: str, completion_text: str) -> dict[str, Any]:
    """统一 LangChain/OpenAI usage 字段，缺失时退回字符估算。"""
    usage: dict[str, Any] = {}
    usage_metadata = getattr(response, "usage_metadata", None)
    response_metadata = getattr(response, "response_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}

    if isinstance(usage_metadata, dict):
        usage = {
            "prompt_tokens": _read_int(usage_metadata, "input_tokens", "prompt_tokens"),
            "completion_tokens": _read_int(usage_metadata, "output_tokens", "completion_tokens"),
            "total_tokens": _read_int(usage_metadata, "total_tokens"),
            "source": "model",
        }
    elif isinstance(token_usage, dict) and token_usage:
        usage = {
            "prompt_tokens": _read_int(token_usage, "prompt_tokens", "input_tokens"),
            "completion_tokens": _read_int(token_usage, "completion_tokens", "output_tokens"),
            "total_tokens": _read_int(token_usage, "total_tokens"),
            "source": "model",
        }

    if not usage:
        usage = {
            "prompt_tokens": estimate_token_count(prompt_text),
            "completion_tokens": estimate_token_count(completion_text),
            "total_tokens": 0,
            "source": "estimated",
        }

    if not usage.get("total_tokens"):
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    return usage


def merge_token_usage(*usages: dict[str, Any]) -> dict[str, Any]:
    """合并多次 LLM 调用的 token 使用量。"""
    merged = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "source": "none",
    }
    sources = set()
    for usage in usages:
        if not usage:
            continue
        merged["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        merged["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        merged["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        sources.add(str(usage.get("source", "unknown")))
    if sources:
        merged["source"] = "+".join(sorted(sources))
    return merged


def semantic_vectors_bytes(semantic: dict[str, Any]) -> int:
    """计算语义层向量占用的字节数（float64 = 8 bytes per dim）。"""
    total = 0
    for key in ("sentence_vec", "task_vec", "context_vec"):
        vec = semantic.get(key, [])
        total += len(vec) * 8
    return total


def new_metrics(mode: TransferMode) -> dict[str, Any]:
    """创建一次对话的指标容器，区分三层。"""
    return {
        "mode": mode,
        "started_at": time.time(),
        "turns": [],
    }


def record_agent_turn(
    metrics: dict[str, Any],
    *,
    agent: str,
    mode: TransferMode,
    request_payload: str,
    response_payload: str,
    token_usage: dict[str, Any],
    duration_seconds: float,
    memory_hits: int,
    non_text_state_transfers: int = 0,
    non_text_state_bytes: int = 0,
) -> dict[str, Any]:
    """记录单个 Agent 内部对话的三层指标。"""
    turn = {
        "agent": agent,
        "mode": mode,
        # Precise 层指标（唯一消耗 token 的层）
        "precise_chars": len(request_payload or "") + len(response_payload or ""),
        "precise_request_chars": len(request_payload or ""),
        "precise_response_chars": len(response_payload or ""),
        "token_usage": token_usage,
        # Semantic 层指标（零 token）
        "semantic_transfers": non_text_state_transfers,
        "semantic_bytes": non_text_state_bytes,
        # Control 层指标（零 token）
        "control_decisions": 1,  # 每轮至少1次路由决策
        # 通用指标
        "duration_seconds": round(duration_seconds, 4),
        "memory_hits": memory_hits,
    }
    metrics.setdefault("turns", []).append(turn)
    return turn


def finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """汇总一次对话的所有 Agent 指标，区分三层。"""
    turns = metrics.get("turns", [])
    token_total = merge_token_usage(*(turn.get("token_usage", {}) for turn in turns))
    memory_hits = sum(int(turn.get("memory_hits", 0) or 0) for turn in turns)
    message_count = len(turns)

    # 三层分别统计
    precise_chars = sum(int(turn.get("precise_chars", 0) or 0) for turn in turns)
    semantic_bytes = sum(int(turn.get("semantic_bytes", 0) or 0) for turn in turns)
    semantic_transfers = sum(int(turn.get("semantic_transfers", 0) or 0) for turn in turns)
    control_decisions = sum(int(turn.get("control_decisions", 0) or 0) for turn in turns)
    memory_events = metrics.get("memory_events", [])
    queries = sum(e.get("query_count", 0) for e in memory_events)
    hit_queries = sum(e.get("hit_query", 0) for e in memory_events)
    if not memory_events:
        queries = len(turns)
        hit_queries = sum(bool(t.get("memory_hits")) for t in turns)

    summary = {
        "mode": metrics.get("mode", "structured"),
        "message_count": message_count,
        # Precise 层
        "precise_chars": precise_chars,
        "token_usage": token_total,
        # Semantic 层
        "semantic_transfers": semantic_transfers,
        "semantic_bytes": semantic_bytes,
        # Control 层
        "control_decisions": control_decisions,
        # 通用
        "total_duration_seconds": round(
            time.time() - float(metrics.get("started_at", time.time())), 4
        ),
        "memory_hits": memory_hits,
        "memory_hit_rate": round(hit_queries / queries, 4) if queries else 0.0,
        "memory_query_count": queries,
        "memory_hit_queries": hit_queries,
        "cross_task_memory_hits": sum(e.get("cross_task_count", 0) for e in memory_events),
        # 自报引用有显式 ID 校验，但不等于经人工/标准答案核验的正确复用。
        "cross_task_memory_citations": sum(e.get("cross_task_cited_count", 0) for e in memory_events),
        "memory_context_chars": sum(e.get("context_chars", 0) for e in memory_events),
        "memory_state_transfers": sum(e.get("state_transfer_count", 0) for e in memory_events),
        "memory_state_bytes": sum(e.get("vector_bytes", 0) for e in memory_events),
        "memory_retrieval_seconds": sum(e.get("duration_seconds", 0) for e in memory_events),
        "memory_consolidation_seconds": metrics.get("memory_consolidation", {}).get("duration_seconds", 0),
        "memory_consolidation_usage_unknown": metrics.get("memory_consolidation", {}).get("usage_unknown", False),
        "embedding_cache_hits": metrics.get("embedding_cache_hits", 0),
        "embedding_encoded_texts": metrics.get("embedding_encoded_texts", 0),
    }
    metrics["summary"] = summary
    return metrics
