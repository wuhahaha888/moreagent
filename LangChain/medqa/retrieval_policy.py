"""程序只过滤可执行性；预期医学价值及顺序来自已有Agent输出。"""
import re


def query_key(query):
    return re.sub(r"\s+", " ", query).strip().casefold()


def choose(item, *, history, tasks, available_inputs, remaining, stalled, limits, reopen=False):
    if remaining <= 0:
        return "stop_resource", "search_fuse"
    if stalled >= limits.max_stalled_rounds:
        return "stop_no_progress", "consecutive_no_progress"
    if not set(item.get("required_inputs", [])) <= set(available_inputs):
        return "need_input", "missing_required_input"
    ids = item["knowledge_task_ids"]
    if not reopen and all(tasks[i]["support"] == "supported" for i in ids):
        return "finalize", "targets_already_reviewed_supported"
    previous = [h for h in history if query_key(h["query"]) == query_key(item["query"])]
    if previous:
        if any(not h.get("error") for h in previous):
            return "reuse", "completed_query"
        if len(previous) > limits.max_tool_retries:
            return "revise_plan", "tool_retry_fuse"
    return "execute", "new_executable_plan" if not previous else "retry_tool_error"
