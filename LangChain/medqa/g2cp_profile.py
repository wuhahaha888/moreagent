"""MedQA配置复用公共三层承载；只有Precise进入模型。"""
from copy import deepcopy
from ..protocol.g2cp import build_control, build_semantic_object
from ..agents.medqa_prompts import ROLES, PLANNER, INVESTIGATOR, REASONER, MEDQA_INPUT_FIELDS

PROFILE = {
    "capabilities": {role: {"actions": actions} for role, actions in zip(ROLES, (
        ["plan"], ["tool_plan", "search", "read_memory", "expand_evidence", "finish"],
        ["answer", "retrieve", "expand_evidence", "stop"]))},
    "routes": {PLANNER: [INVESTIGATOR], INVESTIGATOR: [REASONER],
               REASONER: [INVESTIGATOR, REASONER]},
}


def message(role, values, *, trace, operation_id, step, max_steps, semantic_refs=None):
    if role not in ROLES:
        raise ValueError("未知MedQA角色")
    control = build_control(trace_id=trace, src="medqa.scheduler", dst=role,
                            step=step, max_steps=max_steps, profile=PROFILE)
    control.update(operation_id=operation_id, profile="medqa-v6")
    precise = {key: deepcopy(values[key]) for key in MEDQA_INPUT_FIELDS[role] if key in values}
    return build_semantic_object(control=control,
        semantic={"references": deepcopy(semantic_refs or {})}, precise=precise)
