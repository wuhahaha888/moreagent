"""程序校验引用与审核完整性；原文匹配不等于医学语义为真。"""
import re


class ContractError(ValueError):
    def __init__(self, message, *, category="agent_output"):
        super().__init__(message)
        self.category = category


def validate_plan(output, memories, *, sparse_memories=False):
    conditions, tasks = output.get("key_conditions"), output.get("knowledge_tasks")
    if not isinstance(conditions, list) or not 1 <= len(conditions) <= 12:
        raise ContractError("规划需要1至12项题干关键条件")
    condition_ids = set()
    for condition in conditions:
        if (not isinstance(condition, dict) or not isinstance(condition.get("id"), str)
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", condition["id"])
                or condition["id"] in condition_ids or not isinstance(condition.get("text"), str)
                or not condition["text"].strip()):
            raise ContractError("题干条件ID或内容无效")
        condition_ids.add(condition["id"])
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 3:
        raise ContractError("规划需要1至3个知识需求")
    done, covered = set(), set()
    for task in tasks:
        if (not isinstance(task, dict) or not isinstance(task.get("id"), str)
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", task["id"])
                or task["id"] in done or not isinstance(task.get("goal"), str) or not task["goal"].strip()):
            raise ContractError("知识需求ID或目标无效")
        identifiers(task.get("depends_on", []), done, empty=True)
        covered |= identifiers(task.get("condition_ids", []), condition_ids, empty=True)
        done.add(task["id"])
    criteria = output.get("criteria")
    if not isinstance(criteria, list) or not criteria or any(not isinstance(c, str) or not c.strip() for c in criteria):
        raise ContractError("验收条件不能为空")
    assessments = output.get("memory_assessments")
    if not isinstance(assessments, list):
        raise ContractError("需要显式评估候选记忆")
    seen = set()
    for assessment in assessments:
        if (not isinstance(assessment, dict) or assessment.get("memory_id") not in memories
                or assessment["memory_id"] in seen or assessment.get("decision") not in {"read", "reject"}
                or not isinstance(assessment.get("reason"), str) or not assessment["reason"].strip()):
            raise ContractError("仅有记忆摘要时须明确read或reject，不能宣称直接可用")
        seen.add(assessment["memory_id"])
        identifiers(assessment.get("knowledge_task_ids"), done, empty=assessment["decision"] == "reject")
    if not sparse_memories and seen != set(memories):
        raise ContractError("Planner须评估提供的每条候选记忆")
    # 规划是需求假设；未映射条件交给最终核验，不能把规划当作证据充分性证明。
    return sorted(condition_ids - covered)




def identifiers(values, available, *, empty=False):
    if not isinstance(values, list) or (not values and not empty):
        raise ContractError("需要来源/任务 ID 列表")
    if any(not isinstance(i, str) or i not in available for i in values):
        raise ContractError("出现未提供的来源/任务 ID")
    if len(set(values)) != len(values):
        raise ContractError("ID 列表不能重复")
    return set(values)
