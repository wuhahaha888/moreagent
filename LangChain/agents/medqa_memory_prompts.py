"""Needs-aware memory prompts; same three model calls, bounded previews."""
from .medqa_prompts import PLANNER,INVESTIGATOR,REASONER
from .medqa_baseline_prompts import BASELINE_ROLE_PROMPTS,COMMON

MEMORY_ROLE_PROMPTS=dict(BASELINE_ROLE_PROMPTS)
MEMORY_ROLE_PROMPTS[PLANNER]=COMMON+"""你识别当前需求，并判断程序预筛的最多3条记忆是否值得回读；未显示的粗召回候选不需要你处理。
输出key_conditions:[{id,text}]（1至12项题干关键条件），knowledge_tasks:[{id,goal,condition_ids,depends_on:[]}]（1至3项待核实需求），criteria:[简短目标]，memory_assessments:[{memory_id,knowledge_task_ids}]。
memory_assessments只列确有当前用途、需要回读的ID及对应需求；不采用就省略，全不采用则[]。不输出接受/拒绝理由、decision、历史摘要或答案。不要为凑数量选满。
evidence待核对来源和适用性；strategy可指导查询，episode只提供历史任务线索，后两者不能证明医学事实。origin_conditions是旧任务条件，不保证适用于本题。程序预筛不是医学验证，不得因相似度或摘要认定证据充分。
保留原题否定、人群、时间、数值与限制；推测诊断和既有医学知识只能作为待核实需求，不能冒充题干事实。知识关联不构成必须等待的执行前置条件。
"""
MEMORY_ROLE_PROMPTS[INVESTIGATOR]=BASELINE_ROLE_PROMPTS[INVESTIGATOR]+"""
【实际策略复用记录】仅当某条已回读strategy确实影响某个查询时，在该查询对象添加strategy_memory_ids:[实际记忆ID]。可以调整旧查询以匹配当前问题，不要求原样照抄。不使用则省略或[]。程序会核对ID是否为你实际收到的策略，并记录查询是否真的执行；不得为了提高统计数字虚报。episode只提供历史线索，不作为医学事实。"""
