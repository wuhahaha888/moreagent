"""Agent 状态结构：适配 SemanticObject 三层协议。

State 在 LangGraph 节点间传递，承载三层分离的协议对象。
Control/Semantic 层不序列化给 LLM，只有 Precise 层消耗 token。
"""

from typing import Any, Literal, TypedDict

from ..protocol.g2cp import (
    AgentName,
    ControlInfo,
    PreciseStructure,
    SemanticVectors,
)


class AgentState(TypedDict, total=False):
    """LangGraph 在各 Agent 节点间传递的共享状态结构。

    三层分离：
    - control_stack: 每轮的 ControlInfo，系统消费，零 token
    - semantic_stack: 每轮的 SemanticVectors，向量系统消费，零 token
    - precise_stack: 每轮的 PreciseStructure，LLM 消耗 token
    """

    # ── 追踪与模式 ──
    trace_id: str
    mode: Literal["structured", "text"]

    # ── 用户输入 ──
    user_question: str
    memory_context: dict[str, Any]

    # ── 三层协议栈（每轮一个元素）──
    control_stack: list[ControlInfo]
    semantic_stack: list[SemanticVectors]
    precise_stack: list[PreciseStructure]

    # ── 当前语义向量（最新，用于记忆检索）──
    task_vec: list[float]           # 任务级向量（用户问题，不变）
    context_vec: list[float]        # 上下文级向量（累积更新）

    # ── 各 Agent 输出的 Precise 字段（供下游按需读取）──
    assignment: PreciseStructure    # 任务分配输出
    research: PreciseStructure     # 资料搜索输出
    execution: PreciseStructure    # 任务执行输出
    verification: PreciseStructure # 总结验证输出

    # ── 记忆命中（由语义层检索，传给 Precise 层）──
    memory_hits: list[dict]

    # ── 路由与步数 ──
    next_agent: str
    steps: int
    max_steps: int

    # ── 最终输出 ──
    final_answer: str

    # ── 指标 ──
    metrics: dict[str, Any]
