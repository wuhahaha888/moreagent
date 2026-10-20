"""SemanticObject 三层协议：Control + Semantic + Precise

- ControlInfo：系统路由/调度/决策，零 token
- SemanticVectors：向量检索/路由/缓存，零 token
- PreciseStructure：LLM 执行用最小原子字段，唯一消耗 token 的层
"""

import time
from typing import Any, Literal, TypedDict

# ─── Agent 身份与能力 ───────────────────────────────────────────────

AgentName = Literal["任务分配", "资料搜索", "任务执行", "总结验证", "记忆管理"]
MessageKind = Literal["request", "result", "broadcast", "error"]
Priority = Literal[0, 1, 2, 3]  # 0=低 1=普通 2=高 3=紧急

AGENT_CAPABILITIES: dict[AgentName, dict[str, Any]] = {
    "记忆管理": {
        "actions": ["retrieve", "read", "consolidate"],
        "inputs": ["task_vec", "context_vec", "role", "task_outputs"],
        "outputs": ["memory_hits", "memory_ids", "source_ids"],
    },
    "任务分配": {
        "actions": ["plan"],
        "inputs": ["question", "memory_hits"],
        "outputs": ["goal", "tasks", "criteria"],
    },
    "资料搜索": {
        "actions": ["research"],
        "inputs": ["question", "goal", "tasks", "tool_results", "memory_hits"],
        "outputs": ["findings", "assumptions", "gaps"],
    },
    "任务执行": {
        "actions": ["execute"],
        "inputs": ["question", "goal", "findings", "gaps", "memory_hits"],
        "outputs": ["draft", "steps", "risks"],
    },
    "总结验证": {
        "actions": ["verify"],
        "inputs": ["question", "goal", "draft", "criteria", "memory_hits"],
        "outputs": ["checks", "fixes", "final_answer"],
    },
}

# 每个 Agent 允许的下一跳（系统层路由约束，不交给 LLM 决策）
ALLOWED_NEXT: dict[AgentName, list[str]] = {
    "任务分配": ["资料搜索", "任务执行", "总结验证"],
    "资料搜索": ["任务执行", "总结验证"],
    "任务执行": ["资料搜索", "总结验证"],
    "总结验证": [],
}

# 每个 Agent 需要的 Precise 输入字段（最小集）
AGENT_PRECISE_INPUTS: dict[AgentName, list[str]] = {
    "任务分配": ["question", "memory_hits"],
    "资料搜索": ["question", "goal", "tasks", "tool_results", "memory_hits"],
    "任务执行": ["question", "goal", "tasks", "findings", "gaps", "memory_hits"],
    "总结验证": ["question", "goal", "draft", "criteria", "memory_hits"],
}


# ─── 第1层：ControlInfo（控制层）────────────────────────────────────

class ControlInfo(TypedDict, total=False):
    """控制层：给系统看，零 token 开销。

    消费者：LangGraph 路由器 / 编排器 / 调度器
    不序列化给 LLM。
    """
    # 身份与追踪
    trace_id: str
    timestamp: float
    # 路由
    src: str                   # 发送者
    dst: str                   # 接收者
    kind: MessageKind          # request | result | broadcast | error
    action: str                # plan | research | execute | verify
    # 调度决策
    priority: Priority         # 0=低 1=普通 2=高 3=紧急
    cacheable: bool            # 结果是否可缓存复用
    approximable: bool         # 是否允许近似处理
    # 能力发现
    src_caps: dict[str, Any]   # 发送者能力
    dst_caps: dict[str, Any]   # 接收者能力
    all_caps: dict[str, Any]   # 全局能力表
    # 路由约束
    allowed_next: list[str]    # 允许的下一跳
    step: int                  # 当前步数
    max_steps: int             # 最大步数
    # 统计指标
    usage: dict[str, Any]      # token/字符/耗时
    operation_id: str
    profile: str


# ─── 第2层：SemanticVectors（语义层）────────────────────────────────

class SemanticVectors(TypedDict, total=False):
    """语义层：给向量系统看，零 token 开销，固定维度。

    消费者：向量检索 / Agent 路由 / 缓存匹配 / 记忆召回
    不序列化给 LLM。
    """
    # 多粒度嵌入
    sentence_vec: list[float]  # 句子级：当前输出的语义位置
    task_vec: list[float]      # 任务级：整体任务目标的语义位置
    context_vec: list[float]   # 上下文级：累积上下文的语义位置
    # 向量元数据
    model: str                 # embedding 模型标识
    dim: int                   # 向量维度
    created_at: float          # 向量生成时间
    references: dict[str, Any]


# ─── 第3层：PreciseStructure（精确执行层）────────────────────────────

class PreciseStructure(TypedDict, total=False):
    """精确执行层：给 LLM 看，唯一消耗 token 的层。

    消费者：LLM
    最小原子字段，零歧义，机器可读。
    """
    # ── 通用输入 ──
    question: str              # 用户原始问题
    memory_hits: list[dict]    # [{id, role, summary, score}] 记忆命中摘要

    # ── 任务分配 输出 ──
    goal: str                  # 一句话目标
    tasks: list[str]           # 任务列表
    criteria: list[str]        # 验收标准

    # ── 资料搜索 输出 ──
    findings: list[dict]       # [{claim, url}] 结构化发现
    assumptions: list[str]     # 假设
    gaps: list[str]            # 缺失信息

    # ── 任务执行 输出 ──
    draft: str                 # 答案草稿
    steps: list[str]           # 执行步骤
    risks: list[str]           # 风险

    # ── 总结验证 输出 ──
    checks: list[str]          # 检查点
    fixes: list[str]           # 修正点
    final_answer: str          # 最终答案

    # ── 工具结果（精确字段） ──
    tool_results: list[dict]   # [{title, url, snippet}]

    # ── LLM 意图提示（轻量，非路由命令） ──
    need: str                  # "research" | "execute" | "verify" | "end"
    # 业务字段由配置白名单选择；公共层不解释医学支持或预算规则。
    question_id: str
    options: dict[str, str]
    evidence_pool: list[dict]
    budget: dict[str, Any]


# ─── 复合结构：SemanticObject ────────────────────────────────────────

class SemanticObject(TypedDict, total=False):
    """SemanticObject = ControlInfo + SemanticVectors + PreciseStructure

    三层物理分离，各自有独立的消费者和生命周期。
    """
    control: ControlInfo
    semantic: SemanticVectors
    precise: PreciseStructure


# ─── 构建函数 ────────────────────────────────────────────────────────

def build_control(
    *,
    trace_id: str,
    src: str,
    dst: AgentName,
    kind: MessageKind = "request",
    action: str = "",
    priority: Priority = 1,
    cacheable: bool = False,
    approximable: bool = False,
    step: int = 0,
    max_steps: int = 6,
    usage: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
) -> ControlInfo:
    """构建控制层对象。"""
    capabilities = profile["capabilities"] if profile else AGENT_CAPABILITIES
    routes = profile["routes"] if profile else ALLOWED_NEXT
    return {
        "trace_id": trace_id,
        "timestamp": time.time(),
        "src": src,
        "dst": dst,
        "kind": kind,
        "action": action or capabilities.get(dst, {}).get("actions", [""])[0],
        "priority": priority,
        "cacheable": cacheable,
        "approximable": approximable,
        "src_caps": capabilities.get(src, {}),
        "dst_caps": capabilities.get(dst, {}),
        "all_caps": capabilities,
        "allowed_next": routes.get(dst, []),
        "step": step,
        "max_steps": max_steps,
        "usage": usage or {},
    }


def build_semantic(
    *,
    sentence_vec: list[float] | None = None,
    task_vec: list[float] | None = None,
    context_vec: list[float] | None = None,
    model: str = "",
    dim: int = 0,
) -> SemanticVectors:
    """构建语义层对象。"""
    return {
        "sentence_vec": sentence_vec or [],
        "task_vec": task_vec or [],
        "context_vec": context_vec or [],
        "model": model,
        "dim": dim,
        "created_at": time.time(),
    }


def build_precise_for_agent(
    agent: AgentName,
    *,
    question: str = "",
    memory_hits: list[dict] | None = None,
    # 上游输出（按需传入）
    goal: str = "",
    tasks: list[str] | None = None,
    criteria: list[str] | None = None,
    findings: list[dict] | None = None,
    assumptions: list[str] | None = None,
    gaps: list[str] | None = None,
    draft: str = "",
    steps: list[str] | None = None,
    risks: list[str] | None = None,
    checks: list[str] | None = None,
    fixes: list[str] | None = None,
    final_answer: str = "",
    tool_results: list[dict] | None = None,
) -> PreciseStructure:
    """按 Agent 角色构建最小 Precise 输入，只包含该 Agent 需要的字段。"""
    required = set(AGENT_PRECISE_INPUTS.get(agent, []))
    precise: PreciseStructure = {}

    # 通用字段
    if "question" in required and question:
        precise["question"] = question
    if "memory_hits" in required and memory_hits:
        precise["memory_hits"] = memory_hits

    # 任务分配输出
    if "goal" in required and goal:
        precise["goal"] = goal
    if "tasks" in required and tasks:
        precise["tasks"] = tasks
    if "criteria" in required and criteria:
        precise["criteria"] = criteria

    # 资料搜索输出
    if "findings" in required and findings:
        precise["findings"] = findings
    if "assumptions" in required and assumptions:
        precise["assumptions"] = assumptions
    if "gaps" in required and gaps:
        precise["gaps"] = gaps
    if "tool_results" in required and tool_results:
        precise["tool_results"] = tool_results

    # 任务执行输出
    if "draft" in required and draft:
        precise["draft"] = draft
    if "steps" in required and steps:
        precise["steps"] = steps
    if "risks" in required and risks:
        precise["risks"] = risks

    # 总结验证输出
    if "checks" in required and checks:
        precise["checks"] = checks
    if "fixes" in required and fixes:
        precise["fixes"] = fixes
    if "final_answer" in required and final_answer:
        precise["final_answer"] = final_answer

    return precise


def build_semantic_object(
    *,
    control: ControlInfo,
    semantic: SemanticVectors,
    precise: PreciseStructure,
) -> SemanticObject:
    """构建完整的 SemanticObject。"""
    return {
        "control": control,
        "semantic": semantic,
        "precise": precise,
    }


# ─── 路由决策（系统层，不交给 LLM）──────────────────────────────────

def decide_next_agent(
    current_agent: AgentName,
    precise_output: PreciseStructure,
    control: ControlInfo,
) -> str:
    """根据 Control 层约束和 Precise 层意图提示决定下一跳。

    LLM 可以在 precise.need 中给出轻量提示，但最终路由由系统决定。
    """
    allowed = set(control.get("allowed_next", ALLOWED_NEXT.get(current_agent, [])))

    # 步数超限，强制结束
    if control.get("step", 0) >= control.get("max_steps", 6):
        return "END"

    # LLM 意图提示映射
    need = str(precise_output.get("need", "")).strip().lower()
    need_map = {
        "research": "资料搜索", "search": "资料搜索", "资料搜索": "资料搜索",
        "execute": "任务执行", "executor": "任务执行", "任务执行": "任务执行",
        "verify": "总结验证", "verification": "总结验证", "总结验证": "总结验证",
        "end": "END", "done": "END", "完成": "END",
    }
    hinted = need_map.get(need, "")

    # 提示在允许范围内则采纳
    if hinted and hinted in allowed:
        return hinted

    # 总结验证后必定结束
    if current_agent == "总结验证":
        return "END"

    # 默认路由
    defaults = {
        "任务分配": "资料搜索",
        "资料搜索": "任务执行",
        "任务执行": "总结验证",
    }
    default_next = defaults.get(current_agent, "END")
    return default_next if default_next in allowed else "END"


# ─── 向量规模计算 ────────────────────────────────────────────────────

def semantic_vectors_bytes(semantic: SemanticVectors) -> int:
    """计算语义层向量占用的字节数（float64 = 8 bytes per dim）。"""
    total = 0
    for key in ("sentence_vec", "task_vec", "context_vec"):
        vec = semantic.get(key, [])
        total += len(vec) * 8
    return total


def agent_action(agent: AgentName) -> str:
    """返回指定 Agent 的默认动作类型。"""
    return AGENT_CAPABILITIES.get(agent, {}).get("actions", [""])[0]
