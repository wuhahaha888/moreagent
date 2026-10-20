"""Flask 后端 API：封装 LangChain 多 Agent 协作系统，供 Vue 前端消费。"""

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

# ── 将项目根目录加入模块搜索路径 ──
# 支持 from LangChain.xxx 形式的导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LangChain.framework.bootstrap import create_multi_agent_system
from LangChain.framework.config import load_settings
from LangChain.framework.json_utils import compact_json, pretty_json
from LangChain.memory.chat_log import JsonChatLogger
from LangChain.memory.sessions import ConversationSession, MemorySessionManager

from LangChain.comparison.comparator import build_comparison
from LangChain.comparison.text_runner import TextModeRunner, build_text_model

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── 全局状态（线程安全） ──
_lock = threading.Lock()
_settings = None
_system = None
_text_runner = None
_session_manager = None
_current_session: ConversationSession | None = None
_chat_logger: JsonChatLogger | None = None


def _ensure_initialized():
    """延迟初始化：首次请求时加载模型和记忆。"""
    global _settings, _system, _text_runner, _session_manager, _current_session, _chat_logger
    with _lock:
        if _settings is not None:
            return
        _settings = load_settings()
        _session_manager = MemorySessionManager(_settings.memory_dir, _settings.chat_dir)
        # 默认新建会话
        _current_session = _session_manager.new_session()
        _system, model_name = create_multi_agent_system(_current_session.memory_path)
        _chat_logger = JsonChatLogger(_current_session.chat_path)
        _text_runner = TextModeRunner(build_text_model(_settings))
        print(f"[Flask] 系统初始化完成，模型: {model_name}")


def _serialize_state(final_state: dict[str, Any]) -> dict[str, Any]:
    """将 AgentState 转换为 JSON 可序列化的结构，供前端消费。"""
    control_stack = final_state.get("control_stack", [])
    semantic_stack = final_state.get("semantic_stack", [])
    precise_stack = final_state.get("precise_stack", [])
    metrics = final_state.get("metrics", {})
    summary = metrics.get("summary", {})
    turns = metrics.get("turns", [])
    token_usage = summary.get("token_usage", {})

    # 构建每个 Agent 的三层协议信息 + 每步性能指标
    agent_steps = []
    for i, precise in enumerate(precise_stack):
        control = control_stack[i] if i < len(control_stack) else {}
        semantic = semantic_stack[i] if i < len(semantic_stack) else {}
        # 从 turns 中提取该步骤的指标
        turn_metrics = turns[i] if i < len(turns) else {}
        turn_tokens = turn_metrics.get("token_usage", {})
        agent_steps.append({
            "index": i + 1,
            "agent": control.get("dst", ""),
            "action": control.get("action", ""),
            "step": control.get("step", 0),
            "control": {
                "src": control.get("src", ""),
                "dst": control.get("dst", ""),
                "kind": control.get("kind", ""),
                "action": control.get("action", ""),
                "priority": control.get("priority", 1),
                "step": control.get("step", 0),
                "max_steps": control.get("max_steps", 6),
            },
            "semantic": {
                "sentence_vec_dim": len(semantic.get("sentence_vec", [])),
                "task_vec_dim": len(semantic.get("task_vec", [])),
                "context_vec_dim": len(semantic.get("context_vec", [])),
                "model": semantic.get("model", ""),
            },
            "precise": _json_safe(precise),
            # 每步性能指标
            "step_metrics": {
                "prompt_tokens": turn_tokens.get("prompt_tokens", 0),
                "completion_tokens": turn_tokens.get("completion_tokens", 0),
                "total_tokens": turn_tokens.get("total_tokens", 0),
                "precise_chars": turn_metrics.get("precise_chars", 0),
                "duration_seconds": turn_metrics.get("duration_seconds", 0),
                "memory_hits": turn_metrics.get("memory_hits", 0),
                "semantic_transfers": turn_metrics.get("semantic_transfers", 0),
                "semantic_bytes": turn_metrics.get("semantic_bytes", 0),
                "control_decisions": turn_metrics.get("control_decisions", 0),
            },
        })

    # 展平指标供前端直接使用
    return {
        "trace_id": final_state.get("trace_id", ""),
        "mode": final_state.get("mode", "structured"),
        "user_question": final_state.get("user_question", ""),
        "final_answer": final_state.get("final_answer", ""),
        "agent_steps": agent_steps,
        "metrics": {
            # 前端直接使用的展平字段
            "total_tokens": token_usage.get("total_tokens", 0),
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "duration_seconds": summary.get("total_duration_seconds", 0),
            "total_chars": summary.get("precise_chars", 0),
            "total_messages": summary.get("message_count", 0),
            "memory_hits": summary.get("memory_hits", 0),
            "memory_hit_rate": summary.get("memory_hit_rate", 0),
            # 三层协议指标
            "protocol_metrics": {
                "control_tokens": 0,  # Control 层零 token
                "control_decisions": summary.get("control_decisions", 0),
                "semantic_tokens": 0,  # Semantic 层零 token
                "semantic_transfers": summary.get("semantic_transfers", 0),
                "semantic_bytes": summary.get("semantic_bytes", 0),
                "precise_tokens": token_usage.get("total_tokens", 0),
                "precise_chars": summary.get("precise_chars", 0),
            },
            # 每步 Agent 指标
            "agent_metrics": [
                {
                    "name": t.get("agent", ""),
                    "tokens": t.get("token_usage", {}).get("total_tokens", 0),
                    "prompt_tokens": t.get("token_usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": t.get("token_usage", {}).get("completion_tokens", 0),
                    "duration": t.get("duration_seconds", 0),
                    "precise_chars": t.get("precise_chars", 0),
                    "memory_hits": t.get("memory_hits", 0),
                    "semantic_bytes": t.get("semantic_bytes", 0),
                }
                for t in turns
            ],
            # 原始汇总（供高级分析）
            "summary": _json_safe(summary),
        },
        "memory_hits": _json_safe(final_state.get("memory_hits", [])),
        "memory_events": _json_safe(metrics.get("memory_events", [])),
        "memory_consolidation": _json_safe(metrics.get("memory_consolidation", {})),
    }


def _json_safe(obj: Any) -> Any:
    """递归转换不可序列化的对象为 JSON 安全格式。"""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


# ══════════════════════════════════════════════════════════════════════
# API 路由
# ══════════════════════════════════════════════════════════════════════


@app.route("/api/health", methods=["GET"])
def health():
    """健康检查。"""
    return jsonify({"status": "ok", "initialized": _settings is not None})


@app.route("/api/chat", methods=["POST"])
def chat():
    """运行结构化模式多 Agent 协作。"""
    _ensure_initialized()
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    try:
        final_state = _system.run(question, mode="structured")
        _chat_logger.append_turn(final_state)
        result = _serialize_state(final_state)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _serialize_text_result(text_result: dict[str, Any]) -> dict[str, Any]:
    """将纯文本模式结果展平为前端友好的结构。"""
    metrics = text_result.get("metrics", {})
    summary = metrics.get("summary", {})
    turns = metrics.get("turns", [])
    token_usage = summary.get("token_usage", {})

    return {
        "final_answer": text_result.get("final_answer", ""),
        "agent_outputs": _json_safe(text_result.get("agent_outputs", [])),
        "metrics": {
            "total_tokens": token_usage.get("total_tokens", 0),
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "duration_seconds": summary.get("total_duration_seconds", 0),
            "total_chars": summary.get("total_chars", 0),
            "total_messages": summary.get("message_count", 0),
            "agent_metrics": [
                {
                    "name": t.get("agent", ""),
                    "tokens": t.get("token_usage", {}).get("total_tokens", 0),
                    "prompt_tokens": t.get("token_usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": t.get("token_usage", {}).get("completion_tokens", 0),
                    "duration": t.get("duration_seconds", 0),
                    "total_chars": t.get("total_chars", 0),
                }
                for t in turns
            ],
            "summary": _json_safe(summary),
        },
    }


@app.route("/api/chat/compare", methods=["POST"])
def chat_compare():
    """对比模式：结构化 vs 纯文本。"""
    _ensure_initialized()
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    try:
        # 第1步：结构化模式
        final_state = _system.run(question, mode="structured")
        _chat_logger.append_turn(final_state)
        structured_result = _serialize_state(final_state)

        # 第2步：纯文本模式
        text_result = _text_runner.run(question)
        text_mode_result = _serialize_text_result(text_result)

        # 第3步：构建对比
        structured_for_comp = {
            "final_answer": final_state.get("final_answer", ""),
            "metrics": final_state.get("metrics", {}),
            "agent_count": len(final_state.get("precise_stack", [])),
        }
        comparison = build_comparison(structured_for_comp, text_result)

        return jsonify({
            "structured": structured_result,
            "text_mode": text_mode_result,
            "comparison": _json_safe(comparison),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """列出历史会话。"""
    _ensure_initialized()
    sessions = _session_manager.list_sessions()
    result = []
    for s in sessions:
        result.append({
            "index": s.index,
            "timestamp": s.timestamp,
            "name": s.name,
            "memory_path": str(s.memory_path),
            "chat_path": str(s.chat_path),
            "updated_at": s.updated_at,
        })
    return jsonify(result)


@app.route("/api/sessions", methods=["POST"])
def create_session():
    """新建会话。"""
    _ensure_initialized()
    global _current_session, _chat_logger, _system
    with _lock:
        session = _session_manager.new_session()
        _current_session = session
        _chat_logger = JsonChatLogger(session.chat_path)
        # 重新创建系统（新记忆文件）
        _system, _ = create_multi_agent_system(session.memory_path)
    return jsonify({
        "index": session.index,
        "timestamp": session.timestamp,
        "name": session.name,
        "memory_path": str(session.memory_path),
        "chat_path": str(session.chat_path),
    })


@app.route("/api/sessions/<int:session_index>/resume", methods=["POST"])
def resume_session(session_index):
    """恢复历史会话。"""
    _ensure_initialized()
    global _current_session, _chat_logger, _system
    sessions = _session_manager.list_sessions()
    target = None
    for s in sessions:
        if s.index == session_index:
            target = s
            break
    if not target:
        return jsonify({"error": f"会话 {session_index} 不存在"}), 404
    with _lock:
        _current_session = target
        _chat_logger = JsonChatLogger(target.chat_path)
        _system, _ = create_multi_agent_system(target.memory_path)
    return jsonify({
        "index": target.index,
        "timestamp": target.timestamp,
        "name": target.name,
    })


@app.route("/api/memories", methods=["GET"])
def list_memories():
    """查询当前会话的记忆摘要。"""
    _ensure_initialized()
    summaries = _system.memory.summaries()
    return jsonify(_json_safe(summaries))


@app.route("/api/chats", methods=["GET"])
def get_chat_log():
    """获取当前会话的对话记录。"""
    _ensure_initialized()
    if not _chat_logger or not _chat_logger.path.exists():
        return jsonify([])
    try:
        data = json.loads(_chat_logger.path.read_text(encoding="utf-8"))
        records = data.get("records", []) if isinstance(data, dict) else data
        return jsonify(_json_safe(records))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    """获取当前系统状态。"""
    _ensure_initialized()
    session_info = {}
    if _current_session:
        session_info = {
            "timestamp": _current_session.timestamp,
            "memory_path": str(_current_session.memory_path),
            "chat_path": str(_current_session.chat_path),
        }
    return jsonify({
        "initialized": True,
        "model": _settings.deepseek_model if _settings else "",
        "session": session_info,
        "memory_count": len(_system.memory.items) if _system else 0,
    })


if __name__ == "__main__":
    print("[Flask] StruckFlow API Server 启动中...")
    _ensure_initialized()
    app.run(host="0.0.0.0", port=5000, debug=True)
