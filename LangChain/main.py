import sys
from pathlib import Path
from typing import Any


_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from LangChain.framework.bootstrap import create_multi_agent_system
from LangChain.framework.config import load_settings
from LangChain.framework.json_utils import pretty_json
from LangChain.memory.chat_log import JsonChatLogger
from LangChain.memory.sessions import ConversationSession, MemorySessionManager

from LangChain.comparison.comparator import build_comparison, print_comparison_report, print_text_result
from LangChain.comparison.text_runner import TextModeRunner, build_text_model


def print_json(title: str, data: Any) -> None:
    """以统一标题块打印结构化数据。"""
    print(f"\n========== {title} ==========")
    print(pretty_json(data))


def choose_memory_session(manager: MemorySessionManager) -> tuple[str, ConversationSession]:
    """选择新建或导入历史记忆会话。"""
    sessions = manager.list_sessions()
    print("\n========== 记忆会话 ==========")
    print("0. 开启新对话")
    for session in sessions:
        print(f"{session.index}. 导入 {session.name}")

    while True:
        try:
            choice = input("请选择记忆会话编号> ").strip()
        except EOFError:
            choice = "0"
        if choice == "" or choice == "0":
            return "new", manager.new_session()
        if choice.isdigit():
            selected = find_session(sessions, int(choice))
            if selected:
                return "resume", selected
        print("无效选择，请重新输入。")


def find_session(sessions: list[ConversationSession], index: int) -> ConversationSession | None:
    """按编号查找历史会话。"""
    for session in sessions:
        if session.index == index:
            return session
    return None





def print_metrics(metrics: dict[str, Any]) -> None:
    """打印三轮指标，区分 Control / Semantic / Precise 层。"""
    summary = metrics.get("summary", {})
    token_usage = summary.get("token_usage", {})
    print("\n========== 本轮指标 ==========")
    print(f"模式: SemanticObject 三层协议")
    print(f"消息次数: {summary.get('message_count', 0)}")

    # Precise 层（唯一消耗 token 的层）
    print("\n── Precise 层（消耗 token）──")
    print(f"字符开销: {summary.get('precise_chars', 0)}")
    print(
        "Token 开销: "
        f"prompt={token_usage.get('prompt_tokens', 0)}, "
        f"completion={token_usage.get('completion_tokens', 0)}, "
        f"total={token_usage.get('total_tokens', 0)}, "
        f"source={token_usage.get('source', 'none')}"
    )

    # Semantic 层（零 token）
    print("\n── Semantic 层（零 token）──")
    print(f"向量传递次数: {summary.get('semantic_transfers', 0)}")
    print(f"向量数据规模: {summary.get('semantic_bytes', 0)} bytes")

    # Control 层（零 token）
    print("\n── Control 层（零 token）──")
    print(f"路由决策次数: {summary.get('control_decisions', 0)}")

    # 通用
    print(f"\n总耗时: {summary.get('total_duration_seconds', 0)} 秒")
    print(
        f"共享记忆命中: {summary.get('memory_hits', 0)} "
        f"(命中率 {summary.get('memory_hit_rate', 0)})"
    )


def print_semantic_object(title: str, so: dict[str, Any]) -> None:
    """打印 SemanticObject 三层结构，Control/Semantic 层标注零 token。"""
    print(f"\n========== {title} ==========")
    control = so.get("control", {})
    semantic = so.get("semantic", {})
    precise = so.get("precise", {})
    print("── Control 层（零 token）──")
    print(f"  src={control.get('src')} → dst={control.get('dst')}, "
          f"action={control.get('action')}, step={control.get('step')}")
    print("── Semantic 层（零 token）──")
    svec = semantic.get("sentence_vec", [])
    tvec = semantic.get("task_vec", [])
    print(f"  sentence_vec={len(svec)}维, task_vec={len(tvec)}维, "
          f"model={semantic.get('model', '')}")
    print("── Precise 层（消耗 token）──")
    print(pretty_json(precise))


def legacy_main() -> None:
    """旧通用四角色交互CLI；不作为MedQA三Agent实验入口。"""
    settings = load_settings()
    session_mode, session = choose_memory_session(
        MemorySessionManager(settings.memory_dir, settings.chat_dir)
    )
    system, model_name = create_multi_agent_system(session.memory_path)
    chat_logger = JsonChatLogger(session.chat_path)

    # 创建纯文本模式运行器（自建无 JSON 约束的 LLM，独立运行）
    text_runner = TextModeRunner(build_text_model(settings))

    print("LangChain + LangGraph 多 Agent 协作 CLI（SemanticObject 三层协议）")
    print(f"模型: {model_name}")
    print(f"记忆文件: {session.memory_path}")
    print(f"对话日志: {session.chat_path}")
    print(f"记忆模式: {'新对话' if session_mode == 'new' else '导入历史'}")
    print("协议: SemanticObject v2.0 (Control + Semantic + Precise)")
    print("输入问题后回车执行；输入 exit / quit / q 退出。")
    print("问题前加 + 号开启对比模式")

    while True:
        try:
            question = input("\n用户问题> ").strip()
        except EOFError:
            print("\n已退出。")
            break
        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break
        if not question:
            continue

        # 检查是否开启对比模式
        compare_mode = question.startswith("+")
        if compare_mode:
            question = question[1:].strip()
            if not question:
                continue

        try:
            if compare_mode:
                # ── 对比模式：先结构化（正常流程+写日志），再纯文本（仅对比不存储）──
                print("\n[对比模式] 第1步：运行结构化协议模式...")
                final_state = system.run(question, mode="structured")
                chat_logger.append_turn(final_state)

                # 打印结构化模式结果
                for index, so in enumerate(final_state.get("precise_stack", []), start=1):
                    control_stack = final_state.get("control_stack", [])
                    semantic_stack = final_state.get("semantic_stack", [])
                    combined = {
                        "control": control_stack[index - 1] if index <= len(control_stack) else {},
                        "semantic": semantic_stack[index - 1] if index <= len(semantic_stack) else {},
                        "precise": so,
                    }
                    print_semantic_object(f"角色 {index} / SemanticObject", combined)

                print("\n========== 结构化模式最终输出 ==========")
                print(final_state.get("final_answer") or "没有生成 final_answer。")
                print_metrics(final_state.get("metrics", {}))

                # 运行纯文本模式（不写日志、不写记忆）
                print("\n[对比模式] 第2步：运行纯文本协作模式...")
                text_result = text_runner.run(question)
                print_text_result(text_result)

                # 生成并打印对比报告
                structured_result = {
                    "final_answer": final_state.get("final_answer", ""),
                    "metrics": final_state.get("metrics", {}),
                    "agent_count": len(final_state.get("precise_stack", [])),
                }
                comparison_result = {
                    "structured_result": structured_result,
                    "text_result": text_result,
                    "comparison": build_comparison(structured_result, text_result),
                }
                print_comparison_report(comparison_result)

            else:
                # ── 正常模式：只运行结构化协议 ──
                final_state = system.run(question, mode="structured")
                chat_logger.append_turn(final_state)

                # 打印三层协议栈
                for index, so in enumerate(final_state.get("precise_stack", []), start=1):
                    control_stack = final_state.get("control_stack", [])
                    semantic_stack = final_state.get("semantic_stack", [])
                    combined = {
                        "control": control_stack[index - 1] if index <= len(control_stack) else {},
                        "semantic": semantic_stack[index - 1] if index <= len(semantic_stack) else {},
                        "precise": so,
                    }
                    print_semantic_object(f"角色 {index} / SemanticObject", combined)

                print("\n========== 最终输出 ==========")
                print(final_state.get("final_answer") or "没有生成 final_answer。")
                print_metrics(final_state.get("metrics", {}))

        except KeyboardInterrupt:
            print("\n已中断。")
            break
        except Exception as exc:
            print(f"\n执行失败: {exc}")


def main() -> None:
    """默认使用MedQA三Agent入口；旧交互流程通过legacy显式进入。"""
    if len(sys.argv) > 1 and sys.argv[1] == "legacy":
        del sys.argv[1]
        legacy_main()
    else:
        from LangChain.medqa.cli import main as medqa_main
        medqa_main()


if __name__ == "__main__":
    main()
