from pathlib import Path

from ..agents.workflow import MultiAgentSystem
from .config import load_settings
from .llm import build_chat_model
from ..memory.embedding import QwenEmbeddingService
from ..memory.store import JsonMemoryStore
from ..memory.advanced import SQLiteMemoryStore
from ..memory.agent import MemoryAgent
from ..tools.web_search import DisabledSearchTool, SearxngSearchTool, WebSearchConfig


def create_multi_agent_system(memory_path: Path) -> tuple[MultiAgentSystem, str]:
    """按配置组装 LLM、embedding、共享记忆和检索工具。"""
    settings = load_settings()
    llm = build_chat_model(settings)
    embeddings = QwenEmbeddingService.from_settings(settings)
    memory_agent = None
    if settings.memory_backend == "agentic":
        memory = SQLiteMemoryStore(settings.memory_dir / "shared.sqlite3",
            settings.memory_namespace, memory_path.stem, embeddings.cache_key,
            settings.memory_search_limit)
        embeddings.cache_store = memory
        memory_agent = MemoryAgent(memory, embeddings, llm if settings.memory_curator_enabled else None,
            max_chars=settings.memory_context_chars, ttl_days=settings.memory_ttl_days,
            strategy=settings.memory_retrieval_strategy,
            curation_mode=settings.memory_curation_mode,
            include_task_local=settings.memory_include_task_local)
    elif settings.memory_backend == "json":
        memory = JsonMemoryStore(memory_path, settings.memory_search_limit)
    else:
        raise ValueError("MEMORY_BACKEND 必须为 agentic 或 json")
    search_tool = (
        SearxngSearchTool(
            WebSearchConfig(
                base_url=settings.searxng_base_url,
                max_results=settings.web_search_max_results,
                timeout=settings.web_search_timeout,
                language=settings.web_search_language,
            )
        )
        if settings.web_search_enabled
        else DisabledSearchTool()
    )
    return MultiAgentSystem(llm, embeddings, memory, search_tool,
                            memory_agent=memory_agent), settings.deepseek_model
