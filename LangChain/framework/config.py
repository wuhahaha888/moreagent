import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"


@dataclass(frozen=True)
class Settings:
    """项目运行配置，统一描述模型、embedding、记忆和搜索工具参数。"""

    openai_api_key: str
    openai_base_url: str
    openai_model: str
    openai_temperature: float
    embedding_model_path: str
    embedding_batch_size: int
    embedding_normalize: bool
    embedding_show_progress: bool
    memory_dir: Path
    chat_dir: Path
    memory_search_limit: int
    web_search_enabled: bool
    searxng_base_url: str
    web_search_max_results: int
    web_search_timeout: float
    web_search_language: str
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    memory_backend: str = "agentic"
    memory_namespace: str = "default"
    memory_context_chars: int = 3200
    memory_ttl_days: int = 30
    embedding_cache_version: str = "v1"
    memory_curator_enabled: bool = True
    memory_retrieval_strategy: str = "hybrid"
    memory_curation_mode: str = "auto"
    memory_include_task_local: bool = False


def load_env_file(path: Path = ENV_FILE) -> None:
    """读取 .env 文件并写入缺失的环境变量，已存在变量不会被覆盖。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_env(name: str) -> str:
    """读取必填环境变量，缺失时给出明确配置错误。"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少配置项 {name}，请在 .env 中配置。")
    return value


def parse_bool(value: str) -> bool:
    """解析常见布尔配置写法。"""
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_project_path(value: str) -> Path:
    """将相对路径解析到项目根目录，绝对路径保持不变。"""
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def load_settings() -> Settings:
    """加载完整运行配置并应用默认值。"""
    load_env_file()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
        openai_model=os.getenv("OPENAI_MODEL", ""),
        openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        embedding_model_path=os.getenv("EMBEDDING_MODEL_PATH", r"D:\Models\Qwen3-Embedding-0.6B"),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
        embedding_normalize=parse_bool(os.getenv("EMBEDDING_NORMALIZE", "true")),
        embedding_show_progress=parse_bool(os.getenv("EMBEDDING_SHOW_PROGRESS", "false")),
        memory_dir=resolve_project_path(os.getenv("MEMORY_DIR", "data/memories")),
        chat_dir=resolve_project_path(os.getenv("CHAT_DIR", "data/chats")),
        memory_search_limit=int(os.getenv("MEMORY_SEARCH_LIMIT", "3")),
        web_search_enabled=parse_bool(os.getenv("WEB_SEARCH_ENABLED", "true")),
        searxng_base_url=os.getenv("SEARXNG_BASE_URL", "http://localhost:8082"),
        web_search_max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5")),
        web_search_timeout=float(os.getenv("WEB_SEARCH_TIMEOUT", "20")),
        web_search_language=os.getenv("WEB_SEARCH_LANGUAGE", "zh-CN"),
        deepseek_api_key=require_env("DEEPSEEK_API_KEY"),
        deepseek_base_url=require_env("DEEPSEEK_BASE_URL"),
        deepseek_model=require_env("DEEPSEEK_MODEL"),
        memory_backend=os.getenv("MEMORY_BACKEND", "agentic"),
        memory_namespace=os.getenv("MEMORY_NAMESPACE", "default"),
        memory_context_chars=int(os.getenv("MEMORY_CONTEXT_CHARS", "3200")),
        memory_ttl_days=int(os.getenv("MEMORY_TTL_DAYS", "30")),
        embedding_cache_version=os.getenv("EMBEDDING_CACHE_VERSION", "v1"),
        memory_curator_enabled=parse_bool(os.getenv("MEMORY_CURATOR_ENABLED", "true")),
        memory_retrieval_strategy=os.getenv("MEMORY_RETRIEVAL_STRATEGY", "hybrid"),
        memory_curation_mode=os.getenv("MEMORY_CURATION_MODE", "auto"),
        memory_include_task_local=parse_bool(os.getenv("MEMORY_INCLUDE_TASK_LOCAL", "false")),
    )
