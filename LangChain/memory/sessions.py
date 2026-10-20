from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ConversationSession:
    """一次可导入的会话，绑定同一时间戳下的记忆文件和对话日志。"""

    index: int
    timestamp: str
    memory_path: Path
    chat_path: Path
    updated_at: float

    @property
    def name(self) -> str:
        """展示给 CLI 用户的会话名称。"""
        return self.timestamp


class MemorySessionManager:
    """管理历史记忆会话，支持新建和恢复跨任务上下文。"""

    def __init__(self, memory_dir: Path, chat_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.chat_dir = chat_dir

    def list_sessions(self) -> list[ConversationSession]:
        """按最近更新时间倒序列出可导入的记忆会话。"""
        if not self.memory_dir.exists():
            return []
        # 新存储使用共享数据库，按对话日志恢复 session ID；兼容旧 JSON 会话。
        paths = {p.stem.removeprefix("mem-"): p for p in self.memory_dir.glob("mem-*.json")}
        for chat_path in self.chat_dir.glob("chat-*.json"):
            timestamp = chat_path.stem.removeprefix("chat-")
            paths.setdefault(timestamp, self.memory_dir / f"mem-{timestamp}.json")
        def updated(path):
            chat = self.chat_dir / f"chat-{path.stem.removeprefix('mem-')}.json"
            return max(path.stat().st_mtime if path.exists() else 0,
                       chat.stat().st_mtime if chat.exists() else 0)
        files = sorted(paths.values(), key=updated, reverse=True)
        return [
            self._session_from_memory_path(index, path)
            for index, path in enumerate(files, start=1)
        ]

    def new_session(self) -> ConversationSession:
        """创建新的记忆/日志路径；文件在首次写入时落盘。"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.chat_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return ConversationSession(
            index=0,
            timestamp=timestamp,
            memory_path=self.memory_dir / f"mem-{timestamp}.json",
            chat_path=self.chat_dir / f"chat-{timestamp}.json",
            updated_at=0,
        )

    def _session_from_memory_path(self, index: int, path: Path) -> ConversationSession:
        """根据记忆文件名推导同时间戳的对话日志路径。"""
        timestamp = path.stem.removeprefix("mem-")
        return ConversationSession(
            index=index,
            timestamp=timestamp,
            memory_path=path,
            chat_path=self.chat_dir / f"chat-{timestamp}.json",
            updated_at=path.stat().st_mtime if path.exists() else (self.chat_dir / f"chat-{timestamp}.json").stat().st_mtime,
        )
