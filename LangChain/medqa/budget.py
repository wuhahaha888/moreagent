"""资源保护上限不是完成判据；不预分配初查/补查。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchBudget:
    max_searches: int = 6
    max_reads: int = 4
    max_stalled_rounds: int = 2
    max_tool_retries: int = 1

    def __post_init__(self):
        for value in (self.max_searches, self.max_reads, self.max_stalled_rounds):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("工具保护上限须为正整数")
        if self.max_tool_retries not in (0, 1, 2):
            raise ValueError("工具故障重试上限须为0至2")
