"""紧凑提示保留最小字段示例，避免省去契约后增加修复调用。"""
import json
from .medqa_prompts import COMMON_RULES, ROLE_INSTRUCTIONS, OUTPUT_EXAMPLES

COMPACT_ROLE_PROMPTS = {
    role: COMMON_RULES + "\n" + text + "\nJSON形状示例（按实际审核填字段；无外部证据仍可answer，不能伪造充分性）：\n"
    + json.dumps(OUTPUT_EXAMPLES[role], ensure_ascii=False, separators=(",", ":"))
    for role, text in ROLE_INSTRUCTIONS.items()
}
