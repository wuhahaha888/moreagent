"""题目与评分答案分离。运行时只接收 Question，不接收 Dataset。"""
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


# 已用于调试提示词/证据协议，不能再作为独立 heldout 成绩。
DEVELOPMENT_IDS = frozenset({"0990"})


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    options: dict[str, str]

    def public(self):
        return {"question_id": self.id, "question": self.question, "options": dict(self.options)}


class Dataset:
    def __init__(self, path: Path):
        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        rows = json.loads(raw)["medqa"]
        if not isinstance(rows, dict):
            raise ValueError("MedQA 需要 ID → 题目 的映射")
        self.questions, self._gold = [], {}
        for key, row in rows.items():
            options = row["options"]
            if len(options) != 4 or row["answer"] not in options:
                raise ValueError("MedQA 题目或答案格式错误")
            self.questions.append(Question(str(key), row["question"], dict(options)))
            self._gold[str(key)] = row["answer"]

    def select(self, *, limit=None, seed=None, ids=None, split="heldout"):
        if split not in {"heldout", "development", "all"}:
            raise ValueError("未知数据划分")
        rows = [q for q in self.questions if split == "all" or
                (q.id in DEVELOPMENT_IDS) == (split == "development")]
        if ids is not None:
            if len(set(ids)) != len(ids):
                raise ValueError("顺序文件不能包含重复题目")
            mapping = {q.id: q for q in rows}
            if not set(ids) <= set(mapping):
                raise ValueError("顺序文件含不属于所选划分的题目；全量流请显式使用 --split all")
            rows = [mapping[i] for i in ids]
        elif seed is not None:
            random.Random(seed).shuffle(rows)
        return rows if limit is None else rows[:limit]

    def grade(self, result):
        """仅实验评分侧调用；返回值不送回运行时或记忆。"""
        return result["answer"] == self._gold[result["question_id"]]
