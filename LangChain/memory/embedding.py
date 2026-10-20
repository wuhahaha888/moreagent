"""多粒度 Embedding 服务：句子级 / 任务级 / 上下文级。

支持 SemanticObject 协议的 SemanticVectors 层：
- sentence_vec: 每个 Agent 输出的语义位置
- task_vec: 用户问题的语义位置（全局不变）
- context_vec: 累积上下文的语义位置（每轮更新）
"""

import math
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from sentence_transformers import SentenceTransformer

from ..framework.config import Settings


@dataclass(frozen=True)
class QwenEmbeddingConfig:
    """Qwen embedding 模型配置。"""

    model_path: str
    batch_size: int = 8
    normalize_embeddings: bool = True
    show_progress_bar: bool = False
    cache_version: str = "v1"


class QwenEmbeddingService:
    """延迟加载的 embedding 服务，支持多粒度向量生成。"""

    def __init__(self, config: QwenEmbeddingConfig) -> None:
        self.config = config
        self._model: SentenceTransformer | None = None
        self.cache_store = None
        self.cache_hits = 0
        self.encoded_texts = 0
        self.encoded_seconds = 0.

    @property
    def cache_key(self) -> str:
        # 模型/归一化/编码策略版本必须全部匹配；更换权重时提升 cache_version。
        return f"{self.config.model_path}|norm={self.config.normalize_embeddings}|{self.config.cache_version}"

    @classmethod
    def from_settings(cls, settings: Settings) -> "QwenEmbeddingService":
        """从全局配置构造 embedding 服务。"""
        return cls(
            QwenEmbeddingConfig(
                model_path=settings.embedding_model_path,
                batch_size=settings.embedding_batch_size,
                normalize_embeddings=settings.embedding_normalize,
                show_progress_bar=settings.embedding_show_progress,
                cache_version=settings.embedding_cache_version,
            )
        )

    @property
    def model(self) -> SentenceTransformer:
        """首次使用时加载模型，自动选择 CUDA 或 CPU。"""
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(
                self.config.model_path,
                trust_remote_code=True,
                device=device,
            )
        return self._model

    @property
    def device(self) -> str:
        """返回实际承载 embedding 模型的设备名称。"""
        return str(self.model.device)

    @property
    def dim(self) -> int:
        """返回向量维度。"""
        return self.model.get_embedding_dimension()

    @property
    def model_name(self) -> str:
        """返回模型标识，用于 SemanticVectors.model 字段。"""
        return self.config.model_path.split("/")[-1].split("\\")[-1]

    # ─── 单条/批量嵌入 ────────────────────────────────────────────────

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        """批量生成文本 embedding。"""
        text_list = [text or "" for text in texts]
        if not text_list:
            return []
        cached = {}
        missing = []
        for text in dict.fromkeys(text_list):
            value = self.cache_store.cache_get(self.cache_key, text) if self.cache_store else None
            if value is None:
                missing.append(text)
            else:
                cached[text] = value
                self.cache_hits += text_list.count(text)
        if missing:
            encode_started=time.perf_counter()
            try:
                embeddings = self.model.encode(
                    missing,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=self.config.show_progress_bar,
                ).astype(float).tolist()
            finally:
                self.encoded_seconds += time.perf_counter()-encode_started
            self.encoded_texts += len(missing)
            for text, values in zip(missing, embeddings):
                cached[text] = values
                if self.cache_store:
                    self.cache_store.cache_put(self.cache_key, text, values)
        return [list(cached[text]) for text in text_list]

    def embed_text(self, text: str) -> list[float]:
        """生成单条文本 embedding。"""
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []

    # ─── 多粒度向量生成 ───────────────────────────────────────────────

    def embed_sentence(self, text: str) -> list[float]:
        """句子级嵌入：Agent 单次输出的语义位置。

        对 Agent 输出的 PreciseStructure JSON 做嵌入，
        用于 Agent 间意图匹配和输出相似度计算。
        """
        return self.embed_text(text)

    def embed_task(self, question: str) -> list[float]:
        """任务级嵌入：用户问题的语义位置。

        全局不变，用于任务路由、跨任务记忆召回。
        """
        return self.embed_text(question)

    def embed_context(self, texts: list[str]) -> list[float]:
        """上下文级嵌入：累积上下文的语义位置。

        将多轮 Agent 输出拼接后做嵌入，用于上下文相似度和缓存命中。
        如果输入为空则返回空向量。
        """
        if not texts:
            return []
        combined = " ".join(text.strip() for text in texts if text.strip())
        if not combined:
            return []
        return self.embed_text(combined)

    # ─── 向量运算 ─────────────────────────────────────────────────────

    @staticmethod
    def mean_pool(vectors: list[list[float]]) -> list[float]:
        """对多个向量取均值，用于 context_vec 的增量更新。"""
        if not vectors:
            return []
        dim = len(vectors[0])
        result = [0.0] * dim
        for vec in vectors:
            for i in range(min(dim, len(vec))):
                result[i] += vec[i]
        count = len(vectors)
        return [v / count for v in result]

    @staticmethod
    def weighted_pool(
        vectors: list[list[float]], weights: list[float]
    ) -> list[float]:
        """对多个向量做加权平均，近期输出权重更高。"""
        if not vectors or not weights:
            return []
        dim = len(vectors[0])
        result = [0.0] * dim
        total_weight = 0.0
        for vec, w in zip(vectors, weights):
            for i in range(min(dim, len(vec))):
                result[i] += vec[i] * w
            total_weight += w
        if total_weight == 0:
            return [0.0] * dim
        return [v / total_weight for v in result]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量的余弦相似度，维度不一致时按共同长度比较。"""
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
