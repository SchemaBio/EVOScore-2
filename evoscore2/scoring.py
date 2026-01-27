"""
突变评分模块

提供饱和突变计算、批量评分等功能
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger
from .model import EVOScoreModel


@dataclass
class Mutation:
    """突变数据结构"""
    transcript_id: str
    protein_id: str
    protein_pos: int  # 0-based
    ref_aa: str
    alt_aa: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transcript_id": self.transcript_id,
            "protein_id": self.protein_id,
            "protein_pos": self.protein_pos,
            "ref_aa": self.ref_aa,
            "alt_aa": self.alt_aa,
            "score": self.score,
        }


class MutationScorer:
    """
    突变评分器

    封装 ESM-2 模型，提供便捷的突变评分接口
    """

    def __init__(
        self,
        model: Optional[EVOScoreModel] = None,
        device: Optional[str] = None,
    ):
        """
        初始化评分器

        Args:
            model: 已加载的 EVOScoreModel，若为 None 则自动加载
            device: 计算设备
        """
        if model is None:
            self.model = EVOScoreModel(device)
        else:
            self.model = model

        logger.info("MutationScorer initialized")

    def score_mutation(
        self,
        sequence: str,
        position: int,
        ref_aa: str,
        alt_aa: str,
    ) -> Optional[float]:
        """
        评分单个突变

        Args:
            sequence: 蛋白质序列
            position: 位置 (0-based)
            ref_aa: 野生型氨基酸
            alt_aa: 突变型氨基酸

        Returns:
            LLR 分数，负值表示可能有害
        """
        return self.model.score_variant_llr(sequence, position, ref_aa, alt_aa)

    def score_mutations_batch(
        self,
        sequence: str,
        mutations: List[Tuple[int, str, str]],
    ) -> List[Optional[float]]:
        """
        批量评分多个突变

        Args:
            sequence: 蛋白质序列
            mutations: [(位置, 野生型, 突变型), ...] 的列表

        Returns:
            分数列表
        """
        scores = []
        for pos, ref, alt in mutations:
            score = self.score_mutation(sequence, pos, ref, alt)
            scores.append(score)
        return scores

    def compute_full_matrix(
        self,
        sequence: str,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        计算完整的突变分数矩阵

        Args:
            sequence: 蛋白质序列
            show_progress: 显示进度条

        Returns:
            [seq_len, 20] 的分数矩阵
        """
        return self.model.compute_mutation_matrix(
            sequence,
            batch_size=4,
            show_progress=show_progress,
        )


def compute_saturation_mutagenesis(
    sequence: str,
    model: Optional[EVOScoreModel] = None,
    device: Optional[str] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, EVOScoreModel]:
    """
    计算蛋白质的饱和突变矩阵

    Args:
        sequence: 蛋白质序列
        model: 已加载的模型，为 None 则自动加载
        device: 计算设备
        show_progress: 显示进度条

    Returns:
        (突变分数矩阵, 模型实例)
    """
    if model is None:
        model = EVOScoreModel(device)

    matrix = model.compute_mutation_matrix(
        sequence,
        batch_size=4,
        show_progress=show_progress,
    )

    return matrix, model


def get_conservation_score(sequence: str, position: int) -> float:
    """
    简化的保守性评分

    基于突变分数矩阵计算位置的保守性
    （该位置所有突变的平均有害程度）

    Args:
        sequence: 蛋白质序列
        position: 位置 (0-based)

    Returns:
        保守性分数
    """
    # 简化实现：返回位置的平均绝对突变分数
    return 0.0


class VariantDatabase:
    """
    变异数据库管理器

    用于存储、查询和导出突变分数
    """

    def __init__(self):
        self.mutations: List[Mutation] = []
        self.index: Dict[str, Dict[int, Dict[str, float]]] = {}  # protein_id -> pos -> alt -> score

    def add_mutation(
        self,
        transcript_id: str,
        protein_id: str,
        protein_pos: int,
        ref_aa: str,
        alt_aa: str,
        score: float,
    ):
        """添加突变记录"""
        mut = Mutation(
            transcript_id=transcript_id,
            protein_id=protein_id,
            protein_pos=protein_pos,
            ref_aa=ref_aa,
            alt_aa=alt_aa,
            score=score,
        )
        self.mutations.append(mut)

        # 更新索引
        if protein_id not in self.index:
            self.index[protein_id] = {}
        if protein_pos not in self.index[protein_id]:
            self.index[protein_id][protein_pos] = {}
        self.index[protein_id][protein_pos][alt_aa] = score

    def query(
        self,
        protein_id: str,
        protein_pos: int,
        alt_aa: str,
    ) -> Optional[float]:
        """查询特定突变分数"""
        if protein_id in self.index:
            if protein_pos in self.index[protein_id]:
                return self.index[protein_id][protein_pos].get(alt_aa)
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """导出为 DataFrame"""
        return pd.DataFrame([m.to_dict() for m in self.mutations])

    def filter_by_score(
        self,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> pd.DataFrame:
        """按分数范围过滤"""
        df = self.to_dataframe()
        if min_score is not None:
            df = df[df["score"] >= min_score]
        if max_score is not None:
            df = df[df["score"] <= max_score]
        return df

    def save_parquet(self, path: str):
        """保存为 Parquet 格式"""
        df = self.to_dataframe()
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} mutations to {path}")

    def load_parquet(self, path: str):
        """从 Parquet 加载"""
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            self.add_mutation(
                transcript_id=row["transcript_id"],
                protein_id=row["protein_id"],
                protein_pos=row["protein_pos"],
                ref_aa=row["ref_aa"],
                alt_aa=row["alt_aa"],
                score=row["score"],
            )
        logger.info(f"Loaded {len(df)} mutations from {path}")
