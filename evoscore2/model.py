"""
ESM-2 模型加载与推理模块

提供 ESM-2 模型的加载、Log-Likelihood Ratio (LLR) 计算功能
"""

import torch
from transformers import EsmForMaskedLM, AutoTokenizer
from loguru import logger
from typing import Optional, List, Tuple, Dict
import numpy as np


class EVOScoreModel:
    """
    ESM-2 蛋白质语言模型封装类

    支持加载不同规模的 ESM-2 模型，提供掩码语言模型推理和 LLR 分数计算
    """

    # 固定的 Hugging Face 模型名称
    MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    MODEL_PARAMS = 650_000_000

    # 标准氨基酸列表
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
    AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        half_precision: bool = True,
    ):
        """
        初始化 ESM-2 模型 (esm2_t33_650M_UR50D)

        Args:
            model_path: 模型路径，支持:
                        - None: 从 Hugging Face Hub 下载
                        - 本地路径: 从本地目录加载预下载的模型
            device: 计算设备，None 表示自动检测
            half_precision: 是否使用半精度 (FP16) 推理
        """
        self.half_precision = half_precision
        self.model_path = model_path

        # 自动检测设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing EVOScoreModel on {self.device}")

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载模型和分词器"""
        # 确定模型来源
        if self.model_path:
            source = self.model_path
            logger.info(f"Loading model from local path: {self.model_path}")
        else:
            source = self.MODEL_NAME
            logger.info(f"Loading model from Hugging Face Hub: {self.MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = EsmForMaskedLM.from_pretrained(source).to(self.device)

        if self.half_precision and self.device == "cuda":
            self.model = self.model.half()

        self.model.eval()

        logger.info(f"Model loaded successfully: {self.MODEL_PARAMS/1e6:.1f}M parameters")

        # 预计算所有氨基酸的 token ID
        self.aa_token_ids = {}
        for aa in self.AMINO_ACIDS:
            token_id = self.tokenizer.convert_tokens_to_ids(aa)
            if token_id == self.tokenizer.unk_token_id:
                logger.warning(f"Unknown token for amino acid: {aa}")
            self.aa_token_ids[aa] = token_id

    @torch.no_grad()
    def get_logits(
        self,
        sequences: List[str],
        batch_size: int = 1,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        获取多个序列的模型logits

        Args:
            sequences: 蛋白质序列列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条

        Returns:
            logits: [batch_seq_len, vocab_size] 的张量
        """
        from tqdm import tqdm

        all_logits = []

        iterator = range(0, len(sequences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Getting logits")

        for i in iterator:
            batch_seq = sequences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_seq,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            logits = self.model(**inputs).logits
            all_logits.append(logits.cpu())

        return torch.cat(all_logits, dim=0)

    @torch.no_grad()
    def get_log_probs(
        self,
        sequences: List[str],
        batch_size: int = 4,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        获取多个序列的对数概率

        Args:
            sequences: 蛋白质序列列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条

        Returns:
            log_probs: [batch, seq_len, 20] 的张量
        """
        from tqdm import tqdm

        all_log_probs = []

        iterator = range(0, len(sequences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Getting log probs")

        for i in iterator:
            batch_seq = sequences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_seq,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            logits = self.model(**inputs).logits
            log_probs = torch.log_softmax(logits, dim=-1)

            # 提取序列位置对应的概率（跳过 CLS 和 EOS token）
            all_log_probs.append(log_probs.cpu())

        return torch.cat(all_log_probs, dim=0)

    @torch.no_grad()
    def score_variant_llr(
        self,
        sequence: str,
        position: int,
        ref_aa: str,
        alt_aa: str,
    ) -> Optional[float]:
        """
        计算单个错义突变的 LLR 分数

        Args:
            sequence: 原始蛋白质序列
            position: 突变位置 (0-based)
            ref_aa: 野生型氨基酸 (单字母)
            alt_aa: 突变型氨基酸 (单字母)

        Returns:
            LLR分数，负值表示可能有害
        """
        # 验证输入
        if position < 0 or position >= len(sequence):
            logger.error(f"Position {position} out of range [0, {len(sequence)})")
            return None

        if ref_aa not in self.AMINO_ACIDS or alt_aa not in self.AMINO_ACIDS:
            logger.error(f"Invalid amino acid: ref={ref_aa}, alt={alt_aa}")
            return None

        # 验证参考氨基酸匹配
        actual_ref = sequence[position]
        if actual_ref != ref_aa:
            logger.warning(
                f"Reference mismatch at position {position}: "
                f"sequence has '{actual_ref}', expected '{ref_aa}'"
            )
            return None

        # Tokenize
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        token_ids = inputs["input_ids"]

        # ESM tokenizer 添加 CLS (index 0) 和 EOS (末尾)
        target_pos_idx = position + 1

        # 创建掩码输入
        masked_input = token_ids.clone()
        masked_input[0, target_pos_idx] = self.tokenizer.mask_token_id

        # 推理
        logits = self.model(masked_input).logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # 获取 log 概率
        ref_token_id = self.aa_token_ids.get(ref_aa)
        alt_token_id = self.aa_token_ids.get(alt_aa)

        if ref_token_id is None or alt_token_id is None:
            logger.error(f"Cannot find token ID for {ref_aa} or {alt_aa}")
            return None

        log_prob_ref = log_probs[0, target_pos_idx, ref_token_id].item()
        log_prob_alt = log_probs[0, target_pos_idx, alt_token_id].item()

        # 计算 LLR
        llr_score = log_prob_alt - log_prob_ref
        return llr_score

    @torch.no_grad()
    def compute_mutation_matrix(
        self,
        sequence: str,
        batch_size: int = 4,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        计算序列所有位置的突变分数矩阵

        Args:
            sequence: 蛋白质序列
            batch_size: 批处理大小
            show_progress: 是否显示进度条

        Returns:
            matrix: [seq_len, 20] 的 numpy 数组，每行对应一个位置，
                   每列对应20种氨基酸的 LLR 分数
        """
        from tqdm import tqdm

        seq_len = len(sequence)
        matrix = np.full((seq_len, 20), np.nan, dtype=np.float32)

        iterator = range(seq_len)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing mutation matrix")

        for pos in iterator:
            ref_aa = sequence[pos]

            for alt_aa in self.AMINO_ACIDS:
                if alt_aa == ref_aa:
                    matrix[pos, self.AA_TO_IDX[alt_aa]] = 0.0
                else:
                    score = self.score_variant_llr(sequence, pos, ref_aa, alt_aa)
                    if score is not None:
                        matrix[pos, self.AA_TO_IDX[alt_aa]] = score

        return matrix

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": self.MODEL_NAME,
            "parameters": self.MODEL_PARAMS,
            "device": self.device,
            "half_precision": self.half_precision,
            "vocab_size": self.tokenizer.vocab_size,
        }
