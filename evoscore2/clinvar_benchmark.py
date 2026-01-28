"""
ClinVar Benchmark 模块

提供 ClinVar 数据筛选、拆分、阈值校准和性能评估功能
"""

import gzip
try:
    from typing import NoReturn
except ImportError:
    from typing_extensions import NoReturn
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
)


# ClinVar 临床显著性定义
PATHOGENIC_LABELS: Set[str] = {"Pathogenic", "Likely_pathogenic"}
BENIGN_LABELS: Set[str] = {"Benign", "Likely_benign"}
IGNORED_LABELS: Set[str] = {
    "Conflicting_interpretations_of_pathogenicity",
    "Uncertain_significance",
    "Drug_response",
    "Other",
    "Association",
    "Risk_factor",
    "Protective",
    "Affects",
    "Not_provided",
}


@dataclass
class ClinVarRecord:
    """ClinVar 变异记录"""
    chrom: str
    pos: int
    ref: str
    alt: str
    clinical_significance: str
    review_status: str
    stars: int
    gene: Optional[str] = None


class ClinVarFilter:
    """ClinVar 数据过滤器"""

    # ClinVar 变异类型
    MISSENSE = "single_nucleotide_variant"

    @staticmethod
    def filter_vcf(
        input_vcf: str,
        output_vcf: str,
        min_stars: int = 1,
        include_types: Optional[Set[str]] = None,
        exclude_types: Optional[Set[str]] = None,
        only_missense: bool = True,
    ) -> int:
        """
        过滤 ClinVar VCF

        Args:
            input_vcf: 输入 VCF 路径
            output_vcf: 输出 VCF 路径
            min_stars: 最小星级
            include_types: 只保留的临床显著性类型 (Pat/Likely_path, Benign/Likely_benign)
            exclude_types: 排除的临床显著性类型
            only_missense: 是否只保留 Missense 突变 (默认 True)

        Returns:
            保留的记录数
        """
        if include_types is None:
            include_types = PATHOGENIC_LABELS | BENIGN_LABELS
        if exclude_types is None:
            exclude_types = IGNORED_LABELS

        kept = 0
        total = 0

        with gzip.open(input_vcf, "rt") as fin, gzip.open(output_vcf, "wt") as fout:
            for line in fin:
                if line.startswith("#"):
                    fout.write(line)
                    continue

                total += 1
                fields = line.strip().split("\t")
                info = ClinVarFilter._parse_info(fields[7])

                clin_sig = info.get("CLNSIG", "")
                clin_vc = info.get("CLNVC", "")  # Variant Type
                clnrevstat = info.get("CLNREVSTAT", "")
                stars = ClinVarFilter._get_stars(clnrevstat)

                if stars < min_stars:
                    continue
                if clin_sig not in include_types:
                    continue
                if clin_sig in exclude_types:
                    continue
                if only_missense and clin_vc != "single_nucleotide_variant":
                    continue

                fout.write(line)
                kept += 1

        # 统计各分类数量
        counts = {"Pathogenic": 0, "Likely_pathogenic": 0, "Benign": 0, "Likely_benign": 0}
        with gzip.open(output_vcf, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                info = ClinVarFilter._parse_info(line.split("\t")[7])
                clin_sig = info.get("CLNSIG", "")
                if clin_sig in counts:
                    counts[clin_sig] += 1

        logger.info(f"ClinVar Filtering Summary:")
        logger.info(f"  Input: {total} records")
        logger.info(f"  Output: {kept} records")
        logger.info(f"  Pathogenic (P): {counts['Pathogenic']}")
        logger.info(f"  Likely_pathogenic (LP): {counts['Likely_pathogenic']}")
        logger.info(f"  Benign (B): {counts['Benign']}")
        logger.info(f"  Likely_benign (LB): {counts['Likely_benign']}")

        return kept

    @staticmethod
    def _parse_info(info_str: str) -> Dict[str, str]:
        """解析 VCF INFO 字段"""
        info = {}
        for item in info_str.split(";"):
            if "=" in item:
                key, value = item.split("=", 1)
                info[key] = value
        return info

    # ClinVar 星级映射（基于 CLNREVSTAT）
    STAR_MAPPING: Dict[str, int] = {
        "criteria_provided,_multiple_submitters": 4,
        "reviewed_by_expert_panel": 3,
        "criteria_provided,_conflicting_interpretations": 2,
        "criteria_provided,_single_submitter": 2,
        "no_criteria": 1,
    }

    @staticmethod
    def _get_stars(clnrevstat: str) -> int:
        """
        根据 CLNREVSTAT 计算星级

        星级规则（基于 ClinVar 官方标准）：
        - 4 stars: criteria provided, multiple submitters (no conflicts)
        - 3 stars: reviewed by expert panel
        - 2 stars: criteria provided, conflicting interpretations / single submitter
        - 1 star: no criteria provided / single submitter
        """
        if not clnrevstat:
            return 0

        clnrevstat = clnrevstat.lower()

        # 按优先级匹配
        if "guideline" in clnrevstat or "criteria_provided,_multiple_submitters" in clnrevstat:
            return 4
        elif "reviewed_by_expert_panel" in clnrevstat:
            return 3
        elif "criteria_provided,_conflicting_interpretations" in clnrevstat:
            return 2
        elif "criteria_provided,_single_submitter" in clnrevstat:
            return 2
        elif "no_criteria" in clnrevstat or "single_submitter" in clnrevstat:
            return 1
        else:
            return 0

    @staticmethod
    def load_filtered_vcf(vcf_path: str) -> List[ClinVarRecord]:
        """加载过滤后的 VCF"""
        records = []

        open_func = gzip.open if vcf_path.endswith(".gz") else open
        with open_func(vcf_path, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue

                fields = line.strip().split("\t")
                info = ClinVarFilter._parse_info(fields[7])

                clin_sig = info.get("CLNSIG", "")
                clnrevstat = info.get("CLNREVSTAT", "")
                gene = info.get("GENEINFO", "").split(":")[0] if "GENEINFO" in info else None

                # 根据 CLNREVSTAT 计算星级
                stars = ClinVarFilter._get_stars(clnrevstat)

                record = ClinVarRecord(
                    chrom=fields[0],
                    pos=int(fields[1]),
                    ref=fields[3],
                    alt=fields[4],
                    clinical_significance=clin_sig,
                    review_status=clnrevstat,
                    stars=stars,
                    gene=gene,
                )
                records.append(record)

        logger.info(f"Loaded {len(records)} records from {vcf_path}")
        return records


class ClinVarSplitter:
    """ClinVar 数据拆分器"""

    @staticmethod
    def stratified_split(
        records: List[ClinVarRecord],
        scores: Dict[Tuple[str, int, str, str], float],
        test_size: float = 0.8,
        random_state: int = 42,
    ) -> Tuple[List, List, np.ndarray, np.ndarray]:
        """
        分层抽样拆分数据 (2:8 拆分)

        Args:
            records: ClinVar 记录列表
            scores: 突变分数字典
            test_size: 测试集比例 (默认 0.8 = 80%)
            random_state: 随机种子

        Returns:
            (测试集记录, 训练集记录, 测试集分数, 训练集分数)
        """
        # 准备数据
        features = []
        labels = []
        valid_records = []

        for rec in records:
            key = (rec.chrom, rec.pos, rec.ref, rec.alt)
            score = scores.get(key)

            if score is None:
                continue

            if rec.clinical_significance in PATHOGENIC_LABELS:
                label = 1
            elif rec.clinical_significance in BENIGN_LABELS:
                label = 0
            else:
                continue

            features.append(score)
            labels.append(label)
            valid_records.append(rec)

        X = np.array(features)
        y = np.array(labels)

        # 分层拆分
        (
            X_train, X_test,
            y_train, y_test,
            idx_train, idx_test,
        ) = train_test_split(
            X, y, np.arange(len(X)),
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

        train_records = [valid_records[i] for i in idx_train]
        test_records = [valid_records[i] for i in idx_test]

        n_path_train = sum(1 for l in y_train if l == 1)
        n_path_test = sum(1 for l in y_test if l == 1)
        n_ben_train = sum(1 for l in y_train if l == 0)
        n_ben_test = sum(1 for l in y_test if l == 0)

        logger.info(f"Split: Train {len(X_train)} ({n_path_train} path, {n_ben_train} ben), "
                    f"Test {len(X_test)} ({n_path_test} path, {n_ben_test} ben)")

        return train_records, test_records, X_train, X_test


class ThresholdCalibrator:
    """阈值校准器"""

    @staticmethod
    def calibrate_by_specificity(
        X_train: np.ndarray,
        y_train: np.ndarray,
        target_specificity: float = 0.95,
    ) -> Tuple[float, Dict]:
        """
        基于特异性校准阈值

        Args:
            X_train: 训练集分数
            y_train: 训练集标签
            target_specificity: 目标特异性

        Returns:
            (阈值, 校准指标)
        """
        fpr, tpr, thresholds = roc_curve(y_train, X_train)
        specificities = 1 - fpr

        # 找到最接近目标特异性的阈值
        spe_idx = np.argmin(np.abs(specificities - target_specificity))
        threshold = thresholds[spe_idx]

        metrics = {
            "target_specificity": target_specificity,
            "calibrated_threshold": float(threshold),
            "actual_specificity": float(1 - fpr[spe_idx]),
            "sensitivity_at_threshold": float(tpr[spe_idx]),
        }

        logger.info(f"Calibrated threshold: {threshold:.4f} "
                    f"(specificity: {1-fpr[spe_idx]:.4f}, sensitivity: {tpr[spe_idx]:.4f})")

        return threshold, metrics

    @staticmethod
    def calibrate_by_youden(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, Dict]:
        """基于 Youden's J 统计量校准阈值"""
        fpr, tpr, thresholds = roc_curve(y_train, X_train)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        threshold = thresholds[best_idx]

        metrics = {
            "method": "youden",
            "threshold": float(threshold),
            "sensitivity": float(tpr[best_idx]),
            "specificity": float(1 - fpr[best_idx]),
        }

        logger.info(f"Youden threshold: {threshold:.4f}")
        return threshold, metrics

    @staticmethod
    def calibrate_three_class(
        X_train: np.ndarray,
        y_train: np.ndarray,
        target_specificity: float = 0.95,
        target_specificity_b: float = 0.95,
    ) -> Tuple[float, float, Dict]:
        """
        三分类阈值校准：P_threshold 和 B_threshold

        使用独立的 ROC 曲线分析：
        - P 侧（高分区域）：优化 Youden's J，最大化灵敏度
        - B 侧（低分区域）：基于特异性校准

        Args:
            X_train: 训练集分数（取反后，越正越致病）
            y_train: 训练集标签 (1=致病, 0=良性)
            target_specificity: P 侧目标特异性
            target_specificity_b: B 侧目标特异性

        Returns:
            (p_threshold, b_threshold, 校准指标)
        """
        # 计算完整 ROC 曲线
        fpr, tpr, thresholds = roc_curve(y_train, X_train)
        specificities = 1 - fpr

        # ========== P 侧分析：高特异性的前提下最大化灵敏度 ==========
        # 策略：在满足目标特异性的高分区域点中，选择灵敏度最高的
        # 这样可以提高PPV（减少假阳性）
        median_thr = np.median(X_train)
        high_thr_mask = thresholds >= median_thr
        high_thr_indices = np.where(high_thr_mask)[0]

        if len(high_thr_indices) > 0:
            high_specs = specificities[high_thr_indices]
            high_tpr = tpr[high_thr_indices]

            # 筛选满足目标特异性的高分区域点
            p_candidate_mask = high_specs >= target_specificity
            p_candidate_indices = high_thr_indices[p_candidate_mask]

            if len(p_candidate_indices) > 0:
                # 在满足特异性的候选点中，选择灵敏度最高的
                p_idx = p_candidate_indices[np.argmax(tpr[p_candidate_indices])]
            else:
                # 如果没有满足特异性的点，选择高分区域中特异性最接近目标的
                best_idx = np.argmin(np.abs(high_specs - target_specificity))
                p_idx = high_thr_indices[best_idx]
        else:
            # 如果没有高分区域点，用全局满足特异性的点
            p_idx = np.argmin(np.abs(specificities - target_specificity))

        # ========== B 侧分析：低分区域找目标特异性 ==========
        # 低分区域：thresholds <= median，预测 B = X < threshold
        low_thr_mask = thresholds <= median_thr
        low_thr_indices = np.where(low_thr_mask)[0]

        if len(low_thr_indices) > 0:
            low_specs = specificities[low_thr_indices]
            # 在低分区域找最接近目标特异性的点
            best_idx = np.argmin(np.abs(low_specs - target_specificity_b))
            b_idx = low_thr_indices[best_idx]
        else:
            b_idx = np.argmin(np.abs(specificities - target_specificity_b))

        p_thr_neg = thresholds[p_idx]
        b_thr_neg = thresholds[b_idx]

        # 转回原始尺度
        # 原始尺度：P_threshold < B_threshold
        p_threshold = -p_thr_neg
        b_threshold = -b_thr_neg

        # 确保 P < B
        if p_threshold >= b_threshold:
            p_thr_neg = np.percentile(X_train, 85)
            b_thr_neg = np.percentile(X_train, 15)
            p_threshold = -p_thr_neg
            b_threshold = -b_thr_neg
            logger.warning(f"Thresholds adjusted: P={p_threshold:.4f}, B={b_threshold:.4f}")

        metrics = {
            "p_threshold_config": {
                "method": "max_sens_high_spec",
                "target_specificity": target_specificity,
                "actual_specificity": float(1 - fpr[p_idx]),
                "sensitivity_at_p_threshold": float(tpr[p_idx]),
            },
            "b_threshold_config": {
                "target_specificity": target_specificity_b,
                "actual_specificity": float(1 - fpr[b_idx]),
                "sensitivity_at_b_threshold": float(tpr[b_idx]),
            },
            "p_threshold": float(p_threshold),
            "b_threshold": float(b_threshold),
        }

        logger.info(f"Three-class thresholds (original scale): P={p_threshold:.4f}, B={b_threshold:.4f}")
        logger.info(f"  P_region (max sens at spec>={target_specificity}): "
                    f"spec={1-fpr[p_idx]:.4f}, sens={tpr[p_idx]:.4f}")
        logger.info(f"  B_region (spec>={target_specificity_b}): "
                    f"spec={1-fpr[b_idx]:.4f}, sens={tpr[b_idx]:.4f}")

        return p_threshold, b_threshold, metrics


class BenchmarkEvaluator:
    """基准测试评估器"""

    @staticmethod
    def evaluate(
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float,
    ) -> Dict:
        """
        在测试集上评估性能（二分类）

        Args:
            X_test: 测试集分数
            y_test: 测试集标签
            threshold: 决策阈值

        Returns:
            评估指标字典
        """
        y_pred = (X_test < threshold).astype(int)  # 负分数 = 有害

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        fpr, tpr, _ = roc_curve(y_test, X_test)
        roc_auc = auc(fpr, tpr)

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, X_test)

        results = {
            "auc": float(roc_auc),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "npv": float(npv),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

        logger.info(f"Benchmark AUC: {roc_auc:.4f}, Sens: {sensitivity:.4f}, Spec: {specificity:.4f}")
        return results

    @staticmethod
    def evaluate_three_class(
        X_test: np.ndarray,
        y_test: np.ndarray,
        p_threshold: float,
        b_threshold: float,
    ) -> Dict:
        """
        在测试集上评估三分类性能（P / B / VUS）

        X_test 已经是取反后的分数（越正越致病）
        p_threshold 和 b_threshold 是原始尺度的阈值

        分类规则（原始尺度）：
          - score < P_threshold → P（致病）
          - score > B_threshold → B（良性）
          - P_threshold <= score <= B_threshold → VUS

        分类规则（取反后 X_test = -score）：
          - X_test > -P_threshold → P（因为 -score > -P_threshold 意味着 score < P_threshold）
          - X_test < -B_threshold → B（因为 -score < -B_threshold 意味着 score > B_threshold）

        Args:
            X_test: 测试集分数（取反后）
            y_test: 测试集标签 (1=致病, 0=良性)
            p_threshold: P 阈值（原始尺度）
            b_threshold: B 阈值（原始尺度）

        Returns:
            评估指标字典
        """
        # 转换阈值到取反后尺度
        # 原始尺度: P_threshold < B_threshold (e.g., -8.29 < -3.95)
        # 取反后: p_thr_neg > b_thr_neg (e.g., 8.29 > 3.95)
        p_thr_neg = -p_threshold  # e.g., 8.29
        b_thr_neg = -b_threshold  # e.g., 3.95

        # 三分类预测
        # 原始尺度分类规则:
        #   - score < P_threshold → P
        #   - score > B_threshold → B
        #   - B_threshold <= score < P_threshold → VUS
        #
        # 取反后 (X_test = -score):
        #   - score < P_threshold → -score > -P_threshold → X_test > p_thr_neg → P
        #   - score > B_threshold → -score < -B_threshold → X_test < b_thr_neg → B
        #   - B_threshold <= score < P_threshold → b_thr_neg <= X_test <= p_thr_neg → VUS
        #
        # 由于 p_thr_neg > b_thr_neg:
        #   - P region: X_test > p_thr_neg (8.29)
        #   - B region: X_test < b_thr_neg (3.95)
        #   - VUS region: b_thr_neg <= X_test <= p_thr_neg (3.95 to 8.29)

        y_pred = np.full_like(X_test, -1, dtype=int)  # 默认 VUS
        y_pred[X_test > p_thr_neg] = 1   # P: X_test > -P_threshold
        y_pred[X_test < b_thr_neg] = 0   # B: X_test < -B_threshold

        # 分别统计
        # 预测为 P 的样本
        pred_p_mask = y_pred == 1
        n_pred_p = int(np.sum(pred_p_mask))
        n_true_p_in_pred_p = int(np.sum(y_test[pred_p_mask] == 1))

        # 预测为 B 的样本
        pred_b_mask = y_pred == 0
        n_pred_b = int(np.sum(pred_b_mask))
        n_true_b_in_pred_b = int(np.sum(y_test[pred_b_mask] == 0))

        # VUS 区域
        vus_mask = ~pred_p_mask & ~pred_b_mask
        n_vus = int(np.sum(vus_mask))
        n_true_p_in_vus = int(np.sum(y_test[vus_mask] == 1))
        n_true_b_in_vus = int(np.sum(y_test[vus_mask] == 0))

        # 计算指标
        # P 侧的精确率：在预测为 P 的样本中，有多少是真 P
        ppv_p = n_true_p_in_pred_p / n_pred_p if n_pred_p > 0 else 0

        # B 侧的精确率（实际上是对 B 的 NPV）
        npv_b = n_true_b_in_pred_b / n_pred_b if n_pred_b > 0 else 0

        # 总体统计
        total_p = int(np.sum(y_test == 1))
        total_b = int(np.sum(y_test == 0))

        # P 侧的敏感性（真 P 中有多少被正确判为 P，不包括 VUS）
        sensitivity_p = n_true_p_in_pred_p / total_p if total_p > 0 else 0

        # B 侧的特异性（真 B 中有多少被正确判为 B，不包括 VUS）
        specificity_b = n_true_b_in_pred_b / total_b if total_b > 0 else 0

        # AUC
        fpr, tpr, _ = roc_curve(y_test, X_test)
        roc_auc = auc(fpr, tpr)

        results = {
            "thresholds": {
                "p_threshold": float(p_threshold),
                "b_threshold": float(b_threshold),
            },
            "predictions": {
                "predicted_P": n_pred_p,
                "predicted_B": n_pred_b,
                "VUS": n_vus,
            },
            "true_labels_in_prediction": {
                "true_P_predicted_P": n_true_p_in_pred_p,
                "true_P_in_VUS": n_true_p_in_vus,
                "true_B_predicted_B": n_true_b_in_pred_b,
                "true_B_in_VUS": n_true_b_in_vus,
            },
            "performance": {
                "AUC": float(roc_auc),
                "PPV_P": float(ppv_p),  # P 预测的精确率
                "NPV_B": float(npv_b),  # B 预测的阴性预测值
                "sensitivity_P": float(sensitivity_p),  # 真 P 中被判为 P 的比例
                "specificity_B": float(specificity_b),  # 真 B 中被判为 B 的比例
                "coverage_P": float(n_pred_p / total_p) if total_p > 0 else 0,
                "coverage_B": float(n_pred_b / total_b) if total_b > 0 else 0,
            },
            "totals": {
                "total_P": total_p,
                "total_B": total_b,
            }
        }

        logger.info("=" * 50)
        logger.info("Three-class Benchmark Results:")
        logger.info("=" * 50)
        logger.info(f"  P_threshold: {p_threshold:.4f}, B_threshold: {b_threshold:.4f}")
        logger.info(f"  Predicted P: {n_pred_p} (PPV: {ppv_p:.4f})")
        logger.info(f"  Predicted B: {n_pred_b} (NPV: {npv_b:.4f})")
        logger.info(f"  VUS: {n_vus}")
        logger.info(f"  AUC: {roc_auc:.4f}")
        logger.info("=" * 50)

        return results


def run_clinvar_benchmark(
    clinvar_vcf: str,
    scores_tsv: str,
    output_dir: str,
    min_stars: int = 1,
    test_size: float = 0.8,
    target_specificity: float = 0.95,
) -> Dict:
    """
    运行完整的 ClinVar 基准测试流程

    Args:
        clinvar_vcf: ClinVar VCF 文件路径
        scores_tsv: 分数 TSV 文件 (CHROM, POS, REF, ALT, score)
        output_dir: 输出目录
        min_stars: 最小星级
        test_size: 测试集比例
        target_specificity: 目标特异性

    Returns:
        完整结果字典
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. 过滤 ClinVar
    filtered_vcf = os.path.join(output_dir, "filtered.vcf.gz")
    ClinVarFilter.filter_vcf(
        clinvar_vcf, filtered_vcf,
        min_stars=min_stars,
    )

    # 2. 加载过滤后的数据
    records = ClinVarFilter.load_filtered_vcf(filtered_vcf)

    # 3. 加载分数
    scores_df = pd.read_csv(scores_tsv, sep="\t")
    scores_dict = {
        (row["CHROM"], row["POS"], row["REF"], row["ALT"]): row["score"]
        for _, row in scores_df.iterrows()
    }
    logger.info(f"Loaded {len(scores_dict)} scores")

    # 4. 分层拆分 (2:8)
    train_records, test_records, X_train, X_test = ClinVarSplitter.stratified_split(
        records, scores_dict, test_size=test_size
    )

    # 5. 训练集校准阈值
    threshold, cal_metrics = ThresholdCalibrator.calibrate_by_specificity(
        X_train, X_train, target_specificity
    )

    # 6. 测试集评估
    eval_metrics = BenchmarkEvaluator.evaluate(X_test, test_records, threshold)

    # 7. 汇总结果
    results = {
        "parameters": {
            "min_stars": min_stars,
            "test_size": test_size,
            "target_specificity": target_specificity,
        },
        "calibration": {
            "n_train": len(X_train),
            "threshold": threshold,
            **cal_metrics,
        },
        "evaluation": {
            "n_test": len(X_test),
            **eval_metrics,
        },
    }

    # 8. 保存结果
    pd.DataFrame({
        "CHROM": [r.chrom for r in test_records],
        "POS": [r.pos for r in test_records],
        "REF": [r.ref for r in test_records],
        "ALT": [r.alt for r in test_records],
        "CLNSIG": [r.clinical_significance for r in test_records],
        "score": X_test,
        "label": eval_metrics["tp"] + eval_metrics["fn"],
    }).to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    # 保存 JSON
    import json
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Benchmark complete. Results saved to {output_dir}")
    return results
