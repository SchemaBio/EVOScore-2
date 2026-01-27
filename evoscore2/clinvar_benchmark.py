"""
ClinVar Benchmark 模块

提供 ClinVar 数据筛选、拆分、阈值校准和性能评估功能
"""

import gzip
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
                stars = ClinVarFilter._get_stars(info.get("CLNDISDB", ""))

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

    @staticmethod
    def _get_stars(clndisdb: str) -> int:
        """提取星级"""
        if not clndisdb:
            return 0
        try:
            return int(clndisdb.split(":")[0].split(",")[0].split(";")[0])
        except (ValueError, IndexError):
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
                gene = info.get("GENEINFO", "").split(":")[0] if "GENEINFO" in info else None

                record = ClinVarRecord(
                    chrom=fields[0],
                    pos=int(fields[1]),
                    ref=fields[3],
                    alt=fields[4],
                    clinical_significance=clin_sig,
                    review_status=info.get("CLNREVSTAT", ""),
                    stars=ClinVarFilter._get_stars(info.get("CLNDISDB", "")),
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


class BenchmarkEvaluator:
    """基准测试评估器"""

    @staticmethod
    def evaluate(
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float,
    ) -> Dict:
        """
        在测试集上评估性能

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
