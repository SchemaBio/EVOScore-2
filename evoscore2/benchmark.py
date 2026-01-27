"""
ClinVar Benchmark 模块 (Backward Compatible)

提供 ClinVar 数据筛选、拆分、阈值校准和性能评估功能
"""

from .clinvar_benchmark import (
    ClinVarFilter,
    ClinVarSplitter,
    ThresholdCalibrator,
    BenchmarkEvaluator,
    run_clinvar_benchmark,
    ClinVarRecord,
    PATHOGENIC_LABELS,
    BENIGN_LABELS,
)

__all__ = [
    "ClinVarFilter",
    "ClinVarSplitter",
    "ThresholdCalibrator",
    "BenchmarkEvaluator",
    "run_clinvar_benchmark",
    "ClinVarRecord",
    "PATHOGENIC_LABELS",
    "BENIGN_LABELS",
]
