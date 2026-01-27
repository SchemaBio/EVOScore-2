"""
EVOScore-2: ESM-2 based protein mutation scoring for clinical variant interpretation

MIT License - Free for commercial use

Model: esm2_t33_650M_UR50D (facebook)
"""

from .model import EVOScoreModel
from .scoring import (
    MutationScorer,
    compute_saturation_mutagenesis,
    VariantDatabase,
    Mutation,
)
from .vcf_generator import VCFGenerator, GenomeData, VCFRecord
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
from .benchmark import *  # Backward compatibility

__version__ = "2.0.0"
__author__ = "SchemaBio"
__license__ = "MIT"
__all__ = [
    # Model
    "EVOScoreModel",
    # Scoring
    "MutationScorer",
    "compute_saturation_mutagenesis",
    "VariantDatabase",
    "Mutation",
    # VCF
    "VCFGenerator",
    "GenomeData",
    "VCFRecord",
    # Benchmark
    "ClinVarFilter",
    "ClinVarSplitter",
    "ThresholdCalibrator",
    "BenchmarkEvaluator",
    "run_clinvar_benchmark",
    "ClinVarRecord",
    "PATHOGENIC_LABELS",
    "BENIGN_LABELS",
]
