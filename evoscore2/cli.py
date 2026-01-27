"""
EVOScore-2 CLI

Usage:
    evoscore2 score-all --genome <path> --gff <path> --transcript <id> --output <path>
    evoscore2 to-vcf --scores <path> --output <path>
    evoscore2 filter-clinvar --input <path> --output <path> [--min-stars <int>]
    evoscore2 split-clinvar --input <path> --scores <path> --output <path> [--test-size <float>]
    evoscore2 calibrate --clinvar <path> --scores <path> --output <path> [--specificity <float>]
    evoscore2 benchmark --clinvar <path> --scores <path> --threshold <float> --output <path>
"""

import argparse
import sys
import os
from loguru import logger


def setup_logger(debug: bool = False):
    """配置日志"""
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(sys.stderr, level=level, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def cmd_score_all(args):
    """对所有位点进行 ESM-2 评分，输出 Parquet (带基因组坐标)"""
    import pandas as pd
    from .model import EVOScoreModel
    from .vcf_generator import GenomeData, VCFGenerator

    logger.info(f"Loading model: esm2_t33_650M_UR50D")
    model = EVOScoreModel(model_path=args.model_path, device=args.device)

    logger.info(f"Loading genome data from {args.genome} and {args.gff}")
    genome_data = GenomeData(
        genome_fasta=args.genome,
        gff_annotation=args.gff,
    )

    generator = VCFGenerator(genome_data, model)

    if args.transcript:
        # 单个转录本
        records = generator.generate_vcf(
            transcript_id=args.transcript,
            protein_id=args.protein,
            show_progress=not args.quiet,
        )
        # 保存为 Parquet
        df = generator.records_to_dataframe(records)
        df.to_parquet(args.output, index=False)
        logger.info(f"Saved {len(records)} records to {args.output}")
    else:
        # 全量转录本：分批处理，支持断点续传
        generator.generate_all_to_parquet(
            output_path=args.output,
            protein_id=args.protein,
            show_progress=not args.quiet,
        )

    genome_data.close()


def cmd_to_vcf(args):
    """将 Parquet 转换为 VCF (仅格式转换，坐标已在 score-all 时生成)"""
    from .vcf_generator import VCFGenerator

    df = pd.read_parquet(args.scores)

    generator = VCFGenerator(None, None)
    records = generator.dataframe_to_records(df)
    generator.save_vcf(records, args.output)

    logger.info(f"Converted {len(records)} records to VCF: {args.output}")


def cmd_filter_clinvar(args):
    """清洗 ClinVar"""
    from .clinvar_benchmark import ClinVarFilter

    kept = ClinVarFilter.filter_vcf(
        input_vcf=args.input,
        output_vcf=args.output,
        min_stars=args.min_stars,
    )
    logger.info(f"Filtered ClinVar: {kept} records saved to {args.output}")


def cmd_split_clinvar(args):
    """拆分 ClinVar 数据集"""
    from .clinvar_benchmark import ClinVarFilter, ClinVarSplitter

    # 加载过滤后的数据
    records = ClinVarFilter.load_filtered_vcf(args.input)

    # 加载分数
    scores_df = pd.read_csv(args.scores, sep="\t")
    scores_dict = {
        (row["CHROM"], row["POS"], row["REF"], row["ALT"]): row["score"]
        for _, row in scores_df.iterrows()
    }

    # 分层拆分
    train_records, test_records, X_train, X_test = ClinVarSplitter.stratified_split(
        records, scores_dict, test_size=args.test_size
    )

    # 保存拆分结果
    os.makedirs(args.output, exist_ok=True)

    train_df = pd.DataFrame([
        {
            "CHROM": r.chrom, "POS": r.pos, "REF": r.ref, "ALT": r.alt,
            "CLNSIG": r.clinical_significance, "score": X_train[i]
        }
        for i, r in enumerate(train_records)
    ])
    train_df.to_csv(f"{args.output}/train.csv", index=False)

    test_df = pd.DataFrame([
        {
            "CHROM": r.chrom, "POS": r.pos, "REF": r.ref, "ALT": r.alt,
            "CLNSIG": r.clinical_significance, "score": X_test[i]
        }
        for i, r in enumerate(test_records)
    ])
    test_df.to_csv(f"{args.output}/test.csv", index=False)

    logger.info(f"Split: Train={len(train_records)}, Test={len(test_records)}")
    logger.info(f"Saved to {args.output}/train.csv and {args.output}/test.csv")


def cmd_calibrate(args):
    """计算阈值"""
    from .clinvar_benchmark import ClinVarFilter, ThresholdCalibrator
    import json

    # 加载训练集
    train_df = pd.read_csv(args.input)

    X_train = train_df["score"].values
    y_train = (train_df["CLNSIG"].isin(["Pathogenic", "Likely_pathogenic"])).astype(int).values

    # 校准阈值
    threshold, metrics = ThresholdCalibrator.calibrate_by_specificity(
        X_train, y_train, target_specificity=args.specificity
    )

    # 保存结果
    result = {
        "threshold": threshold,
        "specificity": args.specificity,
        **metrics,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Calibrated threshold: {threshold:.4f}")
    logger.info(f"Saved to {args.output}")


def cmd_benchmark(args):
    """Benchmark 评估"""
    from .clinvar_benchmark import BenchmarkEvaluator
    from .clinvar_benchmark import ClinVarFilter
    import json

    # 加载测试集
    test_df = pd.read_csv(args.input)

    X_test = test_df["score"].values
    y_test = (test_df["CLNSIG"].isin(["Pathogenic", "Likely_pathogenic"])).astype(int).values

    # 评估
    metrics = BenchmarkEvaluator.evaluate(X_test, y_test, args.threshold)

    # 保存结果
    result = {
        "threshold": args.threshold,
        **metrics,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']:.4f}")
    logger.info(f"Saved to {args.output}")


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="EVOScore-2: ESM-2 based protein mutation scoring pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # 1. score-all: 对所有位点评分，输出 Parquet
    p1 = subparsers.add_parser("score-all", help="Score all sites, output Parquet")
    p1.add_argument("--genome", required=True, help="Reference genome FASTA")
    p1.add_argument("--gff", required=True, help="GFF3 annotation")
    p1.add_argument("--transcript", help="Transcript ID (optional, score all if not specified)")
    p1.add_argument("--protein", help="Protein ID (optional)")
    p1.add_argument("--output", required=True, help="Output Parquet path")
    p1.add_argument("--model-path", help="Local model path (if not specified, download from Hugging Face)")
    p1.add_argument("--device", default=None, help="Device (cuda/cpu)")
    p1.add_argument("--quiet", action="store_true", help="Suppress progress bar")

    # 2. to-vcf: Parquet 转 VCF
    p2 = subparsers.add_parser("to-vcf", help="Convert Parquet to VCF")
    p2.add_argument("--scores", required=True, help="Input Parquet file")
    p2.add_argument("--output", required=True, help="Output VCF path")

    # 3. filter-clinvar: 清洗 ClinVar
    p3 = subparsers.add_parser("filter-clinvar", help="Filter ClinVar VCF")
    p3.add_argument("--input", required=True, help="Input ClinVar VCF")
    p3.add_argument("--output", required=True, help="Output filtered VCF")
    p3.add_argument("--min-stars", type=int, default=1, help="Minimum stars (default: 1)")

    # 4. split-clinvar: 拆分 ClinVar
    p4 = subparsers.add_parser("split-clinvar", help="Stratified split ClinVar")
    p4.add_argument("--input", required=True, help="Filtered ClinVar VCF")
    p4.add_argument("--scores", required=True, help="Scores Parquet/CSV")
    p4.add_argument("--output", required=True, help="Output directory")
    p4.add_argument("--test-size", type=float, default=0.8, help="Test proportion (default: 0.8)")

    # 5. calibrate: 计算阈值
    p5 = subparsers.add_parser("calibrate", help="Calibrate threshold from training set")
    p5.add_argument("--input", required=True, help="Training set CSV")
    p5.add_argument("--scores", help="Scores CSV (if not in input)")
    p5.add_argument("--output", required=True, help="Output JSON")
    p5.add_argument("--specificity", type=float, default=0.95, help="Target specificity (default: 0.95)")

    # 6. benchmark: 评估
    p6 = subparsers.add_parser("benchmark", help="Benchmark on test set")
    p6.add_argument("--input", required=True, help="Test set CSV")
    p6.add_argument("--threshold", type=float, required=True, help="Decision threshold")
    p6.add_argument("--output", required=True, help="Output JSON")

    args = parser.parse_args()

    setup_logger(args.debug)

    if args.command == "score-all":
        cmd_score_all(args)
    elif args.command == "to-vcf":
        cmd_to_vcf(args)
    elif args.command == "filter-clinvar":
        cmd_filter_clinvar(args)
    elif args.command == "split-clinvar":
        cmd_split_clinvar(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
