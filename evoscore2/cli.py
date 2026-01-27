"""
EVOScore-2 CLI

## 模型打分流程:
    evoscore2 score-all --genome <path> --gff <path> --output <path>
    evoscore2 to-vcf --scores <path> --output <path>

## 基于预打分文件流程 (hg38_VESM_3B_scores.parquet.gzip):
    evoscore2 query --scores <path> --chrom <str> --pos <int> --ref <str> --alt <str>
    evoscore2 annotate --scores <path> --input <vcf> --output <vcf>
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


# ==================== 模型打分流程 ====================

def cmd_score_all(args):
    """对所有位点进行 ESM-2 评分，输出 Parquet"""
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
        df = generator.records_to_dataframe(records)
        df.to_parquet(args.output, index=False)
        logger.info(f"Saved {len(records)} records to {args.output}")
    else:
        # 全量转录本
        generator.generate_all_to_parquet(
            output_path=args.output,
            protein_id=args.protein,
            show_progress=not args.quiet,
        )

    genome_data.close()


def cmd_to_vcf(args):
    """Parquet 转 VCF"""
    import pandas as pd
    from .vcf_generator import VCFGenerator
    import gzip

    # 处理 .parquet.gzip 压缩文件
    if args.scores.endswith(".parquet.gzip") or args.scores.endswith(".gz"):
        with gzip.open(args.scores, 'rb') as f:
            df = pd.read_parquet(f)
    else:
        df = pd.read_parquet(args.scores)

    # 标准化列名 (处理 "VESM (3B)" 这种特殊列名)
    column_mapping = {
        "VESM (3B)": "score",
        "VESM(3B)": "score",
        "VESM_3B": "score",
        "VESM": "score",
        "EVOScore": "score",
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})

    # 确保必需列存在
    required_cols = ["CHROM", "POS", "REF", "ALT", "score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    generator = VCFGenerator(None, None)
    records = generator.dataframe_to_records(df)
    generator.save_vcf(records, args.output)
    logger.info(f"Converted {len(records)} records to VCF: {args.output}")


# ==================== 基于预打分文件流程 ====================

def _load_scores(args):
    """加载分数文件，统一处理列名"""
    import pandas as pd
    import gzip

    if args.scores.endswith(".parquet.gzip") or args.scores.endswith(".parquet.gz"):
        with gzip.open(args.scores, 'rb') as f:
            df = pd.read_parquet(f)
    elif args.scores.endswith(".parquet"):
        df = pd.read_parquet(args.scores)
    else:
        df = pd.read_csv(args.scores, sep="\t")

    # 标准化列名
    column_mapping = {
        "VESM (3B)": "score",
        "VESM(3B)": "score",
        "VESM_3B": "score",
        "VESM": "score",
        "EVOScore": "score",
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})

    return df


def cmd_query(args):
    """查询单个位点分数"""
    df = _load_scores(args)

    # 精确匹配
    mask = (
        (df["CHROM"].astype(str) == str(args.chrom)) &
        (df["POS"] == args.pos) &
        (df["REF"].astype(str) == str(args.ref)) &
        (df["ALT"].astype(str) == str(args.alt))
    )

    results = df[mask]
    if len(results) > 0:
        for _, row in results.iterrows():
            print(f"CHROM: {row['CHROM']}")
            print(f"POS: {row['POS']}")
            print(f"REF: {row['REF']}")
            print(f"ALT: {row['ALT']}")
            print(f"score: {row['score']}")
    else:
        print("No match found")


def cmd_annotate(args):
    """VCF 批量注释"""
    import pysam

    # 加载分数文件
    scores_df = _load_scores(args)

    # 构建查询索引
    scores_dict = {}
    for _, row in scores_df.iterrows():
        key = (str(row["CHROM"]), int(row["POS"]), str(row["REF"]), str(row["ALT"]))
        scores_dict[key] = row["score"]

    # 读取并注释 VCF
    with pysam.VariantFile(args.input, "r") as vcf_in:
        # 添加 INFO 字段
        vcf_in.header.add_line(
            '##INFO=<ID=EVOScore,Number=1,Type=Float,Description="ESM-2 based pathogenicity score">'
        )

        vcf_out = pysam.VariantFile(args.output, "w", header=vcf_in.header)

        for record in vcf_in:
            for alt in record.alts:
                key = (str(record.chrom), record.pos, str(record.ref), str(alt))
                if key in scores_dict:
                    record.info["EVOScore"] = scores_dict[key]
            vcf_out.write(record)

    logger.info(f"Annotated VCF saved to {args.output}")


# ==================== ClinVar 流程 ====================

def cmd_filter_clinvar(args):
    """清洗 ClinVar"""
    from .clinvar_benchmark import ClinVarFilter

    ClinVarFilter.filter_vcf(
        input_vcf=args.input,
        output_vcf=args.output,
        min_stars=args.min_stars,
    )
    logger.info(f"ClinVar filtering complete")


def cmd_split_clinvar(args):
    """拆分 ClinVar 数据集"""
    from .clinvar_benchmark import ClinVarFilter, ClinVarSplitter

    records = ClinVarFilter.load_filtered_vcf(args.input)

    # 加载分数
    scores_df = _load_scores(args)

    scores_dict = {
        (str(row["CHROM"]), int(row["POS"]), str(row["REF"]), str(row["ALT"])): row["score"]
        for _, row in scores_df.iterrows()
    }

    train_records, test_records, X_train, X_test = ClinVarSplitter.stratified_split(
        records, scores_dict, test_size=args.test_size
    )

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


def cmd_calibrate(args):
    """计算阈值"""
    from .clinvar_benchmark import ThresholdCalibrator
    import json

    train_df = pd.read_csv(args.input)
    X_train = train_df["score"].values
    y_train = (train_df["CLNSIG"].isin(["Pathogenic", "Likely_pathogenic"])).astype(int).values

    threshold, metrics = ThresholdCalibrator.calibrate_by_specificity(
        X_train, y_train, target_specificity=args.specificity
    )

    result = {"threshold": threshold, "specificity": args.specificity, **metrics}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Calibrated threshold: {threshold:.4f}")
    logger.info(f"Saved to {args.output}")


def cmd_benchmark(args):
    """Benchmark 评估"""
    from .clinvar_benchmark import BenchmarkEvaluator
    import json

    test_df = pd.read_csv(args.input)
    X_test = test_df["score"].values
    y_test = (test_df["CLNSIG"].isin(["Pathogenic", "Likely_pathogenic"])).astype(int).values

    metrics = BenchmarkEvaluator.evaluate(X_test, y_test, args.threshold)

    result = {"threshold": args.threshold, **metrics}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']:.4f}")


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="EVOScore-2: ESM-2 based protein mutation scoring",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ---- 模型打分流程 ----
    p1 = subparsers.add_parser("score-all", help="Score all sites with ESM-2, output Parquet")
    p1.add_argument("--genome", required=True, help="Reference genome FASTA")
    p1.add_argument("--gff", required=True, help="GFF3 annotation")
    p1.add_argument("--transcript", help="Transcript ID (optional)")
    p1.add_argument("--protein", help="Protein ID (optional)")
    p1.add_argument("--output", required=True, help="Output Parquet path")
    p1.add_argument("--model-path", help="Local model path")
    p1.add_argument("--device", default=None, help="Device (cuda/cpu)")
    p1.add_argument("--quiet", action="store_true", help="Suppress progress bar")

    p2 = subparsers.add_parser("to-vcf", help="Convert Parquet to VCF")
    p2.add_argument("--scores", required=True, help="Input Parquet file")
    p2.add_argument("--output", required=True, help="Output VCF path")

    # ---- 基于预打分文件流程 ----
    p3 = subparsers.add_parser("query", help="Query single variant score")
    p3.add_argument("--scores", required=True, help="Scores Parquet/CSV")
    p3.add_argument("--chrom", required=True, help="Chromosome")
    p3.add_argument("--pos", required=True, type=int, help="Position")
    p3.add_argument("--ref", required=True, help="Reference allele")
    p3.add_argument("--alt", required=True, help="Alternative allele")

    p4 = subparsers.add_parser("annotate", help="Annotate VCF with EVOScore")
    p4.add_argument("--scores", required=True, help="Scores Parquet/CSV")
    p4.add_argument("--input", required=True, help="Input VCF")
    p4.add_argument("--output", required=True, help="Output VCF")

    # ---- ClinVar 流程 ----
    p5 = subparsers.add_parser("filter-clinvar", help="Filter ClinVar VCF")
    p5.add_argument("--input", required=True, help="Input ClinVar VCF")
    p5.add_argument("--output", required=True, help="Output filtered VCF")
    p5.add_argument("--min-stars", type=int, default=1, help="Minimum stars")

    p6 = subparsers.add_parser("split-clinvar", help="Split ClinVar dataset")
    p6.add_argument("--input", required=True, help="Filtered ClinVar VCF")
    p6.add_argument("--scores", required=True, help="Scores Parquet/CSV")
    p6.add_argument("--output", required=True, help="Output directory")
    p6.add_argument("--test-size", type=float, default=0.8, help="Test proportion")

    p7 = subparsers.add_parser("calibrate", help="Calibrate threshold")
    p7.add_argument("--input", required=True, help="Training set CSV")
    p7.add_argument("--output", required=True, help="Output JSON")
    p7.add_argument("--specificity", type=float, default=0.95, help="Target specificity")

    p8 = subparsers.add_parser("benchmark", help="Benchmark on test set")
    p8.add_argument("--input", required=True, help="Test set CSV")
    p8.add_argument("--threshold", type=float, required=True, help="Decision threshold")
    p8.add_argument("--output", required=True, help="Output JSON")

    args = parser.parse_args()
    setup_logger(args.debug)

    commands = {
        "score-all": cmd_score_all,
        "to-vcf": cmd_to_vcf,
        "query": cmd_query,
        "annotate": cmd_annotate,
        "filter-clinvar": cmd_filter_clinvar,
        "split-clinvar": cmd_split_clinvar,
        "calibrate": cmd_calibrate,
        "benchmark": cmd_benchmark,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
