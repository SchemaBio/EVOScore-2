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


def check_gpu_available() -> bool:
    """检测是否有可用的 GPU 和 cuDF"""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        # 检查 cuDF 是否可用
        import cudf
        return True
    except ImportError:
        return False


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
    """Parquet 转 VCF（支持多种后端：polars/pandas/cudf）"""
    import gzip

    # 标准化列名映射
    column_mapping = {
        "VESM (3B)": "score",
        "VESM(3B)": "score",
        "VESM_3B": "score",
        "VESM": "score",
        "EVOScore": "score",
    }

    # 确定文件类型
    is_gzip_parquet = args.scores.endswith(".parquet.gzip") or args.scores.endswith(".parquet.gz")
    is_parquet = args.scores.endswith(".parquet") or is_gzip_parquet

    # 确定使用的后端
    backend = getattr(args, 'backend', 'auto')

    if backend == 'auto':
        # 自动选择：优先 polars > cudf > pandas
        try:
            import polars
            backend = 'polars'
        except ImportError:
            if check_gpu_available():
                backend = 'cudf'
            else:
                backend = 'pandas'
        logger.info(f"Auto-selected backend: {backend}")

    # 根据后端选择处理函数
    if is_parquet:
        if backend == 'polars':
            _to_vcf_polars(args, column_mapping)
        elif backend == 'cudf':
            if not check_gpu_available():
                logger.warning("cuDF not available, falling back to pandas")
                _to_vcf_cpu(args, column_mapping, is_gzip_parquet)
            else:
                _to_vcf_gpu(args, column_mapping, is_gzip_parquet)
        else:  # pandas
            _to_vcf_cpu(args, column_mapping, is_gzip_parquet)
    else:
        # CSV 文件使用 pandas
        _to_vcf_csv(args, column_mapping)


def _to_vcf_cpu(args, column_mapping, is_gzip_parquet):
    """CPU 版本的 Parquet 转 VCF"""
    import gzip
    from pyarrow.parquet import ParquetFile
    from .vcf_generator import VCFGenerator

    # 尝试 gzip 方式
    if is_gzip_parquet:
        try:
            with gzip.open(args.scores, 'rb') as f:
                pf = ParquetFile(f)
        except gzip.BadGzipFile:
            pf = ParquetFile(args.scores)
    else:
        pf = ParquetFile(args.scores)

    # 获取列名并检查必需列
    schema = pf.schema_arrow
    col_names = [s.name for s in schema]
    required_cols = ["CHROM", "POS", "REF", "ALT"]

    # 检查是否有 score 列
    score_col = None
    for old_col in column_mapping.keys():
        if old_col in col_names:
            score_col = old_col
            break
    if score_col is None and "score" in col_names:
        score_col = "score"

    if not all(c in col_names for c in required_cols):
        raise ValueError(f"Missing required columns. Found: {col_names}")
    if score_col is None:
        raise ValueError(f"No score column found. Available: {col_names}")

    logger.info(f"Converting {pf.metadata.num_rows} records (CPU)...")

    generator = VCFGenerator(None, None)
    open_func = gzip.open if args.output.endswith(".gz") else open
    mode = "wt" if args.output.endswith(".gz") else "w"

    # 处理第一块
    first_batch = pf.read_row_group(0)
    first_df = first_batch.to_pandas()
    for old_col, new_col in column_mapping.items():
        if old_col in first_df.columns:
            first_df = first_df.rename(columns={old_col: new_col})

    records = generator.dataframe_to_records(first_df)
    with open_func(args.output, mode) as f:
        generator.save_vcf_to_file(records, f)

    # 分块处理剩余数据
    total_groups = pf.metadata.num_row_groups
    for i in range(1, total_groups):
        batch = pf.read_row_group(i)
        df = batch.to_pandas()
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        records = generator.dataframe_to_records(df)
        with open_func(args.output, "at" if args.output.endswith(".gz") else "a") as f:
            generator.save_vcf_to_file(records, f, append=True)

    logger.info(f"Converted to VCF: {args.output}")


def _to_vcf_polars(args, column_mapping):
    """Polars 高性能版本的 Parquet 转 VCF（流式分块处理）"""
    import gzip
    import polars as pl

    logger.info("Using Polars backend (high performance)...")

    # 使用 scan_parquet 懒加载，不立即读入内存
    lf = pl.scan_parquet(args.scores)
    col_names = lf.collect_schema().names()

    # 检查必需列
    required_cols = ["CHROM", "POS", "REF", "ALT"]
    if not all(c in col_names for c in required_cols):
        raise ValueError(f"Missing required columns. Found: {col_names}")

    # 找到 score 列
    score_col = None
    for old_col in column_mapping.keys():
        if old_col in col_names:
            score_col = old_col
            break
    if score_col is None and "score" in col_names:
        score_col = "score"
    if score_col is None:
        raise ValueError(f"No score column found. Available: {col_names}")

    # 重命名 score 列
    if score_col != "score":
        lf = lf.rename({score_col: "score"})

    # 获取总行数（用于进度显示）
    total_rows = pl.scan_parquet(args.scores).select(pl.len()).collect().item()
    logger.info(f"Converting {total_rows} records (Polars, streaming)...")

    # 写入文件
    open_func = gzip.open if args.output.endswith(".gz") else open
    mode = "wt" if args.output.endswith(".gz") else "w"

    # 分块大小
    chunk_size = 2_000_000  # 每次处理 200 万行

    with open_func(args.output, mode) as f:
        # 写入 VCF 头
        f.write("##fileformat=VCFv4.2\n")
        f.write('##INFO=<ID=EVOScore,Number=1,Type=Float,Description="ESM-2 based pathogenicity score">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # 分块处理
        for offset in range(0, total_rows, chunk_size):
            # 读取一块数据
            chunk_df = lf.slice(offset, chunk_size).collect()

            # 向量化生成 VCF 行
            vcf_lines = chunk_df.select(
                pl.concat_str([
                    pl.col("CHROM").cast(pl.Utf8),
                    pl.lit("\t"),
                    pl.col("POS").cast(pl.Utf8),
                    pl.lit("\t.\t"),
                    pl.col("REF").cast(pl.Utf8),
                    pl.lit("\t"),
                    pl.col("ALT").cast(pl.Utf8),
                    pl.lit("\t.\t.\tEVOScore="),
                    pl.col("score").round(4).cast(pl.Utf8),
                ]).alias("line")
            )

            # 写入文件
            for line in vcf_lines["line"]:
                f.write(line)
                f.write("\n")

            processed = min(offset + chunk_size, total_rows)
            logger.info(f"Progress: {processed}/{total_rows} ({100*processed//total_rows}%)")

    logger.info(f"Converted to VCF: {args.output}")


def _to_vcf_csv(args, column_mapping):
    """CSV 文件转 VCF（使用 pandas）"""
    import pandas as pd
    from .vcf_generator import VCFGenerator

    df = pd.read_csv(args.scores, sep="\t")
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})

    required_cols = ["CHROM", "POS", "REF", "ALT", "score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    generator = VCFGenerator(None, None)
    records = generator.dataframe_to_records(df)
    generator.save_vcf(records, args.output)
    logger.info(f"Converted {len(records)} records to VCF: {args.output}")


def _to_vcf_gpu(args, column_mapping, is_gzip_parquet):
    """GPU 加速版本的 Parquet 转 VCF（使用 cuDF）"""
    import gzip
    import cudf
    from .vcf_generator import VCFGenerator

    tmp_path = None
    scores_path = args.scores

    # 检测文件是否真的是 gzip 压缩的（检查魔数）
    def is_real_gzip(filepath):
        with open(filepath, 'rb') as f:
            magic = f.read(2)
            return magic == b'\x1f\x8b'

    # 如果扩展名暗示 gzip 且文件确实是 gzip 压缩的
    if is_gzip_parquet and is_real_gzip(args.scores):
        import tempfile
        import shutil
        logger.info("Decompressing gzip parquet for GPU processing...")
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = tmp.name
            with gzip.open(args.scores, 'rb') as f_in:
                shutil.copyfileobj(f_in, tmp)
        scores_path = tmp_path
    elif is_gzip_parquet:
        # 扩展名是 .gzip 但实际不是 gzip 压缩，直接读取
        logger.debug("File has .gzip extension but is not gzip compressed, reading directly")

    try:
        # 使用 cuDF 读取 parquet（GPU 加速）
        gdf = cudf.read_parquet(scores_path)
        col_names = list(gdf.columns)
        required_cols = ["CHROM", "POS", "REF", "ALT"]

        # 检查是否有 score 列
        score_col = None
        for old_col in column_mapping.keys():
            if old_col in col_names:
                score_col = old_col
                break
        if score_col is None and "score" in col_names:
            score_col = "score"

        if not all(c in col_names for c in required_cols):
            raise ValueError(f"Missing required columns. Found: {col_names}")
        if score_col is None:
            raise ValueError(f"No score column found. Available: {col_names}")

        # 在 GPU 上重命名列
        for old_col, new_col in column_mapping.items():
            if old_col in gdf.columns:
                gdf = gdf.rename(columns={old_col: new_col})

        logger.info(f"Converting {len(gdf)} records (GPU)...")

        # 分块转换并写入（避免一次性转换太多数据到 CPU）
        chunk_size = 1_000_000  # 每次处理 100 万行
        generator = VCFGenerator(None, None)
        open_func = gzip.open if args.output.endswith(".gz") else open
        mode = "wt" if args.output.endswith(".gz") else "w"

        total_rows = len(gdf)
        first_chunk = True

        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            # 在 GPU 上切片，然后转换到 CPU
            chunk_gdf = gdf.iloc[start:end]
            chunk_df = chunk_gdf.to_pandas()

            records = generator.dataframe_to_records(chunk_df)
            append_mode = "at" if args.output.endswith(".gz") else "a"

            with open_func(args.output, mode if first_chunk else append_mode) as f:
                generator.save_vcf_to_file(records, f, append=not first_chunk)

            first_chunk = False
            logger.debug(f"Processed {end}/{total_rows} records")

        logger.info(f"Converted to VCF: {args.output}")

    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==================== 基于预打分文件流程 ====================

def _load_scores(args):
    """加载分数文件，统一处理列名"""
    import pandas as pd
    import gzip

    if args.scores.endswith(".parquet.gzip") or args.scores.endswith(".parquet.gz"):
        try:
            with gzip.open(args.scores, 'rb') as f:
                df = pd.read_parquet(f)
        except gzip.BadGzipFile:
            df = pd.read_parquet(args.scores)
    elif args.scores.endswith(".parquet"):
        df = pd.read_parquet(args.scores)
    elif args.scores.endswith(".gz"):
        try:
            with gzip.open(args.scores, 'rb') as f:
                df = pd.read_parquet(f)
        except gzip.BadGzipFile:
            df = pd.read_csv(args.scores, sep="\t")
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


def _normalize_chrom(chrom):
    """标准化染色体名称（去除 chr 前缀）"""
    chrom = str(chrom)
    if chrom.startswith("chr"):
        return chrom[3:]
    return chrom


def _load_scores_for_variants(scores_path, query_keys):
    """
    高效加载指定变异位点的分数（使用 Polars 流式处理）

    Args:
        scores_path: 分数文件路径
        query_keys: 需要查询的位点集合 {(chrom, pos, ref, alt), ...}

    Returns:
        scores_dict: {(chrom, pos, ref, alt): score, ...}
    """
    import polars as pl

    # 标准化列名映射
    column_mapping = {
        "VESM (3B)": "score",
        "VESM(3B)": "score",
        "VESM_3B": "score",
        "VESM": "score",
        "EVOScore": "score",
    }

    # 使用 Polars 懒加载
    lf = pl.scan_parquet(scores_path)
    col_names = lf.collect_schema().names()

    # 找到 score 列
    score_col = None
    for old_col in column_mapping.keys():
        if old_col in col_names:
            score_col = old_col
            break
    if score_col is None and "score" in col_names:
        score_col = "score"

    if score_col and score_col != "score":
        lf = lf.rename({score_col: "score"})

    # 标准化查询键的 CHROM（去除 chr 前缀）
    normalized_query = {}
    for k in query_keys:
        norm_chrom = _normalize_chrom(k[0])
        norm_key = (norm_chrom, k[1], k[2], k[3])
        normalized_query[norm_key] = k  # 映射回原始键

    # 构建查询 DataFrame（使用标准化的 CHROM）
    query_data = {
        "CHROM_NORM": [k[0] for k in normalized_query.keys()],
        "POS": [k[1] for k in normalized_query.keys()],
        "REF": [k[2] for k in normalized_query.keys()],
        "ALT": [k[3] for k in normalized_query.keys()],
    }
    query_df = pl.DataFrame(query_data)

    # 显示样本数据用于调试
    sample = lf.head(3).collect()
    logger.debug(f"Scores file sample CHROM: {sample['CHROM'].to_list()}")
    sample_query = list(query_keys)[:3]
    logger.debug(f"Query sample CHROM: {[k[0] for k in sample_query]}")

    # 分块处理
    chunk_size = 5_000_000
    total_rows = pl.scan_parquet(scores_path).select(pl.len()).collect().item()

    scores_dict = {}

    for offset in range(0, total_rows, chunk_size):
        chunk = lf.slice(offset, chunk_size).collect()

        # 标准化 CHROM 列（去除 chr 前缀）
        chunk = chunk.with_columns([
            pl.col("CHROM").cast(pl.Utf8).str.replace("^chr", "").alias("CHROM_NORM"),
            pl.col("REF").cast(pl.Utf8),
            pl.col("ALT").cast(pl.Utf8),
        ])

        # 内连接找到匹配的行
        matched = chunk.join(
            query_df,
            on=["CHROM_NORM", "POS", "REF", "ALT"],
            how="inner"
        )

        # 添加到结果字典（使用原始的查询键）
        for row in matched.iter_rows(named=True):
            norm_key = (row["CHROM_NORM"], int(row["POS"]), str(row["REF"]), str(row["ALT"]))
            if norm_key in normalized_query:
                orig_key = normalized_query[norm_key]
                scores_dict[orig_key] = row["score"]

        processed = min(offset + chunk_size, total_rows)
        if offset == 0 or processed == total_rows or len(scores_dict) > 0:
            logger.info(f"Progress: {processed}/{total_rows}, found {len(scores_dict)} matches")

    return scores_dict


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


def _count_clinsig(records):
    """统计临床显著性分类数量"""
    counts = {
        "Pathogenic": 0,
        "Likely_pathogenic": 0,
        "Benign": 0,
        "Likely_benign": 0,
    }
    for r in records:
        if r.clinical_significance in counts:
            counts[r.clinical_significance] += 1
    return counts


def cmd_split_clinvar(args):
    """拆分 ClinVar 数据集（内存优化版）"""
    import pandas as pd
    from .clinvar_benchmark import ClinVarFilter, ClinVarSplitter

    records = ClinVarFilter.load_filtered_vcf(args.input)
    logger.info(f"Loaded {len(records)} ClinVar records")

    # 构建需要查询的位点集合
    query_keys = set()
    for r in records:
        query_keys.add((str(r.chrom), int(r.pos), str(r.ref), str(r.alt)))

    logger.info(f"Looking up {len(query_keys)} variants in scores file...")

    # 使用 Polars 高效查询（只加载需要的数据）
    scores_dict = _load_scores_for_variants(args.scores, query_keys)

    logger.info(f"Found {len(scores_dict)} matching scores")

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

    # 输出各组统计
    train_counts = _count_clinsig(train_records)
    test_counts = _count_clinsig(test_records)

    logger.info("=" * 50)
    logger.info("ClinVar Split Statistics:")
    logger.info("=" * 50)
    logger.info(f"Train set ({len(train_records)} records):")
    logger.info(f"  Pathogenic (P):      {train_counts['Pathogenic']:>6}")
    logger.info(f"  Likely_pathogenic:   {train_counts['Likely_pathogenic']:>6}")
    logger.info(f"  Benign (B):          {train_counts['Benign']:>6}")
    logger.info(f"  Likely_benign:       {train_counts['Likely_benign']:>6}")
    logger.info(f"  Total P:             {train_counts['Pathogenic'] + train_counts['Likely_pathogenic']:>6}")
    logger.info(f"  Total B:             {train_counts['Benign'] + train_counts['Likely_benign']:>6}")
    logger.info("")
    logger.info(f"Test set ({len(test_records)} records):")
    logger.info(f"  Pathogenic (P):      {test_counts['Pathogenic']:>6}")
    logger.info(f"  Likely_pathogenic:   {test_counts['Likely_pathogenic']:>6}")
    logger.info(f"  Benign (B):          {test_counts['Benign']:>6}")
    logger.info(f"  Likely_benign:       {test_counts['Likely_benign']:>6}")
    logger.info(f"  Total P:             {test_counts['Pathogenic'] + test_counts['Likely_pathogenic']:>6}")
    logger.info(f"  Total B:             {test_counts['Benign'] + test_counts['Likely_benign']:>6}")
    logger.info("=" * 50)

    # 保存统计信息
    stats = {
        "train": {
            "n_records": len(train_records),
            **train_counts,
            "Total_P": train_counts["Pathogenic"] + train_counts["Likely_pathogenic"],
            "Total_B": train_counts["Benign"] + train_counts["Likely_benign"],
        },
        "test": {
            "n_records": len(test_records),
            **test_counts,
            "Total_P": test_counts["Pathogenic"] + test_counts["Likely_pathogenic"],
            "Total_B": test_counts["Benign"] + test_counts["Likely_benign"],
        },
    }
    import json
    with open(f"{args.output}/split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def cmd_calibrate(args):
    """计算阈值（二分类）"""
    import pandas as pd
    from .clinvar_benchmark import ThresholdCalibrator
    import json

    train_df = pd.read_csv(args.input)
    # 取反分数：ESM-2 分数越负越致病，取反后越高越致病
    X_train = -train_df["score"].values
    y_train = (train_df["CLNSIG"].isin(["Pathogenic", "Likely_pathogenic"])).astype(int).values

    threshold, metrics = ThresholdCalibrator.calibrate_by_specificity(
        X_train, y_train, target_specificity=args.specificity
    )

    # 保存时将阈值转回原始尺度（取反）
    result = {
        "threshold": -threshold,  # 转回原始尺度
        "threshold_negated": threshold,  # 取反后的阈值（用于内部计算）
        "specificity": args.specificity,
        **metrics
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Calibrated threshold: {-threshold:.4f} (sensitivity: {metrics.get('sensitivity', 'N/A')})")
    logger.info(f"Saved to {args.output}")


def cmd_calibrate_three(args):
    """计算三分类阈值（P_threshold 和 B_threshold）

    原始分数：越负越致病（P），越正越良性（B）
    分类规则：
      - score < P_threshold → P（致病）
      - score > B_threshold → B（良性）
      - B_threshold < score < P_threshold → VUS
    """
    import pandas as pd
    from .clinvar_benchmark import ThresholdCalibrator
    import json

    train_df = pd.read_csv(args.input)
    # 取反分数：ESM-2 分数越负越致病，取反后越高越致病
    X_train = -train_df["score"].values
    y_train = (train_df["CLNSIG"].isin(["Pathogenic", "Likely_pathogenic"])).astype(int).values

    p_threshold, b_threshold, metrics = ThresholdCalibrator.calibrate_three_class(
        X_train, y_train,
        target_specificity=args.specificity,
        target_specificity_b=args.specificity_b,
    )

    result = {
        "p_threshold": float(p_threshold),
        "b_threshold": float(b_threshold),
        "specificity": args.specificity,
        "specificity_b": args.specificity_b,
        **metrics
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("=" * 50)
    logger.info("Three-class Thresholds (original scale):")
    logger.info(f"  P_threshold: {p_threshold:.4f}  (score < {p_threshold:.4f} → P)")
    logger.info(f"  B_threshold: {b_threshold:.4f}  (score > {b_threshold:.4f} → B)")
    logger.info(f"  VUS: {b_threshold:.4f} < score < {p_threshold:.4f}")
    logger.info("=" * 50)
    logger.info(f"Saved to {args.output}")


def cmd_benchmark(args):
    """Benchmark 评估（支持二分类和三分类）"""
    import pandas as pd
    from .clinvar_benchmark import BenchmarkEvaluator
    import json

    test_df = pd.read_csv(args.input)
    # 取反分数：与 calibrate 保持一致
    X_test = -test_df["score"].values
    y_test = (test_df["CLNSIG"].isin(["Pathogenic", "Likely_pathogenic"])).astype(int).values

    # 三分类模式
    if args.p_threshold is not None and args.b_threshold is not None:
        # 传入原始尺度的阈值
        # P_threshold: 负值（越负越致病）
        # B_threshold: 负值（但比 P_threshold 大/更正）
        metrics = BenchmarkEvaluator.evaluate_three_class(
            X_test, y_test, args.p_threshold, args.b_threshold
        )
        result = {
            "p_threshold": args.p_threshold,
            "b_threshold": args.b_threshold,
            **metrics
        }
    else:
        # 二分类模式（向后兼容）
        threshold_negated = -args.threshold
        metrics = BenchmarkEvaluator.evaluate(X_test, y_test, threshold_negated)
        result = {"threshold": args.threshold, **metrics}

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Benchmark complete. Results saved to {args.output}")


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
    p2.add_argument("--backend", default="auto", choices=["auto", "polars", "pandas", "cudf"],
                    help="Processing backend: auto (default), polars (recommended), pandas, cudf (GPU)")

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

    p7 = subparsers.add_parser("calibrate", help="Calibrate threshold (binary)")
    p7.add_argument("--input", required=True, help="Training set CSV")
    p7.add_argument("--output", required=True, help="Output JSON")
    p7.add_argument("--specificity", type=float, default=0.95, help="Target specificity")

    p7b = subparsers.add_parser("calibrate-three", help="Calibrate three-class thresholds (P/B/VUS)")
    p7b.add_argument("--input", required=True, help="Training set CSV")
    p7b.add_argument("--output", required=True, help="Output JSON")
    p7b.add_argument("--specificity", type=float, default=0.95, help="Target specificity for P threshold")
    p7b.add_argument("--specificity-b", type=float, default=0.95, dest="specificity_b", help="Target specificity for B threshold")

    p8 = subparsers.add_parser("benchmark", help="Benchmark on test set")
    p8.add_argument("--input", required=True, help="Test set CSV")
    p8.add_argument("--threshold", type=float, help="Decision threshold (binary, mutually exclusive with --p-threshold)")
    p8.add_argument("--p-threshold", type=float, help="P threshold (three-class)")
    p8.add_argument("--b-threshold", type=float, help="B threshold (three-class)")
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
        "calibrate-three": cmd_calibrate_three,
        "benchmark": cmd_benchmark,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
