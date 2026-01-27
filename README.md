# EVOScore-2

MIT License

基于 ESM-2 的蛋白质突变评分与后处理工具。

## 命令

### 模型打分流程 (从基因组生成)

```bash
# 1. 全基因组突变评分 → Parquet
evoscore2 score-all --genome hg38.fa --gff annotation.gff3 --output scores.parquet

# 2. Parquet 转 VCF
evoscore2 to-vcf --scores scores.parquet --output scores.vcf.gz
```

### 基于预打分文件流程 (hg38_VESM_3B_scores.parquet.gzip)

```bash
# 查询单个位点分数
evoscore2 query --scores hg38_VESM_3B_scores.parquet.gzip --chrom 17 --pos 43044295 --ref A --alt G

# VCF 批量注释
evoscore2 annotate --scores hg38_VESM_3B_scores.parquet.gzip --input variants.vcf --output annotated.vcf
```

### ClinVar 基准测试流程

```bash
# 1. 清洗 ClinVar
evoscore2 filter-clinvar --input clinvar.vcf.gz --output clinvar_filtered.vcf.gz

# 2. 拆分数据集 (2:8)
evoscore2 split-clinvar --input clinvar_filtered.vcf.gz --scores scores.parquet --output split_results

# 3. 计算阈值 (训练集)
evoscore2 calibrate --input split_results/train.csv --output threshold.json --specificity 0.95

# 4. Benchmark (测试集)
evoscore2 benchmark --input split_results/test.csv --threshold -4.5 --output benchmark.json
```

## 安装

```bash
pip install -e .
```
