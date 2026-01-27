# EVOScore-2

MIT License

基于 ESM-2 的蛋白质突变评分与后处理工具。

## 功能

- **VCF 注释**: 使用预打分文件批量注释 VCF
- **ClinVar 基准测试**: 过滤、拆分、阈值校准、性能评估

## 安装

### 基础安装（推荐，用于预打分文件流程）

```bash
pip install -e .
```

### 完整安装（包含 ESM-2 模型打分功能）

```bash
pip install -e .[model]
```

## 命令

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

### 模型打分流程（需要 `.[model]` 依赖）

```bash
# 1. 全基因组突变评分 → Parquet
evoscore2 score-all --genome hg38.fa --gff annotation.gff3 --output scores.parquet

# 2. Parquet 转 VCF
evoscore2 to-vcf --scores scores.parquet --output scores.vcf.gz
```
