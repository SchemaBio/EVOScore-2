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

### 转换Parquet到Vcf

```bash
evoscore2 to-vcf --scores hg38_VESM_3B_scores.parquet.gzip --output hg38_VESM_3B_scores.vcf.gz
```

### ClinVar 基准测试流程

```bash
# 1. 清洗 ClinVar
evoscore2 filter-clinvar --input clinvar.vcf.gz --output clinvar_filtered.vcf.gz

# 2. 拆分数据集 (2:8)
evoscore2 split-clinvar --input clinvar_filtered.vcf.gz --scores scores.parquet --output split_results

# 3. 计算阈值 (训练集)
evoscore2 calibrate-precision --input split_results/train.csv --output threshold.json --ppv 0.95 --npv 0.95

# 4. Benchmark (测试集)
evoscore2 benchmark --p-threshold -11.6 --b-threshold -8.5 --output benchmark.json
```
