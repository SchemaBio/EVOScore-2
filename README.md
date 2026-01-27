# EVOScore-2

MIT License - 可自由用于商业用途

基于 ESM-2 (esm2_t33_650M_UR50D) 的蛋白质突变评分工具。

## 工作流

```
Genome + GFF3 → [score-all] → Parquet (含基因组坐标) → [to-vcf] → VCF.gz
                                                              ↓
ClinVar VCF → [filter-clinvar] → [split-clinvar] → train.csv / test.csv
                                                               ↓
                                          [calibrate] → threshold.json
                                                               ↓
                                          [benchmark] → benchmark.json
```

## 命令

### 1. 全基因组突变评分 → Parquet

```bash
# 使用本地预下载的模型
evoscore2 score-all \
    --genome hg38.fa \
    --gff MANE.GRCh38.v1.0.gff3.gz \
    --model-path ./esm2_t33_650M_UR50D \
    --output scores.parquet

# 或从 Hugging Face Hub 自动下载
evoscore2 score-all \
    --genome hg38.fa \
    --gff MANE.GRCh38.v1.0.gff3.gz \
    --output scores.parquet
```

**预下载模型（推荐离线使用）:**
```bash
git lfs install
git clone https://huggingface.co/facebook/esm2_t33_650M_UR50D ./esm2_t33_650M_UR50D
```

输出 Parquet 包含：`CHROM, POS, REF, ALT, score, TranscriptID, ProteinID, ProteinPos, RefAA, AltAA`

### 2. Parquet 转 VCF

```bash
evoscore2 to-vcf \
    --scores scores.parquet \
    --output scores.vcf.gz
```

### 3. 清洗 ClinVar

```bash
evoscore2 filter-clinvar \
    --input clinvar.vcf.gz \
    --output clinvar_filtered.vcf.gz \
    --min-stars 1
```

### 4. 拆分 ClinVar 数据集

```bash
evoscore2 split-clinvar \
    --input clinvar_filtered.vcf.gz \
    --scores scores.parquet \
    --output split_results \
    --test-size 0.8
```

### 5. 计算阈值 (训练集)

```bash
evoscore2 calibrate \
    --input split_results/train.csv \
    --output threshold.json \
    --specificity 0.95
```

### 6. Benchmark (测试集)

```bash
evoscore2 benchmark \
    --input split_results/test.csv \
    --threshold -4.5 \
    --output benchmark.json
```

## 安装

```bash
pip install -e .
```
