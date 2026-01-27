"""
VCF 生成模块

将蛋白质突变分数映射回基因组坐标，生成标准 VCF 文件
"""

import gzip
import pysam
import gffutils
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from loguru import logger
import numpy as np
import pandas as pd
from Bio.Seq import Seq


@dataclass
class VCFRecord:
    """VCF 记录"""
    chrom: str
    pos: int  # 1-based
    ref: str
    alt: str
    score: float
    info: Dict[str, str] = field(default_factory=dict)

    def to_vcf_line(self) -> str:
        """转换为 VCF 行"""
        info_str = ";".join([f"{k}={v}" for k, v in self.info.items()])
        return f"{self.chrom}\t{self.pos}\t.\t{self.ref}\t{self.alt}\t.\t.\t{info_str}"


class GenomeData:
    """
    基因组数据管理器

    负责加载参考基因组和基因注释文件
    """

    def __init__(
        self,
        genome_fasta: str,
        gff_annotation: str,
        disable_gff_updates: bool = True,
    ):
        """
        初始化基因组数据

        Args:
            genome_fasta: 参考基因组 FASTA 文件路径
            gff_annotation: GFF3 注释文件路径 (.gz 支持自动解压)
            disable_gff_updates: 禁止 GFF 自动更新
        """
        self.genome_fasta = genome_fasta
        self.gff_annotation = gff_annotation

        # 打开基因组文件
        self.fasta_file = pysam.FastaFile(genome_fasta)

        # 初始化 GFF 数据库
        gff_db_path = gff_annotation.replace(".gz", "").replace(".gff3", "") + ".db"
        if disable_gff_updates:
            gffutils.disable_feature_name_normalization()
            gffutils.disable_merge_records()

        logger.info(f"Loading GFF annotation from {gff_annotation}...")
        self.gff_db = gffutils.create_db(
            gff_annotation,
            dbfn=gff_db_path,
            disable_gff_updates=disable_gff_updates,
        )

        logger.info("GenomeData initialized")

    def get_transcript_cds(
        self,
        transcript_id: str,
    ) -> List[Dict[str, Any]]:
        """
        获取转录本的 CDS 区域列表

        Args:
            transcript_id: 转录本 ID (如 ENST 或 NM_ 开头)

        Returns:
            CDS 区域列表，每个区域包含 chrom, start, end, strand
        """
        # 尝试不同的 ID 格式
        feature = None

        # 尝试作为转录本特征查找
        try:
            feature = self.gff_db[transcript_id]
        except KeyError:
            # 尝试查找该转录本的所有 CDS
            pass

        if feature is None:
            # 从 GFF 中查找该转录本关联的特征
            for f in self.gff_db.features_of_type("CDS"):
                if hasattr(f, "transcript_id") and f.transcript_id == transcript_id:
                    feature = f
                    break

        if feature is None:
            raise ValueError(f"Transcript {transcript_id} not found in annotation")

        # 获取所有 CDS 子特征
        cds_regions = []
        for cds in self.gff_db.children(transcript_id, featuretype="CDS"):
            cds_regions.append({
                "chrom": cds.chrom,
                "start": cds.start,  # 1-based
                "end": cds.end,      # 1-based (exclusive in pysam)
                "strand": cds.strand,
                "phase": cds.frame,
            })

        # 按位置排序
        cds_regions.sort(key=lambda x: x["start"])

        return cds_regions

    def extract_cds_sequence(self, cds_regions: List[Dict]) -> str:
        """
        从基因组提取 CDS 序列

        Args:
            cds_regions: CDS 区域列表

        Returns:
            CDS 核苷酸序列
        """
        seq_parts = []

        for region in cds_regions:
            # pysam 使用 0-based，半开区间
            seq = self.fasta_file.fetch(
                region["chrom"],
                region["start"] - 1,  # 转换为 0-based
                region["end"],
            )
            seq_parts.append(seq)

        full_seq = "".join(seq_parts)

        # 检查链方向
        if cds_regions and cds_regions[0]["strand"] == "-":
            full_seq = str(Seq(full_seq).reverse_complement())

        return full_seq

    def cds_to_genomic_pos(
        self,
        cds_regions: List[Dict],
        cds_offset: int,
    ) -> Tuple[str, int]:
        """
        将 CDS 内的偏移转换为基因组坐标

        Args:
            cds_regions: CDS 区域列表
            cds_offset: CDS 内的碱基偏移 (0-based)

        Returns:
            (染色体, 基因组位置 1-based)
        """
        cumulative = 0

        for region in cds_regions:
            region_len = region["end"] - region["start"] + 1

            if cumulative + region_len > cds_offset:
                # 找到目标区域
                offset_in_region = cds_offset - cumulative

                if region["strand"] == "+":
                    genomic_pos = region["start"] + offset_in_region
                else:
                    # 负链：位置是反向的
                    genomic_pos = region["end"] - offset_in_region

                return region["chrom"], genomic_pos

            cumulative += region_len

        raise ValueError(f"Offset {cds_offset} out of CDS range")

    def translate_cds(self, cds_seq: str) -> str:
        """翻译 CDS 为蛋白质序列"""
        return str(Seq(cds_seq).translate())

    def get_all_transcripts(self) -> List[str]:
        """获取所有转录本 ID"""
        transcripts = set()
        for feature in self.gff_db.features_of_type("mRNA"):
            if feature.id:
                transcripts.add(feature.id)
        return sorted(transcripts)

    def close(self):
        """关闭文件句柄"""
        self.fasta_file.close()
        self.gff_db.close()


class VCFGenerator:
    """
    VCF 生成器

    将 ESM-2 突变分数映射到基因组坐标
    """

    # 氨基酸到密码子的映射（简化的标准遗传密码）
    CODON_TABLE = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }

    def __init__(
        self,
        genome_data: GenomeData,
        esm_model: "EVOScoreModel",  # 前向引用
    ):
        """
        初始化 VCF 生成器

        Args:
            genome_data: 基因组数据管理器
            esm_model: ESM-2 模型实例
        """
        self.genome_data = genome_data
        self.model = esm_model

        # 预计算所有密码子
        self._build_codon_lookup()

    def _build_codon_lookup(self):
        """构建密码子到氨基酸的查找表"""
        self.codon_to_aa = self.CODON_TABLE

        # 反向查找：氨基酸到密码子
        self.aa_to_codons: Dict[str, List[str]] = {}
        for codon, aa in self.codon_to_aa.items():
            if aa not in self.aa_to_codons:
                self.aa_to_codons[aa] = []
            self.aa_to_codons[aa].append(codon)

    def _get_codon_mutations(
        self,
        original_codon: str,
    ) -> List[Tuple[str, str, str]]:
        """
        获取密码子的所有可能突变

        Args:
            original_codon: 原始密码子 (3个碱基)

        Returns:
            [(突变后密码子, 原始氨基酸, 突变氨基酸), ...]
        """
        original_aa = self.codon_to_aa.get(original_codon.upper())
        if original_aa is None:
            return []

        mutations = []
        bases = ['A', 'T', 'C', 'G']

        for i in range(3):
            for alt_base in bases:
                if alt_base == original_codon[i]:
                    continue

                mut_codon = list(original_codon.upper())
                mut_codon[i] = alt_base
                mut_codon = "".join(mut_codon)

                mut_aa = self.codon_to_aa.get(mut_codon)
                if mut_aa and mut_aa != original_aa:
                    mutations.append((mut_codon, original_aa, mut_aa))

        return mutations

    def generate_vcf(
        self,
        transcript_id: str,
        protein_id: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[VCFRecord]:
        """
        生成转录本的 VCF 记录

        Args:
            transcript_id: 转录本 ID
            protein_id: 蛋白质 ID（用于记录）
            show_progress: 显示进度条

        Returns:
            VCF 记录列表
        """
        from tqdm import tqdm

        logger.info(f"Processing transcript {transcript_id}...")

        # 获取 CDS 区域
        cds_regions = self.genome_data.get_transcript_cds(transcript_id)
        if not cds_regions:
            raise ValueError(f"No CDS found for transcript {transcript_id}")

        # 提取 CDS 序列
        cds_seq = self.genome_data.extract_cds_sequence(cds_regions)
        if len(cds_seq) % 3 != 0:
            logger.warning(f"CDS sequence length {len(cds_seq)} is not divisible by 3")

        # 翻译验证
        protein_seq = self.genome_data.translate_cds(cds_seq)
        logger.info(f"Protein sequence length: {len(protein_seq)}")

        # 预计算 ESM-2 突变矩阵
        logger.info("Computing ESM-2 mutation matrix...")
        mutation_matrix = self.model.compute_mutation_matrix(
            protein_seq,
            batch_size=4,
            show_progress=show_progress,
        )

        # 遍历每个密码子
        vcf_records = []
        num_codons = len(cds_seq) // 3

        iterator = range(num_codons)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating VCF")

        for codon_idx in iterator:
            codon_start = codon_idx * 3
            original_codon = cds_seq[codon_start:codon_start + 3]
            original_aa = protein_seq[codon_idx]

            # 获取该位置的所有错义突变
            codon_mutations = self._get_codon_mutations(original_codon)

            for mut_codon, _, mut_aa in codon_mutations:
                # 获取 ESM-2 分数
                alt_idx = self.model.AA_TO_IDX.get(mut_aa)
                if alt_idx is None:
                    continue

                score = mutation_matrix[codon_idx, alt_idx]
                if np.isnan(score):
                    continue

                # 找出突变发生在哪个碱基
                for i in range(3):
                    if mut_codon[i] != original_codon[i]:
                        ref_base = original_codon[i]
                        alt_base = mut_codon[i]

                        # 转换到基因组坐标
                        cds_pos = codon_start + i
                        chrom, genomic_pos = self.genome_data.cds_to_genomic_pos(
                            cds_regions, cds_pos
                        )

                        # 创建 VCF 记录
                        record = VCFRecord(
                            chrom=chrom,
                            pos=genomic_pos,
                            ref=ref_base.upper(),
                            alt=alt_base.upper(),
                            score=float(score),
                            info={
                                "EVOScore": f"{score:.4f}",
                                "ProteinID": protein_id or transcript_id,
                                "ProteinPos": str(codon_idx + 1),  # 1-based
                                "RefAA": original_aa,
                                "AltAA": mut_aa,
                            },
                        )
                        vcf_records.append(record)
                        break

        logger.info(f"Generated {len(vcf_records)} VCF records")

        return vcf_records

    def generate_all_transcripts(
        self,
        protein_id: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[VCFRecord]:
        """
        对所有转录本进行突变评分 (全量加载到内存)

        Args:
            protein_id: 蛋白质 ID 前缀 (可选)
            show_progress: 显示进度条

        Returns:
            所有转录本的 VCF 记录
        """
        from tqdm import tqdm

        all_records = []
        transcripts = self.genome_data.get_all_transcripts()

        logger.info(f"Found {len(transcripts)} transcripts")

        iterator = transcripts
        if show_progress:
            iterator = tqdm(iterator, desc="Processing transcripts")

        for transcript_id in iterator:
            try:
                records = self.generate_vcf(
                    transcript_id=transcript_id,
                    protein_id=protein_id,
                    show_progress=False,
                )
                all_records.extend(records)
            except Exception as e:
                logger.warning(f"Failed to process {transcript_id}: {e}")

        logger.info(f"Total: {len(all_records)} records from {len(transcripts)} transcripts")
        return all_records

    def generate_all_to_parquet(
        self,
        output_path: str,
        protein_id: Optional[str] = None,
        batch_size: int = 100,
        show_progress: bool = True,
    ):
        """
        分批处理所有转录本，增量保存到 Parquet (支持断点续传)

        Args:
            output_path: 输出 Parquet 路径
            protein_id: 蛋白质 ID 前缀
            batch_size: 每批处理的转录本数量
            show_progress: 显示进度条
        """
        import pandas as pd
        from tqdm import tqdm
        import os

        transcripts = self.genome_data.get_all_transcripts()
        logger.info(f"Found {len(transcripts)} transcripts")

        # 检查是否需要断点续传
        processed_transcripts = set()
        if os.path.exists(output_path):
            try:
                df = pd.read_parquet(output_path)
                if "TranscriptID" in df.columns:
                    processed_transcripts = set(df["TranscriptID"].unique())
                    remaining = [t for t in transcripts if t not in processed_transcripts]
                    logger.info(f"Resuming: {len(processed_transcripts)} already processed, {len(remaining)} remaining")
                else:
                    remaining = transcripts
            except Exception:
                remaining = transcripts
        else:
            remaining = transcripts

        if not remaining:
            logger.info(f"Already complete: {output_path}")
            return

        # 分批处理
        all_records = []
        total_new = 0

        iterator = remaining
        if show_progress:
            iterator = tqdm(iterator, desc="Processing transcripts")

        for i, transcript_id in enumerate(iterator):
            try:
                records = self.generate_vcf(
                    transcript_id=transcript_id,
                    protein_id=protein_id,
                    show_progress=False,
                )
                for r in records:
                    r.info["TranscriptID"] = transcript_id
                all_records.extend(records)

                # 每 batch_size 个转录本保存一次
                if (i + 1) % batch_size == 0:
                    df = self.records_to_dataframe(all_records)
                    df.to_parquet(output_path, index=False, append=True if os.path.exists(output_path) else False)
                    total_new += len(all_records)
                    all_records = []
                    logger.debug(f"Saved batch {i + 1}/{len(remaining)}")

            except Exception as e:
                logger.warning(f"Failed to process {transcript_id}: {e}")

        # 保存剩余记录
        if all_records:
            df = self.records_to_dataframe(all_records)
            df.to_parquet(output_path, index=False, append=True if os.path.exists(output_path) else False)
            total_new += len(all_records)

        logger.info(f"Complete: {total_new} new records saved to {output_path}")

    def save_vcf(
        self,
        records: List[VCFRecord],
        output_path: str,
        compress: bool = True,
    ):
        """
        保存 VCF 文件

        Args:
            records: VCF 记录列表
            output_path: 输出路径
            compress: 是否压缩为 .gz
        """
        if compress and not output_path.endswith(".gz"):
            output_path += ".gz"

        # 写入文件
        with gzip.open(output_path, "wt") as f:
            # 写入 Header
            f.write("##fileformat=VCFv4.2\n")
            f.write('##INFO=<ID=EVOScore,Number=1,Type=Float,Description="ESM-2 based pathogenicity score. Log-Likelihood Ratio. Negative values indicate pathogenicity.">\n')
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

            # 写入记录
            for record in records:
                f.write(f"{record.chrom}\t{record.pos}\t.\t{record.ref}\t{record.alt}\t.\t.\tEVOScore={record.score:.4f}\n")

        logger.info(f"Saved {len(records)} records to {output_path}")

        # 创建索引
        if compress:
            pysam.tabix_index(output_path, preset="vcf", force=True)
            logger.info(f"Created index for {output_path}")

    def records_to_dataframe(self, records: List[VCFRecord]) -> pd.DataFrame:
        """将 VCF 记录转换为 DataFrame"""
        data = []
        for r in records:
            data.append({
                "CHROM": r.chrom,
                "POS": r.pos,
                "REF": r.ref,
                "ALT": r.alt,
                "score": r.score,
                "TranscriptID": r.info.get("TranscriptID", ""),
                "ProteinID": r.info.get("ProteinID", ""),
                "ProteinPos": r.info.get("ProteinPos", ""),
                "RefAA": r.info.get("RefAA", ""),
                "AltAA": r.info.get("AltAA", ""),
            })
        return pd.DataFrame(data)

    @staticmethod
    def dataframe_to_records(df: pd.DataFrame) -> List[VCFRecord]:
        """将 DataFrame 转换为 VCF 记录"""
        records = []
        for _, row in df.iterrows():
            record = VCFRecord(
                chrom=str(row["CHROM"]),
                pos=int(row["POS"]),
                ref=str(row["REF"]),
                alt=str(row["ALT"]),
                score=float(row["score"]),
                info={
                    "EVOScore": f"{row['score']:.4f}",
                    "ProteinID": str(row.get("ProteinID", "")),
                    "ProteinPos": str(row.get("ProteinPos", "")),
                    "RefAA": str(row.get("RefAA", "")),
                    "AltAA": str(row.get("AltAA", "")),
                },
            )
            records.append(record)
        return records

    @staticmethod
    def save_vcf_to_file(
        records: List[VCFRecord],
        file_handle,
        append: bool = False,
    ):
        """
        将 VCF 记录写入已打开的文件句柄（流式/分块写入友好）

        Args:
            records: VCF 记录列表
            file_handle: 已打开的文件句柄
            append: 是否为追加模式（追加时跳过表头）
        """
        # 写入表头（仅首次）
        if not append:
            file_handle.write("##fileformat=VCFv4.2\n")
            file_handle.write('##INFO=<ID=EVOScore,Number=1,Type=Float,Description="ESM-2 based pathogenicity score. Log-Likelihood Ratio. Negative values indicate pathogenicity.">\n')
            file_handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # 写入记录（只保留 EVOScore）
        for record in records:
            file_handle.write(f"{record.chrom}\t{record.pos}\t.\t{record.ref}\t{record.alt}\t.\t.\tEVOScore={record.score:.4f}\n")
