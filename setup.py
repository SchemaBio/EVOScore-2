from setuptools import setup, find_packages

setup(
    name="evoscore2",
    version="2.0.0",
    description="EVOScore-2: ESM-2 based protein mutation scoring for clinical variant interpretation",
    author="SchemaBio",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pysam>=0.21.0",
        "gffutils>=0.12",
        "biopython>=1.81",
        "scikit-learn>=1.3.0",
        "pytabix>=0.0.2",
        "loguru>=0.7.0",
        "tqdm>=4.65.0",
        "pyarrow>=12.0.0",
        "fastparquet>=2023.0.0",
        "typing_extensions>=4.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0"],
    },
    entry_points={
        "console_scripts": [
            "evoscore2=evoscore2.cli:main",
        ],
    },
)
