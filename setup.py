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
        "numpy>=1.16.0",
        "pandas>=1.1.0",
        "pysam>=0.16.0",
        "biopython>=1.78",
        "scikit-learn>=0.23.0",
        "pytabix>=0.0.2",
        "loguru>=0.6.0",
        "tqdm>=4.60.0",
        "pyarrow>=5.0.0",
        "fastparquet>=0.5.0",
        "typing_extensions>=4.0.0",
        "aiocontextvars>=0.2.0",  # Python 3.6 backport for loguru
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0"],
        "model": ["torch>=2.0.0", "transformers>=4.30.0", "gffutils>=0.12"],
    },
    entry_points={
        "console_scripts": [
            "evoscore2=evoscore2.cli:main",
        ],
    },
)
