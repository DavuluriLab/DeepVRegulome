# DeepVRegulome

<p align="center">
  <img src="assets/flowchart.png" alt="DeepVRegulome Pipeline" width="800">
</p>

<p align="center">
  <a href="https://pypi.org/project/deepvregulome/"><img src="https://img.shields.io/pypi/v/deepvregulome?color=blue" alt="PyPI"></a>
  [![PyPI Downloads](https://static.pepy.tech/personalized-badge/deepvregulome?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/deepvregulome)
  <a href="https://huggingface.co/duttaprat/DeepVRegulome"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow" alt="HuggingFace"></a>
  <a href="https://huggingface.co/spaces/duttaprat/DeepVRegulome"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-blue" alt="HuggingFace Space"></a>
  <a href="https://arxiv.org/abs/2511.09026"><img src="https://img.shields.io/badge/arXiv-2511.09026-b31b1b" alt="arXiv"></a>
  <a href="https://deepvregulome.streamlit.app"><img src="https://img.shields.io/badge/demo-Streamlit-ff4b4b" alt="Streamlit"></a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img src="https://img.shields.io/badge/license-CC--BY--NC--4.0-green" alt="License"></a>
</p>

**DeepVRegulome** is a DNABERT-based deep-learning framework for predicting the functional impact of short genomic variants on the human regulome. It provides **464 fine-tuned models** (458 transcription factors + 4 histone modifications + 2 splice sites) trained on ENCODE ChIP-seq and GENCODE data, covering splice-site and transcription factor binding site disruption analysis.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Splice-Site Scoring](#splice-site-scoring)
- [Python API Reference](#python-api-reference)
- [External Data Requirements](#external-data-requirements)
- [Repository Structure](#repository-structure)
- [Tutorial Notebooks](#tutorial-notebooks)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Model Checkpoints](#model-checkpoints)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## Installation

### Requirements

- **Python ≥ 3.11**
- **GPU recommended** — DNABERT inference runs on CPU but is significantly faster on CUDA-enabled GPUs


### 1. Install the Python package

```bash
pip install deepvregulome
```
This installs the core package with PyTorch, Transformers, and HuggingFace Hub.

> **Upgrading:** We release updates frequently with new features and bug fixes. To get the latest version:
> ```bash
> pip install deepvregulome --upgrade
> ```
> Check your installed version: `python -c "import deepvregulome; print(deepvregulome.__version__)"`

### 2. Install optional dependencies

DeepVRegulome has optional extras depending on your use case:

```bash
# For variant scoring from genomic coordinates (requires samtools/htslib)
pip install deepvregulome[genome]        # installs pysam

# For VCF file processing
pip install deepvregulome[vcf]           # installs cyvcf2

# For visualization (attention maps, motif logos)
pip install deepvregulome[viz]           # installs matplotlib, seaborn

# For motif interpretation (logo plots, statistical tests)
pip install deepvregulome[interpret]     # installs logomaker, scipy

# Install everything
pip install deepvregulome[all]
```
### Recommended: Use a dedicated conda environment

If you're on a shared server (e.g., HPC, JupyterHub), we strongly recommend creating
a dedicated conda environment to avoid dependency conflicts:

```bash
# Create and activate environment
conda create -n dvr python=3.11 -y
conda activate dvr

# Install deepvregulome with all optional dependencies
pip install deepvregulome[all]

# If using JupyterHub, register as a selectable kernel
pip install jupyterlab ipykernel
python -m ipykernel install --user --name dvr --display-name "DVR (Python 3.11)"
```

Then select **"DVR (Python 3.11)"** as your kernel in JupyterHub (Kernel → Change Kernel).

---

## External Data Requirements

DeepVRegulome requires two external data files that are **not** included in the package. You must download these before running variant-level analyses.

### Human Reference Genome (hg38)

Required for extracting flanking sequences around variant positions.

```bash
# Option 1: UCSC hg38
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
samtools faidx hg38.fa

# Option 2: GENCODE GRCh38 primary assembly
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.primary_assembly.genome.fa.gz
gunzip GRCh38.primary_assembly.genome.fa.gz
samtools faidx GRCh38.primary_assembly.genome.fa
```

> **Note:** The `.fai` index file is required. Run `samtools faidx` on your FASTA file if the index does not exist.

### JASPAR Motif Database (optional, for motif analysis)

Required only if you want to run motif-level interpretation and TF binding site overlap analysis.

```bash
# Download JASPAR 2024 vertebrate motifs in MEME format
wget https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt
```

---

## Quick Start

### Score a single variant

```python
from deepvregulome import DVR

# Initialize with path to your reference genome FASTA
dvr = DVR(
    genome="/path/to/hg38.fa",
    model_dir="/path/to/preferred_models",
)

# Score a variant against specific TF models
result = dvr.score_variant(
    chrom="chr1",
    pos=3456782,
    ref="A",
    alt="C",
    models=["CTCFL", "SP1", "MYC"],
)
print(result)
```

### Download models into a preferred directory

```python
from deepvregulome import download_models

download_models(
    models=["CTCFL", "SP1"],
    model_dir="/path/to/preferred_models",
)
```

DeepVRegulome now checks `model_dir` first for local checkpoints and falls back to the Hugging Face cache only when a requested model is not present there.

Output is a pandas DataFrame with columns: `chrom`, `pos`, `ref`, `alt`, `model`, `type`, `prob_ref`, `prob_alt`, `log_odds_change`, `disrupted`.

### Score variants from a VCF file

```python
results = dvr.score_vcf(
    "/path/to/patient.vcf",
    models=["CTCFL", "SP1", "GATA3"],
    batch_size=100,
    gpus=[0, 1]         # multi-GPU support
)
results.head()
```
###  Batch Scoring from DataFrame (`score_variants`)

Auto-detects column names: `chrom/pos/ref/alt` or `CHROM/start/end/REF/ALT`.

```python
import pandas as pd

# VCF-style DataFrame
# variant_df = pd.DataFrame({
#     "chrom": ["chr21", "chr1", "chr17", "chr12", "chr7"],
#     "pos":   [10448027, 3456782, 7674220, 25245350, 55181378],
#     "ref":   ["C", "A", "C", "G", "T"],
#     "alt":   ["T", "C", "T", "A", "C"],
# })
# Create a DataFrame of variants to score
variant_df = pd.read_csv("test_vcf.tsv", sep="\t")

results = dvr.score_variants(
    variant_df,
    models=["CTCFL", "SP1", "ATF4"],
    batch_size=5,
)
```

### Score pre-extracted sequences

```python
# TF/histone models use 301bp sequences
result = dvr.score_sequence(
    ref_seq="ATCGATCG...",   # 301bp reference sequence
    alt_seq="ATCGTTCG...",   # 301bp alternate sequence
    models=["CTCFL"]
)
```

---

## Splice-Site Scoring

DeepVRegulome includes two splice-site models fine-tuned on GENCODE v41 exon-intron junctions using 90bp input sequences:

| Model | Description |
|---|---|
| `splice_acceptor` | Predicts acceptor (3') splice site recognition |
| `splice_donor` | Predicts donor (5') splice site recognition |

### List splice models

```python
from deepvregulome import DVR

dvr = DVR()

# List only splice-site models
splice_models = dvr.list_models(model_type="SPLICE")
for m in splice_models:
    print(f"{m.name}  (input: {m.input_length}bp)")
# splice_acceptor  (input: 90bp)
# splice_donor     (input: 90bp)
```

### Score a splice-site variant

```python
# Splice-site models use 90bp sequences centered on the exon-intron boundary
result = dvr.score_sequence(
    ref_seq="ATCG...90bp...GATC",   # 90bp reference sequence
    alt_seq="ATCG...90bp...GTTC",   # 90bp with variant
    models=["splice_acceptor", "splice_donor"],
)
print(result)
```

### Use without the DVR wrapper

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load splice acceptor model directly from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(
    "duttaprat/DeepVRegulome", subfolder="models/splice_acceptor"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "duttaprat/DeepVRegulome", subfolder="models/splice_acceptor"
)
```

> **Note:** Splice-site models use **90bp** input sequences, not 301bp like TF/histone models.
> When using `score_variant()` or `score_vcf()`, ensure the variant falls within a
> known exon-intron junction region for meaningful predictions.

---

## Python API Reference

### `DVR` class

| Method | Description |
|---|---|
| `DVR(genome=..., model_dir=...)` | Initialize with reference genome FASTA path and an optional preferred model directory |
| `dvr.score_variant(chrom, pos, ref, alt, models)` | Score a single variant by genomic coordinates |
| `dvr.score_variants(df, models)` | Batch-score a DataFrame of variants (columns: chrom, pos, ref, alt) |
| `dvr.score_vcf(vcf_path, models)` | Score all variants in a VCF file |
| `dvr.score_sequence(ref_seq, alt_seq, models)` | Score pre-extracted REF/ALT sequences (301bp for TF/histone, 90bp for splice) |
| `dvr.list_models(model_type=None)` | List available models; filter by `"TF"`, `"HISTONE"`, or `"SPLICE"` |
| `dvr.search_models(query)` | Search models by name (e.g., `"ZNF"`, `"GATA"`, `"splice"`) |

### Scoring parameters

All scoring methods accept these optional arguments:

| Parameter | Default | Description |
|---|---|---|
| `models` | required | List of model names (e.g., `["CTCFL", "SP1"]` or `["splice_acceptor"]`) |
| `model_type` | `None` | Score all models of a type: `"TF"` (458 models), `"HISTONE"` (4 models), or `"SPLICE"` (2 models) |
| `batch_size` | `32` | Batch size for GPU inference |
| `gpus` | `[0]` | List of GPU device IDs for parallel inference |
| `return_attention` | `False` | Extract DNABERT attention weights for interpretability |
| `coordinate_system` | `"1-based"` | VCF standard is 1-based; set to `"0-based"` if needed |

### CLI

```bash
# Score a single variant from the command line
deepvregulome score --chrom chr1 --pos 3456782 --ref A --alt C \
    --models CTCFL SP1 --genome /path/to/hg38.fa

# Score from VCF
deepvregulome score-vcf --vcf patient.vcf --models CTCFL SP1 MYC \
    --genome /path/to/hg38.fa --batch-size 100 --gpus 0 1
```

---

## Repository Structure

```
DeepVRegulome/
├── src/deepvregulome/              # Python package (pip install deepvregulome)
│   ├── __init__.py                 #   Public API: DVR class
│   ├── dvr.py                      #   Main scoring engine
│   ├── registry.py                 #   Model registry (464 models + metadata)
│   ├── utils.py                    #   k-mer tokenization, sequence extraction
│   └── cli.py                      #   Command-line interface
├── notebooks/                      # Tutorial notebooks (see below)
│   ├── 01_quickstart.ipynb         #   Getting started with DVR
│   ├── 02_vcf_scoring.ipynb        #   Batch VCF analysis pipeline
│   ├── 03_attention_motifs.ipynb   #   Attention visualization & motif analysis
│   └── 04_clinical_pipeline.ipynb  #   End-to-end clinical variant analysis
├── streamlit_app/                  # Interactive web dashboard
│   └── app_variant_clinical_dashboard.py
├── assets/                         # Figures for README
│   └── flowchart.png
├── pyproject.toml                  # Package metadata & dependencies
├── LICENSE
└── README.md
```

---

## Tutorial Notebooks

The `notebooks/` folder contains Jupyter notebooks that walk through common use cases:

| Notebook | Description |
|---|---|
| `01_quickstart.ipynb` | Install, load models, score your first variant |
| `02_vcf_scoring.ipynb` | Parse a VCF, batch-score variants, filter candidates |
| `03_attention_motifs.ipynb` | Extract DNABERT attention weights, plot motif disruption |
| `04_clinical_pipeline.ipynb` | Full pipeline: VCF → TFBS intersection → scoring → candidate ranking |

To run the notebooks:

```bash
pip install deepvregulome[all] jupyterlab
jupyter lab notebooks/
```

---

## Streamlit Dashboard

An interactive dashboard for exploring variant predictions and clinical stratification is available:

- **Live demo:** [https://deepvregulome.streamlit.app](https://deepvregulome.streamlit.app)
- **HuggingFace Space (try predictions live):** [https://huggingface.co/spaces/duttaprat/DeepVRegulome](https://huggingface.co/spaces/duttaprat/DeepVRegulome)


---

## Model Checkpoints

All 464 fine-tuned DNABERT models (458 TFs + 4 histone marks + 2 splice sites) are hosted on HuggingFace:

**[https://huggingface.co/duttaprat/DeepVRegulome](https://huggingface.co/duttaprat/DeepVRegulome)**

Models can still be fetched automatically through the Hugging Face cache, but you can now pre-download them into your own directory with `download_models(..., model_dir=...)` and point `DVR(..., model_dir=...)` or any scoring call at that location.

For direct access without the `deepvregulome` package:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load a TF model
tokenizer = AutoTokenizer.from_pretrained(
    "duttaprat/DeepVRegulome", subfolder="models/CTCFL"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "duttaprat/DeepVRegulome", subfolder="models/CTCFL"
)

# Load a splice-site model
tokenizer = AutoTokenizer.from_pretrained(
    "duttaprat/DeepVRegulome", subfolder="models/splice_acceptor"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "duttaprat/DeepVRegulome", subfolder="models/splice_acceptor"
)
```

---

## Roadmap

**Current capabilities (v0.4.0):**
- Single-variant and batch VCF scoring with 464 models (458 TF + 4 histone + 2 splice)
- Splice-site disruption scoring (acceptor + donor models)
- Multi-GPU inference support
- Attention-based interpretability
- CLI and Python API

**In development:**
- JASPAR motif enrichment integration
- Expanded model zoo: additional cell types and epigenomic marks
- Conda package

**Planned:**
- REST API for web-based scoring
- Integration with ClinVar and gnomAD annotation

---

## Citation

If you use DeepVRegulome in your research, please cite:

```bibtex
@article{dutta2025deepvregulome,
  title={DeepVRegulome: DNABERT-based deep-learning framework for predicting
         the functional impact of short genomic variants on the human regulome},
  author={Dutta, Pratik and Obusan, Matthew and Sathian, Rekha and Chao, Max
          and Surana, Pallavi and Papineni, Nimisha and Ji, Yanrong
          and Zhou, Zhihan and Liu, Han and Yurovsky, Alisa
          and Davuluri, Ramana V},
  journal={arXiv preprint arXiv:2511.09026},
  year={2025},
  url={https://arxiv.org/abs/2511.09026}
}
```

---

## License

CC-BY-NC-4.0. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b><a href="https://davulurilab.github.io/">Davuluri Lab</a></b> · Department of Biomedical Informatics · Stony Brook University<br>
  <a href="https://github.com/DavuluriLab">GitHub</a> ·
</p>
