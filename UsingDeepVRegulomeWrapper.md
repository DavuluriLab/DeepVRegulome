# Using DeepVRegulome Wrapper

## Installation

DeepVRegulome is currently most reliable in a Python 3.11 virtual environment.

```bash
mkdir -p DVR
cd DVR

python3.11 -m venv venv
source venv/bin/activate
```

Install the package and the common genome-scoring dependency:

```bash
pip install --upgrade pip
pip install deepvregulome
pip install pysam tqdm
```

Download an hg38 FASTA for coordinate-based scoring:

```bash
curl -L -o hg38.fa.gz http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
```

If needed, create the FASTA index:

```bash
samtools faidx hg38.fa
```

## CLI Overview

After installation, the CLI should be available:

```bash
deepvregulome -h
```

Current top-level commands include:

- `score`: score a single variant from genomic coordinates
- `score-seq`: score a pair of pre-extracted REF/ALT sequences
- `score-vcf`: score a VCF file
- `list`: show available models
- `search`: search model names
- `download-models`: download model files into a preferred local directory

## Preferred Model Directory Workflow

DeepVRegulome now supports a preferred local model directory.

Recommended workflow:

1. Download models into a directory you control.
2. Point scoring commands at that directory with `--model-dir`.
3. Use the Hugging Face cache only as a fallback when a requested model is not present locally.

The wrapper now checks these local layouts first:

- `MODEL_DIR/models/<MODEL_NAME>`
- `MODEL_DIR/<MODEL_NAME>`
- a direct model folder whose name matches the model, such as `/path/to/CTCFL`

This behavior was verified in the Python 3.11 development venv.

## Download Models

To download specific models into a preferred directory:

```bash
deepvregulome download-models \
    --models ATF4 CTCFL SP1 \
    --model-dir /path/to/dvr_models
```

To download all models of a type:

```bash
deepvregulome download-models \
    --type TF \
    --model-dir /path/to/dvr_models
```

Optional:

- `--cache-dir /path/to/hf_cache` to choose the fallback Hugging Face cache location
- `--force-download` to refresh files even if they already exist

## Running Variant Scoring on the CLI

If you omit `--output`, scores print to stdout. If you provide `--output`, results are written as a TSV.

Score a single variant while preferring a local model directory:

```bash
deepvregulome score \
    --chrom chr9 \
    --pos 65385776 \
    --ref G \
    --alt A \
    --models ATF4 CTCFL SP1 \
    --genome hg38.fa \
    --model-dir /path/to/dvr_models
```

Score a VCF:

```bash
deepvregulome score-vcf \
    patient.vcf \
    --models CTCFL SP1 \
    --genome hg38.fa \
    --model-dir /path/to/dvr_models \
    -o results.tsv
```

Score two sequences directly:

```bash
deepvregulome score-seq \
    --ref ACTG... \
    --alt ACTA... \
    --models CTCFL \
    --model-dir /path/to/dvr_models
```

## Python API

You can use the same preferred-directory workflow from Python:

```python
from deepvregulome import DVR, Download_models

Download_models(
    models=["CTCFL", "SP1"],
    model_dir="/path/to/dvr_models",
)

dvr = DVR(
    genome="hg38.fa",
    model_dir="/path/to/dvr_models",
)

result = dvr.score_variant(
    chrom="chr9",
    pos=65385776,
    ref="G",
    alt="A",
    models=["CTCFL"],
)
```

You can also override the model directory per call:

```python
result = dvr.score_vcf(
    "patient.vcf",
    models=["CTCFL", "SP1"],
    model_dir="/path/to/dvr_models",
)
```

## Helper Commands

- `deepvregulome list` shows available TF and histone models
- `deepvregulome search ZNF` searches for model names containing a string

## Notes

- Python 3.11 is the recommended environment for this wrapper.
- Use `model_dir` when you want models stored in a project-owned location.
- `cache_dir` remains available as a fallback cache path.
