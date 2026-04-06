# DeepVRegulome Manual

## Overview

DeepVRegulome is a package and command-line interface for scoring the regulatory effect of short genomic variants with fine-tuned DNABERT models. Each model represents a specific regulatory target, usually a transcription factor or histone mark. For a given variant, DeepVRegulome compares the reference sequence and the alternate sequence and reports how the model's predicted binding probability changes between the two.

At a high level, the package does three things:

1. Resolves which model or set of models should be used.
2. Builds or accepts REF and ALT sequences.
3. Runs both sequences through the selected DNABERT model checkpoints and returns a scored table.

The package supports both Python usage and a CLI wrapper. The CLI is useful for quick scoring jobs, while the Python API is better for pipelines, notebooks, and larger workflows.

## Core Concepts

### What a model represents

Each DeepVRegulome model is a separate fine-tuned checkpoint associated with a regulatory target such as `CTCFL`, `SP1`, or a histone mark. When you score a variant against a model, you are asking:

"How much does this alternate allele change the model's predicted probability that the local sequence is a binding site or regulatory signal for this target?"

### What gets scored

There are two ways to score:

- Sequence-based scoring: you already have a 301 bp reference sequence and a 301 bp alternate sequence.
- Variant-based scoring: you provide `chrom`, `pos`, `ref`, and `alt`, and DeepVRegulome extracts the flanking sequence around the variant from a reference genome.

### Why both REF and ALT are needed

DeepVRegulome is designed around comparative inference. The key output is not only the reference probability or the alternate probability alone, but the difference between them. This comparison is what makes the package useful for variant prioritization.

## Installation and Environment

### Recommended Python version

Python 3.11 is strongly recommended for this package. In practice, it is the most reliable environment for the current dependency stack.

### Recommended environment setup

Use a dedicated virtual environment:

```bash
mkdir -p DVR
cd DVR

python3.11 -m venv venv
source venv/bin/activate
```

Install the package:

```bash
pip install --upgrade pip
pip install deepvregulome
pip install pysam tqdm
```

For coordinate-based scoring, you also need an hg38 FASTA:

```bash
curl -L -o hg38.fa.gz http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
samtools faidx hg38.fa
```

## Model Storage Strategy

### Recommended workflow

The recommended workflow is to download models into a directory you control and then point DeepVRegulome at that directory during scoring.

This is preferred over relying entirely on the default Hugging Face cache.

Recommended sequence:

1. Create a stable directory for model storage.
2. Download the model checkpoints into that directory.
3. Use `--model-dir` in the CLI or `model_dir=` in Python.
4. Let the Hugging Face cache act only as a fallback.

### Why a preferred directory is better than cache-only usage

Using a known directory has several advantages:

- You know exactly where the model files live.
- It is easier to manage storage on shared systems or servers.
- It is easier to back up or move the models.
- It makes reproducibility better across sessions and machines.
- It avoids confusion when multiple Python environments share or compete for cache state.

### What happens when `model_dir` is provided

If `model_dir` is provided, DeepVRegulome checks local model paths first. The current local lookup supports:

- `MODEL_DIR/models/<MODEL_NAME>`
- `MODEL_DIR/<MODEL_NAME>`
- a direct model folder whose name matches the model, such as `/path/to/CTCFL`

If the requested model is found locally, DeepVRegulome loads from that local path. If not, it falls back to the standard Hugging Face model resolution path and cache behavior.

### What happens when `model_dir` is not provided

If you do not provide `model_dir`, DeepVRegulome behaves like the legacy flow:

- it does not perform a preferred local-directory lookup
- it resolves the model from Hugging Face
- it uses `cache_dir` if one is supplied
- otherwise it uses the default Hugging Face cache location

This means the new `model_dir` feature is additive and backward-compatible.

## CLI Quickstart

Check the CLI:

```bash
deepvregulome -h
```

Top-level commands:

- `score`
- `score-seq`
- `score-vcf`
- `list`
- `search`
- `download-models`

## Downloading Models

### Download specific models

```bash
deepvregulome download-models \
    --models ATF4 CTCFL SP1 \
    --model-dir /path/to/dvr_models
```

### Download all models of a type

```bash
deepvregulome download-models \
    --type TF \
    --model-dir /path/to/dvr_models
```

### Optional download flags

- `--cache-dir /path/to/hf_cache`
- `--force-download`

### When to use `download-models`

Use `download-models` if:

- you want predictable storage
- you are working on a server or cluster
- you want to prepare an environment before running scoring jobs
- you do not want runtime downloads during an analysis

## CLI Commands in Detail

### `deepvregulome list`

This command lists the available models shipped in the package registry.

Example:

```bash
deepvregulome list
deepvregulome list --type TF
```

Use this when you need to see valid model names or limit the search space to transcription factors or histone models.

### `deepvregulome search`

This command searches model names by substring.

Example:

```bash
deepvregulome search ZNF
deepvregulome search CTC
```

Use this when you know part of a model name but not the exact identifier.

### `deepvregulome score`

This command scores one genomic variant by extracting flanking sequence from a reference genome.

Required inputs:

- `--chrom`
- `--pos`
- `--ref`
- `--alt`
- `--genome`
- either `--models` or `--type`

Example:

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

When to use it:

- you have a single variant to inspect
- you want fast manual checking from a shell
- you are debugging a pipeline or validating a candidate site

### `deepvregulome score-vcf`

This command scores many variants from a VCF file.

Example:

```bash
deepvregulome score-vcf \
    patient.vcf \
    --models CTCFL SP1 \
    --genome hg38.fa \
    --model-dir /path/to/dvr_models \
    -o results.tsv
```

When to use it:

- you have many variants in VCF form
- you want a tabular batch result
- you are processing a patient or cohort file

### `deepvregulome score-seq`

This command scores a reference sequence and an alternate sequence directly, without genome lookup.

Example:

```bash
deepvregulome score-seq \
    --ref ACTG... \
    --alt ACTA... \
    --models CTCFL \
    --model-dir /path/to/dvr_models
```

When to use it:

- you have already generated the sequences
- you want to avoid genome dependency
- you are testing a sequence design or perturbation directly

## Python API

The Python API exposes the same logic as the CLI, with more flexibility.

### Import pattern

```python
from deepvregulome import DVR, Download_models, download_models
```

Both `Download_models` and `download_models` are available. They refer to the same function.

### Download models from Python

```python
from deepvregulome import Download_models

Download_models(
    models=["CTCFL", "SP1"],
    model_dir="/path/to/dvr_models",
)
```

### Initialize a DVR instance

```python
from deepvregulome import DVR

dvr = DVR(
    genome="hg38.fa",
    model_dir="/path/to/dvr_models",
    cache_dir="/path/to/hf_cache",
)
```

Key initialization parameters:

- `genome`: path to the FASTA used for genomic coordinate scoring
- `model_dir`: preferred local directory for model lookup
- `cache_dir`: fallback cache location for Hugging Face files
- `device`: usually `cpu` or `cuda`
- `coordinate_system`: optional manual override for `1-based` or `0-based`

### `score_variant`

```python
result = dvr.score_variant(
    chrom="chr9",
    pos=65385776,
    ref="G",
    alt="A",
    models=["CTCFL"],
)
```

Use this when you have a single genomic location and want automatic sequence extraction.

Important behavior:

- the package performs coordinate sanity checking
- the reference genome is required
- the output is a pandas DataFrame

### `score_vcf`

```python
result = dvr.score_vcf(
    "patient.vcf",
    models=["CTCFL", "SP1"],
    model_dir="/path/to/dvr_models",
)
```

Use this when processing a full VCF from Python rather than the shell.

### `score_variants`

```python
import pandas as pd

variants = pd.DataFrame([
    {"chrom": "chr9", "pos": 65385776, "ref": "G", "alt": "A"},
    {"chrom": "chr1", "pos": 3456782, "ref": "A", "alt": "C"},
])

result = dvr.score_variants(
    variants,
    models=["CTCFL", "SP1"],
)
```

This is useful when your variant data is already in memory as a DataFrame.

The function can auto-detect common column names such as:

- `chrom`, `chr`, `#chrom`
- `pos`, `start`, `position`
- `ref`, `reference`, `ref_allele`
- `alt`, `alternative`, `alt_allele`

### `score_sequence`

```python
result = dvr.score_sequence(
    ref_seq="ACTG...",
    alt_seq="ACTA...",
    models=["CTCFL"],
)
```

This is the cleanest entry point when you already have the sequences and do not want the package to extract them from the genome.

### `download_models` as an instance method

```python
dvr.download_models(
    models=["CTCFL"],
    model_dir="/path/to/dvr_models",
)
```

This is a convenience wrapper around the package-level download helper.

## Understanding the Output Table

The primary output is a tab-separated or DataFrame-style table like:

```text
chrom      pos ref alt model type  prob_ref  prob_alt  log_odds_ratio  score_change
 chr9 65385776   G   A CTCFL   TF    0.3238  0.870793          -3.815      0.476318
```

### Column definitions

- `chrom`: chromosome name
- `pos`: genomic position of the variant
- `ref`: reference allele
- `alt`: alternate allele
- `model`: model used for scoring
- `type`: model class, such as `TF` or `HISTONE`
- `prob_ref`: model probability for the reference sequence
- `prob_alt`: model probability for the alternate sequence
- `log_odds_ratio`: the log-odds difference between reference and alternate prediction
- `score_change`: the probability change term used by the package scoring routine

### Interpreting `prob_ref` and `prob_alt`

These values describe how strongly the model believes each sequence resembles a regulatory site for that model.

In simple terms:

- high `prob_ref` and low `prob_alt` suggests loss of binding or disruption
- low `prob_ref` and high `prob_alt` suggests gain of binding
- similar probabilities suggest limited effect for that specific model

### Interpreting `log_odds_ratio`

This is often the most useful ranking value.

Large absolute values mean larger predicted differences between REF and ALT. The sign tells you the direction of change under the package's scoring convention, while the magnitude tells you how strong the shift is.

### Interpreting `score_change`

This gives an additional effect-size style metric based on the difference between alternate and reference probabilities.

In practice:

- use `log_odds_ratio` to rank strong effects
- inspect `prob_ref` and `prob_alt` to understand the biological direction
- use `score_change` as supporting context

## Practical Recommendations

### Strong recommendations

- use Python 3.11
- use a dedicated virtual environment
- download models into a known `model_dir`
- keep the reference genome indexed with `samtools faidx`
- test one variant interactively before launching large jobs

### Why local model storage should be the default practice

If you rely only on the Hugging Face cache, several practical issues can appear:

- cache location may not be obvious
- cache can be shared across unrelated projects
- cleanup becomes harder
- storage policies on clusters may remove or relocate cache data
- runtime downloads can interrupt reproducibility

Using a known `model_dir` avoids most of these issues and makes troubleshooting much easier.

## Common Failure Modes

### Model not found

Cause:

- typo in the model name
- local model directory does not contain the requested checkpoint

What to do:

- run `deepvregulome search <text>`
- run `deepvregulome list`
- confirm the expected directory structure under `model_dir`

### Genome errors during `score` or `score-vcf`

Cause:

- missing FASTA
- FASTA not indexed
- chromosome naming mismatch such as `1` versus `chr1`

What to do:

- confirm the FASTA path
- run `samtools faidx hg38.fa`
- check whether the VCF and FASTA use compatible chromosome names

### Dependency issues

Cause:

- incompatible Python version
- mismatched NumPy, Torch, or Transformers installation

What to do:

- use Python 3.11
- create a fresh virtual environment
- reinstall the package cleanly

## Relationship to the Wrapper Quickstart

The quickstart in [docs/UsingDeepVRegulomeWrapper.md](/Users/Max_1/Documents/code/test_realms/DeepVRegulome/docs/UsingDeepVRegulomeWrapper.md) is meant for fast setup and immediate use.

This manual is the long-form companion document. Use it when you need:

- a fuller explanation of the workflow
- command-by-command behavior
- guidance on model storage decisions
- interpretation help for the output table
