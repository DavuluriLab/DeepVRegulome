# Installation
1. (Skippable) Create dir you want to run this in. I prefer having this in a python virtual env. **Recommendation is to run this on python 3.11 due to package fits.**
```
mkdir -p DVR
cd DVR

python3.11 -m venv venv
source venv/bin/activate
```

2. Install deepvregulome wrapper via pip.
```
pip install deepvregulome
pip install deepvregulome --upgrade 
```

3. Further installation of pysam & tdqm is also required
```
pip install pysam tdqm
```

4. Install hg.38; for example using UCSC's hg38 build:
```
curl -L -o hg38.fa.gz http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
```

## Functionalities

If installed properly the DeepVRegulome, we should be able to run things onto the CLI:

```
deepvregulome -h                                                                

usage: deepvregulome [-h] {score,score-seq,score-vcf,list,search} ...

DeepVRegulome: Regulatory variant effect prediction

positional arguments:
  {score,score-seq,score-vcf,list,search}
    score               Score a single variant
    score-seq           Score from pre-extracted sequences
    score-vcf           Score variants from a VCF file
    list                List available models
    search              Search model names

options:
  -h, --help            show this help message and exit
```

The helper functions:
- list: outputs the list of available TF and histone marker models available on the [HuggingFace Repo](). 
- search: an exact string search function to search for available model names given a string

We have several scoring methods: 
- score: Scores a single variant given the chromosome, position, reference and alternate allele. Requires a direct hg38 path for location functionalities.
- score-seq: Scores two sequences given a reference and alternate sequence given the models, type and output file
- score-vcf: scores a vcf or vcf-like table (given its path).

 

## Running Variant Scoring on the CLI

Deepvregulome is a wrapper to call and download the given models into cache and runs them locally for inference. If given no output, variant scores will be outputted directly into stdout. If given `--output` variant predictions are given directly as a tsv.

To score a single variant:

```
deepvregulome score --chrom chr9 --pos 65385776 --ref G --alt A \
    --models ATF4 CTCFL SP1 --genome hg38.fa
```

To score for an entire VCF:
```
deepvregulome score-vcf patient.vcf --models CTCFL SP1 --genome hg38.fa -o results.tsv
```

