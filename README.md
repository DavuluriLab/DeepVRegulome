# DeepVRegulome
![DeepVRegulome Pipeline](assets/flowchart.png)

DeepVRegulome is an end‑to‑end framework for predicting the functional impact of small somatic variants in non‑coding regulatory regions (splice sites and transcription‑factor‑binding sites) using fine‑tuned DNABERT models.

---

## ✨ Key Features

- ✅ DNABERT-based classifiers for:
  - Splice sites (acceptor, donor)
  - ~700 TFBS models
- ✅ Region-aware scoring of somatic variants using Δp and log₂ odds
- ✅ Batch processing with multiprocessing and BED/VCF support
- ✅ Interactive Streamlit dashboard with:
  - Variant tables, plots, and survival analysis
  - Attention score visualizations

---

📁 Repository Structure
```
DeepVRegulome/
├── .devcontainer/
├── .streamlit/
├── data/
│   └── Brain/
├── figures/                         # Exported visualizations (e.g. attention maps)
│   └── attention/
│       ├── CTCFL/
│       └── ZNF384/
├── notebooks/                      # Jupyter notebooks for key pipeline steps
│   ├── 01_parse_and_merge_vcfs.ipynb            # Merge and parse VCFs
│   ├── 02_tfbs_intersection.ipynb               # Intersect VCF with TFBS BEDs
│   ├── 03_dnabert_input_generation.ipynb        # Generate sequences for DNABERT
│   ├── 04_scoring_candidate_variants.ipynb      # Compute Δp / logOR & rank variants
│   └── 05_tfbs_attention_motif_visualization.ipynb  # Plot attention scores & motifs
├── scripts/                       # Shell scripts for batch inference
│   ├── run_prediction_tfbs.sh                 # Predict with TFBS models
│   └── run_prediction_splice_acceptor.sh      # Predict with acceptor models
├── src/
│   └── deepvregulome/             # Core Python modules
│       ├── __init__.py
│       ├── dnabert_data_generation.py         # Wild/mutated seq generation
│       ├── intersect.py                       # BED/VCF overlap engine
│       ├── vcf_loader.py                      # VCF parsing utilities
│       └── config.yaml                        # Centralized path config
├── streamlit_app/
│   └── app_variant_clinical_dashboard.py      # Live clinical dashboard
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore

```
## 🧪 Installation
```bash
git clone https://github.com/DavuluriLab//DeepVRegulome.git
cd DeepVRegulome
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```



## ⚙️ Typical Pipeline Flow
| Step | Description | Location |
|------|-------------|----------|
| 1️⃣ | Parse + merge somatic VCFs | `01_parse_and_merge_vcfs.ipynb` |
| 2️⃣ | Intersect variants with TFBS BEDs | `02_tfbs_intersection.ipynb` |
| 3️⃣ | Generate ref/mutated k-mers for DNABERT | `03_dnabert_input_generation.ipynb` |
| 4️⃣ | Predict with DNABERT models | `scripts/run_prediction_tfbs.sh` |
| 5️⃣ | Compute Δp, find candidate variants | `04_scoring_candidate_variants.ipynb` |
| 6️⃣ | Visualize attention scores and motifs | `05_tfbs_attention_motif_visualization.ipynb` |
| 7️⃣ | Browse results interactively | `streamlit_app/app_variant_clinical_dashboard.py` |


## 📊 Example Outputs
  * Candidate variant count by TFBS
  * DNABERT attention heatmaps
  * High-impact motif shifts due to mutations
  * Kaplan–Meier plots for clinical stratification

See figures/attention/ for examples like CTCFL.


## 🌐 Live Demo

An interactive instance of the DeepVRegulome dashboard is hosted here:
➡️ [https://deepvregulome.streamlit.app/](https://deepvregulome.streamlit.app/)
The deployed app allows you to explore model performance metrics and variant-effect predictions without the need to install any software locally.

## 🧬 Model Checkpoints
Full DNABERT fine-tuned weights (acceptor, donor, and 700 TFBS models) will be deposited in Zenodo and made publicly available immediately upon journal acceptance.
In the meantime, researchers may request access by emailing pratik.dutta@stonybrook.edu and ramana.davuluri@stonybrookmedicine.edu  with a brief statement of intended use.

## Citation
If you use DeepVRegulome in your research, please cite:
```
@misc{dutta2025deepvregulomednabertbaseddeeplearningframework,
      title={DeepVRegulome: DNABERT-based deep-learning framework for predicting the functional impact of short genomic variants on the human regulome}, 
      author={Pratik Dutta and Matthew Obusan and Rekha Sathian and Max Chao and Pallavi Surana and Nimisha Papineni and Yanrong Ji and Zhihan Zhou and Han Liu and Alisa Yurovsky and Ramana V Davuluri},
      year={2025},
      eprint={2511.09026},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN},
      url={https://arxiv.org/abs/2511.09026}, 
}
```


## 🧬 Model Checkpoints
462 fine-tuned DNABERT models (458 TFs + 4 histone marks) are available on HuggingFace:
➡️ [duttaprat/DeepVRegulome](https://huggingface.co/duttaprat/DeepVRegulome)
```python
pip install deepvregulome

from deepvregulome import DVR
dvr = DVR(genome="hg38.fa")
result = dvr.score_variant("chr1", 3456782, "A", "C", models=["CTCFL", "SP1"])
```

**After editing both files on GitHub**, rebuild and re-upload to PyPI:
```bash
cd /vast/home/pdutta/Github/dvr_package
git pull   # get the README changes
python -m build
~/.local/bin/twine upload dist/* --skip-existing
```

The `--skip-existing` flag is important — it won't fail on the already-uploaded files and will only upload the new version if you bump the version number. Actually, for a new release you'll need to change `version = "0.1.1"` in `pyproject.toml` first, since PyPI won't accept the same version twice.


MIT. See [LICENSE](LICENSE) for details.
