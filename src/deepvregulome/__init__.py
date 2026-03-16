"""
DeepVRegulome: DNABERT-based regulatory variant effect prediction.

Usage:
    from deepvregulome import DVR

    dvr = DVR(genome="/path/to/hg38.fa")
    result = dvr.score_variant("chr1", 3456782, "A", "C", models=["CTCFL", "SP1"])
    results = dvr.score_vcf("variants.vcf", model_type="TF", batch_size=100, gpus=[0,1,2,3])
"""

__version__ = "0.1.4"

from deepvregulome.registry import ModelRegistry


def __getattr__(name):
    if name == "DVR":
        from deepvregulome.dvr import DVR
        return DVR
    raise AttributeError(f"module 'deepvregulome' has no attribute {name}")


__all__ = ["DVR", "ModelRegistry", "__version__"]
