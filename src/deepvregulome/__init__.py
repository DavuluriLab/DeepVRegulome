"""
DeepVRegulome: DNABERT-based regulatory variant effect prediction.

Usage:
    from deepvregulome import DVR

    dvr = DVR(genome="/path/to/hg38.fa")

    # Score a single variant (auto-detects coordinate system)
    result = dvr.score_variant("chr1", 3456782, "A", "TA", models=["CTCF", "SP1"])

    # Score from sequences directly
    result = dvr.score_sequence(ref_seq, alt_seq, models=["CTCF"])

    # Score a VCF file
    results = dvr.score_vcf("variants.vcf", models=["CTCF", "SP1", "MYC"])
"""

__version__ = "0.1.3"

from deepvregulome.registry import ModelRegistry


def __getattr__(name):
    """Lazy import DVR so registry works even without torch installed."""
    if name == "DVR":
        from deepvregulome.dvr import DVR
        return DVR
    raise AttributeError(f"module 'deepvregulome' has no attribute {name}")


__all__ = ["DVR", "ModelRegistry", "__version__"]
