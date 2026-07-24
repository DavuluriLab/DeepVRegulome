"""
DeepVRegulome: DNABERT-based regulatory variant effect prediction.

Usage:
    from deepvregulome import DVR
    dvr = DVR(genome="hg38.fa")
    result = dvr.score_variant("chr1", 3456782, "A", "C", models=["CTCFL"])

    from deepvregulome.interpret import MotifAnalyzer
    analyzer = MotifAnalyzer()
    report = analyzer.analyze_variant(dvr, "CTCFL")
"""

__version__ = "0.4.0"

from deepvregulome.registry import ModelRegistry


def __getattr__(name):
    if name == "DVR":
        from deepvregulome.dvr import DVR
        return DVR
    if name in {"download_models", "Download_models"}:
        from deepvregulome.dvr import Download_models, download_models
        return download_models if name == "download_models" else Download_models
    if name == "MotifAnalyzer":
        from deepvregulome.interpret import MotifAnalyzer
        return MotifAnalyzer
    raise AttributeError(f"module 'deepvregulome' has no attribute {name}")


__all__ = [
    "DVR",
    "ModelRegistry",
    "MotifAnalyzer",
    "download_models",
    "Download_models",
    "__version__",
]
