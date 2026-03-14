"""
DVR: Main interface for DeepVRegulome variant effect prediction.

Supports three input modes:
    1. Variant coordinates (chr, pos, ref, alt) + reference genome
    2. Pre-extracted REF/ALT sequences
    3. VCF file (batch scoring)

Outputs:
    - Binding probability (REF and ALT)
    - Log-odds ratio change
    - Disruption flag
    - Attention score change (optional — DVR's unique interpretability feature)
"""

import math
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from deepvregulome.registry import ModelInfo, ModelRegistry
from deepvregulome.utils import to_kmer, extract_variant_sequences


class DVR:
    """
    DeepVRegulome: Score regulatory variant effects using fine-tuned DNABERT models.

    Usage:
        # --- Mode 1: Variant coordinates (needs reference genome) ---
        dvr = DVR(genome="/path/to/hg38.fa")
        result = dvr.score_variant(
            chrom="chr1", pos=3456782, ref="A", alt="TA",
            models=["CTCFL", "SP1", "MYC"],
        )

        # --- Mode 2: Pre-extracted sequences ---
        dvr = DVR()
        result = dvr.score_sequence(
            ref_seq="ATCG...",   # 301bp reference
            alt_seq="ATCG...",   # 301bp with variant
            models=["CTCFL"],
        )

        # --- Mode 3: VCF batch scoring ---
        dvr = DVR(genome="/path/to/hg38.fa")
        results = dvr.score_vcf("variants.vcf", models=["CTCFL", "SP1"])

        # --- With attention scores (DVR's unique feature) ---
        result = dvr.score_variant(
            "chr1", 3456782, "A", "C",
            models=["CTCFL"],
            return_attention=True,
        )
    """

    def __init__(
        self,
        genome: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        disruption_threshold: float = 2.0,
    ):
        """
        Args:
            genome: Path to reference genome FASTA file (hg38).
                    Required for score_variant() and score_vcf().
                    Install pysam: pip install deepvregulome[genome]
            device: "cuda" or "cpu" (auto-detected if None)
            cache_dir: Directory for caching downloaded HF models
            disruption_threshold: Log-odds change threshold for disruption flag
        """
        self.registry = ModelRegistry()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.disruption_threshold = disruption_threshold

        # Lazy-loaded genome
        self._genome_path = genome
        self._genome = None

        # Lazy-loaded model cache: name -> (model, tokenizer)
        self._loaded: Dict[str, tuple] = {}

    @property
    def genome(self):
        """Lazy-load the reference genome."""
        if self._genome is None:
            if self._genome_path is None:
                raise ValueError(
                    "Reference genome not provided. Either:\n"
                    "  1. Pass genome='/path/to/hg38.fa' to DVR()\n"
                    "  2. Use dvr.score_sequence() with pre-extracted sequences"
                )
            try:
                import pysam
            except ImportError:
                raise ImportError(
                    "pysam is required for genome-based scoring. "
                    "Install with: pip install deepvregulome[genome]"
                )
            self._genome = pysam.FastaFile(self._genome_path)
        return self._genome

    def _load_model(self, name: str):
        """Load a model from HuggingFace Hub (cached after first load)."""
        if name in self._loaded:
            return self._loaded[name]

        info = self.registry.get(name)
        print(f"  Loading {info.name} from {info.hf_repo}/{info.subfolder}...")

        tokenizer = AutoTokenizer.from_pretrained(
            info.hf_repo,
            subfolder=info.subfolder,
            cache_dir=self.cache_dir,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            info.hf_repo,
            subfolder=info.subfolder,
            cache_dir=self.cache_dir,
            output_attentions=True,  # Always enable for optional attention extraction
        )
        model = model.to(self.device)
        model.eval()

        self._loaded[name] = (model, tokenizer)
        return model, tokenizer

    def _predict(
        self,
        model,
        tokenizer,
        sequence: str,
        return_attention: bool = False,
    ) -> dict:
        """
        Run inference on a single sequence.

        Returns dict with:
            - prob: binding probability (float)
            - attention: attention tensor [num_layers, num_heads, seq_len, seq_len] (optional)
        """
        kmer_seq = to_kmer(sequence, k=6)
        inputs = tokenizer(
            kmer_seq,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=return_attention)

        probs = torch.softmax(outputs.logits, dim=-1)
        result = {"prob": probs[0][1].item()}

        if return_attention and outputs.attentions:
            # Stack attention from all layers: [layers, heads, seq, seq]
            attn = torch.stack(outputs.attentions).squeeze(1)  # remove batch dim
            result["attention"] = attn.cpu().numpy()

        return result

    def _compute_attention_change(
        self,
        attn_ref: np.ndarray,
        attn_alt: np.ndarray,
    ) -> dict:
        """
        Compute attention disruption metrics between REF and ALT.

        This is DVR's unique interpretability feature. Unlike AlphaGenome's
        brute-force ISM, attention change directly reveals which sequence
        positions the model focuses on differently after a mutation.

        Returns:
            attention_score_change: L2 norm of attention difference (scalar)
            max_attention_shift: Maximum single-position attention change
            disrupted_layers: Number of layers with significant attention change
        """
        # Average across heads for each layer
        ref_avg = attn_ref.mean(axis=1)  # [layers, seq, seq]
        alt_avg = attn_alt.mean(axis=1)

        # Per-layer attention difference
        diff = alt_avg - ref_avg  # [layers, seq, seq]

        # Global attention score change (L2 norm across all layers)
        attention_score_change = float(np.sqrt((diff ** 2).sum()))

        # Max single-position shift
        max_attention_shift = float(np.abs(diff).max())

        # Count layers with significant change (> 0.01 mean absolute diff)
        layer_changes = np.abs(diff).mean(axis=(1, 2))  # [layers]
        disrupted_layers = int((layer_changes > 0.01).sum())

        return {
            "attention_score_change": round(attention_score_change, 6),
            "max_attention_shift": round(max_attention_shift, 6),
            "disrupted_layers": disrupted_layers,
            "total_layers": len(layer_changes),
        }

    # ================================================================
    # Public API: score_sequence
    # ================================================================
    def score_sequence(
        self,
        ref_seq: str,
        alt_seq: str,
        models: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        return_attention: bool = False,
    ) -> pd.DataFrame:
        """
        Score a variant given pre-extracted REF and ALT sequences.

        Args:
            ref_seq: Reference DNA sequence (typically 301bp)
            alt_seq: Alternative (mutant) DNA sequence
            models: List of model names (e.g., ["CTCFL", "SP1"])
            model_type: Score all models of this type ("TF" or "HISTONE")
            return_attention: If True, include attention-based metrics

        Returns:
            DataFrame with columns:
                model, type, prob_ref, prob_alt, log_odds_ref, log_odds_alt,
                log_odds_change, disrupted
                [+ attention_score_change, max_attention_shift if return_attention]
        """
        model_names = self._resolve_models(models, model_type)
        results = []

        for name in model_names:
            model, tokenizer = self._load_model(name)
            info = self.registry.get(name)

            ref_out = self._predict(model, tokenizer, ref_seq, return_attention)
            alt_out = self._predict(model, tokenizer, alt_seq, return_attention)

            p_ref = ref_out["prob"]
            p_alt = alt_out["prob"]

            eps = 1e-7
            lo_ref = math.log((p_ref + eps) / (1 - p_ref + eps))
            lo_alt = math.log((p_alt + eps) / (1 - p_alt + eps))
            lo_change = lo_alt - lo_ref

            row = {
                "model": name,
                "type": info.model_type,
                "prob_ref": round(p_ref, 6),
                "prob_alt": round(p_alt, 6),
                "log_odds_ref": round(lo_ref, 4),
                "log_odds_alt": round(lo_alt, 4),
                "log_odds_change": round(lo_change, 4),
                "abs_log_odds_change": round(abs(lo_change), 4),
                "disrupted": abs(lo_change) > self.disruption_threshold,
            }

            # Attention-based interpretability
            if return_attention and "attention" in ref_out and "attention" in alt_out:
                attn_metrics = self._compute_attention_change(
                    ref_out["attention"], alt_out["attention"]
                )
                row.update(attn_metrics)

            results.append(row)

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("abs_log_odds_change", ascending=False)
        return df.reset_index(drop=True)

    # ================================================================
    # Public API: score_variant
    # ================================================================
    def score_variant(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        models: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        flank: int = 150,
        return_attention: bool = False,
    ) -> pd.DataFrame:
        """
        Score a variant by genomic coordinates.

        Args:
            chrom: Chromosome (e.g., "chr1")
            pos: 1-based position
            ref: Reference allele (e.g., "A")
            alt: Alternate allele (e.g., "TA")
            models: List of model names to score
            model_type: Score all models of this type
            flank: Flanking bases on each side (default: 150 → 301bp for SNV)
            return_attention: Include attention disruption metrics

        Returns:
            DataFrame with scoring results (same columns as score_sequence,
            plus chrom, pos, ref, alt columns)

        Example:
            dvr = DVR(genome="/path/to/hg38.fa")
            result = dvr.score_variant("chr1", 3456782, "A", "TA",
                                       models=["CTCFL", "SP1", "MYC"])
        """
        # Extract sequences from genome
        ref_seq, alt_seq = extract_variant_sequences(
            self.genome, chrom, pos, ref, alt, flank=flank
        )

        # Score
        df = self.score_sequence(
            ref_seq, alt_seq,
            models=models,
            model_type=model_type,
            return_attention=return_attention,
        )

        # Add variant info columns
        df.insert(0, "chrom", chrom)
        df.insert(1, "pos", pos)
        df.insert(2, "ref", ref)
        df.insert(3, "alt", alt)

        return df

    # ================================================================
    # Public API: score_vcf
    # ================================================================
    def score_vcf(
        self,
        vcf_path: str,
        models: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        flank: int = 150,
        return_attention: bool = False,
        max_variants: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Score all variants in a VCF file.

        Args:
            vcf_path: Path to VCF file (.vcf or .vcf.gz)
            models: List of model names
            model_type: Score all models of this type
            flank: Flanking bases
            return_attention: Include attention metrics
            max_variants: Stop after this many variants (None = all)

        Returns:
            DataFrame with one row per (variant × model) combination

        Example:
            results = dvr.score_vcf("patient_variants.vcf",
                                     models=["CTCFL", "SP1", "MYC"])
            disrupted = results[results["disrupted"]]
        """
        try:
            import cyvcf2
        except ImportError:
            # Fallback to simple text parsing
            return self._score_vcf_simple(
                vcf_path, models, model_type, flank, return_attention, max_variants
            )

        all_results = []
        reader = cyvcf2.VCF(vcf_path)

        for i, variant in enumerate(reader):
            if max_variants and i >= max_variants:
                break

            chrom = variant.CHROM
            pos = variant.POS
            ref = variant.REF

            for alt in variant.ALT:
                try:
                    df = self.score_variant(
                        chrom, pos, ref, alt,
                        models=models,
                        model_type=model_type,
                        flank=flank,
                        return_attention=return_attention,
                    )
                    all_results.append(df)
                except Exception as e:
                    warnings.warn(f"Skipping {chrom}:{pos} {ref}>{alt}: {e}")

            if (i + 1) % 100 == 0:
                print(f"  Scored {i + 1} variants...")

        reader.close()

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()

    def _score_vcf_simple(
        self, vcf_path, models, model_type, flank, return_attention, max_variants
    ):
        """Fallback VCF parser (no cyvcf2 dependency)."""
        import gzip

        opener = gzip.open if vcf_path.endswith(".gz") else open
        all_results = []

        with opener(vcf_path, "rt") as f:
            for i, line in enumerate(f):
                if line.startswith("#"):
                    continue
                if max_variants and len(all_results) >= max_variants:
                    break

                fields = line.strip().split("\t")
                chrom, pos, _, ref, alt_str = fields[0], int(fields[1]), fields[2], fields[3], fields[4]

                for alt in alt_str.split(","):
                    try:
                        df = self.score_variant(
                            chrom, pos, ref, alt,
                            models=models,
                            model_type=model_type,
                            flank=flank,
                            return_attention=return_attention,
                        )
                        all_results.append(df)
                    except Exception as e:
                        warnings.warn(f"Skipping {chrom}:{pos} {ref}>{alt}: {e}")

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()

    # ================================================================
    # Helpers
    # ================================================================
    def _resolve_models(
        self,
        models: Optional[List[str]],
        model_type: Optional[str],
    ) -> List[str]:
        """Resolve which models to run."""
        if models and model_type:
            raise ValueError("Specify either 'models' or 'model_type', not both")
        if models:
            # Validate
            for m in models:
                self.registry.get(m)  # raises KeyError if not found
            return models
        if model_type:
            return [m.name for m in self.registry.list(model_type=model_type)]
        raise ValueError(
            "Must specify either 'models' (list of names) or 'model_type' ('TF'/'HISTONE')"
        )

    def list_models(self, model_type: Optional[str] = None) -> List[ModelInfo]:
        """List available models."""
        return self.registry.list(model_type=model_type)

    def search_models(self, query: str) -> List[str]:
        """Search model names by substring."""
        return self.registry.search(query)

    def __repr__(self) -> str:
        genome_str = f", genome='{self._genome_path}'" if self._genome_path else ""
        return f"DVR(device='{self.device}'{genome_str}, {self.registry})"

    def __del__(self):
        """Close genome file handle if open."""
        if self._genome is not None:
            try:
                self._genome.close()
            except Exception:
                pass
