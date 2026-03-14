"""
DVR: Main interface for DeepVRegulome variant effect prediction.

Supports:
    - Single variant scoring (score_variant)
    - Sequence-based scoring (score_sequence)
    - Batch scoring from DataFrame (score_variants)
    - VCF file scoring (score_vcf)
    - Multi-GPU parallelism (gpus parameter)
    - Batched inference (batch_size parameter)
    - Automatic GPU memory management (no OOM)
    - Attention-based interpretability
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from deepvregulome.registry import ModelInfo, ModelRegistry
from deepvregulome.utils import (
    to_kmer,
    extract_variant_sequences,
    extract_variant_sequences_batch,
    parse_vcf,
)


# ---------------------------------------------------------------------------
# GPU worker function (runs on each GPU in multi-GPU mode)
# ---------------------------------------------------------------------------
def _gpu_worker(
    gpu_id: int,
    model_names: List[str],
    ref_seqs: List[str],
    alt_seqs: List[str],
    batch_size: int,
    return_attention: bool,
    disruption_threshold: float,
    cache_dir: Optional[str],
    result_queue,
):
    """
    Worker function for multi-GPU scoring.
    Each worker loads its assigned models on its GPU, scores all sequences,
    and puts results into the shared queue.
    """
    device = f"cuda:{gpu_id}"
    registry = ModelRegistry()
    all_results = []

    for model_idx, name in enumerate(model_names):
        info = registry.get(name)

        # Load model to this GPU
        tokenizer = AutoTokenizer.from_pretrained(
            info.hf_repo, subfolder=info.subfolder, cache_dir=cache_dir,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            info.hf_repo, subfolder=info.subfolder, cache_dir=cache_dir,
            output_attentions=return_attention,
        )
        model = model.to(device)
        model.eval()

        # Score all sequences in batches
        for batch_start in range(0, len(ref_seqs), batch_size):
            batch_end = min(batch_start + batch_size, len(ref_seqs))
            batch_refs = ref_seqs[batch_start:batch_end]
            batch_alts = alt_seqs[batch_start:batch_end]

            # Tokenize batch
            ref_kmers = [to_kmer(s) for s in batch_refs]
            alt_kmers = [to_kmer(s) for s in batch_alts]

            ref_inputs = tokenizer(
                ref_kmers, return_tensors="pt", max_length=512,
                truncation=True, padding=True,
            )
            alt_inputs = tokenizer(
                alt_kmers, return_tensors="pt", max_length=512,
                truncation=True, padding=True,
            )

            ref_inputs = {k: v.to(device) for k, v in ref_inputs.items()}
            alt_inputs = {k: v.to(device) for k, v in alt_inputs.items()}

            with torch.no_grad():
                ref_outputs = model(**ref_inputs, output_attentions=return_attention)
                alt_outputs = model(**alt_inputs, output_attentions=return_attention)

            ref_probs = torch.softmax(ref_outputs.logits, dim=-1)[:, 1].cpu().numpy()
            alt_probs = torch.softmax(alt_outputs.logits, dim=-1)[:, 1].cpu().numpy()

            # Compute scores for each variant in batch
            for i in range(len(batch_refs)):
                var_idx = batch_start + i
                p_ref = float(ref_probs[i])
                p_alt = float(alt_probs[i])

                eps = 1e-7
                lo_ref = math.log((p_ref + eps) / (1 - p_ref + eps))
                lo_alt = math.log((p_alt + eps) / (1 - p_alt + eps))
                lo_change = lo_alt - lo_ref

                row = {
                    "_var_idx": var_idx,
                    "model": name,
                    "type": info.model_type,
                    "prob_ref": round(p_ref, 6),
                    "prob_alt": round(p_alt, 6),
                    "log_odds_ref": round(lo_ref, 4),
                    "log_odds_alt": round(lo_alt, 4),
                    "log_odds_change": round(lo_change, 4),
                    "abs_log_odds_change": round(abs(lo_change), 4),
                    "disrupted": abs(lo_change) > disruption_threshold,
                }

                # Attention metrics
                if return_attention and ref_outputs.attentions and alt_outputs.attentions:
                    ref_attn = torch.stack(ref_outputs.attentions)[:, i].cpu().numpy()
                    alt_attn = torch.stack(alt_outputs.attentions)[:, i].cpu().numpy()
                    ref_avg = ref_attn.mean(axis=1)
                    alt_avg = alt_attn.mean(axis=1)
                    diff = alt_avg - ref_avg
                    row["attention_score_change"] = round(float(np.sqrt((diff ** 2).sum())), 6)
                    row["max_attention_shift"] = round(float(np.abs(diff).max()), 6)
                    layer_changes = np.abs(diff).mean(axis=(1, 2))
                    row["disrupted_layers"] = int((layer_changes > 0.01).sum())
                    row["total_layers"] = len(layer_changes)

                all_results.append(row)

        # FREE GPU MEMORY — critical to avoid OOM
        del model
        torch.cuda.empty_cache()

        if (model_idx + 1) % 10 == 0:
            print(f"  [GPU {gpu_id}] Scored {model_idx + 1}/{len(model_names)} models")

    result_queue.put(all_results)


class DVR:
    """
    DeepVRegulome: Score regulatory variant effects using fine-tuned DNABERT models.

    Usage:
        dvr = DVR(genome="hg38.fa")

        # Single variant
        result = dvr.score_variant("chr1", 3456782, "A", "C", models=["CTCFL"])

        # Batch: 20K variants × 462 models × 4 GPUs
        results = dvr.score_vcf(
            "variants.vcf",
            model_type="TF",
            batch_size=100,
            gpus=[0, 1, 2, 3],
        )

        # From DataFrame
        results = dvr.score_variants(
            variant_df,  # columns: chrom, pos, ref, alt
            models=["CTCFL", "SP1"],
            batch_size=50,
            gpus=[0, 1, 2, 3, 4],
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
            genome: Path to reference genome FASTA (hg38). Required for
                    score_variant() and score_vcf().
            device: "cuda", "cuda:0", "cpu", etc. Auto-detected if None.
            cache_dir: Directory for caching downloaded HF models.
            disruption_threshold: |log_odds_change| threshold for disruption flag.
        """
        self.registry = ModelRegistry()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.disruption_threshold = disruption_threshold

        self._genome_path = genome
        self._genome = None

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

    # ------------------------------------------------------------------
    # Internal: single-model, single-sequence prediction
    # ------------------------------------------------------------------
    def _load_model_to_device(self, name: str, device: str):
        """Load a model onto a specific device. Returns (model, tokenizer)."""
        info = self.registry.get(name)
        tokenizer = AutoTokenizer.from_pretrained(
            info.hf_repo, subfolder=info.subfolder, cache_dir=self.cache_dir,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            info.hf_repo, subfolder=info.subfolder, cache_dir=self.cache_dir,
            output_attentions=True,
        )
        model = model.to(device)
        model.eval()
        return model, tokenizer

    def _predict_single(self, model, tokenizer, sequence: str, device: str,
                        return_attention: bool = False) -> dict:
        """Run inference on a single sequence."""
        kmer_seq = to_kmer(sequence, k=6)
        inputs = tokenizer(
            kmer_seq, return_tensors="pt", max_length=512,
            truncation=True, padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=return_attention)

        probs = torch.softmax(outputs.logits, dim=-1)
        result = {"prob": probs[0][1].item()}

        if return_attention and outputs.attentions:
            attn = torch.stack(outputs.attentions).squeeze(1)
            result["attention"] = attn.cpu().numpy()

        return result

    def _predict_batch(self, model, tokenizer, sequences: List[str], device: str,
                       return_attention: bool = False) -> List[dict]:
        """Run inference on a batch of sequences."""
        kmer_seqs = [to_kmer(s) for s in sequences]
        inputs = tokenizer(
            kmer_seqs, return_tensors="pt", max_length=512,
            truncation=True, padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=return_attention)

        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

        results = []
        for i in range(len(sequences)):
            r = {"prob": float(probs[i])}
            if return_attention and outputs.attentions:
                attn = torch.stack(outputs.attentions)[:, i].cpu().numpy()
                r["attention"] = attn
            results.append(r)

        return results

    def _compute_attention_change(self, attn_ref: np.ndarray, attn_alt: np.ndarray) -> dict:
        """Compute attention disruption metrics between REF and ALT."""
        ref_avg = attn_ref.mean(axis=1)
        alt_avg = attn_alt.mean(axis=1)
        diff = alt_avg - ref_avg
        attention_score_change = float(np.sqrt((diff ** 2).sum()))
        max_attention_shift = float(np.abs(diff).max())
        layer_changes = np.abs(diff).mean(axis=(1, 2))
        disrupted_layers = int((layer_changes > 0.01).sum())
        return {
            "attention_score_change": round(attention_score_change, 6),
            "max_attention_shift": round(max_attention_shift, 6),
            "disrupted_layers": disrupted_layers,
            "total_layers": len(layer_changes),
        }

    # ------------------------------------------------------------------
    # Internal: score sequences against models (single GPU, memory-safe)
    # ------------------------------------------------------------------
    def _score_sequences_single_gpu(
        self,
        ref_seqs: List[str],
        alt_seqs: List[str],
        model_names: List[str],
        batch_size: int = 1,
        return_attention: bool = False,
        device: Optional[str] = None,
    ) -> List[dict]:
        """
        Score REF/ALT sequence pairs against multiple models on a single GPU.
        Loads one model at a time to avoid OOM.
        """
        device = device or self.device
        results = []

        for model_idx, name in enumerate(model_names):
            info = self.registry.get(name)

            # Load model
            model, tokenizer = self._load_model_to_device(name, device)

            # Score in batches
            for batch_start in range(0, len(ref_seqs), batch_size):
                batch_end = min(batch_start + batch_size, len(ref_seqs))
                batch_refs = ref_seqs[batch_start:batch_end]
                batch_alts = alt_seqs[batch_start:batch_end]

                if batch_size == 1:
                    # Single sequence path (original behavior)
                    ref_out = self._predict_single(
                        model, tokenizer, batch_refs[0], device, return_attention
                    )
                    alt_out = self._predict_single(
                        model, tokenizer, batch_alts[0], device, return_attention
                    )
                    ref_probs = [ref_out["prob"]]
                    alt_probs = [alt_out["prob"]]
                    ref_attns = [ref_out.get("attention")] if return_attention else [None]
                    alt_attns = [alt_out.get("attention")] if return_attention else [None]
                else:
                    # Batched path
                    ref_outs = self._predict_batch(
                        model, tokenizer, batch_refs, device, return_attention
                    )
                    alt_outs = self._predict_batch(
                        model, tokenizer, batch_alts, device, return_attention
                    )
                    ref_probs = [r["prob"] for r in ref_outs]
                    alt_probs = [r["prob"] for r in alt_outs]
                    ref_attns = [r.get("attention") for r in ref_outs] if return_attention else [None] * len(ref_outs)
                    alt_attns = [r.get("attention") for r in alt_outs] if return_attention else [None] * len(alt_outs)

                # Build result rows
                for i in range(len(batch_refs)):
                    var_idx = batch_start + i
                    p_ref = ref_probs[i]
                    p_alt = alt_probs[i]

                    eps = 1e-7
                    lo_ref = math.log((p_ref + eps) / (1 - p_ref + eps))
                    lo_alt = math.log((p_alt + eps) / (1 - p_alt + eps))
                    lo_change = lo_alt - lo_ref

                    row = {
                        "_var_idx": var_idx,
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

                    if return_attention and ref_attns[i] is not None and alt_attns[i] is not None:
                        attn_metrics = self._compute_attention_change(ref_attns[i], alt_attns[i])
                        row.update(attn_metrics)

                    results.append(row)

            # FREE GPU MEMORY after each model
            del model, tokenizer
            torch.cuda.empty_cache()

            if (model_idx + 1) % 10 == 0:
                print(f"  Scored {model_idx + 1}/{len(model_names)} models...")

        return results

    # ------------------------------------------------------------------
    # Internal: multi-GPU scoring
    # ------------------------------------------------------------------
    def _score_sequences_multi_gpu(
        self,
        ref_seqs: List[str],
        alt_seqs: List[str],
        model_names: List[str],
        gpus: List[int],
        batch_size: int = 32,
        return_attention: bool = False,
    ) -> List[dict]:
        """
        Distribute models across multiple GPUs and score in parallel.
        """
        import torch.multiprocessing as mp

        # Split models across GPUs
        n_gpus = len(gpus)
        models_per_gpu = []
        for i in range(n_gpus):
            start = i * len(model_names) // n_gpus
            end = (i + 1) * len(model_names) // n_gpus
            models_per_gpu.append(model_names[start:end])

        print(f"Distributing {len(model_names)} models across {n_gpus} GPUs: "
              f"{[len(m) for m in models_per_gpu]} models each")

        # Launch workers
        mp.set_start_method("spawn", force=True)
        result_queue = mp.Queue()

        processes = []
        for i, gpu_id in enumerate(gpus):
            if not models_per_gpu[i]:
                continue
            p = mp.Process(
                target=_gpu_worker,
                args=(
                    gpu_id,
                    models_per_gpu[i],
                    ref_seqs,
                    alt_seqs,
                    batch_size,
                    return_attention,
                    self.disruption_threshold,
                    self.cache_dir,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        # Collect results
        all_results = []
        for _ in processes:
            worker_results = result_queue.get()
            all_results.extend(worker_results)

        for p in processes:
            p.join()

        return all_results

    # ==================================================================
    # PUBLIC API: score_sequence
    # ==================================================================
    def score_sequence(
        self,
        ref_seq: Union[str, List[str]],
        alt_seq: Union[str, List[str]],
        models: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        batch_size: int = 1,
        gpus: Optional[List[int]] = None,
        return_attention: bool = False,
    ) -> pd.DataFrame:
        """
        Score variant(s) given pre-extracted REF and ALT sequences.

        Args:
            ref_seq: Reference DNA sequence(s). String for single, list for batch.
            alt_seq: Alternative DNA sequence(s). String for single, list for batch.
            models: List of model names (e.g., ["CTCFL", "SP1"])
            model_type: Score all models of this type ("TF" or "HISTONE")
            batch_size: Number of sequences per forward pass (default: 1)
            gpus: List of GPU IDs for multi-GPU (e.g., [0,1,2,3]). None = single GPU.
            return_attention: Include attention disruption metrics.

        Returns:
            DataFrame with scoring results.
        """
        # Normalize to lists
        if isinstance(ref_seq, str):
            ref_seqs = [ref_seq]
            alt_seqs = [alt_seq]
        else:
            ref_seqs = list(ref_seq)
            alt_seqs = list(alt_seq)

        model_names = self._resolve_models(models, model_type)

        # Choose single-GPU or multi-GPU path
        if gpus and len(gpus) > 1:
            raw_results = self._score_sequences_multi_gpu(
                ref_seqs, alt_seqs, model_names,
                gpus=gpus, batch_size=batch_size,
                return_attention=return_attention,
            )
        else:
            device = f"cuda:{gpus[0]}" if gpus else self.device
            raw_results = self._score_sequences_single_gpu(
                ref_seqs, alt_seqs, model_names,
                batch_size=batch_size,
                return_attention=return_attention,
                device=device,
            )

        df = pd.DataFrame(raw_results)
        if "_var_idx" in df.columns:
            df = df.drop(columns=["_var_idx"])
        if len(df) > 0:
            df = df.sort_values("abs_log_odds_change", ascending=False)
        return df.reset_index(drop=True)

    # ==================================================================
    # PUBLIC API: score_variant (single variant, convenience method)
    # ==================================================================
    def score_variant(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        models: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        flank: int = 150,
        batch_size: int = 1,
        gpus: Optional[List[int]] = None,
        return_attention: bool = False,
    ) -> pd.DataFrame:
        """
        Score a single variant by genomic coordinates.

        Args:
            chrom: Chromosome (e.g., "chr1")
            pos: 1-based position
            ref: Reference allele
            alt: Alternate allele
            models: List of model names
            model_type: Score all models of this type
            flank: Flanking bases (default: 150 → 301bp for SNV)
            batch_size: Sequences per forward pass (only relevant for multi-model)
            gpus: GPU IDs for multi-GPU scoring
            return_attention: Include attention metrics

        Returns:
            DataFrame with scoring results + chrom, pos, ref, alt columns.
        """
        ref_seq, alt_seq = extract_variant_sequences(
            self.genome, chrom, pos, ref, alt, flank=flank
        )

        df = self.score_sequence(
            ref_seq, alt_seq,
            models=models, model_type=model_type,
            batch_size=batch_size, gpus=gpus,
            return_attention=return_attention,
        )

        df.insert(0, "chrom", chrom)
        df.insert(1, "pos", pos)
        df.insert(2, "ref", ref)
        df.insert(3, "alt", alt)

        return df

    # ==================================================================
    # PUBLIC API: score_variants (batch from DataFrame)
    # ==================================================================
    def score_variants(
        self,
        variants: pd.DataFrame,
        models: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        flank: int = 150,
        batch_size: int = 32,
        gpus: Optional[List[int]] = None,
        return_attention: bool = False,
    ) -> pd.DataFrame:
        """
        Score multiple variants from a DataFrame.

        Args:
            variants: DataFrame with columns: chrom, pos, ref, alt
            models: List of model names
            model_type: Score all models of this type ("TF" or "HISTONE")
            flank: Flanking bases
            batch_size: Variants per forward pass
            gpus: GPU IDs for multi-GPU (e.g., [0,1,2,3])
            return_attention: Include attention metrics

        Returns:
            DataFrame with one row per (variant × model), including
            chrom, pos, ref, alt from the input.

        Example:
            import pandas as pd
            variants = pd.DataFrame({
                "chrom": ["chr1", "chr1", "chr2"],
                "pos": [3456782, 1234567, 9876543],
                "ref": ["A", "G", "T"],
                "alt": ["C", "A", "TA"],
            })
            results = dvr.score_variants(variants, model_type="TF",
                                          batch_size=100, gpus=[0,1,2,3])
        """
        # Validate columns
        required = {"chrom", "pos", "ref", "alt"}
        if not required.issubset(variants.columns):
            missing = required - set(variants.columns)
            raise ValueError(f"Missing columns: {missing}. Required: {required}")

        # Extract sequences
        print(f"Extracting sequences for {len(variants)} variants...")
        variant_list = variants[["chrom", "pos", "ref", "alt"]].to_dict("records")
        seq_pairs = extract_variant_sequences_batch(self.genome, variant_list, flank)

        # Filter out failed extractions
        valid_indices = []
        ref_seqs = []
        alt_seqs = []
        for i, (ref_s, alt_s) in enumerate(seq_pairs):
            if ref_s is not None and alt_s is not None:
                valid_indices.append(i)
                ref_seqs.append(ref_s)
                alt_seqs.append(alt_s)

        if len(valid_indices) < len(variants):
            n_failed = len(variants) - len(valid_indices)
            warnings.warn(f"Failed to extract sequences for {n_failed} variants")

        if not ref_seqs:
            return pd.DataFrame()

        print(f"Scoring {len(ref_seqs)} variants × {len(self._resolve_models(models, model_type))} models...")

        model_names = self._resolve_models(models, model_type)

        # Score
        if gpus and len(gpus) > 1:
            raw_results = self._score_sequences_multi_gpu(
                ref_seqs, alt_seqs, model_names,
                gpus=gpus, batch_size=batch_size,
                return_attention=return_attention,
            )
        else:
            device = f"cuda:{gpus[0]}" if gpus else self.device
            raw_results = self._score_sequences_single_gpu(
                ref_seqs, alt_seqs, model_names,
                batch_size=batch_size,
                return_attention=return_attention,
                device=device,
            )

        # Build DataFrame and attach variant info
        df = pd.DataFrame(raw_results)

        if len(df) > 0 and "_var_idx" in df.columns:
            # Map _var_idx back to original variant info
            var_info = variants.iloc[valid_indices].reset_index(drop=True)

            df["chrom"] = df["_var_idx"].map(lambda i: var_info.loc[i, "chrom"])
            df["pos"] = df["_var_idx"].map(lambda i: var_info.loc[i, "pos"])
            df["ref"] = df["_var_idx"].map(lambda i: var_info.loc[i, "ref"])
            df["alt"] = df["_var_idx"].map(lambda i: var_info.loc[i, "alt"])
            df = df.drop(columns=["_var_idx"])

            # Reorder: variant info first
            variant_cols = ["chrom", "pos", "ref", "alt"]
            other_cols = [c for c in df.columns if c not in variant_cols]
            df = df[variant_cols + other_cols]

            df = df.sort_values("abs_log_odds_change", ascending=False)

        return df.reset_index(drop=True)

    # ==================================================================
    # PUBLIC API: score_vcf
    # ==================================================================
    def score_vcf(
        self,
        vcf_path: str,
        models: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        flank: int = 150,
        batch_size: int = 32,
        gpus: Optional[List[int]] = None,
        return_attention: bool = False,
        max_variants: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Score all variants in a VCF file.
        Handles VCF files with or without headers, and .vcf.gz files.

        Args:
            vcf_path: Path to VCF file (.vcf or .vcf.gz)
            models: List of model names
            model_type: Score all models of this type
            flank: Flanking bases
            batch_size: Variants per forward pass
            gpus: GPU IDs for multi-GPU
            return_attention: Include attention metrics
            max_variants: Maximum variants to score (None = all)

        Returns:
            DataFrame with one row per (variant × model).

        Example:
            dvr = DVR(genome="hg38.fa")
            results = dvr.score_vcf(
                "merged_CaVEMan.vcf",
                model_type="TF",
                batch_size=100,
                gpus=[0, 1, 2, 3],
                max_variants=1000,
            )
        """
        # Parse VCF
        print(f"Parsing VCF: {vcf_path}")
        variant_list = parse_vcf(vcf_path, max_variants=max_variants)
        print(f"  Found {len(variant_list)} variants")

        if not variant_list:
            return pd.DataFrame()

        # Convert to DataFrame and use score_variants
        variant_df = pd.DataFrame(variant_list)

        return self.score_variants(
            variant_df,
            models=models, model_type=model_type,
            flank=flank, batch_size=batch_size,
            gpus=gpus, return_attention=return_attention,
        )

    # ==================================================================
    # Helpers
    # ==================================================================
    def _resolve_models(
        self,
        models: Optional[List[str]],
        model_type: Optional[str],
    ) -> List[str]:
        """Resolve which models to run."""
        if models and model_type:
            raise ValueError("Specify either 'models' or 'model_type', not both")
        if models:
            for m in models:
                self.registry.get(m)
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
        if self._genome is not None:
            try:
                self._genome.close()
            except Exception:
                pass
