"""
DVR: Main interface for DeepVRegulome variant effect prediction.

v0.1.4:
    - tqdm progress bars
    - Parallel sequence extraction (multiprocessing)
    - log2 scoring matching published pipeline
    - Coordinate sanity check
    - Multi-GPU model distribution, OOM-safe
    - Position-level attention: dvr.last_attention + dvr.plot_attention()
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
    detect_coordinate_system,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        if desc and total:
            print(f"  {desc} ({total} items)...")
        return iterable


# ---------------------------------------------------------------------------
# Scoring (matching published DeepVRegulome pipeline)
# ---------------------------------------------------------------------------
def _compute_scores(p_ref: float, p_alt: float) -> dict:
    eps = 1e-7
    p_ref_c = max(eps, min(1 - eps, p_ref))
    p_alt_c = max(eps, min(1 - eps, p_alt))
    lo_ref = math.log2(p_ref_c / (1 - p_ref_c))
    lo_alt = math.log2(p_alt_c / (1 - p_alt_c))
    return {
        "log_odds_ratio": round(lo_ref - lo_alt, 4),
        "score_change": round((p_alt - p_ref) * max(p_ref, p_alt), 6),
        "_log_odds_ref": round(lo_ref, 4),
        "_log_odds_alt": round(lo_alt, 4),
    }


def _attention_summary(attn_ref: np.ndarray, attn_alt: np.ndarray) -> dict:
    """Compute summary attention metrics."""
    ref_avg = attn_ref.mean(axis=1)  # avg across heads: [layers, seq, seq]
    alt_avg = attn_alt.mean(axis=1)
    diff = alt_avg - ref_avg
    return {
        "attention_score_change": round(float(np.sqrt((diff ** 2).sum())), 6),
        "max_attention_shift": round(float(np.abs(diff).max()), 6),
        "disrupted_layers": int((np.abs(diff).mean(axis=(1, 2)) > 0.01).sum()),
        "total_layers": diff.shape[0],
    }


def _position_attention(attn: np.ndarray) -> np.ndarray:
    """
    Convert raw attention [layers, heads, seq_len, seq_len] to per-position scores.
    Returns: [seq_len] array — mean attention received at each position,
             averaged across all layers and heads.
    """
    # Mean across layers and heads, then sum columns (attention received)
    # attn shape: [layers, heads, seq_len, seq_len]
    return attn.mean(axis=(0, 1)).sum(axis=0)  # [seq_len]


# ---------------------------------------------------------------------------
# GPU worker (multi-GPU mode)
# ---------------------------------------------------------------------------
def _gpu_worker(
    gpu_id, model_names, ref_seqs, alt_seqs,
    batch_size, return_attention, verbose, cache_dir, result_queue,
):
    device = f"cuda:{gpu_id}"
    registry = ModelRegistry()
    all_results = []

    for model_idx, name in enumerate(model_names):
        info = registry.get(name)
        tokenizer = AutoTokenizer.from_pretrained(
            info.hf_repo, subfolder=info.subfolder, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            info.hf_repo, subfolder=info.subfolder, cache_dir=cache_dir,
            output_attentions=return_attention)
        model = model.to(device)
        model.eval()

        n_batches = math.ceil(len(ref_seqs) / batch_size)
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(ref_seqs))
            batch_refs = ref_seqs[batch_start:batch_end]
            batch_alts = alt_seqs[batch_start:batch_end]

            ref_kmers = [to_kmer(s) for s in batch_refs]
            alt_kmers = [to_kmer(s) for s in batch_alts]

            ref_inputs = tokenizer(ref_kmers, return_tensors="pt", max_length=512,
                                   truncation=True, padding=True)
            alt_inputs = tokenizer(alt_kmers, return_tensors="pt", max_length=512,
                                   truncation=True, padding=True)
            ref_inputs = {k: v.to(device) for k, v in ref_inputs.items()}
            alt_inputs = {k: v.to(device) for k, v in alt_inputs.items()}

            with torch.no_grad():
                ref_out = model(**ref_inputs, output_attentions=return_attention)
                alt_out = model(**alt_inputs, output_attentions=return_attention)

            ref_probs = torch.softmax(ref_out.logits, dim=-1)[:, 1].cpu().numpy()
            alt_probs = torch.softmax(alt_out.logits, dim=-1)[:, 1].cpu().numpy()

            for i in range(len(batch_refs)):
                scores = _compute_scores(float(ref_probs[i]), float(alt_probs[i]))
                row = {
                    "_var_idx": batch_start + i,
                    "model": name, "type": info.model_type,
                    "prob_ref": round(float(ref_probs[i]), 6),
                    "prob_alt": round(float(alt_probs[i]), 6),
                    "log_odds_ratio": scores["log_odds_ratio"],
                    "score_change": scores["score_change"],
                }
                if verbose:
                    row["log_odds_ref"] = scores["_log_odds_ref"]
                    row["log_odds_alt"] = scores["_log_odds_alt"]
                if return_attention and ref_out.attentions and alt_out.attentions:
                    ra = torch.stack(ref_out.attentions)[:, i].cpu().numpy()
                    aa = torch.stack(alt_out.attentions)[:, i].cpu().numpy()
                    row.update(_attention_summary(ra, aa))
                all_results.append(row)

        del model
        torch.cuda.empty_cache()
        print(f"  [GPU {gpu_id}] ✓ {name} ({model_idx + 1}/{len(model_names)})")

    result_queue.put(all_results)


class DVR:
    """
    DeepVRegulome: Score regulatory variant effects using fine-tuned DNABERT models.

    Output columns (default):
        model, type, prob_ref, prob_alt, log_odds_ratio, score_change

    Position-level attention (when return_attention=True):
        After scoring, access dvr.last_attention[model_name] for raw data.
        Use dvr.plot_attention(model_name) for visualization.
    """

    def __init__(
        self,
        genome: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        coordinate_system: Optional[str] = None,
    ):
        self.registry = ModelRegistry()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self._genome_path = genome
        self._genome = None
        self._coordinate_system = coordinate_system
        self._coord_offset = None

        # Position-level attention storage (populated when return_attention=True)
        self.last_attention: Dict[str, dict] = {}

    @property
    def genome(self):
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

    def _run_sanity_check(self, variants: list):
        if self._coord_offset is not None:
            return
        if self._coordinate_system == "1-based":
            self._coord_offset = 1
            print("✓ Coordinate system: 1-based (user override). pos → pos-1")
            return
        elif self._coordinate_system == "0-based":
            self._coord_offset = 0
            print("✓ Coordinate system: 0-based (user override). pos used as-is.")
            return
        result = detect_coordinate_system(self.genome, variants)
        self._coord_offset = result["offset"]
        print(result["message"])

    def _adjust_pos(self, pos: int) -> int:
        if self._coord_offset is None:
            return pos - 1
        return pos - self._coord_offset

    # ------------------------------------------------------------------
    # Internal: model loading and prediction
    # ------------------------------------------------------------------
    def _load_model_to_device(self, name: str, device: str):
        info = self.registry.get(name)
        tokenizer = AutoTokenizer.from_pretrained(
            info.hf_repo, subfolder=info.subfolder, cache_dir=self.cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            info.hf_repo, subfolder=info.subfolder, cache_dir=self.cache_dir,
            output_attentions=True)
        model = model.to(device)
        model.eval()
        return model, tokenizer

    def _predict_single(self, model, tokenizer, sequence, device, return_attention=False):
        kmer_seq = to_kmer(sequence, k=6)
        inputs = tokenizer(kmer_seq, return_tensors="pt", max_length=512,
                           truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=return_attention)
        probs = torch.softmax(outputs.logits, dim=-1)
        result = {"prob": probs[0][1].item()}
        if return_attention and outputs.attentions:
            result["attention"] = torch.stack(outputs.attentions).squeeze(1).cpu().numpy()
        return result

    def _predict_batch(self, model, tokenizer, sequences, device, return_attention=False):
        kmer_seqs = [to_kmer(s) for s in sequences]
        inputs = tokenizer(kmer_seqs, return_tensors="pt", max_length=512,
                           truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=return_attention)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        results = []
        for i in range(len(sequences)):
            r = {"prob": float(probs[i])}
            if return_attention and outputs.attentions:
                r["attention"] = torch.stack(outputs.attentions)[:, i].cpu().numpy()
            results.append(r)
        return results

    # ------------------------------------------------------------------
    # Internal: single-GPU scoring (memory-safe, with tqdm)
    # ------------------------------------------------------------------
    def _score_sequences_single_gpu(
        self, ref_seqs, alt_seqs, model_names,
        batch_size=1, return_attention=False, verbose=False, device=None,
        _store_attention_seqs=None,
    ):
        """
        _store_attention_seqs: if provided, a dict of {var_idx: (ref_seq, alt_seq)}
            used to store raw attention data in self.last_attention
        """
        device = device or self.device
        results = []

        model_pbar = tqdm(model_names, desc="Models", unit="model")
        for model_idx, name in enumerate(model_pbar):
            model_pbar.set_postfix(current=name)
            info = self.registry.get(name)
            model, tokenizer = self._load_model_to_device(name, device)

            n_batches = math.ceil(len(ref_seqs) / batch_size)
            batch_pbar = tqdm(range(n_batches), desc=f"  {name}",
                              unit="batch", leave=False)

            for batch_idx in batch_pbar:
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(ref_seqs))
                br = ref_seqs[batch_start:batch_end]
                ba = alt_seqs[batch_start:batch_end]

                if batch_size == 1:
                    ro = self._predict_single(model, tokenizer, br[0], device, return_attention)
                    ao = self._predict_single(model, tokenizer, ba[0], device, return_attention)
                    rp, ap = [ro["prob"]], [ao["prob"]]
                    ra_list = [ro.get("attention")] if return_attention else [None]
                    aa_list = [ao.get("attention")] if return_attention else [None]
                else:
                    ros = self._predict_batch(model, tokenizer, br, device, return_attention)
                    aos = self._predict_batch(model, tokenizer, ba, device, return_attention)
                    rp = [r["prob"] for r in ros]
                    ap = [r["prob"] for r in aos]
                    ra_list = [r.get("attention") for r in ros] if return_attention else [None]*len(ros)
                    aa_list = [r.get("attention") for r in aos] if return_attention else [None]*len(aos)

                for i in range(len(br)):
                    var_idx = batch_start + i
                    scores = _compute_scores(rp[i], ap[i])
                    row = {
                        "_var_idx": var_idx,
                        "model": name, "type": info.model_type,
                        "prob_ref": round(rp[i], 6),
                        "prob_alt": round(ap[i], 6),
                        "log_odds_ratio": scores["log_odds_ratio"],
                        "score_change": scores["score_change"],
                    }
                    if verbose:
                        row["log_odds_ref"] = scores["_log_odds_ref"]
                        row["log_odds_alt"] = scores["_log_odds_alt"]

                    if return_attention and ra_list[i] is not None and aa_list[i] is not None:
                        row.update(_attention_summary(ra_list[i], aa_list[i]))

                        # Store position-level attention
                        ref_pos_attn = _position_attention(ra_list[i])
                        alt_pos_attn = _position_attention(aa_list[i])

                        if name not in self.last_attention:
                            self.last_attention[name] = {}
                        self.last_attention[name][var_idx] = {
                            "ref_attention": ref_pos_attn,   # [seq_len]
                            "alt_attention": alt_pos_attn,   # [seq_len]
                            "diff_attention": alt_pos_attn - ref_pos_attn,  # [seq_len]
                            "ref_raw": ra_list[i],           # [layers, heads, seq, seq]
                            "alt_raw": aa_list[i],
                            "ref_seq": br[i] if _store_attention_seqs else None,
                            "alt_seq": ba[i] if _store_attention_seqs else None,
                            "prob_ref": rp[i],
                            "prob_alt": ap[i],
                        }

                    results.append(row)

            del model, tokenizer
            torch.cuda.empty_cache()

        return results

    # ------------------------------------------------------------------
    # Internal: multi-GPU scoring
    # ------------------------------------------------------------------
    def _score_sequences_multi_gpu(
        self, ref_seqs, alt_seqs, model_names,
        gpus, batch_size=32, return_attention=False, verbose=False,
    ):
        import torch.multiprocessing as mp

        n_gpus = len(gpus)
        splits = []
        for i in range(n_gpus):
            s = i * len(model_names) // n_gpus
            e = (i + 1) * len(model_names) // n_gpus
            splits.append(model_names[s:e])

        print(f"Distributing {len(model_names)} models across {n_gpus} GPUs: "
              f"{[len(s) for s in splits]} models each")

        mp.set_start_method("spawn", force=True)
        result_queue = mp.Queue()
        processes = []

        for i, gpu_id in enumerate(gpus):
            if not splits[i]:
                continue
            p = mp.Process(target=_gpu_worker, args=(
                gpu_id, splits[i], ref_seqs, alt_seqs,
                batch_size, return_attention, verbose, self.cache_dir,
                result_queue))
            p.start()
            processes.append(p)

        all_results = []
        for _ in processes:
            all_results.extend(result_queue.get())
        for p in processes:
            p.join()

        return all_results

    # ==================================================================
    # PUBLIC API
    # ==================================================================
    def score_sequence(
        self,
        ref_seq: Union[str, List[str]],
        alt_seq: Union[str, List[str]],
        models=None, model_type=None,
        batch_size: int = 1, gpus=None,
        return_attention: bool = False,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Score variant(s) from pre-extracted REF and ALT sequences."""
        if isinstance(ref_seq, str):
            ref_seqs, alt_seqs = [ref_seq], [alt_seq]
        else:
            ref_seqs, alt_seqs = list(ref_seq), list(alt_seq)

        model_names = self._resolve_models(models, model_type)

        # Clear previous attention data
        if return_attention:
            self.last_attention = {}

        if gpus and len(gpus) > 1:
            raw = self._score_sequences_multi_gpu(
                ref_seqs, alt_seqs, model_names,
                gpus=gpus, batch_size=batch_size,
                return_attention=return_attention, verbose=verbose)
        else:
            device = f"cuda:{gpus[0]}" if gpus else self.device
            raw = self._score_sequences_single_gpu(
                ref_seqs, alt_seqs, model_names,
                batch_size=batch_size, return_attention=return_attention,
                verbose=verbose, device=device,
                _store_attention_seqs=return_attention)

        df = pd.DataFrame(raw)
        if "_var_idx" in df.columns:
            df = df.drop(columns=["_var_idx"])
        if len(df) > 0:
            df = df.sort_values("log_odds_ratio", ascending=False, key=abs)
        return df.reset_index(drop=True)

    def score_variant(
        self, chrom: str, pos: int, ref: str, alt: str,
        models=None, model_type=None, flank: int = 150,
        batch_size: int = 1, gpus=None,
        return_attention: bool = False, verbose: bool = False,
    ) -> pd.DataFrame:
        """Score a single variant by genomic coordinates."""
        self._run_sanity_check([{"chrom": chrom, "pos": pos, "ref": ref, "alt": alt}])
        pos_0 = self._adjust_pos(pos)
        ref_seq, alt_seq = extract_variant_sequences(
            self.genome, chrom, pos_0, ref, alt, flank=flank)
        df = self.score_sequence(
            ref_seq, alt_seq, models=models, model_type=model_type,
            batch_size=batch_size, gpus=gpus,
            return_attention=return_attention, verbose=verbose)
        df.insert(0, "chrom", chrom)
        df.insert(1, "pos", pos)
        df.insert(2, "ref", ref)
        df.insert(3, "alt", alt)

        # Store sequence info in attention data
        if return_attention:
            for model_name in self.last_attention:
                for var_idx in self.last_attention[model_name]:
                    self.last_attention[model_name][var_idx]["ref_seq"] = ref_seq
                    self.last_attention[model_name][var_idx]["alt_seq"] = alt_seq
                    self.last_attention[model_name][var_idx]["variant_pos"] = flank
                    self.last_attention[model_name][var_idx]["chrom"] = chrom
                    self.last_attention[model_name][var_idx]["genomic_pos"] = pos

        return df

    def score_variants(
        self, variants: pd.DataFrame,
        models=None, model_type=None, flank: int = 150,
        batch_size: int = 32, gpus=None,
        return_attention: bool = False, verbose: bool = False,
    ) -> pd.DataFrame:
        """Score multiple variants from a DataFrame (columns: chrom, pos, ref, alt)."""
        # Auto-detect column names
        col_map = {}
        for col in variants.columns:
            cl = col.lower().strip()
            if cl in ("chrom", "chr", "#chrom"):
                col_map[col] = "chrom"
            elif cl in ("pos", "start", "position"):
                col_map[col] = "pos"
            elif cl in ("ref", "reference", "ref_allele"):
                col_map[col] = "ref"
            elif cl in ("alt", "alternative", "alt_allele"):
                col_map[col] = "alt"

        if col_map:
            variants = variants.rename(columns=col_map)

        required = {"chrom", "pos", "ref", "alt"}
        if not required.issubset(variants.columns):
            missing = required - set(variants.columns)
            raise ValueError(
                f"Missing columns: {missing}. "
                f"Expected: chrom, pos, ref, alt (or CHROM, start, REF, ALT)"
            )

        variant_list = variants[["chrom", "pos", "ref", "alt"]].to_dict("records")
        self._run_sanity_check(variant_list)

        adjusted = []
        for v in variant_list:
            adjusted.append({
                "chrom": v["chrom"], "pos": self._adjust_pos(v["pos"]),
                "ref": v["ref"], "alt": v["alt"],
            })

        print(f"Extracting sequences for {len(variants)} variants...")
        seq_pairs = extract_variant_sequences_batch(self._genome_path, adjusted, flank)

        valid_indices, ref_seqs, alt_seqs = [], [], []
        for i, (r, a) in enumerate(seq_pairs):
            if r is not None and a is not None:
                valid_indices.append(i)
                ref_seqs.append(r)
                alt_seqs.append(a)

        if len(valid_indices) < len(variants):
            warnings.warn(f"Failed to extract sequences for {len(variants) - len(valid_indices)} variants")
        if not ref_seqs:
            return pd.DataFrame()

        model_names = self._resolve_models(models, model_type)
        print(f"Scoring {len(ref_seqs)} variants × {len(model_names)} models...")

        if return_attention:
            self.last_attention = {}

        if gpus and len(gpus) > 1:
            raw = self._score_sequences_multi_gpu(
                ref_seqs, alt_seqs, model_names,
                gpus=gpus, batch_size=batch_size,
                return_attention=return_attention, verbose=verbose)
        else:
            device = f"cuda:{gpus[0]}" if gpus else self.device
            raw = self._score_sequences_single_gpu(
                ref_seqs, alt_seqs, model_names,
                batch_size=batch_size, return_attention=return_attention,
                verbose=verbose, device=device,
                _store_attention_seqs=return_attention)

        df = pd.DataFrame(raw)
        if len(df) > 0 and "_var_idx" in df.columns:
            var_info = variants.iloc[valid_indices].reset_index(drop=True)
            df["chrom"] = df["_var_idx"].map(lambda i: var_info.loc[i, "chrom"])
            df["pos"] = df["_var_idx"].map(lambda i: var_info.loc[i, "pos"])
            df["ref"] = df["_var_idx"].map(lambda i: var_info.loc[i, "ref"])
            df["alt"] = df["_var_idx"].map(lambda i: var_info.loc[i, "alt"])
            df = df.drop(columns=["_var_idx"])
            cols = ["chrom", "pos", "ref", "alt"] + [c for c in df.columns if c not in ["chrom", "pos", "ref", "alt"]]
            df = df[cols]
            df = df.sort_values("log_odds_ratio", ascending=False, key=abs)
        return df.reset_index(drop=True)

    def score_vcf(
        self, vcf_path: str,
        models=None, model_type=None, flank: int = 150,
        batch_size: int = 32, gpus=None,
        return_attention: bool = False, verbose: bool = False,
        max_variants: Optional[int] = None,
    ) -> pd.DataFrame:
        """Score all variants in a VCF file."""
        print(f"Parsing VCF: {vcf_path}")
        variant_list = parse_vcf(vcf_path, max_variants=max_variants)
        print(f"  Found {len(variant_list)} variants")
        if not variant_list:
            return pd.DataFrame()
        return self.score_variants(
            pd.DataFrame(variant_list), models=models, model_type=model_type,
            flank=flank, batch_size=batch_size, gpus=gpus,
            return_attention=return_attention, verbose=verbose)

    # ==================================================================
    # ATTENTION VISUALIZATION
    # ==================================================================
    def get_attention(self, model_name: str, var_idx: int = 0) -> dict:
        """
        Get position-level attention data for a scored variant.

        Args:
            model_name: Model name (e.g., "ATF4")
            var_idx: Variant index (0 for single variant scoring)

        Returns:
            dict with keys:
                ref_attention: [seq_len] per-position attention for REF
                alt_attention: [seq_len] per-position attention for ALT
                diff_attention: [seq_len] ALT - REF attention difference
                ref_seq: REF DNA sequence
                alt_seq: ALT DNA sequence
                variant_pos: position of variant within window
                prob_ref, prob_alt: binding probabilities
                ref_raw: [layers, heads, seq, seq] full attention tensor
                alt_raw: same for ALT
        """
        if model_name not in self.last_attention:
            available = list(self.last_attention.keys())
            raise KeyError(
                f"No attention data for '{model_name}'. "
                f"Available: {available}. "
                f"Did you run scoring with return_attention=True?"
            )
        if var_idx not in self.last_attention[model_name]:
            available = list(self.last_attention[model_name].keys())
            raise KeyError(f"Variant index {var_idx} not found. Available: {available}")
        return self.last_attention[model_name][var_idx]

    def plot_attention(
        self,
        model_name: str,
        var_idx: int = 0,
        window: int = 20,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None,
    ):
        """
        Plot position-level attention around the variant site.

        Shows REF and ALT attention scores in a ±window bp region around
        the variant, with the variant position highlighted.

        Args:
            model_name: Model name (e.g., "ATF4")
            var_idx: Variant index (0 for single variant)
            window: Bases to show on each side of variant (default: 20)
            figsize: Figure size
            save_path: If provided, save figure to this path

        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib is required for plotting. pip install matplotlib")

        data = self.get_attention(model_name, var_idx)
        ref_attn = data["ref_attention"]
        alt_attn = data["alt_attention"]
        diff_attn = data["diff_attention"]
        variant_pos = data.get("variant_pos", len(ref_attn) // 2)
        ref_seq = data.get("ref_seq", "")
        alt_seq = data.get("alt_seq", "")
        p_ref = data.get("prob_ref", 0)
        p_alt = data.get("prob_alt", 0)
        chrom = data.get("chrom", "")
        genomic_pos = data.get("genomic_pos", "")

        # Window around variant (in k-mer space, variant_pos maps to similar position)
        # Attention is in k-mer space (seq_len = len(seq) - k + 1 for k=6)
        kmer_variant_pos = max(0, variant_pos - 5)  # approximate k-mer position
        start = max(0, kmer_variant_pos - window)
        end = min(len(ref_attn), kmer_variant_pos + window + 1)

        ref_window = ref_attn[start:end]
        alt_window = alt_attn[start:end]
        diff_window = diff_attn[start:end]
        positions = np.arange(start, end)

        # Get nucleotides for x-axis labels
        if ref_seq and len(ref_seq) > end + 5:
            # Map k-mer positions back to sequence positions (k-mer i covers seq[i:i+6])
            seq_labels = [ref_seq[p + 3] if p + 3 < len(ref_seq) else "" for p in positions]
        else:
            seq_labels = [str(p) for p in positions]

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                                  gridspec_kw={"height_ratios": [2, 2, 1.5]})

        # Plot 1: REF attention
        axes[0].bar(positions, ref_window, color="#3498db", alpha=0.8, width=0.8)
        axes[0].axvline(x=kmer_variant_pos, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
        axes[0].set_ylabel("Attention")
        axes[0].set_title(
            f"{model_name} — REF (wild-type)   |   "
            f"P(binding) = {p_ref:.4f}",
            fontsize=11
        )

        # Plot 2: ALT attention
        axes[1].bar(positions, alt_window, color="#e74c3c", alpha=0.8, width=0.8)
        axes[1].axvline(x=kmer_variant_pos, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
        axes[1].set_ylabel("Attention")
        axes[1].set_title(
            f"{model_name} — ALT (mutant)   |   "
            f"P(binding) = {p_alt:.4f}",
            fontsize=11
        )

        # Plot 3: Difference (ALT - REF)
        colors_diff = ["#e74c3c" if d < 0 else "#2ecc71" for d in diff_window]
        axes[2].bar(positions, diff_window, color=colors_diff, alpha=0.8, width=0.8)
        axes[2].axvline(x=kmer_variant_pos, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
        axes[2].axhline(y=0, color="black", linewidth=0.5)
        axes[2].set_ylabel("Δ Attention")
        axes[2].set_xlabel("Position (k-mer index)")
        axes[2].set_title("Attention Change (ALT − REF)", fontsize=11)

        # X-axis labels
        if len(seq_labels) <= 50:
            axes[2].set_xticks(positions)
            axes[2].set_xticklabels(seq_labels, fontsize=7, rotation=0)

        # Supertitle
        variant_str = f"{chrom}:{genomic_pos}" if chrom else "variant"
        fig.suptitle(
            f"DeepVRegulome Attention: {model_name} at {variant_str}   |   "
            f"±{window}bp window",
            fontsize=13, fontweight="bold", y=1.02
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")

        return fig

    def plot_attention_heatmap(
        self,
        model_name: str,
        var_idx: int = 0,
        window: int = 20,
        layer: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
    ):
        """
        Plot attention heatmap (layer × position) around the variant site.

        Shows how each BERT layer attends to positions near the variant,
        for both REF and ALT side by side.

        Args:
            model_name: Model name
            var_idx: Variant index
            window: Bases on each side of variant
            layer: Specific layer to plot (None = all layers summary)
            figsize: Figure size
            save_path: Save path
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required. pip install matplotlib")

        data = self.get_attention(model_name, var_idx)
        ref_raw = data["ref_raw"]   # [layers, heads, seq, seq]
        alt_raw = data["alt_raw"]
        variant_pos = data.get("variant_pos", ref_raw.shape[2] // 2)
        kmer_vp = max(0, variant_pos - 5)
        start = max(0, kmer_vp - window)
        end = min(ref_raw.shape[2], kmer_vp + window + 1)

        # Average across heads, extract window
        ref_layers = ref_raw.mean(axis=1)[:, start:end, start:end]  # [layers, win, win]
        alt_layers = alt_raw.mean(axis=1)[:, start:end, start:end]

        # Sum columns to get "attention received" per position per layer
        ref_heatmap = ref_layers.sum(axis=2)  # [layers, win]
        alt_heatmap = alt_layers.sum(axis=2)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        im0 = axes[0].imshow(ref_heatmap, aspect="auto", cmap="Blues")
        axes[0].set_title(f"REF — {model_name}")
        axes[0].set_ylabel("Layer")
        axes[0].set_xlabel("Position")
        plt.colorbar(im0, ax=axes[0], shrink=0.6)

        im1 = axes[1].imshow(alt_heatmap, aspect="auto", cmap="Reds")
        axes[1].set_title(f"ALT — {model_name}")
        axes[1].set_xlabel("Position")
        plt.colorbar(im1, ax=axes[1], shrink=0.6)

        diff_heatmap = alt_heatmap - ref_heatmap
        vmax = max(abs(diff_heatmap.min()), abs(diff_heatmap.max()))
        im2 = axes[2].imshow(diff_heatmap, aspect="auto", cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax)
        axes[2].set_title(f"Δ (ALT − REF)")
        axes[2].set_xlabel("Position")
        plt.colorbar(im2, ax=axes[2], shrink=0.6)

        # Mark variant position
        vp_in_window = kmer_vp - start
        for ax in axes:
            ax.axvline(x=vp_in_window, color="lime", linewidth=1.5, linestyle="--", alpha=0.8)

        chrom = data.get("chrom", "")
        gpos = data.get("genomic_pos", "")
        fig.suptitle(f"Attention Heatmap: {model_name} at {chrom}:{gpos}  ±{window}bp",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")

        return fig

    # ==================================================================
    # Helpers
    # ==================================================================
    def _resolve_models(self, models, model_type):
        if models and model_type:
            raise ValueError("Specify either 'models' or 'model_type', not both")
        if models:
            for m in models:
                self.registry.get(m)
            return models
        if model_type:
            return [m.name for m in self.registry.list(model_type=model_type)]
        raise ValueError("Must specify either 'models' or 'model_type' ('TF'/'HISTONE')")

    def list_models(self, model_type=None):
        return self.registry.list(model_type=model_type)

    def search_models(self, query: str):
        return self.registry.search(query)

    def __repr__(self):
        g = f", genome='{self._genome_path}'" if self._genome_path else ""
        cs = f", coords={self._coordinate_system}" if self._coordinate_system else ""
        return f"DVR(device='{self.device}'{g}{cs}, {self.registry})"

    def __del__(self):
        if self._genome is not None:
            try:
                self._genome.close()
            except Exception:
                pass
