"""
DVR: Main interface for DeepVRegulome variant effect prediction.

v0.1.4:
    - tqdm progress bars for all long-running operations
    - Parallel sequence extraction (multiprocessing)
    - log2 scoring matching published pipeline
    - Coordinate sanity check (auto-detect 1-based vs 0-based)
    - Multi-GPU model distribution
    - OOM-safe (1 model at a time per GPU)
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
    # Fallback: no progress bar, just iterate
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
                    diff = aa.mean(axis=1) - ra.mean(axis=1)
                    row["attention_score_change"] = round(float(np.sqrt((diff**2).sum())), 6)
                    row["max_attention_shift"] = round(float(np.abs(diff).max()), 6)
                    lc = np.abs(diff).mean(axis=(1, 2))
                    row["disrupted_layers"] = int((lc > 0.01).sum())
                    row["total_layers"] = len(lc)
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

    def _compute_attention_change(self, attn_ref, attn_alt):
        diff = attn_alt.mean(axis=1) - attn_ref.mean(axis=1)
        return {
            "attention_score_change": round(float(np.sqrt((diff**2).sum())), 6),
            "max_attention_shift": round(float(np.abs(diff).max()), 6),
            "disrupted_layers": int((np.abs(diff).mean(axis=(1, 2)) > 0.01).sum()),
            "total_layers": diff.shape[0],
        }

    # ------------------------------------------------------------------
    # Internal: single-GPU scoring (memory-safe, with tqdm)
    # ------------------------------------------------------------------
    def _score_sequences_single_gpu(
        self, ref_seqs, alt_seqs, model_names,
        batch_size=1, return_attention=False, verbose=False, device=None,
    ):
        device = device or self.device
        results = []

        model_pbar = tqdm(model_names, desc="Models", unit="model")
        for model_idx, name in enumerate(model_pbar):
            model_pbar.set_postfix(current=name)
            info = self.registry.get(name)
            model, tokenizer = self._load_model_to_device(name, device)

            n_batches = math.ceil(len(ref_seqs) / batch_size)
            batch_pbar = tqdm(
                range(n_batches),
                desc=f"  {name}",
                unit="batch",
                leave=False,
            )

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
                    scores = _compute_scores(rp[i], ap[i])
                    row = {
                        "_var_idx": batch_start + i,
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
                        row.update(self._compute_attention_change(ra_list[i], aa_list[i]))
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
                verbose=verbose, device=device)

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
        return df

    def score_variants(
        self, variants: pd.DataFrame,
        models=None, model_type=None, flank: int = 150,
        batch_size: int = 32, gpus=None,
        return_attention: bool = False, verbose: bool = False,
    ) -> pd.DataFrame:
        """Score multiple variants from a DataFrame (columns: chrom, pos, ref, alt)."""
        required = {"chrom", "pos", "ref", "alt"}
        if not required.issubset(variants.columns):
            raise ValueError(f"Missing columns: {required - set(variants.columns)}")

        variant_list = variants[["chrom", "pos", "ref", "alt"]].to_dict("records")

        # Sanity check
        self._run_sanity_check(variant_list)

        # Adjust positions to 0-based
        adjusted = []
        for v in variant_list:
            adjusted.append({
                "chrom": v["chrom"],
                "pos": self._adjust_pos(v["pos"]),
                "ref": v["ref"],
                "alt": v["alt"],
            })

        # Parallel sequence extraction with tqdm
        print(f"Extracting sequences for {len(variants)} variants...")
        seq_pairs = extract_variant_sequences_batch(
            self._genome_path, adjusted, flank
        )

        valid_indices, ref_seqs, alt_seqs = [], [], []
        for i, (r, a) in enumerate(seq_pairs):
            if r is not None and a is not None:
                valid_indices.append(i)
                ref_seqs.append(r)
                alt_seqs.append(a)

        if len(valid_indices) < len(variants):
            n_failed = len(variants) - len(valid_indices)
            warnings.warn(f"Failed to extract sequences for {n_failed} variants")
        if not ref_seqs:
            return pd.DataFrame()

        model_names = self._resolve_models(models, model_type)
        print(f"Scoring {len(ref_seqs)} variants × {len(model_names)} models...")

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
                verbose=verbose, device=device)

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
