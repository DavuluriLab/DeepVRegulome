"""
DVR: Main interface for DeepVRegulome variant effect prediction.

v0.1.5:
    - Suppress tokenizer parallelism warnings
    - Fixed plots: same colormap/scale for REF and ALT
    - plot_sequence_attention() — nucleotide-colored attention
    - Auto-detect column names (CHROM/start/REF/ALT)
    - Motif analysis via deepvregulome.interpret
    - Position-level attention storage
    - tqdm progress bars
    - Multi-GPU, OOM-safe, batched inference
"""

import os
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

# Suppress tokenizer fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


def _attention_summary(attn_ref, attn_alt):
    ref_avg = attn_ref.mean(axis=1)
    alt_avg = attn_alt.mean(axis=1)
    diff = alt_avg - ref_avg
    return {
        "attention_score_change": round(float(np.sqrt((diff**2).sum())), 6),
        "max_attention_shift": round(float(np.abs(diff).max()), 6),
        "disrupted_layers": int((np.abs(diff).mean(axis=(1, 2)) > 0.01).sum()),
        "total_layers": diff.shape[0],
    }


def _position_attention(attn):
    return attn.mean(axis=(0, 1)).sum(axis=0)


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
            bs = batch_idx * batch_size
            be = min(bs + batch_size, len(ref_seqs))
            br = ref_seqs[bs:be]
            ba = alt_seqs[bs:be]

            ri = tokenizer([to_kmer(s) for s in br], return_tensors="pt",
                           max_length=512, truncation=True, padding=True)
            ai = tokenizer([to_kmer(s) for s in ba], return_tensors="pt",
                           max_length=512, truncation=True, padding=True)
            ri = {k: v.to(device) for k, v in ri.items()}
            ai = {k: v.to(device) for k, v in ai.items()}

            with torch.no_grad():
                ro = model(**ri, output_attentions=return_attention)
                ao = model(**ai, output_attentions=return_attention)

            rp = torch.softmax(ro.logits, dim=-1)[:, 1].cpu().numpy()
            ap = torch.softmax(ao.logits, dim=-1)[:, 1].cpu().numpy()

            for i in range(len(br)):
                scores = _compute_scores(float(rp[i]), float(ap[i]))
                row = {"_var_idx": bs + i, "model": name, "type": info.model_type,
                       "prob_ref": round(float(rp[i]), 6), "prob_alt": round(float(ap[i]), 6),
                       "log_odds_ratio": scores["log_odds_ratio"],
                       "score_change": scores["score_change"]}
                if verbose:
                    row["log_odds_ref"] = scores["_log_odds_ref"]
                    row["log_odds_alt"] = scores["_log_odds_alt"]
                if return_attention and ro.attentions and ao.attentions:
                    ra = torch.stack(ro.attentions)[:, i].cpu().numpy()
                    aa = torch.stack(ao.attentions)[:, i].cpu().numpy()
                    row.update(_attention_summary(ra, aa))
                all_results.append(row)

        del model
        torch.cuda.empty_cache()
        print(f"  [GPU {gpu_id}] ✓ {name} ({model_idx+1}/{len(model_names)})")

    result_queue.put(all_results)


class DVR:
    """
    DeepVRegulome: Score regulatory variant effects using fine-tuned DNABERT models.
    """

    def __init__(self, genome=None, device=None, cache_dir=None, coordinate_system=None):
        self.registry = ModelRegistry()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self._genome_path = genome
        self._genome = None
        self._coordinate_system = coordinate_system
        self._coord_offset = None
        self.last_attention: Dict[str, dict] = {}

    @property
    def genome(self):
        if self._genome is None:
            if self._genome_path is None:
                raise ValueError("Reference genome not provided.")
            try:
                import pysam
            except ImportError:
                raise ImportError("pysam required. pip install deepvregulome[genome]")
            self._genome = pysam.FastaFile(self._genome_path)
        return self._genome

    def _run_sanity_check(self, variants):
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

    def _adjust_pos(self, pos):
        if self._coord_offset is None:
            return pos - 1
        return pos - self._coord_offset

    # --- Model loading and prediction ---
    def _load_model_to_device(self, name, device):
        info = self.registry.get(name)
        tok = AutoTokenizer.from_pretrained(info.hf_repo, subfolder=info.subfolder, cache_dir=self.cache_dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            info.hf_repo, subfolder=info.subfolder, cache_dir=self.cache_dir, output_attentions=True)
        mdl = mdl.to(device).eval()
        return mdl, tok

    def _predict_single(self, model, tok, seq, device, return_attention=False):
        inp = tok(to_kmer(seq), return_tensors="pt", max_length=512, truncation=True, padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp, output_attentions=return_attention)
        r = {"prob": torch.softmax(out.logits, dim=-1)[0][1].item()}
        if return_attention and out.attentions:
            r["attention"] = torch.stack(out.attentions).squeeze(1).cpu().numpy()
        return r

    def _predict_batch(self, model, tok, seqs, device, return_attention=False):
        inp = tok([to_kmer(s) for s in seqs], return_tensors="pt", max_length=512, truncation=True, padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp, output_attentions=return_attention)
        probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        results = []
        for i in range(len(seqs)):
            r = {"prob": float(probs[i])}
            if return_attention and out.attentions:
                r["attention"] = torch.stack(out.attentions)[:, i].cpu().numpy()
            results.append(r)
        return results

    # --- Single-GPU scoring ---
    def _score_sequences_single_gpu(self, ref_seqs, alt_seqs, model_names,
                                     batch_size=1, return_attention=False,
                                     verbose=False, device=None):
        device = device or self.device
        results = []

        model_pbar = tqdm(model_names, desc="Models", unit="model")
        for name in model_pbar:
            model_pbar.set_postfix(current=name)
            info = self.registry.get(name)
            model, tok = self._load_model_to_device(name, device)

            n_batches = math.ceil(len(ref_seqs) / batch_size)
            batch_pbar = tqdm(range(n_batches), desc=f"  {name}", unit="batch", leave=False)

            for batch_idx in batch_pbar:
                bs = batch_idx * batch_size
                be = min(bs + batch_size, len(ref_seqs))
                br, ba = ref_seqs[bs:be], alt_seqs[bs:be]

                if batch_size == 1:
                    ro = self._predict_single(model, tok, br[0], device, return_attention)
                    ao = self._predict_single(model, tok, ba[0], device, return_attention)
                    rp, ap = [ro["prob"]], [ao["prob"]]
                    ra = [ro.get("attention")] if return_attention else [None]
                    aa = [ao.get("attention")] if return_attention else [None]
                else:
                    ros = self._predict_batch(model, tok, br, device, return_attention)
                    aos = self._predict_batch(model, tok, ba, device, return_attention)
                    rp = [r["prob"] for r in ros]
                    ap = [r["prob"] for r in aos]
                    ra = [r.get("attention") for r in ros] if return_attention else [None]*len(ros)
                    aa = [r.get("attention") for r in aos] if return_attention else [None]*len(aos)

                for i in range(len(br)):
                    vi = bs + i
                    scores = _compute_scores(rp[i], ap[i])
                    row = {"_var_idx": vi, "model": name, "type": info.model_type,
                           "prob_ref": round(rp[i], 6), "prob_alt": round(ap[i], 6),
                           "log_odds_ratio": scores["log_odds_ratio"],
                           "score_change": scores["score_change"]}
                    if verbose:
                        row["log_odds_ref"] = scores["_log_odds_ref"]
                        row["log_odds_alt"] = scores["_log_odds_alt"]

                    if return_attention and ra[i] is not None and aa[i] is not None:
                        row.update(_attention_summary(ra[i], aa[i]))
                        ref_pos = _position_attention(ra[i])
                        alt_pos = _position_attention(aa[i])
                        if name not in self.last_attention:
                            self.last_attention[name] = {}
                        self.last_attention[name][vi] = {
                            "ref_attention": ref_pos, "alt_attention": alt_pos,
                            "diff_attention": alt_pos - ref_pos,
                            "ref_raw": ra[i], "alt_raw": aa[i],
                            "ref_seq": br[i], "alt_seq": ba[i],
                            "prob_ref": rp[i], "prob_alt": ap[i],
                        }

                    results.append(row)

            del model, tok
            torch.cuda.empty_cache()

        return results

    # --- Multi-GPU scoring ---
    def _score_sequences_multi_gpu(self, ref_seqs, alt_seqs, model_names,
                                    gpus, batch_size=32, return_attention=False, verbose=False):
        import torch.multiprocessing as mp
        n = len(gpus)
        splits = [model_names[i*len(model_names)//n:(i+1)*len(model_names)//n] for i in range(n)]
        print(f"Distributing {len(model_names)} models across {n} GPUs: {[len(s) for s in splits]} each")

        mp.set_start_method("spawn", force=True)
        q = mp.Queue()
        procs = []
        for i, gid in enumerate(gpus):
            if not splits[i]: continue
            p = mp.Process(target=_gpu_worker, args=(
                gid, splits[i], ref_seqs, alt_seqs, batch_size,
                return_attention, verbose, self.cache_dir, q))
            p.start()
            procs.append(p)

        all_r = []
        for _ in procs:
            all_r.extend(q.get())
        for p in procs:
            p.join()
        return all_r

    # ==================================================================
    # PUBLIC API
    # ==================================================================
    def score_sequence(self, ref_seq, alt_seq, models=None, model_type=None,
                       batch_size=1, gpus=None, return_attention=False, verbose=False):
        if isinstance(ref_seq, str):
            ref_seqs, alt_seqs = [ref_seq], [alt_seq]
        else:
            ref_seqs, alt_seqs = list(ref_seq), list(alt_seq)

        model_names = self._resolve_models(models, model_type)
        if return_attention:
            self.last_attention = {}

        if gpus and len(gpus) > 1:
            raw = self._score_sequences_multi_gpu(ref_seqs, alt_seqs, model_names,
                                                   gpus=gpus, batch_size=batch_size,
                                                   return_attention=return_attention, verbose=verbose)
        else:
            device = f"cuda:{gpus[0]}" if gpus else self.device
            raw = self._score_sequences_single_gpu(ref_seqs, alt_seqs, model_names,
                                                    batch_size=batch_size, return_attention=return_attention,
                                                    verbose=verbose, device=device)

        df = pd.DataFrame(raw)
        if "_var_idx" in df.columns:
            df = df.drop(columns=["_var_idx"])
        if len(df) > 0:
            df = df.sort_values("log_odds_ratio", ascending=False, key=abs)
        return df.reset_index(drop=True)

    def score_variant(self, chrom, pos, ref, alt, models=None, model_type=None,
                      flank=150, batch_size=1, gpus=None, return_attention=False, verbose=False):
        self._run_sanity_check([{"chrom": chrom, "pos": pos, "ref": ref, "alt": alt}])
        pos_0 = self._adjust_pos(pos)
        ref_seq, alt_seq = extract_variant_sequences(self.genome, chrom, pos_0, ref, alt, flank=flank)

        df = self.score_sequence(ref_seq, alt_seq, models=models, model_type=model_type,
                                  batch_size=batch_size, gpus=gpus,
                                  return_attention=return_attention, verbose=verbose)
        df.insert(0, "chrom", chrom)
        df.insert(1, "pos", pos)
        df.insert(2, "ref", ref)
        df.insert(3, "alt", alt)

        if return_attention:
            for mn in self.last_attention:
                for vi in self.last_attention[mn]:
                    self.last_attention[mn][vi].update({
                        "ref_seq": ref_seq, "alt_seq": alt_seq,
                        "variant_pos": flank, "chrom": chrom, "genomic_pos": pos,
                    })
        return df

    def score_variants(self, variants, models=None, model_type=None, flank=150,
                       batch_size=32, gpus=None, return_attention=False, verbose=False):
        # Auto-detect column names
        col_map = {}
        for col in variants.columns:
            cl = col.lower().strip()
            if cl in ("chrom", "chr", "#chrom"): col_map[col] = "chrom"
            elif cl in ("pos", "start", "position"): col_map[col] = "pos"
            elif cl in ("ref", "reference", "ref_allele"): col_map[col] = "ref"
            elif cl in ("alt", "alternative", "alt_allele"): col_map[col] = "alt"
        if col_map:
            variants = variants.rename(columns=col_map)

        required = {"chrom", "pos", "ref", "alt"}
        if not required.issubset(variants.columns):
            raise ValueError(f"Missing columns: {required - set(variants.columns)}. "
                             f"Expected: chrom, pos, ref, alt (or CHROM, start, REF, ALT)")

        vlist = variants[["chrom", "pos", "ref", "alt"]].to_dict("records")
        self._run_sanity_check(vlist)

        adjusted = [{"chrom": v["chrom"], "pos": self._adjust_pos(v["pos"]),
                      "ref": v["ref"], "alt": v["alt"]} for v in vlist]

        print(f"Extracting sequences for {len(variants)} variants...")
        seq_pairs = extract_variant_sequences_batch(self._genome_path, adjusted, flank)

        vi_list, ref_seqs, alt_seqs = [], [], []
        for i, (r, a) in enumerate(seq_pairs):
            if r is not None and a is not None:
                vi_list.append(i)
                ref_seqs.append(r)
                alt_seqs.append(a)

        if len(vi_list) < len(variants):
            warnings.warn(f"Failed to extract sequences for {len(variants)-len(vi_list)} variants")
        if not ref_seqs:
            return pd.DataFrame()

        model_names = self._resolve_models(models, model_type)
        print(f"Scoring {len(ref_seqs)} variants × {len(model_names)} models...")

        if return_attention:
            self.last_attention = {}

        if gpus and len(gpus) > 1:
            raw = self._score_sequences_multi_gpu(ref_seqs, alt_seqs, model_names,
                                                   gpus=gpus, batch_size=batch_size,
                                                   return_attention=return_attention, verbose=verbose)
        else:
            device = f"cuda:{gpus[0]}" if gpus else self.device
            raw = self._score_sequences_single_gpu(ref_seqs, alt_seqs, model_names,
                                                    batch_size=batch_size, return_attention=return_attention,
                                                    verbose=verbose, device=device)

        df = pd.DataFrame(raw)
        if len(df) > 0 and "_var_idx" in df.columns:
            vi = variants.iloc[vi_list].reset_index(drop=True)
            df["chrom"] = df["_var_idx"].map(lambda i: vi.loc[i, "chrom"])
            df["pos"] = df["_var_idx"].map(lambda i: vi.loc[i, "pos"])
            df["ref"] = df["_var_idx"].map(lambda i: vi.loc[i, "ref"])
            df["alt"] = df["_var_idx"].map(lambda i: vi.loc[i, "alt"])
            df = df.drop(columns=["_var_idx"])
            cols = ["chrom","pos","ref","alt"] + [c for c in df.columns if c not in ["chrom","pos","ref","alt"]]
            df = df[cols].sort_values("log_odds_ratio", ascending=False, key=abs)
        return df.reset_index(drop=True)

    def score_vcf(self, vcf_path, models=None, model_type=None, flank=150,
                  batch_size=32, gpus=None, return_attention=False, verbose=False, max_variants=None):
        print(f"Parsing VCF: {vcf_path}")
        vlist = parse_vcf(vcf_path, max_variants=max_variants)
        print(f"  Found {len(vlist)} variants")
        if not vlist:
            return pd.DataFrame()
        return self.score_variants(pd.DataFrame(vlist), models=models, model_type=model_type,
                                    flank=flank, batch_size=batch_size, gpus=gpus,
                                    return_attention=return_attention, verbose=verbose)

    # ==================================================================
    # ATTENTION ACCESS AND VISUALIZATION
    # ==================================================================
    def get_attention(self, model_name, var_idx=0):
        if model_name not in self.last_attention:
            raise KeyError(f"No attention data for '{model_name}'. Available: {list(self.last_attention.keys())}")
        if var_idx not in self.last_attention[model_name]:
            raise KeyError(f"Variant index {var_idx} not found. Available: {list(self.last_attention[model_name].keys())}")
        return self.last_attention[model_name][var_idx]

    def plot_attention(self, model_name, var_idx=0, window=20, figsize=(14, 6), save_path=None):
        """Bar chart: REF vs ALT attention. FIXED: same color, same y-axis scale."""
        import matplotlib.pyplot as plt

        data = self.get_attention(model_name, var_idx)
        ref_attn, alt_attn, diff_attn = data["ref_attention"], data["alt_attention"], data["diff_attention"]
        vp = data.get("variant_pos", len(ref_attn)//2)
        kvp = max(0, vp - 5)
        s, e = max(0, kvp - window), min(len(ref_attn), kvp + window + 1)

        rw, aw, dw = ref_attn[s:e], alt_attn[s:e], diff_attn[s:e]
        pos = np.arange(s, e)

        # SAME y-axis scale
        y_max = max(rw.max(), aw.max()) * 1.1

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 2, 1.5]})

        # Same color (#3498db) for both REF and ALT
        axes[0].bar(pos, rw, color="#3498db", alpha=0.8, width=0.8)
        axes[0].axvline(x=kvp, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
        axes[0].set_ylim(0, y_max)
        axes[0].set_ylabel("Attention")
        axes[0].set_title(f"{model_name} — REF (wild-type)  |  P(binding)={data['prob_ref']:.4f}", fontsize=11)

        axes[1].bar(pos, aw, color="#3498db", alpha=0.8, width=0.8)
        axes[1].axvline(x=kvp, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
        axes[1].set_ylim(0, y_max)
        axes[1].set_ylabel("Attention")
        axes[1].set_title(f"{model_name} — ALT (mutant)  |  P(binding)={data['prob_alt']:.4f}", fontsize=11)

        colors_d = ["#e74c3c" if d < 0 else "#2ecc71" for d in dw]
        axes[2].bar(pos, dw, color=colors_d, alpha=0.8, width=0.8)
        axes[2].axvline(x=kvp, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
        axes[2].axhline(y=0, color="black", linewidth=0.5)
        axes[2].set_ylabel("Δ Attention")
        axes[2].set_xlabel("Position (k-mer index)")
        axes[2].set_title("Attention Change (ALT − REF)", fontsize=11)

        # Nucleotide labels
        ref_seq = data.get("ref_seq", "")
        if ref_seq and len(pos) <= 50:
            labels = [ref_seq[p+3] if p+3 < len(ref_seq) else "" for p in pos]
            axes[2].set_xticks(pos)
            axes[2].set_xticklabels(labels, fontsize=7)

        chrom = data.get("chrom", "")
        gpos = data.get("genomic_pos", "")
        fig.suptitle(f"DeepVRegulome Attention: {model_name} at {chrom}:{gpos}  |  ±{window}bp",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_attention_heatmap(self, model_name, var_idx=0, window=20, figsize=(12, 10), save_path=None):
        """Layer×position heatmap. FIXED: same scale for REF and ALT."""
        import matplotlib.pyplot as plt

        data = self.get_attention(model_name, var_idx)
        ref_raw, alt_raw = data["ref_raw"], data["alt_raw"]
        vp = data.get("variant_pos", ref_raw.shape[2]//2)
        kvp = max(0, vp - 5)
        s, e = max(0, kvp - window), min(ref_raw.shape[2], kvp + window + 1)

        rh = ref_raw.mean(axis=1)[:, s:e, s:e].sum(axis=2)
        ah = alt_raw.mean(axis=1)[:, s:e, s:e].sum(axis=2)

        # SAME scale for REF and ALT
        vmin = min(rh.min(), ah.min())
        vmax = max(rh.max(), ah.max())

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        im0 = axes[0].imshow(rh, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"REF — {model_name}")
        axes[0].set_ylabel("Layer")
        plt.colorbar(im0, ax=axes[0], shrink=0.6)

        im1 = axes[1].imshow(ah, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"ALT — {model_name}")
        plt.colorbar(im1, ax=axes[1], shrink=0.6)

        dh = ah - rh
        dmax = max(abs(dh.min()), abs(dh.max()))
        im2 = axes[2].imshow(dh, aspect="auto", cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[2].set_title("Δ (ALT − REF)")
        plt.colorbar(im2, ax=axes[2], shrink=0.6)

        vp_w = kvp - s
        for ax in axes:
            ax.axvline(x=vp_w, color="lime", linewidth=1.5, linestyle="--", alpha=0.8)
            ax.set_xlabel("Position")

        chrom = data.get("chrom", "")
        gpos = data.get("genomic_pos", "")
        fig.suptitle(f"Attention Heatmap: {model_name} at {chrom}:{gpos}  ±{window}bp",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_sequence_attention(self, model_name, var_idx=0, window=10, figsize=(16, 4), save_path=None):
        """Nucleotide-colored attention boxes. Same colormap and scale for REF and ALT."""
        import matplotlib.pyplot as plt

        data = self.get_attention(model_name, var_idx)
        ref_seq, alt_seq = data["ref_seq"], data["alt_seq"]
        ref_attn, alt_attn = data["ref_attention"], data["alt_attention"]
        vp = data.get("variant_pos", len(ref_seq)//2)
        p_ref, p_alt = data["prob_ref"], data["prob_alt"]
        chrom = data.get("chrom", "")
        gpos = data.get("genomic_pos", "")

        s = max(0, vp - window)
        e = min(len(ref_seq), vp + window + 1)
        ref_region = ref_seq[s:e]
        alt_region = alt_seq[s:e]

        def kmer_to_nuc(attn, seq_len, start):
            na = np.zeros(seq_len)
            ct = np.zeros(seq_len)
            for ki in range(len(attn)):
                sp = ki + 3
                if start <= sp < start + seq_len:
                    idx = sp - start
                    na[idx] += attn[ki]
                    ct[idx] += 1
            ct[ct == 0] = 1
            return na / ct

        rna = kmer_to_nuc(ref_attn, len(ref_region), s)
        ana = kmer_to_nuc(alt_attn, len(alt_region), s)

        # SAME scale
        gmax = max(rna.max(), ana.max()) + 1e-8
        rn = rna / gmax
        an = ana / gmax

        cmap = plt.cm.YlGnBu
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        vw = vp - s

        for ax, seq, norm, label, prob in [
            (axes[0], ref_region, rn, "Reference Sequence", p_ref),
            (axes[1], alt_region, an, "Alternative Sequence", p_alt),
        ]:
            ax.set_xlim(-0.5, len(seq) - 0.5)
            ax.set_ylim(-0.3, 0.8)
            ax.set_title(f"{label}  |  P(binding) = {prob:.4f}", fontsize=11)

            for i, (base, score) in enumerate(zip(seq, norm)):
                color = cmap(score)
                ec = "red" if i == vw else "gray"
                lw = 2.5 if i == vw else 0.5
                rect = plt.Rectangle((i-0.45, -0.15), 0.9, 0.75,
                                      linewidth=lw, edgecolor=ec, facecolor=color, alpha=0.9)
                ax.add_patch(rect)
                fc = "white" if score > 0.6 else "black"
                ax.text(i, 0.2, base, ha="center", va="center", fontsize=10, fontweight="bold", color=fc)

            ax.set_xticks(range(len(seq)))
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)

        # Genomic position labels
        if gpos:
            offset = data.get("variant_pos", 150)
            co = self._coord_offset or 1
            gs = gpos - offset + s + co
            labels = [str(gs + i) if i % 5 == 0 else "" for i in range(len(ref_region))]
            axes[1].set_xticks(range(len(ref_region)))
            axes[1].set_xticklabels(labels, fontsize=7, rotation=45)

        fig.suptitle(f"Attention: {model_name} at {chrom}:{gpos}  |  ±{window}bp  |  "
                     f"REF={ref_region[vw]}→ALT={alt_region[vw]}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ==================================================================
    # Helpers
    # ==================================================================
    def _resolve_models(self, models, model_type):
        if models and model_type:
            raise ValueError("Specify either 'models' or 'model_type', not both")
        if models:
            for m in models: self.registry.get(m)
            return models
        if model_type:
            return [m.name for m in self.registry.list(model_type=model_type)]
        raise ValueError("Must specify either 'models' or 'model_type'")

    def list_models(self, model_type=None):
        return self.registry.list(model_type=model_type)

    def search_models(self, query):
        return self.registry.search(query)

    def __repr__(self):
        g = f", genome='{self._genome_path}'" if self._genome_path else ""
        return f"DVR(device='{self.device}'{g}, {self.registry})"

    def __del__(self):
        if self._genome is not None:
            try: self._genome.close()
            except: pass
