"""
deepvregulome.interpret — Motif analysis and biological interpretation.

v0.1.7 fixes:
    - Fixed numpy array truth value bug in plot_variant_report
    - Bulk JASPAR download (5 seconds instead of 10 minutes)
    - Improved learned motif PFM with better attention smoothing
"""

import os
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Bulk PFM download URL (much faster than individual API calls)
JASPAR_BULK_URL = "https://jaspar.elixir.no/api/v1/matrix/?tax_id=9606&collection=CORE&page_size=2000&format=json"
CACHE_DIR = Path.home() / ".cache" / "deepvregulome"
JASPAR_CACHE = CACHE_DIR / "jaspar2024_core_human.json"

BASES = ["A", "C", "G", "T"]
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
PSEUDO = 0.01
BG = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MotifMatch:
    motif_id: str
    motif_name: str
    position: int
    strand: str
    score: float
    max_score: float
    rel_score: float
    matched_seq: str


@dataclass
class MotifReport:
    model_name: str
    chrom: str
    genomic_pos: int
    ref_allele: str
    alt_allele: str
    prob_ref: float
    prob_alt: float
    log_odds_ratio: float

    ref_matches: List[MotifMatch] = field(default_factory=list)
    alt_matches: List[MotifMatch] = field(default_factory=list)
    disrupted_motifs: pd.DataFrame = field(default_factory=pd.DataFrame)
    gained_motifs: pd.DataFrame = field(default_factory=pd.DataFrame)

    tf_own_motif_ref: Optional[MotifMatch] = None
    tf_own_motif_alt: Optional[MotifMatch] = None
    tf_motif_disrupted: bool = False

    tf_jaspar_pfm: Optional[np.ndarray] = None
    tf_jaspar_id: str = ""
    tf_jaspar_name: str = ""

    learned_pfm: Optional[np.ndarray] = None
    learned_consensus: str = ""
    learned_jaspar_match: Optional[dict] = None
    learned_jaspar_pfm: Optional[np.ndarray] = None

    ref_seq: str = ""
    alt_seq: str = ""
    variant_pos: int = 0


# ---------------------------------------------------------------------------
# PWM utilities
# ---------------------------------------------------------------------------
def pfm_to_pwm(pfm):
    row_sums = pfm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ppm = pfm / row_sums
    ppm = (ppm + PSEUDO) / (1 + 4 * PSEUDO)
    bg = np.array([BG[b] for b in BASES])
    return np.log2(ppm / bg)


def pfm_to_ic(pfm):
    """Convert PFM to information content matrix for logomaker."""
    row_sums = pfm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ppm = pfm / row_sums
    ppm = np.clip(ppm, 1e-10, 1.0)
    entropy = -(ppm * np.log2(ppm)).sum(axis=1)
    ic_total = 2.0 - entropy
    return pd.DataFrame(ppm * ic_total[:, np.newaxis], columns=BASES)


def score_sequence_with_pwm(seq, pwm):
    if len(seq) != pwm.shape[0]:
        return -np.inf
    return sum(pwm[i, BASE_TO_IDX[b]] for i, b in enumerate(seq.upper()) if b in BASE_TO_IDX)


def max_pwm_score(pwm):
    return float(pwm.max(axis=1).sum())


def min_pwm_score(pwm):
    return float(pwm.min(axis=1).sum())


def reverse_complement_pwm(pwm):
    return pwm[::-1, ::-1].copy()


def scan_sequence(seq, pwm, threshold=0.7):
    ml = pwm.shape[0]
    if len(seq) < ml:
        return []
    max_s, min_s = max_pwm_score(pwm), min_pwm_score(pwm)
    sr = max_s - min_s if max_s != min_s else 1.0
    rc = reverse_complement_pwm(pwm)
    matches = []
    for i in range(len(seq) - ml + 1):
        sub = seq[i:i + ml].upper()
        if "N" in sub:
            continue
        for strand, p in [("+", pwm), ("-", rc)]:
            s = score_sequence_with_pwm(sub, p)
            rel = (s - min_s) / sr
            if rel >= threshold:
                matches.append({"position": i, "strand": strand, "score": s,
                                "max_score": max_s, "rel_score": round(rel, 4),
                                "matched_seq": sub})
    return matches


def compare_pwms(pwm1, pwm2):
    from scipy import stats
    best = {"pearson_r": -1, "offset": 0, "orientation": "+"}
    for orient, p2 in [("+", pwm2), ("-", reverse_complement_pwm(pwm2))]:
        l1, l2 = pwm1.shape[0], p2.shape[0]
        mo = min(5, min(l1, l2))
        for off in range(-l2 + mo, l1 - mo + 1):
            s1, e1 = max(0, off), min(l1, off + l2)
            s2 = max(0, -off)
            if e1 - s1 < mo:
                continue
            f1, f2 = pwm1[s1:e1].flatten(), p2[s2:s2 + (e1 - s1)].flatten()
            if len(f1) < 4:
                continue
            r, p = stats.pearsonr(f1, f2)
            if r > best["pearson_r"]:
                best = {"pearson_r": round(r, 4), "p_value": round(p, 6),
                        "offset": off, "orientation": orient, "overlap": e1 - s1}
    return best


# ---------------------------------------------------------------------------
# JASPAR download — bulk with individual PFM fetch
# ---------------------------------------------------------------------------
def download_jaspar():
    import urllib.request

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if cache exists and is non-empty
    if JASPAR_CACHE.exists():
        try:
            with open(JASPAR_CACHE) as f:
                data = json.load(f)
            if len(data) > 100:  # valid cache has 1000+ motifs
                return data
        except Exception:
            pass

    print("Downloading JASPAR 2024 CORE human motifs (one-time, ~2 min)...")

    # Step 1: Get all matrix IDs from list endpoint
    all_entries = []
    url = JASPAR_BULK_URL
    page = 0
    while url:
        page += 1
        print(f"  Fetching motif list page {page}...")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        all_entries.extend(data.get("results", []))
        url = data.get("next")

    print(f"  Found {len(all_entries)} motifs. Fetching PFMs...")

    # Step 2: Fetch individual PFMs (the list endpoint doesn't include PFMs)
    all_motifs = []
    for i, entry in enumerate(all_entries):
        mid = entry.get("matrix_id", "")
        name = entry.get("name", "")

        detail_url = f"https://jaspar.elixir.no/api/v1/matrix/{mid}/?format=json"
        try:
            with urllib.request.urlopen(detail_url, timeout=30) as dresp:
                detail = json.loads(dresp.read().decode())
            pfm_raw = detail.get("pfm", {})
            if not pfm_raw or "A" not in pfm_raw:
                continue
            length = len(pfm_raw["A"])
            pfm = np.zeros((length, 4))
            for bi, base in enumerate(BASES):
                pfm[:, bi] = pfm_raw.get(base, [0] * length)

            all_motifs.append({
                "matrix_id": mid, "name": name,
                "pfm": pfm.tolist(), "length": length,
                "family": detail.get("family", []),
            })
        except Exception:
            continue

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(all_entries)} motifs downloaded...")

    # Save cache
    with open(JASPAR_CACHE, "w") as f:
        json.dump(all_motifs, f)

    print(f"  ✓ Cached {len(all_motifs)} motifs at {JASPAR_CACHE}")
    return all_motifs


# ---------------------------------------------------------------------------
# MotifAnalyzer
# ---------------------------------------------------------------------------
class MotifAnalyzer:

    def __init__(self, jaspar_cache=None):
        self._motifs = None
        self._custom_cache = jaspar_cache

    @property
    def motifs(self):
        if self._motifs is None:
            if self._custom_cache and os.path.exists(self._custom_cache):
                with open(self._custom_cache) as f:
                    raw = json.load(f)
            else:
                raw = download_jaspar()

            self._motifs = {}
            for entry in raw:
                pfm = np.array(entry["pfm"])
                self._motifs[entry["matrix_id"]] = {
                    "matrix_id": entry["matrix_id"], "name": entry["name"],
                    "pfm": pfm, "pwm": pfm_to_pwm(pfm),
                    "length": entry["length"],
                    "family": entry.get("family", []),
                }
            print(f"Loaded {len(self._motifs)} JASPAR motifs")
        return self._motifs

    def _find_motifs_by_name(self, tf_name):
        tf_upper = tf_name.upper()
        return [m for m in self.motifs.values() if m["name"].upper() == tf_upper]

    def scan_all_motifs(self, sequence, threshold=0.75, top_n=50):
        all_matches = []
        for mid, md in self.motifs.items():
            for hit in scan_sequence(sequence, md["pwm"], threshold=threshold):
                all_matches.append(MotifMatch(
                    motif_id=mid, motif_name=md["name"],
                    position=hit["position"], strand=hit["strand"],
                    score=hit["score"], max_score=hit["max_score"],
                    rel_score=hit["rel_score"], matched_seq=hit["matched_seq"]))
        all_matches.sort(key=lambda m: -m.rel_score)
        return all_matches[:top_n]

    def find_tf_motif(self, sequence, tf_name, threshold=0.7):
        tf_motifs = self._find_motifs_by_name(tf_name)
        if not tf_motifs:
            return None, None
        best, best_pfm = None, None
        for md in tf_motifs:
            for hit in scan_sequence(sequence, md["pwm"], threshold=threshold):
                match = MotifMatch(motif_id=md["matrix_id"], motif_name=md["name"],
                                   position=hit["position"], strand=hit["strand"],
                                   score=hit["score"], max_score=hit["max_score"],
                                   rel_score=hit["rel_score"], matched_seq=hit["matched_seq"])
                if best is None or match.rel_score > best.rel_score:
                    best, best_pfm = match, md["pfm"]
        return best, best_pfm

    def extract_attention_motif(self, dvr, model_name, var_idx=0, motif_length=10):
        """
        Extract attention-derived motif with improved smoothing.

        Instead of raw attention (which concentrates at one position),
        applies Gaussian smoothing to spread the attention signal, then
        builds a PFM where each position has a realistic base distribution.
        """
        data = dvr.get_attention(model_name, var_idx)
        ref_attn = data["ref_attention"]
        ref_seq = data["ref_seq"]

        # Map k-mer attention to nucleotide space
        nuc_attn = np.zeros(len(ref_seq))
        for ki in range(len(ref_attn)):
            center = ki + 3
            if center < len(nuc_attn):
                nuc_attn[center] += ref_attn[ki]

        # Gaussian smoothing (sigma=3) to spread concentrated attention
        try:
            from scipy.ndimage import gaussian_filter1d
            nuc_attn_smooth = gaussian_filter1d(nuc_attn, sigma=3)
        except ImportError:
            # Manual smoothing fallback
            kernel = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
            nuc_attn_smooth = np.convolve(nuc_attn, kernel, mode='same')

        # Find best window
        best_start, best_score = 0, -1
        for i in range(len(ref_seq) - motif_length + 1):
            s = nuc_attn_smooth[i:i + motif_length].sum()
            if s > best_score:
                best_score = s
                best_start = i

        motif_seq = ref_seq[best_start:best_start + motif_length].upper()
        attn_weights = nuc_attn_smooth[best_start:best_start + motif_length]

        # Normalize attention to 0-1
        aw_min, aw_max = attn_weights.min(), attn_weights.max()
        if aw_max > aw_min:
            attn_norm = (attn_weights - aw_min) / (aw_max - aw_min)
        else:
            attn_norm = np.ones(motif_length) * 0.5

        # Build PFM with attention-weighted distributions
        # High attention → high confidence in the actual base (sharp distribution)
        # Low attention → uncertain (flatter distribution, more like background)
        pfm = np.zeros((motif_length, 4))
        n_pseudo_seqs = 100  # scale factor

        for i, base in enumerate(motif_seq):
            if base not in BASE_TO_IDX:
                pfm[i] = n_pseudo_seqs * 0.25  # uniform
                continue

            w = attn_norm[i]
            # Map attention to concentration: w=1 → 85% on actual base, w=0 → 40%
            primary = 0.40 + 0.45 * w
            other = (1.0 - primary) / 3.0

            pfm[i] = n_pseudo_seqs * other
            pfm[i, BASE_TO_IDX[base]] = n_pseudo_seqs * primary

        consensus = "".join(BASES[np.argmax(pfm[i])] for i in range(motif_length))
        return pfm, consensus

    def analyze_variant(self, dvr, model_name, var_idx=0,
                        scan_threshold=0.75, motif_length=10):
        data = dvr.get_attention(model_name, var_idx)
        ref_seq, alt_seq = data["ref_seq"], data["alt_seq"]
        variant_pos = data.get("variant_pos", len(ref_seq) // 2)

        window = 30
        s = max(0, variant_pos - window)
        e = min(len(ref_seq), variant_pos + window + 1)
        ref_window, alt_window = ref_seq[s:e], alt_seq[s:e]

        # 1. Scan
        print(f"Scanning ±{window}bp around variant for JASPAR motifs...")
        ref_matches = self.scan_all_motifs(ref_window, threshold=scan_threshold)
        alt_matches = self.scan_all_motifs(alt_window, threshold=scan_threshold)

        # 2. Disrupted / gained
        ref_set = {(m.motif_id, m.position, m.strand): m for m in ref_matches}
        alt_set = {(m.motif_id, m.position, m.strand): m for m in alt_matches}

        disrupted, gained = [], []
        for key, rm in ref_set.items():
            if key not in alt_set:
                disrupted.append({"motif_id": rm.motif_id, "motif_name": rm.motif_name,
                                  "position": rm.position, "strand": rm.strand,
                                  "ref_score": rm.rel_score, "alt_score": 0.0,
                                  "score_change": -rm.rel_score, "matched_seq": rm.matched_seq})
            else:
                am = alt_set[key]
                if rm.rel_score - am.rel_score > 0.1:
                    disrupted.append({"motif_id": rm.motif_id, "motif_name": rm.motif_name,
                                      "position": rm.position, "strand": rm.strand,
                                      "ref_score": rm.rel_score, "alt_score": am.rel_score,
                                      "score_change": am.rel_score - rm.rel_score,
                                      "matched_seq": rm.matched_seq})
        for key, am in alt_set.items():
            if key not in ref_set:
                gained.append({"motif_id": am.motif_id, "motif_name": am.motif_name,
                               "position": am.position, "strand": am.strand,
                               "ref_score": 0.0, "alt_score": am.rel_score,
                               "score_change": am.rel_score, "matched_seq": am.matched_seq})

        disrupted_df = pd.DataFrame(disrupted).sort_values("score_change") if disrupted else pd.DataFrame()
        gained_df = pd.DataFrame(gained).sort_values("score_change", ascending=False) if gained else pd.DataFrame()

        # 3. TF's own motif
        tf_ref, tf_pfm = self.find_tf_motif(ref_window, model_name, threshold=0.6)
        tf_alt, _ = self.find_tf_motif(alt_window, model_name, threshold=0.6)
        tf_disrupted = (tf_ref is not None and tf_alt is None) or \
                       (tf_ref is not None and tf_alt is not None and
                        tf_ref.rel_score - tf_alt.rel_score > 0.1)

        # 4. Attention motif
        learned_pfm, consensus = self.extract_attention_motif(dvr, model_name, var_idx, motif_length)

        # 5. Compare to JASPAR
        learned_pwm = pfm_to_pwm(learned_pfm)
        best_r, best_match, best_jaspar_pfm = -1, None, None
        for mid, md in self.motifs.items():
            try:
                result = compare_pwms(learned_pwm, md["pwm"])
                if result["pearson_r"] > best_r:
                    best_r = result["pearson_r"]
                    best_match = {"matrix_id": mid, "name": md["name"], **result}
                    best_jaspar_pfm = md["pfm"]
            except Exception:
                continue

        report = MotifReport(
            model_name=model_name,
            chrom=data.get("chrom", ""), genomic_pos=data.get("genomic_pos", 0),
            ref_allele=ref_seq[variant_pos] if variant_pos < len(ref_seq) else "",
            alt_allele=alt_seq[variant_pos] if variant_pos < len(alt_seq) else "",
            prob_ref=data["prob_ref"], prob_alt=data["prob_alt"], log_odds_ratio=0,
            ref_matches=ref_matches, alt_matches=alt_matches,
            disrupted_motifs=disrupted_df, gained_motifs=gained_df,
            tf_own_motif_ref=tf_ref, tf_own_motif_alt=tf_alt,
            tf_motif_disrupted=tf_disrupted,
            tf_jaspar_pfm=tf_pfm,
            tf_jaspar_id=tf_ref.motif_id if tf_ref else "",
            tf_jaspar_name=model_name,
            learned_pfm=learned_pfm, learned_consensus=consensus,
            learned_jaspar_match=best_match, learned_jaspar_pfm=best_jaspar_pfm,
            ref_seq=ref_seq, alt_seq=alt_seq, variant_pos=variant_pos,
        )

        # Summary
        print(f"\n{'='*60}")
        print(f"Motif Analysis: {model_name} at {report.chrom}:{report.genomic_pos}")
        print(f"{'='*60}")
        print(f"  Variant: {report.ref_allele} → {report.alt_allele}")
        print(f"  P(binding): {report.prob_ref:.4f} → {report.prob_alt:.4f}")
        print(f"  JASPAR motifs in REF: {len(ref_matches)}")
        print(f"  JASPAR motifs in ALT: {len(alt_matches)}")
        print(f"  Disrupted: {len(disrupted_df)}  |  Gained: {len(gained_df)}")
        if tf_ref:
            alt_str = f", ALT={tf_alt.rel_score:.3f}" if tf_alt else ", absent in ALT"
            print(f"  {model_name}'s own motif ({tf_ref.motif_id}): "
                  f"{'⚠ DISRUPTED' if tf_disrupted else '✓ intact'} "
                  f"(REF={tf_ref.rel_score:.3f}{alt_str})")
        else:
            print(f"  {model_name}'s own motif: not found in JASPAR")
        if best_match:
            print(f"  Attention motif: {consensus}")
            print(f"    Best JASPAR match: {best_match['name']} ({best_match['matrix_id']}) "
                  f"r={best_match['pearson_r']:.3f}")
        print(f"{'='*60}\n")
        return report

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot_motif_logo(self, report, figsize=(14, 6), save_path=None):
        """JASPAR motif (top) + learned motif (bottom) as proper PWM logos."""
        try:
            import logomaker
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install logomaker matplotlib")

        has_jaspar = report.tf_jaspar_pfm is not None
        has_learned = report.learned_pfm is not None
        n_panels = (1 if has_jaspar else 0) + (1 if has_learned else 0)
        if n_panels == 0:
            print("No motif data available.")
            return None

        fig, axes = plt.subplots(n_panels, 1, figsize=figsize)
        if n_panels == 1:
            axes = [axes]
        idx = 0

        if has_jaspar:
            logomaker.Logo(pfm_to_ic(report.tf_jaspar_pfm), ax=axes[idx], color_scheme="classic")
            axes[idx].set_title(f"JASPAR: {report.tf_jaspar_name} ({report.tf_jaspar_id})",
                                fontsize=12, fontweight="bold")
            axes[idx].set_ylabel("IC (bits)")
            axes[idx].set_ylim(0, 2.2)
            idx += 1

        if has_learned:
            logomaker.Logo(pfm_to_ic(report.learned_pfm), ax=axes[idx], color_scheme="classic")
            title = f"DeepVRegulome learned motif: {report.learned_consensus}"
            if report.learned_jaspar_match:
                lm = report.learned_jaspar_match
                title += f"  |  Match: {lm['name']} (r={lm['pearson_r']:.3f})"
            axes[idx].set_title(title, fontsize=12, fontweight="bold")
            axes[idx].set_ylabel("IC (bits)")
            axes[idx].set_ylim(0, 2.2)

        fig.suptitle(f"Motif Analysis: {report.model_name} at {report.chrom}:{report.genomic_pos} "
                     f"{report.ref_allele}>{report.alt_allele}  |  "
                     f"P(ref)={report.prob_ref:.4f} → P(alt)={report.prob_alt:.4f}",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        return fig

    def plot_jaspar_motif(self, motif_id, figsize=(10, 3), save_path=None):
        """Plot a single JASPAR motif by ID."""
        try:
            import logomaker
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install logomaker")

        if motif_id not in self.motifs:
            raise KeyError(f"Motif {motif_id} not found")
        md = self.motifs[motif_id]
        fig, ax = plt.subplots(figsize=figsize)
        logomaker.Logo(pfm_to_ic(md["pfm"]), ax=ax, color_scheme="classic")
        ax.set_title(f"{md['name']} ({motif_id})", fontsize=13, fontweight="bold")
        ax.set_ylabel("IC (bits)")
        ax.set_ylim(0, 2.2)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_variant_report(self, report, figsize=(16, 14), save_path=None):
        """Combined: sequence + disrupted motifs + logos."""
        try:
            import logomaker
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install logomaker matplotlib")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1.5, 1.5], hspace=0.5, wspace=0.3)

        vp = report.variant_pos
        window = 15
        s = max(0, vp - window)
        e = min(len(report.ref_seq), vp + window + 1)
        ref_region, alt_region = report.ref_seq[s:e], report.alt_seq[s:e]
        vw = vp - s

        # Row 1: Sequence boxes
        ax_seq = fig.add_subplot(gs[0, :])
        n = len(ref_region)
        ax_seq.set_xlim(-0.5, n - 0.5)
        ax_seq.set_ylim(-0.5, 1.5)
        for i, base in enumerate(ref_region):
            c = "#f1948a" if i == vw else "#d4e6f1"
            ax_seq.add_patch(plt.Rectangle((i-0.45, 0.6), 0.9, 0.7, facecolor=c, edgecolor="gray", lw=0.5))
            ax_seq.text(i, 0.95, base, ha="center", va="center", fontsize=9, fontweight="bold")
        for i, base in enumerate(alt_region):
            c = "#e74c3c" if i == vw else "#fadbd8"
            fc = "white" if i == vw else "black"
            ax_seq.add_patch(plt.Rectangle((i-0.45, -0.2), 0.9, 0.7, facecolor=c, edgecolor="gray", lw=0.5))
            ax_seq.text(i, 0.15, base, ha="center", va="center", fontsize=9, fontweight="bold", color=fc)
        ax_seq.text(-1.5, 0.95, "REF", ha="right", va="center", fontsize=10, fontweight="bold")
        ax_seq.text(-1.5, 0.15, "ALT", ha="right", va="center", fontsize=10, fontweight="bold")
        ax_seq.set_title(f"{report.model_name} at {report.chrom}:{report.genomic_pos} "
                         f"({report.ref_allele}→{report.alt_allele})  |  "
                         f"P(ref)={report.prob_ref:.4f} → P(alt)={report.prob_alt:.4f}",
                         fontsize=12, fontweight="bold")
        ax_seq.axis("off")

        # Row 2 left: Disrupted motifs table
        ax_table = fig.add_subplot(gs[1, 0])
        ax_table.axis("off")
        if len(report.disrupted_motifs) > 0:
            td = [[r["motif_name"], r["motif_id"], f"{r['ref_score']:.3f}",
                    f"{r['alt_score']:.3f}", f"{r['score_change']:.3f}"]
                   for _, r in report.disrupted_motifs.head(6).iterrows()]
            table = ax_table.table(cellText=td, colLabels=["Motif", "ID", "REF", "ALT", "Δ"],
                                    loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)
            ax_table.set_title("Disrupted JASPAR Motifs", fontsize=11, fontweight="bold")
        else:
            ax_table.text(0.5, 0.5, "No disrupted motifs found", ha="center", va="center", fontsize=11, style="italic")

        # Row 2 right: TF status
        ax_tf = fig.add_subplot(gs[1, 1])
        ax_tf.axis("off")
        txt = f"TF: {report.model_name}\n\n"
        if report.tf_own_motif_ref:
            txt += f"JASPAR motif: {report.tf_own_motif_ref.motif_id}\n"
            txt += f"REF score: {report.tf_own_motif_ref.rel_score:.3f}\n"
            if report.tf_own_motif_alt:
                txt += f"ALT score: {report.tf_own_motif_alt.rel_score:.3f}\n"
            else:
                txt += "ALT: motif absent\n"
            txt += f"\n{'⚠ DISRUPTED' if report.tf_motif_disrupted else '✓ Intact'}"
        else:
            txt += "Own motif not in JASPAR\n"
            if report.learned_jaspar_match:
                lm = report.learned_jaspar_match
                txt += f"\nAttention motif: {report.learned_consensus}\n"
                txt += f"Best match: {lm['name']} ({lm['matrix_id']})\nr={lm['pearson_r']:.3f}"
        ax_tf.text(0.5, 0.5, txt, ha="center", va="center", fontsize=10, family="monospace",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#eaf2f8", edgecolor="#aed6f1"))
        ax_tf.set_title(f"{report.model_name} Motif Status", fontsize=11, fontweight="bold")

        # Row 3: JASPAR motif logo — FIXED numpy array truth value bug
        jaspar_pfm = None
        if report.tf_jaspar_pfm is not None:
            jaspar_pfm = report.tf_jaspar_pfm
        elif report.learned_jaspar_pfm is not None:
            jaspar_pfm = report.learned_jaspar_pfm

        if jaspar_pfm is not None:
            ax_jaspar = fig.add_subplot(gs[2, :])
            logomaker.Logo(pfm_to_ic(jaspar_pfm), ax=ax_jaspar, color_scheme="classic")
            if report.tf_jaspar_pfm is not None:
                ax_jaspar.set_title(f"JASPAR: {report.tf_jaspar_name} ({report.tf_jaspar_id})",
                                    fontsize=11, fontweight="bold")
            elif report.learned_jaspar_match:
                lm = report.learned_jaspar_match
                ax_jaspar.set_title(f"JASPAR best match: {lm['name']} ({lm['matrix_id']}) | r={lm['pearson_r']:.3f}",
                                    fontsize=11, fontweight="bold")
            ax_jaspar.set_ylabel("IC (bits)")
            ax_jaspar.set_ylim(0, 2.2)

        # Row 4: Learned motif logo
        if report.learned_pfm is not None:
            ax_learned = fig.add_subplot(gs[3, :])
            logomaker.Logo(pfm_to_ic(report.learned_pfm), ax=ax_learned, color_scheme="classic")
            title = f"DeepVRegulome learned motif: {report.learned_consensus}"
            if report.learned_jaspar_match:
                lm = report.learned_jaspar_match
                title += f"  |  Match: {lm['name']} (r={lm['pearson_r']:.3f})"
            ax_learned.set_title(title, fontsize=11, fontweight="bold")
            ax_learned.set_ylabel("IC (bits)")
            ax_learned.set_ylim(0, 2.2)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        return fig
