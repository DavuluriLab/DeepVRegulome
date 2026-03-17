"""
deepvregulome.interpret — Motif analysis and biological interpretation.

Features:
    1. JASPAR motif scanning (PWM-based, pure Python)
    2. Attention-based motif extraction from DNABERT
    3. Motif comparison (Pearson correlation, like TOMTOM)
    4. Proper PWM-based web logo generation

Usage:
    from deepvregulome.interpret import MotifAnalyzer
    analyzer = MotifAnalyzer()
    report = analyzer.analyze_variant(dvr, model_name="ATF4")
    analyzer.plot_motif_logo(report)
"""

import os
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
JASPAR_URL = "https://jaspar.elixir.no/api/v1/matrix/?tax_id=9606&format=json&page_size=1000&collection=CORE"
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

    # JASPAR PFM for the TF's own motif (for logo plotting)
    tf_jaspar_pfm: Optional[np.ndarray] = None
    tf_jaspar_id: str = ""
    tf_jaspar_name: str = ""

    # Attention-derived motif
    learned_pfm: Optional[np.ndarray] = None     # [length, 4] weighted PFM
    learned_consensus: str = ""
    learned_jaspar_match: Optional[dict] = None
    learned_jaspar_pfm: Optional[np.ndarray] = None  # best matching JASPAR PFM

    ref_seq: str = ""
    alt_seq: str = ""
    variant_pos: int = 0


# ---------------------------------------------------------------------------
# PWM utilities
# ---------------------------------------------------------------------------
def pfm_to_pwm(pfm: np.ndarray) -> np.ndarray:
    row_sums = pfm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ppm = pfm / row_sums
    ppm = (ppm + PSEUDO) / (1 + 4 * PSEUDO)
    bg = np.array([BG[b] for b in BASES])
    return np.log2(ppm / bg)


def pfm_to_ic(pfm: np.ndarray) -> pd.DataFrame:
    """Convert PFM to information content matrix for logomaker."""
    row_sums = pfm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ppm = pfm / row_sums
    ppm = np.clip(ppm, 1e-10, 1.0)

    # Information content: IC_i = 2 + sum(p * log2(p))
    entropy = -(ppm * np.log2(ppm)).sum(axis=1)
    ic_total = 2.0 - entropy  # bits

    # Scale each base by its IC contribution
    ic_matrix = ppm * ic_total[:, np.newaxis]

    return pd.DataFrame(ic_matrix, columns=BASES)


def score_sequence_with_pwm(seq, pwm):
    if len(seq) != pwm.shape[0]:
        return -np.inf
    score = 0.0
    for i, base in enumerate(seq.upper()):
        if base in BASE_TO_IDX:
            score += pwm[i, BASE_TO_IDX[base]]
    return score


def max_pwm_score(pwm):
    return float(pwm.max(axis=1).sum())


def min_pwm_score(pwm):
    return float(pwm.min(axis=1).sum())


def reverse_complement_pwm(pwm):
    return pwm[::-1, ::-1].copy()


def scan_sequence(seq, pwm, threshold=0.7):
    motif_len = pwm.shape[0]
    if len(seq) < motif_len:
        return []

    max_s = max_pwm_score(pwm)
    min_s = min_pwm_score(pwm)
    score_range = max_s - min_s if max_s != min_s else 1.0
    rc_pwm = reverse_complement_pwm(pwm)

    matches = []
    for i in range(len(seq) - motif_len + 1):
        subseq = seq[i:i + motif_len].upper()
        if "N" in subseq:
            continue

        for strand, p in [("+", pwm), ("-", rc_pwm)]:
            s = score_sequence_with_pwm(subseq, p)
            rel = (s - min_s) / score_range
            if rel >= threshold:
                matches.append({
                    "position": i, "strand": strand,
                    "score": s, "max_score": max_s,
                    "rel_score": round(rel, 4), "matched_seq": subseq,
                })
    return matches


def compare_pwms(pwm1, pwm2):
    from scipy import stats

    best = {"pearson_r": -1, "offset": 0, "orientation": "+"}
    for orient, p2 in [("+", pwm2), ("-", reverse_complement_pwm(pwm2))]:
        l1, l2 = pwm1.shape[0], p2.shape[0]
        min_ov = min(5, min(l1, l2))

        for off in range(-l2 + min_ov, l1 - min_ov + 1):
            s1, e1 = max(0, off), min(l1, off + l2)
            s2 = max(0, -off)
            if e1 - s1 < min_ov:
                continue
            f1 = pwm1[s1:e1].flatten()
            f2 = p2[s2:s2 + (e1 - s1)].flatten()
            if len(f1) < 4:
                continue
            r, p = stats.pearsonr(f1, f2)
            if r > best["pearson_r"]:
                best = {"pearson_r": round(r, 4), "p_value": round(p, 6),
                        "offset": off, "orientation": orient, "overlap": e1 - s1}
    return best


# ---------------------------------------------------------------------------
# JASPAR database
# ---------------------------------------------------------------------------
def download_jaspar():
    import urllib.request
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if JASPAR_CACHE.exists():
        with open(JASPAR_CACHE) as f:
            return json.load(f)

    print("Downloading JASPAR 2024 CORE human motifs (one-time)...")
    all_motifs = []
    url = JASPAR_URL

    while url:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        for entry in data.get("results", []):
            pfm_raw = entry.get("pfm", {})
            if not pfm_raw:
                continue
            length = len(pfm_raw.get("A", []))
            if length == 0:
                continue
            pfm = np.zeros((length, 4))
            for bi, base in enumerate(BASES):
                pfm[:, bi] = pfm_raw.get(base, [0] * length)

            all_motifs.append({
                "matrix_id": entry.get("matrix_id", ""),
                "name": entry.get("name", ""),
                "pfm": pfm.tolist(),
                "length": length,
                "family": entry.get("family", []),
            })
        url = data.get("next")

    with open(JASPAR_CACHE, "w") as f:
        json.dump(all_motifs, f)

    print(f"  Downloaded {len(all_motifs)} motifs → cached at {JASPAR_CACHE}")
    return all_motifs


# ---------------------------------------------------------------------------
# MotifAnalyzer
# ---------------------------------------------------------------------------
class MotifAnalyzer:

    def __init__(self, jaspar_cache=None):
        self._jaspar_raw = None
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
                pwm = pfm_to_pwm(pfm)
                mid = entry["matrix_id"]
                self._motifs[mid] = {
                    "matrix_id": mid, "name": entry["name"],
                    "pfm": pfm, "pwm": pwm, "length": entry["length"],
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
                    rel_score=hit["rel_score"], matched_seq=hit["matched_seq"],
                ))
        all_matches.sort(key=lambda m: -m.rel_score)
        return all_matches[:top_n]

    def find_tf_motif(self, sequence, tf_name, threshold=0.7):
        tf_motifs = self._find_motifs_by_name(tf_name)
        if not tf_motifs:
            return None, None

        best = None
        best_pfm = None
        best_mid = None
        for md in tf_motifs:
            for hit in scan_sequence(sequence, md["pwm"], threshold=threshold):
                match = MotifMatch(
                    motif_id=md["matrix_id"], motif_name=md["name"],
                    position=hit["position"], strand=hit["strand"],
                    score=hit["score"], max_score=hit["max_score"],
                    rel_score=hit["rel_score"], matched_seq=hit["matched_seq"],
                )
                if best is None or match.rel_score > best.rel_score:
                    best = match
                    best_pfm = md["pfm"]
        return best, best_pfm

    def extract_attention_motif(self, dvr, model_name, var_idx=0, motif_length=10):
        data = dvr.get_attention(model_name, var_idx)
        ref_attn = data["ref_attention"]
        ref_seq = data["ref_seq"]

        # Map k-mer attention to nucleotide space
        nuc_attn = np.zeros(len(ref_seq))
        for ki in range(len(ref_attn)):
            center = ki + 3
            if center < len(nuc_attn):
                nuc_attn[center] += ref_attn[ki]

        # Find region with highest attention
        best_start, best_score = 0, -1
        for i in range(len(ref_seq) - motif_length + 1):
            s = nuc_attn[i:i + motif_length].sum()
            if s > best_score:
                best_score = s
                best_start = i

        motif_seq = ref_seq[best_start:best_start + motif_length].upper()
        attn_weights = nuc_attn[best_start:best_start + motif_length]
        attn_weights = attn_weights / (attn_weights.max() + 1e-8)

        # Build attention-weighted PFM
        # For each position, create a distribution biased by the actual base
        # and spread by (1 - attention_weight) to other bases
        pfm = np.zeros((motif_length, 4))
        for i, base in enumerate(motif_seq):
            if base not in BASE_TO_IDX:
                pfm[i] = 0.25  # uniform for N
                continue
            w = attn_weights[i]
            # High attention → sharper distribution toward actual base
            # Low attention → more uniform
            primary_weight = 0.4 + 0.6 * w  # ranges from 0.4 to 1.0
            other_weight = (1.0 - primary_weight) / 3.0
            pfm[i] = other_weight
            pfm[i, BASE_TO_IDX[base]] = primary_weight

        # Scale to counts (like a PFM with 100 sequences)
        pfm = pfm * 100

        consensus = "".join(BASES[np.argmax(pfm[i])] for i in range(motif_length))
        return pfm, consensus

    def analyze_variant(self, dvr, model_name, var_idx=0,
                        scan_threshold=0.75, motif_length=10):
        data = dvr.get_attention(model_name, var_idx)
        ref_seq = data["ref_seq"]
        alt_seq = data["alt_seq"]
        variant_pos = data.get("variant_pos", len(ref_seq) // 2)

        window = 30
        s = max(0, variant_pos - window)
        e = min(len(ref_seq), variant_pos + window + 1)
        ref_window = ref_seq[s:e]
        alt_window = alt_seq[s:e]

        # 1. Scan JASPAR motifs
        print(f"Scanning ±{window}bp around variant for JASPAR motifs...")
        ref_matches = self.scan_all_motifs(ref_window, threshold=scan_threshold)
        alt_matches = self.scan_all_motifs(alt_window, threshold=scan_threshold)

        # 2. Find disrupted/gained
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

        # 4. Attention-derived motif
        learned_pfm, consensus = self.extract_attention_motif(dvr, model_name, var_idx, motif_length)

        # 5. Compare learned motif to JASPAR
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

        # Build report
        report = MotifReport(
            model_name=model_name,
            chrom=data.get("chrom", ""),
            genomic_pos=data.get("genomic_pos", 0),
            ref_allele=ref_seq[variant_pos] if variant_pos < len(ref_seq) else "",
            alt_allele=alt_seq[variant_pos] if variant_pos < len(alt_seq) else "",
            prob_ref=data["prob_ref"], prob_alt=data["prob_alt"],
            log_odds_ratio=0,
            ref_matches=ref_matches, alt_matches=alt_matches,
            disrupted_motifs=disrupted_df, gained_motifs=gained_df,
            tf_own_motif_ref=tf_ref, tf_own_motif_alt=tf_alt,
            tf_motif_disrupted=tf_disrupted,
            tf_jaspar_pfm=tf_pfm,
            tf_jaspar_id=tf_ref.motif_id if tf_ref else "",
            tf_jaspar_name=model_name,
            learned_pfm=learned_pfm,
            learned_consensus=consensus,
            learned_jaspar_match=best_match,
            learned_jaspar_pfm=best_jaspar_pfm,
            ref_seq=ref_seq, alt_seq=alt_seq, variant_pos=variant_pos,
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"Motif Analysis: {model_name} at {report.chrom}:{report.genomic_pos}")
        print(f"{'='*60}")
        print(f"  Variant: {report.ref_allele} → {report.alt_allele}")
        print(f"  P(binding): {report.prob_ref:.4f} → {report.prob_alt:.4f}")
        print(f"  JASPAR motifs in REF: {len(ref_matches)}")
        print(f"  JASPAR motifs in ALT: {len(alt_matches)}")
        print(f"  Disrupted: {len(disrupted_df)}  |  Gained: {len(gained_df)}")

        if tf_ref:
            print(f"  {model_name}'s own motif ({tf_ref.motif_id}): "
                  f"{'⚠ DISRUPTED' if tf_disrupted else '✓ intact'} "
                  f"(REF={tf_ref.rel_score:.3f}"
                  f"{', ALT=' + f'{tf_alt.rel_score:.3f}' if tf_alt else ', absent in ALT'})")
        else:
            print(f"  {model_name}'s own motif: not found in JASPAR")

        if best_match:
            print(f"  Attention motif: {consensus}")
            print(f"    Best JASPAR match: {best_match['name']} ({best_match['matrix_id']}) "
                  f"r={best_match['pearson_r']:.3f}")
        print(f"{'='*60}\n")

        return report

    # ------------------------------------------------------------------
    # Visualization: Proper PWM logos
    # ------------------------------------------------------------------
    def plot_motif_logo(self, report, figsize=(14, 6), save_path=None):
        """
        Plot JASPAR motif PWM logo and attention-derived motif logo.

        Top: JASPAR known motif (from the TF's own entry or best match)
        Bottom: DeepVRegulome attention-derived motif
        Both are proper PWM logos with letter heights = information content.
        """
        try:
            import logomaker
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("logomaker and matplotlib required. "
                              "pip install logomaker matplotlib")

        # Determine which JASPAR PFM to show
        jaspar_pfm = None
        jaspar_label = ""

        if report.tf_jaspar_pfm is not None:
            jaspar_pfm = report.tf_jaspar_pfm
            jaspar_label = f"JASPAR: {report.tf_jaspar_name} ({report.tf_jaspar_id})"
        elif report.learned_jaspar_pfm is not None and report.learned_jaspar_match:
            jaspar_pfm = report.learned_jaspar_pfm
            lm = report.learned_jaspar_match
            jaspar_label = (f"JASPAR best match: {lm['name']} ({lm['matrix_id']}) "
                           f"| r={lm['pearson_r']:.3f}")

        has_jaspar = jaspar_pfm is not None
        has_learned = report.learned_pfm is not None
        n_panels = (1 if has_jaspar else 0) + (1 if has_learned else 0)

        if n_panels == 0:
            print("No motif data available for logo plot.")
            return None

        fig, axes = plt.subplots(n_panels, 1, figsize=figsize)
        if n_panels == 1:
            axes = [axes]

        panel_idx = 0

        # Panel 1: JASPAR motif logo
        if has_jaspar:
            ic_df = pfm_to_ic(jaspar_pfm)
            logomaker.Logo(ic_df, ax=axes[panel_idx], color_scheme="classic")
            axes[panel_idx].set_title(jaspar_label, fontsize=12, fontweight="bold")
            axes[panel_idx].set_ylabel("IC (bits)")
            axes[panel_idx].set_ylim(0, 2.2)
            panel_idx += 1

        # Panel 2: Attention-derived motif logo
        if has_learned:
            ic_df = pfm_to_ic(report.learned_pfm)
            logomaker.Logo(ic_df, ax=axes[panel_idx], color_scheme="classic")

            title = f"DeepVRegulome learned motif: {report.learned_consensus}"
            if report.learned_jaspar_match:
                lm = report.learned_jaspar_match
                title += f"  |  Match: {lm['name']} (r={lm['pearson_r']:.3f})"
            axes[panel_idx].set_title(title, fontsize=12, fontweight="bold")
            axes[panel_idx].set_ylabel("IC (bits)")
            axes[panel_idx].set_ylim(0, 2.2)

        fig.suptitle(
            f"Motif Analysis: {report.model_name} at "
            f"{report.chrom}:{report.genomic_pos} "
            f"{report.ref_allele}>{report.alt_allele}  |  "
            f"P(ref)={report.prob_ref:.4f} → P(alt)={report.prob_alt:.4f}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        return fig

    def plot_jaspar_motif(self, motif_id, figsize=(10, 3), save_path=None):
        """
        Plot a single JASPAR motif logo by matrix ID.

        Usage:
            analyzer.plot_jaspar_motif("MA0833.3")
        """
        try:
            import logomaker
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("logomaker required. pip install logomaker")

        if motif_id not in self.motifs:
            raise KeyError(f"Motif {motif_id} not found in JASPAR database")

        md = self.motifs[motif_id]
        ic_df = pfm_to_ic(md["pfm"])

        fig, ax = plt.subplots(figsize=figsize)
        logomaker.Logo(ic_df, ax=ax, color_scheme="classic")
        ax.set_title(f"{md['name']} ({motif_id})", fontsize=13, fontweight="bold")
        ax.set_ylabel("IC (bits)")
        ax.set_ylim(0, 2.2)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Full variant report
    # ------------------------------------------------------------------
    def plot_variant_report(self, report, dvr=None, figsize=(16, 14), save_path=None):
        """
        Combined figure:
            Row 1: Sequence with attention coloring (REF and ALT)
            Row 2: Disrupted motifs table + TF status
            Row 3: JASPAR motif logo (top) + Learned motif logo (bottom)
        """
        try:
            import logomaker
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("logomaker and matplotlib required.")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1.5, 1.5], hspace=0.5, wspace=0.3)

        vp = report.variant_pos
        window = 15
        s = max(0, vp - window)
        e = min(len(report.ref_seq), vp + window + 1)
        ref_region = report.ref_seq[s:e]
        alt_region = report.alt_seq[s:e]
        vw = vp - s

        # --- Row 1: Sequence boxes ---
        ax_seq = fig.add_subplot(gs[0, :])
        n = len(ref_region)
        ax_seq.set_xlim(-0.5, n - 0.5)
        ax_seq.set_ylim(-0.5, 1.5)

        for i, base in enumerate(ref_region):
            c = "#f1948a" if i == vw else "#d4e6f1"
            rect = plt.Rectangle((i-0.45, 0.6), 0.9, 0.7, facecolor=c, edgecolor="gray", lw=0.5)
            ax_seq.add_patch(rect)
            ax_seq.text(i, 0.95, base, ha="center", va="center", fontsize=9, fontweight="bold")

        for i, base in enumerate(alt_region):
            c = "#e74c3c" if i == vw else "#fadbd8"
            fc = "white" if i == vw else "black"
            rect = plt.Rectangle((i-0.45, -0.2), 0.9, 0.7, facecolor=c, edgecolor="gray", lw=0.5)
            ax_seq.add_patch(rect)
            ax_seq.text(i, 0.15, base, ha="center", va="center", fontsize=9, fontweight="bold", color=fc)

        ax_seq.text(-1.5, 0.95, "REF", ha="right", va="center", fontsize=10, fontweight="bold")
        ax_seq.text(-1.5, 0.15, "ALT", ha="right", va="center", fontsize=10, fontweight="bold")
        ax_seq.set_title(f"{report.model_name} at {report.chrom}:{report.genomic_pos} "
                         f"({report.ref_allele}→{report.alt_allele})  |  "
                         f"P(ref)={report.prob_ref:.4f} → P(alt)={report.prob_alt:.4f}",
                         fontsize=12, fontweight="bold")
        ax_seq.axis("off")

        # --- Row 2 left: Disrupted motifs table ---
        ax_table = fig.add_subplot(gs[1, 0])
        ax_table.axis("off")
        if len(report.disrupted_motifs) > 0:
            td = [[r["motif_name"], r["motif_id"], f"{r['ref_score']:.3f}",
                    f"{r['alt_score']:.3f}", f"{r['score_change']:.3f}"]
                   for _, r in report.disrupted_motifs.head(6).iterrows()]
            table = ax_table.table(cellText=td,
                                    colLabels=["Motif", "ID", "REF", "ALT", "Δ"],
                                    loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)
            ax_table.set_title("Disrupted JASPAR Motifs", fontsize=11, fontweight="bold")
        else:
            ax_table.text(0.5, 0.5, "No disrupted motifs found",
                          ha="center", va="center", fontsize=11, style="italic")

        # --- Row 2 right: TF status ---
        ax_tf = fig.add_subplot(gs[1, 1])
        ax_tf.axis("off")
        txt = f"TF: {report.model_name}\n\n"
        if report.tf_own_motif_ref:
            txt += f"JASPAR motif: {report.tf_own_motif_ref.motif_id}\n"
            txt += f"REF score: {report.tf_own_motif_ref.rel_score:.3f}\n"
            txt += f"ALT score: {report.tf_own_motif_alt.rel_score:.3f}\n" if report.tf_own_motif_alt else "ALT: absent\n"
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

        # --- Row 3: JASPAR motif logo ---
        jaspar_pfm = report.tf_jaspar_pfm or report.learned_jaspar_pfm
        if jaspar_pfm is not None:
            ax_jaspar = fig.add_subplot(gs[2, :])
            ic_df = pfm_to_ic(jaspar_pfm)
            logomaker.Logo(ic_df, ax=ax_jaspar, color_scheme="classic")
            if report.tf_jaspar_pfm is not None:
                ax_jaspar.set_title(f"JASPAR: {report.tf_jaspar_name} ({report.tf_jaspar_id})",
                                    fontsize=11, fontweight="bold")
            elif report.learned_jaspar_match:
                lm = report.learned_jaspar_match
                ax_jaspar.set_title(f"JASPAR best match: {lm['name']} ({lm['matrix_id']}) | r={lm['pearson_r']:.3f}",
                                    fontsize=11, fontweight="bold")
            ax_jaspar.set_ylabel("IC (bits)")
            ax_jaspar.set_ylim(0, 2.2)

        # --- Row 4: Learned motif logo ---
        if report.learned_pfm is not None:
            ax_learned = fig.add_subplot(gs[3, :])
            ic_df = pfm_to_ic(report.learned_pfm)
            logomaker.Logo(ic_df, ax=ax_learned, color_scheme="classic")
            title = f"DeepVRegulome learned motif: {report.learned_consensus}"
            ax_learned.set_title(title, fontsize=11, fontweight="bold")
            ax_learned.set_ylabel("IC (bits)")
            ax_learned.set_ylim(0, 2.2)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        return fig
