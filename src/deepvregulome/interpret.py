"""
deepvregulome.interpret — Motif analysis and biological interpretation.

Features:
    1. JASPAR motif scanning (PWM-based, pure Python)
    2. Attention-based motif extraction from DNABERT
    3. Motif comparison (Pearson correlation, like TOMTOM)
    4. Web logo generation (using logomaker)

Usage:
    from deepvregulome.interpret import MotifAnalyzer

    analyzer = MotifAnalyzer()  # auto-downloads JASPAR on first use

    # After scoring with attention
    result = dvr.score_variant("chr9", 65385776, "G", "A",
                                models=["ATF4"], return_attention=True)

    # Full motif analysis
    report = analyzer.analyze_variant(dvr, model_name="ATF4")
    print(report.jaspar_matches)
    print(report.disrupted_motifs)
    analyzer.plot_motif_logo(report)
    analyzer.plot_variant_report(report)
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

# Pseudocount for PWM computation
PSEUDO = 0.01

# Background frequencies (uniform)
BG = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MotifMatch:
    """A single motif match at a specific position."""
    motif_id: str
    motif_name: str
    position: int       # position in sequence (0-based)
    strand: str         # "+" or "-"
    score: float        # PWM log-likelihood score
    max_score: float    # maximum possible score for this motif
    rel_score: float    # score / max_score (0-1)
    matched_seq: str    # the actual sequence at this position


@dataclass
class MotifReport:
    """Complete motif analysis report for a variant."""
    model_name: str
    chrom: str
    genomic_pos: int
    ref_allele: str
    alt_allele: str
    prob_ref: float
    prob_alt: float
    log_odds_ratio: float

    # JASPAR scanning results
    ref_matches: List[MotifMatch] = field(default_factory=list)
    alt_matches: List[MotifMatch] = field(default_factory=list)
    disrupted_motifs: pd.DataFrame = field(default_factory=pd.DataFrame)
    gained_motifs: pd.DataFrame = field(default_factory=pd.DataFrame)

    # TF-specific results
    tf_own_motif_ref: Optional[MotifMatch] = None
    tf_own_motif_alt: Optional[MotifMatch] = None
    tf_motif_disrupted: bool = False

    # Attention-derived motif
    learned_pwm: Optional[np.ndarray] = None    # [length, 4]
    learned_consensus: str = ""
    learned_jaspar_match: Optional[dict] = None  # best JASPAR match

    # Sequences
    ref_seq: str = ""
    alt_seq: str = ""
    variant_pos: int = 0


# ---------------------------------------------------------------------------
# PWM utilities
# ---------------------------------------------------------------------------
def pfm_to_pwm(pfm: np.ndarray) -> np.ndarray:
    """
    Convert Position Frequency Matrix to Position Weight Matrix (log-odds).

    Args:
        pfm: [length, 4] count matrix (A, C, G, T)

    Returns:
        pwm: [length, 4] log-odds matrix
    """
    # Normalize to probabilities
    row_sums = pfm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ppm = pfm / row_sums

    # Add pseudocount
    ppm = (ppm + PSEUDO) / (1 + 4 * PSEUDO)

    # Log-odds vs background
    bg = np.array([BG["A"], BG["C"], BG["G"], BG["T"]])
    pwm = np.log2(ppm / bg)

    return pwm


def score_sequence_with_pwm(seq: str, pwm: np.ndarray) -> float:
    """Score a sequence against a PWM."""
    if len(seq) != pwm.shape[0]:
        return -np.inf
    score = 0.0
    for i, base in enumerate(seq.upper()):
        if base in BASE_TO_IDX:
            score += pwm[i, BASE_TO_IDX[base]]
        else:
            score += 0  # N or other
    return score


def max_pwm_score(pwm: np.ndarray) -> float:
    """Maximum possible score for a PWM."""
    return float(pwm.max(axis=1).sum())


def min_pwm_score(pwm: np.ndarray) -> float:
    """Minimum possible score for a PWM."""
    return float(pwm.min(axis=1).sum())


def reverse_complement_pwm(pwm: np.ndarray) -> np.ndarray:
    """Reverse complement a PWM. Column order: A,C,G,T → T,G,C,A reversed."""
    return pwm[::-1, ::-1].copy()


def scan_sequence(seq: str, pwm: np.ndarray, threshold: float = 0.7) -> List[dict]:
    """
    Scan a sequence for PWM matches on both strands.

    Args:
        seq: DNA sequence
        pwm: [motif_len, 4] position weight matrix
        threshold: minimum relative score (0-1) to report

    Returns:
        List of matches with position, strand, score, rel_score
    """
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

        # Forward strand
        fwd_score = score_sequence_with_pwm(subseq, pwm)
        fwd_rel = (fwd_score - min_s) / score_range

        if fwd_rel >= threshold:
            matches.append({
                "position": i,
                "strand": "+",
                "score": fwd_score,
                "max_score": max_s,
                "rel_score": round(fwd_rel, 4),
                "matched_seq": subseq,
            })

        # Reverse strand
        rev_score = score_sequence_with_pwm(subseq, rc_pwm)
        rev_rel = (rev_score - min_s) / score_range

        if rev_rel >= threshold:
            matches.append({
                "position": i,
                "strand": "-",
                "score": rev_score,
                "max_score": max_s,
                "rel_score": round(rev_rel, 4),
                "matched_seq": subseq,
            })

    return matches


def compare_pwms(pwm1: np.ndarray, pwm2: np.ndarray) -> dict:
    """
    Compare two PWMs using Pearson correlation (like TOMTOM).
    Tries all offsets and both orientations, returns best match.

    Returns:
        dict with: pearson_r, offset, orientation, p_value_approx
    """
    from scipy import stats

    best = {"pearson_r": -1, "offset": 0, "orientation": "+"}

    for orientation, p2 in [("+", pwm2), ("-", reverse_complement_pwm(pwm2))]:
        len1, len2 = pwm1.shape[0], p2.shape[0]
        min_overlap = min(5, min(len1, len2))

        for offset in range(-len2 + min_overlap, len1 - min_overlap + 1):
            # Overlapping region
            start1 = max(0, offset)
            end1 = min(len1, offset + len2)
            start2 = max(0, -offset)
            end2 = start2 + (end1 - start1)

            if end1 - start1 < min_overlap:
                continue

            flat1 = pwm1[start1:end1].flatten()
            flat2 = p2[start2:end2].flatten()

            if len(flat1) < 4:
                continue

            r, p = stats.pearsonr(flat1, flat2)

            if r > best["pearson_r"]:
                best = {
                    "pearson_r": round(r, 4),
                    "p_value": round(p, 6),
                    "offset": offset,
                    "orientation": orientation,
                    "overlap": end1 - start1,
                }

    return best


# ---------------------------------------------------------------------------
# JASPAR database
# ---------------------------------------------------------------------------
def download_jaspar() -> list:
    """Download JASPAR 2024 CORE human motifs via API."""
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

            # Convert {A: [...], C: [...], G: [...], T: [...]} to numpy
            length = len(pfm_raw.get("A", []))
            if length == 0:
                continue

            pfm = np.zeros((length, 4))
            for bi, base in enumerate(BASES):
                vals = pfm_raw.get(base, [0] * length)
                pfm[:, bi] = vals

            all_motifs.append({
                "matrix_id": entry.get("matrix_id", ""),
                "name": entry.get("name", ""),
                "pfm": pfm.tolist(),
                "length": length,
                "family": entry.get("family", []),
                "uniprot_ids": entry.get("uniprot_ids", []),
            })

        url = data.get("next")

    # Cache
    with open(JASPAR_CACHE, "w") as f:
        json.dump(all_motifs, f)

    print(f"  Downloaded {len(all_motifs)} motifs → cached at {JASPAR_CACHE}")
    return all_motifs


# ---------------------------------------------------------------------------
# MotifAnalyzer
# ---------------------------------------------------------------------------
class MotifAnalyzer:
    """
    Motif analysis for DeepVRegulome variants.

    Usage:
        analyzer = MotifAnalyzer()
        report = analyzer.analyze_variant(dvr, "ATF4")
        print(report.disrupted_motifs)
        analyzer.plot_motif_logo(report)
    """

    def __init__(self, jaspar_cache: Optional[str] = None):
        """
        Args:
            jaspar_cache: Path to cached JASPAR JSON. If None, uses default cache.
        """
        self._jaspar_raw = None
        self._motifs = None  # dict: name -> {matrix_id, pwm, ...}
        self._custom_cache = jaspar_cache

    @property
    def motifs(self) -> dict:
        """Lazy-load JASPAR motifs."""
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
                name = entry["name"]
                mid = entry["matrix_id"]

                # Store by matrix_id (unique) and also index by name
                self._motifs[mid] = {
                    "matrix_id": mid,
                    "name": name,
                    "pfm": pfm,
                    "pwm": pwm,
                    "length": entry["length"],
                    "family": entry.get("family", []),
                }

            print(f"Loaded {len(self._motifs)} JASPAR motifs")

        return self._motifs

    def _find_motifs_by_name(self, tf_name: str) -> List[dict]:
        """Find all JASPAR motifs matching a TF name (case-insensitive)."""
        tf_upper = tf_name.upper()
        matches = []
        for mid, m in self.motifs.items():
            if m["name"].upper() == tf_upper:
                matches.append(m)
        return matches

    # ------------------------------------------------------------------
    # Core: scan a sequence for all JASPAR motif matches
    # ------------------------------------------------------------------
    def scan_all_motifs(
        self,
        sequence: str,
        threshold: float = 0.75,
        top_n: int = 50,
    ) -> List[MotifMatch]:
        """
        Scan a sequence against all JASPAR motifs.

        Args:
            sequence: DNA sequence
            threshold: minimum relative score to report (0-1)
            top_n: maximum number of matches to return

        Returns:
            List of MotifMatch, sorted by score descending
        """
        all_matches = []

        for mid, motif_data in self.motifs.items():
            pwm = motif_data["pwm"]
            hits = scan_sequence(sequence, pwm, threshold=threshold)

            for hit in hits:
                all_matches.append(MotifMatch(
                    motif_id=mid,
                    motif_name=motif_data["name"],
                    position=hit["position"],
                    strand=hit["strand"],
                    score=hit["score"],
                    max_score=hit["max_score"],
                    rel_score=hit["rel_score"],
                    matched_seq=hit["matched_seq"],
                ))

        # Sort by relative score
        all_matches.sort(key=lambda m: -m.rel_score)
        return all_matches[:top_n]

    # ------------------------------------------------------------------
    # Core: scan for a specific TF's motif
    # ------------------------------------------------------------------
    def find_tf_motif(
        self,
        sequence: str,
        tf_name: str,
        threshold: float = 0.7,
    ) -> Optional[MotifMatch]:
        """
        Check if a specific TF's JASPAR motif is present in the sequence.

        Returns the best match, or None if no match above threshold.
        """
        tf_motifs = self._find_motifs_by_name(tf_name)
        if not tf_motifs:
            return None

        best = None
        for motif_data in tf_motifs:
            hits = scan_sequence(sequence, motif_data["pwm"], threshold=threshold)
            for hit in hits:
                match = MotifMatch(
                    motif_id=motif_data["matrix_id"],
                    motif_name=motif_data["name"],
                    position=hit["position"],
                    strand=hit["strand"],
                    score=hit["score"],
                    max_score=hit["max_score"],
                    rel_score=hit["rel_score"],
                    matched_seq=hit["matched_seq"],
                )
                if best is None or match.rel_score > best.rel_score:
                    best = match

        return best

    # ------------------------------------------------------------------
    # Attention-based motif extraction
    # ------------------------------------------------------------------
    def extract_attention_motif(
        self,
        dvr,
        model_name: str,
        var_idx: int = 0,
        motif_length: int = 10,
        top_fraction: float = 0.2,
    ) -> Tuple[np.ndarray, str]:
        """
        Extract a PWM from high-attention positions in the DNABERT model.

        Uses the REF sequence attention: finds the region with highest
        attention density, extracts the subsequence, and converts to a PWM.

        Args:
            dvr: DVR instance with attention data
            model_name: model name
            var_idx: variant index
            motif_length: length of motif to extract
            top_fraction: fraction of positions to consider "high attention"

        Returns:
            (pwm, consensus_sequence)
        """
        data = dvr.get_attention(model_name, var_idx)
        ref_attn = data["ref_attention"]
        ref_seq = data["ref_seq"]

        # Find the region of length motif_length with highest total attention
        best_start = 0
        best_score = -1

        # Map k-mer attention to nucleotide space
        nuc_attn = np.zeros(len(ref_seq))
        for ki in range(len(ref_attn)):
            center = ki + 3
            if center < len(nuc_attn):
                nuc_attn[center] += ref_attn[ki]

        for i in range(len(ref_seq) - motif_length + 1):
            region_score = nuc_attn[i:i + motif_length].sum()
            if region_score > best_score:
                best_score = region_score
                best_start = i

        # Extract subsequence
        motif_seq = ref_seq[best_start:best_start + motif_length].upper()

        # Convert to PFM (single sequence → one-hot)
        pfm = np.zeros((motif_length, 4))
        for i, base in enumerate(motif_seq):
            if base in BASE_TO_IDX:
                pfm[i, BASE_TO_IDX[base]] = 1.0

        # Weight by attention
        attn_weights = nuc_attn[best_start:best_start + motif_length]
        attn_weights = attn_weights / (attn_weights.max() + 1e-8)

        # Create weighted PFM (attention-weighted counts)
        weighted_pfm = pfm * attn_weights[:, np.newaxis]

        # Normalize to get PWM
        pwm = pfm_to_pwm(weighted_pfm + PSEUDO)

        # Consensus
        consensus = ""
        for i in range(motif_length):
            consensus += BASES[np.argmax(pfm[i])]

        return weighted_pfm, consensus

    # ------------------------------------------------------------------
    # Full variant analysis
    # ------------------------------------------------------------------
    def analyze_variant(
        self,
        dvr,
        model_name: str,
        var_idx: int = 0,
        scan_threshold: float = 0.75,
        motif_length: int = 10,
    ) -> MotifReport:
        """
        Complete motif analysis for a scored variant.

        Args:
            dvr: DVR instance (must have been called with return_attention=True)
            model_name: model name (e.g., "ATF4")
            var_idx: variant index
            scan_threshold: PWM match threshold (0-1)
            motif_length: length for attention-derived motif

        Returns:
            MotifReport with all analysis results
        """
        data = dvr.get_attention(model_name, var_idx)
        ref_seq = data["ref_seq"]
        alt_seq = data["alt_seq"]
        variant_pos = data.get("variant_pos", len(ref_seq) // 2)

        # Extract window around variant for scanning
        window = 30
        start = max(0, variant_pos - window)
        end = min(len(ref_seq), variant_pos + window + 1)
        ref_window = ref_seq[start:end]
        alt_window = alt_seq[start:end]

        # 1. Scan REF and ALT for all JASPAR motifs
        print(f"Scanning ±{window}bp around variant for JASPAR motifs...")
        ref_matches = self.scan_all_motifs(ref_window, threshold=scan_threshold)
        alt_matches = self.scan_all_motifs(alt_window, threshold=scan_threshold)

        # 2. Find disrupted motifs (in REF but not in ALT at same position)
        ref_set = {(m.motif_id, m.position, m.strand): m for m in ref_matches}
        alt_set = {(m.motif_id, m.position, m.strand): m for m in alt_matches}

        disrupted = []
        for key, ref_m in ref_set.items():
            if key not in alt_set:
                disrupted.append({
                    "motif_id": ref_m.motif_id,
                    "motif_name": ref_m.motif_name,
                    "position": ref_m.position,
                    "strand": ref_m.strand,
                    "ref_score": ref_m.rel_score,
                    "alt_score": 0.0,
                    "score_change": -ref_m.rel_score,
                    "matched_seq": ref_m.matched_seq,
                })
            else:
                alt_m = alt_set[key]
                if ref_m.rel_score - alt_m.rel_score > 0.1:
                    disrupted.append({
                        "motif_id": ref_m.motif_id,
                        "motif_name": ref_m.motif_name,
                        "position": ref_m.position,
                        "strand": ref_m.strand,
                        "ref_score": ref_m.rel_score,
                        "alt_score": alt_m.rel_score,
                        "score_change": alt_m.rel_score - ref_m.rel_score,
                        "matched_seq": ref_m.matched_seq,
                    })

        gained = []
        for key, alt_m in alt_set.items():
            if key not in ref_set:
                gained.append({
                    "motif_id": alt_m.motif_id,
                    "motif_name": alt_m.motif_name,
                    "position": alt_m.position,
                    "strand": alt_m.strand,
                    "ref_score": 0.0,
                    "alt_score": alt_m.rel_score,
                    "score_change": alt_m.rel_score,
                    "matched_seq": alt_m.matched_seq,
                })

        disrupted_df = pd.DataFrame(disrupted).sort_values("score_change") if disrupted else pd.DataFrame()
        gained_df = pd.DataFrame(gained).sort_values("score_change", ascending=False) if gained else pd.DataFrame()

        # 3. Check TF's own motif
        tf_ref = self.find_tf_motif(ref_window, model_name, threshold=0.6)
        tf_alt = self.find_tf_motif(alt_window, model_name, threshold=0.6)
        tf_disrupted = (tf_ref is not None and tf_alt is None) or \
                       (tf_ref is not None and tf_alt is not None and
                        tf_ref.rel_score - tf_alt.rel_score > 0.1)

        # 4. Extract attention-based motif
        learned_pwm, consensus = self.extract_attention_motif(
            dvr, model_name, var_idx, motif_length=motif_length
        )

        # 5. Compare learned motif to JASPAR
        learned_jaspar_match = None
        if learned_pwm is not None:
            learned_pwm_logodds = pfm_to_pwm(learned_pwm + PSEUDO)
            best_r = -1
            best_match = None

            for mid, motif_data in self.motifs.items():
                try:
                    result = compare_pwms(learned_pwm_logodds, motif_data["pwm"])
                    if result["pearson_r"] > best_r:
                        best_r = result["pearson_r"]
                        best_match = {
                            "matrix_id": mid,
                            "name": motif_data["name"],
                            **result,
                        }
                except Exception:
                    continue

            learned_jaspar_match = best_match

        # Build report
        report = MotifReport(
            model_name=model_name,
            chrom=data.get("chrom", ""),
            genomic_pos=data.get("genomic_pos", 0),
            ref_allele=ref_seq[variant_pos] if variant_pos < len(ref_seq) else "",
            alt_allele=alt_seq[variant_pos] if variant_pos < len(alt_seq) else "",
            prob_ref=data["prob_ref"],
            prob_alt=data["prob_alt"],
            log_odds_ratio=0,  # computed externally
            ref_matches=ref_matches,
            alt_matches=alt_matches,
            disrupted_motifs=disrupted_df,
            gained_motifs=gained_df,
            tf_own_motif_ref=tf_ref,
            tf_own_motif_alt=tf_alt,
            tf_motif_disrupted=tf_disrupted,
            learned_pwm=learned_pwm,
            learned_consensus=consensus,
            learned_jaspar_match=learned_jaspar_match,
            ref_seq=ref_seq,
            alt_seq=alt_seq,
            variant_pos=variant_pos,
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"Motif Analysis: {model_name} at {report.chrom}:{report.genomic_pos}")
        print(f"{'='*60}")
        print(f"  Variant: {report.ref_allele} → {report.alt_allele}")
        print(f"  P(binding): {report.prob_ref:.4f} → {report.prob_alt:.4f}")
        print(f"  JASPAR motifs in REF window: {len(ref_matches)}")
        print(f"  JASPAR motifs in ALT window: {len(alt_matches)}")
        print(f"  Disrupted motifs: {len(disrupted_df)}")
        print(f"  Gained motifs: {len(gained_df)}")

        if tf_ref:
            print(f"  {model_name}'s own motif ({tf_ref.motif_id}): "
                  f"{'DISRUPTED' if tf_disrupted else 'intact'} "
                  f"(REF={tf_ref.rel_score:.3f}"
                  f"{', ALT=' + str(tf_alt.rel_score) if tf_alt else ', absent in ALT'})")
        else:
            print(f"  {model_name}'s own motif: not found in JASPAR")

        if learned_jaspar_match:
            lm = learned_jaspar_match
            print(f"  Attention-derived motif: {consensus}")
            print(f"    Best JASPAR match: {lm['name']} ({lm['matrix_id']}) "
                  f"r={lm['pearson_r']:.3f} p={lm.get('p_value', 'N/A')}")

        print(f"{'='*60}\n")

        return report

    # ------------------------------------------------------------------
    # Visualization: Motif Logo
    # ------------------------------------------------------------------
    def plot_motif_logo(
        self,
        report: MotifReport,
        figsize: Tuple[int, int] = (12, 4),
        save_path: Optional[str] = None,
    ):
        """
        Plot web logos for REF and ALT sequences at the variant site,
        weighted by attention.

        Requires: pip install logomaker
        """
        try:
            import logomaker
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "logomaker and matplotlib required for logo plots. "
                "Install with: pip install deepvregulome[interpret]"
            )

        variant_pos = report.variant_pos
        window = 10

        start = max(0, variant_pos - window)
        end = min(len(report.ref_seq), variant_pos + window + 1)

        ref_region = report.ref_seq[start:end].upper()
        alt_region = report.alt_seq[start:end].upper()

        # Create information content matrices
        def seq_to_ic_matrix(seq):
            """Convert sequence to information content matrix for logomaker."""
            mat = np.zeros((len(seq), 4))
            for i, base in enumerate(seq):
                if base in BASE_TO_IDX:
                    mat[i, BASE_TO_IDX[base]] = 2.0  # max IC = 2 bits
            df = pd.DataFrame(mat, columns=["A", "C", "G", "T"])
            return df

        ref_df = seq_to_ic_matrix(ref_region)
        alt_df = seq_to_ic_matrix(alt_region)

        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # REF logo
        logomaker.Logo(ref_df, ax=axes[0], color_scheme="classic")
        axes[0].set_title(
            f"REF — {report.model_name} | P(binding)={report.prob_ref:.4f}",
            fontsize=11
        )
        axes[0].set_ylabel("Bits")

        # Highlight variant
        vp = variant_pos - start
        axes[0].axvspan(vp - 0.5, vp + 0.5, alpha=0.2, color="red")

        # ALT logo
        logomaker.Logo(alt_df, ax=axes[1], color_scheme="classic")
        axes[1].set_title(
            f"ALT — {report.model_name} | P(binding)={report.prob_alt:.4f}",
            fontsize=11
        )
        axes[1].set_ylabel("Bits")
        axes[1].axvspan(vp - 0.5, vp + 0.5, alpha=0.2, color="red")

        fig.suptitle(
            f"Motif Logo: {report.chrom}:{report.genomic_pos} "
            f"{report.ref_allele}>{report.alt_allele}  ±{window}bp",
            fontsize=13, fontweight="bold"
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")

        return fig

    # ------------------------------------------------------------------
    # Visualization: Full Variant Report
    # ------------------------------------------------------------------
    def plot_variant_report(
        self,
        report: MotifReport,
        dvr=None,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
    ):
        """
        Combined figure: attention + motif + disruption in one panel.

        Panel 1: Sequence attention (nucleotide-colored boxes)
        Panel 2: Disrupted motifs table
        Panel 3: Attention-derived motif logo + best JASPAR match
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib required. pip install matplotlib")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1.5], hspace=0.4, wspace=0.3)

        variant_pos = report.variant_pos
        window = 15
        start = max(0, variant_pos - window)
        end = min(len(report.ref_seq), variant_pos + window + 1)

        ref_region = report.ref_seq[start:end]
        alt_region = report.alt_seq[start:end]
        vp_in_win = variant_pos - start

        # --- Panel 1: Sequence with colored nucleotides (top, full width) ---
        ax_seq = fig.add_subplot(gs[0, :])
        cmap = plt.cm.YlGnBu

        # Simple attention coloring (uniform for now, will use real attention if dvr provided)
        n = len(ref_region)
        ax_seq.set_xlim(-0.5, n - 0.5)
        ax_seq.set_ylim(-0.5, 1.5)

        # REF row (top)
        for i, base in enumerate(ref_region):
            color = "#d4e6f1" if i != vp_in_win else "#f1948a"
            rect = plt.Rectangle((i - 0.45, 0.6), 0.9, 0.7,
                                  facecolor=color, edgecolor="gray", linewidth=0.5)
            ax_seq.add_patch(rect)
            ax_seq.text(i, 0.95, base, ha="center", va="center",
                        fontsize=9, fontweight="bold")

        # ALT row (bottom)
        for i, base in enumerate(alt_region):
            color = "#fadbd8" if i != vp_in_win else "#e74c3c"
            rect = plt.Rectangle((i - 0.45, -0.2), 0.9, 0.7,
                                  facecolor=color, edgecolor="gray", linewidth=0.5)
            ax_seq.add_patch(rect)
            fontcolor = "white" if i == vp_in_win else "black"
            ax_seq.text(i, 0.15, base, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=fontcolor)

        ax_seq.text(-1.5, 0.95, "REF", ha="right", va="center", fontsize=10, fontweight="bold")
        ax_seq.text(-1.5, 0.15, "ALT", ha="right", va="center", fontsize=10, fontweight="bold")
        ax_seq.set_title(
            f"{report.model_name} at {report.chrom}:{report.genomic_pos} "
            f"({report.ref_allele}→{report.alt_allele})  |  "
            f"P(ref)={report.prob_ref:.4f} → P(alt)={report.prob_alt:.4f}",
            fontsize=12, fontweight="bold"
        )
        ax_seq.axis("off")

        # --- Panel 2: Disrupted motifs (bottom-left) ---
        ax_table = fig.add_subplot(gs[1, 0])
        ax_table.axis("off")

        if len(report.disrupted_motifs) > 0:
            top_disrupted = report.disrupted_motifs.head(8)
            table_data = []
            for _, row in top_disrupted.iterrows():
                table_data.append([
                    row["motif_name"],
                    row["motif_id"],
                    f"{row['ref_score']:.3f}",
                    f"{row['alt_score']:.3f}",
                    f"{row['score_change']:.3f}",
                ])

            table = ax_table.table(
                cellText=table_data,
                colLabels=["Motif", "ID", "REF", "ALT", "Δ"],
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)
            ax_table.set_title("Disrupted JASPAR Motifs", fontsize=11, fontweight="bold")
        else:
            ax_table.text(0.5, 0.5, "No disrupted motifs found",
                          ha="center", va="center", fontsize=11, style="italic")
            ax_table.set_title("Disrupted JASPAR Motifs", fontsize=11, fontweight="bold")

        # --- Panel 3: TF's own motif status (bottom-right) ---
        ax_tf = fig.add_subplot(gs[1, 1])
        ax_tf.axis("off")

        tf_text = f"TF: {report.model_name}\n\n"
        if report.tf_own_motif_ref:
            tf_text += f"JASPAR motif: {report.tf_own_motif_ref.motif_id}\n"
            tf_text += f"REF score: {report.tf_own_motif_ref.rel_score:.3f}\n"
            if report.tf_own_motif_alt:
                tf_text += f"ALT score: {report.tf_own_motif_alt.rel_score:.3f}\n"
            else:
                tf_text += "ALT: motif absent\n"
            tf_text += f"\nStatus: {'⚠ DISRUPTED' if report.tf_motif_disrupted else '✓ Intact'}"
        else:
            tf_text += "Own motif not found in JASPAR\n"
            if report.learned_jaspar_match:
                lm = report.learned_jaspar_match
                tf_text += f"\nAttention motif: {report.learned_consensus}\n"
                tf_text += f"Best match: {lm['name']} ({lm['matrix_id']})\n"
                tf_text += f"Pearson r = {lm['pearson_r']:.3f}"

        ax_tf.text(0.5, 0.5, tf_text, ha="center", va="center",
                   fontsize=10, family="monospace",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#eaf2f8", edgecolor="#aed6f1"))
        ax_tf.set_title(f"{report.model_name} Motif Status", fontsize=11, fontweight="bold")

        # --- Panel 4: Learned motif logo (bottom, full width) ---
        if report.learned_pwm is not None:
            try:
                import logomaker
                ax_logo = fig.add_subplot(gs[2, :])

                pwm_df = pd.DataFrame(
                    report.learned_pwm,
                    columns=["A", "C", "G", "T"]
                )
                # Normalize for logo
                row_sums = pwm_df.sum(axis=1)
                row_sums[row_sums == 0] = 1
                pwm_norm = pwm_df.div(row_sums, axis=0)

                # Information content
                ic = pwm_norm.copy()
                for col in ic.columns:
                    ic[col] = pwm_norm[col] * np.log2(pwm_norm[col] / 0.25 + 1e-10)
                ic = ic.clip(lower=0)

                logomaker.Logo(ic, ax=ax_logo, color_scheme="classic")
                ax_logo.set_title(
                    f"Attention-Derived Motif: {report.learned_consensus}"
                    + (f"  |  Best JASPAR match: {report.learned_jaspar_match['name']} "
                       f"(r={report.learned_jaspar_match['pearson_r']:.3f})"
                       if report.learned_jaspar_match else ""),
                    fontsize=11, fontweight="bold"
                )
                ax_logo.set_ylabel("IC (bits)")
            except ImportError:
                pass

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")

        return fig
