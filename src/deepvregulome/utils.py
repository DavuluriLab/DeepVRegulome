"""
Sequence processing utilities for DeepVRegulome.

Handles DNA ↔ k-mer conversion, reverse complement,
variant sequence extraction (with multiprocessing),
VCF parsing, and coordinate sanity checking.
"""

import os
from typing import List, Tuple, Optional
from multiprocessing import Pool


COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def to_kmer(sequence: str, k: int = 6) -> str:
    """Convert a DNA sequence to space-separated k-mer representation for DNABERT."""
    seq = sequence.upper()
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return sequence.translate(COMPLEMENT)[::-1]


def extract_variant_sequences(
    genome,
    chrom: str,
    pos_0based: int,
    ref: str,
    alt: str,
    flank: int = 150,
) -> Tuple[str, str]:
    """
    Extract REF and ALT sequences centered on a variant.

    Args:
        genome: pysam.FastaFile
        chrom: Chromosome (e.g., "chr1")
        pos_0based: 0-based variant position
        ref: Reference allele
        alt: Alternate allele
        flank: Flanking bases on each side

    Returns:
        (ref_seq, alt_seq) tuple
    """
    left_flank = genome.fetch(chrom, pos_0based - flank, pos_0based)
    right_flank = genome.fetch(chrom, pos_0based + len(ref), pos_0based + len(ref) + flank)

    ref_seq = left_flank + ref + right_flank
    alt_seq = left_flank + alt + right_flank

    return ref_seq.upper(), alt_seq.upper()


# ---------------------------------------------------------------------------
# Parallel sequence extraction
# ---------------------------------------------------------------------------

# Module-level global for multiprocessing workers
_worker_genome = None
_worker_flank = None


def _init_extraction_worker(genome_path: str, flank: int):
    """Initialize pysam.FastaFile in each worker process."""
    global _worker_genome, _worker_flank
    import pysam
    _worker_genome = pysam.FastaFile(genome_path)
    _worker_flank = flank


def _extract_one_variant(variant: dict) -> Tuple[Optional[str], Optional[str]]:
    """Extract sequences for a single variant (called by worker)."""
    global _worker_genome, _worker_flank
    try:
        ref_seq, alt_seq = extract_variant_sequences(
            _worker_genome,
            variant["chrom"],
            variant["pos"],
            variant["ref"],
            variant["alt"],
            _worker_flank,
        )
        return (ref_seq, alt_seq)
    except Exception:
        return (None, None)


def extract_variant_sequences_batch(
    genome_path: str,
    variants: list,
    flank: int = 150,
    n_workers: int = 0,
) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Extract REF and ALT sequences for a batch of variants.
    Uses multiprocessing when n_workers > 1 for speed.

    Args:
        genome_path: Path to reference genome FASTA (string, not pysam object)
        variants: List of dicts with chrom, pos, ref, alt (pos already 0-based)
        flank: Flanking bases on each side
        n_workers: Number of parallel workers. 0 = auto (cpu_count/2, min 1).

    Returns:
        List of (ref_seq, alt_seq) tuples. Failed extractions return (None, None).
    """
    if n_workers == 0:
        n_workers = max(1, os.cpu_count() // 2)

    if n_workers == 1 or len(variants) < 100:
        # Single-threaded for small inputs
        import pysam
        genome = pysam.FastaFile(genome_path)
        results = []
        for v in variants:
            try:
                ref_seq, alt_seq = extract_variant_sequences(
                    genome, v["chrom"], v["pos"], v["ref"], v["alt"], flank
                )
                results.append((ref_seq, alt_seq))
            except Exception:
                results.append((None, None))
        genome.close()
        return results

    # Multi-threaded
    with Pool(
        processes=n_workers,
        initializer=_init_extraction_worker,
        initargs=(genome_path, flank),
    ) as pool:
        results = pool.map(_extract_one_variant, variants, chunksize=500)

    return results


def parse_vcf(vcf_path: str, max_variants: Optional[int] = None) -> list:
    """
    Parse a VCF file into a list of variant dicts.
    Handles both with-header and headerless VCFs.

    Returns:
        List of dicts with keys: chrom, pos, ref, alt
    """
    import gzip

    opener = gzip.open if vcf_path.endswith(".gz") else open
    variants = []

    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 5:
                continue

            chrom = fields[0]
            try:
                pos = int(fields[1])
            except ValueError:
                continue
            ref = fields[3]
            alt_str = fields[4]

            for alt in alt_str.split(","):
                alt = alt.strip()
                if alt == "." or alt == "*":
                    continue
                variants.append({
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                })

            if max_variants and len(variants) >= max_variants:
                break

    return variants[:max_variants] if max_variants else variants


# ──────────────────────────────────────────────────────────────
# COORDINATE SANITY CHECK
# ──────────────────────────────────────────────────────────────

def detect_coordinate_system(genome, variants: list, n_check: int = 10) -> dict:
    """
    Auto-detect whether variant positions are 1-based (VCF) or 0-based (BED)
    by checking REF alleles against the reference genome.
    """
    to_check = variants[:min(n_check, len(variants))]

    votes_1based = 0
    votes_0based = 0
    ambiguous = 0
    no_match = 0

    for v in to_check:
        chrom = v["chrom"]
        pos = v["pos"]
        ref = v["ref"].upper()

        try:
            if len(ref) > 1:
                seq_at_minus1 = genome.fetch(chrom, pos - 1, pos - 1 + len(ref)).upper()
                seq_at_pos = genome.fetch(chrom, pos, pos + len(ref)).upper()
            else:
                seq_at_minus1 = genome.fetch(chrom, pos - 1, pos).upper()
                seq_at_pos = genome.fetch(chrom, pos, pos + 1).upper()

            match_minus1 = (seq_at_minus1 == ref)
            match_pos = (seq_at_pos == ref)

            if match_minus1 and not match_pos:
                votes_1based += 1
            elif match_pos and not match_minus1:
                votes_0based += 1
            elif match_minus1 and match_pos:
                ambiguous += 1
            else:
                no_match += 1

        except Exception:
            no_match += 1

    total_checked = len(to_check)

    if votes_1based >= 3 and votes_0based == 0:
        system, offset, confidence = "1-based", 1, "high"
    elif votes_0based >= 3 and votes_1based == 0:
        system, offset, confidence = "0-based", 0, "high"
    elif votes_1based > votes_0based:
        system, offset, confidence = "1-based", 1, "medium"
    elif votes_0based > votes_1based:
        system, offset, confidence = "0-based", 0, "medium"
    else:
        system, offset, confidence = "1-based", 1, "low"

    if confidence == "high":
        message = (
            f"✓ Sanity check PASSED — {system} coordinates detected\n"
            f"  Checked {total_checked} variants: "
            f"{votes_1based} confirm 1-based, {votes_0based} confirm 0-based, "
            f"{ambiguous} ambiguous, {no_match} no-match\n"
            f"  REF alleles match the reference genome. Coordinates look correct."
        )
    elif confidence == "medium":
        message = (
            f"⚠ Sanity check: likely {system} but not fully certain\n"
            f"  Checked {total_checked} variants: "
            f"{votes_1based} suggest 1-based, {votes_0based} suggest 0-based, "
            f"{ambiguous} ambiguous, {no_match} no-match\n"
            f"  Proceeding with {system}. Override with coordinate_system= if needed."
        )
    else:
        message = (
            f"⚠ Sanity check: could not confidently determine coordinate system\n"
            f"  Checked {total_checked} variants: "
            f"{votes_1based} suggest 1-based, {votes_0based} suggest 0-based, "
            f"{ambiguous} ambiguous, {no_match} no-match\n"
            f"  Defaulting to 1-based (VCF standard). Override with coordinate_system='0-based' if needed."
        )

    return {
        "system": system,
        "offset": offset,
        "confidence": confidence,
        "votes_1based": votes_1based,
        "votes_0based": votes_0based,
        "ambiguous": ambiguous,
        "no_match": no_match,
        "message": message,
    }
