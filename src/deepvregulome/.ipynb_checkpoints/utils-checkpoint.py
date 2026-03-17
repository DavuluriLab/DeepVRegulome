"""
Sequence processing utilities for DeepVRegulome.

Handles DNA ↔ k-mer conversion, reverse complement,
variant sequence extraction, VCF parsing, and coordinate sanity checking.
"""

from typing import List, Tuple, Optional


COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def to_kmer(sequence: str, k: int = 6) -> str:
    """
    Convert a DNA sequence to space-separated k-mer representation for DNABERT.

    Example:
        >>> to_kmer("ATCGATCG", k=6)
        'ATCGAT TCGATC CGATCG'
    """
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
        pos_0based: 0-based variant position (already adjusted by sanity check)
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


def extract_variant_sequences_batch(
    genome,
    variants: list,
    flank: int = 150,
) -> List[Tuple[str, str]]:
    """
    Extract REF and ALT sequences for a batch of variants.
    Expects variants with pos already in 0-based (adjusted by sanity check).
    """
    results = []
    for v in variants:
        try:
            ref_seq, alt_seq = extract_variant_sequences(
                genome, v["chrom"], v["pos"], v["ref"], v["alt"], flank
            )
            results.append((ref_seq, alt_seq))
        except Exception:
            results.append((None, None))
    return results


def parse_vcf(vcf_path: str, max_variants: Optional[int] = None) -> list:
    """
    Parse a VCF file into a list of variant dicts.
    Handles both with-header and headerless VCFs, plus .vcf.gz.

    Returns:
        List of dicts with keys: chrom, pos, ref, alt
        pos is kept as-is from the file (sanity check determines offset later)
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
    by checking if the REF allele matches the reference genome.

    For each variant, checks genome at pos-1 and pos (0-based fetch).
    Only votes when the result is unambiguous (different bases at the two positions).

    Args:
        genome: pysam.FastaFile
        variants: list of dicts with chrom, pos, ref, alt
        n_check: number of variants to check (default: 10)

    Returns:
        dict with:
            system: "1-based" | "0-based" | "unknown"
            offset: 1 | 0 (subtract this from pos to get 0-based)
            message: str (human-readable summary)
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

    # Decision
    total_checked = len(to_check)
    informative = votes_1based + votes_0based

    if votes_1based >= 3 and votes_0based == 0:
        system, offset = "1-based", 1
        confidence = "high"
    elif votes_0based >= 3 and votes_1based == 0:
        system, offset = "0-based", 0
        confidence = "high"
    elif votes_1based > votes_0based:
        system, offset = "1-based", 1
        confidence = "medium"
    elif votes_0based > votes_1based:
        system, offset = "0-based", 0
        confidence = "medium"
    else:
        system, offset = "1-based", 1
        confidence = "low"

    # Build message
    if confidence == "high" and system == "1-based":
        message = (
            f"✓ Sanity check PASSED — 1-based coordinates (VCF format) detected\n"
            f"  Checked {total_checked} variants: {votes_1based} confirmed 1-based, "
            f"{ambiguous} ambiguous, {no_match} no-match\n"
            f"  REF alleles match the reference genome. Coordinates look correct."
        )
    elif confidence == "high" and system == "0-based":
        message = (
            f"✓ Sanity check PASSED — 0-based coordinates (BED format) detected\n"
            f"  Checked {total_checked} variants: {votes_0based} confirmed 0-based, "
            f"{ambiguous} ambiguous, {no_match} no-match\n"
            f"  REF alleles match the reference genome. Coordinates look correct."
        )
    elif confidence == "medium":
        message = (
            f"⚠ Sanity check: likely {system} but not fully certain\n"
            f"  Checked {total_checked} variants: {votes_1based} suggest 1-based, "
            f"{votes_0based} suggest 0-based, {ambiguous} ambiguous, {no_match} no-match\n"
            f"  Proceeding with {system}. Override with coordinate_system= if needed."
        )
    else:
        message = (
            f"⚠ Sanity check: could not confidently determine coordinate system\n"
            f"  Checked {total_checked} variants: {votes_1based} suggest 1-based, "
            f"{votes_0based} suggest 0-based, {ambiguous} ambiguous, {no_match} no-match\n"
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
