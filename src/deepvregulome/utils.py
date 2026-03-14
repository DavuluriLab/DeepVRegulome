"""
Sequence processing utilities for DeepVRegulome.

Handles DNA ↔ k-mer conversion, reverse complement, and
variant sequence extraction from reference genomes.
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
    pos: int,
    ref: str,
    alt: str,
    flank: int = 150,
) -> Tuple[str, str]:
    """
    Extract REF and ALT sequences centered on a variant.

    Args:
        genome: pysam.FastaFile or object with .fetch(chrom, start, end)
        chrom: Chromosome name (e.g., "chr1")
        pos: 1-based variant position
        ref: Reference allele (e.g., "A")
        alt: Alternate allele (e.g., "TA")
        flank: Number of flanking bases on each side (default: 150 → 301bp for SNV)

    Returns:
        (ref_seq, alt_seq) tuple of DNA strings
    """
    start_0 = pos - 1  # 0-based position of first ref base
    left_flank = genome.fetch(chrom, start_0 - flank, start_0)
    right_flank = genome.fetch(chrom, start_0 + len(ref), start_0 + len(ref) + flank)

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

    Args:
        genome: pysam.FastaFile
        variants: List of dicts with keys: chrom, pos, ref, alt
        flank: Flanking bases on each side

    Returns:
        List of (ref_seq, alt_seq) tuples
    """
    results = []
    for v in variants:
        try:
            ref_seq, alt_seq = extract_variant_sequences(
                genome, v["chrom"], v["pos"], v["ref"], v["alt"], flank
            )
            results.append((ref_seq, alt_seq))
        except Exception as e:
            # Return None pair for failed extractions
            results.append((None, None))
    return results


def parse_vcf(vcf_path: str, max_variants: Optional[int] = None) -> list:
    """
    Parse a VCF file into a list of variant dicts.
    Handles both with-header and headerless VCF files, plus .vcf.gz.

    Args:
        vcf_path: Path to VCF file
        max_variants: Maximum number of variants to parse (None = all)

    Returns:
        List of dicts with keys: chrom, pos, ref, alt
        (multi-allelic ALTs are split into separate entries)
    """
    import gzip

    opener = gzip.open if vcf_path.endswith(".gz") else open
    variants = []

    with opener(vcf_path, "rt") as f:
        for line in f:
            # Skip header lines
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

            # Split multi-allelic
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
