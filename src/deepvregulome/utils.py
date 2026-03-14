"""
Sequence processing utilities for DeepVRegulome.

Handles DNA ↔ k-mer conversion, reverse complement, and
variant sequence extraction from reference genomes.
"""

from typing import Tuple


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

    For SNVs: both sequences are (2*flank + 1) bp.
    For indels: REF and ALT lengths differ by len(alt) - len(ref).

    Args:
        genome: pysam.FastaFile or dict-like with .fetch(chrom, start, end)
        chrom: Chromosome name (e.g., "chr1")
        pos: 1-based variant position
        ref: Reference allele (e.g., "A")
        alt: Alternate allele (e.g., "TA")
        flank: Number of flanking bases on each side (default: 150 → 301bp for SNV)

    Returns:
        (ref_seq, alt_seq) tuple of DNA strings

    Example:
        For chr1:3456782 A>TA with flank=150:
        - Extracts 150bp upstream + "A" + 150bp downstream → ref_seq (301bp)
        - Replaces "A" with "TA" → alt_seq (302bp)
        Both are then k-merized by the scorer.
    """
    # Convert to 0-based half-open coordinates
    start_0 = pos - 1  # 0-based position of first ref base

    # Extract flanking sequences
    left_flank = genome.fetch(chrom, start_0 - flank, start_0)
    right_flank = genome.fetch(chrom, start_0 + len(ref), start_0 + len(ref) + flank)

    ref_seq = left_flank + ref + right_flank
    alt_seq = left_flank + alt + right_flank

    return ref_seq.upper(), alt_seq.upper()


def extract_variant_sequences_from_fasta(
    fasta_path: str,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    flank: int = 150,
) -> Tuple[str, str]:
    """
    Convenience wrapper that opens a FASTA file, extracts sequences, and closes it.

    Requires: pip install pysam
    """
    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is required for genome-based variant extraction. "
            "Install with: pip install deepvregulome[genome]"
        )

    with pysam.FastaFile(fasta_path) as genome:
        return extract_variant_sequences(genome, chrom, pos, ref, alt, flank)
