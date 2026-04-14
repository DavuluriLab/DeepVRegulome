"""
DeepVRegulome utility functions.

v0.2.0 changes:
- Parallel sequence extraction via multiprocessing.Pool
- Disk caching of REF/ALT sequences (TSV format)
- Disk caching of tokenized features (.pt format)
- Auto-detection of n_processes
"""

import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# n_processes auto-detection
# ---------------------------------------------------------------------------
def get_default_n_processes(user_value: Optional[int] = None) -> int:
    """
    Auto-detect a sensible number of worker processes.

    Strategy: leave 4 cores for the OS and DataLoader workers, cap at 32
    because tokenization stops scaling beyond that.

    On a 64-core server: returns 32
    On an 8-core laptop: returns 4
    On a 4-core machine: returns 1
    """
    if user_value is not None:
        return max(1, int(user_value))
    n_cpu = os.cpu_count() or 4
    return min(max(n_cpu - 4, 1), 32)


# ---------------------------------------------------------------------------
# k-mer tokenization
# ---------------------------------------------------------------------------
def to_kmer(seq: str, k: int = 6) -> str:
    """Convert a DNA sequence to space-separated k-mers."""
    seq = seq.upper()
    return " ".join([seq[i : i + k] for i in range(len(seq) - k + 1)])


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------
def compute_cache_key(
    vcf_path: Optional[str],
    genome_path: Optional[str],
    n_variants: int,
    flank: int,
    extra: str = "",
) -> str:
    """
    Generate a stable hash key for caching based on inputs.
    """
    h = hashlib.sha256()
    h.update(str(vcf_path or "").encode())
    h.update(str(genome_path or "").encode())
    h.update(str(n_variants).encode())
    h.update(str(flank).encode())
    h.update(extra.encode())
    return h.hexdigest()[:16]


def get_cache_dir(base: Optional[str] = None) -> Path:
    """Return the cache directory, creating it if needed."""
    if base is None:
        base = os.path.expanduser("~/.cache/deepvregulome")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Parallel sequence extraction
# ---------------------------------------------------------------------------
# Module-level globals for multiprocessing workers (cannot pickle pysam objects)
_WORKER_GENOME_PATH = None
_WORKER_FLANK = None


def _init_worker(genome_path: str, flank: int):
    """Initializer for each worker process — opens its own pysam handle."""
    global _WORKER_GENOME_PATH, _WORKER_FLANK
    _WORKER_GENOME_PATH = genome_path
    _WORKER_FLANK = flank


def _extract_one(args):
    """
    Worker function: extract REF/ALT sequences for one variant.
    Each worker opens its own pysam.FastaFile (pysam objects are not picklable).
    """
    import pysam

    chrom, pos, ref, alt, coord_offset = args
    try:
        fa = pysam.FastaFile(_WORKER_GENOME_PATH)
        # coord_offset: 1 means 1-based input (subtract 1 to get 0-based for pysam)
        zero_pos = pos - coord_offset
        start = zero_pos - _WORKER_FLANK
        end = zero_pos + _WORKER_FLANK + 1
        if start < 0:
            return None
        ref_seq = fa.fetch(chrom, start, end).upper()
        if len(ref_seq) != 2 * _WORKER_FLANK + 1:
            return None
        # Build ALT sequence by replacing the center base(s)
        center = _WORKER_FLANK
        ref_len = len(ref)
        alt_seq = ref_seq[:center] + alt.upper() + ref_seq[center + ref_len :]
        # Pad/truncate ALT to same length as REF
        if len(alt_seq) > len(ref_seq):
            alt_seq = alt_seq[: len(ref_seq)]
        elif len(alt_seq) < len(ref_seq):
            alt_seq = alt_seq + "N" * (len(ref_seq) - len(alt_seq))
        fa.close()
        return (chrom, pos, ref, alt, ref_seq, alt_seq)
    except Exception:
        return None


def extract_sequences_parallel(
    variants: List[Tuple[str, int, str, str]],
    genome_path: str,
    flank: int = 150,
    coord_offset: int = 1,
    n_processes: Optional[int] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Extract REF/ALT sequences for many variants in parallel.

    Parameters
    ----------
    variants : list of (chrom, pos, ref, alt) tuples
    genome_path : path to indexed FASTA
    flank : flanking bases on each side (default 150 → 301bp window)
    coord_offset : 1 if input is 1-based (VCF), 0 if input is 0-based (BED)
    n_processes : number of worker processes (auto-detected if None)

    Returns
    -------
    DataFrame with columns: chrom, pos, ref, alt, ref_seq, alt_seq
    """
    n_proc = get_default_n_processes(n_processes)
    args_list = [(c, p, r, a, coord_offset) for (c, p, r, a) in variants]

    print(f"Extracting sequences for {len(variants):,} variants using {n_proc} processes...")

    results = []
    with Pool(
        processes=n_proc,
        initializer=_init_worker,
        initargs=(genome_path, flank),
    ) as pool:
        if show_progress:
            try:
                from tqdm.auto import tqdm

                for r in tqdm(
                    pool.imap_unordered(_extract_one, args_list, chunksize=500),
                    total=len(args_list),
                    desc="Extracting",
                    unit="var",
                ):
                    if r is not None:
                        results.append(r)
            except ImportError:
                results = [r for r in pool.imap_unordered(_extract_one, args_list, chunksize=500) if r is not None]
        else:
            results = [r for r in pool.imap_unordered(_extract_one, args_list, chunksize=500) if r is not None]

    df = pd.DataFrame(results, columns=["chrom", "pos", "ref", "alt", "ref_seq", "alt_seq"])
    print(f"  Successfully extracted {len(df):,} / {len(variants):,} sequences")
    return df


# ---------------------------------------------------------------------------
# Sequence cache (TSV)
# ---------------------------------------------------------------------------
def save_sequences_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save REF/ALT sequences to a TSV file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df[["chrom", "pos", "ref", "alt", "ref_seq", "alt_seq"]].to_csv(
        cache_path, sep="\t", index=False
    )


def load_sequences_cache(cache_path: Path) -> Optional[pd.DataFrame]:
    """Load REF/ALT sequences from TSV cache, or None if missing."""
    if not cache_path.exists():
        return None
    return pd.read_csv(cache_path, sep="\t", dtype={"chrom": str, "pos": int, "ref": str, "alt": str})


def save_meta(meta: dict, meta_path: Path) -> None:
    """Save cache metadata as JSON."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Parallel k-mer tokenization
# ---------------------------------------------------------------------------
def _kmer_one(seq: str) -> str:
    return to_kmer(seq)


def kmerize_parallel(sequences: List[str], n_processes: Optional[int] = None) -> List[str]:
    """Convert many sequences to k-mer strings in parallel."""
    n_proc = get_default_n_processes(n_processes)
    if n_proc == 1 or len(sequences) < 1000:
        return [to_kmer(s) for s in sequences]
    with Pool(processes=n_proc) as pool:
        return list(pool.imap(_kmer_one, sequences, chunksize=1000))


# ---------------------------------------------------------------------------
# VCF parsing (unchanged)
# ---------------------------------------------------------------------------
def parse_vcf(vcf_path: str, max_variants: Optional[int] = None) -> List[dict]:
    """Parse a VCF file into a list of variant dicts."""
    variants = []
    try:
        import pysam

        vf = pysam.VariantFile(vcf_path)
        for i, record in enumerate(vf):
            if max_variants and i >= max_variants:
                break
            if record.alts is None:
                continue
            for alt in record.alts:
                variants.append({
                    "chrom": record.chrom,
                    "pos": record.pos,
                    "ref": record.ref,
                    "alt": alt,
                })
        vf.close()
    except (ImportError, Exception):
        # Fallback to plain text parsing
        with open(vcf_path) as f:
            count = 0
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                chrom, pos, _, ref, alt_field = parts[:5]
                for alt in alt_field.split(","):
                    variants.append({
                        "chrom": chrom,
                        "pos": int(pos),
                        "ref": ref,
                        "alt": alt,
                    })
                    count += 1
                    if max_variants and count >= max_variants:
                        return variants
    return variants
