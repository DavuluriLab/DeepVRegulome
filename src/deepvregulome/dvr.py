"""
DeepVRegulome v0.3.0 — high-performance variant effect scoring.

Changes from v0.2.x:
6. fp16 inference (--use_fp16, default True on CUDA) — ~2x GPU speedup
7. Explicit SDPA attention implementation
8. In-memory model cache (avoid reloading same model in same session)

Previous v0.2.0 changes:
1. Parallel sequence extraction (multiprocessing.Pool)
2. Disk caching of REF/ALT sequences and tokenized features
3. Tokenization done ONCE outside the model loop
4. PyTorch DataLoader with prefetching workers
5. DataParallel mode for few-models-many-variants case

API is backward-compatible. All v0.1/v0.2 calls still work.
"""

import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from deepvregulome.registry import ModelInfo, ModelRegistry
from deepvregulome.utils import (
    compute_cache_key,
    extract_sequences_parallel,
    get_cache_dir,
    get_default_n_processes,
    kmerize_parallel,
    load_sequences_cache,
    parse_vcf,
    save_meta,
    save_sequences_cache,
    to_kmer,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_FLANK = 150          # → 301bp window
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 4      # DataLoader prefetch workers
DEFAULT_MAX_LENGTH = 512     # DNABERT max sequence length
EPS = 1e-7


# ---------------------------------------------------------------------------
# DVR class
# ---------------------------------------------------------------------------
class DVR:
    """
    DeepVRegulome v0.2.0 main interface.

    Example:
        dvr = DVR(genome="hg38.fa")
        results = dvr.score_vcf(
            "patient.vcf",
            models=["CTCFL", "SP1"],
            cache_dir="./dvr_cache",     # NEW: disk caching
            gpus=[0, 1, 2, 3],
            multi_gpu_mode="auto",       # NEW: auto-select DataParallel vs model-parallel
            n_processes=32,              # NEW: explicit CPU parallelism
        )
    """

    def __init__(
        self,
        genome: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        coordinate_system: Optional[str] = None,
        n_processes: Optional[int] = None,
        use_fp16: Optional[bool] = None,
    ):
        self.registry = ModelRegistry()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_cache_dir = None  # for HuggingFace model downloads
        self.dvr_cache_dir = cache_dir  # for sequence/feature caching
        self._genome_path = genome
        self._coordinate_system = coordinate_system
        self._coord_offset = 1 if coordinate_system in (None, "1-based") else 0
        self._n_processes = n_processes
        self.last_attention: Dict[str, dict] = {}

        # v0.3: fp16 enabled by default on CUDA, off on CPU
        if use_fp16 is None:
            use_fp16 = torch.cuda.is_available()
        self.use_fp16 = use_fp16

        # v0.3: in-memory model cache (avoid reloading across calls in same session)
        # Maps model_name -> (model, tokenizer, device_id, fp16_flag)
        self._model_cache: Dict[str, Tuple] = {}

    # ==================================================================
    # v0.3: Centralized model loading with fp16 + SDPA + in-memory cache
    # ==================================================================
    def _load_model(
        self,
        name: str,
        device: str,
        return_attention: bool = False,
    ):
        """
        Load a model with fp16 + SDPA + in-memory cache.

        Returns (model, tokenizer). Subsequent calls for the same model on
        the same device with the same fp16 setting return the cached instance.
        """
        cache_key = (name, device, self.use_fp16, return_attention)

        # Check in-memory cache
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        info = self.registry.get(name)

        tokenizer = AutoTokenizer.from_pretrained(
            info.hf_repo, subfolder=info.subfolder, cache_dir=self.hf_cache_dir,
        )

        # Try SDPA attention implementation (faster, available in transformers >= 4.36)
        # Falls back gracefully on older transformers
        load_kwargs = dict(
            cache_dir=self.hf_cache_dir,
            output_attentions=return_attention,
        )
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                info.hf_repo, subfolder=info.subfolder,
                attn_implementation="sdpa",
                **load_kwargs,
            )
        except (TypeError, ValueError):
            # Older transformers without attn_implementation argument
            model = AutoModelForSequenceClassification.from_pretrained(
                info.hf_repo, subfolder=info.subfolder,
                **load_kwargs,
            )

        model = model.to(device).eval()

        # fp16 (only on CUDA, and not when extracting attention to keep numerics stable)
        if self.use_fp16 and "cuda" in str(device) and not return_attention:
            model = model.half()

        self._model_cache[cache_key] = (model, tokenizer)
        return model, tokenizer

    def clear_model_cache(self):
        """Free GPU memory by clearing the in-memory model cache."""
        self._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==================================================================
    # Public scoring API (backward compatible)
    # ==================================================================
    def score_variant(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        models: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        return_attention: bool = False,
    ) -> pd.DataFrame:
        """Score a single variant."""
        df = pd.DataFrame([{"chrom": chrom, "pos": pos, "ref": ref, "alt": alt}])
        return self.score_variants(
            df, models=models, model_type=model_type,
            return_attention=return_attention,
            batch_size=1, gpus=[0] if torch.cuda.is_available() else None,
        )

    def score_variants(
        self,
        variants: Union[pd.DataFrame, List[dict]],
        models: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        flank: int = DEFAULT_FLANK,
        batch_size: int = DEFAULT_BATCH_SIZE,
        gpus: Optional[List[int]] = None,
        cache_dir: Optional[str] = None,
        n_processes: Optional[int] = None,
        num_workers: int = DEFAULT_NUM_WORKERS,
        multi_gpu_mode: str = "auto",
        return_attention: bool = False,
    ) -> pd.DataFrame:
        """
        Score a batch of variants from a DataFrame.

        Parameters
        ----------
        variants : DataFrame with columns chrom, pos, ref, alt
        models : list of model names (e.g., ["CTCFL", "SP1"])
        model_type : "TF" or "HISTONE" — score against all models of this type
        flank : flanking bases for sequence extraction (default 150 → 301bp)
        batch_size : sequences per GPU forward pass
        gpus : list of GPU device IDs
        cache_dir : directory for disk caching (default ~/.cache/deepvregulome)
        n_processes : CPU parallelism for extraction/tokenization (auto if None)
        num_workers : DataLoader prefetch workers (default 4)
        multi_gpu_mode : "auto", "data_parallel", or "model_parallel"
            - data_parallel: split each batch across GPUs (best for few models)
            - model_parallel: one model per GPU (best for many models)
            - auto: pick based on len(models) vs len(gpus)
        return_attention : extract attention weights for interpretability
        """
        if isinstance(variants, list):
            variants = pd.DataFrame(variants)
        if len(variants) == 0:
            return pd.DataFrame()

        model_names = self._resolve_models(models, model_type)
        cache_dir = cache_dir or self.dvr_cache_dir
        n_processes = n_processes if n_processes is not None else self._n_processes

        # ===== STEP 1: Sequence extraction (with disk cache) =====
        sequences_df = self._extract_sequences_with_cache(
            variants, flank=flank, cache_dir=cache_dir, n_processes=n_processes,
        )

        # ===== STEP 2: Tokenize ONCE (outside model loop) =====
        # All DNABERT 6-mer models share the same vocab, so tokenize once and reuse
        ref_input_ids, ref_attention_mask, alt_input_ids, alt_attention_mask = (
            self._tokenize_with_cache(
                sequences_df, cache_dir=cache_dir, n_processes=n_processes,
            )
        )

        # ===== STEP 3: Build TensorDataset =====
        ref_dataset = TensorDataset(ref_input_ids, ref_attention_mask)
        alt_dataset = TensorDataset(alt_input_ids, alt_attention_mask)

        # ===== STEP 4: Score against models =====
        gpus = gpus or ([0] if torch.cuda.is_available() else None)
        mode = self._resolve_multi_gpu_mode(multi_gpu_mode, model_names, gpus)
        print(f"Scoring {len(sequences_df):,} variants × {len(model_names)} models "
              f"(mode={mode}, gpus={gpus}, batch_size={batch_size})...")

        if mode == "data_parallel" and gpus and len(gpus) > 1:
            results = self._score_data_parallel(
                ref_dataset, alt_dataset, model_names,
                gpus=gpus, batch_size=batch_size, num_workers=num_workers,
                return_attention=return_attention,
            )
        else:
            results = self._score_single_or_model_parallel(
                ref_dataset, alt_dataset, model_names,
                gpus=gpus, batch_size=batch_size, num_workers=num_workers,
                return_attention=return_attention,
            )

        # ===== STEP 5: Build result DataFrame =====
        return self._build_result_df(sequences_df, results, model_names)

    def score_vcf(
        self,
        vcf_path: str,
        models: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        max_variants: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Score all variants in a VCF file."""
        print(f"Parsing VCF: {vcf_path}")
        variant_list = parse_vcf(vcf_path, max_variants=max_variants)
        print(f"  Found {len(variant_list):,} variants")
        if not variant_list:
            return pd.DataFrame()

        # Pass vcf_path as cache key hint
        return self.score_variants(
            pd.DataFrame(variant_list),
            models=models, model_type=model_type,
            **kwargs,
        )

    def score_sequence(
        self,
        ref_seq: str,
        alt_seq: str,
        models: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        return_attention: bool = False,
    ) -> pd.DataFrame:
        """Score pre-extracted REF/ALT sequence pair."""
        sequences_df = pd.DataFrame([{
            "chrom": "user", "pos": 0, "ref": "N", "alt": "N",
            "ref_seq": ref_seq, "alt_seq": alt_seq,
        }])
        model_names = self._resolve_models(models, model_type)
        ref_ids, ref_mask, alt_ids, alt_mask = self._tokenize_sequences(
            sequences_df, n_processes=1,
        )
        ref_dataset = TensorDataset(ref_ids, ref_mask)
        alt_dataset = TensorDataset(alt_ids, alt_mask)
        gpus = [0] if torch.cuda.is_available() else None
        results = self._score_single_or_model_parallel(
            ref_dataset, alt_dataset, model_names,
            gpus=gpus, batch_size=1, num_workers=0,
            return_attention=return_attention,
        )
        return self._build_result_df(sequences_df, results, model_names)

    # ==================================================================
    # STEP 1: Sequence extraction with disk cache
    # ==================================================================
    def _extract_sequences_with_cache(
        self,
        variants: pd.DataFrame,
        flank: int,
        cache_dir: Optional[str],
        n_processes: Optional[int],
    ) -> pd.DataFrame:
        """Extract REF/ALT sequences, using disk cache when possible."""
        # If user already provided ref_seq/alt_seq columns, skip extraction
        if "ref_seq" in variants.columns and "alt_seq" in variants.columns:
            return variants

        if not self._genome_path:
            raise ValueError(
                "Reference genome not provided. Either:\n"
                "  1. Pass genome='/path/to/hg38.fa' to DVR()\n"
                "  2. Add ref_seq/alt_seq columns to your DataFrame"
            )

        # Compute cache key
        cache_key = compute_cache_key(
            vcf_path=None,
            genome_path=self._genome_path,
            n_variants=len(variants),
            flank=flank,
            extra=str(variants.iloc[0].to_dict()) + str(variants.iloc[-1].to_dict()),
        )

        if cache_dir is not None:
            cache_root = get_cache_dir(cache_dir) / cache_key
            seq_cache = cache_root / "sequences.tsv"

            cached = load_sequences_cache(seq_cache)
            if cached is not None and len(cached) == len(variants):
                print(f"✓ Loaded cached sequences from {seq_cache}")
                return cached

        # Extract in parallel
        var_tuples = list(zip(
            variants["chrom"].astype(str),
            variants["pos"].astype(int),
            variants["ref"].astype(str),
            variants["alt"].astype(str),
        ))
        sequences_df = extract_sequences_parallel(
            var_tuples,
            genome_path=self._genome_path,
            flank=flank,
            coord_offset=self._coord_offset,
            n_processes=n_processes,
        )

        # Save to cache
        if cache_dir is not None:
            cache_root = get_cache_dir(cache_dir) / cache_key
            seq_cache = cache_root / "sequences.tsv"
            save_sequences_cache(sequences_df, seq_cache)
            save_meta({
                "genome": self._genome_path,
                "n_variants": len(variants),
                "flank": flank,
                "coord_offset": self._coord_offset,
            }, cache_root / "meta.json")
            print(f"✓ Saved sequence cache to {seq_cache}")

        return sequences_df

    # ==================================================================
    # STEP 2: Tokenization with disk cache
    # ==================================================================
    def _tokenize_with_cache(
        self,
        sequences_df: pd.DataFrame,
        cache_dir: Optional[str],
        n_processes: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize REF/ALT sequences once, with disk cache."""
        if cache_dir is not None:
            cache_key = compute_cache_key(
                vcf_path=None,
                genome_path=self._genome_path,
                n_variants=len(sequences_df),
                flank=DEFAULT_FLANK,
                extra=str(sequences_df.iloc[0].to_dict()),
            )
            cache_root = get_cache_dir(cache_dir) / cache_key
            features_cache = cache_root / "features.pt"

            if features_cache.exists():
                print(f"✓ Loaded cached features from {features_cache}")
                cached = torch.load(features_cache)
                return (
                    cached["ref_input_ids"],
                    cached["ref_attention_mask"],
                    cached["alt_input_ids"],
                    cached["alt_attention_mask"],
                )

        # Tokenize fresh
        ref_ids, ref_mask, alt_ids, alt_mask = self._tokenize_sequences(
            sequences_df, n_processes=n_processes,
        )

        # Save tensor cache
        if cache_dir is not None:
            cache_root = get_cache_dir(cache_dir) / cache_key
            features_cache = cache_root / "features.pt"
            cache_root.mkdir(parents=True, exist_ok=True)
            torch.save({
                "ref_input_ids": ref_ids,
                "ref_attention_mask": ref_mask,
                "alt_input_ids": alt_ids,
                "alt_attention_mask": alt_mask,
            }, features_cache)
            print(f"✓ Saved feature cache to {features_cache}")

        return ref_ids, ref_mask, alt_ids, alt_mask

    def _tokenize_sequences(
        self,
        sequences_df: pd.DataFrame,
        n_processes: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize REF and ALT sequences with the DNABERT 6-mer tokenizer.
        All 462 models share the same vocab, so we only need to do this once.
        """
        # Load any DNABERT tokenizer (vocab is identical across all models)
        first_model = self.registry.list()[0]
        tokenizer = AutoTokenizer.from_pretrained(
            first_model.hf_repo, subfolder=first_model.subfolder,
            cache_dir=self.hf_cache_dir,
        )

        n_proc = get_default_n_processes(n_processes)
        print(f"Tokenizing {len(sequences_df):,} REF + {len(sequences_df):,} ALT "
              f"sequences using {n_proc} processes...")

        ref_seqs = sequences_df["ref_seq"].tolist()
        alt_seqs = sequences_df["alt_seq"].tolist()

        # Parallel k-mer conversion
        ref_kmers = kmerize_parallel(ref_seqs, n_processes=n_proc)
        alt_kmers = kmerize_parallel(alt_seqs, n_processes=n_proc)

        # Single tokenizer call for all sequences (HF tokenizers are already fast in Rust)
        ref_inputs = tokenizer(
            ref_kmers, return_tensors="pt", max_length=DEFAULT_MAX_LENGTH,
            truncation=True, padding="max_length",
        )
        alt_inputs = tokenizer(
            alt_kmers, return_tensors="pt", max_length=DEFAULT_MAX_LENGTH,
            truncation=True, padding="max_length",
        )
        return (
            ref_inputs["input_ids"],
            ref_inputs["attention_mask"],
            alt_inputs["input_ids"],
            alt_inputs["attention_mask"],
        )

    # ==================================================================
    # STEP 4: GPU scoring
    # ==================================================================
    def _resolve_multi_gpu_mode(
        self,
        mode: str,
        model_names: List[str],
        gpus: Optional[List[int]],
    ) -> str:
        """Auto-select DataParallel vs model-parallel based on workload."""
        if mode != "auto":
            return mode
        if not gpus or len(gpus) <= 1:
            return "single"
        # If many models compared to GPUs → model-parallel
        # If few models (or single model) → data-parallel
        if len(model_names) >= 2 * len(gpus):
            return "model_parallel"
        return "data_parallel"

    def _score_single_or_model_parallel(
        self,
        ref_dataset: TensorDataset,
        alt_dataset: TensorDataset,
        model_names: List[str],
        gpus: Optional[List[int]],
        batch_size: int,
        num_workers: int,
        return_attention: bool,
    ) -> Dict[str, dict]:
        """
        Score one model at a time on gpu[0]. Uses the in-memory model cache
        and fp16 inference for speed.
        """
        device = f"cuda:{gpus[0]}" if gpus else "cpu"
        all_results: Dict[str, dict] = {}

        try:
            from tqdm.auto import tqdm
            model_iter = tqdm(model_names, desc="Models", unit="model")
        except ImportError:
            model_iter = model_names

        for name in model_iter:
            info = self.registry.get(name)
            model, tokenizer = self._load_model(name, device, return_attention)

            ref_probs = self._run_dataloader(
                model, ref_dataset, device, batch_size, num_workers,
            )
            alt_probs = self._run_dataloader(
                model, alt_dataset, device, batch_size, num_workers,
            )
            all_results[name] = {
                "prob_ref": ref_probs,
                "prob_alt": alt_probs,
                "type": info.model_type,
            }

        return all_results

    def _score_data_parallel(
        self,
        ref_dataset: TensorDataset,
        alt_dataset: TensorDataset,
        model_names: List[str],
        gpus: List[int],
        batch_size: int,
        num_workers: int,
        return_attention: bool,
    ) -> Dict[str, dict]:
        """
        Use torch.nn.DataParallel to split each batch across multiple GPUs.
        Best for few models, many variants.
        """
        primary_device = f"cuda:{gpus[0]}"
        all_results: Dict[str, dict] = {}

        try:
            from tqdm.auto import tqdm
            model_iter = tqdm(model_names, desc="Models", unit="model")
        except ImportError:
            model_iter = model_names

        for name in model_iter:
            info = self.registry.get(name)
            # Note: DataParallel-wrapped models are NOT cached because the
            # wrapping is GPU-set specific. We cache the base model instead.
            model, tokenizer = self._load_model(name, primary_device, return_attention)

            # Wrap with DataParallel — splits each batch across all listed GPUs
            if len(gpus) > 1:
                dp_model = torch.nn.DataParallel(model, device_ids=gpus)
            else:
                dp_model = model

            # Effective batch = batch_size × n_gpus (each GPU handles batch_size)
            effective_batch = batch_size * len(gpus)
            ref_probs = self._run_dataloader(
                dp_model, ref_dataset, primary_device, effective_batch, num_workers,
            )
            alt_probs = self._run_dataloader(
                dp_model, alt_dataset, primary_device, effective_batch, num_workers,
            )
            all_results[name] = {
                "prob_ref": ref_probs,
                "prob_alt": alt_probs,
                "type": info.model_type,
            }

            # Don't delete `model` — it's in the cache. Just drop the DP wrapper.
            del dp_model

        return all_results

    def _run_dataloader(
        self,
        model,
        dataset: TensorDataset,
        device: str,
        batch_size: int,
        num_workers: int,
    ) -> np.ndarray:
        """
        Run a model over a dataset using DataLoader with prefetching.
        Returns a 1D numpy array of class-1 probabilities.
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        all_probs = []
        softmax = torch.nn.Softmax(dim=-1)

        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                probs = softmax(logits)[:, 1].detach().cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs)

    # ==================================================================
    # STEP 5: Vectorized result DataFrame construction
    # ==================================================================
    def _build_result_df(
        self,
        sequences_df: pd.DataFrame,
        scoring_results: Dict[str, dict],
        model_names: List[str],
    ) -> pd.DataFrame:
        """Build the result DataFrame using vectorized NumPy ops."""
        rows = []
        n_var = len(sequences_df)
        chrom_arr = sequences_df["chrom"].values
        pos_arr = sequences_df["pos"].values
        ref_arr = sequences_df["ref"].values
        alt_arr = sequences_df["alt"].values

        for name in model_names:
            r = scoring_results[name]
            p_ref = r["prob_ref"]
            p_alt = r["prob_alt"]

            # Vectorized log-odds and score change (matches v0.1.x math)
            # Cast to float32 (fp16 outputs need higher precision for log-odds)
            # and clip to avoid log(0) / division by zero
            p_ref = np.clip(p_ref.astype(np.float32), 1e-6, 1 - 1e-6)
            p_alt = np.clip(p_alt.astype(np.float32), 1e-6, 1 - 1e-6)

            log_odds_ref = np.log2(p_ref / (1 - p_ref))
            log_odds_alt = np.log2(p_alt / (1 - p_alt))
            log_odds_ratio = log_odds_ref - log_odds_alt
            score_change = (p_alt - p_ref) * np.maximum(p_ref, p_alt)

            df_model = pd.DataFrame({
                "chrom": chrom_arr,
                "pos": pos_arr,
                "ref": ref_arr,
                "alt": alt_arr,
                "model": name,
                "type": r["type"],
                "prob_ref": np.round(p_ref, 6),
                "prob_alt": np.round(p_alt, 6),
                "log_odds_ratio": np.round(log_odds_ratio, 4),
                "score_change": np.round(score_change, 6),
            })
            rows.append(df_model)

        return pd.concat(rows, ignore_index=True)

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
        return f"DVR(device='{self.device}'{g}, {self.registry})"
