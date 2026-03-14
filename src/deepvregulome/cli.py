"""
Command-line interface for DeepVRegulome.

Usage:
    # Score a single variant
    deepvregulome score --chrom chr1 --pos 3456782 --ref A --alt TA \
        --models CTCFL SP1 MYC --genome hg38.fa

    # Score a VCF file
    deepvregulome score-vcf variants.vcf --models CTCFL SP1 --genome hg38.fa -o results.tsv

    # Score from sequences
    deepvregulome score-seq --ref ATCG... --alt ATCG... --models CTCFL

    # List available models
    deepvregulome list
    deepvregulome list --type TF
    deepvregulome search ZNF
"""

import argparse
import sys


def cmd_score(args):
    from deepvregulome import DVR

    dvr = DVR(genome=args.genome, device=args.device)
    result = dvr.score_variant(
        chrom=args.chrom,
        pos=args.pos,
        ref=args.ref,
        alt=args.alt,
        models=args.models,
        model_type=args.type,
        return_attention=args.attention,
    )

    if args.output:
        result.to_csv(args.output, sep="\t", index=False)
        print(f"Results saved to {args.output}")
    else:
        print(result.to_string(index=False))


def cmd_score_seq(args):
    from deepvregulome import DVR

    dvr = DVR(device=args.device)
    result = dvr.score_sequence(
        ref_seq=args.ref,
        alt_seq=args.alt,
        models=args.models,
        model_type=args.type,
        return_attention=args.attention,
    )

    if args.output:
        result.to_csv(args.output, sep="\t", index=False)
    else:
        print(result.to_string(index=False))


def cmd_score_vcf(args):
    from deepvregulome import DVR

    dvr = DVR(genome=args.genome, device=args.device)
    result = dvr.score_vcf(
        vcf_path=args.vcf,
        models=args.models,
        model_type=args.type,
        return_attention=args.attention,
        max_variants=args.max_variants,
    )

    output = args.output or args.vcf.replace(".vcf", "_dvr_scores.tsv")
    result.to_csv(output, sep="\t", index=False)
    print(f"Scored {len(result)} variant×model combinations → {output}")
    disrupted = result["disrupted"].sum() if "disrupted" in result.columns else 0
    print(f"Disrupted: {disrupted}/{len(result)}")


def cmd_list(args):
    from deepvregulome import ModelRegistry

    reg = ModelRegistry()
    models = reg.list(model_type=args.type)
    print(f"{'Name':<15} {'Type':<8} {'Acc':>6} {'ROC-AUC':>8} {'Peaks':>8}")
    print("-" * 50)
    for m in models:
        print(f"{m.name:<15} {m.model_type:<8} {m.accuracy:>6.1f} {m.roc_auc:>8.2f} {m.peak_count:>8}")
    print(f"\nTotal: {len(models)} models")


def cmd_search(args):
    from deepvregulome import ModelRegistry

    reg = ModelRegistry()
    matches = reg.search(args.query)
    if matches:
        print(f"Models matching '{args.query}': {', '.join(matches)}")
    else:
        print(f"No models found matching '{args.query}'")


def main():
    parser = argparse.ArgumentParser(
        prog="deepvregulome",
        description="DeepVRegulome: Regulatory variant effect prediction",
    )
    sub = parser.add_subparsers(dest="command")

    # --- score ---
    p_score = sub.add_parser("score", help="Score a single variant")
    p_score.add_argument("--chrom", required=True, help="Chromosome (e.g., chr1)")
    p_score.add_argument("--pos", type=int, required=True, help="1-based position")
    p_score.add_argument("--ref", required=True, help="Reference allele")
    p_score.add_argument("--alt", required=True, help="Alternate allele")
    p_score.add_argument("--models", nargs="+", help="Model names (e.g., CTCFL SP1)")
    p_score.add_argument("--type", choices=["TF", "HISTONE"], help="Score all models of this type")
    p_score.add_argument("--genome", required=True, help="Reference genome FASTA")
    p_score.add_argument("--attention", action="store_true", help="Include attention scores")
    p_score.add_argument("--device", default=None, help="cuda or cpu")
    p_score.add_argument("-o", "--output", help="Output TSV file")
    p_score.set_defaults(func=cmd_score)

    # --- score-seq ---
    p_seq = sub.add_parser("score-seq", help="Score from pre-extracted sequences")
    p_seq.add_argument("--ref", required=True, help="Reference sequence (301bp)")
    p_seq.add_argument("--alt", required=True, help="Alternate sequence")
    p_seq.add_argument("--models", nargs="+", help="Model names")
    p_seq.add_argument("--type", choices=["TF", "HISTONE"], help="Score all of type")
    p_seq.add_argument("--attention", action="store_true")
    p_seq.add_argument("--device", default=None)
    p_seq.add_argument("-o", "--output", help="Output TSV file")
    p_seq.set_defaults(func=cmd_score_seq)

    # --- score-vcf ---
    p_vcf = sub.add_parser("score-vcf", help="Score variants from a VCF file")
    p_vcf.add_argument("vcf", help="Input VCF file")
    p_vcf.add_argument("--models", nargs="+", help="Model names")
    p_vcf.add_argument("--type", choices=["TF", "HISTONE"], help="Score all of type")
    p_vcf.add_argument("--genome", required=True, help="Reference genome FASTA")
    p_vcf.add_argument("--attention", action="store_true")
    p_vcf.add_argument("--max-variants", type=int, help="Max variants to score")
    p_vcf.add_argument("--device", default=None)
    p_vcf.add_argument("-o", "--output", help="Output TSV file")
    p_vcf.set_defaults(func=cmd_score_vcf)

    # --- list ---
    p_list = sub.add_parser("list", help="List available models")
    p_list.add_argument("--type", choices=["TF", "HISTONE"], help="Filter by type")
    p_list.set_defaults(func=cmd_list)

    # --- search ---
    p_search = sub.add_parser("search", help="Search model names")
    p_search.add_argument("query", help="Search string (e.g., ZNF, GATA)")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
