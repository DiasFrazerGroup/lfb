import argparse
from lfb.alignment import parse_tsv, filter_alignment
from lfb.inference.scorer import Scorer
from lfb.model.model import ESM, ProGen
from lfb.utils import generate_all_possible_variants, load_variants_from_table


def main():
    parser = argparse.ArgumentParser(
        description="Predict LFB scores for protein variants"
    )

    # Required arguments
    parser.add_argument(
        "--tsv_alignment_file",
        type=str,
        required=True,
        help="Path to alignment TSV file, headerless format. Must contain 'query', 'target', 'qstart', 'qend', 'tstart', 'tend', 'cigar', 'qseq', 'tseq'.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output CSV file for predictions"
    )

    # Optional arguments
    parser.add_argument(
        "--tsv_alignment_header",
        type=str,
        default="query,target,qstart,qend,tstart,tend,cigar,qseq,tseq",
        help="Header of the TSV alignment file, provide as a comma-separated string. Must include 'query', 'target', 'qstart', 'qend', 'tstart', 'tend', 'cigar', 'qseq', 'tseq'.",
    )

    # Model and inference arguments
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="Huggingface model name for ESM, or path to a ProGen2 model if the flag `--progen` is provided",
    )
    parser.add_argument(
        "--progen",
        action="store_true",
        help="Use a ProGen2 model for scoring",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["masked-marginals", "unmasked-marginals", "autoregressive"],
        default="unmasked-marginals",
        help="Inference mode for scoring",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=1024,
        help="Maximum context length for chunking sequences",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use half precision floats for inference",
    )

    # Variant specification
    parser.add_argument(
        "--variants_table",
        type=str,
        default=None,
        help="Path to CSV/TSV file with 'mutant' column. If not provided, all possible variants will be generated.",
    )

    # Alignment filtering options
    parser.add_argument(
        "--coverage",
        type=float,
        default=None,
        help="Minimum coverage threshold for filtering alignment, in range (0,1)",
    )
    parser.add_argument(
        "--percentage_identity",
        type=float,
        default=None,
        help="Minimum fraction of identity threshold for filtering alignment, in range (0,1)",
    )
    parser.add_argument(
        "--random_subsample_num",
        type=int,
        default=None,
        help="Number of sequences to randomly subsample from (filtered) alignment",
    )

    # LFB options
    parser.add_argument(
        "--treat_unmapped_as_zero",
        action="store_true",
        help="Treat unmapped variants as score 0 instead of null",
    )
    parser.add_argument(
        "--no_lfb",
        action="store_true",
        help="Skip LFB aggregation and use only first sequence",
    )

    args = parser.parse_args()

    # Parse and filter alignment data
    print(f"> Parsing TSV file: {args.tsv_alignment_file}")
    alignment_data = parse_tsv(
        args.tsv_alignment_file, header=args.tsv_alignment_header
    )

    print(f"> Original alignment: {len(alignment_data.labels)} sequences")

    alignment_data = filter_alignment(
        alignment_data,
        min_coverage=args.coverage,
        min_pid=args.percentage_identity,
        downsample_num=args.random_subsample_num,
    )

    if len(alignment_data.labels) == 0:
        print(">>> No sequences remaining after filtering <<<")
        exit(1)

    print(f"> Filtered alignment: {len(alignment_data.labels)} sequences")

    # Get variants to score
    if args.variants_table:
        print(f"> Loading variants from: {args.variants_table}")
        variants = load_variants_from_table(args.variants_table)
        print(f"> Loaded {len(variants)} variants from table")
    else:
        print("> Generating all possible single variants...")
        # Use the first sequence as reference for generating variants
        reference_sequence = alignment_data.sequences[0]
        variants = generate_all_possible_variants(reference_sequence)
        print(f"> Generated {len(variants)} possible variants")

    # Initialize model and scorer
    print(f"> Initializing model: {args.model}")
    if args.progen:
        assert (
            args.inference_mode == "autoregressive"
        ), "ProGen only supports autoregressive inference mode"
        model = ProGen(args.model, device=args.device, fp16=args.fp16)
    else:
        assert args.inference_mode in [
            "masked-marginals",
            "unmasked-marginals",
        ], "ESM only supports masked-marginals and unmasked-marginals inference modes"
        model = ESM(args.model, device=args.device, fp16=args.fp16)

    print(f"> Creating scorer with mode: {args.inference_mode}")
    scorer = Scorer(
        model=model,
        scoring_mode=args.inference_mode,
        max_context_length=args.context_length,
    )

    # Score variants
    print(f"> Scoring {len(variants)} variants...")
    if args.no_lfb:
        print("  > Using single sequence (no LFB aggregation)")
        scores_df = scorer.score_variants_without_lfb(
            alignment_data, variants, batch_size=args.batch_size
        )
    else:
        print("  > Using LFB aggregation across alignment")
        scores_df = scorer.score_variants(
            alignment_data,
            variants,
            treat_unmapped_as_zero=args.treat_unmapped_as_zero,
            batch_size=args.batch_size,
        )

    # Save results
    print(f"> Saving predictions to: {args.output}")
    scores_df.write_csv(args.output)

    print(f"> Successfully scored {len(scores_df)} variants")
    print(f"> Score statistics:")
    print(f"  Mean: {scores_df['score'].mean():.4f}")
    print(f"  Std:  {scores_df['score'].std():.4f}")
    print(f"  Min:  {scores_df['score'].min():.4f}")
    print(f"  Max:  {scores_df['score'].max():.4f}")


if __name__ == "__main__":
    main()
