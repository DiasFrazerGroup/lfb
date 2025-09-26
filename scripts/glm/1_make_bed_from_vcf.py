import polars as pl
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Create a BED file from a VCF file")
    parser.add_argument(
        "--variants_vcf",
        required=True,
        help="Path to the variants VCF file with headers: CHROM, POS, ID, ... (standard VCF format)",
    )
    parser.add_argument(
        "--output_bed", required=True, help="Path to the output BED file"
    )
    return parser.parse_args()


def create_bed_from_variants(df, bed_filename):
    """
    Create a BED file from a DataFrame of variants.
    For a single-base variant, BED coordinates are 0-based: [POS-1, POS].
    """
    with open(bed_filename, "w") as f:
        # Iterate over rows; adjust indices if your schema changes.
        score = "0"  # we don't use this score information
        strand = "+"  # All on the positive strand
        for row in df.iter_rows():
            chrom = "chr" + str(row[0])  # CHROM
            pos = row[1]  # POS (assumed 1-based)
            var_id = row[2]  # ID (could be used as a name)
            start = pos - 1  # Convert to 0-based start
            end = pos  # End is non-inclusive
            f.write(f"{chrom}\t{start}\t{end}\t{var_id}\t{score}\t{strand}\n")


if __name__ == "__main__":
    args = parse_args()

    # Load variants data from VCF (skip comment lines starting with #)
    df = pl.read_csv(
        args.variants_vcf,
        separator="\t",
        comment_prefix="##",
        schema_overrides={"#CHROM": pl.String},
    ).rename({"#CHROM": "CHROM"})
    print(f"Loaded {len(df)} variants from VCF")

    # Create the BED file of human variant coordinates
    create_bed_from_variants(df, args.output_bed)
    print(f"Created BED file: {args.output_bed}")
