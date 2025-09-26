import polars as pl
import subprocess
import sys
import os
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lift over ClinVar variants to Zoonomia species"
    )
    parser.add_argument("--bed_file", required=True, help="Path to the BED file")
    parser.add_argument(
        "--output_dir", required=True, help="Directory to store output files"
    )
    parser.add_argument(
        "--alignment", required=True, help="Path to the HAL alignment file"
    )
    parser.add_argument("--species", required=True, help="Name of the target species")
    parser.add_argument(
        "--bin_directory", default="/usr/bin", help="Directory containing hal tools"
    )
    return parser.parse_args()


def run_hal_liftover(
    hal_file, bed_in, target_species, bed_out, bin_directory, src_species="Homo_sapiens"
):
    """
    Run halLiftover to map variant coordinates from human to target species.
    """
    # Command: halLiftover <hal_file> <src_species> <in.bed> <target_species> <out.bed>
    command = [
        f"{bin_directory}/halLiftover",
        hal_file,
        src_species,
        bed_in,
        target_species,
        bed_out,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("halLiftover error:", result.stderr)
    else:
        print("halLiftover completed successfully.")
    return bed_out


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Which species to process
    target_species = args.species

    # Bed file of variants to lift over
    bed_file = args.bed_file

    # Run halLiftover
    mapped_bed = os.path.join(args.output_dir, f"{target_species.lower()}.bed")
    run_hal_liftover(
        args.alignment, bed_file, target_species, mapped_bed, args.bin_directory
    )

    print(f"Processing complete for {target_species}. Saved to {mapped_bed}.")


if __name__ == "__main__":
    main()
