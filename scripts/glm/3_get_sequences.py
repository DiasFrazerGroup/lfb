import polars as pl
import subprocess
import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract sequences for mapped variants and apply mutations"
    )
    parser.add_argument(
        "--clinvar_csv", required=True, help="Path to the filtered ClinVar CSV file"
    )
    parser.add_argument("--bed_file", required=True, help="Path to the mapped BED file")
    parser.add_argument(
        "--output_dir", required=True, help="Directory to store output files"
    )
    parser.add_argument(
        "--alignment", required=True, help="Path to the HAL alignment file"
    )
    parser.add_argument("--species", required=True, help="Target species name")
    parser.add_argument(
        "--final_length", type=int, default=8192, help="Final length of sequences"
    )
    parser.add_argument(
        "--bin_directory",
        default="/usr/bin",
        help="Binary directory containing hal tools",
    )
    return parser.parse_args()


def reverse_complement(seq):
    """
    Return the reverse complement of a DNA sequence.
    """
    complement = {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "a": "t",
        "c": "g",
        "g": "c",
        "t": "a",
        "N": "N",
        "n": "n",
    }
    return "".join(complement.get(base, base) for base in reversed(seq))


def run_hal2fasta(hal_file, species, chrom, start, length, bin_directory):
    """
    Run hal2fasta to extract sequences around mapped coordinates.
    """
    start = str(start)
    length = str(length)
    command = [
        f"{bin_directory}/hal2fasta",
        hal_file,
        species,
        "--sequence",
        chrom,
        "--start",
        start,
        "--length",
        length,
        "--upper",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"hal2fasta error for {chrom}:{start}-{int(start)+int(length)}:",
            result.stderr,
        )
        return None

    fasta_string = result.stdout
    # Skip the header line and join remaining lines
    return "".join(fasta_string.split("\n")[1:]).strip()


def apply_variant(sequence, ref, alt, position, final_length=8192):
    """
    Apply a variant to a sequence and ensure it's the right length.
    Position is 0-based inside the sequence.
    REF/ALT parts will be UPPERCASE, rest will be lowercase for clarity.
    """
    # Convert sequence to lowercase
    sequence = sequence.lower()

    # Create wild-type sequence with REF in uppercase
    wt_sequence = (
        sequence[:position]
        + sequence[position : position + len(ref)].upper()
        + sequence[position + len(ref) :]
    )

    # Create ref substitution sequence "forcing" ref to be in the liftover
    ref_substitution = (
        sequence[:position] + ref.upper() + sequence[position + len(ref) :]
    )

    # Create mutant sequence by applying the variant (ALT in uppercase)
    mutant = sequence[:position] + alt.upper() + sequence[position + len(ref) :]

    # Center both sequences on the variant
    variant_center = position + len(ref) // 2

    # Calculate start positions to achieve final_length
    wt_start = max(0, variant_center - final_length // 2)
    wt_end = min(len(wt_sequence), wt_start + final_length)

    mt_start = max(0, variant_center - final_length // 2)
    mt_end = min(len(mutant), mt_start + final_length)

    # Extract the sequences
    wt_seq = wt_sequence[wt_start:wt_end]
    ref_seq = ref_substitution[wt_start:wt_end]
    mt_seq = mutant[mt_start:mt_end]

    # Pad if necessary to reach final_length
    wt_seq = pad_sequence(wt_seq, final_length)
    ref_seq = pad_sequence(ref_seq, final_length)
    mt_seq = pad_sequence(mt_seq, final_length)

    return wt_seq, ref_seq, mt_seq


def pad_sequence(seq, target_length):
    """Pad sequence with N's to reach target length"""
    if len(seq) < target_length:
        pad_length = target_length - len(seq)
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad
        return "N" * left_pad + seq + "N" * right_pad
    elif len(seq) > target_length:
        # Trim from both ends to keep centered
        excess = len(seq) - target_length
        left_trim = excess // 2
        return seq[left_trim : left_trim + target_length]
    else:
        return seq


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load ClinVar data
    df_clinvar = pl.read_csv(args.clinvar_csv, schema_overrides={"CHROM": pl.String})
    print(f"Loaded {len(df_clinvar)} variants from ClinVar")

    # Create a dictionary for fast lookup by ID
    variant_dict = {row["ID"]: row for row in df_clinvar.to_dicts()}

    # Load BED file
    bed_entries = []
    with open(args.bed_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            bed_entries.append(
                {
                    "chrom": parts[0],
                    "start": int(parts[1]),
                    "end": int(parts[2]),
                    "id": int(parts[3]),
                    "strand": parts[5],
                }
            )

    print(f"Loaded {len(bed_entries)} mapped coordinates from BED file")

    # Process each BED entry
    wt_fasta = os.path.join(args.output_dir, f"{args.species.lower()}_wildtype.fa")
    mt_fasta = os.path.join(args.output_dir, f"{args.species.lower()}_mutant.fa")
    ref_fasta = os.path.join(args.output_dir, f"{args.species.lower()}_ref.fa")

    with (
        open(wt_fasta, "w") as wt_file,
        open(mt_fasta, "w") as mt_file,
        open(ref_fasta, "w") as ref_file,
    ):
        for i, entry in enumerate(bed_entries):
            variant_id = entry["id"]

            # Find matching variant in ClinVar data
            if variant_id not in variant_dict:
                print(f"Warning: Variant ID {variant_id} not found in ClinVar data")
                continue

            variant = variant_dict[variant_id]

            # Calculate sequence extraction parameters
            # Extract a larger region to account for potential large indels
            extraction_length = (
                args.final_length + 128
            )  # Extract extra as indels can be up to 64bp
            extraction_start = max(0, entry["start"] - extraction_length // 2)

            # Extract sequence
            sequence = run_hal2fasta(
                args.alignment,
                args.species,
                entry["chrom"],
                extraction_start,
                extraction_length,
                args.bin_directory,
            )

            if not sequence:
                print(f"Failed to extract sequence for variant {variant_id}")
                continue

            # Get strand information (default to '+' if not available)
            strand = entry["strand"]

            # Calculate position of variant within extracted sequence
            var_position = entry["start"] - extraction_start

            # Get the ref and alt alleles
            ref = variant["REF"]
            alt = variant["ALT"]

            # For negative strand, get reverse complement of sequence and adjust position
            if strand == "-":
                sequence = reverse_complement(sequence)

                # Recalculate position in reverse complemented sequence
                var_position = len(sequence) - var_position - len(ref)

            # Apply variant
            wt_seq, ref_seq, mt_seq = apply_variant(
                sequence, ref, alt, var_position, args.final_length
            )

            if wt_seq and mt_seq:
                # Write to FASTA files
                ref_match = "ref_match" if (ref_seq == wt_seq) else "ref_mismatch"
                descr = f"{args.species},{entry['chrom']},{entry['start']},{entry['end']},{entry['id']},{entry['strand']},{ref_match}"
                wt_file.write(f">{descr},wt\n{wt_seq}\n")
                mt_file.write(f">{descr},mt\n{mt_seq}\n")
                if ref_seq != wt_seq:
                    ref_file.write(f">{descr},ref\n{ref_seq}\n")

                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1}/{len(bed_entries)} variants")

    print(f"Processing complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
