import argparse
import csv
import time
from evo2.models import Evo2


def parse_fasta(fasta_path):
    """
    A simple FASTA parser that yields (label, sequence) tuples.
    """
    with open(fasta_path, "r") as file:
        label = None
        sequence_lines = []
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            if line.startswith(">"):
                if label is not None:
                    yield label, "".join(sequence_lines)
                label = line[1:].strip()  # Remove '>' and any surrounding whitespace
                sequence_lines = []
            else:
                sequence_lines.append(line)
        # Yield the last entry in the file if any
        if label is not None:
            yield label, "".join(sequence_lines)


def filter_sequences(labels, sequences, id_subset_path):
    """
    Filter sequences based on the ID subset.
    """

    if id_subset_path is None:
        print("No ID subset provided, returning all sequences")
        return labels, sequences
    else:
        print(f"Filtering sequences based on ID subset file: {id_subset_path}")

    # Parse the ID subset file using csv.DictReader, which allows the ID column to appear anywhere in the CSV.
    id_subset = []
    with open(id_subset_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            id_subset.append(int(row["ID"]))

    filtered_labels = []
    filtered_sequences = []
    for label, sequence in zip(labels, sequences):
        ID = int(label.split(",")[3])
        if ID in id_subset:
            filtered_labels.append(label)
            filtered_sequences.append(sequence)
    print(f"Filtered to {len(filtered_labels)} sequences out of {len(labels)}")
    if len(filtered_labels) == 0:
        print("No sequences left after filtering, exiting")
        exit(0)
    return filtered_labels, filtered_sequences


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Score sequences in a FASTA file using Evo2 and output the results to a CSV file."
    )
    parser.add_argument(
        "input_fasta", help="Path to the input FASTA file with sequences"
    )
    parser.add_argument(
        "output_csv",
        help="Path to the output CSV file to store the label and score mapping",
    )
    parser.add_argument(
        "--model", help="Path to the Evo2 model to use", default="evo2_1b_base"
    )
    parser.add_argument(
        "--batch_size", help="Batch size to use for scoring", type=int, default=20
    )
    parser.add_argument(
        "--average_reverse_complement",
        action="store_true",
        help="If set, score sequences with average_reverse_complement=True",
    )
    parser.add_argument(
        "--id_subset",
        help="Path to the file containing csv with the IDs to score in ID column",
        default=None,
    )
    args = parser.parse_args()

    # Load the Evo2 model
    model = Evo2(args.model)

    # Parse FASTA file and collect labels and sequences
    labels = []
    sequences = []
    for label, sequence in parse_fasta(args.input_fasta):
        labels.append(label)
        sequences.append(sequence)

    # Filter sequences based on ID subset
    labels, sequences = filter_sequences(labels, sequences, args.id_subset)

    # Capitalise sequences to match the model's input format
    sequences = [seq.upper() for seq in sequences]

    # Score all sequences at once using model.score_sequences and measure time
    print(
        f"Scoring {len(sequences)} sequences with average_reverse_complement={args.average_reverse_complement}, batch_size={args.batch_size}"
    )
    start_time = time.perf_counter()
    scores = model.score_sequences(
        sequences,
        batch_size=args.batch_size,
        average_reverse_complement=args.average_reverse_complement,
    )
    end_time = time.perf_counter()

    # Calculate the scoring rate
    elapsed = end_time - start_time
    seq_per_sec = len(sequences) / elapsed if elapsed > 0 else float("inf")

    # Write the scores to the CSV file (each row: label, score)
    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["LOC", "ID", "STRAND", "REF_MATCH", "ALLELE", "SCORE"])
        for label, score in zip(labels, scores):
            species, chrom, start, end, ref_id, strand, ref_mismatch, ref_allele = (
                label.split(",")
            )
            writer.writerow(
                [
                    f"{species},{chrom}:{start}-{end}",
                    ref_id,
                    strand,
                    ref_mismatch,
                    ref_allele,
                    score,
                ]
            )

    # Print overall timing information
    print(
        f"Scored {len(sequences)} sequences in {elapsed:.2f} seconds ({seq_per_sec:.2f} sequences/second)."
    )


if __name__ == "__main__":
    main()
