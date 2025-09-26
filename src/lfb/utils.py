from collections import defaultdict
from typing import List, Tuple
from lfb.constants import AMINO_ACID_MAP, AMINO_ACIDS
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
import torch
import pandas as pd


def parse_fasta(fasta_path: str) -> Tuple[List[str], List[str]]:
    sequences = []
    labels = []
    current_label = None
    current_seq = []

    with open(fasta_path, "r") as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save previous sequence if it exists
                if current_label is not None:
                    sequences.append("".join(current_seq))
                    labels.append(current_label)

                # Start new sequence
                current_label = line[1:].split()[0]  # Get first word after '>'
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget to save the last sequence
        if current_label is not None:
            sequences.append("".join(current_seq))
            labels.append(current_label)

    return sequences, labels


def parse_fasta_to_tensor(fasta_path: str) -> Tuple[torch.Tensor, List[str]]:
    sequences = []
    labels = []
    amino_map = defaultdict(lambda: 20, AMINO_ACID_MAP)

    current_label = None
    current_seq = []

    with open(fasta_path, "r") as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save previous sequence if it exists
                if current_label is not None:
                    seq = [amino_map[aa.upper()] for aa in "".join(current_seq)]
                    sequences.append(seq)
                    labels.append(current_label)

                # Start new sequence
                current_label = line[1:].split()[0]  # Get first word after '>'
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget to save the last sequence
        if current_label is not None:
            seq = [amino_map[aa.upper()] for aa in "".join(current_seq)]
            sequences.append(seq)
            labels.append(current_label)

    tensor = torch.tensor(sequences, dtype=torch.long)
    return tensor, labels


def write_tensor_to_fasta(tensor: torch.Tensor, labels: List[str], output_fasta: str):
    reverse_map = {v: k for k, v in AMINO_ACID_MAP.items()}

    with open(output_fasta, "w") as f:
        for i, label in enumerate(labels):
            seq = "".join([reverse_map[int(aa)] for aa in tensor[i]])
            f.write(f">{label}\n{seq}\n")


def get_tokenizer(file):
    """
    Get a tokenizer from a json file.
    """
    with open(file, "r") as f:
        tokenizer_object = Tokenizer.from_str(f.read())
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object)
    tokenizer.pad_token = "<|pad|>"
    tokenizer.pad_token_id = 0
    return tokenizer


def generate_all_possible_variants(sequence: str, start_pos: int = 1) -> List[str]:
    """
    Generate all possible single amino acid substitutions for a protein sequence.

    Args:
        sequence: Protein sequence string
        start_pos: Starting position number (default 1 for 1-indexed)

    Returns:
        List of variant strings in format "A123T"
    """
    variants = []
    amino_acids = list(AMINO_ACIDS)

    for i, original_aa in enumerate(sequence):
        position = i + start_pos
        for new_aa in amino_acids:
            if new_aa != original_aa:  # Skip synonymous mutations
                variant = f"{original_aa}{position}{new_aa}"
                variants.append(variant)

    return variants


def load_variants_from_table(variants_table_path: str) -> List[str]:
    """
    Load variants from a table file.

    Args:
        variants_table_path: Path to CSV/TSV file with 'mutant' column

    Returns:
        List of variant strings
    """
    try:
        # Try reading as CSV first
        df = pd.read_csv(variants_table_path)
    except:
        # Try reading as TSV
        df = pd.read_csv(variants_table_path, sep="\t")

    if "mutant" not in df.columns:
        raise ValueError("Variants table must have a 'mutant' column")

    return df["mutant"].tolist()
