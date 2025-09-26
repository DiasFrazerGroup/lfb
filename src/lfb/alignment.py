import torch
from typing import List, Tuple, Union
from dataclasses import dataclass
import re
from lfb.constants import AMINO_ACID_MAP
import numpy as np


@dataclass
class AlignmentData:
    sequences: List[str]
    gap_masks: List[List[bool]]
    mappings: List[List[Union[int, None]]]
    labels: List[str]


def parse_cigar(cigar_str: str) -> List[Tuple[int, str]]:
    """Parse the CIGAR string into a list of (length, operation) tuples."""
    return [
        (int(length), op)
        for length, op in re.findall(r"(\d+)([MIDNSHP=XB])", cigar_str)
    ]


def get_query_to_target_mapping(
    cigar_tuples: List[Tuple[int, str]],
    qstart: int,
    qend: int,
    tstart: int,
    tend: int,
    qlen: int,
) -> List[int]:
    """
    Args:
        cigar_tuples: List of (length, operation) from CIGAR string
        qstart: Start index in the query sequence (1-based)
        tstart: Start index in the target sequence (1-based)

    Returns:
        t_mapping: Mapping from query positions to target sequence positions
    """
    q_pos = qstart - 1  # Position in qseq
    t_pos = tstart - 1  # Position in tseq

    # Overall length: qstart-1 + (qend-qstart+1) + (qlen - qend)
    t_mapping = [None] * (qstart - 1)

    for length, op in cigar_tuples:
        if op == "M" or op == "=" or op == "X":
            # Match or mismatch: advance both sequences
            t_mapping.extend([t_pos + idx for idx in range(length)])
            q_pos += length
            t_pos += length
        elif op == "I":
            # Insertion in query, advance query
            t_mapping.extend([None] * length)
            q_pos += length
        elif op == "D":
            # Deletion in query, advance target
            t_pos += length
        else:
            raise ValueError(f"Unsupported CIGAR operation: {op}")
    t_mapping.extend([None] * (qlen - qend))
    assert len(t_mapping) == qlen

    return t_mapping


def parse_tsv(file_path: str, header: str) -> AlignmentData:
    """Parse TSV file and reconstruct alignments aligned to the query."""

    # Read TSV file
    with open(file_path, "r") as f:
        data_lines = f.readlines()

    # legacy_header = [
    #     "query",
    #     "target",
    #     "fident",
    #     "evalue",
    #     "bits",
    #     "taxid",
    #     "taxname",
    #     "taxlineage",
    #     "qstart",
    #     "qend",
    #     "tstart",
    #     "tend",
    #     "cigar",
    #     "qseq",
    #     "tseq",
    # ]

    header = header.strip().split(",")
    assert all(
        [
            x in header
            for x in [
                "query",
                "target",
                "qstart",
                "qend",
                "tstart",
                "tend",
                "cigar",
                "qseq",
                "tseq",
            ]
        ]
    )

    # Get indices of columns
    column_indices = {col: idx for idx, col in enumerate(header)}

    # Process each alignment
    target_sequences = []
    target_labels = []
    target_mappings = []

    for line in data_lines:
        fields = line.strip().split("\t")
        query = fields[column_indices["query"]]
        target = fields[column_indices["target"]]
        cigar = fields[column_indices["cigar"]]
        qseq = fields[column_indices["qseq"]]
        tseq = fields[column_indices["tseq"]]
        qstart = int(fields[column_indices["qstart"]])  # 0-based indexing
        qend = int(fields[column_indices["qend"]])
        tstart = int(fields[column_indices["tstart"]])
        tend = int(fields[column_indices["tend"]])

        # Parse CIGAR string
        cigar_tuples = parse_cigar(cigar)

        # Get query length
        qlen = len(qseq)

        # Get mapping from query onto target sequence
        target_mapping = get_query_to_target_mapping(
            cigar_tuples,
            qstart,
            qend,
            tstart,
            tend,
            qlen,
        )
        query_sequence = qseq
        query_label = query
        target_labels.append(target)
        target_sequences.append(tseq)
        target_mappings.append(target_mapping)

    # Collect all sequences
    sequences = [query_sequence, *target_sequences]
    labels = [query_label, *target_labels]
    mappings = [list(range(len(query_sequence))), *target_mappings]
    gap_masks = [[x == None for x in mapping] for mapping in mappings]

    return AlignmentData(sequences, gap_masks, mappings, labels)


def map_logits_to_alignments(
    alignment_data: AlignmentData, sequence_logits: List[torch.Tensor], vocab_size: int
) -> Tuple[torch.Tensor, List[str]]:
    """Align logits from different sequences according to sequence alignments."""
    num_sequences = len(alignment_data.sequences)
    query_length = len(alignment_data.sequences[0])

    aligned_logits = torch.zeros(num_sequences, query_length, vocab_size)
    gap_masks = alignment_data.gap_masks
    aligned_sequences = []

    for i in range(num_sequences):
        mapping = alignment_data.mappings[i]
        target_seq = alignment_data.sequences[i]
        target_logits = sequence_logits[i]
        seq = []
        for j in range(query_length):
            k = mapping[j]
            if k == None:
                seq.append("-")
            else:
                seq.append(target_seq[k])
                aligned_logits[i, j, :] = target_logits[k, :]

        seq = "".join(seq)
        aligned_sequences.append(seq)

    return aligned_logits, gap_masks, aligned_sequences


def get_aligned_sequences(alignment_data: AlignmentData) -> List[str]:
    """Get a list of aligned sequences from an AlignmentData object."""
    num_sequences = len(alignment_data.sequences)
    query_length = len(alignment_data.sequences[0])
    aligned_sequences = []
    for i in range(num_sequences):
        mapping = alignment_data.mappings[i]
        target_seq = alignment_data.sequences[i]
        seq = []
        for j in range(query_length):
            k = mapping[j]
            if k == None:
                seq.append("-")
            else:
                seq.append(target_seq[k])

        seq = "".join(seq)
        aligned_sequences.append(seq)
    return aligned_sequences


def get_msa_tensor(alignment_data: AlignmentData) -> torch.Tensor:
    """Get a tensor of shape (n, l) from an AlignmentData object."""
    aligned_sequences = get_aligned_sequences(alignment_data)
    msa = []
    gap_idx = AMINO_ACID_MAP["-"]
    for seq in aligned_sequences:
        seq = [AMINO_ACID_MAP.get(aa, gap_idx) for aa in seq]
        msa.append(seq)
    return torch.tensor(msa, dtype=torch.long)


def filter_alignment(
    alignment_data: AlignmentData,
    min_pid: float = None,
    min_coverage: float = None,
    downsample_num: int = None,
    random_seed: int = 42,
) -> AlignmentData:
    """
    Filter alignment data based on sequence identity, coverage, and optionally downsample.

    Args:
        alignment_data (AlignmentData): AlignmentData object with attributes labels, gap_masks, sequences, mappings.
        min_pid (float): Minimum percentage identity to reference sequence (0.0-1.0). None to skip filter.
        min_coverage (float): Minimum coverage of reference sequence (0.0-1.0). None to skip filter.
        downsample_num (int): Maximum number of sequences to keep (always includes first). None to skip downsampling.
        random_seed (int): Seed for random downsampling

    Returns:
        AlignmentData: Filtered alignment data.
    """
    msa = get_msa_tensor(alignment_data)
    assert len(msa) == len(alignment_data.labels)
    print(f"  > {len(msa)} sequences before filtering")

    # Calculate percentage identity to reference sequence (first sequence)
    reference_seq = msa[0]
    gap_idx = AMINO_ACID_MAP["-"]

    # Initialize keep condition as all True
    keep_condition = torch.ones(len(msa), dtype=torch.bool)

    # Apply PID filter if specified
    if min_pid is not None:
        # For each sequence, calculate identity
        matches = msa == reference_seq.unsqueeze(0)
        pid_scores = matches.float().mean(dim=1)

        pid_condition = pid_scores >= min_pid
        keep_condition = keep_condition & pid_condition
        print(
            f"  > After PID filter (>={min_pid:.2f}): {pid_condition.sum()} sequences"
        )

    # Apply coverage filter if specified
    if min_coverage is not None:
        # Calculate coverage: percentage of reference positions that are not gaps in target
        target_non_gap_mask = msa != gap_idx
        coverage_scores = target_non_gap_mask.float().mean(dim=1)

        coverage_condition = coverage_scores >= min_coverage
        keep_condition = keep_condition & coverage_condition
        print(
            f"  > After coverage filter (>={min_coverage:.2f}): {coverage_condition.sum()} sequences"
        )

    # Always keep the first sequence (reference)
    keep_condition[0] = True

    # Get indices of sequences to keep
    keep_indices = torch.where(keep_condition)[0].tolist()

    print(f"  > After all filters: {len(keep_indices)} sequences")

    # Random downsampling if requested
    if downsample_num is not None and len(keep_indices) > downsample_num:
        np.random.seed(random_seed)

        # Always keep first sequence (reference)
        reference_idx = 0
        other_indices = [idx for idx in keep_indices if idx != reference_idx]

        # Randomly sample from the rest
        n_to_sample = min(downsample_num - 1, len(other_indices))
        sampled_other = np.random.choice(
            other_indices, size=n_to_sample, replace=False
        ).tolist()

        keep_indices = [reference_idx] + sampled_other
        print(f"  > After random downsampling: {len(keep_indices)} sequences")

    # Filter the data
    filtered_labels = [alignment_data.labels[i] for i in keep_indices]
    filtered_gap_masks = [alignment_data.gap_masks[i] for i in keep_indices]
    filtered_sequences = [alignment_data.sequences[i] for i in keep_indices]
    filtered_mappings = [alignment_data.mappings[i] for i in keep_indices]

    return AlignmentData(
        labels=filtered_labels,
        gap_masks=filtered_gap_masks,
        sequences=filtered_sequences,
        mappings=filtered_mappings,
    )


def align_variant(variant, mapping):
    """
    Convert a mutant string from the query sequence to target sequence coordinates (one-based).

    Args:
        mutant (str): Mutation string, e.g., "A123B"
        mapping (List[int or None]): Mapping from query to target coordinates (0-indexed).

    Returns:
        str or None: Aligned mutant string if mapping is possible, otherwise None.
    """
    actual_mutants = []
    for mutation in variant.split(":"):
        try:
            from_AA, position, to_AA = (
                mutation[0],
                int(mutation[1:-1]),
                mutation[-1],
            )
        except Exception as e:
            print("Issue with mutant:", mutation, e)
            continue

        # Map the position from query (1-based) to target (1-based) using mapping.
        mapped_position = mapping[position - 1]  # mapping is 0-indexed
        if mapped_position is not None:
            actual_mutants.append(from_AA + str(mapped_position + 1) + to_AA)
    if len(actual_mutants) == 0:
        return None
    else:
        return ":".join(actual_mutants)
