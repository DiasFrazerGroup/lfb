from typing import Dict, Iterator, List, Tuple
import polars as pl
from polars import DataFrame
from lfb.alignment import AlignmentData, align_variant
from lfb.constants import AMINO_ACID_MAP
from lfb.model.model import Model


def apply_variant(
    sequence: str,
    variant: str,
) -> Tuple[str, str, int]:
    """
    Helper function that mutates an input sequence (sequence) via an input variant (variant).
    """
    alt_sequence = list(sequence)
    ref_sequence = list(sequence)

    positions = []
    for mutation in variant.split(":"):
        try:
            from_AA, position, to_AA = mutation[0], int(mutation[1:-1]), mutation[-1]
        except:
            print("Issue with variant: " + str(variant))

        alt_sequence[position - 1] = to_AA
        ref_sequence[position - 1] = from_AA

        positions.append(position)
    center_position = int(sum(positions) / len(positions))

    return "".join(alt_sequence), "".join(ref_sequence), center_position


def get_chunk(sequence: str, center_position: int, chunk_size: int) -> str:
    """
    Get the start and end indices of a chunk of a sequence around a center position.
    """
    seq_len = len(sequence)
    desired_length = chunk_size

    # If the sequence is too short, return the entire sequence.
    if seq_len <= desired_length:
        return 0, seq_len

    # Compute initial window boundaries.
    start = center_position - (chunk_size // 2)
    end = center_position + (chunk_size // 2)

    # Adjust if the chunk extends before the start of the sequence.
    if start < 0:
        start = 0
        end = desired_length
    # Adjust if the chunk extends past the end of the sequence.
    elif end > seq_len:
        end = seq_len
        start = seq_len - desired_length

    return start, end


class Scorer:
    def __init__(self, model: Model, scoring_mode: str, max_context_length: int):
        self.model = model
        self.scoring_mode = scoring_mode
        self.max_context_length = max_context_length

    def ar_input_generator(
        self, alignment: AlignmentData, variants: List[str]
    ) -> Iterator[Dict]:
        for label, sequence, mapping in zip(
            alignment.labels, alignment.sequences, alignment.mappings
        ):
            # For each original variant in the input, align its coordinates and apply it to the sequence
            for orig_variant in variants:
                aligned_variant = align_variant(orig_variant, mapping)

                if aligned_variant is None:
                    continue

                # apply variant to sequence
                alt_seq, ref_seq, center_position = apply_variant(
                    sequence, aligned_variant
                )

                # Get chunk if sequence is too long, -2 for <cls> and <eos> tokens
                alt_start, alt_end = get_chunk(
                    alt_seq, center_position, chunk_size=self.max_context_length - 2
                )
                ref_start, ref_end = get_chunk(
                    ref_seq, center_position, chunk_size=self.max_context_length - 2
                )

                yield {
                    "sequence": alt_seq,
                    "allele": "alt",
                    "variant": orig_variant,
                    "aligned_variant": aligned_variant,
                    "label": label,
                    "start": alt_start,
                    "end": alt_end,
                }

                yield {
                    "sequence": ref_seq,
                    "allele": "ref",
                    "variant": orig_variant,
                    "aligned_variant": aligned_variant,
                    "label": label,
                    "start": ref_start,
                    "end": ref_end,
                }

    def mlm_input_generator(
        self, alignment: AlignmentData, variants: List[str]
    ) -> Iterator[Dict]:
        """
        Generate inputs for masked-language-model (MLM) scoring.

        The function slides a window of length `self.max_context_length`
        (with 50 % overlap) across each sequence in the alignment.  Every
        window that contains at least one of the variant positions is emitted.
        """
        # -2 for <cls> and <eos> tokens
        window_size = self.max_context_length - 2
        # Overlap windows by half their length.
        step_size = max(window_size // 2, 1)

        for label, sequence, mapping in zip(
            alignment.labels, alignment.sequences, alignment.mappings
        ):

            orig_positions = []
            aligned_positions = []
            for var in variants:
                for mut in var.split(":"):
                    pos = int(mut[1:-1])

                    if mapping[pos - 1] is None:
                        continue
                    else:
                        aln_pos = mapping[pos - 1] + 1

                    if aln_pos not in aligned_positions:
                        orig_positions.append(pos)
                        aligned_positions.append(aln_pos)

            unassigned_idx = set(range(len(aligned_positions)))

            seq_len = len(sequence)
            for start in range(0, seq_len, step_size):
                end = min(start + window_size, seq_len)
                if start >= end:
                    break

                # Indices of variants that fall into this window
                idxs_in_window = [
                    i for i in unassigned_idx if start <= aligned_positions[i] - 1 < end
                ]
                if not idxs_in_window:
                    continue

                chunk_aligned_positions = [aligned_positions[i] for i in idxs_in_window]
                chunk_orig_positions = [orig_positions[i] for i in idxs_in_window]

                yield {
                    "sequence": sequence,
                    "aligned_positions": chunk_aligned_positions,
                    "positions": chunk_orig_positions,
                    "label": label,
                    "start": start,
                    "end": end,
                }

                # Mark these positions as handled so they won't appear again
                unassigned_idx.difference_update(idxs_in_window)

                # Stop early if every position has been assigned
                if not unassigned_idx:
                    break

    def mask_mlm_inputs(self, inputs: Iterator[Dict]) -> Iterator[Dict]:
        """
        Mask the inputs for the MLM model.
        """
        for input in inputs:
            for position, aligned_position in zip(
                input["positions"], input["aligned_positions"]
            ):

                sequence = list(input["sequence"])

                sequence[aligned_position - 1] = self.model.mask_token

                yield {
                    "sequence": "".join(sequence),
                    "aligned_positions": [aligned_position],
                    "positions": [position],
                    "label": input["label"],
                    "start": input["start"],
                    "end": input["end"],
                }

    def marginals_differences(
        self, outputs: List[Dict], variants: List[str]
    ) -> Iterator[Dict]:
        """
        Extract the marginals from the outputs.
        """
        # Mapping from position and label (sequence) to marginal amino acid distribution
        pos_label_to_marginal = {}
        labels = set()
        for output in outputs:
            label = output["label"]
            labels.add(label)
            for position, marginal in zip(output["positions"], output["marginals"]):
                pos_label_to_marginal[(position, label)] = marginal

        # For each variant, extract the marginals for each position and label
        for label in list(labels):
            for variant in variants:
                sum_marginal_diff = 0
                any_mapped = False
                for mut in variant.split(":"):
                    ref, pos, alt = mut[0], int(mut[1:-1]), mut[-1]

                    marginal = pos_label_to_marginal.get((pos, label), None)
                    if marginal is None:
                        continue
                    else:
                        any_mapped = True

                    alt_aa = AMINO_ACID_MAP.get(alt, None)
                    ref_aa = AMINO_ACID_MAP.get(ref, None)

                    if alt_aa is None or ref_aa is None:
                        print(
                            f"  > Warning: Unrecognized amino acid(s) in mutant description: {mut}. Averaging over amino acid possibilities."
                        )

                    alt_marginal = (
                        marginal[alt_aa] if alt_aa is not None else marginal.mean()
                    )
                    ref_marginal = (
                        marginal[ref_aa] if ref_aa is not None else marginal.mean()
                    )

                    marginal_diff = alt_marginal - ref_marginal
                    sum_marginal_diff += marginal_diff

                if any_mapped:
                    yield {
                        "variant": variant,
                        "label": label,
                        "score": sum_marginal_diff,
                    }

    def log_likelihood_differences(
        self, outputs: List[Dict], variants: List[str]
    ) -> Iterator[Dict]:
        """
        Compute the log-likelihood differences between the masked and unmasked outputs.
        """
        label_variant_allele_to_score = {}
        labels = set()
        for output in outputs:
            label = output["label"]
            labels.add(label)
            variant = output["variant"]
            allele = output["allele"]
            score = output["log_likelihood"]

            label_variant_allele_to_score[(label, variant, allele)] = score

        for label in list(labels):
            for variant in variants:
                alt_score = label_variant_allele_to_score.get(
                    (label, variant, "alt"), None
                )
                ref_score = label_variant_allele_to_score.get(
                    (label, variant, "ref"), None
                )
                if ref_score is None or alt_score is None:
                    continue

                yield {
                    "variant": variant,
                    "label": label,
                    "score": alt_score - ref_score,
                }

    def score_variants(
        self,
        alignment: AlignmentData,
        variants: List[str],
        treat_unmapped_as_zero: bool = False,
        batch_size: int = 8,
    ) -> DataFrame:
        """
        Score a list of mutants using LFB.
        """
        if self.scoring_mode == "autoregressive":
            inputs = self.ar_input_generator(alignment, variants)
        elif (
            self.scoring_mode == "masked-marginals"
            or self.scoring_mode == "unmasked-marginals"
        ):
            inputs = self.mlm_input_generator(alignment, variants)
            if self.scoring_mode == "masked-marginals":
                inputs = self.mask_mlm_inputs(inputs)

        outputs = self.model.infer(inputs, batch_size=batch_size)

        if self.scoring_mode == "autoregressive":
            outputs = self.log_likelihood_differences(outputs, variants)
        if (
            self.scoring_mode == "masked-marginals"
            or self.scoring_mode == "unmasked-marginals"
        ):
            outputs = self.marginals_differences(outputs, variants)

        # All scored variants (for those mapped over successfully)
        scores_df = pl.from_dicts(outputs)

        # Extract the unique labels from the alignment data.
        unique_labels = list(set(alignment.labels))

        # Create a complete grid with every combination of variant and label.
        complete_pairs = [
            {"variant": var, "label": lbl} for var in variants for lbl in unique_labels
        ]
        grid_df = pl.from_dicts(complete_pairs)

        # Join the complete grid with the scores, so any missing (variant, label) pair will have a null score.
        merged_df = grid_df.join(scores_df, on=["variant", "label"], how="left")

        # Check the option to treat unmapped variants as zero.
        if treat_unmapped_as_zero:
            merged_df = merged_df.with_columns(pl.col("score").fill_null(0))

        # Group by variant and compute the mean score.
        lfb_scores = (
            merged_df.group_by("variant")
            .agg(pl.col("score").mean().alias("score"))
            .rename({"variant": "mutant"})
            .sort("mutant")
        )

        return lfb_scores

    def score_variants_without_lfb(
        self, alignment: AlignmentData, variants: List[str], batch_size: int = 8
    ) -> DataFrame:
        """
        Score a list of mutants using LFB.
        """
        alignment = AlignmentData(
            labels=alignment.labels[:1],
            sequences=alignment.sequences[:1],
            mappings=alignment.mappings[:1],
            gap_masks=alignment.gap_masks[:1],
        )
        return self.score_variants(alignment, variants, batch_size=batch_size)
