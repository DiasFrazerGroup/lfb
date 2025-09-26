from typing import Dict, Iterator, List
from lfb.constants import AMINO_ACID_TOKENS
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from lfb.model.progen.modeling_progen import ProGenForCausalLM
from lfb.model.progen.tokenizer import tokenizer as progen_tokenizer


class Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def infer(self, inputs: Iterator[Dict]) -> List[Dict]:
        raise NotImplementedError("This method should be implemented by the subclass")

    def sort_inputs(self, inputs: Iterator[Dict]) -> List[Dict]:
        sorted_inputs = sorted(inputs, key=lambda x: len(x["sequence"]))
        return sorted_inputs


class ESM(Model):
    def __init__(self, model_name: str, device: str = "cuda", fp16: bool = False):
        super().__init__(model_name)
        self.model, self.tokenizer = self._get_model_and_tokenizer(
            model_name, device=device, fp16=fp16
        )
        self.device = device
        self.mask_token = "<mask>"
        self.token_indices = self._get_token_indices()

    def _get_model_and_tokenizer(
        self, model_name: str, device: str = "cuda", fp16: bool = False
    ):
        if fp16:
            model = AutoModelForMaskedLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            )
        else:
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.to(device)
        return model, tokenizer

    def _get_token_indices(self):
        token_indices = []
        for aa in AMINO_ACID_TOKENS:
            index = self.tokenizer.convert_tokens_to_ids(aa)
            token_indices.append(index)
        return token_indices

    def infer(self, inputs: Iterator[Dict], batch_size: int = 32) -> Iterator[Dict]:
        # Convert the generator to a list, keeping track of the original order.
        input_items = self.sort_inputs(inputs)

        # Loop over the sorted items in batches.
        for i in tqdm(
            range(0, len(input_items), batch_size),
            desc="ESM Inference",
            total=max((len(input_items) + batch_size - 1) // batch_size, 1),
        ):
            batch_dicts = input_items[i : i + batch_size]

            sequences = [d["sequence"] for d in batch_dicts]
            aligned_positions = [d["aligned_positions"] for d in batch_dicts]
            starts = [d["start"] for d in batch_dicts]
            ends = [d["end"] for d in batch_dicts]

            chunk_input_ids, chunk_attention_masks = [], []
            for seq, s, e in zip(sequences, starts, ends):
                subseq_ids = self.tokenizer.encode(
                    seq[s:e],
                    add_special_tokens=False,
                )

                if s == 0:
                    # prepend <cls> when at sequence start
                    subseq_ids = [self.tokenizer.cls_token_id] + subseq_ids
                if e == len(seq):
                    # append <eos> when at sequence end
                    subseq_ids = subseq_ids + [self.tokenizer.eos_token_id]

                chunk_input_ids.append(subseq_ids)
                chunk_attention_masks.append([1] * len(subseq_ids))

            batch_encoding = self.tokenizer.pad(
                {"input_ids": chunk_input_ids, "attention_mask": chunk_attention_masks},
                padding=True,
                return_tensors="pt",
            )
            tokenized_inputs = {k: v.to(self.device) for k, v in batch_encoding.items()}

            with torch.no_grad():
                outputs = self.model(**tokenized_inputs)
                logits = outputs["logits"]

            for j, input_dict in enumerate(batch_dicts):
                # Offset of 1 only when a <cls> token was prepended
                shift = 1 if starts[j] == 0 else 0
                pos = [(x - starts[j] - 1) + shift for x in aligned_positions[j]]

                token_logits = logits[j, pos, :]
                marginals = torch.log_softmax(token_logits, dim=-1)
                input_dict["marginals"] = marginals[..., self.token_indices].cpu()

                yield input_dict


class ProGen(Model):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        fp16: bool = False,
    ):
        super().__init__(model_name)
        self.device = device
        self.fp16 = fp16

        self.tokenizer = progen_tokenizer
        self.model = self._get_model(model_name, device, fp16)
        self.model.eval()
        self.token_indices = self._get_token_indices()

        self.first_token, self.last_token = 5, 29

    def _get_model(self, model_name: str, device: str = "cuda", fp16: bool = False):
        if self.fp16:
            model = ProGenForCausalLM.from_pretrained(
                model_name,
                revision="float16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(device)
            model = model.half()
        else:
            model = ProGenForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=True
            ).to(device)
        return model

    def _get_token_indices(self):
        token_indices = []
        for aa in AMINO_ACID_TOKENS:
            index = self.tokenizer.convert_tokens_to_ids(aa)
            token_indices.append(index)
        return token_indices

    def infer(
        self,
        inputs: Iterator[Dict],
        average_reverse: bool = True,
        reduction: str = "mean",
        **kwargs,
    ) -> Iterator[Dict]:
        """
        Simple autoregressive inference for ProGen model.

        Parameters
        ----------
        inputs : iterator of dictionaries produced by `ar_input_generator`
        average_reverse : if True, average forward and reverse direction log-likelihoods
        reduction : 'mean' (per-residue) or 'sum' (total log-prob)

        Returns
        -------
        Iterator of dictionaries with 'log_likelihood' added
        """
        input_items = self.sort_inputs(inputs)

        for input_dict in tqdm(input_items, desc="ProGen Inference"):
            sequence = input_dict["sequence"]
            start = input_dict["start"]
            end = input_dict["end"]

            # Extract the subsequence and add special tokens
            subseq = sequence[start:end]
            token_ids = self.tokenizer.encode(subseq, add_special_tokens=False)

            # Add special tokens based on position in full sequence
            if start == 0:
                token_ids = [self.tokenizer.cls_token_id] + token_ids
            if end == len(sequence):
                token_ids = token_ids + [self.tokenizer.eos_token_id]

            # Convert to tensor
            input_ids = torch.tensor([token_ids], device=self.device)

            log_likelihoods = []
            directions = ["forward"] + (["reverse"] if average_reverse else [])

            for direction in directions:
                log_likelihood = self._compute_log_likelihood(input_ids, direction)
                log_likelihoods.append(log_likelihood)

            # Average across directions if requested
            final_log_likelihood = sum(log_likelihoods) / len(log_likelihoods)

            # Apply reduction: normalize by sequence length for "mean"
            if reduction == "mean":
                # Count valid amino acid tokens (exclude special tokens)
                valid_token_count = self._count_valid_tokens(input_ids)
                final_log_likelihood = final_log_likelihood / max(valid_token_count, 1)

            # Return result
            output_dict = input_dict.copy()
            output_dict["log_likelihood"] = final_log_likelihood

            yield output_dict

    def _compute_log_likelihood(self, input_ids: torch.Tensor, direction: str) -> float:
        """
        Compute log-likelihood for a single sequence in given direction.

        Parameters
        ----------
        input_ids : tensor of shape (1, seq_len)
        direction : "forward" or "reverse"

        Returns
        -------
        Log-likelihood as float
        """
        if direction == "reverse":
            # Simple reverse: flip the sequence
            input_ids = input_ids.flip(dims=[1])

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=self.fp16):
            # Flatten input_ids to 1D for processing
            target = input_ids.squeeze(0)

            # Get model logits
            logits = self.model(target, labels=target).logits

            # Shift logits and targets
            logits = logits[:-1, ...]
            target = target[1:]

            # Remove terminal tokens (BOS=3, EOS=4) if present
            bos_token, eos_token = 3, 4
            if len(target) > 0 and target[-1] in [bos_token, eos_token]:
                logits = logits[:-1, ...]
                target = target[:-1]

            # Assert no BOS/EOS tokens remain
            assert (target == bos_token).sum() == 0
            assert (target == eos_token).sum() == 0

            # Filter to amino acid tokens only
            first_token, last_token = 5, 29
            logits = logits[:, first_token : (last_token + 1)]
            target = target - first_token

            # Verify dimensions
            assert logits.shape[1] == (last_token - first_token + 1)

            # Additional safety: ensure all targets are in valid range
            valid_target_mask = (target >= 0) & (target < logits.shape[1])
            if valid_target_mask.sum() < len(target):
                logits = logits[valid_target_mask]
                target = target[valid_target_mask]

            # If no valid tokens remain, return 0
            if len(target) == 0:
                return 0.0

            # This ensures correct tensor shapes for cross_entropy
            log_likelihood = -F.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1), reduction="sum"
            )

            return log_likelihood.item()

    def _count_valid_tokens(self, input_ids: torch.Tensor) -> int:
        """
        Count valid amino acid tokens (excluding special tokens).

        Parameters
        ----------
        input_ids : tensor of shape (1, seq_len)

        Returns
        -------
        Count of valid amino acid tokens
        """
        target = input_ids.squeeze(0)

        # Remove first token (for shifting)
        target = target[1:]

        # Remove terminal tokens if present
        bos_token, eos_token = 3, 4
        if len(target) > 0 and target[-1] in [bos_token, eos_token]:
            target = target[:-1]

        # Filter out any remaining BOS/EOS tokens
        valid_positions = (target != bos_token) & (target != eos_token)
        target = target[valid_positions]

        # Count amino acid tokens (between first_token and last_token)
        first_token, last_token = 5, 29
        valid_count = ((target >= first_token) & (target <= last_token)).sum().item()

        return valid_count
