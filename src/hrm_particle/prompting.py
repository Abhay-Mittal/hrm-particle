"""Official HRM-Text prompt formatting and PrefixLM batch construction.

The checkpoint is not a chat model.  It expects condition tokens inside an
``<|im_start|>...<|im_end|>`` prefix and ``token_type_ids == 1`` on that
entire prefix.  Fixed response-prefix tokens (``"\nSolution:\n"`` by
default) are causal tokens and are the first positions that receive a particle
intervention.  This lets the particle affect the first *sampled* token without
modifying the clean bidirectional prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


CONDITION_TOKENS = {
    "direct": "<|object_ref_start|>",
    "cot": "<|object_ref_end|>",
    "noisy": "<|quad_start|>",
    "synth": "<|quad_end|>",
}


def format_hrm_prompt(question: str, condition: str = "synth,cot") -> str:
    """Format one raw problem exactly as recommended by the model card."""

    question = str(question).strip()
    if not question:
        raise ValueError("question must be non-empty")
    tags = [part.strip() for part in condition.split(",") if part.strip()]
    if not tags:
        raise ValueError("condition must contain at least one tag")
    unknown = [tag for tag in tags if tag not in CONDITION_TOKENS]
    if unknown:
        raise ValueError(f"unknown HRM condition tag(s): {unknown}")
    prefix = "".join(CONDITION_TOKENS[tag] for tag in tags)
    return f"<|im_start|>{prefix}{question}<|im_end|>"


def _encode(tokenizer: Any, text: str) -> list[int]:
    values = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(values, Tensor):
        values = values.tolist()
    result = [int(value) for value in values]
    if not result:
        raise ValueError(f"tokenizer produced no tokens for {text!r}")
    return result


@dataclass(frozen=True)
class PrefixLMBatch:
    """A left-padded prompt plus a fixed causal response prefix."""

    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
    prompt_mask: Tensor
    particle_mask: Tensor
    position_ids: Tensor
    prompt_lengths: Tensor
    response_prefix_length: int

    def repeat_interleave(self, repeats: int) -> "PrefixLMBatch":
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        fields = {
            name: getattr(self, name).repeat_interleave(repeats, dim=0)
            for name in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "prompt_mask",
                "particle_mask",
                "position_ids",
                "prompt_lengths",
            )
        }
        return PrefixLMBatch(
            **fields,
            response_prefix_length=self.response_prefix_length,
        )


def build_prefixlm_batch(
    tokenizer: Any,
    questions: Sequence[str],
    *,
    condition: str = "synth,cot",
    response_prefix: str = "\nSolution:\n",
    device: torch.device | str | None = None,
) -> PrefixLMBatch:
    """Tokenize raw questions with correct PrefixLM and particle masks.

    Prompts are left padded so the fixed response prefix begins at one common
    tensor index.  Explicit position IDs remove the positional shift that left
    padding would otherwise introduce.
    """

    if not questions:
        raise ValueError("questions must be non-empty")
    prompt_ids = [_encode(tokenizer, format_hrm_prompt(question, condition)) for question in questions]
    response_ids = _encode(tokenizer, response_prefix)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")

    batch_size = len(prompt_ids)
    max_prompt = max(len(ids) for ids in prompt_ids)
    total_length = max_prompt + len(response_ids)
    input_ids = torch.full((batch_size, total_length), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    token_type_ids = torch.zeros_like(input_ids)
    prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    particle_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    position_ids = torch.zeros_like(input_ids)

    for row, ids in enumerate(prompt_ids):
        start = max_prompt - len(ids)
        stop = max_prompt
        prompt_tensor = torch.tensor(ids, dtype=torch.long, device=input_ids.device)
        response_tensor = torch.tensor(response_ids, dtype=torch.long, device=input_ids.device)
        input_ids[row, start:stop] = prompt_tensor
        input_ids[row, stop:] = response_tensor
        attention_mask[row, start:] = True
        token_type_ids[row, start:stop] = 1
        prompt_mask[row, start:stop] = True
        particle_mask[row, stop:] = True
        position_ids[row, start:] = torch.arange(
            len(ids) + len(response_ids), dtype=torch.long, device=input_ids.device
        )

    return PrefixLMBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        prompt_mask=prompt_mask,
        particle_mask=particle_mask,
        position_ids=position_ids,
        prompt_lengths=torch.tensor([len(ids) for ids in prompt_ids], device=input_ids.device),
        response_prefix_length=len(response_ids),
    )


__all__ = [
    "CONDITION_TOKENS",
    "PrefixLMBatch",
    "build_prefixlm_batch",
    "format_hrm_prompt",
]

