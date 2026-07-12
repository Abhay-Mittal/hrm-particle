from __future__ import annotations

import torch

from hrm_particle.prompting import build_prefixlm_batch, format_hrm_prompt


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [2 + ord(character) for character in text]


def test_official_condition_order_and_envelope():
    assert format_hrm_prompt("2+2?") == (
        "<|im_start|><|quad_end|><|object_ref_end|>2+2?<|im_end|>"
    )


def test_prefixlm_and_particle_masks_do_not_overlap():
    batch = build_prefixlm_batch(CharacterTokenizer(), ["short", "a considerably longer prompt"])
    assert batch.input_ids.shape[0] == 2
    assert torch.equal(batch.prompt_mask, batch.token_type_ids.bool())
    assert not bool((batch.prompt_mask & batch.particle_mask).any())
    assert torch.equal(batch.attention_mask, batch.prompt_mask | batch.particle_mask)
    # The shorter left-padded prompt still starts at RoPE position zero.
    first_nonpad = batch.attention_mask[0].nonzero()[0, 0]
    assert batch.position_ids[0, first_nonpad].item() == 0
    assert batch.particle_mask[:, -batch.response_prefix_length :].all()


def test_repeat_interleave_keeps_prompt_particle_grouping():
    batch = build_prefixlm_batch(CharacterTokenizer(), ["one", "two"])
    repeated = batch.repeat_interleave(4)
    assert repeated.input_ids.shape[0] == 8
    for row in range(4):
        assert torch.equal(repeated.input_ids[row], batch.input_ids[0])
    for row in range(4, 8):
        assert torch.equal(repeated.input_ids[row], batch.input_ids[1])

