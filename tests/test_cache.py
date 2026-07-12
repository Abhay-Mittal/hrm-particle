from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from hrm_particle.cache import (
    CacheCloneError,
    assert_cache_isolated,
    branch_cache,
    clone_cache,
    repeat_cache_batch,
    select_cache_batch,
)


@dataclass
class FakeLayer:
    keys: torch.Tensor
    values: torch.Tensor


class FakeDynamicCache:
    def __init__(self, batch_size: int = 2) -> None:
        self.layers = [
            FakeLayer(
                keys=torch.randn(batch_size, 2, 3, 4),
                values=torch.randn(batch_size, 2, 3, 4),
            ),
            FakeLayer(
                keys=torch.randn(batch_size, 2, 3, 4),
                values=torch.randn(batch_size, 2, 3, 4),
            ),
        ]
        self.metadata = {"seen": torch.tensor(3)}

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self.layers:
            layer.keys = layer.keys.repeat_interleave(repeats, dim=0)
            layer.values = layer.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for layer in self.layers:
            device_indices = indices.to(layer.keys.device)
            layer.keys = layer.keys.index_select(0, device_indices)
            layer.values = layer.values.index_select(0, device_indices)


def test_clone_is_equal_but_storage_independent() -> None:
    source = FakeDynamicCache()
    cloned = clone_cache(source)

    assert cloned is not source
    assert torch.equal(cloned.layers[0].keys, source.layers[0].keys)
    assert_cache_isolated(source, cloned)

    cloned.layers[0].keys.add_(100)
    assert not torch.equal(cloned.layers[0].keys, source.layers[0].keys)


def test_branch_mutation_cannot_contaminate_siblings_or_source() -> None:
    source = FakeDynamicCache()
    baseline = source.layers[1].values.clone()
    branches = branch_cache(source, 4)

    branches[2].layers[1].values.zero_()

    assert torch.equal(source.layers[1].values, baseline)
    assert torch.equal(branches[0].layers[1].values, baseline)
    assert torch.equal(branches[1].layers[1].values, baseline)
    assert torch.equal(branches[3].layers[1].values, baseline)
    assert_cache_isolated(source, *branches)


def test_repeat_cache_batch_preserves_source_and_order() -> None:
    source = FakeDynamicCache(batch_size=2)
    original = source.layers[0].keys.clone()

    repeated = repeat_cache_batch(source, 3)

    assert repeated.layers[0].keys.shape[0] == 6
    assert torch.equal(repeated.layers[0].keys[0], original[0])
    assert torch.equal(repeated.layers[0].keys[1], original[0])
    assert torch.equal(repeated.layers[0].keys[2], original[0])
    assert torch.equal(repeated.layers[0].keys[3], original[1])
    assert torch.equal(source.layers[0].keys, original)
    assert_cache_isolated(source, repeated)


def test_select_cache_batch_preserves_source() -> None:
    source = FakeDynamicCache(batch_size=3)
    original = source.layers[0].values.clone()

    selected = select_cache_batch(source, [2, 0])

    assert torch.equal(selected.layers[0].values[0], original[2])
    assert torch.equal(selected.layers[0].values[1], original[0])
    assert torch.equal(source.layers[0].values, original)


def test_legacy_tuple_cache_can_be_branched() -> None:
    legacy = ((torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)),)
    branches = branch_cache(legacy, 2)

    assert torch.equal(branches[0][0][0], legacy[0][0])
    assert_cache_isolated(legacy, *branches)


def test_cache_with_autograd_state_is_rejected() -> None:
    cache = FakeDynamicCache()
    cache.layers[0].keys.requires_grad_(True)

    with pytest.raises(CacheCloneError, match="torch.no_grad"):
        clone_cache(cache)


def test_isolation_assertion_detects_alias() -> None:
    source = FakeDynamicCache()
    alias = FakeDynamicCache()
    alias.layers[0].keys = source.layers[0].keys

    with pytest.raises(AssertionError, match="shares tensor storage"):
        assert_cache_isolated(source, alias)
