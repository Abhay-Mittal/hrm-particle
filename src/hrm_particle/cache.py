"""Safe KV-cache branching helpers.

Hugging Face cache objects are mutable: every decoding call appends keys and
values in place.  Sharing one prompt cache between particle branches therefore
silently contaminates branches.  The helpers here make that dangerous action
explicit and verify that cloned caches do not share tensor storage.

The POC intentionally targets detached ``DynamicCache``-style prompt caches.
Static/quantized/offloaded caches may own device resources that cannot be deep
copied and should not be used for branching without an architecture-specific
implementation.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

import torch
from torch import Tensor


CacheT = TypeVar("CacheT")


class CacheCloneError(RuntimeError):
    """Raised when a cache cannot be copied into isolated branch state."""


def iter_cache_tensors(value: Any) -> Iterator[Tensor]:
    """Yield every tensor reachable from a cache-like Python object.

    Traversal is deliberately restricted to containers and ``__dict__``.  It
    covers legacy tuple caches, Transformers ``Cache.layers`` objects, and the
    small fake caches used by tests without depending on a Transformers version.
    """

    seen: set[int] = set()

    def visit(item: Any) -> Iterator[Tensor]:
        identifier = id(item)
        if identifier in seen:
            return
        seen.add(identifier)

        if isinstance(item, Tensor):
            yield item
            return
        if isinstance(item, dict):
            for key, child in item.items():
                yield from visit(key)
                yield from visit(child)
            return
        if isinstance(item, (list, tuple, set, frozenset)):
            for child in item:
                yield from visit(child)
            return
        attributes = getattr(item, "__dict__", None)
        if attributes is not None:
            for child in attributes.values():
                yield from visit(child)

    yield from visit(value)


def _storage_span(tensor: Tensor) -> tuple[str, int, int] | None:
    """Identify a tensor storage, returning ``None`` for empty/meta tensors."""

    if tensor.device.type == "meta" or tensor.numel() == 0:
        return None
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes())


def assert_cache_isolated(*caches: Any) -> None:
    """Raise if any two caches share a non-empty tensor storage."""

    storage_owners: dict[tuple[str, int, int], int] = {}
    for cache_index, cache in enumerate(caches):
        for tensor in iter_cache_tensors(cache):
            span = _storage_span(tensor)
            if span is None:
                continue
            previous = storage_owners.get(span)
            if previous is not None and previous != cache_index:
                raise AssertionError(
                    f"cache {cache_index} shares tensor storage with cache {previous}; "
                    "particle branches must own independent KV state"
                )
            storage_owners[span] = cache_index


def clone_cache(cache: CacheT | None, *, detach: bool = True) -> CacheT | None:
    """Deep-copy a cache and verify tensor-storage isolation.

    ``copy.deepcopy`` is compatible with detached Hugging Face ``DynamicCache``
    objects and legacy tuple caches.  A clear error is raised for cache types
    that hold non-copyable streams or graph-connected non-leaf tensors.
    """

    if cache is None:
        return None
    if detach:
        graph_tensors = [tensor for tensor in iter_cache_tensors(cache) if tensor.requires_grad]
        if graph_tensors:
            raise CacheCloneError(
                "cache contains tensors that require gradients; create/fill the prompt cache under "
                "torch.no_grad() before branching"
            )
    try:
        cloned = copy.deepcopy(cache)
    except Exception as error:  # pragma: no cover - exact exception is cache-type dependent
        raise CacheCloneError(
            f"cannot safely clone cache of type {type(cache).__name__}; use a detached "
            "Transformers DynamicCache or add a cache-specific clone implementation"
        ) from error

    if detach:
        for tensor in iter_cache_tensors(cloned):
            # ``deepcopy`` already cloned storage.  Detaching in place avoids
            # replacing object attributes and works for every tensor container.
            tensor.detach_()
    assert_cache_isolated(cache, cloned)
    return cloned


def branch_cache(
    cache: CacheT | None,
    num_branches: int,
    *,
    detach: bool = True,
) -> list[CacheT | None]:
    """Create ``num_branches`` independent prompt-cache snapshots."""

    if num_branches <= 0:
        raise ValueError(f"num_branches must be positive, got {num_branches}")
    branches = [clone_cache(cache, detach=detach) for _ in range(num_branches)]
    assert_cache_isolated(*[branch for branch in branches if branch is not None])
    return branches


def repeat_cache_batch(cache: CacheT, repeats: int, *, detach: bool = True) -> CacheT:
    """Clone a cache and repeat each batch row for particle-major decoding.

    Transformers cache classes implement ``batch_repeat_interleave`` in place.
    Keeping the source cache untouched makes it reusable for other experimental
    conditions and preserves a clean anchor snapshot.
    """

    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    repeated = clone_cache(cache, detach=detach)
    method = getattr(repeated, "batch_repeat_interleave", None)
    if not callable(method):
        raise CacheCloneError(
            f"cache type {type(cache).__name__} does not implement batch_repeat_interleave"
        )
    method(repeats)
    assert_cache_isolated(cache, repeated)
    return repeated


def select_cache_batch(
    cache: CacheT,
    indices: Tensor | Sequence[int],
    *,
    detach: bool = True,
) -> CacheT:
    """Clone a cache and retain only specified batch rows."""

    selected = clone_cache(cache, detach=detach)
    method = getattr(selected, "batch_select_indices", None)
    if not callable(method):
        raise CacheCloneError(
            f"cache type {type(cache).__name__} does not implement batch_select_indices"
        )
    if not isinstance(indices, Tensor):
        indices = torch.tensor(list(indices), dtype=torch.long)
    else:
        indices = indices.to(dtype=torch.long)
    method(indices)
    assert_cache_isolated(cache, selected)
    return selected


# Explicit name used by rollout code and configs.
branch_past_key_values = branch_cache


__all__ = [
    "CacheCloneError",
    "assert_cache_isolated",
    "branch_cache",
    "branch_past_key_values",
    "clone_cache",
    "iter_cache_tensors",
    "repeat_cache_batch",
    "select_cache_batch",
]
