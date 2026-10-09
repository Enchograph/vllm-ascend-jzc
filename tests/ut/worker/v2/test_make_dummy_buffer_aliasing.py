"""Regression: ``AscendInputBatch.make_dummy`` must alias the persistent buffers.

``InputBatch.make_dummy`` returns slices of the ``InputBuffers`` that live for
the whole process. aclgraph capture runs through this dummy batch, so the
address baked into the graph is whatever ``make_dummy`` hands back. On replay
the runtime writes the real lengths into those same persistent buffers and does
not re-run Python. The two only line up if the dummy batch *aliases* the
buffers rather than copying them.

``AscendInputBatch.make_dummy`` used to rebuild the batch with
``cls(**asdict(input_batch), ...)``. ``dataclasses.asdict`` deep-copies every
leaf field and ``torch.Tensor.__deepcopy__`` allocates new storage, so the
captured graph pointed at a throwaway allocation that nothing updated again.
DSA reads ``seq_lens`` as ``seqused_kv`` and does not rebind graph params, so it
derived KV addresses from stale lengths -- silently wrong output on every
full-graph decode, and an out-of-range MTE fault at ``max_model_len=10240``.
"""

import torch

from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers

# The aliasing only matters for tensors the captured graph reads. These are the
# ones that reach attention metadata via prepare_attn.
_ALIASED_ATTRS = ("seq_lens", "query_start_loc", "positions", "input_ids")

MAX_NUM_REQS = 8
MAX_NUM_TOKENS = 64


def _buffers() -> AscendInputBuffers:
    return AscendInputBuffers(
        max_num_reqs=MAX_NUM_REQS,
        max_num_tokens=MAX_NUM_TOKENS,
        device=torch.device("cpu"),
    )


def test_make_dummy_tensors_alias_persistent_buffers():
    """Every graph-visible tensor must sit inside its persistent buffer."""
    buffers = _buffers()
    batch = AscendInputBatch.make_dummy(4, 16, buffers)

    for name in _ALIASED_ATTRS:
        got = getattr(batch, name)
        want = getattr(buffers, name)
        assert isinstance(got, torch.Tensor), name
        assert got.data_ptr() == want.data_ptr(), (
            f"{name} was detached from its persistent buffer "
            f"(0x{got.data_ptr():x} != 0x{want.data_ptr():x}); aclgraph would "
            f"bake the throwaway address"
        )


def test_make_dummy_batch_observes_later_writes_through_buffers():
    """A write through the buffer must be visible in the dummy batch.

    This is the property graph replay depends on: the runtime writes real
    lengths into ``input_buffers.seq_lens`` and the captured graph reads them at
    the baked address. A copy would freeze the dummy values instead.
    """
    buffers = _buffers()
    batch = AscendInputBatch.make_dummy(4, 16, buffers)

    buffers.seq_lens[:4] = 8192
    assert torch.equal(batch.seq_lens, torch.full((4,), 8192, dtype=batch.seq_lens.dtype)), (
        "dummy batch froze a copy of seq_lens instead of aliasing the buffer"
    )

    buffers.positions[:16] = 7
    assert bool((batch.positions[:16] == 7).all()), (
        "dummy batch froze a copy of positions instead of aliasing the buffer"
    )


def test_make_dummy_shares_one_base_address_across_capture_sizes():
    """All captured sizes must bake the same base address.

    aclgraph captures one graph per size against a single set of buffers. If a
    size allocated its own tensor, the per-size graphs would each point at a
    different address and the asymmetry would be invisible until replay.
    """
    buffers = _buffers()
    seq_lens_ptrs = set()
    query_start_loc_ptrs = set()

    for num_reqs in (1, 2, 4, 8):
        batch = AscendInputBatch.make_dummy(num_reqs, num_reqs * 2, buffers)
        seq_lens_ptrs.add(batch.seq_lens.data_ptr())
        query_start_loc_ptrs.add(batch.query_start_loc.data_ptr())

    assert len(seq_lens_ptrs) == 1, f"seq_lens moved across capture sizes: {[hex(p) for p in sorted(seq_lens_ptrs)]}"
    assert len(query_start_loc_ptrs) == 1, (
        f"query_start_loc moved across capture sizes: {[hex(p) for p in sorted(query_start_loc_ptrs)]}"
    )


def test_make_dummy_still_populates_ascend_specific_fields():
    """The rebuild must not drop what AscendInputBatch adds on top."""
    buffers = _buffers()
    batch = AscendInputBatch.make_dummy(4, 16, buffers)

    assert batch.seq_lens_np is not None
    assert batch.seq_lens_np.shape == (4,)
    assert batch.attn_state is not None
    assert batch.num_reqs == 4
    assert batch.num_tokens == 16
    assert len(batch.req_ids) == 4
