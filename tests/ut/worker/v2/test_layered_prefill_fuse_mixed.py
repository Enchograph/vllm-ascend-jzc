"""Unit tests for fused mixed-batch frontier concat / split."""

import numpy as np
import torch

from vllm.v1.core.layered_prefill import LayeredPrefillStateStore
from vllm_ascend.worker.v2.layered_prefill import (
    concat_req_frontiers,
    pad_activation_rows,
    slice_req_activations,
    store_req_frontiers,
)


def test_store_and_concat_req_frontiers_roundtrip():
    store = LayeredPrefillStateStore()
    hidden = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    residual = hidden + 0.5
    query_start_loc = np.array([0, 1, 6], dtype=np.int32)
    req_ids = ["d0", "p0"]
    store_req_frontiers(
        store, req_ids, query_start_loc, hidden, residual, next_group_id=1
    )
    d_front = store.get("d0")
    p_front = store.get("p0")
    assert d_front is not None and p_front is not None
    assert d_front.query_len == 1
    assert p_front.query_len == 5
    assert d_front.group_id == 1
    restored, restored_res = concat_req_frontiers(
        store, req_ids, expected_group_id=1, num_tokens_padded=8
    )
    assert restored.shape[0] == 8
    assert torch.equal(restored[:6], hidden)
    assert restored_res is not None
    assert torch.equal(restored_res[:6], residual)
    assert torch.equal(restored[6:], torch.zeros(2, 4))


def test_slice_req_activations_keeps_decode_rows():
    hidden = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)
    residual = hidden + 1
    query_start_loc = np.array([0, 1, 6], dtype=np.int32)
    d_h, d_r = slice_req_activations(
        ["d0", "p0"],
        ["d0"],
        query_start_loc,
        hidden,
        residual,
    )
    assert d_h.shape == (1, 3)
    assert torch.equal(d_h, hidden[:1])
    assert d_r is not None
    assert torch.equal(d_r, residual[:1])
    padded, padded_r = pad_activation_rows(d_h, d_r, 4)
    assert padded.shape[0] == 4
    assert torch.equal(padded[:1], d_h)
    assert padded_r is not None
    assert torch.equal(padded[1:], torch.zeros(3, 3))


def test_store_keep_ids_uses_full_batch_loc():
    store = LayeredPrefillStateStore()
    hidden = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    query_start_loc = np.array([0, 1, 6], dtype=np.int32)
    store_req_frontiers(
        store,
        ["p0"],
        query_start_loc,
        hidden,
        None,
        next_group_id=2,
        batch_req_ids=["d0", "p0"],
        keep_ids=["p0"],
    )
    p_front = store.get("p0")
    assert p_front is not None
    assert p_front.query_len == 5
    assert torch.equal(p_front.hidden_states, hidden[1:6])
    assert store.get("d0") is None


def test_store_drops_trailing_pad_and_repads_after_rider_cancel():
    """Padding is a property of this physical batch, not of the last request.

    A Decode rider that leaves before the next group must not leave its
    old pad rows on the surviving prefill, and the prefill must not keep
    pad rows that belonged to the wider mixed batch.
    """
    store = LayeredPrefillStateStore()
    # d0: 1 logical row, p0: 5 logical rows, 2 pad rows at the tail.
    hidden = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    residual = hidden + 0.25
    query_start_loc = np.array([0, 1, 6], dtype=np.int32)
    store_req_frontiers(
        store,
        ["d0", "p0"],
        query_start_loc,
        hidden,
        residual,
        next_group_id=1,
    )
    prefill = store.get("p0")
    rider = store.get("d0")
    assert prefill is not None and rider is not None
    assert prefill.query_len == 5
    assert rider.query_len == 1
    assert prefill.hidden_states.shape == (5, 4)
    assert torch.equal(prefill.hidden_states, hidden[1:6])
    assert torch.equal(prefill.residual, residual[1:6])

    # Rider cancelled. The next group is prefill-only and pads to its own width.
    restored, restored_res = concat_req_frontiers(
        store, ["p0"], expected_group_id=1, num_tokens_padded=8
    )
    assert restored.shape == (8, 4)
    assert torch.equal(restored[:5], hidden[1:6])
    assert torch.equal(restored[5:], torch.zeros(3, 4))
    assert restored_res is not None
    assert torch.equal(restored_res[:5], residual[1:6])
    assert torch.equal(restored_res[5:], torch.zeros(3, 4))


def test_pad_activation_rows_pads_token_dim_for_rank3():
    hidden = torch.ones(5, 4, 3)
    padded, residual = pad_activation_rows(hidden, None, 8)
    assert residual is None
    assert padded.shape == (8, 4, 3)
    assert torch.equal(padded[:5], hidden)
    assert torch.equal(padded[5:], torch.zeros(3, 4, 3))


def test_store_replaces_previous_frontier_for_same_req():
    store = LayeredPrefillStateStore()
    first = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    second = first + 10
    loc = np.array([0, 4], dtype=np.int32)
    store_req_frontiers(store, ["p0"], loc, first, None, next_group_id=1)
    store_req_frontiers(store, ["p0"], loc, second, None, next_group_id=2)
    second_front = store.get("p0")
    assert second_front is not None
    assert second_front.group_id == 2
    assert torch.equal(second_front.hidden_states, second)
    assert len(store.by_req_id) == 1
    assert second_front.hidden_states.data_ptr() == second.data_ptr()


def test_store_single_request_no_pad_takes_storage():
    store = LayeredPrefillStateStore()
    hidden = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2).contiguous()
    loc = np.array([0, 5], dtype=np.int32)
    store_req_frontiers(store, ["p0"], loc, hidden, None, next_group_id=1)
    front = store.get("p0")
    assert front is not None
    assert front.query_len == 5
    assert torch.equal(front.hidden_states, hidden)
    assert front.hidden_states.data_ptr() == hidden.data_ptr()


def test_store_single_request_with_pad_copies_logical_rows_only():
    store = LayeredPrefillStateStore()
    hidden = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
    loc = np.array([0, 5], dtype=np.int32)
    store_req_frontiers(store, ["p0"], loc, hidden, None, next_group_id=1)
    front = store.get("p0")
    assert front is not None
    assert front.query_len == 5
    assert front.hidden_states.shape == (5, 2)
    assert torch.equal(front.hidden_states, hidden[:5])
    assert front.hidden_states.data_ptr() != hidden.data_ptr()


def test_store_multi_request_copies_slices():
    store = LayeredPrefillStateStore()
    hidden = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    loc = np.array([0, 1, 6], dtype=np.int32)
    store_req_frontiers(store, ["d0", "p0"], loc, hidden, None, next_group_id=1)
    d_front = store.get("d0")
    p_front = store.get("p0")
    assert d_front is not None and p_front is not None
    assert torch.equal(d_front.hidden_states, hidden[:1])
    assert torch.equal(p_front.hidden_states, hidden[1:6])
    assert d_front.hidden_states.data_ptr() != hidden.data_ptr()
    assert p_front.hidden_states.data_ptr() != hidden.data_ptr()


def test_slice_req_activations_single_keep_matches_clone_path():
    hidden = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)
    residual = hidden + 1
    query_start_loc = np.array([0, 1, 6], dtype=np.int32)
    d_h, d_r = slice_req_activations(
        ["d0", "p0"],
        ["d0"],
        query_start_loc,
        hidden,
        residual,
    )
    assert torch.equal(d_h, hidden[:1].clone())
    assert d_r is not None
    assert torch.equal(d_r, residual[:1].clone())
