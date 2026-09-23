"""Unit tests for Layered Prefill V2 PP packing / group owner / PPHandler merge (V6)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.pp_utils import compute_need_sampled_mask

from vllm_ascend.worker.v2.layered_prefill import (
    LayeredPPHandlerCapture,
    LayeredV2ExecuteModelState,
    compute_layered_need_sampled_mask,
    concat_pp_input_batches,
    snapshot_pp_input_batch,
)
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _pp_batch(
    req_ids: list[str],
    *,
    computed: list[int],
    prefill: list[int],
    scheduled: list[int],
    max_seq: list[int],
    req_indices: list[int],
):
    num_reqs = len(req_ids)
    return SimpleNamespace(
        req_ids=req_ids,
        num_reqs=num_reqs,
        num_computed_tokens_np=np.array(computed, dtype=np.int32),
        prefill_len_np=np.array(prefill, dtype=np.int32),
        max_seq_len_np=np.array(max_seq, dtype=np.int32),
        num_scheduled_tokens=np.array(scheduled, dtype=np.int32),
        idx_mapping=torch.tensor(req_indices, dtype=torch.int32),
        idx_mapping_np=np.array(req_indices, dtype=np.int32),
    )


class _FakePPHandler:
    def __init__(self, *, is_last_rank: bool, max_sample_len: int = 1):
        self.is_last_rank = is_last_rank
        self.max_sample_len = max_sample_len
        self.broadcasts: list = []
        self.receives: list = []

    def broadcast(self, tokens, num_sampled, num_rejected, batch):
        self.broadcasts.append(
            (tokens.clone(), num_sampled.clone(), num_rejected.clone(), batch)
        )

    def receive(self, batch):
        self.receives.append(batch)
        return False


def test_layered_pp_intermediate_round_trip_preserves_d_and_p_rows():
    d = IntermediateTensors(
        {"hidden_states": torch.ones(2, 4), "residual": torch.zeros(2, 4)}
    )
    p = IntermediateTensors(
        {"hidden_states": torch.full((3, 4), 2), "residual": torch.ones(3, 4)}
    )
    packed = NPUModelRunner._combine_layered_pp_intermediate(d, p)
    d_out, p_out = NPUModelRunner._split_layered_pp_intermediate(packed)

    assert d_out is not None and p_out is not None
    assert torch.equal(d_out["hidden_states"], d["hidden_states"])
    assert torch.equal(p_out["hidden_states"], p["hidden_states"])
    assert int(packed["layered_pp_d_rows"].item()) == 2
    assert int(packed["layered_pp_p_rows"].item()) == 3


def test_layered_pp_intermediate_p_only_has_zero_d_rows():
    p = IntermediateTensors({"hidden_states": torch.full((4, 2), 3.0)})
    packed = NPUModelRunner._combine_layered_pp_intermediate(None, p)
    d_out, p_out = NPUModelRunner._split_layered_pp_intermediate(packed)
    assert d_out is not None and p_out is not None
    assert d_out["hidden_states"].shape[0] == 0
    assert torch.equal(p_out["hidden_states"], p["hidden_states"])
    assert int(packed["layered_pp_d_rows"].item()) == 0


def test_layered_pp_intermediate_d_only_has_zero_p_rows():
    d = IntermediateTensors({"hidden_states": torch.ones(2, 2)})
    packed = NPUModelRunner._combine_layered_pp_intermediate(d, None)
    d_out, p_out = NPUModelRunner._split_layered_pp_intermediate(packed)
    assert d_out is not None and p_out is not None
    assert torch.equal(d_out["hidden_states"], d["hidden_states"])
    assert p_out["hidden_states"].shape[0] == 0
    assert int(packed["layered_pp_p_rows"].item()) == 0


def test_combine_rejects_missing_d_and_p():
    try:
        NPUModelRunner._combine_layered_pp_intermediate(None, None)
    except RuntimeError as error:
        assert "requires D or P" in str(error)
    else:
        raise AssertionError("expected empty-pack error")


def test_split_none_is_none():
    assert NPUModelRunner._split_layered_pp_intermediate(None) == (None, None)


def test_split_rejects_missing_metadata():
    tensors = IntermediateTensors({"hidden_states": torch.ones(2, 2)})
    try:
        NPUModelRunner._split_layered_pp_intermediate(tensors)
    except RuntimeError as error:
        assert "missing D/P row metadata" in str(error)
    else:
        raise AssertionError("expected metadata error")


def test_layered_pp_group_owner_picks_containing_rank(monkeypatch):
    partitions = {0: (0, 24), 1: (24, 48)}

    monkeypatch.setattr(
        "vllm_ascend.worker.v2.model_runner.get_pp_indices",
        lambda _n, rank, _world: partitions[rank],
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.model_runner.get_pp_group",
        lambda: SimpleNamespace(world_size=2),
    )
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=48),
        hf_config=None,
    )
    assert (
        NPUModelRunner._layered_pp_group_owner(
            runner, SimpleNamespace(group_start=0, group_end=24)
        )
        == 0
    )
    assert (
        NPUModelRunner._layered_pp_group_owner(
            runner, SimpleNamespace(group_start=24, group_end=48)
        )
        == 1
    )


def test_clone_intermediate_tensors_is_detached():
    src = IntermediateTensors({"hidden_states": torch.arange(6.0).reshape(2, 3)})
    cloned = NPUModelRunner._clone_intermediate_tensors(src)
    src["hidden_states"].fill_(-1)
    assert cloned["hidden_states"].tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_upstream_need_sampled_mask_treats_layered_intermediate_as_sample():
    batch = _pp_batch(
        ["p0"],
        computed=[0],
        prefill=[128],
        scheduled=[128],
        max_seq=[1024],
        req_indices=[3],
    )
    mask = compute_need_sampled_mask(batch)
    assert mask is not None and bool(mask[0])


def test_layered_mask_drops_intermediate_p_keeps_decode():
    batch = concat_pp_input_batches(
        [
            _pp_batch(
                ["d0"],
                computed=[128],
                prefill=[128],
                scheduled=[1],
                max_seq=[1024],
                req_indices=[0],
            ),
            _pp_batch(
                ["p0"],
                computed=[0],
                prefill=[128],
                scheduled=[128],
                max_seq=[1024],
                req_indices=[1],
            ),
        ]
    )
    intermediate = compute_layered_need_sampled_mask(
        batch, sample_p=False, p_req_ids={"p0"}
    )
    assert intermediate is not None
    assert intermediate.tolist() == [True, False]
    final = compute_layered_need_sampled_mask(batch, sample_p=True, p_req_ids={"p0"})
    assert final is not None
    assert final.tolist() == [True, True]


def test_snapshot_pp_input_batch_is_detached():
    batch = _pp_batch(
        ["d0"],
        computed=[4],
        prefill=[4],
        scheduled=[1],
        max_seq=[32],
        req_indices=[7],
    )
    snap = snapshot_pp_input_batch(batch)
    batch.idx_mapping.fill_(-1)
    batch.num_computed_tokens_np[0] = 99
    assert snap.idx_mapping.tolist() == [7]
    assert snap.num_computed_tokens_np.tolist() == [4]


def test_capture_flush_merges_d_and_final_p_broadcast():
    inner = _FakePPHandler(is_last_rank=True, max_sample_len=2)
    capture = LayeredPPHandlerCapture(inner)
    d_batch = _pp_batch(
        ["d0", "d1"],
        computed=[8, 8],
        prefill=[8, 8],
        scheduled=[1, 1],
        max_seq=[64, 64],
        req_indices=[0, 1],
    )
    p_batch = _pp_batch(
        ["p0"],
        computed=[0],
        prefill=[16],
        scheduled=[16],
        max_seq=[64],
        req_indices=[2],
    )
    d_tokens = torch.tensor([[11, -1], [12, -1]], dtype=torch.int64)
    p_tokens = torch.tensor([[21, -1]], dtype=torch.int64)
    capture.broadcast(
        d_tokens,
        torch.ones(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        d_batch,
    )
    capture.broadcast(
        p_tokens,
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        p_batch,
    )
    d_tokens.fill_(0)
    stats = capture.flush(sample_p=True, p_req_ids={"p0"})

    assert len(inner.broadcasts) == 1
    tokens, num_sampled, _rejected, batch = inner.broadcasts[0]
    assert tokens.tolist() == [[11, -1], [12, -1], [21, -1]]
    assert num_sampled.tolist() == [1, 1, 1]
    assert batch.req_ids == ["d0", "d1", "p0"]
    assert batch.idx_mapping_np.tolist() == [0, 1, 2]
    assert stats["n_d"] == 2
    assert stats["n_p"] == 1
    assert stats["dropped_p_rows"] == 0
    assert stats["skipped"] is False


def test_capture_flush_drops_intermediate_p_from_receive():
    inner = _FakePPHandler(is_last_rank=False, max_sample_len=1)
    capture = LayeredPPHandlerCapture(inner)
    d_batch = _pp_batch(
        ["d0"],
        computed=[8],
        prefill=[8],
        scheduled=[1],
        max_seq=[64],
        req_indices=[0],
    )
    p_batch = _pp_batch(
        ["p0"],
        computed=[0],
        prefill=[16],
        scheduled=[16],
        max_seq=[64],
        req_indices=[1],
    )
    assert capture.receive(d_batch) is False
    assert capture.receive(p_batch) is False
    stats = capture.flush(sample_p=False, p_req_ids={"p0"})

    assert len(inner.receives) == 1
    assert inner.receives[0].req_ids == ["d0"]
    assert stats["n_d"] == 1
    assert stats["n_p"] == 0
    assert stats["dropped_p_rows"] == 1
    assert stats["skipped"] is False


def test_sample_layered_v2_merges_d_and_final_p_on_non_last_rank(monkeypatch):
    sample_calls: list[list[str]] = []

    def fake_sample(self, grammar_output):
        batch = self.execute_model_state.input_batch
        sample_calls.append(list(batch.req_ids))
        if self.pp_handler is not None:
            if getattr(self, "is_last_pp_rank", False):
                tokens = torch.arange(batch.num_reqs, dtype=torch.int64).unsqueeze(1)
                self.pp_handler.broadcast(
                    tokens,
                    torch.ones(batch.num_reqs, dtype=torch.int32),
                    torch.zeros(batch.num_reqs, dtype=torch.int32),
                    batch,
                )
            else:
                self.pp_handler.receive(batch)
        return ModelRunnerOutput(
            req_ids=list(batch.req_ids),
            req_id_to_index={req_id: i for i, req_id in enumerate(batch.req_ids)},
            sampled_token_ids=[[] for _ in batch.req_ids],
        )

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner.sample_tokens",
        fake_sample,
    )

    inner = _FakePPHandler(is_last_rank=False, max_sample_len=1)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.pp_handler = inner
    runner.is_last_pp_rank = False
    runner.kv_connector = MagicMock()

    d_batch = _pp_batch(
        ["d0"],
        computed=[8],
        prefill=[8],
        scheduled=[1],
        max_seq=[64],
        req_indices=[0],
    )
    p_batch = _pp_batch(
        ["p0"],
        computed=[0],
        prefill=[16],
        scheduled=[16],
        max_seq=[64],
        req_indices=[1],
    )
    state = LayeredV2ExecuteModelState(
        scheduler_output=SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={"d0": 1, "p0": 16},
            total_num_scheduled_tokens=17,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        ),
        d_state=SimpleNamespace(input_batch=d_batch),
        p_state=SimpleNamespace(input_batch=p_batch, finished_req_ids=set()),
        sample_p=True,
    )

    output = runner._sample_layered_v2(None, state)
    assert sample_calls == [["d0"], ["p0"]]
    assert len(inner.receives) == 1
    assert inner.receives[0].req_ids == ["d0", "p0"]
    assert output.req_ids == ["d0", "p0"]
    assert runner.pp_handler is inner


def test_sample_layered_v2_skips_pp_slot_for_intermediate_p(monkeypatch):
    sample_calls: list[list[str]] = []

    def fake_sample(self, grammar_output):
        batch = self.execute_model_state.input_batch
        sample_calls.append(list(batch.req_ids))
        if self.pp_handler is not None:
            self.pp_handler.receive(batch)
        return ModelRunnerOutput(
            req_ids=list(batch.req_ids),
            req_id_to_index={req_id: i for i, req_id in enumerate(batch.req_ids)},
            sampled_token_ids=[[7] for _ in batch.req_ids],
        )

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner.sample_tokens",
        fake_sample,
    )

    inner = _FakePPHandler(is_last_rank=False, max_sample_len=1)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.pp_handler = inner
    runner.is_last_pp_rank = False
    runner.kv_connector = MagicMock()
    runner.kv_connector.post_forward.return_value = None

    d_batch = _pp_batch(
        ["d0"],
        computed=[8],
        prefill=[8],
        scheduled=[1],
        max_seq=[64],
        req_indices=[0],
    )
    p_batch = _pp_batch(
        ["p0"],
        computed=[0],
        prefill=[16],
        scheduled=[16],
        max_seq=[64],
        req_indices=[1],
    )
    state = LayeredV2ExecuteModelState(
        scheduler_output=SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={"d0": 1, "p0": 16},
            total_num_scheduled_tokens=17,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        ),
        d_state=SimpleNamespace(input_batch=d_batch),
        p_state=SimpleNamespace(input_batch=p_batch, finished_req_ids=set()),
        sample_p=False,
    )

    output = runner._sample_layered_v2(None, state)
    assert sample_calls == [["d0"]]
    assert len(inner.receives) == 1
    assert inner.receives[0].req_ids == ["d0"]
    assert output.sampled_token_ids == [[7], []]


def test_update_pp_decode_requests_skipped_during_layered_d(monkeypatch):
    called = []

    def fake_update(self):
        called.append(True)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner.update_pp_decode_requests",
        fake_update,
    )
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner._layered_skip_pp_decode_update = False
    runner.update_pp_decode_requests()
    runner._layered_skip_pp_decode_update = True
    runner.update_pp_decode_requests()
    assert called == [True]


def _pp4_activation_runner(monkeypatch, *, rank: int, leftover=None):
    partitions = {0: (0, 12), 1: (12, 24), 2: (24, 36), 3: (36, 48)}
    dummy_h = torch.full((3, 4), 7.0)
    dummy_r = torch.full((3, 4), 8.0)
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.model_runner.get_pp_indices",
        lambda _n, r, _w: partitions[r],
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.model_runner.get_pp_group",
        lambda: SimpleNamespace(rank_in_group=rank, world_size=4),
    )
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=48),
        hf_config=None,
        dtype=torch.float32,
    )
    runner.device = torch.device("cpu")
    runner.layered_prefill_model_adapter = SimpleNamespace(
        make_transport_frontier=lambda n, d, dev: (dummy_h, dummy_r)
    )
    runner.layered_prefill_state = SimpleNamespace(get=lambda _req: leftover)
    plan = SimpleNamespace(
        prefill_req_ids=("p0",),
        group_id=2,
        group_start=24,
        group_end=36,
    )
    return runner, plan, dummy_h, dummy_r


def test_prepare_p_activation_rank_before_owner_uses_transport_frontier(monkeypatch):
    leftover = SimpleNamespace(
        group_id=1,
        hidden_states=torch.ones(3, 4),
        residual=torch.ones(3, 4),
    )
    received = IntermediateTensors({"hidden_states": torch.full((3, 4), 9.0)})
    for rank in (0, 1):
        runner, plan, dummy_h, dummy_r = _pp4_activation_runner(
            monkeypatch, rank=rank, leftover=leftover
        )
        req_ids, frontier, embeds, inter, source = runner._prepare_layered_p_activation(
            plan,
            num_tokens_padded=3,
            inputs_embeds=torch.ones(3, 4),
            intermediate_tensors=received,
        )
        assert req_ids == ["p0"]
        assert source == "transport_frontier"
        assert frontier[0] is dummy_h
        assert frontier[1] is dummy_r
        assert embeds is None
        assert inter is None


def test_prepare_p_activation_owner_uses_local_frontier(monkeypatch):
    leftover = SimpleNamespace(
        group_id=2,
        hidden_states=torch.full((3, 4), 1.5),
        residual=torch.full((3, 4), 2.5),
    )
    runner, plan, dummy_h, _ = _pp4_activation_runner(
        monkeypatch, rank=2, leftover=leftover
    )
    _req_id, frontier, embeds, inter, source = runner._prepare_layered_p_activation(
        plan,
        num_tokens_padded=3,
        inputs_embeds=torch.ones(3, 4),
        intermediate_tensors=IntermediateTensors(
            {"hidden_states": torch.full((3, 4), 9.0)}
        ),
    )
    assert source == "frontier"
    assert frontier[0] is leftover.hidden_states
    assert embeds is None
    assert inter is None
    assert frontier[0] is not dummy_h


def test_prepare_p_activation_owner_missing_frontier_raises(monkeypatch):
    runner, plan, _h, _r = _pp4_activation_runner(monkeypatch, rank=2, leftover=None)
    try:
        runner._prepare_layered_p_activation(
            plan,
            num_tokens_padded=3,
            inputs_embeds=None,
            intermediate_tensors=None,
        )
    except RuntimeError as error:
        assert "Missing layered activation frontier" in str(error)
    else:
        raise AssertionError("expected missing-frontier error")


def test_prepare_p_activation_rank_after_owner_uses_pp_recv(monkeypatch):
    runner, plan, _h, _r = _pp4_activation_runner(monkeypatch, rank=3, leftover=None)
    received = IntermediateTensors({"hidden_states": torch.full((3, 4), 9.0)})
    _req_id, frontier, embeds, inter, source = runner._prepare_layered_p_activation(
        plan,
        num_tokens_padded=3,
        inputs_embeds=torch.ones(3, 4),
        intermediate_tensors=received,
    )
    assert source == "pp_recv"
    assert frontier is None
    assert embeds is None
    assert inter is received


def test_prepare_p_activation_rank_after_owner_missing_recv_raises(monkeypatch):
    runner, plan, _h, _r = _pp4_activation_runner(monkeypatch, rank=3, leftover=None)
    try:
        runner._prepare_layered_p_activation(
            plan,
            num_tokens_padded=3,
            inputs_embeds=None,
            intermediate_tensors=None,
        )
    except RuntimeError as error:
        assert "did not receive P activation" in str(error)
    else:
        raise AssertionError("expected missing PP recv error")


def test_prepare_p_activation_group0_rank0_uses_embed(monkeypatch):
    runner, _plan, _h, _r = _pp4_activation_runner(monkeypatch, rank=0, leftover=None)
    plan = SimpleNamespace(
        prefill_req_ids=("p0",),
        group_id=0,
        group_start=0,
        group_end=12,
    )
    embeds_in = torch.ones(3, 4)
    _req_id, frontier, embeds, inter, source = runner._prepare_layered_p_activation(
        plan,
        num_tokens_padded=3,
        inputs_embeds=embeds_in,
        intermediate_tensors=None,
    )
    assert source == "embed"
    assert frontier is None
    assert embeds is embeds_in
    assert inter is None


def test_prepare_mixed_activation_non_owner_uses_pp_recv(monkeypatch):
    runner, _plan, _h, _r = _pp4_activation_runner(monkeypatch, rank=1, leftover=None)
    plan = SimpleNamespace(
        prefill_req_ids=("p0",),
        group_id=0,
        group_start=0,
        group_end=6,
    )
    received = IntermediateTensors({"hidden_states": torch.full((4, 4), 9.0)})
    req_ids, frontier, embeds, inter, source = runner._prepare_layered_p_activation(
        plan,
        req_ids=["d0", "p0"],
        num_tokens_padded=4,
        inputs_embeds=torch.ones(4, 4),
        intermediate_tensors=received,
    )
    assert req_ids == ["d0", "p0"]
    assert source == "pp_recv"
    assert frontier is None
    assert embeds is None
    assert inter is received


def test_prepare_mixed_activation_owner_concats_frontiers(monkeypatch):
    from vllm.v1.core.layered_prefill import LayeredFrontier, LayeredPrefillStateStore

    runner, _plan, _h, _r = _pp4_activation_runner(monkeypatch, rank=0, leftover=None)
    plan = SimpleNamespace(
        prefill_req_ids=("p0",),
        group_id=1,
        group_start=6,
        group_end=12,
    )
    store = LayeredPrefillStateStore()
    store.put(
        LayeredFrontier(
            req_id="d0",
            group_id=1,
            query_len=1,
            hidden_states=torch.ones(1, 4),
            residual=None,
        )
    )
    store.put(
        LayeredFrontier(
            req_id="p0",
            group_id=1,
            query_len=3,
            hidden_states=torch.full((3, 4), 2.0),
            residual=None,
        )
    )
    runner.layered_prefill_state = store
    req_ids, frontier, embeds, inter, source = runner._prepare_layered_p_activation(
        plan,
        req_ids=["d0", "p0"],
        num_tokens_padded=5,
        inputs_embeds=None,
        intermediate_tensors=None,
    )
    assert req_ids == ["d0", "p0"]
    assert source == "frontier_mixed"
    assert frontier[0].shape == (5, 4)
    assert embeds is None
    assert inter is None


def test_record_layered_activation_counts_transport_frontier(monkeypatch):
    from vllm_ascend.worker.v2.layered_prefill import empty_layered_prefill_counters

    monkeypatch.setattr(
        "vllm_ascend.worker.v2.model_runner.get_pp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=4),
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.model_runner.get_pp_indices",
        lambda _n, rank, _w: {0: (0, 12), 1: (12, 24), 2: (24, 36), 3: (36, 48)}[rank],
    )
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=48),
        hf_config=None,
    )
    runner.layered_prefill_counters = empty_layered_prefill_counters()
    plan = SimpleNamespace(group_id=2, group_start=24, group_end=36)
    runner._record_layered_activation(plan, "transport_frontier")
    runner._record_layered_activation(plan, "embed")
    counters = runner.layered_prefill_counters
    assert counters["transport_frontier_steps"] == 1
    assert [row["source"] for row in counters["activation_sources"]] == [
        "transport_frontier",
        "embed",
    ]
    snap = runner.layered_prefill_snapshot()
    assert snap["transport_frontier_steps"] == 1
    assert snap["pp_rank"] == 0


def test_fused_pp_payload_drops_rows_before_the_group_owner(monkeypatch):
    """Stages ahead of the group owner ship a placeholder nobody reads."""
    partitions = {0: (0, 12), 1: (12, 24), 2: (24, 36), 3: (36, 48)}
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.model_runner.get_pp_indices",
        lambda _n, rank, _world: partitions[rank],
    )
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=48),
        hf_config=None,
    )
    output = SimpleNamespace(
        hidden_states=torch.ones(7, 4), residual=torch.ones(7, 4)
    )
    runner.layered_prefill_model_adapter = SimpleNamespace(
        to_intermediate_tensors=lambda out: IntermediateTensors(
            {"hidden_states": out.hidden_states, "residual": out.residual}
        )
    )
    # Layers [24, 36) sit on stage 2, so stages 0 and 1 only relay.
    plan = SimpleNamespace(group_start=24, group_end=36)

    for rank, expected_rows in ((0, 0), (1, 0), (2, 7), (3, 7)):
        monkeypatch.setattr(
            "vllm_ascend.worker.v2.model_runner.get_pp_group",
            lambda rank=rank: SimpleNamespace(world_size=4, rank_in_group=rank),
        )
        packed = NPUModelRunner._layered_fused_pp_payload(runner, plan, output)
        assert packed.tensors["hidden_states"].shape[0] == expected_rows
        assert packed.tensors["residual"].shape[0] == expected_rows
        # An empty tensor makes both isend_tensor_dict and irecv_tensor_dict
        # skip the transfer, so the ring stays matched.
        assert (packed.tensors["hidden_states"].numel() == 0) == (
            expected_rows == 0
        )


def _relay_runner(*, pp_size=4, dp_size=1, enabled=True, num_layers=48):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner._skip_relay_stages = enabled
    runner.dp_size = dp_size
    runner.parallel_config = SimpleNamespace(pipeline_parallel_size=pp_size)
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=num_layers),
        hf_config=None,
    )
    return runner


def _relay_plan(group_id, num_groups, start, end, *, sampling=False):
    return SimpleNamespace(
        group_id=group_id,
        num_groups=num_groups,
        group_start=start,
        group_end=end,
        is_sampling_step=sampling,
        is_final_group=group_id + 1 == num_groups,
    )


def _patch_pp(monkeypatch, rank, *, pp_size=4, num_layers=48):
    width = num_layers // pp_size
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.model_runner.get_pp_indices",
        lambda _n, r, _w: (r * width, (r + 1) * width),
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.model_runner.get_pp_group",
        lambda: SimpleNamespace(world_size=pp_size, rank_in_group=rank),
    )


@pytest.mark.parametrize(
    "rank, expected",
    # G=4 on PP=4: group 1 is stage 1's and group 2 is stage 2's, so stage 2
    # must run to keep the next frontier. Stage 0 is ahead, stage 3 behind.
    [(0, "empty"), (1, None), (2, None), (3, "forward")],
)
def test_relay_action_spares_the_owner_and_the_next_owner(
    monkeypatch, rank, expected
):
    _patch_pp(monkeypatch, rank)
    runner = _relay_runner()
    plan = _relay_plan(1, 4, 12, 24)
    action = NPUModelRunner._layered_relay_stage_action(runner, plan)
    assert action == expected


def test_next_owner_is_spared_so_its_frontier_survives(monkeypatch):
    """owner_of(group+1) must run: the step leaves its frontier behind."""
    # Group 2 of 4 is stage 2's and group 3 is stage 3's.
    _patch_pp(monkeypatch, 3)
    runner = _relay_runner()
    assert (
        NPUModelRunner._layered_relay_stage_action(
            runner, _relay_plan(2, 4, 24, 36)
        )
        is None
    )
    _patch_pp(monkeypatch, 0)
    runner = _relay_runner()
    assert (
        NPUModelRunner._layered_relay_stage_action(
            runner, _relay_plan(2, 4, 24, 36)
        )
        == "empty"
    )


def test_final_group_never_relays_even_without_sampling(monkeypatch):
    """The final group clears the frontier on every rank; nobody may skip it.

    The last group of a non-final token chunk is not a sampling step, but it
    still runs the clear, and a rank that skipped it would carry a stale
    frontier into the next chunk's group 0.
    """
    _patch_pp(monkeypatch, 0)
    runner = _relay_runner()
    plan = _relay_plan(3, 4, 36, 48)
    plan.is_sampling_step = False
    assert plan.is_final_group is True
    assert NPUModelRunner._layered_relay_stage_action(runner, plan) is None


def test_next_group_owner_tracks_the_stage_partition(monkeypatch):
    _patch_pp(monkeypatch, 0)
    runner = _relay_runner()
    for group_id, expected in ((0, 1), (1, 2), (2, 3), (3, None)):
        assert (
            NPUModelRunner._layered_next_group_owner(
                runner, _relay_plan(group_id, 4, group_id * 12, (group_id + 1) * 12)
            )
            == expected
        )


def test_sampling_step_never_relays(monkeypatch):
    """The last rank broadcasts there; every other rank must stay to receive."""
    _patch_pp(monkeypatch, 0)
    runner = _relay_runner()
    assert (
        NPUModelRunner._layered_relay_stage_action(
            runner, _relay_plan(3, 4, 36, 48, sampling=True)
        )
        is None
    )
    # is_final_group is the fallback when is_sampling_step is absent.
    bare = SimpleNamespace(
        group_id=3, num_groups=4, group_start=36, group_end=48,
        is_final_group=True,
    )
    assert NPUModelRunner._layered_relay_stage_action(runner, bare) is None


@pytest.mark.parametrize(
    "kwargs", [{"enabled": False}, {"pp_size": 1}, {"dp_size": 2}]
)
def test_relay_is_gated_off(monkeypatch, kwargs):
    """Knob off, PP=1, and DP>1 each disable the relay short-circuit."""
    _patch_pp(monkeypatch, 0)
    runner = _relay_runner(**kwargs)
    assert (
        NPUModelRunner._layered_relay_stage_action(runner, _relay_plan(1, 4, 12, 24))
        is None
    )


def test_relay_is_gated_off_by_a_kv_connector(monkeypatch):
    """With a connector the executor aggregates every rank's output."""
    _patch_pp(monkeypatch, 0)
    runner = _relay_runner()
    runner.vllm_config = SimpleNamespace(kv_transfer_config=object())
    assert (
        NPUModelRunner._layered_relay_stage_action(runner, _relay_plan(1, 4, 12, 24))
        is None
    )


def _relay_step_runner(*, is_last=False, rank=0):
    from vllm_ascend.worker.v2.layered_prefill import (
        empty_layered_prefill_counters,
    )

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.is_last_pp_rank = is_last
    runner.layered_prefill_counters = empty_layered_prefill_counters()
    return runner


def test_relay_step_forwards_the_received_activation(monkeypatch):
    """A stage behind the owner passes its input straight on."""
    _patch_pp(monkeypatch, 3)
    runner = _relay_step_runner()
    received = IntermediateTensors({"hidden_states": torch.ones(5, 2)})
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"p0": 8, "d0": 1})

    packed = NPUModelRunner._execute_layered_relay_step(
        runner, scheduler_output, "forward", received, fused=True
    )
    assert packed is received
    assert runner.execute_model_state is None
    # sample_tokens still owes update_from_output one row per request.
    assert runner._layered_relay_output.req_ids == ["p0", "d0"]
    assert runner._layered_relay_output.sampled_token_ids == [[], []]


def test_relay_step_counts_are_reported_per_action(monkeypatch):
    """A smoke probe needs a positive signal that the skip actually fired."""
    _patch_pp(monkeypatch, 3)
    runner = _relay_step_runner()
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"p0": 8})
    received = IntermediateTensors({"hidden_states": torch.ones(5, 2)})
    NPUModelRunner._execute_layered_relay_step(
        runner, scheduler_output, "forward", received, fused=True
    )
    NPUModelRunner._execute_layered_relay_step(
        runner, scheduler_output, "forward", received, fused=True
    )
    _patch_pp(monkeypatch, 0)
    runner.layered_prefill_model_adapter = SimpleNamespace(
        make_transport_frontier=lambda n, dtype, device: (
            torch.zeros(n, 4), torch.zeros(n, 4)
        ),
        to_intermediate_tensors=lambda out: IntermediateTensors(
            {"hidden_states": out.hidden_states, "residual": out.residual}
        ),
    )
    runner.model_config = SimpleNamespace(dtype=torch.float16)
    runner.device = torch.device("cpu")
    NPUModelRunner._execute_layered_relay_step(
        runner, scheduler_output, "empty", None, fused=True
    )
    counters = runner.layered_prefill_counters
    assert counters["relay_forward_steps"] == 2
    assert counters["relay_empty_steps"] == 1


def test_relay_step_on_the_last_rank_returns_no_tensors(monkeypatch):
    _patch_pp(monkeypatch, 3)
    runner = _relay_step_runner(is_last=True)
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"p0": 8})
    assert (
        NPUModelRunner._execute_layered_relay_step(
            runner, scheduler_output, "forward", None, fused=True
        )
        is None
    )
    assert runner._layered_relay_output.req_ids == ["p0"]


def test_relay_step_rejects_a_missing_activation(monkeypatch):
    _patch_pp(monkeypatch, 3)
    runner = _relay_step_runner()
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"p0": 8})
    with pytest.raises(RuntimeError, match="no activation to pass on"):
        NPUModelRunner._execute_layered_relay_step(
            runner, scheduler_output, "forward", None, fused=True
        )


def _placeholder_runner():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.layered_prefill_model_adapter = SimpleNamespace(
        make_transport_frontier=lambda n, dtype, device: (
            torch.zeros(n, 4), torch.zeros(n, 4)
        ),
        to_intermediate_tensors=lambda out: IntermediateTensors(
            {"hidden_states": out.hidden_states, "residual": out.residual}
        ),
    )
    runner.model_config = SimpleNamespace(dtype=torch.float16)
    runner.device = torch.device("cpu")
    return runner


def test_relay_placeholder_unpacks_off_the_fused_path():
    """What the PP=4 smoke caught: the next stage still unpacks a placeholder.

    Off the fused path a stage runs its payload through
    _split_layered_pp_intermediate even when it goes on to restore its own
    stored frontier, and that insists on the D/P row counts.  A placeholder
    without them killed the worker mid-run.
    """
    runner = _placeholder_runner()

    packed = NPUModelRunner._layered_pre_owner_placeholder(
        runner, with_row_metadata=True
    )
    d_part, p_part = NPUModelRunner._split_layered_pp_intermediate(packed)
    assert d_part.tensors["hidden_states"].shape[0] == 0
    assert p_part.tensors["hidden_states"].shape[0] == 0


def test_relay_placeholder_stays_bare_on_the_fused_path():
    """The fused receiver feeds the payload to the model, so no extra keys."""
    runner = _placeholder_runner()
    packed = NPUModelRunner._layered_pre_owner_placeholder(runner)
    assert set(packed.tensors) == {"hidden_states", "residual"}


def test_sample_tokens_consumes_the_relay_output_once():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    stashed = ModelRunnerOutput(
        req_ids=["p0"],
        req_id_to_index={"p0": 0},
        sampled_token_ids=[[]],
    )
    runner._layered_relay_output = stashed
    runner.execute_model_state = None
    assert NPUModelRunner.sample_tokens(runner) is stashed
    assert runner._layered_relay_output is None
