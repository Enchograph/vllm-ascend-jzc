"""RingStepPlan: participation is shared; the local mask only applies rows."""

from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker.gpu.pp_utils import (
    PPHandler,
    RingStepRecord,
    build_ring_step_plan,
    reconcile_ring_records,
    ring_collective_action,
)

from vllm_ascend.worker.v2.layered_prefill import LayeredPPHandlerCapture


def _batch(req_ids, *, computed, prefill, scheduled, max_seq, indices):
    n = len(req_ids)
    return SimpleNamespace(
        req_ids=req_ids,
        num_reqs=n,
        num_computed_tokens_np=np.array(computed, dtype=np.int32),
        prefill_len_np=np.array(prefill, dtype=np.int32),
        max_seq_len_np=np.array(max_seq, dtype=np.int32),
        num_scheduled_tokens=np.array(scheduled, dtype=np.int32),
        idx_mapping=torch.tensor(indices, dtype=torch.int32),
        idx_mapping_np=np.array(indices, dtype=np.int32),
    )


def _plan(order, *, sampling, computed, prefill, scheduled, max_seq, step_id=1, width=1):
    return build_ring_step_plan(
        step_id=step_id,
        pp_size=4,
        sample_width=width,
        request_order=order,
        old_computed=dict(zip(order, computed, strict=True)),
        num_scheduled=dict(zip(order, scheduled, strict=True)),
        prefill_len=dict(zip(order, prefill, strict=True)),
        max_seq_len=dict(zip(order, max_seq, strict=True)),
        sampling_step=sampling,
    )


class _Rank(PPHandler):
    """PPHandler without a process group. The collective itself is recorded."""

    def __init__(self, plan, *, last: bool):
        self.ring_step_plan = plan
        self.ring_ledger = []
        self.is_last_rank = last
        self.entered: list = []

    def bind_plan(self, plan):
        self.ring_step_plan = plan

    def _enter_broadcast(self, tokens, num_sampled, num_rejected):
        self.entered.append(
            ("broadcast", tokens.clone(), num_sampled.clone(), num_rejected.clone())
        )

    def _enter_receive(self, input_batch, plan, apply_mask):
        self.entered.append(
            (
                "receive",
                plan.payload_rows,
                plan.sample_width,
                tuple(plan.sampled_request_order),
                apply_mask.copy(),
            )
        )


def _output(order, scheduled, *, computed, layered=None):
    cached = CachedRequestData.make_empty()
    cached.req_ids = list(order)
    cached.num_computed_tokens = list(computed)
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=dict(zip(order, scheduled, strict=True)),
        total_num_scheduled_tokens=sum(scheduled),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        layered_prefill_plan=layered,
    )


def _scheduler(requests, *, pp_size: int):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        requests=requests,
        num_spec_tokens=0,
        _ring_step_seq=0,
    )


def test_missing_plan_refuses_local_skip():
    rank = _Rank(None, last=True)
    batch = _batch(
        ["d0"],
        computed=[8],
        prefill=[8],
        scheduled=[1],
        max_seq=[64],
        indices=[0],
    )
    try:
        rank.broadcast(
            torch.tensor([[1]], dtype=torch.int64),
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            batch,
        )
    except RuntimeError as error:
        assert "no RingStepPlan" in str(error)
    else:
        raise AssertionError("expected missing-plan error")
    assert rank.entered == []


def test_local_mask_does_not_change_participation():
    """One rank would apply nothing; the other would apply the row. Both enter."""
    plan = _plan(
        ["d0"],
        sampling=True,
        computed=[8],
        prefill=[8],
        scheduled=[1],
        max_seq=[64],
        step_id=10,
    )
    assert plan.collective_required
    finishing = _batch(
        ["d0"],
        computed=[8],
        prefill=[8],
        scheduled=[1],
        max_seq=[9],
        indices=[0],
    )
    continuing = _batch(
        ["d0"],
        computed=[8],
        prefill=[8],
        scheduled=[1],
        max_seq=[64],
        indices=[1],
    )
    last = _Rank(plan, last=True)
    other = _Rank(plan, last=False)
    last.broadcast(
        torch.tensor([[4]], dtype=torch.int64),
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        continuing,
    )
    other.receive(finishing)
    assert [item[0] for item in last.entered] == ["broadcast"]
    assert other.entered[0][1:4] == (1, 1, ("d0",))
    assert other.entered[0][4].tolist() == [False]
    reconcile_ring_records(last.ring_ledger + other.ring_ledger)
    assert {record.participated for record in last.ring_ledger + other.ring_ledger} == {
        True
    }
    assert other.ring_ledger[0].apply_count == 0


def test_global_skip_when_layered_group_does_not_sample():
    requests = {
        "p0": SimpleNamespace(num_prompt_tokens=128, max_tokens=896),
    }
    sched = _scheduler(requests, pp_size=4)
    output = _output(
        ["p0"],
        [128],
        computed=[0],
        layered=SimpleNamespace(is_sampling_step=False),
    )
    Scheduler._attach_ring_step_plan(sched, output)
    plan = output.ring_step_plan
    assert plan is not None
    assert plan.collective_required is False
    assert ring_collective_action(plan) == "skip"

    captured = _batch(
        ["p0"],
        computed=[0],
        prefill=[128],
        scheduled=[128],
        max_seq=[1024],
        indices=[3],
    )
    last = _Rank(plan, last=True)
    first = _Rank(plan, last=False)
    last.broadcast(
        torch.tensor([[1]], dtype=torch.int64),
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        captured,
    )
    first.receive(captured)
    assert last.entered == [] and first.entered == []
    reconcile_ring_records(last.ring_ledger + first.ring_ledger)
    assert all(not record.participated for record in last.ring_ledger + first.ring_ledger)


def test_rider_cancel_keeps_plan_shape():
    plan = _plan(
        ["p0"],
        sampling=True,
        computed=[5],
        prefill=[8],
        scheduled=[3],
        max_seq=[64],
        step_id=11,
        width=1,
    )
    assert plan.collective_required
    wide = _batch(
        ["d0", "p0"],
        computed=[8, 5],
        prefill=[8, 8],
        scheduled=[1, 3],
        max_seq=[64, 64],
        indices=[0, 1],
    )
    last = _Rank(plan, last=True)
    first = _Rank(plan, last=False)
    last.broadcast(
        torch.tensor([[9], [3]], dtype=torch.int64),
        torch.tensor([1, 1], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        wide,
    )
    narrow = _batch(
        ["p0"],
        computed=[5],
        prefill=[8],
        scheduled=[3],
        max_seq=[64],
        indices=[1],
    )
    first.receive(narrow)
    kind, tokens, sampled, _rejected = last.entered[0]
    assert kind == "broadcast"
    assert tokens.shape == (1, 1)
    assert tokens.tolist() == [[3]]
    assert sampled.tolist() == [1]
    assert first.entered[0][1:4] == (1, 1, ("p0",))
    assert first.entered[0][4].tolist() == [True]
    reconcile_ring_records(last.ring_ledger + first.ring_ledger)


def test_empty_output_still_enters():
    plan = _plan(
        ["d0"],
        sampling=True,
        computed=[8],
        prefill=[8],
        scheduled=[1],
        max_seq=[64],
        step_id=12,
    )
    batch = _batch(
        ["d0"],
        computed=[8],
        prefill=[8],
        scheduled=[1],
        max_seq=[64],
        indices=[0],
    )
    last = _Rank(plan, last=True)
    first = _Rank(plan, last=False)
    last.broadcast(
        torch.tensor([[-1]], dtype=torch.int64),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        batch,
    )
    first.receive(batch)
    assert last.entered[0][1].shape == (1, 1)
    assert last.entered[0][2].tolist() == [0]
    assert first.entered[0][0] == "receive"
    reconcile_ring_records(last.ring_ledger + first.ring_ledger)


def test_flush_reorders_interleaved_parents_and_matches_peer():
    plan = _plan(
        ["a", "b"],
        sampling=True,
        computed=[0, 0],
        prefill=[4, 4],
        scheduled=[4, 4],
        max_seq=[32, 32],
        step_id=13,
        width=1,
    )
    parent_b = _batch(
        ["b"],
        computed=[0],
        prefill=[4],
        scheduled=[4],
        max_seq=[32],
        indices=[2],
    )
    parent_a = _batch(
        ["a"],
        computed=[0],
        prefill=[4],
        scheduled=[4],
        max_seq=[32],
        indices=[1],
    )

    class _Fake:
        def __init__(self, *, last: bool):
            self.is_last_rank = last
            self.ring_step_plan = None
            self.calls = []

        def broadcast(self, tokens, num_sampled, num_rejected, batch):
            self.calls.append((list(batch.req_ids), tokens.clone(), num_sampled.clone()))

        def receive(self, batch):
            self.calls.append(list(batch.req_ids))

    sender = _Fake(last=True)
    send = LayeredPPHandlerCapture(sender)
    send.broadcast(
        torch.tensor([[20]], dtype=torch.int64),
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        parent_b,
    )
    send.broadcast(
        torch.tensor([[10]], dtype=torch.int64),
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        parent_a,
    )
    send_stats = send.flush(sample_p=True, p_req_ids={"a", "b"}, plan=plan)

    receiver = _Fake(last=False)
    recv = LayeredPPHandlerCapture(receiver)
    recv.receive(parent_b)
    recv.receive(parent_a)
    recv_stats = recv.flush(sample_p=True, p_req_ids={"a", "b"}, plan=plan)

    assert sender.calls[0][0] == ["a", "b"]
    assert sender.calls[0][1].tolist() == [[10], [20]]
    assert receiver.calls == [["a", "b"]]
    assert send_stats["plan_hash"] == recv_stats["plan_hash"]
    assert send_stats["payload_rows"] == recv_stats["payload_rows"] == 2
    assert send_stats["participated"] and recv_stats["participated"]


def test_flush_empty_sender_still_matches_receiver_shape():
    plan = _plan(
        ["p0"],
        sampling=True,
        computed=[8],
        prefill=[8],
        scheduled=[1],
        max_seq=[64],
        step_id=14,
    )

    class _Fake:
        def __init__(self, *, last: bool):
            self.is_last_rank = last
            self.calls = []

        def broadcast(self, tokens, num_sampled, num_rejected, batch):
            self.calls.append((tokens.shape, num_sampled.tolist(), list(batch.req_ids)))

        def receive(self, batch):
            self.calls.append(list(batch.req_ids))

    sender = _Fake(last=True)
    send_stats = LayeredPPHandlerCapture(sender).flush(
        sample_p=True, p_req_ids={"p0"}, plan=plan
    )
    receiver = _Fake(last=False)
    batch = _batch(
        ["p0"],
        computed=[8],
        prefill=[8],
        scheduled=[1],
        max_seq=[64],
        indices=[0],
    )
    capture = LayeredPPHandlerCapture(receiver)
    capture.receive(batch)
    recv_stats = capture.flush(sample_p=True, p_req_ids={"p0"}, plan=plan)
    assert sender.calls == [((1, 1), [0], ["p0"])]
    assert receiver.calls == [["p0"]]
    assert send_stats["participated"] and recv_stats["participated"]
    assert send_stats["req_ids"] == recv_stats["req_ids"] == ["p0"]


def test_pp1_does_not_issue_a_plan_and_final_group_does():
    requests = {
        "p0": SimpleNamespace(num_prompt_tokens=8, max_tokens=8),
    }
    single = _scheduler(requests, pp_size=1)
    output = _output(["p0"], [1], computed=[8])
    Scheduler._attach_ring_step_plan(single, output)
    assert output.ring_step_plan is None

    multi = _scheduler(requests, pp_size=4)
    final = _output(
        ["p0"],
        [1],
        computed=[8],
        layered=SimpleNamespace(is_sampling_step=True),
    )
    Scheduler._attach_ring_step_plan(multi, final)
    plan = final.ring_step_plan
    assert plan is not None and plan.collective_required
    assert plan.sampled_request_order == ("p0",)
    assert plan.sample_width == 1
    assert isinstance(plan.plan_hash, str) and len(plan.plan_hash) == 12


def test_reconcile_rejects_disagreement():
    left = RingStepRecord(
        step_id=1,
        role="broadcast",
        participated=True,
        payload_rows=1,
        sample_width=1,
        request_order=("a",),
        plan_hash="abc",
        apply_count=1,
    )
    right = RingStepRecord(
        step_id=1,
        role="receive",
        participated=False,
        payload_rows=1,
        sample_width=1,
        request_order=("a",),
        plan_hash="abc",
        apply_count=0,
    )
    try:
        reconcile_ring_records([left, right])
    except AssertionError as error:
        assert "diverged" in str(error)
    else:
        raise AssertionError("expected reconcile failure")
