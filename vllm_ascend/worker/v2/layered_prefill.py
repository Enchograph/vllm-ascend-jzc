# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Layered Prefill helpers for Ascend Model Runner V2.

See /home/jzc/gjc/layered_prefill_v2_migration_plan.md §§3–4.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

from vllm.v1.core.layered_prefill import LayeredFrontier, LayeredPrefillStateStore
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.gpu.model_runner import ExecuteModelState
from vllm.v1.worker.gpu.pp_utils import (
    RingStepPlan,
    align_sampled_payload,
    ring_collective_action,
)

if TYPE_CHECKING:
    from vllm.v1.core.layered_prefill import LayeredPrefillPlan


def empty_layered_prefill_counters() -> dict[str, Any]:
    return {
        "execute_steps": 0,
        "transport_frontier_steps": 0,
        "fused_mixed_steps": 0,
        "same_layer_steps": 0,
        "groups": [],
        "pp_slots": [],
        "activation_sources": [],
    }


@dataclass
class LayeredV2ExecuteModelState:
    """Combined D/P execute state handed to sample_tokens.

    ``sample_p`` is True only on ``LayeredPrefillPlan.is_sampling_step``
    (final layer group of the final prompt chunk). Intermediate groups and
    non-final chunks skip sampler, ``postprocess_num_computed_tokens``, and
    the PPHandler slot for the P rows (plan §3.3 / V6).  Final-P shares that
    one slot with D.

    ``fused_mixed``: D+P ran as one eager layer-group forward. ``d_state`` is
    None and ``p_state`` holds the combined batch. Sample the whole batch on
    ``sample_p``; otherwise emit empty tokens for every mixed row.
    """

    scheduler_output: SchedulerOutput
    d_state: ExecuteModelState | None
    p_state: ExecuteModelState | None
    sample_p: bool
    fused_mixed: bool = False
    # Requests whose logits must be dropped when the prompt chunk is not done.
    prefill_req_ids: tuple[str, ...] = ()


def detach_execute_model_state(state: ExecuteModelState) -> ExecuteModelState:
    """Clone tensors that P forward may overwrite in shared model workspaces.

    ``input_buffers`` are already isolated via a second AscendInputBuffers set.
    Model activations / aux hidden states are not: the P sub-batch reuses the
    same layer scratch as D, so sampling D after P without a clone reads P's
    last write (first token OK on P-only steps, then decode collapses to
    commas / noise under true P+D).
    """
    hidden = state.hidden_states
    aux = state.aux_hidden_states
    return state._replace(
        hidden_states=None if hidden is None else hidden.clone(),
        aux_hidden_states=(
            None if aux is None else [tensor.clone() for tensor in aux]
        ),
    )


def subset_cached_request_data(
    data: CachedRequestData, req_ids: list[str]
) -> CachedRequestData:
    indices = [data.req_ids.index(req_id) for req_id in req_ids if req_id in data.req_ids]
    selected_ids = [data.req_ids[index] for index in indices]

    def aligned(values: list) -> list:
        return [values[index] for index in indices] if len(values) == len(data.req_ids) else []

    return CachedRequestData(
        req_ids=selected_ids,
        resumed_req_ids=data.resumed_req_ids.intersection(selected_ids),
        new_token_ids=aligned(data.new_token_ids),
        all_token_ids={
            req_id: data.all_token_ids[req_id]
            for req_id in selected_ids
            if req_id in data.all_token_ids
        },
        new_block_ids=aligned(data.new_block_ids),
        num_computed_tokens=aligned(data.num_computed_tokens),
        num_output_tokens=aligned(data.num_output_tokens),
    )


def subset_scheduler_output(
    scheduler_output: SchedulerOutput,
    req_ids: list[str],
    *,
    layered_plan: Any = None,
    include_one_time_updates: bool = True,
) -> SchedulerOutput:
    """Crop a SchedulerOutput to ``req_ids``.

    One-time lifecycle fields (finished / preempted / connector / block zeroing)
    must appear on exactly one sub-batch per scheduler step.
    """
    req_id_set = set(req_ids)
    num_scheduled_tokens = {
        req_id: scheduler_output.num_scheduled_tokens[req_id]
        for req_id in req_ids
        if req_id in scheduler_output.num_scheduled_tokens
    }
    scheduled_new_reqs = [
        data for data in scheduler_output.scheduled_new_reqs if data.req_id in req_id_set
    ]
    scheduled_cached_reqs = subset_cached_request_data(
        scheduler_output.scheduled_cached_reqs, req_ids
    )
    scheduled_spec_decode_tokens = {
        req_id: tokens
        for req_id, tokens in scheduler_output.scheduled_spec_decode_tokens.items()
        if req_id in req_id_set
    }
    scheduled_encoder_inputs = {
        req_id: inputs
        for req_id, inputs in scheduler_output.scheduled_encoder_inputs.items()
        if req_id in req_id_set
    }
    num_invalid_spec_tokens = None
    if scheduler_output.num_invalid_spec_tokens is not None:
        num_invalid_spec_tokens = {
            req_id: value
            for req_id, value in scheduler_output.num_invalid_spec_tokens.items()
            if req_id in req_id_set
        }
    partial_tail_offloads = None
    if scheduler_output.partial_tail_offloads is not None:
        partial_tail_offloads = {
            req_id: value
            for req_id, value in scheduler_output.partial_tail_offloads.items()
            if req_id in req_id_set
        }
    return replace(
        scheduler_output,
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=scheduled_cached_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        scheduled_encoder_inputs=scheduled_encoder_inputs,
        scheduled_encoder_input_stats=(
            scheduler_output.scheduled_encoder_input_stats
            if include_one_time_updates
            else None
        ),
        finished_req_ids=(
            scheduler_output.finished_req_ids if include_one_time_updates else set()
        ),
        preempted_req_ids=(
            scheduler_output.preempted_req_ids if include_one_time_updates else set()
        ),
        free_encoder_mm_hashes=(
            scheduler_output.free_encoder_mm_hashes if include_one_time_updates else []
        ),
        new_block_ids_to_zero=(
            scheduler_output.new_block_ids_to_zero if include_one_time_updates else None
        ),
        kv_cache_block_copies=(
            scheduler_output.kv_cache_block_copies if include_one_time_updates else None
        ),
        kv_connector_metadata=(
            scheduler_output.kv_connector_metadata if include_one_time_updates else None
        ),
        ec_connector_metadata=(
            scheduler_output.ec_connector_metadata if include_one_time_updates else None
        ),
        ec_manager_metadata=(
            scheduler_output.ec_manager_metadata if include_one_time_updates else None
        ),
        partial_tail_offloads=partial_tail_offloads,
        num_invalid_spec_tokens=num_invalid_spec_tokens,
        layered_prefill_plan=layered_plan,
    )


def split_d_p_req_ids(
    scheduler_output: SchedulerOutput,
    plan: "LayeredPrefillPlan",
) -> tuple[list[str], list[str]]:
    all_req_ids = list(scheduler_output.num_scheduled_tokens)
    p_req_ids = list(plan.prefill_req_ids)
    if len(p_req_ids) != 1:
        raise RuntimeError("Layered prefill Phase 1 supports exactly one P request")
    p_req_set = set(p_req_ids)
    if not p_req_set.issubset(all_req_ids):
        raise RuntimeError("Layered plan contains an unscheduled P request")
    d_req_ids = [req_id for req_id in all_req_ids if req_id not in p_req_set]
    return d_req_ids, p_req_ids


def snapshot_pp_input_batch(input_batch) -> SimpleNamespace:
    """Copy the fields ``PPHandler`` reads so a later P sample cannot alias them."""
    num_reqs = int(input_batch.num_reqs)
    max_seq = getattr(input_batch, "max_seq_len_np", None)
    return SimpleNamespace(
        req_ids=list(input_batch.req_ids)[:num_reqs],
        num_reqs=num_reqs,
        num_computed_tokens_np=np.array(
            input_batch.num_computed_tokens_np[:num_reqs], copy=True
        ),
        prefill_len_np=np.array(input_batch.prefill_len_np[:num_reqs], copy=True),
        max_seq_len_np=(
            None if max_seq is None else np.array(max_seq[:num_reqs], copy=True)
        ),
        num_scheduled_tokens=np.array(
            input_batch.num_scheduled_tokens[:num_reqs], copy=True
        ),
        idx_mapping=input_batch.idx_mapping[:num_reqs].detach().clone(),
        idx_mapping_np=np.array(input_batch.idx_mapping_np[:num_reqs], copy=True),
    )


def concat_pp_input_batches(batches: list) -> SimpleNamespace:
    """Stack D then P along the request axis for one PPHandler slot."""
    if not batches:
        raise RuntimeError("Layered PP merge received no input batches")
    if len(batches) == 1:
        return batches[0]

    def _cat_np(attr: str) -> np.ndarray:
        return np.concatenate(
            [getattr(batch, attr)[: batch.num_reqs] for batch in batches]
        )

    max_seq_parts = [batch.max_seq_len_np for batch in batches]
    if any(part is None for part in max_seq_parts):
        max_seq = None
    else:
        max_seq = np.concatenate(
            [part[: batch.num_reqs] for part, batch in zip(max_seq_parts, batches)]
        )
    return SimpleNamespace(
        req_ids=[
            req_id
            for batch in batches
            for req_id in list(batch.req_ids)[: batch.num_reqs]
        ],
        num_reqs=sum(batch.num_reqs for batch in batches),
        num_computed_tokens_np=_cat_np("num_computed_tokens_np"),
        prefill_len_np=_cat_np("prefill_len_np"),
        max_seq_len_np=max_seq,
        num_scheduled_tokens=_cat_np("num_scheduled_tokens"),
        idx_mapping=torch.cat(
            [batch.idx_mapping[: batch.num_reqs] for batch in batches], dim=0
        ),
        idx_mapping_np=_cat_np("idx_mapping_np"),
    )


def compute_layered_need_sampled_mask(
    input_batch,
    *,
    sample_p: bool,
    p_req_ids: set[str] | frozenset[str] | None = None,
) -> np.ndarray | None:
    """Upstream mask plus layered-aware exclusion of intermediate P rows.

    ``PPHandler.compute_need_sampled_mask`` treats ``old_computed=0`` and
    ``num_scheduled == prefill_len`` as a final prefill.  Every layered
    intermediate group looks like that, so those rows must not occupy the
    sampled-token slot (migration plan §4 V6).
    """
    from vllm.v1.worker.gpu.pp_utils import compute_need_sampled_mask

    mask = compute_need_sampled_mask(input_batch)
    if mask is None:
        return None
    mask = np.array(mask, copy=True)
    if not sample_p and p_req_ids:
        for index, req_id in enumerate(list(input_batch.req_ids)[: input_batch.num_reqs]):
            if req_id in p_req_ids:
                mask[index] = False
    return mask if mask.any() else None


def _is_p_only_batch(batch, p_req_ids: set[str]) -> bool:
    if not p_req_ids:
        return False
    return set(batch.req_ids[: batch.num_reqs]).issubset(p_req_ids)


def _pp_flush_stats(
    *,
    role: str,
    sample_p: bool,
    kept: list,
    dropped: list,
    p_ids: set[str],
) -> dict[str, Any]:
    def _rows(batches: list) -> tuple[int, int]:
        n_d = 0
        n_p = 0
        for batch in batches:
            n_req = int(batch.num_reqs)
            if _is_p_only_batch(batch, p_ids):
                n_p += n_req
            else:
                n_d += n_req
        return n_d, n_p

    n_d, n_p = _rows(kept)
    dropped_d, dropped_p = _rows(dropped)
    req_ids = [
        req_id
        for batch in kept
        for req_id in list(batch.req_ids)[: batch.num_reqs]
    ]
    return {
        "role": role,
        "sample_p": bool(sample_p),
        "n_d": n_d,
        "n_p": n_p,
        "dropped_p_rows": dropped_p,
        "dropped_d_rows": dropped_d,
        "req_ids": req_ids,
        "skipped": not kept,
    }


def _batch_covering_plan(batches, plan: RingStepPlan):
    """One batch whose rows are exactly ``plan.sampled_request_order``."""
    selected = _select_pp_batches(batches, plan.sampled_request_order)
    by_id = {req_id: index for index, req_id in enumerate(selected.req_ids)}

    def _take(name: str, fill: int) -> np.ndarray:
        out = np.full(plan.payload_rows, fill, dtype=np.int64)
        for row, req_id in enumerate(plan.sampled_request_order):
            index = by_id.get(req_id)
            if index is None:
                continue
            out[row] = int(getattr(selected, name)[index])
        return out

    idx = np.full(plan.payload_rows, -1, dtype=np.int32)
    for row, req_id in enumerate(plan.sampled_request_order):
        index = by_id.get(req_id)
        if index is not None:
            idx[row] = int(selected.idx_mapping_np[index])
    return SimpleNamespace(
        req_ids=list(plan.sampled_request_order),
        num_reqs=plan.payload_rows,
        num_computed_tokens_np=_take("num_computed_tokens_np", 0),
        prefill_len_np=_take("prefill_len_np", 0),
        max_seq_len_np=_take("max_seq_len_np", 0),
        num_scheduled_tokens=_take("num_scheduled_tokens", 0),
        idx_mapping=torch.tensor(idx, dtype=torch.int32),
        idx_mapping_np=idx,
    )


def _empty_pp_batch(plan: RingStepPlan):
    """Batch with no local rows. The collective shape still comes from ``plan``."""
    return SimpleNamespace(
        req_ids=[],
        num_reqs=0,
        num_computed_tokens_np=np.zeros(0, dtype=np.int64),
        prefill_len_np=np.zeros(0, dtype=np.int64),
        max_seq_len_np=np.zeros(0, dtype=np.int64),
        num_scheduled_tokens=np.zeros(0, dtype=np.int64),
        idx_mapping=torch.zeros(0, dtype=torch.int32),
        idx_mapping_np=np.zeros(0, dtype=np.int32),
    )


def _select_pp_batches(batches, request_order: Sequence[str]):
    """Rows named by the plan, in plan order. Extra captured ids are omitted."""
    found: dict[str, tuple] = {}
    for batch in batches:
        ids = list(batch.req_ids)[: batch.num_reqs]
        for index, req_id in enumerate(ids):
            found[req_id] = (batch, index)
    chosen = [req_id for req_id in request_order if req_id in found]
    if not chosen:
        return SimpleNamespace(
            req_ids=[],
            num_reqs=0,
            num_computed_tokens_np=np.zeros(0, dtype=np.int64),
            prefill_len_np=np.zeros(0, dtype=np.int64),
            max_seq_len_np=np.zeros(0, dtype=np.int64),
            num_scheduled_tokens=np.zeros(0, dtype=np.int64),
            idx_mapping=torch.zeros(0, dtype=torch.int32),
            idx_mapping_np=np.zeros(0, dtype=np.int32),
        )

    def _col(name: str) -> np.ndarray:
        parts = []
        for req_id in chosen:
            batch, index = found[req_id]
            parts.append(np.array(getattr(batch, name)[index], copy=True))
        return np.stack(parts)

    idx = torch.tensor(
        [
            int(found[req_id][0].idx_mapping[found[req_id][1]])
            for req_id in chosen
        ],
        dtype=torch.int32,
    )
    return SimpleNamespace(
        req_ids=chosen,
        num_reqs=len(chosen),
        num_computed_tokens_np=_col("num_computed_tokens_np"),
        prefill_len_np=_col("prefill_len_np"),
        max_seq_len_np=_col("max_seq_len_np"),
        num_scheduled_tokens=_col("num_scheduled_tokens"),
        idx_mapping=idx,
        idx_mapping_np=idx.numpy().astype(np.int32, copy=True),
    )


class LayeredPPHandlerCapture:
    """Capture D/P ``broadcast``/``receive`` and flush them as one PPHandler slot.

    ``PPHandler.queue`` holds one entry per scheduler step.  Sampling D then
    the final P group would overwrite that entry unless the two sub-batches
    share a single NCCL payload (concatenated along the request axis).
    Intermediate P groups are dropped: they must not produce a sample.
    """

    def __init__(self, inner):
        self.inner = inner
        self.broadcasts: list[tuple] = []
        self.receives: list = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor | None,
        num_rejected: torch.Tensor | None,
        input_batch,
    ) -> None:
        snapshot = snapshot_pp_input_batch(input_batch)
        num_reqs = snapshot.num_reqs
        device = sampled_token_ids.device
        tokens = sampled_token_ids[:num_reqs].detach().clone()
        if num_sampled is None:
            sampled = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        else:
            sampled = num_sampled[:num_reqs].detach().clone()
        if num_rejected is None:
            rejected = torch.zeros(num_reqs, dtype=sampled.dtype, device=sampled.device)
        else:
            rejected = num_rejected[:num_reqs].detach().clone()
        self.broadcasts.append((tokens, sampled, rejected, snapshot))

    def receive(self, input_batch) -> bool:
        # Always False: D and P are flushed together, so neither sub-batch
        # can claim "all rows decode next" until the merged mask is known.
        # Default model_state.postprocess_state is a no-op for int 0.
        self.receives.append(snapshot_pp_input_batch(input_batch))
        return False

    def flush(
        self,
        *,
        sample_p: bool,
        p_req_ids: set[str] | frozenset[str] | None = None,
        plan: RingStepPlan | None = None,
    ) -> dict[str, Any]:
        """Enter or skip the sampled-token collective from ``plan`` only.

        ``sample_p`` is recorded for diagnostics. It does not drop rows and
        it does not decide participation. An empty local capture still
        enters when the plan says so, with empty rows for requests this
        rank did not sample.
        """
        del sample_p  # participation is the plan's, not this flag's
        p_ids = set(p_req_ids or ())
        inner = self.inner
        if plan is None:
            plan = getattr(inner, "ring_step_plan", None)
        action = ring_collective_action(plan)
        if action == "missing":
            raise RuntimeError(
                "Layered PP flush has no RingStepPlan; refusing to skip "
                "the sampled-token collective from the local batch"
            )
        assert plan is not None
        if hasattr(inner, "bind_plan"):
            inner.bind_plan(plan)
        else:
            inner.ring_step_plan = plan
        role = "broadcast" if getattr(inner, "is_last_rank", False) else "receive"
        captured = (
            [payload[3] for payload in self.broadcasts]
            if role == "broadcast"
            else list(self.receives)
        )
        extra = [
            req_id
            for batch in captured
            for req_id in list(batch.req_ids)[: batch.num_reqs]
            if req_id not in plan.sampled_request_order
        ]
        stats = _pp_flush_stats(
            role=role,
            sample_p=bool(getattr(plan, "collective_required", False)),
            kept=captured,
            dropped=[],
            p_ids=p_ids,
        )
        stats.update(
            step_id=plan.step_id,
            plan_hash=plan.plan_hash,
            participated=action == "enter",
            skipped=action != "enter",
            payload_rows=plan.payload_rows,
            sample_width=plan.sample_width,
            req_ids=list(plan.sampled_request_order),
            dropped_not_in_plan=len(extra),
            dropped_p_rows=0,
            dropped_d_rows=0,
        )
        if action == "skip":
            note = getattr(inner, "record_ring_skip", None)
            if note is not None:
                note(role)
            return stats
        if role == "broadcast":
            self._flush_broadcast(plan)
        else:
            self._flush_receive(plan)
        return stats

    def _flush_broadcast(self, plan: RingStepPlan) -> None:
        req_ids: list[str] = []
        token_rows: list[torch.Tensor] = []
        sampled_rows: list[torch.Tensor] = []
        rejected_rows: list[torch.Tensor] = []
        batches = []
        for tokens, sampled, rejected, batch in self.broadcasts:
            ids = list(batch.req_ids)[: batch.num_reqs]
            req_ids.extend(ids)
            token_rows.append(tokens[: len(ids)])
            sampled_rows.append(sampled[: len(ids)])
            rejected_rows.append(rejected[: len(ids)])
            batches.append(batch)
        if token_rows:
            tokens = torch.cat(token_rows, dim=0)
            num_sampled = torch.cat(sampled_rows, dim=0)
            num_rejected = torch.cat(rejected_rows, dim=0)
        else:
            tokens = torch.empty(
                (0, plan.sample_width), dtype=torch.int64
            )
            num_sampled = torch.empty(0, dtype=torch.int32)
            num_rejected = torch.empty(0, dtype=torch.int32)
        aligned_tokens, aligned_sampled, aligned_rejected = align_sampled_payload(
            plan, req_ids, tokens, num_sampled, num_rejected
        )
        # Token row i belongs to plan order i. The handler aligns again by
        # req id, so this batch must name every plan row in that order.
        # Otherwise a second align would treat the already-aligned tensor
        # as if it were still in capture order.
        ordered = _batch_covering_plan(batches, plan)
        self.inner.broadcast(
            aligned_tokens, aligned_sampled, aligned_rejected, ordered
        )

    def _flush_receive(self, plan: RingStepPlan) -> None:
        ordered = (
            _select_pp_batches(self.receives, plan.sampled_request_order)
            if self.receives
            else _empty_pp_batch(plan)
        )
        self.inner.receive(ordered)


def concat_req_frontiers(
    store: LayeredPrefillStateStore,
    req_ids: Sequence[str],
    *,
    expected_group_id: int,
    num_tokens_padded: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Concatenate per-request frontiers in input-batch order, then pad."""
    if not req_ids:
        raise RuntimeError("concat_req_frontiers requires at least one request")
    hidden_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    residual_mode: bool | None = None
    for req_id in req_ids:
        frontier = store.get(req_id)
        if frontier is None:
            raise RuntimeError(
                f"Missing layered activation frontier for request {req_id}"
            )
        if frontier.group_id != expected_group_id:
            raise RuntimeError(
                f"Layered frontier group mismatch for {req_id}: expected "
                f"{expected_group_id}, got {frontier.group_id}"
            )
        hidden_parts.append(frontier.hidden_states)
        has_residual = frontier.residual is not None
        if residual_mode is None:
            residual_mode = has_residual
        elif residual_mode != has_residual:
            raise RuntimeError(
                "Layered mixed frontiers mix residual and residual-free rows"
            )
        if has_residual:
            residual_parts.append(frontier.residual)
    from vllm_ascend.ops.pack_fused_stage import pack_token_rows

    hidden = pack_token_rows(hidden_parts)
    residual = pack_token_rows(residual_parts) if residual_mode else None
    pad = int(num_tokens_padded) - int(hidden.shape[0])
    if pad < 0:
        raise RuntimeError(
            "Layered mixed frontier rows exceed the padded token width"
        )
    if pad > 0:
        hidden = _pad_token_rows(hidden, pad)
        if residual is not None:
            residual = _pad_token_rows(residual, pad)
    return hidden, residual


def _persist_frontier_rows(
    tensor: torch.Tensor,
    start: int,
    end: int,
    *,
    take_storage: bool,
) -> torch.Tensor:
    """Own logical rows for a frontier entry.

    When one request owns the entire activation tensor, take it with
    ``contiguous()`` (zero-copy if already contiguous) so store does not
    allocate a second full copy beside the forward output. Partial slices
    always ``clone()``: a narrow view would keep the parent storage (and any
    trailing pad) alive, and ``contiguous()`` on a contiguous prefix does
    not detach.
    """
    if take_storage and start == 0 and end == int(tensor.shape[0]):
        return tensor.contiguous()
    return tensor[start:end].clone()


def store_req_frontiers(
    store: LayeredPrefillStateStore,
    req_ids: Sequence[str],
    query_start_loc: np.ndarray,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    next_group_id: int,
    *,
    batch_req_ids: Sequence[str] | None = None,
    keep_ids: Sequence[str] | None = None,
) -> None:
    """Slice mixed hidden/residual back into per-request frontiers.

    ``query_start_loc`` is always aligned to ``batch_req_ids`` (or ``req_ids``
    when that is the full batch). ``keep_ids`` persists a subset without
    treating subset order as loc indices.

    Drops any previous frontier for a request before allocating the new one
    so the old copy does not overlap the handover peak.
    """
    loc_ids = list(batch_req_ids) if batch_req_ids is not None else list(req_ids)
    persist = list(keep_ids) if keep_ids is not None else list(req_ids)
    id_to_index = {req_id: index for index, req_id in enumerate(loc_ids)}
    total_rows = int(hidden_states.shape[0])
    if residual is not None and int(residual.shape[0]) != total_rows:
        raise RuntimeError(
            "Layered frontier hidden and residual row counts differ: "
            f"hidden_rows={total_rows} residual_rows={int(residual.shape[0])}"
        )
    # One request owning every row can take the activation storage; mixed
    # or padded batches must copy logical slices only.
    take_storage = len(persist) == 1
    for req_id in persist:
        if req_id not in id_to_index:
            raise RuntimeError(
                f"Layered frontier store missing {req_id} in batch {loc_ids}"
            )
        index = id_to_index[req_id]
        start = int(query_start_loc[index])
        end = int(query_start_loc[index + 1])
        if end <= start:
            raise RuntimeError(
                f"Layered mixed frontier has empty rows for request {req_id}"
            )
        if end > total_rows:
            raise RuntimeError(
                f"Layered frontier slice for {req_id} ends at {end}, "
                f"past hidden rows {total_rows}"
            )
        # Drop the previous group before allocating/taking the new tensor.
        store.pop(req_id)
        # Padding (sequence-parallel / DSA-CP) sits after the last logical
        # token of this physical batch. It belongs to the batch width, not
        # to whichever request happens to be last. A later group may drop a
        # Decode rider and pad to a different width; carrying the old pad
        # rows on the last request makes the restored row count disagree
        # with the new query_start_loc. Persist logical rows only.
        # concat_req_frontiers pads to the current batch width.
        store.put(
            LayeredFrontier(
                req_id=req_id,
                group_id=next_group_id,
                query_len=end - start,
                hidden_states=_persist_frontier_rows(
                    hidden_states, start, end, take_storage=take_storage
                ),
                residual=(
                    None
                    if residual is None
                    else _persist_frontier_rows(
                        residual, start, end, take_storage=take_storage
                    )
                ),
            )
        )


def slice_req_activations(
    req_ids: Sequence[str],
    keep_ids: Sequence[str],
    query_start_loc: np.ndarray,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Take hidden/residual rows for ``keep_ids`` in ``req_ids`` order."""
    keep = set(keep_ids)
    hidden_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    for index, req_id in enumerate(req_ids):
        if req_id not in keep:
            continue
        start = int(query_start_loc[index])
        end = int(query_start_loc[index + 1])
        if end <= start:
            raise RuntimeError(
                f"Layered activation slice is empty for request {req_id}"
            )
        # Views only; a single part uses contiguous(), multiple parts rely on
        # cat to allocate once instead of clone-then-cat.
        hidden_parts.append(hidden_states[start:end])
        if residual is not None:
            residual_parts.append(residual[start:end])
    if not hidden_parts:
        raise RuntimeError("Layered activation slice matched no requests")
    if len(hidden_parts) == 1:
        hidden = hidden_parts[0].contiguous()
        residual_out = (
            None if not residual_parts else residual_parts[0].contiguous()
        )
    else:
        hidden = torch.cat(hidden_parts, dim=0)
        residual_out = (
            torch.cat(residual_parts, dim=0) if residual_parts else None
        )
    return hidden, residual_out


def _pad_token_rows(tensor: torch.Tensor, pad: int) -> torch.Tensor:
    """Pad dimension 0 (token rows). Feature dims, including DSV4 hc, stay put."""
    if pad == 0:
        return tensor
    # F.pad lists (left, right) pairs from the last dimension backward.
    spec = [0, 0] * tensor.dim()
    spec[-1] = pad
    return torch.nn.functional.pad(tensor, spec)


def pad_activation_rows(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    num_tokens_padded: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    pad = int(num_tokens_padded) - int(hidden_states.shape[0])
    if pad < 0:
        raise RuntimeError("Layered activation rows exceed the padded width")
    if pad == 0:
        return hidden_states, residual
    hidden_states = _pad_token_rows(hidden_states, pad)
    if residual is not None:
        residual = _pad_token_rows(residual, pad)
    return hidden_states, residual
