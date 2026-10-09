# SPDX-License-Identifier: Apache-2.0
"""Diagnostic memory probe for the layered Decode->Prefill handover.

Context: the layered run at ``max_model_len``/``max_num_batched_tokens`` =
10240 dies deterministically at the ``torch.npu.synchronize()`` that separates
the Decode and Prefill sub-batches, reporting CANN 507011 with a kernel
``devmm`` fault at a virtual address that is identical across processes. This
module records a per-step balance sheet at that exact point, so a single
crashing run can answer both "was the allocator under pressure" and "which
allocation is missing from ``profile_run``'s estimate".

Off unless ``VLLM_ASCEND_LAYERED_MEM_PROBE=1``.

All probe output goes through ``logger.warning`` on purpose. In worker
processes the ``vllm_ascend`` logger namespace has no handler attached, so
``INFO`` records are dropped by the root logger while ``WARNING`` and above
still reach it. A diagnostic that is off by default must not be silently
discarded in exactly the run it was enabled for.

HARD CONSTRAINT: every call here is a *query*. Nothing in this file allocates
device memory. The fault address reproduces across processes only because the
allocation sequence is deterministic; a probe that allocated would shift that
sequence and turn a deterministic bug into a heisenbug.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger

from vllm_ascend import envs

logger = init_logger(__name__)

BYTES_PER_MIB = 1024 * 1024

# Points inside one layered step. Recording all three attributes the peak to a
# sub-step in time, not merely to a step.
PROBE_TAG_DECODE_END = "d_end"
PROBE_TAG_HANDOVER = "dp_boundary"
PROBE_TAG_PREFILL_END = "p_end"

_STATE_ATTR = "_layered_mem_probe_state"


@dataclass
class MemProbeState:
    """Per-runner probe counters. Avoids module-level mutable globals."""

    step: int = 0
    peak_bytes_seen: int = 0
    segment_dump_path: str | None = None
    announced: bool = False
    rank: int = -1


def probe_enabled() -> bool:
    return bool(envs.VLLM_ASCEND_LAYERED_MEM_PROBE)


def _segment_dump_enabled() -> bool:
    return bool(envs.VLLM_ASCEND_LAYERED_MEM_PROBE_SNAPSHOT)


def _get_state(runner: Any) -> MemProbeState:
    state = getattr(runner, _STATE_ATTR, None)
    if state is None:
        state = MemProbeState(rank=_current_rank())
        setattr(runner, _STATE_ATTR, state)
    return state


def _current_rank() -> int:
    """Tag every line with its rank: with TP=8 the balance sheets must not mix."""
    try:
        return int(torch.npu.current_device())
    except Exception:  # noqa: BLE001 - fall back to launcher-provided rank
        for name in ("RANK", "LOCAL_RANK"):
            value = os.getenv(name)
            if value is not None and value.lstrip("-").isdigit():
                return int(value)
        return -1


def _frontier_bytes(store: Any) -> tuple[int, int]:
    """Device bytes held by the cross-group frontier store.

    This is the allocation ``profile_run`` never observes: frontiers live
    across scheduler steps, while profiling runs a single plain forward that
    never enters the layered store/restore path.
    """
    total_bytes = 0
    num_frontiers = 0
    by_req_id = getattr(store, "by_req_id", None)
    if not by_req_id:
        return 0, 0
    for frontier in by_req_id.values():
        num_frontiers += 1
        for name in ("hidden_states", "residual"):
            tensor = getattr(frontier, name, None)
            if isinstance(tensor, torch.Tensor):
                total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes, num_frontiers


def _allocator_stats() -> dict[str, int]:
    stats = torch.npu.memory_stats()
    return {
        # Non-zero means the allocator had to free cached blocks to satisfy a
        # request: the single most direct evidence of memory pressure.
        "retries": int(stats.get("num_alloc_retries", 0)),
        "ooms": int(stats.get("num_ooms", 0)),
        "segments": int(stats.get("segment.all.current", 0)),
        "oversize_segments": int(stats.get("oversize_segments.current", 0)),
        "allocated": int(stats.get("allocated_bytes.all.current", 0)),
        "peak": int(stats.get("allocated_bytes.all.peak", 0)),
        "reserved": int(stats.get("reserved_bytes.all.current", 0)),
        "inactive_split": int(stats.get("inactive_split_bytes.all.current", 0)),
    }


def record(tag: str, runner: Any) -> None:
    """Emit one balance-sheet line for ``tag`` inside the current step."""
    if not probe_enabled():
        return
    try:
        state = _get_state(runner)
        if not state.announced:
            state.announced = True
            logger.warning(
                "Layered mem probe enabled (query-only, segment_dump=%s)",
                _segment_dump_enabled(),
            )
        stats = _allocator_stats()
        free_bytes, total_bytes = torch.npu.mem_get_info()
        frontier_bytes, num_frontiers = _frontier_bytes(
            getattr(runner, "layered_prefill_state", None)
        )
        if tag == PROBE_TAG_HANDOVER:
            state.step += 1
        logger.warning(
            "MEMPROBE rank=%d step=%d tag=%s allocated=%dMiB peak=%dMiB "
            "reserved=%dMiB free=%dMiB total=%dMiB retries=%d ooms=%d "
            "segments=%d oversize_segments=%d inactive_split=%dMiB "
            "frontier=%dMiB num_frontiers=%d",
            state.rank,
            state.step,
            tag,
            stats["allocated"] // BYTES_PER_MIB,
            stats["peak"] // BYTES_PER_MIB,
            stats["reserved"] // BYTES_PER_MIB,
            free_bytes // BYTES_PER_MIB,
            total_bytes // BYTES_PER_MIB,
            stats["retries"],
            stats["ooms"],
            stats["segments"],
            stats["oversize_segments"],
            stats["inactive_split"] // BYTES_PER_MIB,
            frontier_bytes // BYTES_PER_MIB,
            num_frontiers,
        )
        if stats["peak"] > state.peak_bytes_seen:
            state.peak_bytes_seen = stats["peak"]
            logger.warning(
                "MEMPROBE NEWPEAK step=%d tag=%s peak_bytes=%d frontier_bytes=%d",
                state.step,
                tag,
                stats["peak"],
                frontier_bytes,
            )
        if _segment_dump_enabled():
            _dump_segments(state, tag)
    except Exception as exc:  # noqa: BLE001 - a probe must never break a run
        logger.warning("MEMPROBE failed at tag=%s: %r", tag, exc)


def _dump_segments(state: MemProbeState, tag: str) -> None:
    """Append the allocator segment table so the faulting VA can be matched.

    dmesg reports the faulting virtual address. If it falls inside a segment
    listed here we learn that segment's size and pool; if it falls outside
    every segment, the allocator had released the range while a captured
    aclgraph still held the baked address.
    """
    if state.segment_dump_path is None:
        # Use the device index already resolved for this state, not RANK from
        # the environment: worker processes do not set RANK, so every one of
        # them would fall back to 0 and overwrite the others' dump in the
        # shared /tmp. The pid keeps a stale file from a previous run from
        # being mistaken for this one's.
        state.segment_dump_path = (
            f"/tmp/memprobe_segments_rank{state.rank}_pid{os.getpid()}.jsonl"
        )
        logger.warning("MEMPROBE segment dumps -> %s", state.segment_dump_path)
    rows = [
        {
            "address": int(segment.get("address", 0)),
            "total_size": int(segment.get("total_size", 0)),
            "allocated_size": int(segment.get("allocated_size", 0)),
            "segment_type": segment.get("segment_type", ""),
        }
        for segment in torch.npu.memory_snapshot()
    ]
    with open(state.segment_dump_path, "a") as handle:
        handle.write(json.dumps({"step": state.step, "tag": tag, "segments": rows}) + "\n")
