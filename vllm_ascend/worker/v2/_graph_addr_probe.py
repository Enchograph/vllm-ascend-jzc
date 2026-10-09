# SPDX-License-Identifier: Apache-2.0
"""Compare tensor addresses baked at aclgraph capture against replay steps.

Why this exists: the ACL 507011 / MTE out-of-range fault at
``max_model_len``=10240 only happens under FULL_DECODE aclgraph, and the
faulting kernel (``SparseAttnSharedkv``) derives KV addresses on device from
``seqused_kv`` and the block table. A probe placed at the kernel call site
recorded nothing for the faulting step, because graph replay does not execute
Python at all -- it replays instructions holding the addresses captured during
warmup.

Capture-time dumps showed four different ``seq_lens`` addresses for the four
captured batch sizes, and two of those sizes shared one address. That is not
what a slice of a persistent buffer looks like, so the question is whether the
address a graph baked is still where ``seq_lens`` lives when that graph is
replayed. This probe answers it by logging the address on every step --
capture and replay alike -- from ``prepare_attn``, which runs outside the
graph and therefore executes on replay steps too.

Enabled by ``VLLM_ASCEND_GRAPH_ADDR_PROBE=1``; off by default.

HARD CONSTRAINTS
  * Reads ``data_ptr()`` and shapes only. No device allocation, no copies, no
    synchronisation -- the fault reproduces only in a narrow configuration and
    is deterministic precisely because the allocation sequence is fixed.
  * Logs at WARNING: worker processes attach no handler to the
    ``vllm_ascend`` logger namespace, so INFO records are dropped.
  * No sampling, no cap, no de-duplication. Earlier probes in this
    investigation lost two runs to filters that discarded exactly the data
    that mattered.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.logger import init_logger

from vllm_ascend import envs

logger = init_logger(__name__)


def probe_enabled() -> bool:
    return bool(envs.VLLM_ASCEND_GRAPH_ADDR_PROBE)


def _addr(tensor: Any) -> str:
    """Address and shape of a tensor, or a marker when there is none."""
    if not isinstance(tensor, torch.Tensor):
        return "none"
    return f"0x{tensor.data_ptr():x}@{tuple(tensor.shape)}"


def record_step_addrs(
    *,
    input_batch: Any,
    block_tables: Any,
    for_capture: bool,
    cudagraph_mode: Any,
    num_reqs: int,
    num_reqs_padded: int,
) -> None:
    """Log the addresses this step hands to attention metadata.

    ``for_capture`` separates the graphs being recorded from the steps that
    replay them; comparing the two is the whole point.
    """
    if not probe_enabled():
        return
    try:
        seq_lens = getattr(input_batch, "seq_lens", None)
        positions = getattr(input_batch, "positions", None)
        query_start_loc = getattr(input_batch, "query_start_loc", None)
        # Block tables come as a tuple, one per KV cache group; the main table
        # is the first and is the one the sparse kernel indexes.
        main_block_table = None
        if isinstance(block_tables, (tuple, list)) and block_tables:
            main_block_table = block_tables[0]
        elif isinstance(block_tables, torch.Tensor):
            main_block_table = block_tables

        logger.warning(
            "GRAPHADDR phase=%s cg_mode=%s num_reqs=%d num_reqs_padded=%d "
            "seq_lens=%s block_table=%s positions=%s query_start_loc=%s",
            "capture" if for_capture else "step",
            getattr(cudagraph_mode, "name", cudagraph_mode),
            num_reqs,
            num_reqs_padded,
            _addr(seq_lens),
            _addr(main_block_table),
            _addr(positions),
            _addr(query_start_loc),
        )
    except Exception as exc:  # noqa: BLE001 - a probe must not break the run
        logger.warning("GRAPHADDR failed: %r", exc)
