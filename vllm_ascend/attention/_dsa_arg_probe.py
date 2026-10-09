# SPDX-License-Identifier: Apache-2.0
"""Diagnostic dump of the arguments handed to the sparse-attention kernel.

CANN names ``SparseAttnSharedkv`` as the faulting kernel for the ACL 507011 /
MTE out-of-range seen at ``max_model_len``=10240 under FULL_DECODE aclgraph.
That kernel computes KV addresses on device from ``ori_block_table`` and the
sequence lengths, so a bad address must come from those arguments -- but the
Python side never records what they actually were.

Two static hypotheses were already refuted by reading the code: graph padding
rows carry zeroed block tables (the gather kernel stores 0 explicitly), and
``seq_lens`` padding rows are zeroed in ``attn_utils.build_attn_metadata``.
This probe exists to replace further guessing with the real arguments from the
step that faults.

Enabled by ``VLLM_ASCEND_DSA_ARG_PROBE=1``; off by default.

HARD CONSTRAINTS
  * Read-only. Nothing here mutates a tensor the kernel will consume.
  * No device allocation and no device->host copies of large tensors; the
    crash reproduces only in a narrow configuration and a probe that perturbs
    memory layout or stream timing could hide it.
  * Logs at WARNING: in worker processes the ``vllm_ascend`` logger namespace
    has no handler, so INFO records are dropped by the root logger.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.logger import init_logger

from vllm_ascend import envs

logger = init_logger(__name__)

# Record every call, unconditionally.
#
# Earlier versions capped the dump count, skipped warmup batches, and planned
# to de-duplicate by (sub-batch, layer, shape). Each of those filters was a
# guess about which data would matter, and two of them cost a run apiece: a
# shared budget went entirely to warmup, then entirely to the Prefill
# sub-batch, while the fault happens during Decode. The warmup batches are not
# noise either -- they are the arguments baked into the captured aclgraph,
# which is exactly the comparison "graph replay is necessary" points at.
#
# The volume does not justify filtering: the crash lands within ~4 seconds of
# the benchmark starting, so a run yields on the order of thousands of lines
# against a launcher.out that is already ~1900 lines and CANN logs of 1.7 GB.
# The probe is query-only and allocates nothing, so line count does not change
# what it observes. Decide what matters in the analyzer, after seeing the data.
_dumps_emitted = 0


def probe_enabled() -> bool:
    return bool(envs.VLLM_ASCEND_DSA_ARG_PROBE)


def _tensor_summary(name: str, tensor: Any) -> str:
    """Shape/dtype/device-pointer summary. Never copies tensor contents."""
    if tensor is None:
        return f"{name}=None"
    if not isinstance(tensor, torch.Tensor):
        return f"{name}=<{type(tensor).__name__}>"
    return (
        f"{name}(shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"ptr=0x{tensor.data_ptr():x} contig={tensor.is_contiguous()})"
    )


def _row_stats(block_table: torch.Tensor, max_rows: int = 12) -> str:
    """Per-row max block id and nonzero count, for up to ``max_rows`` rows.

    A padding row that was correctly zeroed reads max=0 nz=0. A row holding a
    stale block table reads a nonzero max. This is the single comparison the
    probe exists to make, so the reduction runs on device and only the small
    result crosses to host.
    """
    if block_table is None or not isinstance(block_table, torch.Tensor):
        return "rows=<none>"
    if block_table.ndim < 2:
        return f"rows=<ndim {block_table.ndim}>"
    rows = min(int(block_table.shape[0]), max_rows)
    # Block tables are int32, and NPU's aclnnMaxDim rejects DT_INT32. Reduce in
    # int64, which is on its supported list, rather than pulling rows to host.
    head = block_table[:rows].to(torch.int64)
    row_max = head.max(dim=1).values
    row_nz = (head != 0).sum(dim=1)
    max_list = row_max.tolist()
    nz_list = row_nz.tolist()
    parts = [f"r{i}(max={max_list[i]},nz={nz_list[i]})" for i in range(rows)]
    suffix = "" if rows == int(block_table.shape[0]) else f" ...+{int(block_table.shape[0]) - rows}"
    return "rows=[" + " ".join(parts) + "]" + suffix


def record_sparse_attn_args(
    *,
    layer_name: str,
    block_table: Any,
    seqused_kv: Any,
    cu_seqlens_q: Any,
    query: Any,
    ori_win_left: Any,
    has_prefill: Any,
    sparse_indices: Any = None,
    cmp_block_table: Any = None,
) -> None:
    """Emit one line describing what the kernel is about to receive."""
    if not probe_enabled():
        return
    global _dumps_emitted
    kind = "prefill" if has_prefill else "decode"
    try:
        _dumps_emitted += 1
        seq_summary = ""
        if isinstance(seqused_kv, torch.Tensor) and seqused_kv.numel():
            # Padding rows are expected to read 0 here; a nonzero tail would
            # mean the zeroing in build_attn_metadata did not reach this path.
            head = seqused_kv[: min(12, seqused_kv.numel())].tolist()
            seq_summary = (
                f" seqused_kv_head={head} nonzero={int((seqused_kv != 0).sum())}"
                f"/{int(seqused_kv.numel())}"
            )
        logger.warning(
            "DSAPROBE dump=%d kind=%s layer=%s has_prefill=%s "
            "ori_win_left=%s | %s | %s | %s | %s | %s%s | block_table_%s",
            _dumps_emitted,
            kind,
            layer_name,
            has_prefill,
            ori_win_left,
            _tensor_summary("q", query),
            _tensor_summary("block_table", block_table),
            _tensor_summary("seqused_kv", seqused_kv),
            _tensor_summary("cu_seqlens_q", cu_seqlens_q),
            _tensor_summary("sparse_indices", sparse_indices),
            seq_summary,
            _row_stats(block_table),
        )
        if cmp_block_table is not None:
            logger.warning(
                "DSAPROBE dump=%d kind=%s layer=%s cmp | %s | cmp_block_table_%s",
                _dumps_emitted,
                kind,
                layer_name,
                _tensor_summary("cmp_block_table", cmp_block_table),
                _row_stats(cmp_block_table),
            )
    except Exception as exc:  # noqa: BLE001 - a probe must not break the run
        logger.warning("DSAPROBE failed: %r", exc)
