# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/pp_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Materialize the PP sampled-token broadcast buffer at construction time.

``PPHandler`` gives the sampled-token broadcast its own communicator, a sibling
of the PP ``device_group`` (``make_sibling_device_group(group_desc="pp_broadcast")``),
so the broadcast does not serialize with the inter-stage hidden-state p2p.  HCCL
allocates that communicator's buffer lazily, on first use, and sizes it from
``HCCL_BUFFSIZE`` rather than from the payload: a ~128-byte broadcast was
observed asking the driver for 2,148,532,224 B == 2 x 1024 MiB + 1 MiB.

Nothing in ``profile_run`` touches this communicator -- the profiling forward
runs with ``skip_attn=True`` and no PP collective -- so the allocation used to
land *after* memory profiling had already handed every remaining byte to the KV
cache.  The first real mixed step then died in ``HcclBroadcast`` with
``Memory_Allocation_Failure(EL0004)``, and because the failure is asynchronous
the traceback surfaced at an unrelated ``.item()`` sync point.

The fix states the obvious invariant instead of reserving a guessed margin:

    a communicator is not fully constructed until its buffer exists.

``non_torch_increase`` is measured as ``after_profile - before_create`` where
``before_create`` is the worker's ``init_snapshot`` (taken in ``_init_device``,
before the model runner and therefore before ``PPHandler``), so a buffer
materialized here is charged to ``non_kv_cache_memory`` as an ordinary non-torch
allocation -- exactly the category ``memory_profiling`` documents as "memory
used by NCCL".  No magic number enters the budget: the cost is whatever the
driver actually took.

PP=1 never constructs a ``PPHandler`` and pays nothing.  PP>1 always reaches
``receive``/``broadcast`` on every step, independent of async scheduling and of
layered prefill, so there is no configuration that pays for an unused buffer and
no reason to gate this behind an env var -- a collective that some ranks skip and
others enter would deadlock.

Failures here are deliberately not swallowed: if the buffer cannot be
materialized, the memory plan that follows would be wrong, and a loud failure at
startup beats an OOM at the first mixed step.
"""

import torch
from vllm.logger import logger
from vllm.v1.worker.gpu.pp_utils import PPHandler

_pp_broadcast_buffer_patch_applied = False
_original_pp_handler_init = None


def _driver_bytes_in_use() -> int | None:
    """Driver-side bytes in use, or None when unavailable.

    HCCL buffers bypass the torch caching allocator, so ``memory_allocated`` and
    ``memory_reserved`` cannot see them; only the driver's own accounting can.
    """
    try:
        free_bytes, total_bytes = torch.npu.mem_get_info()
    except Exception as exc:  # pragma: no cover - depends on driver/runtime
        logger.debug("Could not read NPU memory info: %s", exc)
        return None
    return int(total_bytes) - int(free_bytes)


def materialize_pp_broadcast_buffer(handler: PPHandler) -> int | None:
    """Force HCCL to allocate the ``pp_broadcast`` buffer, and report its cost.

    Uses the widest payload the steady state can produce, on the same stream the
    live path uses, so the measurement stands even if a future HCCL were to size
    its buffer from the message rather than from ``HCCL_BUFFSIZE``.

    Returns the driver-side byte delta, or None if it could not be measured.
    """
    device = handler.device
    max_num_reqs = int(len(handler.req_idx_gen_np))
    sample_width = int(handler.max_sample_len)

    before = _driver_bytes_in_use()

    # Same dtypes and shapes as _enter_broadcast / _enter_receive, at their
    # maximum: (num_reqs, sample_width) int64 tokens plus a (2, num_reqs) int32
    # [num_sampled, num_rejected] pair.
    sampled_tokens = torch.empty(
        (max_num_reqs, sample_width), dtype=torch.int64, device=device
    )
    combined = torch.empty((2, max_num_reqs), dtype=torch.int32, device=device)

    with torch.npu.stream(handler.broadcast_stream):
        handler.broadcast_stream.wait_stream(handler.main_stream)
        torch.distributed.broadcast(
            sampled_tokens, src=handler.last_rank, group=handler.broadcast_group
        )
        torch.distributed.broadcast(
            combined, src=handler.last_rank, group=handler.broadcast_group
        )

    # The allocation is what we are after, not the transfer: synchronize so it
    # has really happened before the caller goes on to plan memory.
    torch.npu.synchronize()
    del sampled_tokens, combined

    after = _driver_bytes_in_use()
    if before is None or after is None:
        logger.info(
            "PP broadcast buffer materialized (pp_broadcast communicator); "
            "driver-side cost not measurable on this runtime"
        )
        return None

    delta = after - before
    logger.info(
        "PP broadcast buffer materialized: %.2f GiB taken by the pp_broadcast "
        "HCCL communicator, now charged to non-torch memory instead of "
        "surfacing at the first mixed step (payload %sx%s int64 + 2x%s int32)",
        delta / (1 << 30),
        max_num_reqs,
        sample_width,
        max_num_reqs,
    )
    if delta <= 0:
        # Either the buffer predates this call or the driver reading is not
        # attributable. Worth noticing: the KV budget is then still blind to it.
        logger.warning(
            "PP broadcast buffer materialization measured no driver-side growth "
            "(delta=%d B). If the first mixed step still fails in HcclBroadcast, "
            "this allocation is happening somewhere this measurement cannot see.",
            delta,
        )
    return delta


def _apply_pp_broadcast_buffer_patch() -> None:
    global _pp_broadcast_buffer_patch_applied
    global _original_pp_handler_init
    if _pp_broadcast_buffer_patch_applied:
        return

    _original_pp_handler_init = PPHandler.__init__

    # Signature-independent on purpose: everything needed is read back off the
    # constructed handler, so an upstream signature change cannot silently
    # bypass this.
    def _patched_init(self, *args, **kwargs):
        _original_pp_handler_init(self, *args, **kwargs)
        materialize_pp_broadcast_buffer(self)

    PPHandler.__init__ = _patched_init
    _pp_broadcast_buffer_patch_applied = True


_apply_pp_broadcast_buffer_patch()
