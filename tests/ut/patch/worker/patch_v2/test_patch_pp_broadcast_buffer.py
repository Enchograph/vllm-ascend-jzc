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
"""The pp_broadcast HCCL buffer must be paid for while PPHandler is built.

Guards the invariant the patch exists for: by the time ``PPHandler.__init__``
returns, the sibling communicator's buffer has really been allocated, so the KV
budget that is computed afterwards sees it as non-torch memory instead of
meeting it at the first mixed step.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from vllm.v1.worker.gpu.pp_utils import PPHandler

from vllm_ascend.patch.worker.patch_v2 import patch_pp_broadcast_buffer as pp_buf


class _FakeStream:
    """Stand-in for the broadcast stream; records that it was waited on."""

    def __init__(self):
        self.waited_on = []

    def wait_stream(self, other):
        self.waited_on.append(other)


def _fake_handler(*, max_num_reqs=8, sample_width=1):
    return SimpleNamespace(
        device=torch.device("cpu"),
        req_idx_gen_np=np.zeros(max_num_reqs, dtype=np.int32),
        max_sample_len=sample_width,
        broadcast_stream=_FakeStream(),
        main_stream=object(),
        last_rank=3,
        broadcast_group=object(),
    )


def _npu_stub(mem_readings):
    """torch.npu with a scripted mem_get_info sequence: [(free, total), ...]."""
    npu = mock.MagicMock()
    npu.mem_get_info.side_effect = list(mem_readings)
    return npu


def test_materialize_issues_both_broadcasts_at_max_payload():
    """Both collectives go to the sibling group, shaped for the widest step."""
    handler = _fake_handler(max_num_reqs=8, sample_width=2)
    total = 64 << 30
    # 2 GiB disappears from driver-side free memory across the materialization.
    npu = _npu_stub([(total - (10 << 30), total), (total - (12 << 30), total)])

    calls = []

    def _record_broadcast(tensor, src=None, group=None):
        calls.append((tuple(tensor.shape), tensor.dtype, src, group))

    with (
        mock.patch.object(torch, "npu", npu),
        mock.patch.object(torch.distributed, "broadcast", _record_broadcast),
    ):
        delta = pp_buf.materialize_pp_broadcast_buffer(handler)

    assert delta == 2 << 30
    # Widest payload the steady state can produce, not a 1-element probe:
    # (num_reqs, sample_width) int64 tokens, then (2, num_reqs) int32 counters.
    assert calls == [
        ((8, 2), torch.int64, 3, handler.broadcast_group),
        ((2, 8), torch.int32, 3, handler.broadcast_group),
    ]
    # Ordered behind the main stream, as the live broadcast path is.
    assert handler.broadcast_stream.waited_on == [handler.main_stream]
    # The allocation must have landed before the caller plans memory.
    npu.synchronize.assert_called_once_with()


def test_materialize_warns_when_no_driver_growth_is_seen():
    """A zero delta means the KV budget is still blind: say so."""
    handler = _fake_handler()
    total = 64 << 30
    flat = [(total - (10 << 30), total), (total - (10 << 30), total)]

    with (
        mock.patch.object(torch, "npu", _npu_stub(flat)),
        mock.patch.object(torch.distributed, "broadcast", lambda *a, **k: None),
        mock.patch.object(pp_buf.logger, "warning") as warning,
    ):
        delta = pp_buf.materialize_pp_broadcast_buffer(handler)

    assert delta == 0
    warning.assert_called_once()
    assert "HcclBroadcast" in warning.call_args[0][0]


def test_materialize_survives_unreadable_driver_memory():
    """Measurement is diagnostics; failing to read it must not block startup."""
    handler = _fake_handler()
    npu = mock.MagicMock()
    npu.mem_get_info.side_effect = RuntimeError("no driver here")

    with (
        mock.patch.object(torch, "npu", npu),
        mock.patch.object(torch.distributed, "broadcast", lambda *a, **k: None),
    ):
        assert pp_buf.materialize_pp_broadcast_buffer(handler) is None

    npu.synchronize.assert_called_once_with()


def test_patch_is_applied_and_idempotent():
    """Importing the module wraps __init__ once, however often it is applied."""
    assert pp_buf._pp_broadcast_buffer_patch_applied
    wrapped = PPHandler.__init__
    assert wrapped is not pp_buf._original_pp_handler_init

    pp_buf._apply_pp_broadcast_buffer_patch()

    assert PPHandler.__init__ is wrapped
    assert pp_buf._original_pp_handler_init is not wrapped


def test_wrapped_init_materializes_after_the_original_runs():
    """Order matters: the group must exist before we can make its buffer real."""
    events = []

    def _fake_original(self, *args, **kwargs):
        events.append("init")

    with (
        mock.patch.object(pp_buf, "_original_pp_handler_init", _fake_original),
        mock.patch.object(
            pp_buf,
            "materialize_pp_broadcast_buffer",
            lambda handler: events.append("materialize"),
        ),
    ):
        PPHandler.__init__(SimpleNamespace())

    assert events == ["init", "materialize"]
