"""Unit tests for the layered Decode->Prefill memory probe."""

from types import SimpleNamespace

import torch

from vllm.v1.core.layered_prefill import LayeredFrontier, LayeredPrefillStateStore
from vllm_ascend.worker.v2 import _mem_probe


def _store_with(*frontiers: LayeredFrontier) -> LayeredPrefillStateStore:
    store = LayeredPrefillStateStore()
    for frontier in frontiers:
        store.put(frontier)
    return store


def test_frontier_bytes_empty_store():
    assert _mem_probe._frontier_bytes(LayeredPrefillStateStore()) == (0, 0)
    assert _mem_probe._frontier_bytes(None) == (0, 0)


def test_frontier_bytes_counts_hidden_and_residual():
    hidden = torch.zeros(8, 2, 4, dtype=torch.float32)  # 8*2*4*4 = 256 bytes
    residual = torch.zeros(8, 2, 4, dtype=torch.float32)
    store = _store_with(
        LayeredFrontier(
            req_id="p0",
            group_id=1,
            query_len=8,
            hidden_states=hidden,
            residual=residual,
        )
    )
    assert _mem_probe._frontier_bytes(store) == (512, 1)


def test_frontier_bytes_skips_absent_residual():
    """DeepSeek-V4 recreates residual per layer, so it is not in the frontier."""
    store = _store_with(
        LayeredFrontier(
            req_id="p0",
            group_id=1,
            query_len=8,
            hidden_states=torch.zeros(8, 2, 4, dtype=torch.float32),
            residual=None,
        )
    )
    assert _mem_probe._frontier_bytes(store) == (256, 1)


def test_frontier_bytes_sums_concurrent_requests():
    tensor = torch.zeros(4, 4, dtype=torch.float32)  # 64 bytes each
    store = _store_with(
        LayeredFrontier(
            req_id="p0", group_id=1, query_len=4, hidden_states=tensor, residual=None
        ),
        LayeredFrontier(
            req_id="p1", group_id=2, query_len=4, hidden_states=tensor, residual=None
        ),
    )
    assert _mem_probe._frontier_bytes(store) == (128, 2)


def test_record_is_noop_when_disabled(monkeypatch):
    """Default is off: the probe must not touch the runner at all."""
    monkeypatch.setattr(_mem_probe, "probe_enabled", lambda: False)
    runner = SimpleNamespace(layered_prefill_state=LayeredPrefillStateStore())
    _mem_probe.record(_mem_probe.PROBE_TAG_HANDOVER, runner)
    assert not hasattr(runner, _mem_probe._STATE_ATTR)


def test_record_swallows_backend_errors(monkeypatch):
    """A diagnostic must never be able to break the run it observes."""
    monkeypatch.setattr(_mem_probe, "probe_enabled", lambda: True)
    monkeypatch.setattr(_mem_probe, "_segment_dump_enabled", lambda: False)

    def _boom():
        raise RuntimeError("allocator stats unavailable")

    monkeypatch.setattr(_mem_probe, "_allocator_stats", _boom)
    runner = SimpleNamespace(layered_prefill_state=LayeredPrefillStateStore())
    _mem_probe.record(_mem_probe.PROBE_TAG_HANDOVER, runner)


def test_step_counter_advances_only_on_handover(monkeypatch):
    """d_end / p_end share the step that dp_boundary opened."""
    monkeypatch.setattr(_mem_probe, "probe_enabled", lambda: True)
    monkeypatch.setattr(_mem_probe, "_segment_dump_enabled", lambda: False)
    monkeypatch.setattr(
        _mem_probe,
        "_allocator_stats",
        lambda: {
            "retries": 0,
            "ooms": 0,
            "segments": 1,
            "oversize_segments": 0,
            "allocated": 0,
            "peak": 0,
            "reserved": 0,
            "inactive_split": 0,
        },
    )
    monkeypatch.setattr(torch.npu, "mem_get_info", lambda: (1, 2), raising=False)
    runner = SimpleNamespace(layered_prefill_state=LayeredPrefillStateStore())

    _mem_probe.record(_mem_probe.PROBE_TAG_DECODE_END, runner)
    assert getattr(runner, _mem_probe._STATE_ATTR).step == 0
    _mem_probe.record(_mem_probe.PROBE_TAG_HANDOVER, runner)
    assert getattr(runner, _mem_probe._STATE_ATTR).step == 1
    _mem_probe.record(_mem_probe.PROBE_TAG_PREFILL_END, runner)
    assert getattr(runner, _mem_probe._STATE_ATTR).step == 1


def test_probe_state_is_per_runner(monkeypatch):
    """State lives on the runner, not in module globals."""
    monkeypatch.setattr(_mem_probe, "probe_enabled", lambda: True)
    monkeypatch.setattr(_mem_probe, "_segment_dump_enabled", lambda: False)
    monkeypatch.setattr(
        _mem_probe,
        "_allocator_stats",
        lambda: {
            "retries": 0,
            "ooms": 0,
            "segments": 0,
            "oversize_segments": 0,
            "allocated": 0,
            "peak": 0,
            "reserved": 0,
            "inactive_split": 0,
        },
    )
    monkeypatch.setattr(torch.npu, "mem_get_info", lambda: (1, 2), raising=False)
    first = SimpleNamespace(layered_prefill_state=LayeredPrefillStateStore())
    second = SimpleNamespace(layered_prefill_state=LayeredPrefillStateStore())

    _mem_probe.record(_mem_probe.PROBE_TAG_HANDOVER, first)
    _mem_probe.record(_mem_probe.PROBE_TAG_HANDOVER, first)
    _mem_probe.record(_mem_probe.PROBE_TAG_HANDOVER, second)

    assert getattr(first, _mem_probe._STATE_ATTR).step == 2
    assert getattr(second, _mem_probe._STATE_ATTR).step == 1
