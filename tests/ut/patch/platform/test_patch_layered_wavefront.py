from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import LayeredPrefillExtensions
from vllm_ascend.patch.platform import patch_layered_wavefront as wavefront


def _policy(*, layers=48, pp=4, enabled=True, require_pd_mixed=True):
    return SimpleNamespace(
        enabled=enabled,
        num_hidden_layers=layers,
        pipeline_parallel_size=pp,
        config=SimpleNamespace(require_pd_mixed=require_pd_mixed),
        initialize_request=lambda request: None,
    )


def _vllm_config(**layered_cfg):
    return SimpleNamespace(additional_config={"scheduler_config": {"layered_prefill_config": dict(layered_cfg)}})


def _prompt(name, *, num_groups=4, group_id=0, in_flight=0):
    return SimpleNamespace(
        request_id=name,
        pending=True,
        layered_prefill_enabled=True,
        layered_prefill_num_groups=num_groups,
        layered_prefill_group_id=group_id,
        num_in_flight_tokens=in_flight,
    )


class _FakeScheduler:
    """Just the scheduler surface the wavefront policy reads."""

    def __init__(self, *, depth, running=(), waiting=(), pp=4):
        self.layered_prefill_policy = _policy(pp=pp)
        self.vllm_config = _vllm_config(max_concurrent_layered_prefills=depth)
        self.running = list(running)
        self.waiting = list(waiting)
        self.skipped_waiting = []

    @staticmethod
    def _is_layered_prefill_pending(request):
        return request.pending

    @staticmethod
    def _is_layered_request_eligible(request):
        return True

    @staticmethod
    def _has_unsampled_final_prefill():
        return False

    _layered_wavefront_depth = wavefront._layered_wavefront_depth
    _layered_wavefront_state = wavefront._layered_wavefront_state
    _next_layered_admission_candidate = wavefront._next_layered_admission_candidate
    _get_layered_wavefront_candidate = wavefront._get_layered_wavefront_candidate


@pytest.mark.parametrize(
    "num_groups, expected_owners",
    [(4, [0, 1, 2, 3]), (8, [0, 0, 1, 1, 2, 2, 3, 3])],
)
def test_owner_of_group_assigns_every_group_to_one_stage(num_groups, expected_owners):
    policy = _policy()
    owners = [wavefront.owner_of_group(policy, num_groups, g) for g in range(num_groups)]
    assert owners == expected_owners
    assert wavefront.owner_of_group(policy, num_groups, num_groups) is None
    assert wavefront.owner_of_group(policy, num_groups, -1) is None


def test_in_flight_group_owner_trails_the_group_cursor():
    """The cursor advances at schedule time; the worker holds the group behind."""
    policy = _policy()
    request = _prompt("a", group_id=2)
    assert wavefront.group_owner(policy, request) == 2
    assert wavefront.in_flight_group_owner(policy, request) == 1


@pytest.mark.parametrize("configured, pp, expected", [(4, 1, 1), (4, 4, 4), (2, 4, 2), (8, 4, 4), (1, 4, 1)])
def test_wavefront_depth_is_capped_by_pp_size(configured, pp, expected):
    assert _FakeScheduler(depth=configured, pp=pp)._layered_wavefront_depth() == expected


def test_extensions_parse_and_default():
    assert LayeredPrefillExtensions().max_concurrent_layered_prefills == 1
    ext = LayeredPrefillExtensions.from_vllm_config(_vllm_config(max_concurrent_layered_prefills=4))
    assert ext.max_concurrent_layered_prefills == 4
    assert LayeredPrefillExtensions.from_vllm_config(SimpleNamespace(additional_config=None)).max_concurrent_layered_prefills == 1
    assert LayeredPrefillExtensions().skip_relay_stages is False
    assert LayeredPrefillExtensions({"skip_relay_stages": True}).skip_relay_stages is True


def test_extensions_reject_negative_concurrency():
    with pytest.raises(ValueError):
        LayeredPrefillExtensions({"max_concurrent_layered_prefills": -1})


def test_admission_waits_while_stage_zero_holds_a_group():
    """A new prompt enters on stage 0; admitting it now would serialize there."""
    first = _prompt("first", group_id=1, in_flight=8)  # group 0 on stage 0
    second = _prompt("second")
    scheduler = _FakeScheduler(depth=4, running=[first], waiting=[second])
    assert scheduler._get_layered_wavefront_candidate() is None


def test_admission_opens_once_stage_zero_drains():
    first = _prompt("first", group_id=2, in_flight=8)  # group 1 on stage 1
    second = _prompt("second")
    scheduler = _FakeScheduler(depth=4, running=[first], waiting=[second])
    assert scheduler._get_layered_wavefront_candidate() is second
    _, busy, active = scheduler._layered_wavefront_state()
    assert busy == {1}
    assert active == 1


def test_ready_prompt_on_an_idle_stage_goes_first():
    busy = _prompt("busy", group_id=3, in_flight=8)  # group 2 on stage 2
    blocked = _prompt("blocked", group_id=2)  # next group on busy stage 2
    free = _prompt("free", group_id=1)  # next group on idle stage 1
    scheduler = _FakeScheduler(depth=4, running=[busy, blocked, free])
    assert scheduler._get_layered_wavefront_candidate() is free


def test_admission_stops_at_the_depth_cap():
    first = _prompt("first", group_id=2, in_flight=8)
    second = _prompt("second", group_id=1, in_flight=8)
    third = _prompt("third")
    scheduler = _FakeScheduler(depth=2, running=[first, second], waiting=[third])
    assert scheduler._get_layered_wavefront_candidate() is None


def _patched_scheduler_cls():
    """A stand-in with the three upstream methods the patch wraps."""

    class Base:
        _get_layered_prefill_candidate_calls = 0

        def _get_layered_prefill_candidate(self):
            type(self)._get_layered_prefill_candidate_calls += 1
            return "serial"

        def _attach_fused_mixed_decodes(self, scheduler_output, candidate):
            # Mirrors the upstream group-0 rider walk.
            candidate.layered_fused_decode_ids = [
                r.request_id for r in self.running if r is not candidate and self._is_layered_decode_request(r)
            ]

        @staticmethod
        def _is_layered_decode_request(request):
            return request.is_decode

    class Patched(Base, _FakeScheduler):
        pass

    assert wavefront._patch_scheduler(Patched)
    return Patched


def test_concurrency_one_keeps_the_serial_candidate():
    cls = _patched_scheduler_cls()
    scheduler = cls(depth=1)
    assert scheduler._get_layered_prefill_candidate() == "serial"
    assert cls._get_layered_prefill_candidate_calls == 1


def test_a_decode_rides_only_one_overlapping_span():
    """A decode mid-span on one prompt must not be recruited by another.

    Its token progress is frozen until its span samples, so riding two spans
    satisfies neither: it never leaves ``running`` and the engine wedges.
    """
    cls = _patched_scheduler_cls()
    riding = SimpleNamespace(request_id="riding", is_decode=True, layered_fused_decode_slot=True)
    idle = SimpleNamespace(request_id="idle", is_decode=True, layered_fused_decode_slot=False)
    prompt = SimpleNamespace(request_id="prompt", is_decode=False, layered_fused_decode_slot=False)
    output = SimpleNamespace(layered_prefill_plan=SimpleNamespace(group_id=0))

    scheduler = cls(depth=4, running=[riding, idle, prompt])
    scheduler._attach_fused_mixed_decodes(output, prompt)
    assert prompt.layered_fused_decode_ids == ["idle"]
    # The marker is scoped to the rider walk.
    assert cls._is_layered_decode_request(riding)

    serial = cls(depth=1, running=[riding, idle, prompt])
    serial._attach_fused_mixed_decodes(output, prompt)
    assert prompt.layered_fused_decode_ids == ["riding", "idle"]
