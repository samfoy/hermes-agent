"""Tests for transient provider capacity-overload retry backoff.

Regression cover for the failure captured on Kiro claude-opus-5 (2026-08-13):
HTTP 500 ``MODEL_TEMPORARILY_UNAVAILABLE`` classified as a generic
``server_error``, inherited the short 2s-base schedule, and exhausted all three
default attempts in ~19 seconds — far inside a real capacity window — dropping
the turn with a raw provider error.
"""

from types import SimpleNamespace

import agent.retry_utils as retry_utils
from agent.retry_utils import (
    _CAPACITY_OVERLOAD_LONG_BACKOFF,
    _CAPACITY_OVERLOAD_SHORT_ATTEMPTS,
    capacity_overload_backoff,
    capacity_overload_retry_ceiling,
    is_capacity_overload_error,
)

# The verbatim body from the measured failure.
_KIRO_BODY = (
    '{"message":"Encountered unexpectedly high load when processing the '
    'request, please try again.","reason":"MODEL_TEMPORARILY_UNAVAILABLE"}'
)


def _err(message, status: object = 500):
    return SimpleNamespace(status_code=status, message=message)


# ── detection ───────────────────────────────────────────────────────────────

def test_detects_the_measured_kiro_capacity_500():
    """The exact error from the report must be recognised as capacity overload."""
    assert is_capacity_overload_error(_err(_KIRO_BODY)) is True


def test_detects_capacity_phrases_across_transient_statuses():
    for status in (429, 500, 502, 503, 504, 529):
        assert is_capacity_overload_error(_err(_KIRO_BODY, status=status)) is True, status


def test_status_none_still_classifies_on_message_alone():
    """Message-only classification paths have no status to offer."""
    assert is_capacity_overload_error(SimpleNamespace(message=_KIRO_BODY)) is True


def test_deterministic_4xx_carrying_capacity_words_is_not_retried_slowly():
    """The guard that keeps this from being a footgun.

    A capacity-flavoured phrase inside an auth / not-found / malformed-request
    rejection is deterministic. Widening the long backoff to cover it would
    turn an instant failure into minutes of pointless waiting.
    """
    for status in (400, 401, 403, 404, 422):
        assert is_capacity_overload_error(
            _err("model temporarily unavailable", status=status)
        ) is False, status


def test_unrelated_server_error_is_not_capacity_overload():
    """A plain 500 keeps the normal short schedule and fast fallback."""
    assert is_capacity_overload_error(_err("internal server error")) is False
    assert is_capacity_overload_error(_err("upstream connect error")) is False


def test_non_numeric_status_does_not_crash_or_match():
    assert is_capacity_overload_error(_err(_KIRO_BODY, status="bogus")) is False


def test_explicit_status_code_argument_overrides_error_attribute():
    """Callers that already resolved the status get to pass it in."""
    err = _err(_KIRO_BODY, status=404)
    assert is_capacity_overload_error(err, status_code=503) is True
    assert is_capacity_overload_error(_err(_KIRO_BODY, status=503), status_code=404) is False


# ── schedule ────────────────────────────────────────────────────────────────

def test_short_tier_preserves_caller_default_wait():
    """Early attempts stay fast so a momentary blip recovers immediately."""
    for attempt in range(1, _CAPACITY_OVERLOAD_SHORT_ATTEMPTS + 1):
        wait, policy = capacity_overload_backoff(attempt, default_wait=2.0)
        assert wait == 2.0
        assert policy == "capacity_overload_short"


def test_long_tier_waits_are_substantially_longer_than_the_default(monkeypatch):
    """The actual bug: 3 short attempts exhausted in ~19s total.

    Every long-tier wait must exceed the entire old budget so a capacity window
    gets a real chance to clear.
    """
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
    old_total_budget = 19.0
    for attempt in range(_CAPACITY_OVERLOAD_SHORT_ATTEMPTS + 1,
                         _CAPACITY_OVERLOAD_SHORT_ATTEMPTS + len(_CAPACITY_OVERLOAD_LONG_BACKOFF) + 1):
        wait, policy = capacity_overload_backoff(attempt, default_wait=2.0)
        assert policy == "capacity_overload_long"
        assert wait >= 15.0
    assert sum(_CAPACITY_OVERLOAD_LONG_BACKOFF) > old_total_budget


def test_long_tier_is_monotonically_increasing(monkeypatch):
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
    waits = [
        capacity_overload_backoff(a, default_wait=2.0)[0]
        for a in range(_CAPACITY_OVERLOAD_SHORT_ATTEMPTS + 1,
                       _CAPACITY_OVERLOAD_SHORT_ATTEMPTS + len(_CAPACITY_OVERLOAD_LONG_BACKOFF) + 1)
    ]
    assert waits == sorted(waits)
    assert len(set(waits)) == len(waits)


def test_long_tier_clamps_at_final_entry_instead_of_growing(monkeypatch):
    """Attempts past the table reuse the last wait — never unbounded growth."""
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
    final = _CAPACITY_OVERLOAD_LONG_BACKOFF[-1]
    for attempt in (50, 500):
        wait, policy = capacity_overload_backoff(attempt, default_wait=2.0)
        assert wait == final
        assert policy == "capacity_overload_long"


def test_long_tier_is_jittered_to_decorrelate_concurrent_sessions():
    """Real (unpatched) jitter: identical attempts must not align exactly."""
    waits = {capacity_overload_backoff(
        _CAPACITY_OVERLOAD_SHORT_ATTEMPTS + 1, default_wait=2.0)[0] for _ in range(20)}
    assert len(waits) > 1
    base = _CAPACITY_OVERLOAD_LONG_BACKOFF[0]
    assert all(base <= w <= base * 1.25 for w in waits)


# ── ceiling ─────────────────────────────────────────────────────────────────

def test_ceiling_makes_every_long_tier_entry_reachable():
    """The Z.AI bug this mirrors: a ceiling equal to short_attempts left the
    whole long schedule as dead code.

    The loop gives up at ``retry_count >= ceiling`` *before* computing that
    attempt's backoff, so the largest attempt it still schedules (ceiling - 1)
    must reach the final long-tier index.
    """
    ceiling = capacity_overload_retry_ceiling()
    assert ceiling > _CAPACITY_OVERLOAD_SHORT_ATTEMPTS
    last_attempt_with_backoff = ceiling - 1
    assert last_attempt_with_backoff - _CAPACITY_OVERLOAD_SHORT_ATTEMPTS >= len(
        _CAPACITY_OVERLOAD_LONG_BACKOFF)


def test_ceiling_exceeds_the_default_api_max_retries():
    """Default api_max_retries is 3; without raising it the long tier never runs."""
    assert capacity_overload_retry_ceiling() > 3


def test_full_schedule_runs_over_the_attempt_range_the_loop_walks(monkeypatch):
    """End-to-end: with the extended ceiling the whole table is exercised, and
    the total wait window is minutes rather than the measured 19 seconds."""
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
    ceiling = capacity_overload_retry_ceiling()

    long_waits = []
    for attempt in range(1, ceiling):
        wait, policy = capacity_overload_backoff(attempt, default_wait=2.0)
        if policy == "capacity_overload_long":
            long_waits.append(wait)

    assert long_waits == list(_CAPACITY_OVERLOAD_LONG_BACKOFF)
    assert sum(long_waits) >= 180.0
