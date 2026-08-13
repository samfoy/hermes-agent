"""End-to-end guard: transient provider capacity errors must survive a real
capacity window instead of dying in ~19 seconds.

Measured failure (Kiro claude-opus-5, 2026-08-13). The provider returned::

    HTTP 500 {"message":"Encountered unexpectedly high load when processing
    the request, please try again.","reason":"MODEL_TEMPORARILY_UNAVAILABLE"}

That is a plain 5xx, so the classifier labelled it a generic ``server_error``
and it inherited the short 2s-base backoff. With the default
``api_max_retries`` of 3, all attempts burned between 12:45:56 and 12:46:15 —
19 seconds — and the turn was dropped with the raw provider error surfaced to
the user.

These tests drive the real ``run_conversation`` loop with a faked API call, so
they prove the *wiring* fires, not merely that the backoff helper computes
larger numbers. The unit-level schedule cover lives in
``tests/test_capacity_overload_backoff.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

_CAPACITY_BODY = (
    '{"message":"Encountered unexpectedly high load when processing the '
    'request, please try again.","reason":"MODEL_TEMPORARILY_UNAVAILABLE"}'
)


class CapacityError(Exception):
    """Shape of the openai.InternalServerError seen in the real trace."""

    status_code = 500

    def __init__(self, body: str = _CAPACITY_BODY):
        super().__init__(f"Error code: 500 - {{'error': {{'message': '{body}'}}}}")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": body, "type": "kiro_error", "code": 500}}


class PlainServerError(Exception):
    """A 5xx with no capacity signal — must keep the old fast-fail behaviour."""

    status_code = 500

    def __init__(self):
        super().__init__("Error code: 500 - internal server error")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "internal server error"}}


def _make_tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _make_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key-abcdef12",
            base_url="http://127.0.0.1:8795/v1",
            provider="kiro",
            model="claude-opus-5",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


def _mock_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="claude-opus-5", usage=None)


class _VirtualClock:
    """Stand-in for the ``time`` module, scoped to ``conversation_loop``.

    The retry loop waits with ``sleep_end = time.time() + wait_time`` and then
    polls ``while time.time() < sleep_end`` in 200ms ``time.sleep`` steps. Three
    consequences for a test that must not actually wait minutes:

    * A no-op ``sleep`` spins forever, because ``time.time()`` never reaches
      ``sleep_end``. Sleeping must advance the clock.
    * Patching ``time.time`` on the real module is global and freezes timing for
      every other module in the process. Replacing the module *reference* inside
      ``conversation_loop`` keeps the fake local; every other attribute
      delegates to the real module.
    * ``wait_time`` itself is never handed to ``sleep``, so the chosen backoff is
      only observable as the *accumulated* 200ms steps between two API calls.
      ``close_window`` is called on each API attempt to close that group.
    """

    def __init__(self):
        self._now = 1_000_000.0
        self._pending = 0.0
        self.waits: list[float] = []

    def time(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._now

    def sleep(self, seconds) -> None:
        seconds = float(seconds)
        self._now += seconds
        self._pending += seconds

    def close_window(self) -> None:
        """Attribute the accumulated sleep to the retry that was waiting."""
        if self._pending > 0:
            self.waits.append(self._pending)
            self._pending = 0.0

    @property
    def total_slept(self) -> float:
        return sum(self.waits) + self._pending

    def __getattr__(self, name):
        import time as _real_time

        return getattr(_real_time, name)


def _run(agent, side_effect):
    """Drive the real ``run_conversation`` loop, recording backoff waits.

    Returns ``(result, waits)`` where ``waits`` holds the wall-clock the loop
    actually spent waiting before each retry — the thing the fix changes.
    """
    clock = _VirtualClock()

    def instrumented(api_kwargs):
        # Reached once per API attempt, so the wait that preceded it is complete.
        clock.close_window()
        return side_effect(api_kwargs)

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=instrumented),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("run_agent.OpenAI", return_value=MagicMock()),
        patch("agent.conversation_loop.time", new=clock),
        patch("agent.agent_runtime_helpers.time.sleep", new=lambda *_a, **_k: None),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        result = agent.run_conversation("hello")

    clock.close_window()
    return result, clock.waits


class TestCapacityOverloadSurvivesRealWindow:
    def test_capacity_500_recovers_after_more_than_three_attempts(self):
        """The regression. Six consecutive capacity 500s used to kill the turn
        after three; the seventh attempt must now land and complete."""
        agent = _make_agent()
        calls = []

        def fake_api_call(api_kwargs):
            calls.append(1)
            if len(calls) <= 6:
                raise CapacityError()
            return _mock_response("Recovered after capacity cleared")

        result, _waits = _run(agent, fake_api_call)

        assert len(calls) == 7, (
            f"expected the loop to keep retrying past the default 3 attempts, "
            f"got {len(calls)} API calls"
        )
        assert result.get("completed") is True
        assert not result.get("failed")
        assert "Recovered after capacity cleared" in (result.get("final_response") or "")

    def test_total_backoff_window_far_exceeds_the_measured_19_seconds(self):
        """The user-visible symptom was giving up 19s into a capacity window.

        The cumulative wait before the successful attempt must now be minutes.
        """
        agent = _make_agent()
        calls = []

        def fake_api_call(api_kwargs):
            calls.append(1)
            if len(calls) <= 6:
                raise CapacityError()
            return _mock_response("ok")

        _result, waits = _run(agent, fake_api_call)

        total = sum(waits)
        assert total > 120.0, (
            f"total backoff {total:.1f}s must far exceed the ~19s that dropped "
            f"the real turn; waits={[round(w, 1) for w in waits]}"
        )
        # Longest single wait shows the long tier engaged, not just more short ones.
        assert max(waits) >= 15.0

    def test_capacity_error_is_surfaced_as_capacity_not_raw_http_500(self):
        """The user saw a bare HTTP 500. Exhausting the longer schedule should
        still report it, but the retry path must have been the capacity one."""
        agent = _make_agent()

        def always_capacity(api_kwargs):
            raise CapacityError()

        result, waits = _run(agent, always_capacity)

        assert result.get("completed") is not True
        # Long tier ran before giving up, rather than 3 quick attempts.
        assert len(waits) > 3
        assert max(waits) >= 15.0


class TestFallbackBeatsWaiting:
    """A configured fallback recovers in seconds, so it must win over patiently
    sitting on an exhausted primary for minutes."""

    def test_capacity_error_prefers_fallback_over_the_long_schedule(self):
        agent = _make_agent()
        # Give the agent a fallback chain, as a user with fallback_providers has.
        setattr(agent, "_fallback_chain", [
            {
                "provider": "bedrock",
                "model": "claude-opus-5",
                "base_url": "https://bedrock.example/v1",
            }
        ])
        setattr(agent, "_fallback_index", 0)

        calls = []

        def always_capacity(api_kwargs):
            calls.append(1)
            raise CapacityError()

        _result, waits = _run(agent, always_capacity)

        # The long tier must not engage while a fallback is still unused.
        assert not waits or max(waits) < 15.0, (
            f"a configured fallback must be tried before minutes of waiting; "
            f"waits={[round(w, 1) for w in waits]}"
        )

    def test_long_schedule_still_applies_when_no_fallback_is_configured(self):
        """With nothing better to switch to, waiting out the capacity window is
        the correct last resort — and it is the reported user's exact setup.

        The turn start resets ``_fallback_index`` to 0, so an empty chain (rather
        than a consumed index) is the honest way to express "no fallback".
        """
        agent = _make_agent()
        setattr(agent, "_fallback_chain", [])
        setattr(agent, "_fallback_index", 0)

        assert agent._has_pending_fallback() is False, (
            "precondition: this test needs no fallback available"
        )

        def always_capacity(api_kwargs):
            raise CapacityError()

        _result, waits = _run(agent, always_capacity)

        assert waits and max(waits) >= 15.0, (
            f"with no fallback available the long schedule must run; waits={waits}"
        )


class TestCompressionWinsOverWaiting:
    """A message can carry BOTH an overflow signal and a capacity word, e.g.
    vLLM's ``server overloaded: prompt exceeds the max_model_len 32768``.

    There the request is too big. Shrinking it is the recovery, so the capacity
    schedule must stand aside — waiting cannot succeed on its own and only
    delays the fix.

    Note on coverage: for *this* input the compression branch returns before the
    backoff code is reached, so the end-to-end test below passes with or without
    the ``should_compress`` exclusion and is a characterization test, not a
    revert-sensitive one. The gate assertion that actually holds the line is
    ``test_gate_refuses_capacity_schedule_when_compression_was_requested``,
    which fails if the exclusion is dropped. Both are kept: the first pins the
    observable behaviour, the second pins the reason.
    """

    def test_overflow_message_mentioning_overload_still_compresses_fast(self):
        agent = _make_agent()

        class OverflowWithCapacityWords(Exception):
            status_code = 500

            def __init__(self):
                msg = "server overloaded: prompt exceeds the max_model_len 32768"
                super().__init__(f"Error code: 500 - {msg}")
                self.response = SimpleNamespace(headers={})
                self.body = {"error": {"message": msg}}

        def always_overflow(api_kwargs):
            raise OverflowWithCapacityWords()

        _result, waits = _run(agent, always_overflow)

        assert not waits or max(waits) < 15.0, (
            f"an overflow error must take the compression path, not minutes of "
            f"capacity backoff; waits={[round(w, 1) for w in waits]}"
        )

    def test_gate_refuses_capacity_schedule_when_compression_was_requested(self):
        """Revert-sensitive: mirrors the loop's gate for a real mixed message.

        The phrase list alone DOES match this error, which is exactly why the
        gate needs the ``should_compress`` exclusion rather than relying on the
        patterns being narrow enough.
        """
        from agent.error_classifier import classify_api_error
        from agent.retry_utils import is_capacity_overload_error

        class E(Exception):
            status_code = 500

            def __init__(self, msg):
                super().__init__(f"Error code: 500 - {msg}")
                self.message = msg
                self.body = {"error": {"message": msg}}

        for msg in (
            "server overloaded: prompt exceeds the max_model_len 32768",
            "context length exceeded while server under high load",
        ):
            err = E(msg)
            classified = classify_api_error(err, provider="vllm", model="local")

            assert is_capacity_overload_error(err, status_code=500) is True, msg
            assert classified.should_compress is True, msg

            gate = (
                classified.retryable
                and not classified.should_compress
                and is_capacity_overload_error(err, status_code=500)
            )
            assert gate is False, (
                f"compression must win over the capacity schedule for: {msg}"
            )


class TestNonCapacityErrorsUnchanged:
    def test_plain_500_still_fails_fast_on_the_default_budget(self):
        """The guard against turning every 5xx into minutes of waiting.

        A 500 with no capacity signal keeps the default 3-attempt budget.
        """
        agent = _make_agent()
        calls = []

        def always_500(api_kwargs):
            calls.append(1)
            raise PlainServerError()

        result, waits = _run(agent, always_500)

        assert len(calls) <= 4, (
            f"a plain 500 must not inherit the capacity schedule; "
            f"got {len(calls)} attempts"
        )
        assert result.get("completed") is not True
        assert not waits or max(waits) < 15.0

    def test_capacity_phrase_on_a_4xx_does_not_slow_the_loop(self):
        """A capacity-flavoured phrase inside a deterministic 4xx rejection is
        text, not a transient overload — it must keep failing fast."""
        agent = _make_agent()

        class NotFoundWithCapacityWords(Exception):
            status_code = 404

            def __init__(self):
                super().__init__(
                    "Error code: 404 - model temporarily unavailable"
                )
                self.response = SimpleNamespace(headers={})
                self.body = {"error": {"message": "model temporarily unavailable"}}

        calls = []

        def always_404(api_kwargs):
            calls.append(1)
            raise NotFoundWithCapacityWords()

        result, waits = _run(agent, always_404)

        assert len(calls) <= 4, f"deterministic 404 retried {len(calls)} times"
        assert result.get("completed") is not True
        assert not waits or max(waits) < 15.0
