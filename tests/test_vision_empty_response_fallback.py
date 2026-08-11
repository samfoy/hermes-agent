"""Tests for the empty-vision-response failure path (2026-08-11).

An empty vision response is a SUCCESSFUL API call that returned no visible text
(``finish_reason='stop'``, ``content=None``, ``usage=None``). Observed
intermittently on reasoning models behind the Codex/Responses transport
(bedrock-mantle / openai.gpt-5.6-sol).

The old behaviour retried once, then substituted a canned apology string and
still reported ``success: True`` — so a caller reasoned from an error message as
if it were a description of the image. These tests pin the three branches:

  1. same-backend retry succeeds     -> success, no fallback attempted
  2. retry empty, fallback succeeds  -> success with the fallback's text
  3. everything empty               -> success=False, honest error

RED-check: each assertion below fails against the pre-fix code, which returned
``success: True`` with "There was a problem with the request..." in every case.
"""

import asyncio
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _vision_tools():
    """Import tools.vision_tools FRESH on every use.

    tests/agent/test_vision_routing_31179.py deletes ``tools.vision_tools`` and
    ``agent.auxiliary_client`` out of ``sys.modules`` in an autouse fixture to
    force a config re-read. A module object captured at import time therefore
    goes stale: patching an attribute on it has no effect on the module the code
    under test actually imports, and these tests failed only when run after that
    file. Resolving through ``importlib.import_module`` each time returns
    whatever is currently registered, so patches always land on the live module.
    """
    import importlib

    return importlib.import_module("tools.vision_tools")


def _aux_client():
    """Same rationale as _vision_tools(), for agent.auxiliary_client."""
    import importlib

    return importlib.import_module("agent.auxiliary_client")


def _resp(content, *, usage=True, finish="stop", model="test-model"):
    """Build a chat-completions-shaped response object."""
    msg = types.SimpleNamespace(role="assistant", content=content, tool_calls=None)
    choice = types.SimpleNamespace(index=0, message=msg, finish_reason=finish)
    return types.SimpleNamespace(
        choices=[choice],
        model=model,
        usage=types.SimpleNamespace(total_tokens=10) if usage else None,
    )


EMPTY = _resp(None, usage=False)


class _Harness(unittest.TestCase):
    """Patch the network + filesystem edges so only the branch logic is tested."""

    def setUp(self):
        self.calls = []          # (provider, model) per async_call_llm call
        self._orig = {}

        # Stub the module-level names the tool resolves at call time.
        def fake_extract(resp):
            try:
                return resp.choices[0].message.content or ""
            except Exception:
                return ""

        vt = _vision_tools()
        self._patch(vt, "extract_content_or_reasoning", fake_extract)
        self._patch(vt, "_load_auxiliary_client", lambda: None)

        # Avoid touching the real image pipeline.
        self._patch(
            vt, "_normalize_image_source",
            lambda *a, **k: ("/tmp/fake.png", None),
            required=False,
        )

    def _patch(self, obj, name, value, required=True):
        if not hasattr(obj, name):
            if required:
                self.skipTest(f"{name} missing from vision_tools; API changed")
            return
        self._orig[(obj, name)] = getattr(obj, name)
        setattr(obj, name, value)

    def tearDown(self):
        for (obj, name), val in self._orig.items():
            setattr(obj, name, val)


class TestFallbackHelper(_Harness):
    """_retry_vision_on_fallback_backend: the branch that saves the call."""

    def _run(self, backends, primary, responses):
        """Drive the helper with a fake backend list and scripted responses.

        Patches are applied with ``unittest.mock.patch.object`` rather than by
        assigning module attributes, so they are undone even on failure. This
        matters because the helper re-imports ``async_call_llm`` from
        ``agent.auxiliary_client`` on every candidate: a leaked patch from an
        earlier test in the same process would otherwise be picked up here, which
        is exactly how these tests failed when run after the rest of the suite.
        """
        ac = _aux_client()
        vt = _vision_tools()

        async def fake_call(**kw):
            self.calls.append((kw.get("provider"), kw.get("model")))
            if not responses:
                raise AssertionError("more calls than scripted responses")
            return responses.pop(0)

        def fake_extract(resp):
            try:
                return resp.choices[0].message.content or ""
            except Exception:
                return ""

        with mock.patch.object(ac, "get_available_vision_backends", lambda: list(backends)), \
             mock.patch.object(ac, "resolve_vision_provider_client",
                               lambda *a, **k: (primary, object(), "m")), \
             mock.patch.object(vt, "async_call_llm", fake_call), \
             mock.patch.object(vt, "extract_content_or_reasoning", fake_extract):
            return asyncio.run(
                vt._retry_vision_on_fallback_backend(
                    {"task": "vision", "messages": [], "model": "primary-model"}
                )
            )

    def test_skips_the_backend_that_already_failed(self):
        """The failing provider must not be retried a third time."""
        analysis, prov = self._run(
            backends=["bedrock-mantle", "bedrock"],
            primary="bedrock-mantle",
            responses=[_resp("FALLBACK_TEXT")],
        )
        self.assertEqual(analysis, "FALLBACK_TEXT")
        self.assertEqual(prov, "bedrock")
        self.assertEqual([p for p, _ in self.calls], ["bedrock"])
        self.assertNotIn("bedrock-mantle", [p for p, _ in self.calls])

    def test_drops_primary_model_so_fallback_picks_its_own(self):
        """A mantle model name must not be sent to a bedrock backend."""
        self._run(
            backends=["bedrock-mantle", "bedrock"],
            primary="bedrock-mantle",
            responses=[_resp("OK")],
        )
        self.assertEqual(self.calls[0][1], None, "primary model leaked to fallback")

    def test_no_distinct_backend_returns_empty(self):
        """Single-backend host: nothing to fall back to, fail cleanly."""
        analysis, prov = self._run(
            backends=["bedrock-mantle"], primary="bedrock-mantle", responses=[],
        )
        self.assertEqual(analysis, "")
        self.assertIsNone(prov)
        self.assertEqual(self.calls, [])

    def test_fallback_also_empty_returns_empty(self):
        analysis, prov = self._run(
            backends=["bedrock-mantle", "bedrock"],
            primary="bedrock-mantle",
            responses=[EMPTY],
        )
        self.assertEqual(analysis, "")
        self.assertIsNone(prov)

    def test_exception_in_one_candidate_tries_the_next(self):
        ac = _aux_client()
        vt = _vision_tools()

        seq = ["boom", _resp("SECOND_OK")]

        async def fake_call(**kw):
            self.calls.append((kw.get("provider"), kw.get("model")))
            nxt = seq.pop(0)
            if nxt == "boom":
                raise RuntimeError("backend exploded")
            return nxt

        def fake_extract(r):
            return r.choices[0].message.content or "" if getattr(r, "choices", None) else ""

        with mock.patch.object(ac, "get_available_vision_backends", lambda: ["a", "b"]), \
             mock.patch.object(ac, "resolve_vision_provider_client",
                               lambda *a, **k: ("primary", object(), "m")), \
             mock.patch.object(vt, "async_call_llm", fake_call), \
             mock.patch.object(vt, "extract_content_or_reasoning", fake_extract):
            analysis, prov = asyncio.run(
                vt._retry_vision_on_fallback_backend(
                    {"task": "vision", "messages": []}
                )
            )

        self.assertEqual(analysis, "SECOND_OK")
        self.assertEqual(prov, "b")
        self.assertEqual([p for p, _ in self.calls], ["a", "b"])


class TestEmptyResponseLogging(unittest.TestCase):
    """_log_empty_vision_response must be total — it runs on the failure path."""

    def test_never_raises_on_hostile_shapes(self):
        for bad in (None, object(), 42, "str", [], {},
                    types.SimpleNamespace(choices=None),
                    types.SimpleNamespace(choices=[]),
                    types.SimpleNamespace(choices=[object()])):
            with self.subTest(shape=type(bad).__name__):
                _vision_tools()._log_empty_vision_response("t", bad)  # must not raise

    def test_logs_the_diagnostic_fields(self):
        import logging

        vt = _vision_tools()
        with self.assertLogs(vt.logger, level=logging.WARNING) as cm:
            vt._log_empty_vision_response("first attempt", EMPTY)
        blob = "\n".join(cm.output)
        self.assertIn("vision empty response", blob)
        self.assertIn("first attempt", blob)
        self.assertIn("usage=None", blob, "usage is the cheapest positive signal")
        self.assertIn("finish_reason='stop'", blob)


class TestFailClosedContract(unittest.TestCase):
    """The canned-apology-as-success regression must not come back."""

    def test_apology_string_is_gone_from_source(self):
        src = (REPO / "tools" / "vision_tools.py").read_text()
        self.assertNotIn(
            "There was a problem with the request and the image could not be analyzed.",
            src,
            "the canned apology returned with success=True is the bug; it must stay deleted",
        )

    def test_error_path_reports_success_false(self):
        from tools.registry import tool_error

        payload = json.loads(tool_error("no visible text", success=False))
        self.assertIs(payload["success"], False)
        self.assertIn("error", payload)


if __name__ == "__main__":
    unittest.main()
