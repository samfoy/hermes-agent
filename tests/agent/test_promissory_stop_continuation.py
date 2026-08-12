"""Post-tool promissory-stop continuation: detector + call-site wiring.

Covers the Responses-API defect where a mid-task progress note gets stamped
``phase='final_answer'`` after a tool round and terminates the turn (observed
message ids 126162, 133517, 136646 in the local state.db; the user had to
type "continue").  The fix adds ``looks_like_post_tool_promissory_stop`` in
``agent.agent_runtime_helpers`` and a nudge block in
``agent.conversation_loop`` scoped to ``api_mode == "codex_responses"``.

Anti-vacuity design (see the agent-turn-termination-forensics skill):
* The detector is IMPORTED, never copied — reverting the helper fails this
  module at collection with ImportError.
* The conversation-loop call site is extracted by AST, keyed on the
  ``promissory_stop_continuations += 1`` statement inside the ``if`` body —
  reverting the call site raises at import with a named reason.
* True-positive fixtures are the exact recorded texts of the three real
  defects; false-positive fixtures are the exact recorded texts of the
  legitimate promissory-sounding stops the corpus sweep surfaced.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import agent.conversation_loop as conversation_loop
from agent.agent_runtime_helpers import (
    looks_like_post_tool_promissory_stop,
)
from agent.context_compressor import ContextCompressor

_LOOP_SOURCE = Path(conversation_loop.__file__).read_text()

# ── AST gate: the call site must exist and be codex-scoped ──────────────────
_CALL_SITE_TEST = None
_CALL_SITE_BODY = None
for _node in ast.walk(ast.parse(_LOOP_SOURCE)):
    if not isinstance(_node, ast.If):
        continue
    for _stmt in _node.body:
        if (
            isinstance(_stmt, ast.AugAssign)
            and isinstance(_stmt.target, ast.Name)
            and _stmt.target.id == "promissory_stop_continuations"
        ):
            _CALL_SITE_TEST = ast.unparse(_node.test)
            _CALL_SITE_BODY = "\n".join(ast.unparse(s) for s in _node.body)
if _CALL_SITE_TEST is None:
    raise AssertionError(
        "promissory-stop call site not found in conversation_loop — "
        "the fix is missing or the counter was renamed"
    )
assert isinstance(_CALL_SITE_TEST, str)
assert isinstance(_CALL_SITE_BODY, str)


def _agent():
    from agent.agent_runtime_helpers import strip_think_blocks

    return SimpleNamespace(
        _strip_think_blocks=lambda c: strip_think_blocks(None, c),
    )


def _post_tool_messages():
    return [
        {"role": "user", "content": "publish the doc"},
        {"role": "assistant", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "tool_call_id": "x", "content": "SyntaxError: line 3"},
    ]


# The three real defects, verbatim from state.db (ids 136646, 126162, 133517).
DEFECT_136646 = (
    "The browser render completed, but my verification snippet had a syntax "
    "error. I will rerun the check with the fixed parser."
)
DEFECT_126162 = (
    "The queue is broad and includes alarms, pipeline blocks, compliance "
    "notices, and project requests. I am now ranking by severity, ownership, "
    "alarm state, and latest correspondence rather than by title alone."
)
DEFECT_133517 = (
    "I will clean the last lint defects, then publish the corrected source "
    "blocks and decision sentence."
)

# Legitimate stops the corpus sweep surfaced as near-misses, verbatim.
LEGIT_DECLINATION_114235 = (
    "Confirmed slide 9 is the 35-run bimodal chart. Delete the full slide and "
    "its notes. I will not relocate its weak 35-run claims unless another "
    "slide needs the iteration-count diagnostic. Keep going."
)
LEGIT_WAITING_114372 = (
    "The builder is still active and has started the recut. The source tree "
    "remains unchanged so far; its provisional report confirms the correct "
    "15-slide scope. I will continue when the completed diff returns."
)
LEGIT_USER_ASK_75746 = (
    "Need your mwinit to fetch those files. Can you run `mwinit -o` in "
    "another terminal and let me know when it's done? Then I'll retry the "
    "download."
)
LEGIT_BG_MONITOR_56170 = (
    "Uploading steadily — 102,135 / 161,055 (~63%), no errors, sync still "
    "running on pass 1. At this rate it should finish this pass well within "
    "the credential window. I'll report at the next check or when it hits "
    "FINAL."
)


class TestDetectorTruePositives(unittest.TestCase):
    def test_recorded_defects_fire(self):
        agent = _agent()
        msgs = _post_tool_messages()
        for name, text in (
            ("136646", DEFECT_136646),
            ("126162", DEFECT_126162),
            ("133517", DEFECT_133517),
        ):
            with self.subTest(defect=name):
                self.assertTrue(
                    looks_like_post_tool_promissory_stop(agent, text, msgs)
                )

    def test_fires_with_thinking_block_prefix(self):
        agent = _agent()
        text = "<think>scratch</think>" + DEFECT_136646
        self.assertTrue(
            looks_like_post_tool_promissory_stop(
                agent, text, _post_tool_messages()
            )
        )


class TestDetectorFalsePositives(unittest.TestCase):
    def test_recorded_legitimate_stops_do_not_fire(self):
        agent = _agent()
        msgs = _post_tool_messages()
        for name, text in (
            ("declination-114235", LEGIT_DECLINATION_114235),
            ("waiting-114372", LEGIT_WAITING_114372),
            ("user-ask-75746", LEGIT_USER_ASK_75746),
            ("bg-monitor-56170", LEGIT_BG_MONITOR_56170),
        ):
            with self.subTest(case=name):
                self.assertFalse(
                    looks_like_post_tool_promissory_stop(agent, text, msgs)
                )

    def test_terse_answers_do_not_fire(self):
        agent = _agent()
        msgs = _post_tool_messages()
        for text in ("Done.", "PASS", "OK", "42", "Fixed the typo in utils.ts."):
            with self.subTest(text=text):
                self.assertFalse(
                    looks_like_post_tool_promissory_stop(agent, text, msgs)
                )

    def test_long_real_answer_with_promissory_opener_does_not_fire(self):
        # Promise at the START followed by a delivered answer is not a
        # dangling promise — the tail window must not reach it.
        agent = _agent()
        text = (
            "I will check each item in turn. "
            + "Here are the results of the audit. " * 30
            + "All checks passed and the report is published."
        )
        self.assertFalse(
            looks_like_post_tool_promissory_stop(
                agent, text, _post_tool_messages()
            )
        )

    def test_not_post_tool_does_not_fire(self):
        # Same defect text, but the turn did not just execute tools —
        # the start-of-turn intent-ack path owns that case.
        agent = _agent()
        msgs = [{"role": "user", "content": "go"}]
        self.assertFalse(
            looks_like_post_tool_promissory_stop(agent, DEFECT_136646, msgs)
        )
        self.assertFalse(
            looks_like_post_tool_promissory_stop(agent, DEFECT_136646, [])
        )

    def test_question_to_user_does_not_fire(self):
        agent = _agent()
        text = "I will rerun the check — is that the parser you meant?"
        self.assertFalse(
            looks_like_post_tool_promissory_stop(
                agent, text, _post_tool_messages()
            )
        )

    def test_deferred_or_periodic_followups_do_not_fire(self):
        # Adversarial-review finding: a completed turn with a routine
        # future follow-up mention is not a stalled task.
        agent = _agent()
        msgs = _post_tool_messages()
        for name, text in (
            ("next-day recheck",
             "Fixed. I will re-check tomorrow morning after the batch job runs."),
            ("check-back tomorrow",
             "All changes are committed and pushed. I will check back tomorrow "
             "to confirm the metrics look healthy."),
            ("periodic rerun",
             "I fixed the parser bug and reran the check successfully. I will "
             "rerun periodically as a smoke test."),
            ("separate future task",
             "Report published. I will rank the next batch of tickets by "
             "severity in the follow-up review."),
        ):
            with self.subTest(case=name):
                self.assertFalse(
                    looks_like_post_tool_promissory_stop(agent, text, msgs)
                )

    def test_nudge_text_itself_never_matches_the_detector(self):
        # If a future edit to the nudge copy introduced first-person
        # promissory phrasing, a model echoing the nudge back as assistant
        # content could self-trigger. Pin the wording property.
        from agent.agent_runtime_helpers import _PROMISSORY_TAIL_RE

        nudge = conversation_loop._PROMISSORY_STOP_CONTINUATION_NUDGE
        self.assertIsNone(_PROMISSORY_TAIL_RE.search(nudge))


class TestDetectorHostileShapes(unittest.TestCase):
    def test_multimodal_list_content_does_not_raise(self):
        agent = _agent()
        payload = [
            {"type": "text", "text": "I will rerun the check now."},
            {"type": "thinking", "content": "scratch"},
        ]
        # Must classify without raising; the classification itself may be
        # either way depending on how strip_think_blocks flattens lists —
        # the contract under test is no-crash, fail-toward-terminate.
        try:
            looks_like_post_tool_promissory_stop(
                agent, payload, _post_tool_messages()
            )
        except Exception as exc:  # pragma: no cover
            self.fail(f"raised on multimodal content: {exc!r}")

    def test_none_and_empty_do_not_fire(self):
        agent = _agent()
        msgs = _post_tool_messages()
        self.assertFalse(looks_like_post_tool_promissory_stop(agent, None, msgs))
        self.assertFalse(looks_like_post_tool_promissory_stop(agent, "", msgs))


class TestCallSiteWiring(unittest.TestCase):
    def test_call_site_gates_on_codex_and_detector(self):
        # The extracted if-test must consult the detector, the api_mode
        # scope, the per-turn cap, and tool availability.
        self.assertIn("looks_like_post_tool_promissory_stop", _CALL_SITE_TEST)
        self.assertIn("codex_responses", _CALL_SITE_TEST)
        self.assertIn("promissory_stop_continuations < 2", _CALL_SITE_TEST)
        self.assertIn("valid_tool_names", _CALL_SITE_TEST)

    def test_call_site_body_continues_non_finally(self):
        # The nudge body must (a) mark the turn non-final, (b) append the
        # nudge user message, and (c) loop — the trio that makes the
        # continuation actually happen rather than terminating anyway.
        # Extracted by AST from the live source, so reverting any of the
        # three fails here with a named diff.
        self.assertIn("final_response = None", _CALL_SITE_BODY)
        self.assertIn("_PROMISSORY_STOP_CONTINUATION_NUDGE", _CALL_SITE_BODY)
        self.assertIn("continue", _CALL_SITE_BODY.split("\n")[-1])
        self.assertIn("_emit_interim_assistant_message", _CALL_SITE_BODY)

    def test_counter_resets_after_clean_final_answer(self):
        # The per-turn counter must reset on the fall-through path (clean
        # final answer), or two defects across one long session would burn
        # the whole budget. The reset is the statement immediately after
        # the if-block; assert it exists outside the body.
        reset_lines = [
            l for l in _LOOP_SOURCE.splitlines()
            if l.strip() == "promissory_stop_continuations = 0"
        ]
        self.assertGreaterEqual(
            len(reset_lines), 2,
            "expected init + fall-through reset of promissory_stop_continuations",
        )

    def test_nudge_constant_exists_and_is_synthetic(self):
        nudge = conversation_loop._PROMISSORY_STOP_CONTINUATION_NUDGE
        self.assertTrue(nudge.startswith("[System:"))
        # The compressor must recognize the nudge as ephemeral scaffolding so
        # it never survives compaction as a fake human turn.
        self.assertTrue(
            ContextCompressor._is_synthetic_compression_user_turn(
                {"role": "user", "content": nudge}
            )
        )

    def test_no_local_detector_copy_in_this_file(self):
        own = Path(__file__).read_text()
        tree = ast.parse(own)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.assertNotEqual(
                    node.name, "looks_like_post_tool_promissory_stop",
                    "test file must import the detector, not redefine it",
                )


if __name__ == "__main__":
    unittest.main()
