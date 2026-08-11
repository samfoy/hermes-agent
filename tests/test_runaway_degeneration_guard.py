"""Guard against runaway low-entropy degeneration on Responses-API models.

Failure mode: gpt-5.x on the Responses API sometimes emits a repeated token
instead of composing an answer. Observed once on this install (session
b07427f45039, message id 132807, 2026-08-11): 103,088 characters that
decomposed to a single "}", 1,137 lines of "!", and 77,747 blank lines, with
NOT ONE line longer than 12 characters. It persisted with finish_reason=NULL,
so the turn never finalized and the garbage reached the user.

These tests import the production predicate and read the guard expression out
of the source by AST, so a revert fails LOUDLY rather than passing against a
hand-copied duplicate. (Learned from the sentinel-guard work, where 7 of 9
mirrored tests passed with the production fix fully reverted.)
"""
from __future__ import annotations

import ast
import os
import sqlite3
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from agent.codex_responses_adapter import (  # noqa: E402
    _RUNAWAY_BLANK_RATIO,
    _RUNAWAY_DOMINANCE,
    _RUNAWAY_MIN_LINES,
    _RUNAWAY_PROSE_MIN_CHARS,
    _is_runaway_degenerate_text,
)

SOURCE_PATH = os.path.join(REPO, "agent", "codex_responses_adapter.py")
with open(SOURCE_PATH, encoding="utf8") as fh:
    SOURCE = fh.read()


def _extract_guard_condition(flag_name: str) -> str:
    """Return the unparsed ``if`` test that sets ``flag_name = True``.

    The guard lives as an ``if`` condition, so walk for the If node whose body
    assigns the flag True. Raising at import time (rather than asserting inside
    a test) makes a revert fail at COLLECTION with a named reason -- louder and
    earlier than a late assertion.
    """
    for node in ast.walk(ast.parse(SOURCE)):
        if not isinstance(node, ast.If):
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and any(getattr(t, "id", None) == flag_name for t in stmt.targets)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is True
            ):
                return ast.unparse(node.test)
    raise AssertionError(
        f"guard condition setting {flag_name}=True not found in {SOURCE_PATH} -- "
        "the fix is missing, renamed, or reverted."
    )


GUARD_EXPR = _extract_guard_condition("runaway_degenerate_text")

# The real defect payload, rebuilt to the exact measured shape.
REAL_SHAPE = "}\n" + "\n\n" * 300 + "".join("!\n" + "\n" * 60 for _ in range(1137))


class TestRunawayPredicate(unittest.TestCase):
    def test_flags_the_real_recorded_shape(self):
        self.assertTrue(_is_runaway_degenerate_text(REAL_SHAPE))

    def test_flags_pure_repeated_token(self):
        self.assertTrue(_is_runaway_degenerate_text("!\n" * 200))

    def test_flags_overwhelmingly_blank_padding(self):
        self.assertTrue(_is_runaway_degenerate_text("ok\n" + "\n" * 500))

    def test_flags_whitespace_only_many_lines(self):
        self.assertTrue(_is_runaway_degenerate_text("   \n" * 100))

    def test_ignores_short_messages_entirely(self):
        # At or below the line floor a terse reply is plausible; never flag.
        self.assertFalse(_is_runaway_degenerate_text("!\n" * _RUNAWAY_MIN_LINES))

    def test_never_raises_on_hostile_input(self):
        # Deliberately wrong types: the predicate must degrade to False, never
        # raise, because it runs inside the response-normalization hot path.
        hostile: list = [None, 123, [], {}, object(), b"bytes"]
        for bad in hostile:
            with self.subTest(value=type(bad).__name__):
                self.assertFalse(_is_runaway_degenerate_text(bad))  # type: ignore[arg-type]


class TestRunawayFalsePositives(unittest.TestCase):
    """Legitimate answers must never be classified as degenerate."""

    def test_long_prose_answer_is_clean(self):
        para = "This sentence is comfortably longer than the prose floor.\n"
        self.assertFalse(_is_runaway_degenerate_text(para * 400))

    def test_large_code_block_is_clean(self):
        code = "    result = compute_the_thing(alpha, beta, gamma)\n"
        self.assertFalse(_is_runaway_degenerate_text(code * 300))

    def test_markdown_table_is_clean(self):
        row = "| a_column_value | another_column_value | third_value |\n"
        self.assertFalse(_is_runaway_degenerate_text(row * 200))

    def test_repetitive_but_substantive_lines_are_clean(self):
        """High dominance alone must not condemn: these lines carry prose."""
        line = "  - the same long bullet repeated many times over\n"
        text = line * 300
        self.assertFalse(_is_runaway_degenerate_text(text))

    def test_prose_with_heavy_blank_padding_is_clean(self):
        """Blank ratio alone must not condemn when real prose is present."""
        text = ("A genuinely substantive line of explanation.\n" + "\n" * 40) * 30
        self.assertFalse(_is_runaway_degenerate_text(text))

    def test_ascii_diagram_of_short_lines_is_clean_when_any_line_is_prose(self):
        text = "+---+\n" * 100 + "The diagram above shows the layout.\n"
        self.assertFalse(_is_runaway_degenerate_text(text))


class TestGuardWiring(unittest.TestCase):
    """The predicate must actually gate the finish_reason, not sit unused."""

    def test_guard_expression_uses_the_predicate(self):
        self.assertIn("_is_runaway_degenerate_text", GUARD_EXPR)

    def test_guard_requires_no_tool_calls(self):
        # A degenerate-looking text alongside real tool calls must not be
        # condemned: finish_reason="tool_calls" short-circuits first anyway.
        self.assertIn("not tool_calls", GUARD_EXPR)

    def test_guard_requires_non_empty_text(self):
        self.assertIn("final_text", GUARD_EXPR)

    def test_finish_reason_branch_exists_and_yields_incomplete(self):
        code_only = "\n".join(line.split("#", 1)[0] for line in SOURCE.splitlines())
        self.assertIn("elif runaway_degenerate_text:", code_only)
        idx = code_only.index("elif runaway_degenerate_text:")
        tail = code_only[idx : idx + 200]
        self.assertIn('finish_reason = "incomplete"', tail)

    def test_guard_is_evaluated_after_the_leak_guard(self):
        """Leaked tool-call text is the more specific diagnosis; it wins."""
        code_only = "\n".join(line.split("#", 1)[0] for line in SOURCE.splitlines())
        self.assertLess(
            code_only.index("leaked_tool_call_text = True"),
            code_only.index("runaway_degenerate_text = True"),
        )

    def test_evaluating_the_live_guard_expression_on_the_real_payload(self):
        env = {
            "_is_runaway_degenerate_text": _is_runaway_degenerate_text,
            "final_text": REAL_SHAPE,
            "tool_calls": [],
        }
        self.assertTrue(bool(eval(GUARD_EXPR, env)))  # noqa: S307

    def test_live_guard_expression_spares_a_real_answer(self):
        env = {
            "_is_runaway_degenerate_text": _is_runaway_degenerate_text,
            "final_text": "Here is a perfectly ordinary answer to the question.",
            "tool_calls": [],
        }
        self.assertFalse(bool(eval(GUARD_EXPR, env)))  # noqa: S307


class TestThresholdsDocumented(unittest.TestCase):
    """Thresholds are load-bearing; keep them honest and named."""

    def test_thresholds_are_sane(self):
        self.assertGreater(_RUNAWAY_MIN_LINES, 10)
        self.assertGreater(_RUNAWAY_PROSE_MIN_CHARS, 0)
        self.assertGreater(_RUNAWAY_DOMINANCE, 0.0)
        self.assertLessEqual(_RUNAWAY_DOMINANCE, 1.0)
        self.assertGreater(_RUNAWAY_BLANK_RATIO, 0.0)
        self.assertLessEqual(_RUNAWAY_BLANK_RATIO, 1.0)

    def test_no_local_copy_of_the_predicate(self):
        """This file must exercise production code, not a mirrored duplicate."""
        with open(os.path.abspath(__file__), encoding="utf8") as fh:
            own = fh.read()
        code_only = "\n".join(
            line for line in own.splitlines()
            if "assertNotIn" not in line and "assertIn" not in line
        )
        self.assertNotIn("def _is_runaway_degenerate_text", code_only)


class TestAgainstRecordedRow(unittest.TestCase):
    """Validate against the ACTUAL row in state.db when it is available."""

    DB = os.path.expanduser("~/.hermes/state.db")

    def test_real_row_132807_is_flagged(self):
        if not os.path.exists(self.DB):
            self.skipTest("no local state.db")
        con = sqlite3.connect(f"file:{self.DB}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT content FROM messages WHERE id=132807"
            ).fetchone()
        finally:
            con.close()
        if not row or not row[0]:
            self.skipTest("row 132807 not present in this DB")
        self.assertTrue(_is_runaway_degenerate_text(row[0]))

    def test_no_false_positives_across_local_history(self):
        """The predicate must not condemn any other message in the corpus."""
        if not os.path.exists(self.DB):
            self.skipTest("no local state.db")
        con = sqlite3.connect(f"file:{self.DB}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT id, content FROM messages WHERE role='assistant' "
                "AND length(COALESCE(content,'')) > 400"
            ).fetchall()
        finally:
            con.close()
        if not rows:
            self.skipTest("no assistant messages in this DB")
        flagged = [mid for mid, content in rows if _is_runaway_degenerate_text(content)]
        self.assertLessEqual(
            set(flagged), {132807},
            f"predicate flagged unexpected messages: {sorted(set(flagged) - {132807})}",
        )


if __name__ == "__main__":
    unittest.main()
