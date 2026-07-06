"""Tests for the Opus-4.8 accuracy judge in judge.py.

Every test mocks the Anthropic client at the boundary (``client.messages.create``)
— no test ever reaches the real API (CLAUDE.md §6). Each test encodes *why* the
behaviour matters (Rule 9): the judge must grade strictly against the gold text,
short-circuit answers it cannot grade, and degrade to a rationalised 0.0 rather
than crash or silently swallow a bad model response.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import judge
from judge import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_TOOL,
    JUDGE_TOOL_NAME,
    grade_answer,
)
from models import JudgeVerdict


@pytest.fixture(autouse=True)
def _stub_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the settings boundary so tests never require real API keys in the env.

    The judge only reads ``judge_model`` from settings; patching it here keeps the
    tests hermetic (no ``.env``, no real Anthropic/Voyage/Pinecone keys) without
    coupling the judge test to unrelated provider config.
    """
    monkeypatch.setattr(judge, "get_settings", lambda: SimpleNamespace(judge_model="claude-opus-4-8"))


def _tool_use_response(score: float, rationale: str, *, name: str = JUDGE_TOOL_NAME) -> SimpleNamespace:
    """A canned Anthropic response whose single tool_use block carries a verdict.

    Mirrors the real SDK shape (``response.content`` is a list of blocks, each with
    ``.type``; a tool_use block has ``.name`` and a parsed ``.input`` dict) so the
    grader exercises its real parse-from-structured-response path against it.
    """
    block = SimpleNamespace(type="tool_use", name=name, input={"score": score, "rationale": rationale}, id="toolu_1")
    return SimpleNamespace(stop_reason="tool_use", content=[block])


def _text_only_response(text: str = "Sorry, no tool.") -> SimpleNamespace:
    """A malformed response: the model answered in prose, never calling the tool."""
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(stop_reason="end_turn", content=[block])


def _refusal_response() -> SimpleNamespace:
    """A safety refusal: HTTP 200, empty content, stop_reason 'refusal'."""
    return SimpleNamespace(stop_reason="refusal", content=[])


def test_matching_answer_grades_to_one_with_rationale_as_valid_verdict() -> None:
    # Why (Rule 9): the happy path must parse the enforced structured output into a
    # real, valid JudgeVerdict — not free text — so downstream ResultRow.accuracy is
    # a trustworthy {0,0.5,1}. This exercises the real parse path (mocked transport).
    client = Mock()
    client.messages.create.return_value = _tool_use_response(1.0, "The answer states the same profession as the gold.")

    verdict = grade_answer("A consulting detective.", "A consulting detective.", client=client)

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.score == 1.0
    assert verdict.rationale  # one-sentence justification, never empty
    client.messages.create.assert_called_once()


def test_enforces_structured_output_and_grades_against_gold_not_world_knowledge() -> None:
    # Why: structured output must be ENFORCED (forced tool_choice on a strict tool),
    # and the deterministic prompt must instruct grading against the provided gold
    # text rather than the model's own knowledge — otherwise a true-in-reality but
    # off-gold answer would be mis-scored. Assert both the request shape and the prompt.
    client = Mock()
    client.messages.create.return_value = _tool_use_response(0.0, "The answer contradicts the gold text.")

    # An answer that is true in reality but absent from / contradicting the gold.
    verdict = grade_answer("Paris is the capital of France.", "The capital is Berlin.", client=client)

    assert verdict.score == 0.0
    _, kwargs = client.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": JUDGE_TOOL_NAME}  # forced → enforced
    assert JUDGE_TOOL in kwargs["tools"]
    assert JUDGE_TOOL.get("strict") is True  # strict schema → validated structured output
    prompt = JUDGE_SYSTEM_PROMPT.lower()
    assert "gold" in prompt
    assert "knowledge" in prompt  # instructs against grading on own world knowledge


def test_empty_answer_short_circuits_to_zero_without_calling_the_api() -> None:
    # Why: an empty/errored answer can never match the gold; short-circuiting to 0.0
    # avoids a needless (and costly) API call and never crashes. Documented decision.
    client = Mock()

    verdict = grade_answer("", "A consulting detective.", client=client)

    assert verdict.score == 0.0
    assert verdict.rationale  # explains the 0.0
    client.messages.create.assert_not_called()


def test_whitespace_only_answer_also_short_circuits() -> None:
    # Why: a whitespace-only answer is as ungradeable as an empty one — same 0.0 path,
    # still no API call.
    client = Mock()

    verdict = grade_answer("   \n\t ", "A consulting detective.", client=client)

    assert verdict.score == 0.0
    client.messages.create.assert_not_called()


def test_differently_worded_correct_answer_scores_partial_not_spurious_zero() -> None:
    # Why: a correct-but-differently-worded answer must earn 0.5/1.0 with a reason,
    # not a spurious 0 — the judge passes the model's graded verdict through faithfully.
    client = Mock()
    client.messages.create.return_value = _tool_use_response(0.5, "Captures the gist but omits a detail from the gold.")

    verdict = grade_answer("He solves crimes for a living.", "A consulting detective.", client=client)

    assert verdict.score == 0.5
    assert verdict.rationale


def test_refusal_is_retried_then_falls_back_to_zero_with_rationale() -> None:
    # Why (criterion 5): a safety refusal must not crash the sweep — retry once, then
    # score 0.0 with an explaining rationale so the cell is graded, not dropped.
    client = Mock()
    client.messages.create.return_value = _refusal_response()

    verdict = grade_answer("some answer", "the gold", client=client)

    assert verdict.score == 0.0
    assert verdict.rationale
    assert client.messages.create.call_count == 2  # one retry before falling back


def test_malformed_output_is_retried_then_falls_back_to_zero() -> None:
    # Why (criterion 5): if the model never calls the tool (no parseable verdict),
    # the judge retries once and then falls back to a rationalised 0.0 — it never
    # raises on a bad model response and never returns an invalid verdict.
    client = Mock()
    client.messages.create.return_value = _text_only_response()

    verdict = grade_answer("some answer", "the gold", client=client)

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.score == 0.0
    assert client.messages.create.call_count == 2


def test_recovers_on_retry_after_a_single_malformed_response() -> None:
    # Why: a transient malformed first response must not doom the grade — a valid
    # verdict on the retry is honoured rather than discarded.
    client = Mock()
    client.messages.create.side_effect = [
        _text_only_response(),
        _tool_use_response(1.0, "Matches the gold answer."),
    ]

    verdict = grade_answer("some answer", "the gold", client=client)

    assert verdict.score == 1.0
    assert client.messages.create.call_count == 2
