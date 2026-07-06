"""Tests for the plain-LLM baseline in plain_llm.py.

The Anthropic boundary (Sonnet 5 generation + judge) is ALWAYS mocked; no test reaches a live API
(CLAUDE.md §6). Each test encodes *why* the behaviour matters (Rule 9): the baseline must produce a
trustworthy measured ResultRow when the corpus fits, and — the priority credibility behaviour — emit
a *skipped* row (not an error, not a raise) once the corpus passes the ~1M context wall, with the
crossover EXACT and LOGGED. Cost flows through the shared cost.py; a generation failure below the wall
degrades to an error row rather than halting the sweep.
"""

from types import SimpleNamespace

import pytest
import structlog

import cost
import judge
import plain_llm
from corpus import CorpusBook
from corpus_loader import load_for_token_target
from models import BenchmarkQuestion, JudgeVerdict, QuestionTier, ResultRow, SystemName


# ─── Boundary fakes (mirror the SDK shapes plain_llm calls) ───────────────────


class RecordingAnthropic:
    """Anthropic boundary: ``.messages.create(**kwargs)`` returns queued responses, records calls."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs) -> object:
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _generation_response(text: str, *, input_tokens: int = 100, output_tokens: int = 20) -> SimpleNamespace:
    """A canned Sonnet 5 generation response (text block + usage), mirroring the SDK shape."""
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens))


def _judge_tool_response(score: float, rationale: str) -> SimpleNamespace:
    """A canned judge response whose forced tool_use block carries the verdict (real judge parses it)."""
    block = SimpleNamespace(type="tool_use", name=judge.JUDGE_TOOL_NAME, input={"score": score, "rationale": rationale}, id="toolu_1")
    return SimpleNamespace(stop_reason="tool_use", content=[block])


def _settings(**overrides) -> SimpleNamespace:
    """Fake settings so tests need no .env / real keys (plain_llm reads only the generation model)."""
    base = {"rag_generation_model": "claude-sonnet-5"}
    base.update(overrides)
    return SimpleNamespace(**base)


def _question(**overrides) -> BenchmarkQuestion:
    fields = {
        "question_id": "holmes_lookup_1",
        "book_key": "holmes",
        "tier": QuestionTier.LOOKUP,
        "question_text": "What is Holmes's profession?",
        "gold_answer": "A consulting detective.",
    }
    fields.update(overrides)
    return BenchmarkQuestion(**fields)


def _corpus_load(text: str, *, total_token_count: int, sources: int = 1) -> SimpleNamespace:
    """A minimal CorpusLoad stand-in exposing the three attributes plain_llm reads."""
    return SimpleNamespace(text=text, total_token_count=total_token_count, sources=list(range(sources)))


def _loader(corpus_load: SimpleNamespace):
    """Wrap a fixed CorpusLoad as an injectable ``corpus_loader`` callable ``(size) -> CorpusLoad``."""
    return lambda size: corpus_load


# ─── Happy path: whole sub-1M corpus answered -> graded measured row ──────────


def test_happy_path_whole_corpus_answered_graded_measured_row(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 9): the point of the slice — a sub-1M corpus is stuffed WHOLE into Sonnet 5, answered,
    # judged, and yields a fully-populated MEASURED ResultRow with latency and input-token cost set, so
    # it is a trustworthy cell of the crossover chart.
    monkeypatch.setattr(plain_llm, "grade_answer", lambda answer, gold, client=None: JudgeVerdict(score=1.0, rationale="matches gold"))
    corpus_text = "Sherlock Holmes is a consulting detective who lives at Baker Street."
    anthropic = RecordingAnthropic([_generation_response("A consulting detective.")])

    row = plain_llm.run_plain_llm_query(
        _question(),
        500_000,
        anthropic_client=anthropic,
        corpus_loader=_loader(_corpus_load(corpus_text, total_token_count=500_000)),
        settings=_settings(),
    )

    assert isinstance(row, ResultRow)
    assert row.system is SystemName.PLAIN_LLM
    assert row.measured_or_extrapolated == "measured"
    assert row.corpus_token_count == 500_000
    assert row.accuracy == 1.0
    assert row.answer_text == "A consulting detective."
    assert row.skipped_reason == ""  # answered, not skipped
    assert row.error == ""
    assert row.latency_seconds >= 0.0
    assert row.cost_usd > 0.0  # generation priced, not left at the unmeasured default
    # The WHOLE corpus reached the model, and the 1M-context beta header was wired.
    assert corpus_text in anthropic.calls[0]["messages"][0]["content"]
    assert anthropic.calls[0]["extra_headers"]["anthropic-beta"] == plain_llm.CONTEXT_1M_BETA_HEADER


# ─── Context-wall skip (priority credibility test, PRD story #11) ─────────────


def test_corpus_over_wall_returns_skipped_row_and_does_not_raise() -> None:
    # Why (Rule 12): once the corpus passes the context wall the path MUST return a skipped row
    # (skipped_reason="exceeds_context_window") — NOT raise, NOT an error row, and NOT call the model.
    # This is what makes the wall explicit in the data instead of a crash.
    anthropic = RecordingAnthropic([])  # empty: any API call would IndexError -> proves no call is made

    row = plain_llm.run_plain_llm_query(
        _question(),
        2_000_000,
        anthropic_client=anthropic,
        corpus_loader=_loader(_corpus_load("x " * 10, total_token_count=2_000_000)),
        settings=_settings(),
    )

    assert row.system is SystemName.PLAIN_LLM
    assert row.skipped_reason == "exceeds_context_window"
    assert row.skipped_reason == plain_llm.CONTEXT_WALL_SKIPPED_REASON
    assert row.error == ""  # a skip is not an error
    assert row.answer_text == ""
    assert row.cost_usd == 0.0  # a skipped row has no measured cost
    assert row.accuracy == 0.0
    assert row.measured_or_extrapolated == "measured"  # an observed wall, not an extrapolation
    assert row.corpus_token_count == 2_000_000
    assert anthropic.calls == []  # no API call was made


# ─── Crossover exactness (must be exact AND logged) ───────────────────────────


def test_crossover_boundary_is_exact_just_under_answers_just_over_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 9): the crossover is the headline of the whole benchmark, so the wall must be EXACT —
    # a corpus of exactly CONTEXT_WINDOW_TOKEN_LIMIT tokens is answered; one token more is skipped.
    monkeypatch.setattr(plain_llm, "grade_answer", lambda answer, gold, client=None: JudgeVerdict(score=1.0, rationale="ok"))
    limit = plain_llm.CONTEXT_WINDOW_TOKEN_LIMIT

    at_wall = plain_llm.run_plain_llm_query(
        _question(), limit,
        anthropic_client=RecordingAnthropic([_generation_response("answered")]),
        corpus_loader=_loader(_corpus_load("corpus", total_token_count=limit)),
        settings=_settings(),
    )
    over_wall = plain_llm.run_plain_llm_query(
        _question(), limit + 1,
        anthropic_client=RecordingAnthropic([]),
        corpus_loader=_loader(_corpus_load("corpus", total_token_count=limit + 1)),
        settings=_settings(),
    )

    assert at_wall.skipped_reason == ""  # exactly at the wall -> answered
    assert at_wall.answer_text == "answered"
    assert over_wall.skipped_reason == "exceeds_context_window"  # one token over -> skipped


def test_crossover_value_is_logged_not_inferred() -> None:
    # Why (Rule 9): the wall must be a *logged field*, not something a downstream reader infers. Capture
    # the structured log and assert the exact crossover values were recorded on the skip decision.
    with structlog.testing.capture_logs() as logs:
        plain_llm.run_plain_llm_query(
            _question(),
            1_500_000,
            anthropic_client=RecordingAnthropic([]),
            corpus_loader=_loader(_corpus_load("corpus", total_token_count=1_500_000)),
            settings=_settings(),
        )

    skip_events = [entry for entry in logs if entry["event"] == "plain_llm_context_wall_skipped"]
    assert len(skip_events) == 1
    event = skip_events[0]
    assert event["corpus_token_count"] == 1_500_000
    assert event["context_window_limit"] == plain_llm.CONTEXT_WINDOW_TOKEN_LIMIT
    assert event["over_by"] == 1_500_000 - plain_llm.CONTEXT_WINDOW_TOKEN_LIMIT
    assert event["skipped_reason"] == "exceeds_context_window"


# ─── Cost computed by the shared cost.py (not inline) ─────────────────────────


def test_cost_is_computed_from_usage_by_shared_cost_module(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 9): cost is the thesis. The per-query cost must be exactly what cost.py computes from the
    # generation input/output usage, proving it flows through the shared contract (reused across paths)
    # rather than being re-derived inline in plain_llm.py.
    monkeypatch.setattr(plain_llm, "grade_answer", lambda answer, gold, client=None: JudgeVerdict(score=0.5, rationale="partial"))
    anthropic = RecordingAnthropic([_generation_response("partial answer", input_tokens=400_000, output_tokens=60)])

    row = plain_llm.run_plain_llm_query(
        _question(), 500_000,
        anthropic_client=anthropic,
        corpus_loader=_loader(_corpus_load("some corpus text here", total_token_count=500_000)),
        settings=_settings(),
    )

    expected = cost.token_cost_usd("claude-sonnet-5", input_tokens=400_000, output_tokens=60)
    assert row.cost_usd == pytest.approx(expected)
    assert row.latency_seconds >= 0.0


# ─── measured_or_extrapolated is correct on every emitted row ─────────────────


def test_measured_or_extrapolated_is_measured_on_answered_skipped_and_error_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 9): plain-LLM only ever emits *measured* rows (it runs or observes the wall) — never
    # "extrapolated" (that boundary belongs to the wiki path). All three emitted shapes must say so, so
    # the measured-vs-extrapolated distinction in the data is never blurred (Rule 12).
    monkeypatch.setattr(plain_llm, "grade_answer", lambda answer, gold, client=None: JudgeVerdict(score=1.0, rationale="ok"))

    answered = plain_llm.run_plain_llm_query(
        _question(), 100_000,
        anthropic_client=RecordingAnthropic([_generation_response("a")]),
        corpus_loader=_loader(_corpus_load("c", total_token_count=100_000)),
        settings=_settings(),
    )
    skipped = plain_llm.run_plain_llm_query(
        _question(), 5_000_000,
        anthropic_client=RecordingAnthropic([]),
        corpus_loader=_loader(_corpus_load("c", total_token_count=5_000_000)),
        settings=_settings(),
    )

    class ExplodingAnthropic:
        def __init__(self) -> None:
            self.messages = self

        def create(self, **kwargs):
            raise RuntimeError("anthropic 500")

    errored = plain_llm.run_plain_llm_query(
        _question(), 100_000,
        anthropic_client=ExplodingAnthropic(),
        corpus_loader=_loader(_corpus_load("c", total_token_count=100_000)),
        settings=_settings(),
    )

    assert answered.measured_or_extrapolated == "measured" and answered.skipped_reason == ""
    assert skipped.measured_or_extrapolated == "measured" and skipped.skipped_reason == "exceeds_context_window"
    assert errored.measured_or_extrapolated == "measured" and errored.error


# ─── Error/boundary: generation failure BELOW the wall -> error row, not halt ─


def test_generation_failure_below_wall_becomes_error_row_not_a_raise_or_skip() -> None:
    # Why (Rule 12): a failure below the wall (here generation raises) must be recorded as an ERROR
    # ResultRow (populated error) so the sweep continues — it must NOT propagate/halt, and must NOT be
    # mislabelled as a context-wall skip.
    class ExplodingAnthropic:
        def __init__(self) -> None:
            self.messages = self

        def create(self, **kwargs):
            raise RuntimeError("anthropic 500")

    row = plain_llm.run_plain_llm_query(
        _question(), 300_000,
        anthropic_client=ExplodingAnthropic(),
        corpus_loader=_loader(_corpus_load("indexed corpus text", total_token_count=300_000)),
        settings=_settings(),
    )

    assert isinstance(row, ResultRow)
    assert row.error and "anthropic 500" in row.error
    assert row.skipped_reason == ""  # a below-wall failure is an error, NOT a wall skip
    assert row.accuracy == 0.0
    assert row.measured_or_extrapolated == "measured"
    assert row.corpus_token_count == 300_000  # the real size the failure happened at, not zeroed


# ─── Integration: real corpus_loader + real judge parse + real cost.py ────────


def test_integration_real_loader_real_judge_real_cost_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 9 + B3.5): exercise the REAL wiring the sweep runner (#9) will consume — the real
    # corpus_loader assembles the corpus, plain_llm stuffs it whole into Sonnet 5 (mocked), the REAL
    # judge parses a real tool_use verdict, and the REAL cost.py prices it — all landing in a real
    # ResultRow. Not a fully-stubbed happy path.
    monkeypatch.setattr(judge, "get_settings", lambda: SimpleNamespace(judge_model="claude-opus-4-8"))

    books = [CorpusBook("holmes", "A Study in Scarlet", "Sherlock Holmes is a consulting detective who lives at Baker Street.")]

    # Real loader on the injected books; a whitespace counter keeps it hermetic (no tiktoken needed).
    def real_loader(size: int):
        return load_for_token_target(size, books=books, count_tokens=lambda t: len(t.split()))

    anthropic = RecordingAnthropic([
        _generation_response("A consulting detective.", input_tokens=250, output_tokens=8),
        _judge_tool_response(1.0, "The answer conveys the same profession as the gold."),
    ])

    row = plain_llm.run_plain_llm_query(
        _question(), 3, anthropic_client=anthropic, corpus_loader=real_loader, settings=_settings()
    )

    expected_cost = cost.token_cost_usd("claude-sonnet-5", input_tokens=250, output_tokens=8)
    assert row.system is SystemName.PLAIN_LLM
    assert row.measured_or_extrapolated == "measured"
    assert row.accuracy == 1.0  # real judge parsed the real tool_use verdict
    assert row.judge_rationale
    assert row.cost_usd == pytest.approx(expected_cost)
    assert row.corpus_token_count == 11  # real loader's whitespace-token count of the whole book
    assert len(anthropic.calls) == 2  # one generation call, one judge call


def test_skip_path_composes_with_real_loader_token_count(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 9): the skip decision must compare against the SAME token axis corpus_loader produces —
    # not a parallel count. Drive the real loader with a tiny wall so a small real corpus trips it, and
    # prove the skip row's corpus_token_count is exactly what the loader measured.
    monkeypatch.setattr(plain_llm, "CONTEXT_WINDOW_TOKEN_LIMIT", 5)

    books = [CorpusBook("holmes", "A Study in Scarlet", "Sherlock Holmes is a consulting detective in London.")]

    def real_loader(size: int):
        return load_for_token_target(size, books=books, count_tokens=lambda t: len(t.split()))

    row = plain_llm.run_plain_llm_query(
        _question(), 3, anthropic_client=RecordingAnthropic([]), corpus_loader=real_loader, settings=_settings()
    )

    assert row.skipped_reason == "exceeds_context_window"
    assert row.corpus_token_count == 8  # the loader's own whitespace-token count, > the patched wall of 5
