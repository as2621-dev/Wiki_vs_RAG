"""Tests for the wiki path in wiki.py.

The ``claude -p`` subprocess is ALWAYS mocked via an injected ``cli_runner``; NO test invokes a real
CLI (CLAUDE.md §6) — one test proves it by asserting the module never falls back to the real runner.
Each test encodes *why* the behaviour matters (Rule 9): the path must yield a trustworthy MEASURED
``ResultRow`` carrying the CLI's ``total_cost_usd`` (API-equivalent) and wall-clock; ingest must RESUME
from a checkpoint after a rate-limit cutoff rather than re-ingesting; sizes past the measured cap must
be marked ``extrapolated`` (never silent); and a CLI failure must degrade to an ``error`` row, not crash.
"""

import json
from types import SimpleNamespace

import pytest
import structlog

import judge
import wiki
from models import BenchmarkQuestion, JudgeVerdict, QuestionTier, ResultRow, SystemName


# ─── Boundary fakes ───────────────────────────────────────────────────────────


class RecordingCliRunner:
    """``claude -p`` boundary: returns queued CliResults (or raises queued exceptions), records calls."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    def __call__(self, args, *, input_text=None, timeout=None) -> wiki.CliResult:
        self.calls.append({"args": args, "input_text": input_text, "timeout": timeout})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _cli_json(*, result="", total_cost_usd=0.0, duration_ms=None, is_error=False) -> wiki.CliResult:
    """A canned ``claude -p --output-format json`` success CliResult (mirrors the CLI's JSON shape)."""
    payload = {"type": "result", "is_error": is_error, "result": result, "total_cost_usd": total_cost_usd}
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    return wiki.CliResult(returncode=0, stdout=json.dumps(payload), stderr="")


def _judge_tool_response(score: float, rationale: str) -> SimpleNamespace:
    """A canned judge response whose forced tool_use block carries the verdict (real judge parses it)."""
    block = SimpleNamespace(type="tool_use", name=judge.JUDGE_TOOL_NAME, input={"score": score, "rationale": rationale}, id="toolu_1")
    return SimpleNamespace(stop_reason="tool_use", content=[block])


class RecordingJudgeClient:
    """Anthropic judge boundary: ``.messages.create(**kwargs)`` returns queued responses."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.messages = self

    def create(self, **kwargs) -> object:
        return self._responses.pop(0)


def _settings(**overrides) -> SimpleNamespace:
    """Fake settings so tests need no .env / real keys (wiki reads only CLI + MCP + checkpoint config)."""
    base = {"claude_cli_binary": "claude", "llmwiki_mcp_config_path": "/tmp/mcp.json", "wiki_checkpoint_dir": ""}
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


def _source(book_key: str, text: str = "some text") -> SimpleNamespace:
    """A minimal LoadedSource stand-in exposing the two attributes wiki reads (book_key, text)."""
    return SimpleNamespace(book_key=book_key, text=text)


def _corpus_load(*, total_token_count: int, source_keys: list[str]) -> SimpleNamespace:
    """A minimal CorpusLoad stand-in exposing total_token_count + sources (each with book_key/text)."""
    return SimpleNamespace(total_token_count=total_token_count, sources=[_source(key) for key in source_keys])


# ─── Happy path: ingest + answer -> measured ResultRow with cost AND latency ──


def test_happy_path_ingest_and_answer_measured_row_with_cost_and_latency(tmp_path, monkeypatch) -> None:
    # Why (Rule 9, story #14): the point of the slice — claude -p ingests the corpus and answers, and
    # the row carries the CLI's total_cost_usd (API-equivalent, non-zero even though the author pays $0)
    # AND the CLI-reported wall-clock. That measured row is the wiki's cell of the cost/accuracy chart.
    monkeypatch.setattr(wiki, "grade_answer", lambda answer, gold, client=None: JudgeVerdict(score=1.0, rationale="matches gold"))
    corpus = _corpus_load(total_token_count=500_000, source_keys=["holmes", "dracula"])
    runner = RecordingCliRunner([
        _cli_json(result="ingested holmes", total_cost_usd=0.10),   # ingest source 1
        _cli_json(result="ingested dracula", total_cost_usd=0.12),  # ingest source 2
        _cli_json(result="A consulting detective.", total_cost_usd=0.03, duration_ms=4200),  # answer
    ])

    row = wiki.run_wiki_slice(
        _question(), corpus, cli_runner=runner, checkpoint_dir=str(tmp_path), settings=_settings()
    )

    assert isinstance(row, ResultRow)
    assert row.system is SystemName.WIKI
    assert row.measured_or_extrapolated == "measured"
    assert row.corpus_token_count == 500_000
    assert row.accuracy == 1.0
    assert row.answer_text == "A consulting detective."
    assert row.error == ""
    assert row.cost_usd == 0.03  # the ANSWER call's API-equivalent total_cost_usd (one-time ingest cost excluded)
    assert row.latency_seconds == 4.2  # CLI duration_ms (4200) -> seconds, not a measured stopwatch span
    # Three claude -p invocations: two ingest, one answer — no shell string, an arg LIST every time.
    assert len(runner.calls) == 3
    assert all(isinstance(call["args"], list) for call in runner.calls)


def test_answer_prompt_uses_arg_list_and_corpus_never_in_argv(tmp_path) -> None:
    # Why (security/CSO): corpus content must travel via stdin, never argv (no command injection). The
    # ingest call passes the source TEXT as input_text and a fixed instruction as the -p arg.
    corpus = _corpus_load(total_token_count=100_000, source_keys=["holmes"])
    secret_text = "SENSITIVE CORPUS TEXT that must not appear on the command line"
    corpus.sources[0].text = secret_text
    runner = RecordingCliRunner([
        _cli_json(result="ok", total_cost_usd=0.05),
        _cli_json(result="ans", total_cost_usd=0.01, duration_ms=10),
    ])

    wiki.run_wiki_slice(
        _question(), corpus, cli_runner=runner, checkpoint_dir=str(tmp_path), judge_client=RecordingJudgeClient([_judge_tool_response(1.0, "ok")]), settings=_settings()
    )

    ingest_call = runner.calls[0]
    assert ingest_call["input_text"] == secret_text  # corpus via stdin
    assert secret_text not in " ".join(ingest_call["args"])  # NEVER in argv
    assert ingest_call["timeout"] == wiki.CLAUDE_CLI_TIMEOUT_SECONDS  # bounded, doesn't hang forever


# ─── Resumable ingest: rate-limit cutoff resumes from checkpoint, not restart ─


def test_ingest_resumes_from_checkpoint_after_rate_limit_cutoff(tmp_path) -> None:
    # Why (Rule 9, story #13): THE resumability behaviour. A usage-window cutoff mid-ingest must leave a
    # checkpoint so a second run RESUMES — already-ingested sources are NOT re-ingested (asserted by which
    # sources the runner is called with), and the remaining source is picked up.
    corpus = _corpus_load(total_token_count=300_000, source_keys=["a", "b", "c"])

    # First run: ingest "a", then a rate-limit cutoff before "b".
    first_runner = RecordingCliRunner([
        _cli_json(result="ingested a", total_cost_usd=0.10),
        wiki.WikiRateLimitError("usage limit reached"),
    ])
    first = wiki.ingest_corpus(corpus, cli_runner=first_runner, checkpoint_dir=str(tmp_path), settings=_settings())

    assert first.interrupted is True
    assert first.complete is False
    assert first.ingested_source_keys == ("a",)  # only "a" survived the cutoff
    assert first.resumed is False  # this run started fresh
    assert len(first_runner.calls) == 2  # tried "a" (ok) then "b" (rate-limited); "c" never attempted

    # Second run: RESUMES — "a" is skipped (not re-ingested), "b" and "c" are ingested.
    second_runner = RecordingCliRunner([
        _cli_json(result="ingested b", total_cost_usd=0.11),
        _cli_json(result="ingested c", total_cost_usd=0.12),
    ])
    second = wiki.ingest_corpus(corpus, cli_runner=second_runner, checkpoint_dir=str(tmp_path), settings=_settings())

    assert second.resumed is True  # continued from the prior checkpoint
    assert second.complete is True
    assert second.newly_ingested_count == 2  # only "b" and "c" this run — "a" was NOT re-done
    assert second.ingested_source_keys == ("a", "b", "c")
    assert len(second_runner.calls) == 2  # proves "a" was skipped, not re-ingested
    # Cumulative ingest cost carried across the resume (0.10 + 0.11 + 0.12).
    assert second.ingest_cost_usd == pytest.approx(0.33)


def test_ingest_is_idempotent_rerun_of_complete_ingest_is_noop(tmp_path) -> None:
    # Why (Rule 9): running a COMPLETE ingest again must be a no-op — no source re-ingested, no extra CLI
    # call — so an off-hours loop that re-enters a finished size does no redundant (rate-limited) work.
    corpus = _corpus_load(total_token_count=200_000, source_keys=["a", "b"])
    first_runner = RecordingCliRunner([_cli_json(total_cost_usd=0.1), _cli_json(total_cost_usd=0.1)])
    wiki.ingest_corpus(corpus, cli_runner=first_runner, checkpoint_dir=str(tmp_path), settings=_settings())

    noop_runner = RecordingCliRunner([])  # empty: any CLI call would IndexError -> proves none is made
    result = wiki.ingest_corpus(corpus, cli_runner=noop_runner, checkpoint_dir=str(tmp_path), settings=_settings())

    assert result.complete is True
    assert result.newly_ingested_count == 0
    assert noop_runner.calls == []  # no re-ingestion


# ─── Measured ceiling: within cap -> measured; past cap -> extrapolated ───────


def test_size_past_cap_is_extrapolated_and_makes_no_cli_call(monkeypatch) -> None:
    # Why (Rule 12): a size past the measured cap must be marked extrapolated and NOT measured — no
    # claude -p call, never silently dropped or mislabelled as measured.
    over_cap = wiki.MEASURED_TOKEN_CAP + 1
    runner = RecordingCliRunner([])  # empty: any CLI call would IndexError -> proves none is made

    row = wiki.run_wiki_query(_question(), corpus_token_count=over_cap, corpus_size=9, cli_runner=runner, settings=_settings())

    assert row.system is SystemName.WIKI
    assert row.measured_or_extrapolated == "extrapolated"
    assert row.corpus_token_count == over_cap
    assert row.error == ""  # extrapolation is not an error
    assert runner.calls == []  # no measurement past the cap


def test_size_at_cap_is_measured_boundary_is_exact(monkeypatch) -> None:
    # Why (Rule 9): the cap boundary must be exact — a size of exactly MEASURED_TOKEN_CAP is still
    # measured (answered), one token more is extrapolated.
    monkeypatch.setattr(wiki, "grade_answer", lambda answer, gold, client=None: JudgeVerdict(score=1.0, rationale="ok"))
    at_cap_runner = RecordingCliRunner([_cli_json(result="answered", total_cost_usd=0.02, duration_ms=100)])

    at_cap = wiki.run_wiki_query(
        _question(), corpus_token_count=wiki.MEASURED_TOKEN_CAP, corpus_size=5, cli_runner=at_cap_runner, settings=_settings()
    )
    over_cap = wiki.run_wiki_query(
        _question(), corpus_token_count=wiki.MEASURED_TOKEN_CAP + 1, corpus_size=5, cli_runner=RecordingCliRunner([]), settings=_settings()
    )

    assert at_cap.measured_or_extrapolated == "measured"  # exactly at the cap -> measured
    assert at_cap.answer_text == "answered"
    assert over_cap.measured_or_extrapolated == "extrapolated"  # one token over -> extrapolated


def test_extrapolation_boundary_is_logged_not_inferred() -> None:
    # Why (Rule 9): the measured/extrapolated boundary must be a LOGGED field, not something a downstream
    # reader infers — capture the structured log and assert the exact cap values were recorded.
    over_cap = wiki.MEASURED_TOKEN_CAP + 250_000
    with structlog.testing.capture_logs() as logs:
        wiki.run_wiki_query(_question(), corpus_token_count=over_cap, corpus_size=9, cli_runner=RecordingCliRunner([]), settings=_settings())

    events = [entry for entry in logs if entry["event"] == "wiki_extrapolated_past_cap"]
    assert len(events) == 1
    assert events[0]["corpus_token_count"] == over_cap
    assert events[0]["measured_token_cap"] == wiki.MEASURED_TOKEN_CAP
    assert events[0]["over_by"] == over_cap - wiki.MEASURED_TOKEN_CAP


# ─── Edge case: claude -p non-zero exit / partial wiki -> error row, no crash ─


def test_cli_nonzero_exit_becomes_error_row_with_fix_suggestion(monkeypatch) -> None:
    # Why (Rule 12): a claude -p non-zero exit / partial wiki must degrade to an error ResultRow (error
    # populated) with a fix_suggestion logged — NOT raise, NOT halt the sweep.
    failing_runner = RecordingCliRunner([wiki.CliResult(returncode=1, stdout="", stderr="wiki mcp server crashed")])

    with structlog.testing.capture_logs() as logs:
        row = wiki.run_wiki_query(_question(), corpus_token_count=500_000, corpus_size=1, cli_runner=failing_runner, settings=_settings())

    assert isinstance(row, ResultRow)
    assert row.system is SystemName.WIKI
    assert row.error != ""  # the failure is recorded on the row
    assert "wiki mcp server crashed" in row.error
    assert row.accuracy == 0.0
    assert row.cost_usd == 0.0
    fail_events = [entry for entry in logs if entry["event"] == "wiki_query_failed"]
    assert fail_events and fail_events[0]["fix_suggestion"]  # fix_suggestion present (CLAUDE.md §5)


def test_cli_unparseable_json_becomes_error_row(monkeypatch) -> None:
    # Why (Rule 12): a zero-exit but garbage-JSON response must NOT be silently treated as a $0 success —
    # it is an error row, so a broken CLI output can't under-report the cost gap or fake an answer.
    garbage_runner = RecordingCliRunner([wiki.CliResult(returncode=0, stdout="not json at all", stderr="")])

    row = wiki.run_wiki_query(_question(), corpus_token_count=500_000, corpus_size=1, cli_runner=garbage_runner, settings=_settings())

    assert row.error != ""
    assert "unparseable" in row.error.lower()


# ─── No real claude -p is ever invoked (subprocess mocked) ────────────────────


def test_no_real_claude_cli_is_invoked(monkeypatch, tmp_path) -> None:
    # Why (CLAUDE.md §6): prove no test path reaches the real subprocess — replace the real runner with a
    # tripwire that fails loudly, then drive the happy path with an INJECTED runner. The tripwire must
    # never fire, and the injected runner must receive all the calls.
    def _tripwire(*args, **kwargs):
        raise AssertionError("real _run_claude_cli was invoked — a test escaped the mock boundary")

    monkeypatch.setattr(wiki, "_run_claude_cli", _tripwire)
    monkeypatch.setattr(wiki, "grade_answer", lambda answer, gold, client=None: JudgeVerdict(score=1.0, rationale="ok"))
    corpus = _corpus_load(total_token_count=100_000, source_keys=["holmes"])
    injected = RecordingCliRunner([_cli_json(total_cost_usd=0.1), _cli_json(result="ans", total_cost_usd=0.01, duration_ms=5)])

    row = wiki.run_wiki_slice(_question(), corpus, cli_runner=injected, checkpoint_dir=str(tmp_path), settings=_settings())

    assert row.error == ""
    assert len(injected.calls) == 2  # all work went through the injected mock, not the real runner


# ─── Integration: real corpus_loader -> mocked CLI -> real judge -> real checkpoint round-trip ──


def test_integration_real_loader_real_judge_real_checkpoint_roundtrip(tmp_path, monkeypatch) -> None:
    # Why (B3.5): trace two levels out — a REAL corpus_loader (small fixture) feeds a mocked claude -p,
    # the REAL judge parses a mocked tool response into the score, a REAL ResultRow is assembled, and a
    # REAL checkpoint round-trips on disk (write then resume genuinely SKIPS done work, not just claims to).
    from corpus import CorpusBook
    from corpus_loader import load_for_token_target

    # The judge reads only judge_model from settings; patch it so the real judge needs no .env / real key.
    monkeypatch.setattr(judge, "get_settings", lambda: SimpleNamespace(judge_model="claude-opus-4-8"))

    books = [CorpusBook("holmes", "Holmes", "Sherlock Holmes is a consulting detective."),
             CorpusBook("dracula", "Dracula", "Count Dracula is a vampire from Transylvania.")]
    # Target 8 tokens forces BOTH sources to load (source 1 is 6 tokens < 8), so a mid-ingest cutoff is real.
    corpus_load = load_for_token_target(8, books=books, count_tokens=lambda text: len(text.split()))
    assert len(corpus_load.sources) == 2  # the small fixture really loaded both sources

    # First run: ingest source 1, then a usage cutoff before the rest.
    first_runner = RecordingCliRunner([_cli_json(result="ok", total_cost_usd=0.2), wiki.WikiRateLimitError("rate limit")])
    first = wiki.ingest_corpus(corpus_load, cli_runner=first_runner, checkpoint_dir=str(tmp_path), settings=_settings())
    assert first.complete is False

    # The checkpoint really landed on disk keyed by this size's token count.
    checkpoint = wiki.load_checkpoint(str(tmp_path), corpus_load.total_token_count)
    assert checkpoint.ingested_source_keys == list(first.ingested_source_keys)
    assert len(checkpoint.ingested_source_keys) >= 1

    # Resume: only the remaining sources are ingested (done work skipped), then answer + REAL judge.
    remaining = len(corpus_load.sources) - len(first.ingested_source_keys)
    second_runner = RecordingCliRunner([_cli_json(result="ok", total_cost_usd=0.2) for _ in range(remaining)])
    second = wiki.ingest_corpus(corpus_load, cli_runner=second_runner, checkpoint_dir=str(tmp_path), settings=_settings())
    assert second.complete is True
    assert second.newly_ingested_count == remaining  # exactly the not-yet-done sources
    assert len(second_runner.calls) == remaining  # the already-done source was NOT re-called

    answer_runner = RecordingCliRunner([_cli_json(result="A consulting detective.", total_cost_usd=0.03, duration_ms=1500)])
    judge_client = RecordingJudgeClient([_judge_tool_response(1.0, "matches the gold answer")])
    row = wiki.run_wiki_query(
        _question(), corpus_token_count=corpus_load.total_token_count, corpus_size=len(corpus_load.sources),
        cli_runner=answer_runner, judge_client=judge_client, settings=_settings(),
    )

    assert row.measured_or_extrapolated == "measured"
    assert row.accuracy == 1.0  # the REAL judge parsed the mocked tool response
    assert row.judge_rationale == "matches the gold answer"
    assert row.cost_usd == 0.03
    assert row.latency_seconds == 1.5
