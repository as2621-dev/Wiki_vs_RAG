"""Tests for the resumable sweep runner (issue #9).

Every path callable is a FAKE injected through ``run_sweep(paths=...)`` — no test ever touches a live
Anthropic / Voyage / Pinecone API or ``claude -p`` (CLAUDE.md §6). The tests exercise the runner's own
contract: one row per applicable cell, crash-resume that never re-runs a completed cell, per-cell
failure isolation, only-answerable-questions-per-size, the ResultRow output schema, and reconstructable
structured logs.
"""

import csv
import json
from types import SimpleNamespace

import pytest
import structlog

from models import BenchmarkQuestion, QuestionTier, ResultRow, SystemName
from sweep import (
    PROGRESS_STORE_FILENAME,
    RESULTS_CSV_FILENAME,
    RESULTS_JSON_FILENAME,
    build_default_paths,
    run_sweep,
)


# ─── Fixtures / fakes ─────────────────────────────────────────


def _question(question_id: str, book_key: str) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        question_id=question_id,
        book_key=book_key,
        tier=QuestionTier.LOOKUP,
        question_text=f"Q for {book_key}?",
        gold_answer="gold",
    )


def _corpus_load(*, total_token_count: int, source_keys: list[str]) -> SimpleNamespace:
    """A minimal CorpusLoad stand-in exposing the three attributes the runner reads."""
    return SimpleNamespace(
        total_token_count=total_token_count,
        sources=[SimpleNamespace(book_key=key, text="text") for key in source_keys],
        loaded_source_keys=list(source_keys),
    )


def _corpus_loader_for(mapping: dict[int, SimpleNamespace]):
    """Return a ``(size) -> CorpusLoad`` fake driven by a size->CorpusLoad mapping."""
    return lambda size: mapping[size]


def _measured_row(system: SystemName, corpus_load: SimpleNamespace, question: BenchmarkQuestion) -> ResultRow:
    """Build a plausible measured ResultRow the way a real path would for a cell."""
    return ResultRow(
        system=system,
        corpus_size=len(corpus_load.sources),
        corpus_token_count=corpus_load.total_token_count,
        measured_or_extrapolated="measured",
        question_id=question.question_id,
        tier=question.tier,
        latency_seconds=1.0,
        cost_usd=0.5,
        accuracy=1.0,
        answer_text=f"{system.value} answer",
        judge_rationale="ok",
    )


def _recording_path(system: SystemName, calls: list[tuple[int, str]]):
    """A fake path that records (token_count, question_id) per call and returns a measured row."""

    def _path(question: BenchmarkQuestion, corpus_load: SimpleNamespace) -> ResultRow:
        calls.append((corpus_load.total_token_count, question.question_id))
        return _measured_row(system, corpus_load, question)

    return _path


# ─── Acceptance criterion 1: happy path -> one row per cell + results.csv AND results.json ──


def test_happy_path_one_row_per_cell_writes_csv_and_json(tmp_path) -> None:
    # Why (Rule 9, story #16): the whole point — a sweep over (sizes × systems × questions) emits one
    # ResultRow per applicable cell and persists BOTH results.csv and results.json. If the runner
    # dropped a system or forgot a file, downstream (#10 / the UI) would silently read a truncated set.
    loads = {
        100_000: _corpus_load(total_token_count=100_000, source_keys=["holmes"]),
        500_000: _corpus_load(total_token_count=500_000, source_keys=["holmes"]),
    }
    questions = [_question("q1", "holmes"), _question("q2", "holmes")]
    paths = {system: _recording_path(system, []) for system in SystemName}

    result = run_sweep(
        [100_000, 500_000],
        paths=paths,
        out_dir=tmp_path,
        corpus_loader=_corpus_loader_for(loads),
        questions_provider=lambda keys: questions,
    )

    # 2 sizes × 3 systems × 2 questions = 12 cells.
    assert result.total_rows == 12
    assert result.newly_run_cells == 12
    assert result.resumed_cells == 0
    csv_path = tmp_path / RESULTS_CSV_FILENAME
    json_path = tmp_path / RESULTS_JSON_FILENAME
    assert csv_path.exists() and json_path.exists()

    parsed = json.loads(json_path.read_text())
    assert len(parsed) == 12
    assert {row["system"] for row in parsed} == {"plain_llm", "rag", "wiki"}


# ─── Acceptance criterion 2: resumable — crash mid-sweep resumes, done cells not re-run ──


def test_crash_mid_sweep_resumes_and_does_not_rerun_completed_cells(tmp_path) -> None:
    # Why (Rule 9, story #17): a multi-day headless run WILL be killed mid-way. Resuming must skip every
    # already-completed cell (never re-calling the expensive path) and finish the rest — a runner that
    # re-ran done cells would double-bill and could double-count rows.
    load = _corpus_load(total_token_count=100_000, source_keys=["holmes"])
    questions = [_question("q1", "holmes"), _question("q2", "holmes"), _question("q3", "holmes")]

    first_run_calls: list[str] = []

    def _crashing_path(question: BenchmarkQuestion, corpus_load: SimpleNamespace) -> ResultRow:
        if question.question_id == "q2":
            # Simulate a process kill (BaseException — not caught by the runner's per-cell guard) AFTER
            # q1 has been durably appended but BEFORE q2 is recorded.
            raise KeyboardInterrupt("killed mid-sweep")
        first_run_calls.append(question.question_id)
        return _measured_row(SystemName.RAG, corpus_load, question)

    with pytest.raises(KeyboardInterrupt):
        run_sweep(
            [100_000],
            paths={SystemName.RAG: _crashing_path},
            out_dir=tmp_path,
            corpus_loader=_corpus_loader_for({100_000: load}),
            questions_provider=lambda keys: questions,
        )
    assert first_run_calls == ["q1"]  # only q1 completed before the crash
    assert (tmp_path / PROGRESS_STORE_FILENAME).exists()

    second_run_calls: list[str] = []

    def _resume_path(question: BenchmarkQuestion, corpus_load: SimpleNamespace) -> ResultRow:
        second_run_calls.append(question.question_id)
        return _measured_row(SystemName.RAG, corpus_load, question)

    result = run_sweep(
        [100_000],
        paths={SystemName.RAG: _resume_path},
        out_dir=tmp_path,
        corpus_loader=_corpus_loader_for({100_000: load}),
        questions_provider=lambda keys: questions,
    )

    # q1 was already done — NOT re-called; only q2 and q3 run on resume.
    assert second_run_calls == ["q2", "q3"]
    assert result.resumed_cells == 1
    assert result.newly_run_cells == 2
    assert result.total_rows == 3
    # Every question ends up in the final results exactly once (no double-count).
    parsed = json.loads((tmp_path / RESULTS_JSON_FILENAME).read_text())
    assert sorted(row["question_id"] for row in parsed) == ["q1", "q2", "q3"]


def test_rerun_of_completed_sweep_reruns_nothing(tmp_path) -> None:
    # Why: idempotency — running a finished sweep again is a no-op over the paths (duplicate-cell guard),
    # so an accidental re-invocation can never re-bill or corrupt the results.
    load = _corpus_load(total_token_count=100_000, source_keys=["holmes"])
    questions = [_question("q1", "holmes")]
    common = dict(
        out_dir=tmp_path,
        corpus_loader=_corpus_loader_for({100_000: load}),
        questions_provider=lambda keys: questions,
    )
    run_sweep([100_000], paths={SystemName.RAG: _recording_path(SystemName.RAG, [])}, **common)

    second_calls: list[tuple[int, str]] = []
    result = run_sweep([100_000], paths={SystemName.RAG: _recording_path(SystemName.RAG, second_calls)}, **common)
    assert second_calls == []
    assert result.newly_run_cells == 0
    assert result.resumed_cells == 1
    assert result.total_rows == 1


# ─── Acceptance criterion 3: per-cell failure isolation ──


def test_one_system_raising_becomes_error_row_and_does_not_halt_others(tmp_path) -> None:
    # Why (Rule 12, story #17): a single flaky system on a cell must NOT take out the other systems'
    # rows for that cell or halt the sweep — it becomes an error ResultRow so the failure is in the data.
    load = _corpus_load(total_token_count=100_000, source_keys=["holmes"])
    questions = [_question("q1", "holmes")]

    def _raising_rag(question: BenchmarkQuestion, corpus_load: SimpleNamespace) -> ResultRow:
        raise RuntimeError("pinecone exploded")

    paths = {
        SystemName.RAG: _raising_rag,
        SystemName.PLAIN_LLM: _recording_path(SystemName.PLAIN_LLM, []),
        SystemName.WIKI: _recording_path(SystemName.WIKI, []),
    }
    result = run_sweep(
        [100_000],
        paths=paths,
        out_dir=tmp_path,
        corpus_loader=_corpus_loader_for({100_000: load}),
        questions_provider=lambda keys: questions,
    )

    assert result.total_rows == 3  # sweep did not halt — all three systems produced a row
    parsed = {row["system"]: row for row in json.loads((tmp_path / RESULTS_JSON_FILENAME).read_text())}
    assert "RuntimeError: pinecone exploded" in parsed["rag"]["error"]
    assert parsed["rag"]["corpus_token_count"] == 100_000  # error row still carries the real axis value
    assert parsed["plain_llm"]["error"] == ""  # the other systems are unaffected
    assert parsed["wiki"]["error"] == ""


# ─── Acceptance criterion 4: only-answerable-questions-per-size ──


def test_only_questions_answerable_at_a_size_are_run(tmp_path) -> None:
    # Why (correctness, PRD decision #3): a question must only run at sizes where its anchor work is
    # loaded. The runner enumerates questions off the loaded source keys, so a size that loaded only
    # 'holmes' must never run the 'dracula' question.
    from questions import questions_for_token_target

    loads = {
        100_000: _corpus_load(total_token_count=100_000, source_keys=["holmes"]),
        500_000: _corpus_load(total_token_count=500_000, source_keys=["holmes", "dracula"]),
    }
    calls: list[tuple[int, str]] = []
    run_sweep(
        [100_000, 500_000],
        paths={SystemName.RAG: _recording_path(SystemName.RAG, calls)},
        out_dir=tmp_path,
        corpus_loader=_corpus_loader_for(loads),
        questions_provider=questions_for_token_target,  # the REAL answerable-question filter
    )
    at_100k = {qid for ttc, qid in calls if ttc == 100_000}
    at_500k = {qid for ttc, qid in calls if ttc == 500_000}
    assert all(qid.startswith("holmes") for qid in at_100k)  # dracula not answerable yet at 100k
    assert any(qid.startswith("dracula") for qid in at_500k)  # dracula answerable once loaded


# ─── Acceptance criterion 5: measured/extrapolated + corpus_token_count recorded + boundary logged ──


def test_each_row_records_boundary_and_token_count_and_logs_them(tmp_path) -> None:
    # Why (story #18/#26): every row must carry measured_or_extrapolated + corpus_token_count, and the
    # boundary must be a LOGGED field per row so a multi-day run's honesty is reconstructable from logs.
    measured = _corpus_load(total_token_count=100_000, source_keys=["holmes"])
    extrapolated_load = _corpus_load(total_token_count=50_000_000, source_keys=["holmes"])
    question = _question("q1", "holmes")

    def _wiki_path(q: BenchmarkQuestion, cl: SimpleNamespace) -> ResultRow:
        kind = "extrapolated" if cl.total_token_count > 5_000_000 else "measured"
        row = _measured_row(SystemName.WIKI, cl, q)
        return row.model_copy(update={"measured_or_extrapolated": kind})

    with structlog.testing.capture_logs() as logs:
        run_sweep(
            [100_000, 50_000_000],
            paths={SystemName.WIKI: _wiki_path},
            out_dir=tmp_path,
            corpus_loader=_corpus_loader_for({100_000: measured, 50_000_000: extrapolated_load}),
            questions_provider=lambda keys: [question],
        )

    parsed = json.loads((tmp_path / RESULTS_JSON_FILENAME).read_text())
    by_ttc = {row["corpus_token_count"]: row for row in parsed}
    assert by_ttc[100_000]["measured_or_extrapolated"] == "measured"
    assert by_ttc[50_000_000]["measured_or_extrapolated"] == "extrapolated"

    completed = [log for log in logs if log["event"] == "sweep_cell_completed"]
    assert len(completed) == 2
    for log in completed:
        assert "measured_or_extrapolated" in log and "corpus_token_count" in log


# ─── Acceptance criterion: reconstructable structured logs ──


def test_run_is_reconstructable_from_structured_logs(tmp_path) -> None:
    # Why (story #26): a headless run must be reconstructable from logs alone — every cell start and
    # finish is a structured event, bracketed by sweep_started / sweep_completed.
    load = _corpus_load(total_token_count=100_000, source_keys=["holmes"])
    questions = [_question("q1", "holmes"), _question("q2", "holmes")]
    with structlog.testing.capture_logs() as logs:
        run_sweep(
            [100_000],
            paths={SystemName.RAG: _recording_path(SystemName.RAG, [])},
            out_dir=tmp_path,
            corpus_loader=_corpus_loader_for({100_000: load}),
            questions_provider=lambda keys: questions,
        )
    events = [log["event"] for log in logs]
    assert events.count("sweep_cell_started") == 2
    assert events.count("sweep_cell_completed") == 2
    assert "sweep_started" in events and "sweep_completed" in events


# ─── B3.5 integration: real ResultRow serialization round-trips through results.csv ──


def test_results_csv_round_trips_through_real_resultrow_serialization(tmp_path) -> None:
    # Why (contract for #10): the CSV columns must be EXACTLY ResultRow's serialization — #10 reads this
    # as a stable schema. Parse the CSV back into ResultRow objects and assert it reconstructs equal rows
    # (answer text with a comma must survive quoting, not shift columns).
    load = _corpus_load(total_token_count=100_000, source_keys=["holmes"])
    question = _question("q1", "holmes")

    def _comma_answer_path(q: BenchmarkQuestion, cl: SimpleNamespace) -> ResultRow:
        row = _measured_row(SystemName.RAG, cl, q)
        return row.model_copy(update={"answer_text": "a, b, and c\nnext line"})

    run_sweep(
        [100_000],
        paths={SystemName.RAG: _comma_answer_path},
        out_dir=tmp_path,
        corpus_loader=_corpus_loader_for({100_000: load}),
        questions_provider=lambda keys: [question],
    )

    with (tmp_path / RESULTS_CSV_FILENAME).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ResultRow.csv_header()
    reconstructed = ResultRow(**dict(zip(ResultRow.csv_header(), rows[1], strict=True)))
    assert reconstructed.answer_text == "a, b, and c\nnext line"
    assert reconstructed.system == SystemName.RAG
    assert reconstructed.corpus_token_count == 100_000


def test_torn_progress_line_is_skipped_on_resume(tmp_path) -> None:
    # Why (data-integrity): a crash can leave a half-written final JSONL line; the resume must ignore it
    # (loudly) and still recover every complete prior cell, never abort the whole resume.
    store = tmp_path / PROGRESS_STORE_FILENAME
    good = _measured_row(SystemName.RAG, _corpus_load(total_token_count=100_000, source_keys=["holmes"]), _question("q1", "holmes"))
    store.write_text(
        json.dumps({"cell_key": "100000:rag:q1", "row": good.model_dump(mode="json")}) + "\n"
        + '{"cell_key": "100000:rag:q2", "row": {truncated',  # torn final line
        encoding="utf-8",
    )

    load = _corpus_load(total_token_count=100_000, source_keys=["holmes"])
    calls: list[tuple[int, str]] = []
    with structlog.testing.capture_logs() as logs:
        result = run_sweep(
            [100_000],
            paths={SystemName.RAG: _recording_path(SystemName.RAG, calls)},
            out_dir=tmp_path,
            corpus_loader=_corpus_loader_for({100_000: load}),
            questions_provider=lambda keys: [_question("q1", "holmes"), _question("q2", "holmes")],
        )
    assert calls == [(100_000, "q2")]  # q1 recovered from the good line; q2 re-run (its line was torn)
    assert result.resumed_cells == 1
    assert any(log["event"] == "sweep_progress_line_skipped" for log in logs)


def test_duplicate_progress_line_is_not_double_counted(tmp_path) -> None:
    # Why (data-integrity): a duplicate cell key in the store must load the row once — the guard prevents
    # a double-append (however it arose) from inflating the row count.
    store = tmp_path / PROGRESS_STORE_FILENAME
    row = _measured_row(SystemName.RAG, _corpus_load(total_token_count=100_000, source_keys=["holmes"]), _question("q1", "holmes"))
    line = json.dumps({"cell_key": "100000:rag:q1", "row": row.model_dump(mode="json")}) + "\n"
    store.write_text(line + line, encoding="utf-8")  # same cell twice

    result = run_sweep(
        [100_000],
        paths={SystemName.RAG: _recording_path(SystemName.RAG, [])},
        out_dir=tmp_path,
        corpus_loader=_corpus_loader_for({100_000: _corpus_load(total_token_count=100_000, source_keys=["holmes"])}),
        questions_provider=lambda keys: [_question("q1", "holmes")],
    )
    assert result.total_rows == 1  # not 2


# ─── build_default_paths: once-per-size setup memoization ──


def test_build_default_paths_indexes_rag_once_per_size(tmp_path) -> None:
    # Why (architecture): RAG must embed+upsert a size ONCE, then answer each of its questions against
    # the indexed namespace — re-indexing per question would repay the one-time embedding cost per cell.
    index_calls: list[int] = []
    query_calls: list[str] = []
    paths = build_default_paths(
        index_corpus=lambda cl: index_calls.append(cl.total_token_count),
        run_rag_query=lambda q, corpus_token_count, corpus_size: (
            query_calls.append(q.question_id),
            _measured_row(SystemName.RAG, _corpus_load(total_token_count=corpus_token_count, source_keys=["holmes"]), q),
        )[1],
    )
    rag_path = paths[SystemName.RAG]
    load = _corpus_load(total_token_count=100_000, source_keys=["holmes"])
    rag_path(_question("q1", "holmes"), load)
    rag_path(_question("q2", "holmes"), load)
    assert index_calls == [100_000]  # indexed once for the size
    assert query_calls == ["q1", "q2"]  # queried per question


def test_build_default_paths_wiki_incomplete_ingest_yields_error_row(tmp_path) -> None:
    # Why (Rule 12): a rate-limit cutoff leaves the wiki partial — the path must return an error row
    # telling the caller to resume, never answer against a partial wiki.
    answered: list[str] = []
    paths = build_default_paths(
        ingest_corpus=lambda cl: SimpleNamespace(complete=False),
        run_wiki_query=lambda q, corpus_token_count, corpus_size: answered.append(q.question_id),
    )
    row = paths[SystemName.WIKI](_question("q1", "holmes"), _corpus_load(total_token_count=100_000, source_keys=["holmes"]))
    assert "wiki_ingest_incomplete" in row.error
    assert answered == []  # never answered against the partial wiki


def test_build_default_paths_plain_llm_feeds_loaded_corpus_through(tmp_path) -> None:
    # Why: plain-LLM has no per-size setup; the adapter must feed the already-loaded corpus straight to
    # the path (via an overriding corpus_loader) rather than re-loading by token target.
    seen = {}

    def _fake_plain(question, size, *, corpus_loader):
        seen["size"] = size
        seen["loaded"] = corpus_loader(size)
        return _measured_row(SystemName.PLAIN_LLM, seen["loaded"], question)

    paths = build_default_paths(run_plain_llm_query=_fake_plain)
    load = _corpus_load(total_token_count=100_000, source_keys=["holmes"])
    paths[SystemName.PLAIN_LLM](_question("q1", "holmes"), load)
    assert seen["size"] == 100_000
    assert seen["loaded"] is load  # the loaded corpus was passed straight through
