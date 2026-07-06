"""Tests for the aggregate + extrapolate step (issue #10).

These encode WHY the emitted ``results.json`` is trustworthy (Rule 9), not just its shape: a projection
is structurally distinguishable from a measurement (``kind``), the plain-LLM context wall survives as a
``skipped_context`` point rather than a dropped or zeroed row, a thin fit is flagged low-confidence
rather than faked, and a cost series dominated by one expensive point is surfaced. The payload is
validated against the ``ResultsPayload`` contract model so a schema drift fails a test.
"""

import csv
import json
from pathlib import Path

import pytest

from aggregate import (
    COST_DOMINATION_FRACTION,
    DEFAULT_TARGET_TOKENS,
    MIN_MEASURED_POINTS_FOR_FIT,
    _least_squares,
    build_results_json,
)
from models import QuestionTier, ResultRow, ResultsPayload, SeriesPointKind, SystemName


def _row(
    system: SystemName,
    corpus_token_count: int,
    *,
    accuracy: float = 1.0,
    cost_usd: float = 0.5,
    question_id: str = "holmes_lookup_1",
    measured_or_extrapolated: str = "measured",
    skipped_reason: str = "",
    error: str = "",
    answer_text: str = "an answer",
) -> ResultRow:
    """Build a plausible graded ``ResultRow`` the way a real path would for one cell."""
    return ResultRow(
        system=system,
        corpus_size=1,
        corpus_token_count=corpus_token_count,
        measured_or_extrapolated=measured_or_extrapolated,
        question_id=question_id,
        tier=QuestionTier.LOOKUP,
        latency_seconds=1.0,
        cost_usd=cost_usd,
        accuracy=accuracy,
        answer_text=answer_text,
        judge_rationale="ok",
        skipped_reason=skipped_reason,
        error=error,
    )


def _rag_measured_rows() -> list[ResultRow]:
    """RAG measured across the sweep band — enough points for a confident fit."""
    return [
        _row(SystemName.RAG, 100_000, accuracy=0.9, cost_usd=0.10),
        _row(SystemName.RAG, 1_000_000, accuracy=0.8, cost_usd=0.30),
        _row(SystemName.RAG, 10_000_000, accuracy=0.7, cost_usd=0.60),
        _row(SystemName.RAG, 50_000_000, accuracy=0.6, cost_usd=0.90),
    ]


# ─── Least-squares primitive ────────────────────────────────────────────────────────────────────


def test_least_squares_recovers_a_known_line():
    assert _least_squares([0.0, 1.0, 2.0], [1.0, 3.0, 5.0]) == (2.0, 1.0)


def test_least_squares_returns_none_when_unfittable():
    assert _least_squares([5.0], [1.0]) is None  # one point
    assert _least_squares([2.0, 2.0], [1.0, 3.0]) is None  # no x spread


# ─── Acceptance 1: happy path -> fitted + extrapolated series + answers in the documented shape ────


def test_happy_path_emits_contract_shape_extrapolated_to_1b():
    payload = build_results_json(rows=_rag_measured_rows())

    assert isinstance(payload, ResultsPayload)  # validates against the api-contracts model (Rule 9)
    rag_accuracy = [p for p in payload.series.accuracy if p.system == SystemName.RAG]
    rag_cost = [p for p in payload.series.cost if p.system == SystemName.RAG]

    # Both series reach the 1B target as an extrapolated point.
    assert any(p.corpus_token_count == DEFAULT_TARGET_TOKENS for p in rag_accuracy)
    assert any(p.corpus_token_count == DEFAULT_TARGET_TOKENS for p in rag_cost)
    # Answers reflect the graded cell.
    assert payload.answers["holmes_lookup_1"]["100000"]["rag"].accuracy == 0.9


def test_out_path_writes_valid_json_that_reparses_to_the_contract(tmp_path: Path):
    out = tmp_path / "results.json"
    build_results_json(rows=_rag_measured_rows(), out_path=out)

    reparsed = ResultsPayload.model_validate_json(out.read_text(encoding="utf-8"))
    assert reparsed.summary.extrapolated_to_token_count == DEFAULT_TARGET_TOKENS


# ─── Acceptance 2: extrapolated points are structurally distinct from measured ───────────────────


def test_measured_and_extrapolated_points_are_distinctly_kinded():
    payload = build_results_json(rows=_rag_measured_rows())
    rag_accuracy = [p for p in payload.series.accuracy if p.system == SystemName.RAG]

    measured = [p for p in rag_accuracy if p.kind == SeriesPointKind.MEASURED]
    extrapolated = [p for p in rag_accuracy if p.kind == SeriesPointKind.EXTRAPOLATED]

    # Every measured point sits at or below the boundary; every extrapolated point strictly above it —
    # a reader can never mistake a projection for a measurement.
    boundary = max(p.corpus_token_count for p in measured)
    assert boundary == 50_000_000
    assert extrapolated, "expected projected points past the measured ceiling"
    assert all(p.corpus_token_count > boundary for p in extrapolated)
    assert all(p.corpus_token_count <= boundary for p in measured)


def test_meta_states_the_fit_method_and_boundary():
    payload = build_results_json(rows=_rag_measured_rows())
    rag_meta = next(m for m in payload.meta if m.system == SystemName.RAG)

    assert rag_meta.measured_boundary_token_count == 50_000_000
    assert "ln(tokens)" in rag_meta.accuracy_fit_method
    assert "tokens" in rag_meta.cost_fit_method
    assert rag_meta.low_confidence is False


# ─── Acceptance 3: plain-LLM context wall -> accuracy null + skipped_context, present not dropped ──


def test_plain_llm_context_wall_becomes_skipped_context_point():
    rows = [
        _row(SystemName.PLAIN_LLM, 100_000, accuracy=0.95, cost_usd=0.20),
        _row(SystemName.PLAIN_LLM, 1_000_000, accuracy=0.94, cost_usd=0.40),
        _row(
            SystemName.PLAIN_LLM,
            2_000_000,
            accuracy=0.0,
            cost_usd=0.0,
            skipped_reason="exceeds_context_window",
            answer_text="",
        ),
    ]
    payload = build_results_json(rows=rows)
    plain_points = [p for p in payload.series.accuracy if p.system == SystemName.PLAIN_LLM]

    wall = [p for p in plain_points if p.corpus_token_count == 2_000_000]
    assert len(wall) == 1, "the context-wall point must be present, not dropped"
    assert wall[0].accuracy is None  # null, never a zero score
    assert wall[0].kind == SeriesPointKind.SKIPPED_CONTEXT
    # The wall is not silently zeroed in the answers view either.
    assert payload.answers["holmes_lookup_1"]["2000000"]["plain_llm"].accuracy is None


def test_walled_system_is_not_extrapolated_past_its_wall():
    rows = [
        _row(SystemName.PLAIN_LLM, 100_000, accuracy=0.95),
        _row(SystemName.PLAIN_LLM, 1_000_000, accuracy=0.94),
        _row(SystemName.PLAIN_LLM, 2_000_000, skipped_reason="exceeds_context_window", answer_text=""),
    ]
    payload = build_results_json(rows=rows)
    plain_points = [p for p in payload.series.accuracy if p.system == SystemName.PLAIN_LLM]

    # A system that hit the wall must never sprout an extrapolated point — that would fake a run.
    assert not any(p.kind == SeriesPointKind.EXTRAPOLATED for p in plain_points)
    plain_meta = next(m for m in payload.meta if m.system == SystemName.PLAIN_LLM)
    assert "context wall" in plain_meta.accuracy_fit_method


# ─── Acceptance 4: too few measured points -> low-confidence flag, not a silent bad fit ──────────


def test_too_few_measured_points_flags_low_confidence():
    rows = [
        _row(SystemName.WIKI, 100_000, accuracy=0.7, cost_usd=1.0),
        _row(SystemName.WIKI, 1_000_000, accuracy=0.75, cost_usd=2.0),
    ]
    payload = build_results_json(rows=rows)
    wiki_meta = next(m for m in payload.meta if m.system == SystemName.WIKI)

    assert wiki_meta.measured_point_count < MIN_MEASURED_POINTS_FOR_FIT
    assert wiki_meta.low_confidence is True
    assert "LOW CONFIDENCE" in wiki_meta.accuracy_fit_method


# ─── Acceptance 5: cost fit dominated by one expensive point -> surfaced ─────────────────────────


def test_single_expensive_point_domination_is_surfaced():
    # One wiki size costs vastly more than the rest — it drives the whole linear fit.
    rows = [
        _row(SystemName.WIKI, 100_000, accuracy=0.7, cost_usd=1.0),
        _row(SystemName.WIKI, 1_000_000, accuracy=0.72, cost_usd=2.0),
        _row(SystemName.WIKI, 5_000_000, accuracy=0.74, cost_usd=1000.0),
    ]
    payload = build_results_json(rows=rows)
    wiki_meta = next(m for m in payload.meta if m.system == SystemName.WIKI)

    largest = 1000.0
    total = 1.0 + 2.0 + 1000.0
    assert largest >= COST_DOMINATION_FRACTION * total  # guards the fixture, not just the code
    assert wiki_meta.cost_domination_note != ""
    assert "5M" in wiki_meta.cost_domination_note


# ─── Data-integrity: an error cell is excluded from the mean but never vanished ──────────────────


def test_error_row_excluded_from_accuracy_mean_but_present_in_answers():
    rows = [
        _row(SystemName.RAG, 100_000, accuracy=1.0, question_id="q_ok"),
        _row(SystemName.RAG, 100_000, accuracy=0.0, question_id="q_err", error="Boom: timeout", answer_text=""),
    ]
    payload = build_results_json(rows=rows)
    rag_measured = [p for p in payload.series.accuracy if p.system == SystemName.RAG and p.kind == SeriesPointKind.MEASURED]

    # Mean is over the single graded row (1.0), not dragged to 0.5 by the failure.
    assert rag_measured[0].accuracy == 1.0
    # The failed cell is still visible in the answers view.
    assert "q_err" in payload.answers


# ─── Argument guard ──────────────────────────────────────────────────────────────────────────────


def test_requires_exactly_one_source():
    with pytest.raises(ValueError):
        build_results_json()
    with pytest.raises(ValueError):
        build_results_json("results.csv", rows=_rag_measured_rows())


# ─── Integration: sweep writes results.csv the real way -> aggregate reads it -> contract holds ───


def test_reads_a_real_sweep_results_csv_end_to_end(tmp_path: Path):
    """Prove the #9 -> #10 -> #11/#12 contract lines up: serialize rows exactly as the sweep does,
    read them back through the aggregate, and validate the emitted payload against the contract model
    with all three point kinds present and correctly typed."""
    rows = [
        # A full plain-LLM series that walls out, RAG measured across the band, wiki with a cap+skip mix.
        _row(SystemName.PLAIN_LLM, 100_000, accuracy=0.95, cost_usd=0.2),
        _row(SystemName.PLAIN_LLM, 1_000_000, accuracy=0.94, cost_usd=0.4),
        _row(SystemName.PLAIN_LLM, 2_000_000, skipped_reason="exceeds_context_window", answer_text=""),
        *_rag_measured_rows(),
        _row(SystemName.WIKI, 100_000, accuracy=0.7, cost_usd=5.0),
        _row(SystemName.WIKI, 1_000_000, accuracy=0.72, cost_usd=10.0),
        _row(SystemName.WIKI, 5_000_000, accuracy=0.74, cost_usd=40.0),
        _row(SystemName.WIKI, 50_000_000, accuracy=0.75, cost_usd=200.0, measured_or_extrapolated="extrapolated"),
    ]

    # Write results.csv byte-identically to how sweep._write_results does (header + to_csv_row).
    csv_path = tmp_path / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ResultRow.csv_header())
        for row in rows:
            writer.writerow(row.to_csv_row())

    out_path = tmp_path / "results.json"
    build_results_json(csv_path, out_path=out_path)

    # Re-parse the file back into the contract model — a drift in field names/nesting fails here.
    reparsed = ResultsPayload.model_validate(json.loads(out_path.read_text(encoding="utf-8")))
    kinds = {p.kind for p in reparsed.series.accuracy}
    assert SeriesPointKind.MEASURED in kinds
    assert SeriesPointKind.EXTRAPOLATED in kinds
    assert SeriesPointKind.SKIPPED_CONTEXT in kinds
    # The wiki pre-marked extrapolated row survived as an extrapolated cost point (not measured).
    wiki_50m = next(p for p in reparsed.series.cost if p.system == SystemName.WIKI and p.corpus_token_count == 50_000_000)
    assert wiki_50m.kind == SeriesPointKind.EXTRAPOLATED
    # The thesis' cost gap is computed at the 1B target.
    assert reparsed.summary.wiki_to_rag_cost_ratio_at_target is not None
