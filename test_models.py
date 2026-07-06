"""Tests for the shared Pydantic schema in models.py.

These pin the data-layer contract that all three sweep paths write into: the
three-system enum, the cumulative-token sweep axis, the measured-vs-extrapolated
boundary, and the plain-LLM context-wall skip row. Each test encodes *why* the
shape matters (Rule 9), not just that a field happens to exist.
"""

import pytest
from pydantic import ValidationError

from models import QuestionTier, ResultRow, SystemName


def _base_row_kwargs(system: SystemName) -> dict:
    """A fully-populated, valid ResultRow kwargs dict for the given system."""
    return {
        "system": system,
        "corpus_size": 3,
        "corpus_token_count": 1_000_000,
        "measured_or_extrapolated": "measured",
        "question_id": "holmes_lookup_1",
        "tier": QuestionTier.LOOKUP,
        "latency_seconds": 1.5,
        "cost_usd": 0.02,
        "accuracy": 1.0,
        "answer_text": "A consulting detective.",
        "judge_rationale": "Matches the gold answer.",
    }


def test_system_name_has_all_three_paths() -> None:
    # Why: the sweep compares plain-LLM vs RAG vs wiki; a missing member silently
    # drops a whole path from the benchmark.
    assert {member.value for member in SystemName} == {"plain_llm", "rag", "wiki"}


@pytest.mark.parametrize("system", list(SystemName))
def test_result_row_constructs_and_dict_round_trips_for_each_system(system: SystemName) -> None:
    # Why: every graded cell is a ResultRow (PRD #16); the results-JSON shape is
    # built from model_dump, so a dict round-trip must be lossless for all systems.
    row = ResultRow(**_base_row_kwargs(system))
    reconstructed = ResultRow(**row.model_dump())
    assert reconstructed == row
    assert reconstructed.system is system


@pytest.mark.parametrize("system", list(SystemName))
def test_result_row_csv_round_trips_for_each_system(system: SystemName) -> None:
    # Why: one graded cell == one CSV line is the Python<->UI seam; a CSV row must
    # reconstruct to an equal ResultRow (no field lost or reordered).
    row = ResultRow(**_base_row_kwargs(system))
    header = ResultRow.csv_header()
    values = row.to_csv_row()
    assert len(header) == len(values)
    reconstructed = ResultRow(**dict(zip(header, values, strict=True)))
    assert reconstructed == row


def test_result_row_carries_new_sweep_fields_and_retains_corpus_size() -> None:
    # Why: corpus_token_count is the true sweep axis (PRD #18/#20) while corpus_size
    # is retained only as an internal ordering detail — both must coexist.
    row = ResultRow(**_base_row_kwargs(SystemName.RAG))
    assert row.corpus_token_count == 1_000_000
    assert row.measured_or_extrapolated == "measured"
    assert row.skipped_reason == ""  # defaults to empty for a normal graded row
    assert row.corpus_size == 3


def test_plain_llm_context_wall_skip_row_validates() -> None:
    # Why: past the context window the plain-LLM path emits a skipped row with a
    # reason (PRD #11) instead of raising; it must be a representable, valid row.
    kwargs = _base_row_kwargs(SystemName.PLAIN_LLM)
    kwargs["skipped_reason"] = "exceeds_context_window"
    row = ResultRow(**kwargs)
    assert row.skipped_reason == "exceeds_context_window"
    assert row.system is SystemName.PLAIN_LLM


def test_measured_or_extrapolated_rejects_non_literal_value() -> None:
    # Why: the measured-vs-extrapolated boundary must never be blurred (Rule 12);
    # only the two literals are admissible.
    kwargs = _base_row_kwargs(SystemName.WIKI)
    kwargs["measured_or_extrapolated"] = "guessed"
    with pytest.raises(ValidationError):
        ResultRow(**kwargs)


def test_error_row_validates_with_new_required_fields_present() -> None:
    # Why: a failed cell becomes an error ResultRow, not a dropped row — it must
    # still validate with the new required fields (corpus_token_count etc.) present.
    kwargs = _base_row_kwargs(SystemName.RAG)
    kwargs["error"] = "pinecone_timeout"
    kwargs["answer_text"] = ""
    kwargs["accuracy"] = 0.0
    row = ResultRow(**kwargs)
    assert row.error == "pinecone_timeout"
    assert row.corpus_token_count == 1_000_000
    assert row.measured_or_extrapolated == "measured"
