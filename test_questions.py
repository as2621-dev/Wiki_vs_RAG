"""Tests for the frozen golden question bank in questions.py.

These pin the credibility anchor of the whole benchmark (PRD M0 / stories #1–#3): a frozen,
human-verified set spanning both novels and movies across three tiers, checksum-locked so no
accidental edit can silently change a published result, and selectable by the corpus's cumulative
source order. Each test encodes *why* the shape matters (Rule 9), not just that it happens to hold.
"""

import pytest
from pydantic import ValidationError

from corpus import GUTENBERG_BOOKS
from models import BenchmarkQuestion, QuestionTier
from questions import (
    QUESTION_BANK,
    QUESTION_BANK_CHECKSUM,
    compute_question_bank_checksum,
    questions_for_token_target,
    verify_question_bank_checksum,
)

_NOVEL_KEYS: set[str] = {book_key for book_key, _title, _url in GUTENBERG_BOOKS}
_MOVIE_KEYS: set[str] = {q.book_key for q in QUESTION_BANK} - _NOVEL_KEYS


def _novel_questions() -> list[BenchmarkQuestion]:
    return [q for q in QUESTION_BANK if q.book_key in _NOVEL_KEYS]


def _movie_questions() -> list[BenchmarkQuestion]:
    return [q for q in QUESTION_BANK if q.book_key not in _NOVEL_KEYS]


def test_bank_is_sized_and_every_question_is_valid() -> None:
    # Why: the finding rests on ~30–50 hand-verified questions (PRD story #1); each must be a
    # real BenchmarkQuestion with a tier and a non-empty gold answer, or a graded row is junk.
    assert 30 <= len(QUESTION_BANK) <= 50
    for question in QUESTION_BANK:
        assert isinstance(question, BenchmarkQuestion)
        assert isinstance(question.tier, QuestionTier)
        assert question.gold_answer.strip(), question.question_id


def test_question_ids_are_unique() -> None:
    # Why: question_id keys ResultRow, the results JSON, and the checksum's canonical order — a
    # duplicate id would silently overwrite a graded cell downstream.
    ids = [q.question_id for q in QUESTION_BANK]
    assert len(ids) == len(set(ids))


def test_movies_and_novels_both_present() -> None:
    # Why: the golden key must span novels AND movie scripts (PRD story #1) so the corpus reaches
    # scale with checkable content from both source types.
    assert _NOVEL_KEYS, "no novel-anchored questions"
    assert _MOVIE_KEYS, "no movie-anchored questions"


def test_every_tier_is_represented_across_both_novels_and_movies() -> None:
    # Why: tier balance is what lets the chart show *where* each path breaks, not just an average
    # (PRD story #3). A tier missing from either source type would silently skew the comparison.
    all_tiers = set(QuestionTier)
    novel_tiers = {q.tier for q in _novel_questions()}
    movie_tiers = {q.tier for q in _movie_questions()}
    assert novel_tiers == all_tiers, f"novels missing tiers: {all_tiers - novel_tiers}"
    assert movie_tiers == all_tiers, f"movies missing tiers: {all_tiers - movie_tiers}"


def test_no_source_work_is_silently_missing_a_tier() -> None:
    # Why: every anchor work should probe all three tiers so no source contributes a lopsided
    # sample; a work with only lookup questions would understate retrieval degradation on it.
    tiers_by_key: dict[str, set[QuestionTier]] = {}
    for question in QUESTION_BANK:
        tiers_by_key.setdefault(question.book_key, set()).add(question.tier)
    for book_key, tiers in tiers_by_key.items():
        assert tiers == set(QuestionTier), f"{book_key} missing tiers: {set(QuestionTier) - tiers}"


def test_checksum_verifies_on_the_untouched_bank() -> None:
    # Why: the frozen bank must match its stored checksum (PRD story #2); a drift here means the
    # published key and the code disagree.
    assert compute_question_bank_checksum() == QUESTION_BANK_CHECKSUM
    assert verify_question_bank_checksum() is True


@pytest.mark.parametrize("tampered_field", ["question_text", "gold_answer", "tier"])
def test_checksum_fails_when_any_graded_field_is_edited(tampered_field: str) -> None:
    # Why: THIS is the credibility test (Rule 9, PRD story #2) — editing any field a published
    # result depends on must flip the checksum, so no silent edit can change a score.
    original = QUESTION_BANK[0]
    new_value = QuestionTier.TIMELINE if tampered_field == "tier" else "TAMPERED"
    if tampered_field == "tier" and original.tier is QuestionTier.TIMELINE:
        new_value = QuestionTier.LOOKUP
    tampered = original.model_copy(update={tampered_field: new_value})
    tampered_bank = [tampered, *QUESTION_BANK[1:]]
    assert compute_question_bank_checksum(tampered_bank) != QUESTION_BANK_CHECKSUM


def test_checksum_is_order_independent() -> None:
    # Why: the checksum must guard content, not list order — reordering the bank (a harmless edit)
    # must not trip the guard, or every reformat would demand a re-freeze and dull the alarm.
    reversed_bank = list(reversed(QUESTION_BANK))
    assert compute_question_bank_checksum(reversed_bank) == QUESTION_BANK_CHECKSUM


def test_checksum_is_deterministic_across_calls() -> None:
    # Why: a published checksum is worthless if it depends on volatile per-process state (e.g.
    # hash() salting) — recomputation must be identical every time.
    assert compute_question_bank_checksum() == compute_question_bank_checksum()


def test_selection_returns_only_questions_for_loaded_works() -> None:
    # Why: at a small sweep size only the earliest works are loaded (PRD decision #3); selection
    # must return exactly those questions and nothing anchored to an unloaded work.
    selected = questions_for_token_target(["holmes", "dracula"])
    assert {q.book_key for q in selected} == {"holmes", "dracula"}


def test_selection_never_returns_a_question_whose_anchor_is_not_loaded() -> None:
    # Why: returning a question for an unloaded work would ask about content the system can't have
    # retrieved — a false failure that would corrupt the accuracy curve.
    selected = questions_for_token_target(["holmes"])
    assert selected, "holmes questions should be selected"
    assert all(q.book_key == "holmes" for q in selected)
    assert "dracula" not in {q.book_key for q in selected}


def test_selection_composes_with_the_real_corpus_source_order() -> None:
    # Why: selection must depend on the *actual* corpus contract (GUTENBERG_BOOKS order), not a
    # fictional one — loading the first N works in that order must expose exactly their questions.
    ordered_keys = [book_key for book_key, _title, _url in GUTENBERG_BOOKS]
    first_two_loaded = ordered_keys[:2]
    selected = questions_for_token_target(first_two_loaded)
    assert {q.book_key for q in selected} == set(first_two_loaded)


def test_selection_with_no_loaded_works_returns_nothing() -> None:
    # Why: before any work is loaded there is nothing answerable; selection must return empty, not
    # the whole bank.
    assert questions_for_token_target([]) == []


def test_malformed_question_missing_gold_answer_is_rejected() -> None:
    # Why: a gold answer is the judge's reference; a question without one can't be graded and must
    # be rejected at construction, not discovered mid-sweep.
    with pytest.raises(ValidationError):
        BenchmarkQuestion(
            question_id="broken_1",
            book_key="holmes",
            tier=QuestionTier.LOOKUP,
            question_text="What is missing?",
        )  # type: ignore[call-arg]


def test_malformed_question_unknown_tier_is_rejected() -> None:
    # Why: the tier drives per-tier breakdown; an unknown tier would silently fall out of every
    # grouping, so it must be rejected at construction.
    with pytest.raises(ValidationError):
        BenchmarkQuestion(
            question_id="broken_2",
            book_key="holmes",
            tier="impossible",  # type: ignore[arg-type]
            question_text="What tier is this?",
            gold_answer="n/a",
        )
