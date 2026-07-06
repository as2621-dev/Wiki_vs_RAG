"""Tests for the token-target cumulative corpus loader (issue #4).

The loader is a PURE assembly over already-sourced ``CorpusBook`` objects: tests inject a fixed
book list and a fixed token counter, so no test hits the network, disk, or tiktoken. Tests assert
CODE behaviour: fixed source order preserved, whole sources accumulated up to the token target,
provenance (source key / title / char offset) carried on every loaded source, a shortfall flagged
(never a silent under-delivery), an empty source skipped without breaking ordering, determinism
(same target + sources -> identical bytes), and composition with ``questions_for_token_target``.
"""

import pytest
import structlog

from corpus import CorpusBook
from corpus_loader import (
    CorpusLoad,
    LoadedSource,
    count_tokens_tiktoken,
    load_for_token_target,
)
from questions import questions_for_token_target


def _word_counter(text: str) -> int:
    """A deterministic stand-in token counter (one 'token' per whitespace word) for hermetic tests."""
    return len(text.split())


def _books(*specs: tuple[str, str, str]) -> list[CorpusBook]:
    """Build an ordered CorpusBook list from ``(book_key, title, text)`` specs."""
    return [CorpusBook(book_key=key, title=title, text=text) for key, title, text in specs]


# ─── happy path: fixed order, whole sources, up-to-N accumulation ─────────────


def test_accumulates_whole_sources_in_fixed_order_up_to_target() -> None:
    # Injected counter: each source is 40 "tokens". Target 50 -> source0 (40 < 50) keeps going,
    # source1 crosses (80 >= 50) and is included WHOLE, source2 is never loaded.
    books = _books(
        ("holmes", "Holmes", " ".join(["w"] * 40)),
        ("dracula", "Dracula", " ".join(["w"] * 40)),
        ("monte_cristo", "Monte Cristo", " ".join(["w"] * 40)),
    )
    load = load_for_token_target(50, books=books, count_tokens=_word_counter)
    assert load.loaded_source_keys == ["holmes", "dracula"]  # fixed order, crossing source included
    assert load.total_token_count == 80
    assert load.shortfall is False


def test_single_source_larger_than_target_still_loads_whole() -> None:
    # Whole-source policy: a lone source bigger than the target loads in full (so its golden
    # anchor is answerable at the smallest sweep size), never truncated to the boundary.
    books = _books(("holmes", "Holmes", " ".join(["w"] * 100)))
    load = load_for_token_target(50, books=books, count_tokens=_word_counter)
    assert load.loaded_source_keys == ["holmes"]
    assert load.total_token_count == 100
    assert load.shortfall is False


# ─── provenance on every loaded source ────────────────────────────────────────


def test_every_source_carries_key_title_and_char_offset() -> None:
    books = _books(
        ("holmes", "Holmes", "AAAA"),
        ("dracula", "Dracula", "BBBB"),
    )
    load = load_for_token_target(1000, books=books, count_tokens=_word_counter)
    assert [(s.book_key, s.title) for s in load.sources] == [("holmes", "Holmes"), ("dracula", "Dracula")]
    # char_offset must index the source's text inside the assembled corpus (traceability, story #7).
    for source in load.sources:
        assert load.text[source.char_offset : source.char_offset + len(source.text)] == source.text
    assert load.sources[0].char_offset == 0
    assert load.sources[1].char_offset == len("AAAA") + len("\n\n")  # separator accounted for


# ─── edge: target exceeds available corpus -> cap + shortfall flag ────────────


def test_target_exceeding_corpus_caps_at_max_and_flags_shortfall() -> None:
    books = _books(
        ("holmes", "Holmes", " ".join(["w"] * 30)),
        ("dracula", "Dracula", " ".join(["w"] * 30)),
    )
    with structlog.testing.capture_logs() as logs:
        load = load_for_token_target(1_000_000, books=books, count_tokens=_word_counter)
    assert load.loaded_source_keys == ["holmes", "dracula"]  # everything included
    assert load.total_token_count == 60  # capped at max available
    assert load.shortfall is True  # never silently under-delivers (Rule 12)
    shortfall_logs = [log for log in logs if log["event"] == "corpus_load_shortfall"]
    assert shortfall_logs and all("fix_suggestion" in log for log in shortfall_logs)


# ─── edge: empty / zero-length source contributes nothing, order intact ───────


def test_empty_source_contributes_nothing_and_does_not_break_order() -> None:
    books = _books(
        ("holmes", "Holmes", "AAAA"),
        ("empty_work", "Empty Work", ""),  # zero-length source in the middle
        ("dracula", "Dracula", "BBBB"),
    )
    load = load_for_token_target(1000, books=books, count_tokens=_word_counter)
    assert load.loaded_source_keys == ["holmes", "dracula"]  # empty skipped, ordering intact
    # Offsets stay correct: dracula still starts right after holmes + one separator, unaffected.
    assert load.sources[1].char_offset == len("AAAA") + len("\n\n")
    assert load.text == "AAAA\n\nBBBB"


# ─── determinism: same target + sources -> identical bytes every run ──────────


def test_same_target_yields_identical_content_across_runs() -> None:
    books = _books(
        ("holmes", "Holmes", " ".join(["w"] * 40)),
        ("dracula", "Dracula", " ".join(["w"] * 40)),
        ("monte_cristo", "Monte Cristo", " ".join(["w"] * 40)),
    )
    first = load_for_token_target(60, books=books, count_tokens=_word_counter)
    second = load_for_token_target(60, books=books, count_tokens=_word_counter)
    assert first == second  # frozen dataclasses compare by value
    assert first.text == second.text
    assert first.loaded_source_keys == second.loaded_source_keys


# ─── error/boundary: token counter is consistent + documented; bad target loud ─


def test_total_token_count_equals_sum_of_counted_sources() -> None:
    # The reported token axis must equal the counter applied to each loaded source — no hidden
    # re-counting on a different basis (consistency with the sweep axis used elsewhere).
    books = _books(
        ("holmes", "Holmes", " ".join(["w"] * 12)),
        ("dracula", "Dracula", " ".join(["w"] * 7)),
    )
    load = load_for_token_target(1000, books=books, count_tokens=_word_counter)
    assert load.total_token_count == sum(_word_counter(s.text) for s in load.sources) == 19


def test_non_positive_target_raises_loudly() -> None:
    books = _books(("holmes", "Holmes", "AAAA"))
    with pytest.raises(ValueError):
        load_for_token_target(0, books=books, count_tokens=_word_counter)


def test_default_tiktoken_counter_is_deterministic_and_positive() -> None:
    pytest.importorskip("tiktoken")
    text = "Sherlock Holmes is a consulting detective at 221B Baker Street."
    assert count_tokens_tiktoken(text) == count_tokens_tiktoken(text)
    assert count_tokens_tiktoken(text) > 0


# ─── system-wide: loader's loaded keys compose with the question bank ─────────


def test_loaded_source_keys_compose_with_questions_for_token_target() -> None:
    # The loader owns the token->sources decision; questions.py filters on the keys it exposes.
    # For a target that loads only holmes, only holmes questions are answerable (real composition).
    books = _books(
        ("holmes", "Holmes", " ".join(["w"] * 40)),
        ("dracula", "Dracula", " ".join(["w"] * 40)),
    )
    load = load_for_token_target(10, books=books, count_tokens=_word_counter)
    assert load.loaded_source_keys == ["holmes"]
    answerable = questions_for_token_target(load.loaded_source_keys)
    assert answerable and {q.book_key for q in answerable} == {"holmes"}


def test_result_types_are_frozen_value_objects() -> None:
    load = load_for_token_target(5, books=_books(("holmes", "Holmes", "AAAA")), count_tokens=_word_counter)
    assert isinstance(load, CorpusLoad)
    assert isinstance(load.sources[0], LoadedSource)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
