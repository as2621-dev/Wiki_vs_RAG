"""Token-target cumulative corpus loader with provenance (issue #4, PRD stories #4/#6/#7).

Turns the fixed-order corpus (Gutenberg novels then movie scripts) into the "first-N-tokens" for
any sweep target, so the sweep axis is **tokens, not book count** (PRD decision #2). This module is
the shared corpus contract the RAG / plain-LLM / wiki / sweep-runner slices consume.

It COMPOSES with ``corpus.py`` rather than forking it: it reuses the ``CorpusBook`` value object and
the fixed source order (``GUTENBERG_BOOKS`` then ``MOVIE_SCRIPTS``), and exposes the loaded anchor
keys so ``questions.questions_for_token_target`` can filter answerable questions off the SAME
token->sources decision this loader owns (never a parallel ordering).

Boundary policy (documented, deterministic): sources are loaded **whole, in fixed order**, until the
cumulative token count reaches the target — the source that crosses the target is included in full,
never truncated to the token boundary. Rationale: a golden question anchored to a work must see that
work whole to be answerable at the smallest sweep size (PRD decision #3), and whole-source offsets
stay clean for provenance/traceability. Consequence: the loaded corpus is "at least ~N tokens"
(capped at the available maximum), which the coarse sweep targets (100k, 500k, 1M, ...) tolerate.

Token axis: ``tiktoken``'s ``cl100k_base`` BPE — a deterministic, offline proxy for the LLM token
count (exact Anthropic counts require a per-call API round-trip, unusable in a deterministic,
network-free assembly path). The counter is injectable so tests stay hermetic and any future axis
swap is a one-line change; ``total_token_count`` is always the injected counter applied to the
loaded sources, so the reported axis never drifts from how it was measured.
"""

from collections.abc import Callable
from dataclasses import dataclass

import structlog

from corpus import CorpusBook, fetch_corpus, fetch_movie_corpus, loaded_movie_books

logger = structlog.get_logger()

# A token counter maps text -> token count. Injectable (default below) so tests need no tiktoken and
# a future axis change touches one place. See module docstring for why cl100k_base is the default.
TokenCounter = Callable[[str], int]

# Reason: cl100k_base is stable, cached-offline after first load, and version-pinned by name, so the
# same text always yields the same count across runs/machines (determinism, Rule 9).
_TIKTOKEN_ENCODING_NAME = "cl100k_base"
_token_encoder = None  # lazily constructed + cached; keeps import-time free of tiktoken.

# Fixed separator between concatenated sources in the assembled corpus text. Char offsets on each
# LoadedSource account for it, so ``load.text[offset:offset+len(text)]`` round-trips exactly.
_CORPUS_SEPARATOR = "\n\n"


def count_tokens_tiktoken(text: str) -> int:
    """Count tokens with tiktoken's ``cl100k_base`` BPE — the default, deterministic token axis.

    ``disallowed_special=()`` disables special-token interception so any literal string in a novel
    or script (e.g. ``<|endoftext|>``) is encoded as plain text and never raises. The encoder is
    built once and cached, so repeated counts over the corpus don't re-load the vocabulary.

    Args:
        text: The text to count.

    Returns:
        The token count under ``cl100k_base``; deterministic for a given string.

    Example:
        >>> count_tokens_tiktoken("hello world") > 0
        True
    """
    global _token_encoder
    if _token_encoder is None:
        try:
            import tiktoken
        except ImportError as exc:  # Reason: honest hard failure, not a silent wrong count.
            raise RuntimeError(
                "tiktoken not installed — `pip install tiktoken` (declared in requirements.txt)"
            ) from exc
        _token_encoder = tiktoken.get_encoding(_TIKTOKEN_ENCODING_NAME)
    return len(_token_encoder.encode(text, disallowed_special=()))


@dataclass(frozen=True)
class LoadedSource:
    """One whole source (novel or script) loaded into the token-target corpus, with provenance."""

    book_key: str
    title: str
    text: str
    token_count: int  # tokens this source contributes, under the run's token counter
    char_offset: int  # start index of this source's text within the assembled corpus (``CorpusLoad.text``)


@dataclass(frozen=True)
class CorpusLoad:
    """The corpus assembled for a token target: ordered provenance-bearing sources + a shortfall flag.

    ``shortfall`` is True iff the available corpus could not reach the requested target (capped at
    max available) — a loud signal the caller must not ignore (Rule 12), never a silent short read.
    """

    token_target: int
    total_token_count: int
    shortfall: bool
    sources: tuple[LoadedSource, ...]

    @property
    def text(self) -> str:
        """The assembled corpus text: loaded source texts joined by ``_CORPUS_SEPARATOR`` in order."""
        return _CORPUS_SEPARATOR.join(source.text for source in self.sources)

    @property
    def loaded_source_keys(self) -> list[str]:
        """The loaded anchor keys in fixed order — the seam ``questions_for_token_target`` composes with.

        Example:
            >>> CorpusLoad(100, 0, True, ()).loaded_source_keys
            []
        """
        return [source.book_key for source in self.sources]


def assemble_ordered_corpus(*, force_refresh: bool = False) -> list[CorpusBook]:
    """Source the full corpus in fixed order: Gutenberg novels then movie scripts.

    Thin production wiring over ``corpus.fetch_corpus`` and ``corpus.fetch_movie_corpus`` (each of
    which handles its own caching / fallback chain). Skipped movie scripts contribute nothing here
    but stay inspectable on the fetch results. Not exercised in the loader's unit tests, which
    inject an explicit ``books`` list to stay network-free.

    Args:
        force_refresh: Re-download the novels even if cached (passed through to ``fetch_corpus``).

    Returns:
        Novels then successfully-sourced scripts, both in their fixed declaration order.
    """
    novels = fetch_corpus(force_refresh=force_refresh)
    scripts = loaded_movie_books(fetch_movie_corpus())
    return [*novels, *scripts]


def load_for_token_target(
    token_target: int,
    *,
    books: list[CorpusBook] | None = None,
    count_tokens: TokenCounter = count_tokens_tiktoken,
) -> CorpusLoad:
    """Assemble the first-N-tokens of the fixed-order corpus, carrying provenance on every source.

    Accumulates whole sources in fixed order (see the module docstring's boundary policy) until the
    cumulative token count reaches ``token_target``; the crossing source is included whole and the
    loop stops. If the whole corpus can't reach the target it is capped at max available and
    ``shortfall`` is set + logged (never a silent short read — Rule 12). A zero-length source
    contributes nothing and is skipped without shifting the order or offsets of real sources.

    Determinism: for the same ``token_target``, ``books``, and ``count_tokens`` the result is
    byte-identical every run — no set/dict iteration, no wall-clock, no network in this path.

    Args:
        token_target: Desired cumulative token count (must be positive).
        books: The fixed-ordered corpus to load from; defaults to ``assemble_ordered_corpus()``
            (which fetches). Tests inject an explicit list to stay hermetic.
        count_tokens: Token counter defining the axis; defaults to ``count_tokens_tiktoken``.

    Returns:
        A ``CorpusLoad`` with the loaded sources (each with key/title/offset), the total token
        count, and the shortfall flag.

    Raises:
        ValueError: If ``token_target`` is not positive.

    Example:
        >>> from corpus import CorpusBook
        >>> books = [CorpusBook("a", "A", "one two three"), CorpusBook("b", "B", "four five six")]
        >>> load = load_for_token_target(2, books=books, count_tokens=lambda t: len(t.split()))
        >>> load.loaded_source_keys
        ['a']
    """
    if token_target <= 0:
        raise ValueError(f"token_target must be positive, got {token_target}")

    ordered = books if books is not None else assemble_ordered_corpus()
    loaded: list[LoadedSource] = []
    cumulative_tokens = 0
    char_offset = 0
    for book in ordered:
        if not book.text:
            # Reason: a zero-length source has no content and no answerable question; skip it so it
            # neither counts as loaded nor shifts the order/offsets of the real sources after it.
            logger.info("corpus_load_empty_source_skipped", book_key=book.book_key)
            continue
        source_tokens = count_tokens(book.text)
        loaded.append(
            LoadedSource(
                book_key=book.book_key,
                title=book.title,
                text=book.text,
                token_count=source_tokens,
                char_offset=char_offset,
            )
        )
        cumulative_tokens += source_tokens
        char_offset += len(book.text) + len(_CORPUS_SEPARATOR)
        if cumulative_tokens >= token_target:
            break

    shortfall = cumulative_tokens < token_target
    if shortfall:
        logger.warning(
            "corpus_load_shortfall",
            token_target=token_target,
            available_tokens=cumulative_tokens,
            loaded_sources=len(loaded),
            fix_suggestion="Add more sources or lower the token target; corpus capped at max available",
        )
    logger.info(
        "corpus_load_completed",
        token_target=token_target,
        loaded_tokens=cumulative_tokens,
        loaded_sources=len(loaded),
        shortfall=shortfall,
    )
    return CorpusLoad(
        token_target=token_target,
        total_token_count=cumulative_tokens,
        shortfall=shortfall,
        sources=tuple(loaded),
    )
