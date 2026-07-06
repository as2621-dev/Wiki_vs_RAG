"""Public-domain novels + movie-script corpus fetching + chunking for the RAG path.

Downloads five Project Gutenberg novels (dense character/place/faction networks,
no licensing issue), strips the PG boilerplate, and exposes character-window
chunks for embedding. The book ORDER is fixed — the scaling sweep loads the
first N books at corpus size N, so the same books map to the same sizes on every
run.

Movie scripts are sourced from subslikescript detail pages (server-rendered
``div.full-script`` inside ``article.main-article``) with a Playwright fallback for
JS-only/blocked pages and a Kaggle-dump fallback for a zero-scrape path. A fetched
script reuses the same ``CorpusBook`` value object novels use, so both plug into the
same fixed source-order / provenance contract the cumulative loader composes with.

Legal note: movie scripts are copyrighted. They are used here ONLY as private
benchmark input for local scoring — never redistributed, never republished, never
shipped in results. Only public-domain Gutenberg text and hand-written question/gold
strings are published. Treat any scraped or Kaggle-sourced script as private input.
"""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import httpx
import structlog
from bs4 import BeautifulSoup

from settings import get_settings

logger = structlog.get_logger()

DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Reason: fixed order => reproducible size→book mapping across runs.
GUTENBERG_BOOKS: list[tuple[str, str, str]] = [
    # (book_key, human title, Gutenberg plain-text URL)
    ("holmes", "The Adventures of Sherlock Holmes", "https://www.gutenberg.org/files/1661/1661-0.txt"),
    ("dracula", "Dracula", "https://www.gutenberg.org/files/345/345-0.txt"),
    ("monte_cristo", "The Count of Monte Cristo", "https://www.gutenberg.org/files/1184/1184-0.txt"),
    ("les_mis", "Les Misérables", "https://www.gutenberg.org/files/135/135-0.txt"),
    ("war_and_peace", "War and Peace", "https://www.gutenberg.org/files/2600/2600-0.txt"),
]

_PG_START_MARKER = "*** START OF"
_PG_END_MARKER = "*** END OF"


@dataclass(frozen=True)
class CorpusBook:
    """A single fetched, cleaned novel."""

    book_key: str
    title: str
    text: str


@dataclass(frozen=True)
class BookChunk:
    """One embeddable window of a book, with provenance metadata."""

    chunk_id: str
    book_key: str
    title: str
    text: str


def _strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Remove the Project Gutenberg license header/footer, keeping the novel."""
    start_index = raw_text.find(_PG_START_MARKER)
    if start_index != -1:
        # Skip to the end of the START marker line.
        newline = raw_text.find("\n", start_index)
        raw_text = raw_text[newline + 1 :] if newline != -1 else raw_text[start_index:]
    end_index = raw_text.find(_PG_END_MARKER)
    if end_index != -1:
        raw_text = raw_text[:end_index]
    return raw_text.strip()


def fetch_corpus(force_refresh: bool = False) -> list[CorpusBook]:
    """Download and cache all corpus books to ``documents/``.

    Args:
        force_refresh: Re-download even if a cached copy exists on disk.

    Returns:
        The books in fixed corpus order.

    Example:
        >>> books = fetch_corpus()
        >>> books[0].book_key
        'holmes'
    """
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    books: list[CorpusBook] = []
    with httpx.Client(timeout=120.0, follow_redirects=True) as http_client:
        for book_key, title, url in GUTENBERG_BOOKS:
            cache_path = DOCUMENTS_DIR / f"{book_key}.txt"
            if cache_path.exists() and not force_refresh:
                logger.info("corpus_cache_hit", book_key=book_key)
                text = cache_path.read_text(encoding="utf-8")
            else:
                logger.info("corpus_download_started", book_key=book_key, url=url)
                try:
                    response = http_client.get(url)
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    logger.error(
                        "corpus_download_failed",
                        book_key=book_key,
                        error_message=str(exc),
                        fix_suggestion="Check the Gutenberg URL / network; try force_refresh later",
                    )
                    raise
                text = _strip_gutenberg_boilerplate(response.text)
                cache_path.write_text(text, encoding="utf-8")
                logger.info("corpus_download_completed", book_key=book_key, chars=len(text))
            books.append(CorpusBook(book_key=book_key, title=title, text=text))
    return books


def chunk_book(book: CorpusBook) -> list[BookChunk]:
    """Split a book into overlapping character-window chunks for embedding."""
    settings = get_settings()
    size = settings.chunk_size_chars
    overlap = settings.chunk_overlap_chars
    step = max(1, size - overlap)
    chunks: list[BookChunk] = []
    position = 0
    index = 0
    while position < len(book.text):
        window = book.text[position : position + size]
        if window.strip():
            chunks.append(
                BookChunk(
                    chunk_id=f"{book.book_key}_{index}",
                    book_key=book.book_key,
                    title=book.title,
                    text=window,
                )
            )
            index += 1
        position += step
    logger.info("book_chunked", book_key=book.book_key, chunk_count=len(chunks))
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Movie-script source (subslikescript → Playwright → Kaggle dump)
# ══════════════════════════════════════════════════════════════════════════════

# Reason: fixed order => reproducible size→work mapping, exactly like GUTENBERG_BOOKS.
# book_key values are the anchor keys the question bank targets (see questions.QUESTION_BANK);
# the cumulative loader (issue #4) appends these after the novels in this order.
# Detail-page ids are best-effort: a wrong URL/id simply misses and degrades down the fallback
# chain (that graceful degradation is the whole point), so an unverified id never crashes a run.
MOVIE_SCRIPTS: list[tuple[str, str, str]] = [
    # (book_key, human title, subslikescript detail-page URL)
    ("godfather", "The Godfather", "https://subslikescript.com/movie/The_Godfather-68646"),
    ("casablanca", "Casablanca", "https://subslikescript.com/movie/Casablanca-34583"),
    ("pulp_fiction", "Pulp Fiction", "https://subslikescript.com/movie/Pulp_Fiction-110912"),
    ("wizard_of_oz", "The Wizard of Oz", "https://subslikescript.com/movie/The_Wizard_of_Oz-32138"),
    ("star_wars", "Star Wars: A New Hope", "https://subslikescript.com/movie/Star_Wars-76759"),
    ("shawshank", "The Shawshank Redemption", "https://subslikescript.com/movie/The_Shawshank_Redemption-111161"),
]

# DOM contract: transcript is div.full-script INSIDE article.main-article (a full-script node
# elsewhere on the page is a wrong node — Rule 9). Title is the h1.
_MOVIE_ARTICLE_SELECTOR = "article.main-article"
_MOVIE_SCRIPT_SELECTOR = "div.full-script"

# Scraper boilerplate injected around the transcript; dropped line-wise during cleaning.
_SCRIPT_BOILERPLATE_MARKERS: tuple[str, ...] = (
    "Watch the full movie",
    "You can read the",
    "ADVERTISEMENT",
)

# JS-only pages render only this placeholder into div.full-script. Matched on the WHOLE cleaned
# text (not a substring) so a real line containing the word "loading" is never a false miss.
_LOADING_PLACEHOLDERS: frozenset[str] = frozenset({"loading", "loading…", "loading..."})

# Reason: aggressive normalization (case/whitespace/punctuation folded away) so a re-upload that
# differs only in formatting fingerprints identically — a cheap, deterministic near-duplicate test.
_NON_ALNUM_RUN = re.compile(r"[^a-z0-9]+")


class MovieSource(str, Enum):
    """Which layer of the fallback chain actually supplied a script (source provenance)."""

    SUBSLIKESCRIPT_HTTP = "subslikescript_http"
    SUBSLIKESCRIPT_PLAYWRIGHT = "subslikescript_playwright"
    KAGGLE_DUMP = "kaggle_dump"


@dataclass(frozen=True)
class MovieScriptResult:
    """Outcome of sourcing one movie script through the fallback chain.

    On success ``book`` is a ``CorpusBook`` (the same value object novels use, so it flows through
    the existing source-order/provenance contract) and ``fetched_via`` names the layer that
    supplied it. On a total miss ``book``/``fetched_via`` are ``None`` and ``skipped_reason`` is a
    non-empty flag the cumulative loader surfaces — a typed shortfall signal, never an exception.
    """

    book_key: str
    title: str
    book: CorpusBook | None
    fetched_via: MovieSource | None
    skipped_reason: str = ""


ScriptFetcher = Callable[[str], str]
KaggleLoader = Callable[[str, str], str]


def _strip_script_boilerplate(raw_text: str) -> str:
    """Drop scraper boilerplate lines, keeping only transcript lines."""
    kept: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(marker in stripped for marker in _SCRIPT_BOILERPLATE_MARKERS):
            continue
        kept.append(stripped)
    return "\n".join(kept).strip()


def parse_full_script(html: str) -> str:
    """Extract the cleaned transcript from a subslikescript detail page.

    Scrapes ``div.full-script`` *inside* ``article.main-article`` (a full-script node elsewhere
    on the page is a decoy — Rule 9) and strips scraper boilerplate. Returns ``""`` when the
    node is absent so the caller treats it as a plain-fetch miss and falls back.

    Args:
        html: The detail-page HTML (from a plain fetch or a Playwright render).

    Returns:
        The boilerplate-stripped transcript text, or ``""`` if the node is missing.

    Example:
        >>> parse_full_script('<article class="main-article"><div class="full-script">'
        ...                   'ADVERTISEMENT\\nHello.</div></article>')
        'Hello.'
    """
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one(_MOVIE_ARTICLE_SELECTOR)
    if article is None:
        return ""
    node = article.select_one(_MOVIE_SCRIPT_SELECTOR)
    if node is None:
        return ""
    return _strip_script_boilerplate(node.get_text("\n"))


def _is_usable_script(text: str) -> bool:
    """True iff ``text`` is a real transcript, not empty and not a 'Loading…' placeholder."""
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.lower() not in _LOADING_PLACEHOLDERS


def _script_fingerprint(text: str) -> str:
    """SHA-256 over aggressively-normalized text — equal for whitespace/case/punctuation-only diffs."""
    normalized = _NON_ALNUM_RUN.sub("", text.lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _http_fetch_detail_html(url: str) -> str:
    """Plain server-rendered fetch of a detail page (the first, cheapest layer)."""
    with httpx.Client(timeout=60.0, follow_redirects=True) as http_client:
        response = http_client.get(url)
        response.raise_for_status()
        return response.text


def _playwright_fetch_detail_html(url: str) -> str:
    """Render a JS-only detail page with Playwright (second layer; import-guarded).

    Playwright is an optional dependency and is imported lazily so the module (and its tests,
    which inject a fake) never require the package or its browser binaries.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright not installed — `pip install playwright && playwright install chromium`"
        ) from exc
    with sync_playwright() as playwright_runtime:
        browser = playwright_runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            return page.content()
        finally:
            browser.close()


def _kaggle_load_script(book_key: str, title: str) -> str:
    """Read a pre-processed script from the local Kaggle dump (third, zero-scrape layer).

    Expects ``MOVIE_SCRIPTS_KAGGLE_DUMP_PATH`` to point at a directory of ``<book_key>.txt`` files
    derived from the 'Movie Transcripts 59K' Kaggle dump. No Kaggle API credentials are used or
    stored — the dump is downloaded out-of-band and read locally. Returns ``""`` on any miss.
    """
    dump_path = get_settings().movie_scripts_kaggle_dump_path
    if not dump_path:
        return ""
    script_path = Path(dump_path) / f"{book_key}.txt"
    if not script_path.exists():
        logger.warning(
            "movie_kaggle_dump_miss",
            book_key=book_key,
            fix_suggestion="Add <book_key>.txt to MOVIE_SCRIPTS_KAGGLE_DUMP_PATH from the Kaggle dump",
        )
        return ""
    return script_path.read_text(encoding="utf-8").strip()


def fetch_movie_script(
    book_key: str,
    title: str,
    detail_url: str,
    *,
    http_fetch: ScriptFetcher = _http_fetch_detail_html,
    playwright_fetch: ScriptFetcher = _playwright_fetch_detail_html,
    kaggle_load: KaggleLoader = _kaggle_load_script,
) -> MovieScriptResult:
    """Source one movie script, trying plain HTTP → Playwright → Kaggle dump in order.

    Each layer fires only when the prior returned empty/placeholder or raised. The three
    boundaries are injectable so tests mock at the boundary (no live site, no browser binaries).
    A total failure is logged with a ``fix_suggestion`` and returned as a typed skip — never a
    raised exception, so the cumulative loader flags the shortfall instead of crashing (Rule 12).

    Args:
        book_key: Anchor key (matches the question bank, e.g. ``"godfather"``).
        title: Human title, carried as provenance on the returned ``CorpusBook``.
        detail_url: subslikescript detail-page URL for the plain/Playwright layers.
        http_fetch: Plain-fetch boundary returning raw HTML.
        playwright_fetch: Playwright boundary returning rendered HTML.
        kaggle_load: Kaggle-dump boundary returning plain script text.

    Returns:
        A ``MovieScriptResult`` — a loaded ``CorpusBook`` with its source, or a typed skip.

    Example:
        >>> html = '<article class="main-article"><div class="full-script">Hi.</div></article>'
        >>> r = fetch_movie_script("m", "M", "u", http_fetch=lambda _: html)
        >>> r.fetched_via.value
        'subslikescript_http'
    """
    # Reason: (source, produces-html?, runner) — html layers get parsed; Kaggle yields plain text.
    attempts: tuple[tuple[MovieSource, bool, Callable[[], str]], ...] = (
        (MovieSource.SUBSLIKESCRIPT_HTTP, True, lambda: http_fetch(detail_url)),
        (MovieSource.SUBSLIKESCRIPT_PLAYWRIGHT, True, lambda: playwright_fetch(detail_url)),
        (MovieSource.KAGGLE_DUMP, False, lambda: kaggle_load(book_key, title)),
    )
    for source, produces_html, run_layer in attempts:
        try:
            raw = run_layer()
        except Exception as exc:  # Reason: any layer may fail; log + fall through, never swallow.
            logger.warning(
                "movie_script_source_error",
                book_key=book_key,
                source=source.value,
                error_message=str(exc),
                fix_suggestion="Check the source URL/network, Playwright install, or Kaggle dump path",
            )
            continue
        script_text = parse_full_script(raw) if produces_html else raw.strip()
        if _is_usable_script(script_text):
            logger.info("movie_script_sourced", book_key=book_key, source=source.value, chars=len(script_text))
            book = CorpusBook(book_key=book_key, title=title, text=script_text)
            return MovieScriptResult(book_key=book_key, title=title, book=book, fetched_via=source)
        logger.info("movie_script_source_miss", book_key=book_key, source=source.value)
    logger.error(
        "movie_script_all_sources_failed",
        book_key=book_key,
        fix_suggestion="Verify the subslikescript detail URL/id, install Playwright, or set the Kaggle dump path",
    )
    return MovieScriptResult(book_key=book_key, title=title, book=None, fetched_via=None, skipped_reason="all_sources_failed")


def fetch_movie_corpus(
    scripts: list[tuple[str, str, str]] = MOVIE_SCRIPTS,
    *,
    http_fetch: ScriptFetcher = _http_fetch_detail_html,
    playwright_fetch: ScriptFetcher = _playwright_fetch_detail_html,
    kaggle_load: KaggleLoader = _kaggle_load_script,
) -> list[MovieScriptResult]:
    """Source every movie script in fixed order, dropping near-duplicates.

    Runs ``fetch_movie_script`` per entry (through the fallback chain) and de-duplicates by
    normalized-text fingerprint so a re-upload differing only in formatting is not loaded twice.
    Every entry yields a result — a loaded book, a near-duplicate skip, or a fetch-failure skip —
    so the cumulative loader (issue #4) sees the full shortfall picture, never a silently shrunk list.

    Args:
        scripts: Fixed ``(book_key, title, detail_url)`` order; defaults to ``MOVIE_SCRIPTS``.
        http_fetch: Plain-fetch boundary (injected in tests).
        playwright_fetch: Playwright boundary (injected in tests).
        kaggle_load: Kaggle-dump boundary (injected in tests).

    Returns:
        One ``MovieScriptResult`` per input entry, in source order.
    """
    results: list[MovieScriptResult] = []
    seen_fingerprints: set[str] = set()
    for book_key, title, detail_url in scripts:
        result = fetch_movie_script(
            book_key,
            title,
            detail_url,
            http_fetch=http_fetch,
            playwright_fetch=playwright_fetch,
            kaggle_load=kaggle_load,
        )
        if result.book is None:
            results.append(result)  # keep the shortfall visible
            continue
        fingerprint = _script_fingerprint(result.book.text)
        if fingerprint in seen_fingerprints:
            logger.info("movie_script_near_duplicate_skipped", book_key=book_key)
            results.append(
                MovieScriptResult(book_key=book_key, title=title, book=None, fetched_via=None, skipped_reason="near_duplicate")
            )
            continue
        seen_fingerprints.add(fingerprint)
        results.append(result)
    loaded_count = sum(1 for result in results if result.book is not None)
    logger.info("movie_corpus_fetched", requested=len(scripts), loaded=loaded_count)
    return results


def loaded_movie_books(results: list[MovieScriptResult]) -> list[CorpusBook]:
    """Return the successfully-loaded ``CorpusBook`` scripts in source order.

    The seam the cumulative loader (issue #4) consumes: it appends these after the Gutenberg
    novels to build the fixed-order corpus. Skipped/shortfall results contribute nothing here but
    remain inspectable on the original results list.

    Example:
        >>> loaded_movie_books([])
        []
    """
    return [result.book for result in results if result.book is not None]
