"""Public-domain corpus fetching + chunking for the RAG path.

Downloads five Project Gutenberg novels (dense character/place/faction networks,
no licensing issue), strips the PG boilerplate, and exposes character-window
chunks for embedding. The book ORDER is fixed — the scaling sweep loads the
first N books at corpus size N, so the same books map to the same sizes on every
run.
"""

from dataclasses import dataclass
from pathlib import Path

import httpx
import structlog

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
