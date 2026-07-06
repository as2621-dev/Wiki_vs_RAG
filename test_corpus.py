"""Tests for the movie-script corpus source (subslikescript -> Playwright -> Kaggle).

Every external boundary is mocked: the HTTP fetch, the Playwright fetch, and the Kaggle-dump
reader are injected as fakes, so no test hits a live site or needs Playwright browser binaries.
Tests assert CODE behaviour: the right DOM node is scraped (not a decoy), the fallback chain
fires in order, near-duplicates are dropped, and a total failure is a typed skip (not a crash).
"""

import pytest
import structlog

import corpus
from corpus import (
    CorpusBook,
    MovieScriptResult,
    MovieSource,
    fetch_movie_corpus,
    fetch_movie_script,
    loaded_movie_books,
    parse_full_script,
)
from questions import questions_for_token_target


def _detail_page_html(script_text: str, *, decoy_text: str = "WRONG NODE — sidebar advert") -> str:
    """Wrap transcript text in a subslikescript-shaped detail page.

    Includes a DECOY ``div.full-script`` OUTSIDE ``article.main-article`` so a correct parser
    must scrape the node *inside* the article, not the first match on the page (Rule 9).
    """
    return f"""
    <html><body>
      <aside><div class="full-script">{decoy_text}</div></aside>
      <article class="main-article">
        <h1>Some Movie</h1>
        <div class="full-script">Watch the full movie for free\n{script_text}\nADVERTISEMENT</div>
      </article>
    </body></html>
    """


def _loading_page_html() -> str:
    """A JS-only page whose server HTML only holds a 'Loading…' placeholder."""
    return """
    <html><body>
      <article class="main-article">
        <div class="full-script">Loading…</div>
      </article>
    </body></html>
    """


# ─── parse_full_script: DOM contract ──────────────────────────────────────────


def test_parse_full_script_scrapes_node_inside_article_not_decoy() -> None:
    # Reason (Rule 9): the transcript lives in div.full-script INSIDE article.main-article;
    # a parser that grabs the first div.full-script would return the sidebar decoy.
    html = _detail_page_html("RICK: Here's looking at you, kid.")
    cleaned = parse_full_script(html)
    assert "Here's looking at you, kid." in cleaned
    assert "WRONG NODE" not in cleaned  # decoy outside the article must not be scraped


def test_parse_full_script_strips_scraper_boilerplate() -> None:
    html = _detail_page_html("Real dialogue line.")
    cleaned = parse_full_script(html)
    assert "Watch the full movie" not in cleaned
    assert "ADVERTISEMENT" not in cleaned
    assert cleaned == "Real dialogue line."


def test_parse_full_script_missing_node_returns_empty() -> None:
    assert parse_full_script("<html><body><p>no article here</p></body></html>") == ""


def test_parse_full_script_loading_placeholder_is_not_a_real_script() -> None:
    # Boilerplate stripping must NOT be trickable into returning a usable 'Loading…'.
    assert corpus._is_usable_script(parse_full_script(_loading_page_html())) is False


# ─── fetch_movie_script: fallback chain ───────────────────────────────────────


def test_happy_path_uses_plain_http_with_title_and_source_provenance() -> None:
    def http(url: str) -> str:
        return _detail_page_html("MICHAEL: My father made him an offer he couldn't refuse.")

    def never(*args: object, **kwargs: object) -> str:
        raise AssertionError("fallback should not fire on a good plain fetch")

    result = fetch_movie_script(
        "godfather",
        "The Godfather",
        "https://example/movie/The_Godfather",
        http_fetch=http,
        playwright_fetch=never,
        kaggle_load=never,
    )
    assert result.book is not None
    assert result.book_key == "godfather"
    assert result.book.title == "The Godfather"  # title provenance
    assert "offer he couldn't refuse" in result.book.text
    assert result.fetched_via is MovieSource.SUBSLIKESCRIPT_HTTP  # source provenance
    assert result.skipped_reason == ""


def test_loading_page_falls_back_to_playwright() -> None:
    def http(url: str) -> str:
        return _loading_page_html()

    def playwright(url: str) -> str:
        return _detail_page_html("Rendered-by-JS dialogue.")

    def never(*args: object, **kwargs: object) -> str:
        raise AssertionError("Kaggle should not fire when Playwright succeeds")

    result = fetch_movie_script(
        "casablanca", "Casablanca", "url", http_fetch=http, playwright_fetch=playwright, kaggle_load=never
    )
    assert result.fetched_via is MovieSource.SUBSLIKESCRIPT_PLAYWRIGHT
    assert "Rendered-by-JS dialogue." in result.book.text


def test_fallback_chain_order_plain_then_playwright_then_kaggle() -> None:
    # Proves the CHAIN ordering: each layer only fires when the prior returned empty/failed.
    calls: list[str] = []

    def http(url: str) -> str:
        calls.append("http")
        return _loading_page_html()  # miss

    def playwright(url: str) -> str:
        calls.append("playwright")
        raise RuntimeError("browser blocked")  # fail

    def kaggle(book_key: str, title: str) -> str:
        calls.append("kaggle")
        return "Full transcript straight from the Kaggle dump."

    result = fetch_movie_script(
        "pulp_fiction", "Pulp Fiction", "url", http_fetch=http, playwright_fetch=playwright, kaggle_load=kaggle
    )
    assert calls == ["http", "playwright", "kaggle"]
    assert result.fetched_via is MovieSource.KAGGLE_DUMP
    assert "Kaggle dump" in result.book.text


def test_total_failure_returns_typed_skip_not_exception() -> None:
    def miss(*args: object, **kwargs: object) -> str:
        return ""  # every layer returns nothing

    with structlog.testing.capture_logs() as logs:
        result = fetch_movie_script(
            "star_wars", "Star Wars", "url", http_fetch=miss, playwright_fetch=miss, kaggle_load=miss
        )
    # A well-typed skip the cumulative loader can act on — NOT a raised exception (Rule 12).
    assert isinstance(result, MovieScriptResult)
    assert result.book is None
    assert result.fetched_via is None
    assert result.skipped_reason  # shortfall flagged, non-empty
    error_events = [log for log in logs if log["log_level"] == "error"]
    assert error_events and all("fix_suggestion" in log for log in error_events)


def test_source_error_is_logged_and_does_not_crash() -> None:
    def boom(url: str) -> str:
        raise ConnectionError("network down")

    def kaggle(book_key: str, title: str) -> str:
        return "Recovered script text."

    with structlog.testing.capture_logs() as logs:
        result = fetch_movie_script(
            "shawshank", "The Shawshank Redemption", "url",
            http_fetch=boom, playwright_fetch=boom, kaggle_load=kaggle,
        )
    assert result.fetched_via is MovieSource.KAGGLE_DUMP  # recovered via last layer
    warnings = [log for log in logs if log["log_level"] == "warning"]
    assert warnings and all("fix_suggestion" in log for log in warnings)


# ─── fetch_movie_corpus: dedup + shortfall composition ────────────────────────


def test_near_duplicate_scripts_are_detected_and_not_loaded_twice() -> None:
    original = "RICK: Here's looking at you, kid."
    near_dup = "  rick   heres  looking AT you kid  "  # same words, different case/space/punct

    def http(url: str) -> str:
        return _detail_page_html(near_dup if "second" in url else original)

    scripts = [
        ("casablanca", "Casablanca", "first"),
        ("casablanca_dup", "Casablanca (reupload)", "second"),
    ]
    results = fetch_movie_corpus(
        scripts, http_fetch=http, playwright_fetch=_reject, kaggle_load=_reject
    )
    loaded = loaded_movie_books(results)
    assert len(loaded) == 1  # near-duplicate not loaded twice
    assert results[1].skipped_reason == "near_duplicate"
    assert results[1].book is None


def test_failed_source_is_flagged_not_silently_shrunk() -> None:
    def http(url: str) -> str:
        if "bad" in url:
            return ""  # total miss for this one
        return _detail_page_html("Good script.")

    scripts = [
        ("good", "Good Movie", "good"),
        ("bad", "Bad Movie", "bad"),
    ]
    results = fetch_movie_corpus(scripts, http_fetch=http, playwright_fetch=_miss, kaggle_load=_miss)
    assert len(results) == 2  # both entries preserved — corpus not silently shrunk
    assert loaded_movie_books(results) == [results[0].book]
    assert results[1].skipped_reason  # shortfall visible to the cumulative loader


# ─── integration: plugs into the corpus source-order / questions contract ─────


def test_loaded_movie_book_composes_with_question_bank_source_order() -> None:
    def http(url: str) -> str:
        return _detail_page_html("MICHAEL and SONNY are brothers.")

    results = fetch_movie_corpus(
        [("godfather", "The Godfather", "u")], http_fetch=http, playwright_fetch=_reject, kaggle_load=_reject
    )
    books = loaded_movie_books(results)
    assert isinstance(books[0], CorpusBook)  # same value object novels use
    # The movie book_key must flow through questions.py's source-order composition (issue #4 seam).
    keyed = questions_for_token_target([books[0].book_key])
    assert keyed and {q.book_key for q in keyed} == {"godfather"}


def _reject(*args: object, **kwargs: object) -> str:
    raise AssertionError("unexpected fallback")


def _miss(*args: object, **kwargs: object) -> str:
    return ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
