---
title: Layered fetch-fallback source (HTTP → Playwright → local dump) with hermetic boundary mocking
tags: [scraping, fallback, playwright, kaggle, beautifulsoup, dedup, pytest, mocking, corpus]
problem_type: convention
symptoms: A corpus source must try a cheap fetch, fall back to a heavier one, then to a local dump, and its tests must never hit a live site or need Playwright browser binaries.
date: 2026-07-05
---

Pattern used by `corpus.fetch_movie_script` (subslikescript → Playwright → Kaggle dump).
Reusable for any source with a cheap-path / heavy-path / offline-path chain.

## Inject each boundary as a callable, default to a module function
- `fetch_movie_script(..., *, http_fetch=_http_fetch_detail_html, playwright_fetch=..., kaggle_load=...)`.
  Tests pass fakes; production uses the defaults. This mocks **at the boundary**, not deep in
  logic, and needs no `monkeypatch`. Playwright is **lazily imported inside** its default fetcher
  (`from playwright.sync_api import ...` in the function body), so the module — and its tests —
  never require the package or `playwright install`. Declare it in `requirements.txt` anyway
  (honest contract) with a comment that it's optional/lazy.
- Drive the chain with a table `(source_enum, produces_html, lambda: layer())` and loop: each
  layer fires only if the prior returned empty/placeholder or **raised** (catch broad `Exception`,
  log a `warning` with `fix_suggestion`, `continue` — this is graceful degradation, not swallowing).

## Return a typed skip, never raise, on total failure
- On all-layers-miss: `logger.error(..., fix_suggestion=...)` and return a result object with
  `book=None` + non-empty `skipped_reason`. The cumulative loader flags the shortfall instead of
  crashing or silently shrinking the corpus (Rule 12). One `capture_logs()` test asserts the error
  event carries `fix_suggestion`; another asserts the fallback **call order** via a shared list.

## Reuse the existing value object — don't fork provenance
- A loaded script wraps the **same** `CorpusBook(book_key, title, text)` novels use, so it plugs
  into the fixed source-order/provenance contract `questions_for_token_target` composes with. The
  skip/source metadata lives on a thin envelope (`MovieScriptResult`), not a parallel corpus model.

## Two non-obvious gotchas
- **"Loading…" detection must match the WHOLE cleaned text**, not a substring — else a real line
  containing "loading" is a false miss, and boilerplate-stripping could be tricked into surfacing
  the placeholder. Check `cleaned.strip().lower() in {"loading", "loading…", "loading..."}`.
- **Near-duplicate fingerprint: fold separators to EMPTY, not to a space.** `re.sub(r"[^a-z0-9]+",
  "", text.lower())` makes `"Here's"` and `"heres"` collide (a reformatted re-upload). Substituting
  a space leaves `"here s"` ≠ `"heres"` and the dedup misses.

## DOM contract (Rule 9): scope the node, and test a decoy
- Extract `div.full-script` **inside** `article.main-article` (`article = soup.select_one(...)`,
  then `article.select_one("div.full-script")`) — a `full-script` node elsewhere on the page is a
  decoy. The happy-path fixture must include a decoy full-script **outside** the article and assert
  it is NOT scraped, or the test can't fail when the selector loses its scope.
