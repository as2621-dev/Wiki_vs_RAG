# Conventions

**Why this doc exists:** so every slice built by `/grab-issue` matches the scaffolding already in
`corpus.py` / `models.py` / `questions.py` / `settings.py` instead of re-inventing style.
**Update when:** a new cross-cutting pattern is introduced (a new module type, a new log event family,
a schema change to `ResultRow`).

## Language & structure

- **Python 3.12+**, type hints on every signature and attribute. Ruff, line length 120, double quotes.
- One **deep module per file** at repo root (flat until complexity demands a package): `corpus.py`,
  `questions.py`, `models.py`, `settings.py`, and the new `rag.py`, `plain_llm.py`, `wiki.py`,
  `judge.py`, `runner.py`, `analysis.py`, `cost.py`. Keep each **under 500 lines** (agent-code limit);
  wiring thin, logic inside.
- **Pydantic v2 models** (or frozen dataclasses for pure value objects) for every structure crossing a
  module boundary — never raw dicts at a seam. Prior art: `models.py`, `corpus.CorpusBook/BookChunk`.
- Pure, table-driven functions where possible (prior art: `questions_for_corpus_size`).

## Naming

- Intention-revealing, prefixed: `question_id`, `corpus_token_count`, `book_key`, `latency_seconds`,
  `total_cost_usd` — not `id`, `n`, `t`. Match the existing field names exactly.
- `snake_case` functions/vars/modules, `PascalCase` classes, `UPPER_SNAKE_CASE` constants/enum tables.

## Logging (structured JSON via structlog)

- `logger = structlog.get_logger()` per module. Event names are `snake_case` verbs:
  `corpus_fetch_started`, `rag_query_completed`, `wiki_ingest_checkpoint`, `judge_scored`,
  `sweep_cell_failed`.
- Every `logger.error` carries a `fix_suggestion`. Never log secret values.
- A multi-day headless run must be reconstructable from the logs alone — log every cell start/finish,
  every retry, and every measured-vs-extrapolated decision.

## Error handling

- **Fail loud, never silently truncate (Rule 12).** A per-cell failure becomes an **error `ResultRow`**,
  not a swallowed exception and not a halted sweep. A source that won't fetch is logged and skipped —
  but the corpus size it produces is flagged, never silently shrunk.
- Enforce measured ceilings explicitly: the plain-LLM path returns a **skipped row with a reason**
  past the context wall; it does not raise.

## Docstrings

- Google-style with a runnable `Example:` for any non-obvious function (prior art:
  `questions_for_corpus_size`). Inline `# Reason:` for non-obvious *why*.

## Secrets

- All keys via `settings.get_settings()` (`pydantic-settings`), never `os.environ` directly, never
  hardcoded, never logged. `.env` is gitignored; keep `.env.example` current.
