---
title: RAG path — shared path-agnostic cost model, size-keyed namespace isolation, multi-service boundary mocking
tags: [rag, voyage, pinecone, anthropic, cost, pinecone-namespace, mocking, pytest, retry, resultrow]
problem_type: convention
symptoms: A pipeline calls three external services (embeddings + vector DB + LLM); its cost must be computed identically across paths, sizes must not cross-contaminate, and tests must never hit a live API or require the SDKs installed.
date: 2026-07-05
---

Established building the RAG path (`rag.py` + `cost.py`, issue #6). Reuse for the
plain-LLM (#7), wiki (#8), and sweep-runner (#9) slices.

## Shared, path-agnostic cost model (`cost.py`)
- Cost lives in **one** pure module: a `PRICING` table (`{model: ModelPrice(input_usd_per_million, output_usd_per_million)}`)
  + `token_cost_usd(model, *, input_tokens=0, output_tokens=0)`. No network, no
  clients, no path-specific logic — so #7/#8 price their own calls with the same
  function. Document the list-price source + date inline (Sonnet 5 $3/$15, Opus 4.8
  $5/$25, voyage-3 $0.06/1M).
- A path assembles `ResultRow.cost_usd` by **summing** per-call `token_cost_usd`
  results (RAG = query-embedding + generation). That summation is not "inline
  pricing" — the per-token arithmetic stays in `cost.py`.
- Unknown model → **raise**, never return `0.0` (Rule 12): a mispriced cell must
  surface, not silently under-report the cost gap (the thesis). Tests pin the
  documented prices, prove linearity (so summing is valid), and prove the raise.
- Per-query cost = query-embed + generation only. Corpus embedding is a one-time
  **indexing** cost, not a per-query charge — don't fold it into every row.

## Size-keyed Pinecone namespace isolation
- Namespace **must** encode the sweep size: `_namespace_for(corpus_token_count) ->
  f"corpus-{n}"`. Index and query derive it from the same function, so a chunk from
  size A is never retrievable when querying size B.
- Test it for real with a fake Pinecone that stores upserts per-namespace and returns
  only that namespace's vectors on query; index two sizes with distinguishable text
  (`"AAAA…"` vs `"BBBB…"`), query one, assert the **generation prompt** contains only
  that size's text. This proves isolation end-to-end, not just that two dicts differ.
- Split the seam so the corpus is embedded **once per size** (`index_corpus`), reused
  across all its questions (`run_rag_query` embeds only the query) — the perf contract
  the sweep runner depends on.

## Multi-service injected-boundary mocking (Voyage + Pinecone + Anthropic)
- Every external client is a keyword-only injected param (`voyage_client=None,
  pinecone_index=None, anthropic_client=None`); a real one is built from settings via
  a lazy `_build_*` helper **only when the arg is None**. Import the SDKs *inside* the
  builder (lazy) so tests that inject fakes need neither the package nor a `.env`.
  (voyageai/pinecone weren't installed — an accidental live call would `ImportError`,
  which doubles as a guarantee no test hits a live API.)
- Fakes mirror the exact SDK shapes: Voyage `.embed(texts, model, input_type)` ->
  `SimpleNamespace(embeddings=[...], total_tokens=N)`; Pinecone `.upsert(vectors=,
  namespace=)` / `.query(vector=, top_k=, namespace=, include_metadata=)` ->
  `SimpleNamespace(matches=[SimpleNamespace(metadata={"text": ...})])`; Anthropic a
  recording `messages.create` returning queued responses (generation text block +
  `usage`, then the judge's forced `tool_use` block).
- Integration test wires the **real** `corpus_loader` + **real** `judge` parse + **real**
  `cost.py` with only the three service boundaries mocked (Anthropic `side_effect =
  [generation_response, judge_tool_response]`) — catches contract drift a fully-stubbed
  happy path would miss.

## Bounded embedding backoff + per-query error rows
- Wrap embedding in a bounded retry (`MAX_EMBED_ATTEMPTS`, exponential backoff via an
  injectable `sleep=time.sleep` so tests pass a no-op) — rides out a Voyage rate limit,
  then **re-raises** (never infinite-loops). Test both: raises-then-succeeds, and
  gives-up-after-N.
- A per-query failure is caught at the query seam and returned as an **error
  `ResultRow`** (`error` populated, `measured_or_extrapolated="measured"`), so one bad
  cell never halts the sweep (Rule 12) — surface, don't hide, but don't crash the run.
- Free-tier vector cap: flag + log when chunk count would exceed the Pinecone Starter
  cap (`PINECONE_STARTER_VECTOR_CAP`); test by monkeypatching the cap to a small number.
  Surface the ceiling, don't silently degrade.
