---
title: Token-target cumulative corpus loader — deterministic axis, whole-source boundary, loaded-keys seam
tags: [corpus, token-target, tiktoken, determinism, provenance, loader, sweep-axis]
problem_type: convention
symptoms: A loader must assemble "first-N-tokens" of a fixed-order corpus deterministically, carry provenance, flag a shortfall, and stay consistent with the question bank's source order.
date: 2026-07-05
---

Pattern used by `corpus_loader.load_for_token_target` (issue #4). The shared corpus contract the
RAG / plain-LLM / wiki / sweep-runner slices (#6–#9) all consume.

## Token axis: injectable counter, offline-deterministic default
- The sweep axis is **cumulative tokens** (PRD decision #2). Exact Anthropic counts need a per-call
  API round-trip — unusable in a deterministic, network-free assembly path. Use `tiktoken`
  `cl100k_base` as the default proxy: stable, cached-offline after first vocab load, version-pinned
  by name (same text → same count across runs/machines). Encode with `disallowed_special=()` so a
  literal `<|endoftext|>` in a script is counted as plain text, never raises.
- Make the counter **injectable** (`count_tokens: TokenCounter = count_tokens_tiktoken`). Tests pass
  a trivial `len(text.split())` counter → hermetic (no tiktoken, no vocab download); a future axis
  swap is one line. `total_token_count` is always the injected counter applied to the loaded
  sources, so the reported axis never drifts from how it was measured.
- Lazy-import tiktoken inside the counter + cache the encoder in a module global (like corpus.py's
  Playwright lazy import); declare `tiktoken` in requirements.txt anyway (honest contract).

## Boundary policy: whole sources, include the crossing one (document it)
- Accumulate sources **whole, in fixed order** until cumulative ≥ target; the source that crosses
  the target is included in full, never truncated. Rationale: a golden question anchored to a work
  must see it whole to be answerable at the smallest sweep size (PRD decision #3), and whole-source
  char offsets stay clean for provenance. Consequence: the corpus is "at least ~N tokens" (capped at
  max available) — the coarse targets (100k, 500k, 1M…) tolerate the overshoot. Either whole-source
  or truncate-to-boundary is fine **if deterministic and documented**; this repo chose whole-source.

## Provenance + shortfall (Rule 12)
- Each `LoadedSource` carries `book_key`, `title`, and `char_offset` (start index of its text in the
  assembled `CorpusLoad.text`, which joins sources with a fixed `"\n\n"` separator — offsets account
  for the separator so `text[offset:offset+len]` round-trips). Offset is **char**, not token.
- `shortfall = cumulative_tokens < token_target` after the loop: set the flag AND `logger.warning`
  with `fix_suggestion` — never a silent short read. An empty/zero-length source is skipped before
  the offset increment (no phantom separator), so it contributes nothing and never shifts the order
  or offsets of real sources.

## Determinism + composition seam
- No set/dict iteration, no wall-clock, no network in the assembly path. Same target + books +
  counter → byte-identical `CorpusLoad` (frozen dataclasses compare by value — assert `a == b`).
- Expose `loaded_source_keys` (fixed order) as the **single source of truth** for which works are
  loaded at N. `questions.questions_for_token_target(load.loaded_source_keys)` filters answerable
  questions off exactly this — questions.py does NOT re-derive the token→sources decision. An
  integration test asserts real composition (loaded keys → answerable questions), not a stub.
- Callers that load at many targets (the sweep runner) should `assemble_ordered_corpus()` **once**
  and pass `books=` to each `load_for_token_target` call — the `books=None` default re-fetches
  (and `fetch_movie_corpus` re-scrapes, no cache), so inject-and-reuse is the intended pattern.
