---
title: Plain-LLM context-wall skip row + the Sonnet 5 "1M beta header" is a no-op
tags: [anthropic, claude, sonnet-5, context-window, 1m, beta-header, resultrow, skip, crossover, mocking]
problem_type: convention
symptoms: A long-context path must emit a "skipped — exceeds context window" row (not an error, not a raise) past the ~1M wall, price the answered case via the shared cost model, and wire a "1M context beta header" that may not actually be needed.
date: 2026-07-05
---

Established building the plain-LLM baseline (`plain_llm.py`, issue #7). Reuse for the
wiki path's measured ceiling (#8) and the sweep runner (#9).

## Context-wall skip row (the crossover, made explicit in the data)
- The wall is a **module constant** compared on the **same tiktoken axis
  `corpus_loader` owns**: `corpus_load.total_token_count > CONTEXT_WINDOW_TOKEN_LIMIT`
  (`1_000_000`). `> limit` skips; `<= limit` answers — so the crossover is **exact**
  (at the wall → answered, one token over → skipped). Compare against the loader's
  count, never a parallel re-count, or the boundary silently drifts.
- Past the wall: return a `ResultRow(system=PLAIN_LLM, skipped_reason="exceeds_context_window",
  measured_or_extrapolated="measured", cost_usd=0.0, accuracy=0.0, answer_text="")`
  with **no API call and no raise**. Build the Anthropic client *after* the wall check
  so a skip never even constructs a client (belt-and-suspenders: a skipped size can't
  hit a live API). A skip is `measured` (an observed wall) with a `skipped_reason` — it
  is **not** `extrapolated` (that boundary belongs to the wiki path).
- The crossover must be a **logged field, not inferred**: emit a structured
  `plain_llm_context_wall_skipped` event carrying `corpus_token_count`,
  `context_window_limit`, `over_by`, `skipped_reason`. Assert it in tests with
  `structlog.testing.capture_logs()` (no stdlib-logging config needed).
- A generation failure **below** the wall is an **error `ResultRow`** (populated
  `error`), caught by a broad `except` that returns the row — never a raise, never
  mislabelled as a wall skip. Track `corpus_token_count`/`corpus_size` in locals
  initialised to 0 *before* the try so the error row reports the real size the failure
  happened at (honest, not zeroed) when the corpus had already loaded.

## The "1M context beta header" is a no-op on Sonnet 5 (surfaced conflict)
- `reference/integrations.md` says to set a `context-1m` beta header for the baseline.
  The **authoritative `claude-api` skill** says current-generation Claude models —
  including `claude-sonnet-5`, Opus 4.8, etc. — expose the **1M window natively at
  standard pricing with no beta header**. The `context-1m-*` beta gated the older
  Sonnet 4 / 4.5 line; on Sonnet 5 it is a **harmless no-op**.
- Resolution (Rule 7 — surface, don't average): wire the documented value
  `CONTEXT_1M_BETA_HEADER = "context-1m-2025-08-07"` as a const via
  `messages.create(..., extra_headers={"anthropic-beta": CONTEXT_1M_BETA_HEADER})` to
  honour the reference doc and make intent explicit, but **do not expect it to be
  load-bearing**. If it ever starts 400-ing on a future model, drop it — don't "fix"
  the value.

## Seam + mocking (mirror `rag.py`)
- `run_plain_llm_query(question, size, *, anthropic_client=None,
  corpus_loader=load_for_token_target, settings=None) -> ResultRow`. Inject BOTH the
  Anthropic client and the corpus loader so tests stay hermetic (CLAUDE.md §6). Name it
  `anthropic_client` (not `anthropic`) to avoid shadowing `import anthropic`.
- Keep the `messages.create` seam identical to `rag.py` (extra kwargs like
  `extra_headers` ride along) so one `RecordingAnthropic` fake covers both paths; assert
  the whole corpus reached `messages[0]["content"]` and the beta header is on
  `extra_headers`.
- Reuse the model id from `settings.rag_generation_model` (`claude-sonnet-5` — the same
  Sonnet 5 model integrations.md ties to both baseline and RAG generation) and price via
  `cost.token_cost_usd(model, input_tokens=, output_tokens=)`. One pricing source, no
  inline cost math, no forked setting.
