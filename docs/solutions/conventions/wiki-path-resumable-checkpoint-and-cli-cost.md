---
title: Wiki path — resumable atomic checkpoint, `claude -p --output-format json` cost parsing, subprocess-boundary mocking
tags: [wiki, llmwiki, claude-cli, subprocess, checkpoint, resumable, atomic-write, total_cost_usd, mocking, pytest, resultrow]
symptoms: A path drives an external CLI (`claude -p`) to do long, rate-limited work (ingest) that must resume after a usage-window cutoff, capture the CLI's own reported dollar cost, and be tested without ever spawning the real CLI.
---

Established building the wiki path (`wiki.py`, issue #8). Reuse for any path that drives a
CLI/subprocess it can't hit in tests, or needs resumable off-hours work.

## Subprocess boundary = one injected `cli_runner` callable
- Define `CliResult(returncode, stdout, stderr)` + `CliRunner = Callable[..., CliResult]`.
  The ONLY real spawn is a default `_run_claude_cli(args, *, input_text=None, timeout=...)`
  using `subprocess.run(args, input=..., text=True, timeout=..., check=False)` — **arg
  list, `shell=False`**, corpus content on **stdin (`input=`)**, never in argv (no command
  injection; the `-p` prompt is only fixed instructions + controlled golden-set text).
- Every seam takes `cli_runner: CliRunner = _run_claude_cli`. Tests inject a
  `RecordingCliRunner` that returns queued `CliResult`s **or raises queued exceptions**
  (to simulate a mid-run cutoff). Prove hermeticity with a test that `monkeypatch.setattr`s
  `_run_claude_cli` to a tripwire that raises — it must never fire (CLAUDE.md §6).
- Always set a subprocess `timeout` so a wedged CLI becomes an error row, not a hang.

## Cost comes straight from the CLI JSON — do NOT route through `cost.py`
- `claude -p --output-format json` emits `{"result": "...", "total_cost_usd": <float>,
  "duration_ms": <int>, "is_error": <bool>}`. `total_cost_usd` is already **API-equivalent
  USD** (the author pays $0 on Max, but the field reports metered-API cost — the thesis gap).
- So `ResultRow.cost_usd = total_cost_usd` **directly**. `cost.py` prices raw *token usage*
  → USD (RAG/plain-LLM); the wiki already has USD, so it bypasses `cost.py`. Document this —
  it looks like a missed reuse but is the correct call.
- Latency = CLI `duration_ms / 1000`, falling back to a measured `perf_counter` span.
- Parse loudly: non-zero exit / `is_error` / unparseable JSON → raise (never silently treat
  as a $0 success, which would under-report the cost gap). Classify rate-limit messages
  (`"rate limit"`, `"usage limit"`, ...) into a distinct `WikiRateLimitError` = resume signal.

## Resumable ingest = per-item atomic checkpoint keyed by size
- Checkpoint is a small JSON file `wiki-checkpoint-<corpus_token_count>.json` in a
  `checkpoint_dir` (a setting, gitignored) holding `{corpus_token_count, ingested_source_keys,
  ingest_cost_usd}`. Keyed by token count so sizes never share state (mirrors RAG's
  size-keyed Pinecone namespace).
- Load checkpoint → `already = set(ingested_source_keys)` → loop sources, `continue` on ones
  already done (idempotent; a re-run of a complete ingest is a no-op with **zero** CLI calls),
  else ingest → **persist after each source, atomically** (`tmp = dir/.name.<uuid>.tmp;
  tmp.write_text(...); os.replace(tmp, path)` — atomic within a fs, so a crash mid-write can't
  corrupt resume state).
- On `WikiRateLimitError` mid-loop: `break` with the checkpoint intact and `interrupted=True`.
  The current source raised *before* being appended, so it retries next run. The next off-hours
  run RESUMES (skips done keys), never restarts. Test it end-to-end: run 1 does source A then a
  raised rate-limit; assert checkpoint has only A; run 2 with a fresh runner ingests B/C and the
  runner is called **only** for B/C (proves A was skipped, not re-done).

## Seam shape (mirror `rag.py` for the uniform sweep-runner interface)
- `ingest_corpus(corpus_load, *, cli_runner, checkpoint_dir, settings) -> IngestResult`
  (one-time per size, like `index_corpus`); `run_wiki_query(question, *, corpus_token_count,
  corpus_size, cli_runner, judge_client, settings) -> ResultRow` (per question, like
  `run_rag_query`); `run_wiki_slice(...)` convenience = ingest + query.
- Measured ceiling: `corpus_token_count > MEASURED_TOKEN_CAP` (5M, top of the PRD 2–5M band)
  → emit an `extrapolated` ResultRow with **no CLI call** (never silent, never mislabelled);
  boundary is exact (`==` cap is still measured) and **logged**. Within cap → `measured`.
- The judge stays the shared `grade_answer(answer, gold, client=judge_client)`; inject
  `judge_client` and `monkeypatch judge.get_settings` in tests (it reads only `judge_model`).
