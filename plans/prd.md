# PRD — Wiki vs RAG Benchmark Engine

**Date:** 2026-07-05
**Source:** documents/product-brief.md
**Status:** Ready for /to-issues

## Problem Statement

A technical hiring manager skims a candidate's portfolio for ~2 minutes and decides whether the
person can reason rigorously about a real AI-systems trade-off. "RAG vs X" is the most over-written
topic on the internet and almost all of it concludes "it depends" — shallow content reads as shallow
understanding and does not earn a follow-up conversation. The candidate needs a **specific,
defensible, slightly surprising finding** about a real retrieval trade-off, backed by original
evidence, not another generic explainer.

## Solution

A benchmark engine that produces one defensible finding and the charts to prove it:

> A plain long-context LLM wins on accuracy until the corpus stops fitting in its context window
> (~1M tokens). Past that wall you are forced to choose: **RAG is cheap, fast, and shallow; an
> agent-built wiki (LLMWiki) is costly, slow, and deep.** The wiki's ingest cost climbs ~50–100×
> above RAG's — and that cost gap is the finding.

The engine drafts questions along a difficulty axis, answers each via three paths (plain LLM, RAG,
LLMWiki), grades every answer against a **frozen, human-verified golden key** over famous novels and
movie scripts (answers anyone can check — not "LLMs grading LLMs"), and aggregates latency, cost, and
accuracy per path as the corpus scales from 100k to 50M tokens (measured) and extrapolates to 1B. The
output is a crossover accuracy chart and a cost curve, rendered in a simple visualization UI that
doubles as the figures for a blog post — the primary interview artifact.

## Technical Foundation

*This section is the durable technical north star — there is no separate master plan.*

### Tech stack

- **Engine / language:** **Python 3.12+** — matches the existing scaffolding (`corpus.py`,
  `models.py`, `questions.py`, `settings.py`), the batch-pipeline shape, and the Anthropic/Voyage/
  Pinecone SDKs. Pydantic v2 for all cross-module data; `structlog` for JSON logs.
- **RAG data layer:** **Pinecone serverless** (vector store) + **Voyage** (`voyage-3`, 1024-dim)
  embeddings — free Starter tier covers the affordable sweep; embeddings ~$30 one-time. Already
  wired in `settings.py`.
- **LLMs:** **Claude Sonnet 5** = plain-LLM baseline (**1M-token context beta**, `[1m]`) *and* RAG
  generation; **Claude Opus 4.8** = accuracy judge (structured `JudgeVerdict`). Single vendor keeps
  the narrative clean.
- **Wiki path:** **LLMWiki driven via the Claude Code CLI** (`claude -p`, MCP config from
  `./llmwiki mcp-config`) on the author's **Max $200/mo plan** (marginal $0), capturing
  `total_cost_usd` as the reported **API-equivalent** cost.
- **UI:** **Next.js (static export) + Tailwind + a charting lib (Recharts)**, deployed to **Vercel**.
  Deliberately thin — one page reading **pre-baked results JSON** (never live ingestion). Its charts
  are lifted directly into the blog post.
- **Background jobs:** **None (no Trigger.dev).** Off-hours wiki-ingest loops are a plain scheduled
  shell/cron script — a job queue is unjustified complexity here (Rule 2).
- **Agent layer:** **None (no Pydantic AI / LangChain).** The "agents" are `claude -p` subprocess
  calls and direct SDK calls; there is no multi-tool agent to orchestrate.

### Architecture

```
                         ┌───────────────────────────────────────────────┐
                         │             SWEEP RUNNER (harness)             │
                         │  for each token-target size × system × question│
                         └───────────────────────────────────────────────┘
                            │                │                 │        │
              ┌─────────────┘                │                 │        └──────────────┐
              ▼                              ▼                 ▼                        ▼
        ┌───────────┐                 ┌────────────┐    ┌──────────────┐         ┌───────────┐
        │  CORPUS   │                 │ PLAIN-LLM  │    │   RAG PATH   │         │ WIKI PATH │
        │ novels +  │──cumulative────▶│  Sonnet 5  │    │ Voyage embed │         │ llmwiki   │
        │ movie scr.│  token-target   │  1M ctx    │    │  → Pinecone  │         │ via       │
        │ + chunker │  loading        │ (≤~1M only)│    │  → retrieve  │         │ claude -p │
        └───────────┘                 └────────────┘    │  → Sonnet 5  │         │ (Max plan)│
              │                              │           └──────────────┘         └───────────┘
              │  fixed golden anchors        │                 │                        │
              ▼                              └───────┬─────────┴────────────┬───────────┘
        ┌───────────┐                                ▼                      │
        │ GOLDEN SET│                          ┌───────────┐                │
        │ frozen Qs │─────────gold answers────▶│   JUDGE   │◀───answers─────┘
        │ + answers │                          │ Opus 4.8  │
        └───────────┘                          └───────────┘
                                                     │ ResultRow (0/0.5/1, latency, cost)
                                                     ▼
                                        ┌───────────────────────────┐
                                        │  AGGREGATE + EXTRAPOLATE   │  results.csv / results.json
                                        │  measured→1B curve fit     │──────────────┐
                                        └───────────────────────────┘              ▼
                                                                            ┌──────────────┐
                                                                            │  UI (Next.js)│→ blog figures
                                                                            │ pre-baked JSON│
                                                                            └──────────────┘
```

### Key design decisions

1. **Three measured paths, single LLM vendor.** `SystemName` gains `PLAIN_LLM` alongside `RAG` and
   `WIKI`. Rationale: the plain-LLM→retrieval **crossover is the headline chart**; dropping it (as the
   current scaffolding does) removes the finding. All three use Claude → clean narrative, one API key.
   *Rules out* a multi-vendor baseline (Gemini) that would muddy the story.
2. **The sweep axis is cumulative token count, not book count.** Corpus loads famous works in a fixed
   order until each token target is hit (100k → 500k → 1M → 2M → 5M → 10M → 20M → 50M). Rationale:
   the thesis is about *tokens vs the context wall*, so the axis must be tokens. *Rules out* the
   scaffolding's `corpus_size = 1..5` book-count axis (kept only as an internal ordering detail).
3. **Fixed golden anchors, growing haystack.** Golden questions target famous works loaded at the
   **smallest** sweep size; every larger size adds "haystack" content but the same questions are asked
   at every size. Rationale: this is what measures *retrieval degradation as the corpus grows* —
   accuracy decay under dilution is the mechanism behind the crossover. *Rules out* adding new
   questions per size (which would confound difficulty with size).
4. **Each path has an honest measured ceiling, then extrapolate.** Plain-LLM is measured only while
   the corpus ≤ ~1M tokens (its context wall), then drops out. Wiki is measured to a modest cap
   (~2–5M tokens) then extrapolated — building the wiki over 50M tokens is days of wall-clock under
   Max-plan rate limits. RAG is measured to the full 50M. Everything is extrapolated to 1B. The
   measured/extrapolated boundary is a **logged field on every row**, never hidden (Rule 12).
5. **Report API-equivalent cost even though the author pays $0.** The wiki runs on the Max plan
   (marginal $0), but `total_cost_usd` from `claude -p` / LLMWiki is captured and reported so the
   cost gap — the whole thesis — is not buried behind a flat subscription.
6. **Human-verified golden key, frozen before any run.** ~30–50 questions across three tiers, hand-
   verified against the source text, locked. The judge (Opus 4.8) grades *against this key*, not from
   its own knowledge. This is the credibility anchor; it is built and verified **first** (M0).
7. **UI reads pre-baked results only.** The slider selects a pre-computed corpus size; it never
   ingests live. A pre-recorded demo video is the backup so a live failure never becomes the story.

### Module contracts (deep modules — plain prose, NOT tests)

- **Corpus** — *Responsibility:* turn public novels (Gutenberg) + famous movie scripts
  (subslikescript / Kaggle dump) into a deterministically-ordered, cleaned, chunked corpus, and load
  the first-N-tokens for any sweep target. *Requirements:* the same token target must always yield the
  same content on every run (fixed source order); PG/scraper boilerplate stripped; each chunk carries
  provenance (source key, title, offset). *Edge cases:* a source fetch fails (skip + log, don't
  silently shrink the corpus); a token target exceeds available corpus (cap at max available + flag);
  a movie page is JS-only / empty (fall back to Playwright, then to Kaggle dump); duplicate/near-dup
  scripts.
- **Golden set** — *Responsibility:* hold the frozen, human-verified questions and gold answers, and
  return those answerable at a given sweep size. *Requirements:* every question's anchor work is loaded
  at the smallest size it appears; gold answers are reference text (judge input), not exact-match
  strings; the set is immutable once frozen (a checksum guards accidental edits). *Edge cases:* a
  question whose anchor is not yet loaded (never returned for that size); tier balance across sources.
- **RAG path** — *Responsibility:* embed chunks (Voyage) → upsert (Pinecone) → retrieve top-k for a
  question → generate an answer (Sonnet 5) from only the retrieved context. *Requirements:* index is
  namespaced/keyed by corpus size so sizes don't cross-contaminate; retrieval and generation cost +
  latency are captured per query. *Edge cases:* Pinecone free-tier vector cap exceeded (log + switch
  namespace strategy or flag ceiling); empty/zero-hit retrieval; embedding rate limits (backoff).
- **Plain-LLM path** — *Responsibility:* stuff the whole loaded corpus into Sonnet 5's 1M context and
  answer. *Requirements:* refuses to run (returns a skipped row, not an error) once corpus > context
  limit; captures input-token cost. *Edge cases:* corpus just under vs over the 1M wall (the crossover
  point — must be exact and logged); 1M beta header wiring.
- **Wiki path** — *Responsibility:* drive LLMWiki through `claude -p` to ingest the corpus and answer
  a question, capturing `total_cost_usd` and wall-clock. *Requirements:* ingest is idempotent/resumable
  across off-hours loops; API-equivalent cost captured even on the Max plan; measured only to the cap,
  then rows past the cap are marked extrapolated. *Edge cases:* rate-limit / usage-window exhaustion
  mid-ingest (checkpoint + resume, don't restart); CLI non-zero exit; partial wiki.
- **Judge** — *Responsibility:* grade one answer against its gold answer → `JudgeVerdict` (0 / 0.5 /
  1 + one-line rationale). *Requirements:* judges strictly against the gold text, deterministic prompt,
  structured output enforced. *Edge cases:* empty/errored answer (score 0 with rationale); answer
  correct but differently worded than gold (0.5/1 with reason); refusal.
- **Sweep runner** — *Responsibility:* orchestrate every (size × system × question) cell into a
  `ResultRow` and aggregate to `results.csv` / `results.json`. *Requirements:* resumable (a crash
  mid-sweep resumes from the last completed cell); each row records measured-vs-extrapolated; failures
  become error rows, not a halted sweep (Rule 12). *Edge cases:* a single system failing shouldn't kill
  the row for other systems; duplicate-cell guard on resume.
- **Aggregate + extrapolate** — *Responsibility:* fit cost/accuracy curves from measured data and
  extrapolate to 1B, emitting chart-ready series + headline summary stats. *Requirements:* extrapolated
  points are visually/structurally distinct from measured; the fit method and its boundary are stated.
  *Edge cases:* too few measured points to fit (flag low confidence); wiki cost fit dominated by a
  single expensive point.
- **UI** — *Responsibility:* render the crossover chart, the cost curve, and a corpus-size-slider /
  question-picker view of the three paths' answers, from pre-baked JSON. *Requirements:* no live model
  calls; measured vs extrapolated visually marked; works as static export. *Edge cases:* size where a
  path dropped out (show "exceeded context", not blank); missing results file.

### Milestones (coarse — slices come from /to-issues)

- **M0 — Golden set frozen & human-verified:** ~30–50 Qs across 3 tiers over the anchor novels +
  movie scripts, hand-verified against source, checksum-locked. *De-risks the riskiest assumption
  before any sweep runs.*
- **M1 — Corpus pipeline:** novels + movie-script sourcing (subslikescript scraper w/ Kaggle
  fallback), cleaning, deterministic order, chunking, and token-target cumulative loading.
- **M2 — RAG path E2E** at the smallest size: embed → Pinecone → retrieve → Sonnet 5 → judge →
  `ResultRow`.
- **M3 — Plain-LLM baseline** (Sonnet 5, 1M context) with the crossover visible at small sizes.
- **M4 — Wiki path** via `claude -p`, measured to the cap, `total_cost_usd` captured, resumable.
- **M5 — Full sweep + aggregate + extrapolate to 1B:** `results.csv/json`, measured/extrapolated
  boundary logged.
- **M6 — Visualization UI** (Next.js, pre-baked JSON): crossover chart, cost curve, slider view.
- **M7 — Blog post** assembled from the UI's figures + narrative. The unbreakable interview artifact.

### Riskiest assumption + how we de-risk it

That the **accuracy numbers are trustworthy** — a frozen, human-verified golden key graded *against
that key* yields scores a skeptical interviewer can't dismiss as "LLMs grading LLMs." If grading isn't
credible, the whole finding collapses. **De-risked in M0:** build and hand-verify the golden set, and
spot-check the judge's scores against a human pass, before running any sweep.

## User Stories

1. As the author, I want a frozen set of ~30–50 human-verified questions with gold answers over
   famous novels and movie scripts, so that my accuracy numbers are defensible and not "LLMs grading
   LLMs."
2. As the author, I want the golden set checksum-locked, so that no accidental edit can silently
   change a published result.
3. As the author, I want questions tagged by difficulty tier (lookup / relational / timeline), so
   that I can show *where* each path breaks, not just an average.
4. As the author, I want to fetch and clean Project Gutenberg novels in a fixed order, so that the
   same token target always maps to the same content across runs.
5. As the author, I want to source famous movie scripts from subslikescript (with a Kaggle-dump
   fallback), so that the corpus reaches 50M tokens with recognizable, checkable content.
6. As the author, I want the corpus loader to assemble the first-N-tokens for any sweep target, so
   that the sweep axis is tokens, not book count.
7. As the author, I want each chunk to carry provenance (source, title, offset), so that I can trace
   any retrieved context back to its origin.
8. As the author, I want the RAG path to embed with Voyage, upsert to Pinecone, retrieve top-k, and
   answer with Sonnet 5, so that I have a realistic, cheap retrieval baseline.
9. As the author, I want the RAG index keyed by corpus size, so that different sweep sizes don't
   cross-contaminate retrieval.
10. As the author, I want a plain-LLM path that stuffs the whole corpus into Sonnet 5's 1M context,
    so that I have the long-context baseline that defines the crossover.
11. As the author, I want the plain-LLM path to emit a "skipped — exceeds context" row (not an error)
    once the corpus passes ~1M tokens, so that the context wall is explicit in the data.
12. As the author, I want a wiki path that drives LLMWiki via `claude -p` to ingest the corpus and
    answer questions, so that I can measure the agent-built-wiki side of the thesis.
13. As the author, I want the wiki ingest to be resumable across off-hours loops, so that a
    rate-limit or usage-window cutoff doesn't force a restart.
14. As the author, I want the wiki path to capture `total_cost_usd` (API-equivalent) even though I pay
    $0 on the Max plan, so that the cost gap — the thesis — is reported, not buried.
15. As the author, I want an Opus 4.8 judge that grades each answer against the gold answer into a
    structured 0 / 0.5 / 1 verdict with a rationale, so that scoring is consistent and auditable.
16. As the author, I want every graded cell written as a `ResultRow` (system, size, question, latency,
    cost, accuracy, answer, rationale, measured-or-extrapolated, error), so that one CSV is the full
    record of a run.
17. As the author, I want the sweep runner to be resumable and to turn per-cell failures into error
    rows, so that one flaky call never halts or silently truncates a multi-day run.
18. As the author, I want each path's measured ceiling enforced (plain-LLM ~1M, wiki ~2–5M, RAG 50M)
    and the measured/extrapolated boundary logged per row, so that the honest limit of the data is
    visible.
19. As the author, I want cost and accuracy curves fitted from measured data and extrapolated to 1B
    tokens, so that I can show the trajectory past what I can afford to measure.
20. As the author, I want extrapolated points visually distinct from measured points, so that no
    reader mistakes a projection for a measurement.
21. As the author, I want a simple Next.js page that renders the crossover accuracy chart and the cost
    curve from pre-baked JSON, so that I have publication-ready figures for the blog.
22. As the author, I want a corpus-size slider and question picker that shows the three paths' answers
    side by side at the selected size, so that a viewer can *feel* the trade-off, not just read a chart.
23. As a technical hiring manager, I want to see measured-vs-extrapolated clearly marked in the UI, so
    that I trust the methodology instead of discounting it.
24. As the author, I want a pre-recorded demo video of the UI, so that a live failure during an
    interview never becomes the story.
25. As the author, I want a blog post assembled from the engine's charts and a narrative of the
    finding, so that I have one unbreakable artifact that earns the interview conversation.
26. As the author, I want structured JSON logs with a `fix_suggestion` on errors across the pipeline,
    so that a multi-day headless run is debuggable after the fact.

## Implementation Decisions

- **Extend `models.py`, don't fork it.** Add `PLAIN_LLM` to `SystemName`. Extend `ResultRow` with a
  token-count field (the true sweep axis), a `measured_or_extrapolated` flag, and a `source`-agnostic
  `corpus_token_count`; keep `corpus_size` as an internal book/script-count ordering detail. Add a
  `SkippedReason` for the plain-LLM context-wall row.
- **Extend `corpus.py`** with a movie-script source (subslikescript scraper → `div.full-script` in
  `article.main-article`; Playwright fallback; Kaggle-dump fallback) and a `load_for_token_target(n)`
  that assembles cleaned content in fixed order up to `n` tokens.
- **Expand `questions.py`** from 15 to ~30–50 Qs, adding movie-anchored questions, and add a checksum
  guard over the frozen bank.
- **New modules** (one deep module each, per the contracts above): `rag.py`, `plain_llm.py`,
  `wiki.py`, `judge.py`, `runner.py`, `analysis.py`. Keep each under the 500-line agent-code limit;
  business logic in these, wiring thin.
- **Cost accounting** lives in one place (a small `cost.py` or fields on `ResultRow`) so
  API-equivalent pricing for all three paths is computed identically.
- **UI** is a separate `ui/` Next.js app (static export) reading `results.json` copied from the
  engine output; no shared runtime with Python.
- **Config** stays in `settings.py` (`pydantic-settings`); add movie-source + 1M-context-beta +
  sweep-target settings there.

## Testing Decisions

- Test **external behavior, not implementation** (Rule 9). Mock every external boundary — Anthropic,
  Voyage, Pinecone, `claude -p`, HTTP/Playwright — never hit real services in tests (CLAUDE.md §6).
- Per function: happy path + one intentional failure + one edge case (empty corpus, over-context,
  rate-limit, zero-hit retrieval, checksum mismatch).
- The **golden-set checksum** and the **plain-LLM context-wall skip** are the two tests that most
  directly encode *why* the finding is trustworthy — prioritize them.
- Mirror the scaffolding's existing style (frozen dataclasses, Pydantic models, `structlog`). Prior
  art to copy: `questions_for_corpus_size` (pure, table-driven) and the `models.py` schemas.

## Out of Scope

- **Live corpus ingestion in the UI** — the slider only selects pre-baked sizes.
- **A second LLM vendor** for the baseline (Gemini, etc.) — single-vendor by decision #1.
- **Fine-tuning and long-context-vs-RAG-only comparisons** — the novelty is RAG-vs-agent-wiki.
- **A job queue / Trigger.dev** — off-hours loops are a plain scheduled script.
- **Auth, multi-user, payments** — this is a single-author benchmark + a static read-only UI.
- **Measuring the wiki past its cap** — extrapolated, by decision #4.

## Further Notes

- **subslikescript** is JS-rendered at the homepage but detail pages
  (`/movie/<Title>-<id>`) carry server HTML with the transcript in `div.full-script`; try
  `requests`+`bs4` first, fall back to Playwright, then to the Kaggle "59K movie transcripts" dump.
  Movie scripts are copyrighted — used here only as private benchmark input; state this in the blog's
  methodology.
- The **wiki runs on the Max plan** (marginal $0) but must report **API-equivalent** cost; schedule
  ingest loops for off-hours / the tail of usage windows so they don't disrupt real work.
- The 90-day metric is measurable with this stack: the crossover chart + a wiki-ingest cost curve
  ~50–100× RAG's is exactly what `analysis.py` emits.
