# Product Brief

**Date:** 2026-07-05
**Status:** Draft — needs `/cto` to translate into a PRD

## One-liner
A benchmark engine that proves where RAG beats an agent-built wiki — and where it breaks.

## Target user
A **technical hiring manager or senior engineer** reviewing the author's portfolio during a
job-interview process. The moment they'd reach for it: skimming a candidate's work for ~2 minutes,
deciding whether this person can reason rigorously about a real AI-systems trade-off and is worth a
deeper conversation. Secondary reader: the broader technical audience who lands on the resulting
blog post.

## Problem
"RAG vs X" is one of the most over-written topics on the internet, and almost all of it concludes
"it depends." A candidate who wants to stand out in an interview needs a **specific, defensible,
slightly surprising finding** about a real retrieval trade-off — not another generic explainer. The
pain is credibility: shallow content reads as shallow understanding and does not earn the
conversation.

## Today's workaround
Candidates write generic "RAG vs long-context / RAG vs fine-tuning" blog posts, or link tutorials
they followed. These don't differentiate because they carry no original evidence and no rigorous
methodology — a skeptical interviewer discounts them immediately.

## Unique angle
Two things nothing else combines:
1. **The comparison is fresh.** Everyone benchmarks RAG against long-context or fine-tuning. Almost
   nobody benchmarks RAG against an **agent that builds a structured wiki** (LLMWiki) from the same
   corpus. That novelty is the wedge.
2. **The finding is quantified and defensible.** A frozen, **human-verified golden answer key**
   (over famous novels + famous movie transcripts, whose answers anyone can check) means the
   accuracy numbers are trustworthy — not "LLMs grading LLMs." The headline is a real crossover:
   **RAG is cheap / fast / shallow; the wiki is costly / slow / deep — and the plain LLM baseline
   wins until the corpus stops fitting in its context window (~1M tokens), at which point you are
   forced to choose.**

## Smallest provable version (MVP)
1. **The engine** — a batch pipeline: draft questions along a difficulty axis (shallow lookup →
   multi-hop relational/timeline) → answer each question via three paths (plain long-context LLM,
   Pinecone RAG, LLMWiki) → an evaluator agent grades each answer **against the frozen human-verified
   key** (0 / 0.5 / 1) → aggregate latency, cost, and accuracy per path per corpus size.
2. **The frozen golden set** — ~30–50 questions with human-verified reference answers, locked before
   any run.
3. **The scaling sweep** — run at 100k → 500k → 1M → 5M → 10M → 50M tokens; the plain-LLM path drops
   out once the corpus exceeds its context window; extrapolate cost/accuracy curves toward 500M.
4. **The blog post** — charts + narrative built from the engine's output. This is the primary,
   unbreakable interview artifact.
5. **(Stretch) A simple live UI** — a two-panel view with a **slider that selects a pre-baked corpus
   size** (never live ingestion) so a viewer can ask a question and watch the paths answer in real
   time. Pre-record a demo video as backup so a live failure never becomes the story.

## 90-day success metric
Leading indicator: the blog post / repo is used as a talking point in an interview and prompts the
interviewer to ask follow-up questions about the methodology (i.e., it earns the conversation).
Proxy the author can watch: the engine produces a clean accuracy-crossover chart and a cost curve
where the wiki's ingest cost climbs ~50–100x above RAG's — the core evidence the thesis needs.

## Competition
- **Direct:** the sea of "RAG vs long-context / RAG vs fine-tuning" blog posts and vendor
  benchmarks. Beaten by the fresh RAG-vs-agent-wiki angle + a verified golden key.
- **Indirect:** LLMWiki's own docs/examples; Pinecone's marketing benchmarks (not neutral).
- **Do nothing:** the author writes a generic explainer or links a tutorial — cheap, but doesn't
  differentiate in an interview.

## Economics (as analyzed this session)
- **Raw storage is trivial:** 500M tokens ≈ ~2 GB of text.
- **RAG side is cheap:** ~1.1M vectors at 500M tokens; embeddings ~$30 one-time (Voyage 3.5);
  Pinecone **free Starter tier covers the whole affordable sweep** (2 GB / ~300K vectors ≈ ~150M
  tokens; 2M write units + 1M read units/month) — only past ~100–150M tokens does the $50/mo
  Standard plan apply; per-query cost ~1.5¢.
- **Wiki side is the wall:** building the wiki requires an LLM to **read the entire corpus** — the
  API-equivalent cost is ~$1,000 at 500M tokens (bare minimum, single read) and realistically
  $2,000–$5,000+ with re-reads/thinking/writing, over days of wall-clock. **This gap IS the thesis.**
- **Author's real out-of-pocket ≈ $30** (Voyage embeddings). The wiki runs headless on the author's
  **Claude Max $200/mo plan** (marginal $0, constrained by rate limits, not dollars); ingest loops
  scheduled for off-hours / the tail of each usage window so they don't disrupt real work.
- **Honesty guardrail:** the engine must still capture and report the **API-equivalent** wiki cost
  (`total_cost_usd` from `claude -p` / LLMWiki), even though the author pays $0 — otherwise the
  thesis is quietly buried behind a flat subscription.

## What held up under pressure
- The audience and desired outcome (interview/portfolio → earn the conversation).
- The build ordering (engine + blog first; UI as a stretch layer with a recorded-video backup).
- The specific, testable thesis (RAG shallow, wiki deep, plain-LLM baseline until the context wall).
- The credibility anchor (frozen, human-verified golden key over famous, checkable content).
- The economics (RAG cheap, wiki expensive — the cost gap is the finding, verified against live
  vendor pricing).

## What's still soft
- **Exact golden-set size and the difficulty-axis definition** — ~30–50 Qs and a shallow→multi-hop
  spectrum are assumed; `/cto` should pin the rubric.
- **Which components run on the Max plan vs the API** (RAG generation with Sonnet 5, judge with
  Opus 4.8) — affects reported cost and rate-limit budgeting.
- **How far the wiki sweep can actually run** before rate limits/time cap it — determines where
  measured data stops and extrapolation begins. Must be logged, not hidden (Rule 12).
- **Movie-transcript corpus sourcing** (subslikescript.com is JS-rendered; needs headless scraping
  or Kaggle dumps) — feasibility not yet exercised.

## Riskiest assumption
That the **accuracy numbers are trustworthy** — i.e., that a frozen, human-verified golden key plus
an evaluator grading *against that key* produces defensible scores a skeptical interviewer can't
dismiss as "LLMs grading LLMs." If the grading isn't credible, the entire finding collapses. Test
this first: build and hand-verify the golden set before running any sweep.

## Contradictions surfaced
- **"Playground" vs. "automated experiment":** the user first said "definitely a live playground,"
  then described an automated agent pipeline whose deliverable is a blog post. Resolved: the
  **engine + blog post is the core** (unbreakable interview artifact); the live UI is a stretch
  layer. Not blended — ordered.
- **"The wiki is basically free" (Max plan) vs. the thesis that the wiki is expensive:** resolved by
  reporting the API-equivalent cost regardless of what the author personally pays.
- **Live slider ingestion vs. demo safety:** resolved — the slider switches between **pre-baked**
  corpus sizes; it never ingests live (ingesting millions of tokens can't happen in a demo window).

## Open questions
- Final golden-set size and the exact difficulty tiers / question templates.
- Corpus composition and the fixed ingestion order (which sources at which sweep size); movie-vs-book
  mix.
- Embedding model + chunking strategy for RAG (a `/cto` decision).
- Where measured sweep data ends and extrapolation begins on the wiki side.
- Whether generation/judge run on API or Max plan, and how off-hours ingest loops are scheduled.

## Prior art & positioning (from research, 2026-07-05)
The core scientific finding is **already established** — this project's value is packaging + rigor +
honest cost accounting for a portfolio/interview audience, NOT novel discovery. The blog post must
frame it as replication-made-tangible and **cite the prior art** (that literacy is itself the flex).

Closest prior work:
- **[Cochran, "Vector RAG vs LLM-Compiled Wiki" (arXiv 2605.18490)](https://arxiv.org/abs/2605.18490)**
  — the nearest neighbor. Preregistered RAG-vs-agent-wiki study measuring cost+latency+quality.
  Findings mirror this thesis: wiki wins cross-document connectivity + citation accuracy; RAG wins
  single-fact lookup, is ~10x cheaper, and answers in ~3 min vs the wiki's ~22 min; wiki never
  recoups its build cost. **BUT:** 24-paper corpus, 13 questions, no scaling, not narrative, and uses
  Karpathy's *pattern* — not `lucasastorian/llmwiki`.
- **[CorpusQA (arXiv 2601.14952)](https://arxiv.org/pdf/2601.14952)** & **[BEAM (arXiv 2510.27246)](https://arxiv.org/html/2510.27246v1)**
  — already chart the plain-LLM → RAG → structured/memory crossover across 128k→10M tokens. The
  three-tier narrative is **well-trodden**; do not claim it as a discovery.
- **[When to use Graphs in RAG (arXiv 2506.05690)](https://arxiv.org/pdf/2506.05690)** — RAG-variant
  benchmark over novels (56k→1.1M tokens), accuracy + token overhead (no dollars, no wiki method).

The unclaimed gap this project occupies (the intersection, none done together):
1. RAG-variants **+** agent-built-wiki **+** a corpus-size **scaling sweep** **+** cost+latency+accuracy
   **+** a **narrative corpus** — the union is open.
2. **No published benchmark of `lucasastorian/llmwiki` specifically** (its README has zero benchmarks).
3. **No interactive showcase** — findings live only in arXiv PDFs, never a playable two-panel/slider demo.
4. **Ingest dollar-cost** is barely measured anywhere (prior work counts query tokens, not build $).

Defensible positioning one-liner:
> "Cochran showed this on 24 academic papers, 13 questions, no scaling. I extend it to a scaling
> sweep over narrative fiction anyone can fact-check, using the actual LLMWiki tool, packaged as an
> interactive demo."

## Phasing
- **Phase 1 (MVP — this brief):** three arms only — **LLMWiki vs vector RAG vs plain long-context** —
  over famous novels + movie transcripts, scaling sweep, cost+latency+accuracy, pre-baked-size slider.
  This alone occupies the unclaimed gap; keep scope here (Rule 2).
- **Phase 2 (future):** add the RAG-variant *zoo* as selectable arms — **GraphRAG** (the automated peer
  of the wiki; expensive-ingest, cap at a mid sweep size), a **strong-RAG** variant (Contextual
  Retrieval / hybrid+rerank; cheap, full sweep, also a fairness check against the "you used naive RAG"
  critique), and optionally RAPTOR / LightRAG / HippoRAG. Deferred because each Tier-3 contender
  doubles the expensive-ingest cost and none are needed to claim the Phase-1 gap.
