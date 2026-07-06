# Integrations

**Why this doc exists:** the engine depends on five external services, each with an auth pattern and a
rate-limit / sourcing gotcha that will otherwise be rediscovered mid-slice.
**Update when:** a provider's auth, model id, pricing, or scraping structure changes.

## Anthropic (Claude) — baseline, RAG generation, judge

- **Key:** `ANTHROPIC_API_KEY` (settings: `anthropic_api_key`).
- **Plain-LLM baseline & RAG generation:** `claude-sonnet-5`. Baseline needs the **1M-token context
  beta** — set the `context-1m` beta header (verify exact header via the `claude-api` skill before
  wiring). The 1M ceiling is the crossover wall; past it the baseline emits a skipped row.
- **Judge:** `claude-opus-4-8`, structured output → `JudgeVerdict` (0 / 0.5 / 1 + rationale).
- **Cost:** capture input/output token cost per call for the API-equivalent accounting.

## Voyage — RAG embeddings

- **Key:** `VOYAGE_API_KEY` (settings: `voyage_api_key`). Model `voyage-3`, **1024-dim** (must match
  the Pinecone index dimension). ~$30 one-time to embed the affordable sweep. Batch + backoff on rate
  limits (`voyageai` SDK).

## Pinecone — RAG vector store

- **Key:** `PINECONE_API_KEY`. Serverless, `aws` / `us-east-1`, index `wiki-vs-rag`, dim **1024**.
- **Free Starter tier** covers the affordable sweep (~2 GB / ~300K vectors ≈ ~150M tokens; 2M write /
  1M read units/mo). Past ~100–150M tokens the $50/mo Standard plan applies — **log when the ceiling
  is hit**, don't silently degrade.
- **Key the index by corpus size** (namespace per sweep size) so sizes don't cross-contaminate.

## LLMWiki — wiki path (via Claude Code CLI, Max plan)

- Driven through `claude -p` with an MCP config from `./llmwiki mcp-config <dir>` (settings:
  `llmwiki_mcp_config_path`, `claude_cli_binary`). Runs on the author's **Max $200/mo plan** — marginal
  **$0**, constrained by **rate limits, not dollars**.
- **Must still capture `total_cost_usd`** (API-equivalent) from the CLI output — this is the reported
  cost gap, the thesis.
- **Ingest is resumable:** checkpoint progress so a usage-window / rate-limit cutoff resumes rather
  than restarts. Schedule loops for off-hours.
- **Measured only to a ~2–5M-token cap**, then extrapolated (PRD decision #4).

## subslikescript — movie-script corpus source

- Homepage is a **JS-rendered SPA** (a plain fetch returns only "Loading…"). But **detail pages**
  `https://subslikescript.com/movie/<Title>-<id>` carry **server-rendered HTML**: transcript in
  `div.full-script` inside `article.main-article`; title in `h1`.
- **Listing:** `/movies?page=N` (~30/page, ~1,234 pages). **Approach:** try `requests`+`bs4` on detail
  pages first; fall back to **Playwright** if blocked; fall back to the **Kaggle "Movie Transcripts
  59K" dump** (`fayaznoor10/movie-transcripts-59k`) for a zero-scrape path.
- **Legal note:** scripts are copyrighted — used here only as **private benchmark input**, never
  redistributed. State this in the blog's methodology.

## Project Gutenberg — novel corpus source

- Fixed-order plain-text URLs already wired in `corpus.GUTENBERG_BOOKS`. Strip PG boilerplate between
  `*** START OF` / `*** END OF` markers (already implemented). No key, no rate concern at this volume;
  be polite (cache locally, don't re-fetch).
