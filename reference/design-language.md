# Design Language — visualization UI

**Why this doc exists:** the UI is deliberately thin — its job is to render the crossover chart, the
cost curve, and a three-path answer view credibly enough that a skeptical hiring manager trusts the
figures. It is a **data-visualization page, not a product**, so this doc is a lean chart spec, not a
full design-system import.
**Update when:** the chart set changes or the blog's visual identity is locked.

**Before writing any chart code, load the `dataviz` skill.** It carries the palette formula,
mark specs, and light/dark rules. This doc pins only the project-specific choices.

## Character

Clean, technical, editorial — reads like a good engineering blog figure. Spacious, high
data-ink ratio, no chartjunk. The measured-vs-extrapolated distinction must be unmistakable.

## Palette (brand-neutral, self-contained — swap into the dataviz validator)

Three systems get three fixed, colorblind-safe hues, stable across every chart:

- **plain_llm** — `#2563eb` (blue)
- **rag** — `#0d9488` (teal)
- **wiki** — `#d97706` (amber)

Neutrals: ink `#0f172a` / muted `#64748b` / grid `#e2e8f0` (light); ink `#f1f5f9` / muted `#94a3b8` /
grid `#1e293b` (dark). Theme-aware per the Artifact/UI rules — style both light and dark.

## Encoding rules (load-bearing for credibility)

- **Measured = solid line + filled marker. Extrapolated = dashed line + hollow marker.** One legend
  entry explains it. This is the single most important visual decision — a projection must never read
  as a measurement (Rule 12, PRD decision #4 & #20).
- **Context wall:** where plain_llm drops out (~1M tokens), draw a faint vertical rule labeled
  "context window" and end the plain_llm series with an open dot — don't let it vanish silently.
- **X axis is log-scale token count** (100k → 1B); label ticks in human units (100k, 1M, 50M, 1B).
- **Two headline charts:** (1) accuracy crossover (accuracy vs tokens, one line per system);
  (2) cost curve (total_cost_usd vs tokens, log-y — the wiki's ~50–100× gap is the visual).

## Typography & layout

- System UI / `Inter` stack; tabular numerals for axis + cost labels. Generous whitespace.
- One column, max-width ~880px, charts full-width within it. Slider + question picker sit above a
  three-column answer panel (plain_llm / rag / wiki), each cell showing the answer + its 0/0.5/1 score.

## Voice

Neutral and precise. Caption every figure with what's measured vs extrapolated and the corpus
composition. Let the numbers carry it — no hype.
