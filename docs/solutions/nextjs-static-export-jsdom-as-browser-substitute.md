# Next.js static-export + jsdom render tests as a browser-verification substitute

**When this applies:** building a UI slice in a headless sandbox where browser
binaries (Playwright/Chromium, puppeteer) can't be installed or run — but the slice
protocol asks for a browser walkthrough. Also the reusable recipe for the
measured-vs-extrapolated chart encoding.

## Problem

The B8.5 browser walkthrough needs a real browser. In this sandbox `browser-use`,
puppeteer, and Playwright Chromium are all unavailable (no binary, no display). Faking
a browser result violates Rule 12; parking the slice for a missing binary is wrong when
the real acceptance criteria (build + render) actually pass.

## Solution — two honest verification seams, no browser

1. **`next build` with `output: 'export'`** is the load-bearing, browser-free
   acceptance check. It statically prerenders the page (`out/index.html`) — grep the
   emitted HTML for the load-bearing marks (chart testids, the context-wall label, the
   legend text) to prove the charts render from data at build time.
2. **jsdom render tests (Vitest + React Testing Library)** are the durable regression
   lock. Assert the *data-derived* marks, not pixels: query SVG by `data-*` attributes
   (`data-series-line`, `data-marker`, `data-context-wall`, `data-open-end-dot`,
   `data-x-tick`) and check `stroke-dasharray` / inline `style.fill`. These encode WHY
   (Rule 9): a projection must never read as a measurement.

Record the browser gap in `docs/residual-review-findings/<sha>.md`; don't park for it.

## Gotchas that cost time

- **React 19 removed the global `JSX` namespace.** `function C(): JSX.Element` fails
  `next build` type-check. Use `React.JSX.Element` (the `React` namespace is globally
  available via `@types/react`, no import needed).
- **Pick a patched Next version.** `next@15.1.6` carries CVE-2025-66478; the `backport`
  dist-tag (e.g. `15.5.20`) is the patched 15.x line.
- **Testable SVG marks:** drive fills/strokes via inline `style` (not presentation
  attributes) so `var(--chart-surface)` resolves in-browser *and* is assertable in
  jsdom (`el.style.fill === 'var(--chart-surface)'`). Measured = filled hue; extrapolated
  = hollow (surface fill + hue stroke); dashed line via `strokeDasharray`.
- **Deterministic fixtures:** generate `results.json` by running the real Python
  aggregate models (`build_results_json(rows=...)`), not by hand — the fixture then
  *is* the contract, and a contract-lock test catches Python-side drift.
