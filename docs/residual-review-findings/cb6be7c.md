# Residual review findings — issue #11 (crossover + cost charts)

Head at review: `cb6be7c` · slice: `ui/` static-export viewer.

Every concrete, evidenced defect surfaced by the review panel was fixed in the slice
commit. The items below are advisory / environment-bound and were **deferred by
design**, not skipped silently (Rule 12).

## Deferred (advisory)

1. **Shallow runtime validation of `results.json`** — `parseResults` guards the
   top-level shape (`series.accuracy` / `series.cost` are arrays) and degrades to an
   empty state, but does not deep-validate every point's fields. Accepted tradeoff:
   the file is pre-baked by our own `aggregate.py` (the contract is locked by
   `loadResults.test.ts`), and a thin viewer adding a full runtime schema validator
   (e.g. zod) would be over-engineering for a self-produced asset (Rule 2). Revisit
   only if the JSON ever comes from an untrusted producer.

## Environment-bound (could not exercise in sandbox)

2. **Browser walkthrough (B8.5) not run** — `browser-use` / puppeteer / Playwright
   Chromium are unavailable in this sandbox (no browser binary, no display). Verified
   instead via `npm run build` (successful static export; both charts + the context
   wall + the encoding legend appear in `out/index.html`) and 13 jsdom render tests
   (React Testing Library) that assert the data-derived marks. The committed jsdom
   tests are the durable regression lock. Re-run the browser walkthrough on a host
   with a Chromium binary when available.
