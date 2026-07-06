/**
 * Data-loading module for the pre-baked `results.json`.
 *
 * The UI has NO backend: `results.json` is a bundled static asset. This module
 * normalizes an unknown raw import into a typed `ResultsPayload` or a clear "empty"
 * result, so a missing / malformed / empty file shows an empty state instead of
 * crashing. Issue #12 reuses this loader to read the `answers` section.
 */

import { logger } from "@/lib/logger";
import type { ResultsPayload } from "@/lib/types";

/** A discriminated result: either usable data, or an empty state with a reason. */
export type LoadedResults =
  | { status: "ok"; payload: ResultsPayload }
  | { status: "empty"; reason: string };

function hasSeries(value: unknown): value is ResultsPayload {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as { series?: { accuracy?: unknown; cost?: unknown } };
  return (
    typeof candidate.series === "object" &&
    candidate.series !== null &&
    Array.isArray(candidate.series.accuracy) &&
    Array.isArray(candidate.series.cost)
  );
}

/**
 * Parse an unknown raw `results.json` import into a typed payload or an empty state.
 *
 * @param raw - The raw imported JSON (or null/undefined if the file is missing).
 * @returns A discriminated `LoadedResults`; never throws.
 *
 * @example
 * const loaded = parseResults(rawResultsJson);
 * if (loaded.status === "ok") renderCharts(loaded.payload);
 */
export function parseResults(raw: unknown): LoadedResults {
  if (!hasSeries(raw)) {
    logger.error("results_parse_empty", {
      reason: "missing_or_malformed_series",
      fix_suggestion: "Run scripts/generate_sample_results.py (or the sweep + aggregate.py) to emit ui/public/results.json",
    });
    return { status: "empty", reason: "No results file found. Run the sweep to generate results.json." };
  }
  const payload = raw as ResultsPayload;
  if (payload.series.accuracy.length === 0 && payload.series.cost.length === 0) {
    logger.error("results_parse_empty", { reason: "no_series_points", fix_suggestion: "Sweep produced no gradable rows." });
    return { status: "empty", reason: "Results file has no data points yet." };
  }
  logger.info("results_loaded", {
    accuracy_points: payload.series.accuracy.length,
    cost_points: payload.series.cost.length,
  });
  return { status: "ok", payload };
}
