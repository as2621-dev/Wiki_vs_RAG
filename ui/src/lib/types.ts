/**
 * TypeScript mirror of the `results.json` contract emitted by `aggregate.py`
 * (Pydantic models in `models.py`: `ResultsPayload` / `ResultsSeries` /
 * `AccuracySeriesPoint` / `CostSeriesPoint` / `AnswerCell` / `SeriesPointKind`).
 *
 * This file is the single source of truth for the seam between the Python engine
 * and the UI. Field names and nesting MUST match the Python side exactly — a drift
 * means the UI reads garbage. Issue #12 (size slider + answer view) reuses these
 * types, so they are exported for reuse.
 */

/** The three architectures under comparison — mirrors Python `SystemName`. */
export type SystemName = "plain_llm" | "rag" | "wiki";

/** Fixed render order for the three systems (stable hue assignment). */
export const SYSTEM_ORDER: readonly SystemName[] = ["plain_llm", "rag", "wiki"];

/**
 * How a series point was obtained — mirrors Python `SeriesPointKind`.
 * `measured` was run, `extrapolated` is a curve projection, `skipped_context`
 * marks the plain-LLM context wall (`accuracy: null`, not a zero score).
 */
export type SeriesPointKind = "measured" | "extrapolated" | "skipped_context";

/** One point on the accuracy-vs-tokens chart — mirrors Python `AccuracySeriesPoint`. */
export interface AccuracySeriesPoint {
  corpus_token_count: number;
  system: SystemName;
  /** Mean judge score 0..1, or null for a `skipped_context` point. */
  accuracy: number | null;
  kind: SeriesPointKind;
}

/** One point on the cost-vs-tokens chart — mirrors Python `CostSeriesPoint`. */
export interface CostSeriesPoint {
  corpus_token_count: number;
  system: SystemName;
  total_cost_usd: number;
  kind: SeriesPointKind;
}

/** The two chart-ready series the UI reads — mirrors Python `ResultsSeries`. */
export interface ResultsSeries {
  accuracy: AccuracySeriesPoint[];
  cost: CostSeriesPoint[];
}

/** One system's answer to one question at one size — mirrors Python `AnswerCell`. */
export interface AnswerCell {
  answer_text: string;
  accuracy: number | null;
  kind: SeriesPointKind;
}

/** answers[question_id][corpus_token_count][system] — mirrors Python `answers` nesting. */
export type AnswersBlock = Record<string, Record<string, Record<string, AnswerCell>>>;

/**
 * The full `results.json` payload. `generated_note`, `series` and `answers` are the
 * exact contract; `meta` and `summary` are additive metadata the UI may ignore.
 */
export interface ResultsPayload {
  generated_note: string;
  series: ResultsSeries;
  answers: AnswersBlock;
  meta?: unknown[];
  summary?: unknown;
}
