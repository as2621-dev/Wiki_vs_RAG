/**
 * Pure, framework-free helpers for the three-path answer view (issue #12).
 *
 * The `answers` block from `results.json` is intentionally SPARSE: a
 * `(question, size, system)` cell exists only if a sweep row actually ran for it
 * (see `aggregate._build_answers`). So a system can be absent at a size — past the
 * plain-LLM context wall, for example. These helpers resolve each column into an
 * explicit view (a real answer or a labeled dropout) so the UI never renders a blank
 * or `undefined` cell. Kept side-effect-free so they are trivially unit-testable.
 */

import type { AnswerCell, AnswersBlock, SystemName } from "@/lib/types";

/** Shown when a path has no usable answer at a size past its context wall. */
export const DROPOUT_CONTEXT_MESSAGE = "exceeded context window";
/** Shown when a path simply was not run at a size (no row, and not walled). */
export const DROPOUT_NO_DATA_MESSAGE = "no answer recorded at this size";

/** A resolved column: either a real answer cell, or a labeled dropout. */
export type ColumnView =
  | { state: "answer"; cell: AnswerCell }
  | { state: "dropout"; message: string };

/**
 * The pre-baked corpus sizes available for a question, ascending. These are the ONLY
 * sizes the slider may select — it snaps to real data, never an arbitrary value.
 *
 * @example
 * sizesForQuestion(answers, "q_lookup_01"); // [100000, 500000, 1000000, ...]
 */
export function sizesForQuestion(answers: AnswersBlock, questionId: string): number[] {
  const bySize = answers[questionId];
  if (!bySize) {
    return [];
  }
  return Object.keys(bySize)
    .map((size) => Number(size))
    .filter((size) => Number.isFinite(size))
    .sort((a, b) => a - b);
}

/** True if `system` hit its context wall at or below `size` (a `skipped_context` cell). */
function isWalledAtOrBelow(answers: AnswersBlock, questionId: string, size: number, system: SystemName): boolean {
  const bySize = answers[questionId];
  if (!bySize) {
    return false;
  }
  for (const [sizeKey, systems] of Object.entries(bySize)) {
    const cell = systems[system];
    if (cell && cell.kind === "skipped_context" && Number(sizeKey) <= size) {
      return true;
    }
  }
  return false;
}

/**
 * Resolve one (question, size, system) into a displayable column view.
 *
 * A present, gradable cell renders its answer. A `skipped_context` cell, or an absent
 * cell for a system that is walled at/below this size, renders "exceeded context window".
 * Any other absent cell renders a plain "no answer recorded" — never a blank.
 *
 * @example
 * resolveColumn(answers, "q_lookup_01", 5000000, "plain_llm");
 * // { state: "dropout", message: "exceeded context window" }
 */
export function resolveColumn(
  answers: AnswersBlock,
  questionId: string,
  size: number,
  system: SystemName,
): ColumnView {
  const cell = answers[questionId]?.[String(size)]?.[system];
  if (cell && cell.kind !== "skipped_context" && cell.accuracy !== null) {
    return { state: "answer", cell };
  }
  if (cell || isWalledAtOrBelow(answers, questionId, size, system)) {
    return { state: "dropout", message: DROPOUT_CONTEXT_MESSAGE };
  }
  return { state: "dropout", message: DROPOUT_NO_DATA_MESSAGE };
}

/**
 * Format a 0..1 judge score for a badge: `1`, `0.5`, `0.9`, `0.62` — trailing zeros
 * trimmed. Returns `null` for a null accuracy so the caller renders a dropout instead.
 */
export function formatScore(accuracy: number | null): string | null {
  if (accuracy === null) {
    return null;
  }
  return String(Number(accuracy.toFixed(2)));
}
