/**
 * Pure scale + tick helpers for the log-token x axis and log-cost y axis.
 * Kept framework-free and side-effect-free so they are trivially unit-testable.
 */

import type { SystemName } from "@/lib/types";

/** Fixed, colorblind-safe system hues (reference/design-language.md). */
export const SYSTEM_HUE: Record<SystemName, string> = {
  plain_llm: "#2563eb",
  rag: "#0d9488",
  wiki: "#d97706",
};

/** Human-readable system labels for legends. */
export const SYSTEM_LABEL: Record<SystemName, string> = {
  plain_llm: "Plain LLM",
  rag: "RAG",
  wiki: "Wiki",
};

/** The human-unit x-axis ticks required by the design language. */
export const X_AXIS_TICKS: readonly number[] = [100_000, 1_000_000, 50_000_000, 1_000_000_000];

/**
 * Render a token count as a compact human label (`1_000_000` -> `"1M"`).
 *
 * @example
 * humanizeTokens(100_000); // "100k"
 * humanizeTokens(1_000_000_000); // "1B"
 */
export function humanizeTokens(tokenCount: number): string {
  if (tokenCount >= 1_000_000_000) {
    return `${trimZero(tokenCount / 1_000_000_000)}B`;
  }
  if (tokenCount >= 1_000_000) {
    return `${trimZero(tokenCount / 1_000_000)}M`;
  }
  if (tokenCount >= 1_000) {
    return `${trimZero(tokenCount / 1_000)}k`;
  }
  return String(tokenCount);
}

function trimZero(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

/** Format a USD cost with tabular-friendly precision. */
export function formatCost(costUsd: number): string {
  if (costUsd >= 100) {
    return `$${Math.round(costUsd)}`;
  }
  if (costUsd >= 1) {
    return `$${costUsd.toFixed(1)}`;
  }
  return `$${costUsd.toFixed(3)}`;
}

/**
 * Build a log10 scale mapping a value domain to a pixel range.
 * Values <= 0 are clamped to `floor` so a zero/near-zero cost never yields NaN/-Infinity.
 */
export function makeLogScale(
  domainMin: number,
  domainMax: number,
  rangeMin: number,
  rangeMax: number,
  floor = 1e-6,
): (value: number) => number {
  const safeMin = Math.max(domainMin, floor);
  const safeMax = Math.max(domainMax, safeMin * 10);
  const logMin = Math.log10(safeMin);
  const logMax = Math.log10(safeMax);
  const span = logMax - logMin || 1;
  return (value: number): number => {
    const clamped = Math.max(value, floor);
    const fraction = (Math.log10(clamped) - logMin) / span;
    return rangeMin + fraction * (rangeMax - rangeMin);
  };
}

/** Build a linear scale mapping a value domain to a pixel range. */
export function makeLinearScale(
  domainMin: number,
  domainMax: number,
  rangeMin: number,
  rangeMax: number,
): (value: number) => number {
  const span = domainMax - domainMin || 1;
  return (value: number): number => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
}

/** "Nice" log-y ticks (powers of ten) spanning a cost domain, for the cost chart. */
export function logCostTicks(domainMin: number, domainMax: number): number[] {
  const floorExp = Math.floor(Math.log10(Math.max(domainMin, 1e-6)));
  const ceilExp = Math.ceil(Math.log10(Math.max(domainMax, 1e-5)));
  const ticks: number[] = [];
  for (let exp = floorExp; exp <= ceilExp; exp += 1) {
    ticks.push(10 ** exp);
  }
  return ticks;
}
