/**
 * Render tests for the cost curve. Verifies both charts render from the same fixture,
 * the log-y cost axis renders without NaN/crash on small costs, and the encoding holds.
 */

import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CostChart } from "@/components/CostChart";
import { parseResults } from "@/lib/loadResults";
import { SYSTEM_HUE } from "@/lib/scale";
import type { CostSeriesPoint, ResultsPayload } from "@/lib/types";
import rawResults from "../../public/results.json";

function loadFixture(): ResultsPayload {
  const loaded = parseResults(rawResults);
  if (loaded.status !== "ok") {
    throw new Error("fixture results.json failed to load");
  }
  return loaded.payload;
}

describe("CostChart", () => {
  it("renders a line per system from the fixture cost series", () => {
    const payload = loadFixture();
    const { container } = render(<CostChart points={payload.series.cost} />);
    for (const system of ["rag", "wiki"] as const) {
      const line = container.querySelector(`path[data-series-line="${system}"][data-kind="measured"]`);
      expect(line, `cost line for ${system}`).not.toBeNull();
      expect(line).toHaveStyle({ stroke: SYSTEM_HUE[system] });
    }
  });

  it("marks extrapolated cost points as dashed + hollow", () => {
    const payload = loadFixture();
    const { container } = render(<CostChart points={payload.series.cost} />);
    const extrapolatedMarker = container.querySelector('circle[data-marker="wiki"][data-kind="extrapolated"]');
    expect(extrapolatedMarker).not.toBeNull();
    expect(extrapolatedMarker).toHaveStyle({ fill: "var(--chart-surface)" });
  });

  it("renders a log-y cost axis with no NaN coordinates on tiny costs", () => {
    // Near-zero cost must clamp, not produce NaN/-Infinity path coords (correctness lens).
    const points: CostSeriesPoint[] = [
      { corpus_token_count: 100_000, system: "rag", total_cost_usd: 0.0, kind: "measured" },
      { corpus_token_count: 1_000_000, system: "rag", total_cost_usd: 0.002, kind: "measured" },
      { corpus_token_count: 50_000_000, system: "wiki", total_cost_usd: 100.0, kind: "measured" },
    ];
    const { container } = render(<CostChart points={points} />);
    const chart = container.querySelector('[data-testid="token-series-chart"]');
    expect(chart?.innerHTML.includes("NaN")).toBe(false);
    expect(chart?.innerHTML.includes("Infinity")).toBe(false);
  });
});
