/**
 * Render tests for the accuracy crossover chart. These are the honest seam tests:
 * given the real fixture `results.json`, the chart must render one line per system
 * with the data-derived measured/extrapolated marks, the context wall, and the hues.
 * They encode WHY (Rule 9): a projection must never read as a measurement, and a
 * walled series must not silently vanish.
 */

import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AccuracyChart } from "@/components/AccuracyChart";
import { parseResults } from "@/lib/loadResults";
import { SYSTEM_HUE } from "@/lib/scale";
import type { ResultsPayload } from "@/lib/types";
import rawResults from "../../public/results.json";

function loadFixture(): ResultsPayload {
  const loaded = parseResults(rawResults);
  if (loaded.status !== "ok") {
    throw new Error("fixture results.json failed to load");
  }
  return loaded.payload;
}

describe("AccuracyChart", () => {
  it("renders one line per system (three fixed hues) from the fixture data", () => {
    const payload = loadFixture();
    const { container } = render(<AccuracyChart points={payload.series.accuracy} />);

    for (const system of ["plain_llm", "rag", "wiki"] as const) {
      // Every system that has >=2 measured points gets a solid measured line.
      const measuredLine = container.querySelector(`path[data-series-line="${system}"][data-kind="measured"]`);
      expect(measuredLine, `measured line for ${system}`).not.toBeNull();
      expect(measuredLine).toHaveStyle({ stroke: SYSTEM_HUE[system] });
    }
  });

  it("encodes measured as solid+filled and extrapolated as dashed+hollow", () => {
    const payload = loadFixture();
    const { container } = render(<AccuracyChart points={payload.series.accuracy} />);

    // Extrapolated line is dashed; measured line is not.
    const extrapolatedLine = container.querySelector('path[data-series-line="wiki"][data-kind="extrapolated"]');
    expect(extrapolatedLine).not.toBeNull();
    expect(extrapolatedLine?.getAttribute("stroke-dasharray")).toBe("6 4");
    const measuredLine = container.querySelector('path[data-series-line="wiki"][data-kind="measured"]');
    expect(measuredLine?.getAttribute("stroke-dasharray")).toBeNull();

    // Measured marker is filled with the hue; extrapolated marker is hollow (surface fill).
    const measuredMarker = container.querySelector('circle[data-marker="wiki"][data-kind="measured"]');
    const extrapolatedMarker = container.querySelector('circle[data-marker="wiki"][data-kind="extrapolated"]');
    expect(measuredMarker).toHaveStyle({ fill: SYSTEM_HUE.wiki });
    expect(extrapolatedMarker).toHaveStyle({ fill: "var(--chart-surface)" });
  });

  it("shows one legend entry explaining the measured-vs-extrapolated encoding", () => {
    const payload = loadFixture();
    const { getByText, container } = render(<AccuracyChart points={payload.series.accuracy} />);
    expect(getByText(/solid \+ filled = measured, dashed \+ hollow = extrapolated/i)).toBeInTheDocument();
    expect(container.querySelector("[data-legend-encoding]")).not.toBeNull();
  });

  it("draws the context wall + an open end-dot where plain_llm drops out — it does not vanish", () => {
    const payload = loadFixture();
    const { container, getByText } = render(<AccuracyChart points={payload.series.accuracy} />);

    // Plain-LLM is still present up to the wall (has measured markers), not silently gone.
    const plainMarkers = container.querySelectorAll('circle[data-marker="plain_llm"]');
    expect(plainMarkers.length).toBeGreaterThan(0);

    // The wall rule + open end-dot exist and are labeled.
    expect(container.querySelector('[data-context-wall="plain_llm"]')).not.toBeNull();
    expect(container.querySelector('circle[data-open-end-dot="plain_llm"]')).not.toBeNull();
    expect(getByText("context window")).toBeInTheDocument();
    expect(getByText("exceeded context window")).toBeInTheDocument();
  });

  it("labels the x axis with human-unit log ticks (100k, 1M, 50M, 1B)", () => {
    const payload = loadFixture();
    const { getByText, container } = render(<AccuracyChart points={payload.series.accuracy} />);
    for (const label of ["100k", "1M", "50M", "1B"]) {
      expect(getByText(label)).toBeInTheDocument();
    }
    expect(container.querySelectorAll("[data-x-tick]").length).toBe(4);
  });
});
