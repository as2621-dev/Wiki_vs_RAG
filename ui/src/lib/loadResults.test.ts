/**
 * Contract-lock + empty-state tests for the data loader. The fixture must parse into
 * the typed model exactly (so #12 can build on the same shape), and any missing /
 * malformed / empty input must degrade to an empty state, never a crash.
 */

import { describe, expect, it } from "vitest";

import { parseResults } from "@/lib/loadResults";
import rawResults from "../../public/results.json";

describe("parseResults — contract lock", () => {
  it("parses the fixture results.json into the typed payload (both series present)", () => {
    const loaded = parseResults(rawResults);
    expect(loaded.status).toBe("ok");
    if (loaded.status !== "ok") return;
    expect(loaded.payload.series.accuracy.length).toBeGreaterThan(0);
    expect(loaded.payload.series.cost.length).toBeGreaterThan(0);
    expect(typeof loaded.payload.generated_note).toBe("string");
    expect(loaded.payload.answers).toBeDefined();
  });

  it("carries every SeriesPointKind through unchanged (measured/extrapolated/skipped_context)", () => {
    const loaded = parseResults(rawResults);
    if (loaded.status !== "ok") throw new Error("fixture failed to load");
    const kinds = new Set(loaded.payload.series.accuracy.map((point) => point.kind));
    expect(kinds.has("measured")).toBe(true);
    expect(kinds.has("extrapolated")).toBe(true);
    expect(kinds.has("skipped_context")).toBe(true);
    // The skipped_context point carries null accuracy, not a zero score.
    const skipped = loaded.payload.series.accuracy.find((point) => point.kind === "skipped_context");
    expect(skipped?.accuracy).toBeNull();
  });
});

describe("parseResults — empty state", () => {
  it("returns empty for a missing file (null)", () => {
    expect(parseResults(null).status).toBe("empty");
  });

  it("returns empty for a malformed payload (no series)", () => {
    expect(parseResults({ generated_note: "x" }).status).toBe("empty");
  });

  it("returns empty when both series are present but contain no points", () => {
    const loaded = parseResults({ generated_note: "x", series: { accuracy: [], cost: [] }, answers: {} });
    expect(loaded.status).toBe("empty");
  });
});
