/**
 * Page-level seam test: ONE `results.json` load must drive BOTH the charts (issue #11)
 * and the new answer explorer (issue #12) on the same static page — they share the
 * loader, types, and fixture, so this guards that neither half regresses the other.
 */

import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import HomePage from "@/app/page";

describe("HomePage", () => {
  it("renders both charts and the answer explorer from a single results.json load", () => {
    const { container, getByTestId } = render(<HomePage />);

    // Issue #11 charts.
    expect(getByTestId("accuracy-chart")).toBeInTheDocument();
    expect(container.querySelector('[data-testid="empty-state"]')).toBeNull();

    // Issue #12 answer explorer, wired below the charts from the same payload.
    expect(getByTestId("answer-explorer")).toBeInTheDocument();
    expect(container.querySelectorAll('[data-testid="answer-column"]').length).toBe(3);
  });
});
