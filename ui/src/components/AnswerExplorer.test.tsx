/**
 * Render + interaction tests for the answer explorer. These are the honest seam tests
 * (Rule 9): against the REAL fixture, moving the slider and picking a question must
 * drive the three-column panel to the exact answer_text + score in the data — a static
 * viewer with no live calls. They also lock the intent that a dropped-out path shows
 * "exceeded context window" (never a blank) and an empty block shows a note, not a crash.
 */

import { fireEvent, render, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AnswerExplorer } from "@/components/AnswerExplorer";
import { parseResults } from "@/lib/loadResults";
import type { AnswersBlock, ResultsPayload } from "@/lib/types";
import rawResults from "../../public/results.json";

function loadFixtureAnswers(): AnswersBlock {
  const loaded = parseResults(rawResults);
  if (loaded.status !== "ok") {
    throw new Error("fixture results.json failed to load");
  }
  return (loaded.payload as ResultsPayload).answers;
}

/** Read the answer/dropout text of one system column by its data-system marker. */
function columnText(container: HTMLElement, system: string): string {
  const column = container.querySelector(`[data-testid="answer-column"][data-system="${system}"]`);
  if (!column) {
    throw new Error(`no answer column for ${system}`);
  }
  return column.textContent ?? "";
}

describe("AnswerExplorer (real fixture)", () => {
  it("shows three columns with each path's answer_text + score at the smallest size", () => {
    const answers = loadFixtureAnswers();
    const { container } = render(<AnswerExplorer answers={answers} />);

    // Smallest pre-baked size (100k) has all three measured — the happy path.
    for (const system of ["plain_llm", "rag", "wiki"]) {
      const cell = answers.q_lookup_01["100000"][system];
      const text = columnText(container, system);
      expect(text).toContain(cell.answer_text);
      expect(text).toContain("score");
    }
    // The plain_llm score (0.9) is read from the data, not hardcoded downstream.
    expect(columnText(container, "plain_llm")).toContain("0.9");
  });

  it("updates all three answers when the corpus-size slider moves", () => {
    const answers = loadFixtureAnswers();
    const { container, getByRole } = render(<AnswerExplorer answers={answers} />);

    const slider = getByRole("slider");
    // Move to 50M (index 5 in [100k,500k,1M,2M,5M,50M]): rag/wiki answer, plain_llm past the wall.
    fireEvent.change(slider, { target: { value: "5" } });

    expect(columnText(container, "rag")).toContain(answers.q_lookup_01["50000000"].rag.answer_text);
    expect(columnText(container, "wiki")).toContain(answers.q_lookup_01["50000000"].wiki.answer_text);
    // plain_llm dropped out past the wall — shows the message, not a blank.
    expect(columnText(container, "plain_llm")).toContain("exceeded context window");
  });

  it("marks each present answer as measured vs extrapolated (story #23)", () => {
    const answers = loadFixtureAnswers();
    const { container } = render(<AnswerExplorer answers={answers} />);
    // Every present column at 100k is measured in the fixture — the mark is shown + data-driven.
    const badges = container.querySelectorAll('[data-testid="kind-badge"]');
    expect(badges.length).toBe(3);
    for (const badge of badges) {
      expect(badge.getAttribute("data-kind")).toBe("measured");
      expect(badge.textContent).toContain("measured");
    }
  });

  it("shows 'exceeded context window' for plain_llm at its skipped_context size (2M)", () => {
    const answers = loadFixtureAnswers();
    const { container, getByRole } = render(<AnswerExplorer answers={answers} />);
    fireEvent.change(getByRole("slider"), { target: { value: "3" } }); // index 3 = 2M
    const column = container.querySelector('[data-testid="answer-column"][data-system="plain_llm"]');
    expect(within(column as HTMLElement).getByTestId("answer-dropout").textContent).toBe(
      "exceeded context window",
    );
  });

  it("renders an empty note (not a crash) when the answers block is empty", () => {
    const { getByTestId } = render(<AnswerExplorer answers={{}} />);
    expect(getByTestId("answer-explorer-empty")).toBeInTheDocument();
  });
});

describe("AnswerExplorer (synthetic multi-question picker)", () => {
  const answers: AnswersBlock = {
    q_a: {
      "100000": {
        plain_llm: { answer_text: "A-plain", accuracy: 1.0, kind: "measured" },
        rag: { answer_text: "A-rag", accuracy: 0.5, kind: "measured" },
        wiki: { answer_text: "A-wiki", accuracy: 1.0, kind: "extrapolated" },
      },
    },
    q_b: {
      "100000": {
        plain_llm: { answer_text: "B-plain", accuracy: 0.0, kind: "measured" },
        rag: { answer_text: "B-rag", accuracy: 1.0, kind: "measured" },
        wiki: { answer_text: "B-wiki", accuracy: 0.5, kind: "measured" },
      },
    },
  };

  it("swaps all three answers when a different question is picked", () => {
    const { container, getByTestId } = render(<AnswerExplorer answers={answers} />);
    expect(columnText(container, "plain_llm")).toContain("A-plain");

    const select = within(getByTestId("question-picker")).getByRole("combobox");
    fireEvent.change(select, { target: { value: "q_b" } });

    expect(columnText(container, "plain_llm")).toContain("B-plain");
    expect(columnText(container, "rag")).toContain("B-rag");
    expect(columnText(container, "wiki")).toContain("B-wiki");
  });

  it("renders the extrapolated marker for an extrapolated answer cell", () => {
    const { container } = render(<AnswerExplorer answers={answers} />);
    const wikiBadge = container
      .querySelector('[data-testid="answer-column"][data-system="wiki"]')
      ?.querySelector('[data-testid="kind-badge"]');
    expect(wikiBadge?.getAttribute("data-kind")).toBe("extrapolated");
    expect(wikiBadge?.textContent).toContain("extrapolated");
  });
});
