/**
 * Unit tests for the pure answer-view resolvers. These encode WHY (Rule 9): a sparse
 * `answers` block must never surface a blank/undefined column — a walled path reads
 * "exceeded context window", an un-run path reads a plain no-data note, and a present
 * cell surfaces its exact answer + score. Derived from the same `kind` the charts use.
 */

import { describe, expect, it } from "vitest";

import {
  DROPOUT_CONTEXT_MESSAGE,
  DROPOUT_NO_DATA_MESSAGE,
  formatScore,
  resolveColumn,
  sizesForQuestion,
} from "@/lib/answerView";
import type { AnswersBlock } from "@/lib/types";

const answers: AnswersBlock = {
  q1: {
    "100000": {
      plain_llm: { answer_text: "p-100k", accuracy: 0.9, kind: "measured" },
      rag: { answer_text: "r-100k", accuracy: 0.5, kind: "measured" },
      wiki: { answer_text: "w-100k", accuracy: 1.0, kind: "measured" },
    },
    "2000000": {
      plain_llm: { answer_text: "", accuracy: null, kind: "skipped_context" },
    },
    "5000000": {
      rag: { answer_text: "r-5M", accuracy: 0.62, kind: "measured" },
      wiki: { answer_text: "w-5M", accuracy: 0.82, kind: "extrapolated" },
    },
  },
};

describe("sizesForQuestion", () => {
  it("returns the pre-baked sizes ascending (the only selectable sizes)", () => {
    expect(sizesForQuestion(answers, "q1")).toEqual([100000, 2000000, 5000000]);
  });

  it("returns [] for an unknown question rather than throwing", () => {
    expect(sizesForQuestion(answers, "nope")).toEqual([]);
  });
});

describe("resolveColumn", () => {
  it("surfaces a present cell's answer + score", () => {
    const view = resolveColumn(answers, "q1", 100000, "plain_llm");
    expect(view.state).toBe("answer");
    if (view.state === "answer") {
      expect(view.cell.answer_text).toBe("p-100k");
      expect(view.cell.accuracy).toBe(0.9);
    }
  });

  it("reads a skipped_context cell as 'exceeded context window', not a blank", () => {
    const view = resolveColumn(answers, "q1", 2000000, "plain_llm");
    expect(view).toEqual({ state: "dropout", message: DROPOUT_CONTEXT_MESSAGE });
  });

  it("reads an ABSENT cell past the wall as 'exceeded context window'", () => {
    // plain_llm has no cell at 5M, but it is walled at 2M -> context dropout, not no-data.
    const view = resolveColumn(answers, "q1", 5000000, "plain_llm");
    expect(view).toEqual({ state: "dropout", message: DROPOUT_CONTEXT_MESSAGE });
  });

  it("reads an absent, non-walled cell as a plain no-data note", () => {
    // rag was not run at 2M and never walled -> honest no-data, still not a blank.
    const view = resolveColumn(answers, "q1", 2000000, "rag");
    expect(view).toEqual({ state: "dropout", message: DROPOUT_NO_DATA_MESSAGE });
  });

  it("preserves the extrapolated kind on a present cell", () => {
    const view = resolveColumn(answers, "q1", 5000000, "wiki");
    expect(view.state).toBe("answer");
    if (view.state === "answer") {
      expect(view.cell.kind).toBe("extrapolated");
    }
  });
});

describe("formatScore", () => {
  it("trims trailing zeros and returns null for a null score", () => {
    expect(formatScore(1.0)).toBe("1");
    expect(formatScore(0.5)).toBe("0.5");
    expect(formatScore(0.62)).toBe("0.62");
    expect(formatScore(null)).toBeNull();
  });
});
