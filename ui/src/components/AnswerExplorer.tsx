"use client";

/**
 * Interactive three-path answer explorer (issue #12). Owns the two selections — the
 * question and the corpus-size index — and wires them into the pre-baked `answers`
 * block. No live calls: it only reads the static payload, and the slider snaps to
 * pre-baked sizes only. An empty `answers` block shows a clear note, never a crash.
 */

import { useState } from "react";

import { AnswerPanel } from "@/components/AnswerPanel";
import { QuestionPicker } from "@/components/QuestionPicker";
import { SizeSlider } from "@/components/SizeSlider";
import { sizesForQuestion } from "@/lib/answerView";
import type { AnswersBlock } from "@/lib/types";

export interface AnswerExplorerProps {
  answers: AnswersBlock;
}

export function AnswerExplorer({ answers }: AnswerExplorerProps): React.JSX.Element {
  const questionIds = Object.keys(answers);
  const [selectedQuestionId, setSelectedQuestionId] = useState<string>(questionIds[0] ?? "");
  const [selectedSizeIndex, setSelectedSizeIndex] = useState<number>(0);

  if (questionIds.length === 0) {
    return (
      <section className="flex flex-col gap-4" data-testid="answer-explorer">
        <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">Compare the three answers</h2>
        <p className="text-sm text-slate-500 dark:text-slate-400" data-testid="answer-explorer-empty">
          No answers recorded yet — run the sweep to populate the answer view.
        </p>
      </section>
    );
  }

  const sizes = sizesForQuestion(answers, selectedQuestionId);
  // A different question may expose different sizes; clamp so the index stays valid.
  const sizeIndex = Math.min(selectedSizeIndex, Math.max(sizes.length - 1, 0));
  const selectedSize = sizes[sizeIndex];

  function handleSelectQuestion(questionId: string): void {
    setSelectedQuestionId(questionId);
    setSelectedSizeIndex(0);
  }

  return (
    <section className="flex flex-col gap-6" data-testid="answer-explorer">
      <div className="flex flex-col gap-1">
        <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">Compare the three answers</h2>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Slide the corpus size and pick a question to see each path&apos;s answer and its judge score.
        </p>
      </div>

      <div className="flex flex-col gap-5 md:flex-row md:items-start md:gap-8">
        <div className="md:w-56">
          <QuestionPicker
            questionIds={questionIds}
            selectedQuestionId={selectedQuestionId}
            onSelect={handleSelectQuestion}
          />
        </div>
        <div className="flex-1">
          <SizeSlider sizes={sizes} selectedIndex={sizeIndex} onSelectIndex={setSelectedSizeIndex} />
        </div>
      </div>

      <AnswerPanel answers={answers} questionId={selectedQuestionId} size={selectedSize} />
    </section>
  );
}
