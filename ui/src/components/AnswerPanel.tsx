/**
 * The three-path answer panel: one column per system (plain_llm / rag / wiki) at the
 * selected (question, size). Each column shows the path's answer text + its 0/0.5/1
 * judge score, and — reusing the charts' encoding — a measured (filled) vs extrapolated
 * (hollow) marker so a projection never reads as a measurement. A dropped-out path shows
 * "exceeded context window" instead of a blank cell.
 */

import { formatScore, resolveColumn } from "@/lib/answerView";
import { SYSTEM_HUE, SYSTEM_LABEL } from "@/lib/scale";
import type { AnswersBlock, SeriesPointKind, SystemName } from "@/lib/types";
import { SYSTEM_ORDER } from "@/lib/types";

export interface AnswerPanelProps {
  answers: AnswersBlock;
  questionId: string;
  size: number;
}

/** Measured = filled dot; extrapolated = hollow dot — mirrors the chart legend encoding. */
function KindBadge({ system, kind }: { system: SystemName; kind: SeriesPointKind }): React.JSX.Element {
  const isExtrapolated = kind === "extrapolated";
  const hue = SYSTEM_HUE[system];
  return (
    <span
      className="inline-flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400"
      data-testid="kind-badge"
      data-kind={kind}
    >
      <span
        aria-hidden
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{
          backgroundColor: isExtrapolated ? "var(--chart-surface)" : hue,
          border: isExtrapolated ? `1.5px solid ${hue}` : "none",
        }}
      />
      {isExtrapolated ? "extrapolated" : "measured"}
    </span>
  );
}

function AnswerColumn({
  answers,
  questionId,
  size,
  system,
}: {
  answers: AnswersBlock;
  questionId: string;
  size: number;
  system: SystemName;
}): React.JSX.Element {
  const view = resolveColumn(answers, questionId, size, system);
  const hue = SYSTEM_HUE[system];

  return (
    <div
      className="flex flex-1 flex-col gap-3 rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900"
      data-testid="answer-column"
      data-system={system}
    >
      <div className="flex items-center justify-between">
        <span className="inline-flex items-center gap-2 font-semibold text-slate-900 dark:text-slate-100">
          <span aria-hidden className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: hue }} />
          {SYSTEM_LABEL[system]}
        </span>
        {view.state === "answer" && <KindBadge system={system} kind={view.cell.kind} />}
      </div>

      {view.state === "answer" ? (
        <>
          <span
            className="inline-flex w-fit items-center rounded-full bg-slate-100 px-2.5 py-0.5 text-sm font-semibold tabular-nums text-slate-700 dark:bg-slate-800 dark:text-slate-200"
            data-testid="score-badge"
          >
            score {formatScore(view.cell.accuracy)} / 1
          </span>
          <p className="text-sm leading-relaxed text-slate-700 dark:text-slate-300" data-testid="answer-text">
            {view.cell.answer_text}
          </p>
        </>
      ) : (
        <p
          className="text-sm font-medium italic text-slate-500 dark:text-slate-400"
          data-testid="answer-dropout"
        >
          {view.message}
        </p>
      )}
    </div>
  );
}

export function AnswerPanel({ answers, questionId, size }: AnswerPanelProps): React.JSX.Element {
  return (
    <div className="flex flex-col gap-4 md:flex-row" data-testid="answer-panel">
      {SYSTEM_ORDER.map((system) => (
        <AnswerColumn key={system} answers={answers} questionId={questionId} size={size} system={system} />
      ))}
    </div>
  );
}
