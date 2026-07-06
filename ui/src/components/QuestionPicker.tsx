/**
 * Question picker for the answer view: a native `<select>` over the question ids
 * present in the pre-baked `answers` block. Native select keeps it accessible and
 * dependency-free (matches the chart components' zero-runtime-lib style).
 */

export interface QuestionPickerProps {
  questionIds: string[];
  selectedQuestionId: string;
  onSelect: (questionId: string) => void;
}

export function QuestionPicker({ questionIds, selectedQuestionId, onSelect }: QuestionPickerProps): React.JSX.Element {
  return (
    <label className="flex flex-col gap-1 text-sm" data-testid="question-picker">
      <span className="font-medium text-slate-700 dark:text-slate-200">Question</span>
      <select
        className="rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
        value={selectedQuestionId}
        onChange={(event) => onSelect(event.target.value)}
      >
        {questionIds.map((questionId) => (
          <option key={questionId} value={questionId}>
            {questionId}
          </option>
        ))}
      </select>
    </label>
  );
}
