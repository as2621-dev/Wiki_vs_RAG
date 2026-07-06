/**
 * Clear empty-state panel shown when `results.json` is missing / malformed / empty —
 * the UI must never crash or render a blank when there is no data.
 */

export interface EmptyStateProps {
  reason: string;
}

export function EmptyState({ reason }: EmptyStateProps): React.JSX.Element {
  return (
    <div
      className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-10 text-center dark:border-slate-700 dark:bg-slate-900"
      data-testid="empty-state"
    >
      <p className="text-lg font-semibold text-slate-700 dark:text-slate-200">No results yet</p>
      <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">{reason}</p>
    </div>
  );
}
