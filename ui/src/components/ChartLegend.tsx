/**
 * Shared legend: the three fixed system hues + the single entry that explains the
 * measured-vs-extrapolated encoding (solid+filled vs dashed+hollow). Identity is
 * never color-alone — each system is labeled.
 */

import { SYSTEM_HUE, SYSTEM_LABEL } from "@/lib/scale";
import { SYSTEM_ORDER } from "@/lib/types";

export function ChartLegend(): React.JSX.Element {
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm" data-testid="chart-legend">
      {SYSTEM_ORDER.map((system) => (
        <span key={system} className="inline-flex items-center gap-2" data-legend-system={system}>
          <span
            aria-hidden
            className="inline-block h-3 w-3 rounded-full"
            style={{ backgroundColor: SYSTEM_HUE[system] }}
          />
          <span className="text-slate-700 dark:text-slate-200">{SYSTEM_LABEL[system]}</span>
        </span>
      ))}
      <span className="inline-flex items-center gap-2 text-slate-500 dark:text-slate-400" data-legend-encoding>
        <svg width="52" height="14" aria-hidden className="shrink-0">
          <line x1="2" y1="7" x2="24" y2="7" stroke="currentColor" strokeWidth="2" />
          <circle cx="13" cy="7" r="4" fill="currentColor" />
          <line x1="30" y1="7" x2="50" y2="7" stroke="currentColor" strokeWidth="2" strokeDasharray="5 3" />
          <circle cx="40" cy="7" r="4" fill="var(--chart-surface)" stroke="currentColor" strokeWidth="1.5" />
        </svg>
        <span>solid + filled = measured, dashed + hollow = extrapolated</span>
      </span>
    </div>
  );
}
