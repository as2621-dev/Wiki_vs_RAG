/**
 * Accuracy crossover chart: mean judge accuracy (0..1, linear y) vs corpus tokens
 * (log x), one line per system. Renders through the shared `TokenSeriesChart`.
 */

import type { ChartPoint } from "@/components/TokenSeriesChart";
import { TokenSeriesChart } from "@/components/TokenSeriesChart";
import { ChartLegend } from "@/components/ChartLegend";
import { makeLinearScale } from "@/lib/scale";
import type { AccuracySeriesPoint } from "@/lib/types";

const PLOT_TOP = 28;
const PLOT_BOTTOM = 372; // HEIGHT - bottom margin, mirrors TokenSeriesChart geometry.
const Y_TICKS = [0, 0.25, 0.5, 0.75, 1];

export interface AccuracyChartProps {
  points: AccuracySeriesPoint[];
}

export function AccuracyChart({ points }: AccuracyChartProps): React.JSX.Element {
  // Accuracy is a linear 0..1 axis; skipped_context points carry null and drive the wall.
  const yScale = makeLinearScale(0, 1, PLOT_BOTTOM, PLOT_TOP);
  const chartPoints: ChartPoint[] = points.map((point) => ({
    system: point.system,
    tokenCount: point.corpus_token_count,
    value: point.accuracy,
    kind: point.kind,
  }));

  return (
    <figure className="flex flex-col gap-3" data-testid="accuracy-chart">
      <figcaption className="text-base font-semibold text-slate-900 dark:text-slate-100">
        Accuracy vs corpus size
      </figcaption>
      <TokenSeriesChart
        points={chartPoints}
        yScale={yScale}
        yTicks={Y_TICKS.map((value) => ({ value, label: value.toFixed(2) }))}
        yAxisLabel="Accuracy"
        formatValue={(value) => value.toFixed(2)}
      />
      <ChartLegend />
    </figure>
  );
}
