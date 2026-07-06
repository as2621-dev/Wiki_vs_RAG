/**
 * Cost curve: total API-equivalent cost (USD, log y) vs corpus tokens (log x),
 * one line per system — the wiki's ~50-100x cost gap is the visual. Renders through
 * the shared `TokenSeriesChart`.
 */

import type { ChartPoint } from "@/components/TokenSeriesChart";
import { TokenSeriesChart } from "@/components/TokenSeriesChart";
import { ChartLegend } from "@/components/ChartLegend";
import { formatCost, logCostTicks, makeLogScale } from "@/lib/scale";
import type { CostSeriesPoint } from "@/lib/types";

const PLOT_TOP = 28;
const PLOT_BOTTOM = 372;

export interface CostChartProps {
  points: CostSeriesPoint[];
}

export function CostChart({ points }: CostChartProps): React.JSX.Element {
  const costs = points.map((point) => point.total_cost_usd).filter((cost) => cost > 0);
  const domainMin = costs.length > 0 ? Math.min(...costs) : 0.001;
  const domainMax = costs.length > 0 ? Math.max(...costs) : 1;
  // Log-y so a near-zero RAG cost and a $100 wiki cost coexist without one flattening the other.
  const yScale = makeLogScale(domainMin, domainMax, PLOT_BOTTOM, PLOT_TOP);
  const yTicks = logCostTicks(domainMin, domainMax).map((value) => ({ value, label: formatCost(value) }));

  const chartPoints: ChartPoint[] = points.map((point) => ({
    system: point.system,
    tokenCount: point.corpus_token_count,
    value: point.total_cost_usd,
    kind: point.kind,
  }));

  return (
    <figure className="flex flex-col gap-3" data-testid="cost-chart">
      <figcaption className="text-base font-semibold text-slate-900 dark:text-slate-100">
        Total cost vs corpus size (log scale)
      </figcaption>
      <TokenSeriesChart
        points={chartPoints}
        yScale={yScale}
        yTicks={yTicks}
        yAxisLabel="Total cost (USD)"
        formatValue={formatCost}
      />
      <ChartLegend />
    </figure>
  );
}
