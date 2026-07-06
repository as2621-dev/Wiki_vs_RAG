/**
 * Shared SVG chart primitive: value-vs-log(token-count), one line per system,
 * with the measured-vs-extrapolated encoding and the plain-LLM context wall.
 *
 * Self-contained SVG (no chart lib) so we fully control the load-bearing marks:
 * measured = solid line + filled marker, extrapolated = dashed line + hollow marker,
 * a faint labeled "context window" rule + an open end-dot where a series drops out.
 * Both headline charts (accuracy, cost) render through this one primitive.
 */

import type { SeriesPointKind, SystemName } from "@/lib/types";
import { SYSTEM_ORDER } from "@/lib/types";
import {
  humanizeTokens,
  makeLogScale,
  SYSTEM_HUE,
  SYSTEM_LABEL,
  X_AXIS_TICKS,
} from "@/lib/scale";

/** One system point normalized for plotting (value null == dropped out at the wall). */
export interface ChartPoint {
  system: SystemName;
  tokenCount: number;
  value: number | null;
  kind: SeriesPointKind;
}

export interface TokenSeriesChartProps {
  /** Normalized points across all systems. */
  points: ChartPoint[];
  /** Y-axis pixel scale (linear for accuracy, log for cost) — maps value -> y. */
  yScale: (value: number) => number;
  /** Y-axis tick values with formatted labels, top-to-bottom order handled by yScale. */
  yTicks: { value: number; label: string }[];
  /** Accessible axis titles. */
  yAxisLabel: string;
  /** Formatter for a point's value (used in marker tooltips/titles). */
  formatValue: (value: number) => string;
}

const WIDTH = 800;
const HEIGHT = 420;
const MARGIN = { top: 28, right: 28, bottom: 48, left: 64 };
const PLOT_LEFT = MARGIN.left;
const PLOT_RIGHT = WIDTH - MARGIN.right;
const PLOT_TOP = MARGIN.top;
const PLOT_BOTTOM = HEIGHT - MARGIN.bottom;

const X_DOMAIN_MIN = 100_000;
const X_DOMAIN_MAX = 1_000_000_000;

function polylinePath(pixels: { x: number; y: number }[]): string {
  return pixels.map((p, index) => `${index === 0 ? "M" : "L"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
}

/**
 * Render the value-vs-token chart. Pure of side effects; all data comes via props.
 */
export function TokenSeriesChart({ points, yScale, yTicks, yAxisLabel, formatValue }: TokenSeriesChartProps): React.JSX.Element {
  const xScale = makeLogScale(X_DOMAIN_MIN, X_DOMAIN_MAX, PLOT_LEFT, PLOT_RIGHT);

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label={`${yAxisLabel} versus corpus token count, one line per system`}
      className="w-full h-auto"
      data-testid="token-series-chart"
    >
      {/* Y grid + ticks */}
      {yTicks.map((tick) => {
        const y = yScale(tick.value);
        return (
          <g key={`y-${tick.value}`}>
            <line x1={PLOT_LEFT} x2={PLOT_RIGHT} y1={y} y2={y} className="chart-grid" />
            <text x={PLOT_LEFT - 10} y={y} textAnchor="end" dominantBaseline="middle" className="chart-tick-label">
              {tick.label}
            </text>
          </g>
        );
      })}

      {/* X ticks (log-scale, human units) */}
      {X_AXIS_TICKS.map((tick) => {
        const x = xScale(tick);
        return (
          <g key={`x-${tick}`} data-x-tick={tick}>
            <line x1={x} x2={x} y1={PLOT_TOP} y2={PLOT_BOTTOM} className="chart-grid" />
            <text x={x} y={PLOT_BOTTOM + 20} textAnchor="middle" className="chart-tick-label">
              {humanizeTokens(tick)}
            </text>
          </g>
        );
      })}

      {/* Axis titles */}
      <text
        x={(PLOT_LEFT + PLOT_RIGHT) / 2}
        y={HEIGHT - 8}
        textAnchor="middle"
        className="chart-axis-title"
      >
        corpus tokens (log scale)
      </text>

      {/* Per-system marks */}
      {SYSTEM_ORDER.map((system) => {
        const hue = SYSTEM_HUE[system];
        const systemPoints = points
          .filter((point) => point.system === system)
          .sort((a, b) => a.tokenCount - b.tokenCount);
        if (systemPoints.length === 0) {
          return null;
        }

        const plotted = systemPoints.filter((point): point is ChartPoint & { value: number } => point.value !== null);
        const measured = plotted.filter((point) => point.kind === "measured");
        const extrapolated = plotted.filter((point) => point.kind === "extrapolated");
        const wallPoints = systemPoints.filter((point) => point.kind === "skipped_context");

        const toPixel = (point: ChartPoint & { value: number }) => ({
          x: xScale(point.tokenCount),
          y: yScale(point.value),
        });

        const measuredPixels = measured.map(toPixel);
        const lastMeasured = measured[measured.length - 1];
        // Dashed segment bridges the last measured point into the extrapolated tail.
        const extraPixels = (lastMeasured ? [lastMeasured, ...extrapolated] : extrapolated).map(toPixel);

        return (
          <g key={system} data-system={system}>
            {/* Solid measured line */}
            {measuredPixels.length >= 2 && (
              <path
                d={polylinePath(measuredPixels)}
                fill="none"
                style={{ stroke: hue }}
                strokeWidth={2}
                data-series-line={system}
                data-kind="measured"
              />
            )}
            {/* Dashed extrapolated line */}
            {extraPixels.length >= 2 && (
              <path
                d={polylinePath(extraPixels)}
                fill="none"
                style={{ stroke: hue }}
                strokeWidth={2}
                strokeDasharray="6 4"
                data-series-line={system}
                data-kind="extrapolated"
              />
            )}
            {/* Markers: filled = measured, hollow = extrapolated */}
            {plotted.map((point) => {
              const { x, y } = toPixel(point);
              const isMeasured = point.kind === "measured";
              return (
                <circle
                  key={`${system}-${point.tokenCount}`}
                  cx={x}
                  cy={y}
                  r={5}
                  strokeWidth={2}
                  style={isMeasured ? { fill: hue, stroke: hue } : { fill: "var(--chart-surface)", stroke: hue }}
                  data-marker={system}
                  data-kind={point.kind}
                >
                  <title>{`${SYSTEM_LABEL[system]} · ${humanizeTokens(point.tokenCount)} · ${formatValue(point.value)} (${point.kind})`}</title>
                </circle>
              );
            })}
            {/* Context wall: faint vertical rule + open end-dot where the series drops out */}
            {wallPoints.map((wall) => {
              const wallX = xScale(wall.tokenCount);
              return (
                <g key={`wall-${system}-${wall.tokenCount}`} data-context-wall={system}>
                  <line
                    x1={wallX}
                    x2={wallX}
                    y1={PLOT_TOP}
                    y2={PLOT_BOTTOM}
                    className="chart-context-wall"
                    strokeDasharray="4 4"
                  />
                  <text x={wallX + 6} y={PLOT_TOP + 12} className="chart-wall-label">
                    context window
                  </text>
                  <text x={wallX + 6} y={PLOT_TOP + 28} className="chart-wall-sublabel">
                    exceeded context window
                  </text>
                  {lastMeasured && (
                    <circle
                      cx={xScale(lastMeasured.tokenCount)}
                      cy={yScale(lastMeasured.value)}
                      r={6}
                      strokeWidth={2}
                      style={{ fill: "var(--chart-surface)", stroke: hue }}
                      data-open-end-dot={system}
                    >
                      <title>{`${SYSTEM_LABEL[system]} drops out here — exceeded context window`}</title>
                    </circle>
                  )}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}
