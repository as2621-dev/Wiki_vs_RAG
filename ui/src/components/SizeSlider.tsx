/**
 * Corpus-size slider: a discrete range input over the pre-baked sizes. `min/max/step`
 * are indices into the `sizes` array, so the thumb SNAPS to real data only — an
 * arbitrary token count can never be selected (Out of Scope: live ingestion). The
 * current size is labeled in human units (100k, 1M, ...).
 */

import { humanizeTokens } from "@/lib/scale";

export interface SizeSliderProps {
  sizes: number[];
  selectedIndex: number;
  onSelectIndex: (index: number) => void;
}

export function SizeSlider({ sizes, selectedIndex, onSelectIndex }: SizeSliderProps): React.JSX.Element {
  const maxIndex = Math.max(sizes.length - 1, 0);
  const currentSize = sizes[selectedIndex];

  return (
    <div className="flex flex-col gap-1 text-sm" data-testid="size-slider">
      <div className="flex items-baseline justify-between">
        <span className="font-medium text-slate-700 dark:text-slate-200">Corpus size</span>
        <span className="font-semibold tabular-nums text-slate-900 dark:text-slate-100" data-testid="size-slider-value">
          {currentSize === undefined ? "—" : humanizeTokens(currentSize)}
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={maxIndex}
        step={1}
        value={selectedIndex}
        disabled={sizes.length <= 1}
        onChange={(event) => onSelectIndex(Number(event.target.value))}
        aria-label="Corpus size"
        aria-valuetext={currentSize === undefined ? undefined : humanizeTokens(currentSize)}
        className="w-full accent-slate-800 dark:accent-slate-200"
      />
      <div className="flex justify-between text-xs text-slate-400 dark:text-slate-500">
        <span>{sizes.length > 0 ? humanizeTokens(sizes[0]) : ""}</span>
        <span>{sizes.length > 0 ? humanizeTokens(sizes[maxIndex]) : ""}</span>
      </div>
    </div>
  );
}
