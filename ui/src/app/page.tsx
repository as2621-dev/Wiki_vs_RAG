/**
 * The single page: renders the two headline figures from the pre-baked, bundled
 * `results.json`. No backend, no live model calls — `results.json` is imported as a
 * static asset and validated by the shared loader. A missing/empty file shows the
 * empty state instead of crashing.
 */

import rawResults from "../../public/results.json";
import { AccuracyChart } from "@/components/AccuracyChart";
import { CostChart } from "@/components/CostChart";
import { EmptyState } from "@/components/EmptyState";
import { parseResults } from "@/lib/loadResults";

export default function HomePage(): React.JSX.Element {
  const loaded = parseResults(rawResults);

  return (
    <main className="mx-auto flex max-w-[880px] flex-col gap-12 px-6 py-12">
      <header className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">
          Long-context LLM vs RAG vs Wiki — scaling figures
        </h1>
        {loaded.status === "ok" && (
          <p className="text-sm text-slate-500 dark:text-slate-400">{loaded.payload.generated_note}</p>
        )}
      </header>

      {loaded.status === "empty" ? (
        <EmptyState reason={loaded.reason} />
      ) : (
        <>
          <AccuracyChart points={loaded.payload.series.accuracy} />
          <CostChart points={loaded.payload.series.cost} />
        </>
      )}
    </main>
  );
}
