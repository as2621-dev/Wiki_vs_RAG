"""Generate a sample ``results.json`` fixture for the UI (issue #11).

Builds a small synthetic sweep of ``ResultRow`` objects covering all three systems,
all three ``SeriesPointKind`` values (measured / extrapolated / skipped_context), and a
plain-LLM ``skipped_context`` context-wall row, then runs the real ``aggregate.build_results_json``
so the fixture is byte-for-byte the contract the UI consumes. Writes to ``ui/public/results.json``.

Run: ``python scripts/generate_sample_results.py``
"""

from __future__ import annotations

from pathlib import Path

from aggregate import build_results_json
from models import QuestionTier, ResultRow, SystemName

# Measured sweep axis (subset of the documented targets), extrapolated to 1B by aggregate.
MEASURED_TOKENS: list[int] = [100_000, 500_000, 1_000_000, 5_000_000, 50_000_000]
CONTEXT_WALL_TOKEN: int = 2_000_000  # plain_llm drops out just past 1M

# Hand-tuned per-system curves so the fixture tells the thesis' story (wiki accurate but costly).
ACCURACY_BY_SYSTEM: dict[SystemName, float] = {
    SystemName.PLAIN_LLM: 0.90,
    SystemName.RAG: 0.62,
    SystemName.WIKI: 0.82,
}
# Cost per measured point scales roughly with tokens; wiki is the expensive path.
COST_PER_MILLION: dict[SystemName, float] = {
    SystemName.PLAIN_LLM: 0.010,
    SystemName.RAG: 0.015,
    SystemName.WIKI: 2.0,
}

QUESTION_ID: str = "q_lookup_01"


def _measured_row(system: SystemName, token_count: int) -> ResultRow:
    """Build one measured ``ResultRow`` for a system at a token count."""
    corpus_size = max(1, token_count // 100_000)
    cost = COST_PER_MILLION[system] * (token_count / 1_000_000)
    return ResultRow(
        system=system,
        corpus_size=corpus_size,
        corpus_token_count=token_count,
        measured_or_extrapolated="measured",
        question_id=QUESTION_ID,
        tier=QuestionTier.LOOKUP,
        latency_seconds=1.2,
        cost_usd=cost,
        accuracy=ACCURACY_BY_SYSTEM[system],
        answer_text=f"{system.value} answer at {token_count} tokens.",
        judge_rationale="Matches the gold answer.",
    )


def _wall_row(token_count: int) -> ResultRow:
    """Build the plain-LLM context-wall skip row (``skipped_context``)."""
    return ResultRow(
        system=SystemName.PLAIN_LLM,
        corpus_size=max(1, token_count // 100_000),
        corpus_token_count=token_count,
        measured_or_extrapolated="measured",
        question_id=QUESTION_ID,
        tier=QuestionTier.LOOKUP,
        latency_seconds=0.0,
        cost_usd=0.0,
        accuracy=0.0,
        answer_text="",
        skipped_reason="exceeds_context_window",
    )


def build_rows() -> list[ResultRow]:
    """Assemble the synthetic sweep rows."""
    rows: list[ResultRow] = []
    # plain_llm: measured up to 1M, then walled at 2M.
    for token_count in [100_000, 500_000, 1_000_000]:
        rows.append(_measured_row(SystemName.PLAIN_LLM, token_count))
    rows.append(_wall_row(CONTEXT_WALL_TOKEN))
    # rag + wiki: measured across the full axis.
    for system in (SystemName.RAG, SystemName.WIKI):
        for token_count in MEASURED_TOKENS:
            rows.append(_measured_row(system, token_count))
    return rows


def main() -> None:
    """Generate the fixture and write it under the UI's public assets."""
    out_path = Path(__file__).resolve().parent.parent / "ui" / "public" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_results_json(rows=build_rows(), out_path=out_path)
    print(f"wrote {out_path} — {len(payload.series.accuracy)} accuracy pts, {len(payload.series.cost)} cost pts")


if __name__ == "__main__":
    main()
