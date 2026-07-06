"""Sweep runner: orchestrate every (size × system × question) cell into a ``ResultRow`` and aggregate
to ``results.csv`` / ``results.json`` (issue #9, PRD stories #16/#17/#18/#26).

This is the orchestrator that composes the three measured paths (``rag`` / ``plain_llm`` / ``wiki``)
over the token sweep axis. It owns three durability properties, and re-implements none of the path
logic (each path already returns a graded/skipped/extrapolated/error ``ResultRow``):

- **Resumable (story #17).** Every completed cell is appended to a crash-safe JSONL progress store
  (``sweep_progress.jsonl``) the instant it finishes (write + ``flush`` + ``fsync``). On restart the
  store is replayed into a set of completed cell keys; a cell already in that set is **skipped** — the
  path callable is never called again for it — so a kill mid-sweep resumes from the last completed
  cell and never double-counts (duplicate-cell guard).
- **Per-cell failure isolation (Rule 12).** Each path call is wrapped: if a path *raises*, the cell
  becomes an ``error`` ``ResultRow`` (with ``error`` populated) and the sweep continues — one flaky
  system on one cell never kills the other systems' rows for that cell nor halts the run. (The real
  paths already catch internally; this guards the injected seam and any un-caught bug.)
- **Reconstructable logs (story #26).** Every cell start / finish / resume-skip / failure is a
  structured ``structlog`` event (``fix_suggestion`` on failures), so a multi-day headless run is
  reconstructable from the logs alone; the measured-vs-extrapolated boundary is logged per row.

The path callables are **injected** as a uniform per-cell seam ``(question, corpus_load) -> ResultRow``
(``paths={SystemName.RAG: fn, ...}``), so tests pass fakes and never hit a live API or ``claude -p``
(CLAUDE.md §6). ``build_default_paths`` wires the real path seams (with once-per-size index/ingest) for
an actual run. The output schema is exactly ``ResultRow``'s serialization — the stable contract the
aggregate step (#10) reads.
"""

import csv
import io
import json
import os
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import structlog

from corpus_loader import CorpusLoad, load_for_token_target
from models import BenchmarkQuestion, ResultRow, SystemName
from questions import questions_for_token_target

logger = structlog.get_logger()

# PRD / api-contracts sweep axis (measured band). Extrapolation to 1B is the aggregate step's job (#10).
DEFAULT_SWEEP_TARGETS: tuple[int, ...] = (
    100_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
    50_000_000,
)

# Fixed system order so cell enumeration and the emitted CSV are deterministic run-to-run.
SWEEP_SYSTEM_ORDER: tuple[SystemName, ...] = (SystemName.PLAIN_LLM, SystemName.RAG, SystemName.WIKI)

PROGRESS_STORE_FILENAME = "sweep_progress.jsonl"
RESULTS_CSV_FILENAME = "results.csv"
RESULTS_JSON_FILENAME = "results.json"

# Uniform per-cell path seam: (question, corpus_load) -> a graded ``ResultRow``. The runner injects one
# per system; tests pass fakes; ``build_default_paths`` wires the real paths.
PathQuery = Callable[[BenchmarkQuestion, CorpusLoad], ResultRow]
CorpusLoader = Callable[[int], CorpusLoad]
QuestionsProvider = Callable[[Iterable[str]], list[BenchmarkQuestion]]


def _cell_key(size: int, system: SystemName, question_id: str) -> str:
    """Return the stable identity for one sweep cell — the resume/dedup unit.

    Uses the sweep INPUT ``size`` (the token target), not the derived corpus token count, so the key
    is byte-identical across runs even if two targets both cap to the same available-corpus token
    count (a shortfall collision would otherwise let one row overwrite the other's resume slot).

    Example:
        >>> _cell_key(100_000, SystemName.RAG, "holmes_lookup_1")
        '100000:rag:holmes_lookup_1'
    """
    return f"{size}:{system.value}:{question_id}"


@dataclass(frozen=True)
class SweepResult:
    """Summary of a sweep run — where the outputs are and how many cells ran vs resumed."""

    results_csv_path: str
    results_json_path: str
    progress_store_path: str
    total_rows: int
    newly_run_cells: int
    resumed_cells: int


@dataclass
class _SweepProgress:
    """Replayed progress: which cell keys are already done, and the rows for them (in store order)."""

    completed_cell_keys: set[str]
    completed_rows: list[ResultRow]


def _load_progress(store_path: Path) -> _SweepProgress:
    """Replay the append-only JSONL progress store into completed cell keys + rows.

    A torn final line (a crash mid-append) is skipped loudly rather than aborting the resume — the
    prior complete lines stay valid because the store is append-only. A duplicate cell key keeps the
    first occurrence, so a replay can never double-count a row.

    Args:
        store_path: Path to ``sweep_progress.jsonl`` (may not exist yet — treated as empty).

    Returns:
        The completed cell keys and their rows in store order.
    """
    completed_cell_keys: set[str] = set()
    completed_rows: list[ResultRow] = []
    if not store_path.exists():
        return _SweepProgress(completed_cell_keys, completed_rows)

    for line_number, raw_line in enumerate(store_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            cell_key = str(record["cell_key"])
            row = ResultRow(**record["row"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            # Reason: a crash mid-write can leave a torn final line; skip it (loudly) and resume from
            # the complete lines rather than losing the whole prior run (Rule 12 — surfaced, not hidden).
            logger.warning(
                "sweep_progress_line_skipped",
                store_path=str(store_path),
                line_number=line_number,
                error_message=str(exc),
                fix_suggestion="Torn/partial progress line (likely a crash mid-append); ignoring it and resuming",
            )
            continue
        if cell_key in completed_cell_keys:
            continue  # duplicate-cell guard: never load the same cell twice
        completed_cell_keys.add(cell_key)
        completed_rows.append(row)
    return _SweepProgress(completed_cell_keys, completed_rows)


def _append_completed_cell(store_path: Path, cell_key: str, row: ResultRow) -> None:
    """Append one completed cell to the JSONL progress store, crash-safely (write + flush + fsync).

    An append-only log is inherently crash-safe for every prior line; ``fsync`` forces this line to
    disk before we treat the cell as done, so a kill immediately after can never lose a completed cell.
    """
    store_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"cell_key": cell_key, "row": row.model_dump(mode="json")})
    with store_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_atomic(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (temp file + ``os.replace``) — reusing the #8 pattern.

    A reader (or a re-run) always sees either the previous complete file or the new complete one,
    never a half-written results file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _write_results(out_dir: Path, rows: list[ResultRow]) -> tuple[Path, Path]:
    """Aggregate all rows to ``results.csv`` + ``results.json`` via ``ResultRow``'s own serialization.

    The CSV header/rows come from ``ResultRow.csv_header`` / ``to_csv_row`` (the ``csv`` module quotes
    any commas/newlines in answer text); the JSON is the list of ``model_dump`` dicts. No ad-hoc
    columns — this is the exact contract the aggregate step (#10) and the UI read.

    Returns:
        ``(results_csv_path, results_json_path)``.
    """
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(ResultRow.csv_header())
    for row in rows:
        writer.writerow(row.to_csv_row())
    csv_path = out_dir / RESULTS_CSV_FILENAME
    _write_atomic(csv_path, csv_buffer.getvalue())

    json_path = out_dir / RESULTS_JSON_FILENAME
    _write_atomic(json_path, json.dumps([row.model_dump(mode="json") for row in rows], indent=2))
    return csv_path, json_path


def run_sweep(
    sizes: Sequence[int],
    *,
    paths: dict[SystemName, PathQuery],
    out_dir: str | Path,
    corpus_loader: CorpusLoader = load_for_token_target,
    questions_provider: QuestionsProvider = questions_for_token_target,
    system_order: Sequence[SystemName] = SWEEP_SYSTEM_ORDER,
) -> SweepResult:
    """Run the full (size × system × question) sweep, resumably, into ``results.csv`` / ``results.json``.

    For each size the corpus is loaded once (``corpus_loader``) and only the questions answerable at
    that size are enumerated (``questions_provider`` off the loaded source keys). Each (size, system,
    question) cell dispatches to its injected path callable and collects a ``ResultRow``. A completed
    cell is appended to the crash-safe progress store immediately; on a re-run, cells already in the
    store are skipped (the path callable is not called again) so a killed run resumes from where it
    stopped. A path that raises becomes an ``error`` row and the sweep continues (Rule 12).

    Args:
        sizes: The sweep token targets (e.g. ``DEFAULT_SWEEP_TARGETS``), in order.
        paths: ``SystemName -> (question, corpus_load) -> ResultRow`` — the injected per-cell seam.
            A system absent from this map is skipped (lets a run cover a subset of systems).
        out_dir: Directory for the progress store and the ``results.csv`` / ``results.json`` outputs.
        corpus_loader: ``(size) -> CorpusLoad``; defaults to ``corpus_loader.load_for_token_target``.
            Tests inject a fake returning a small ``CorpusLoad`` to stay hermetic.
        questions_provider: ``(loaded_source_keys) -> [BenchmarkQuestion]``; defaults to
            ``questions.questions_for_token_target`` so only answerable questions run at each size.
        system_order: Fixed system iteration order (determinism); defaults to ``SWEEP_SYSTEM_ORDER``.

    Returns:
        A ``SweepResult`` with the output paths and the run/resume cell counts.
    """
    out_dir = Path(out_dir)
    store_path = out_dir / PROGRESS_STORE_FILENAME
    progress = _load_progress(store_path)
    rows: list[ResultRow] = list(progress.completed_rows)
    resumed_cells = len(progress.completed_cell_keys)
    newly_run_cells = 0

    logger.info(
        "sweep_started",
        sizes=list(sizes),
        systems=[system.value for system in system_order if system in paths],
        out_dir=str(out_dir),
        already_completed_cells=resumed_cells,
    )

    for size in sizes:
        corpus_load = corpus_loader(size)
        corpus_token_count = corpus_load.total_token_count
        corpus_size = len(corpus_load.sources)
        questions = questions_provider(corpus_load.loaded_source_keys)
        for question in questions:
            for system in system_order:
                path_query = paths.get(system)
                if path_query is None:
                    continue
                cell_key = _cell_key(size, system, question.question_id)
                if cell_key in progress.completed_cell_keys:
                    logger.info(
                        "sweep_cell_skipped_resumed",
                        cell_key=cell_key,
                        size=size,
                        system=system.value,
                        question_id=question.question_id,
                    )
                    continue

                logger.info(
                    "sweep_cell_started",
                    cell_key=cell_key,
                    size=size,
                    system=system.value,
                    corpus_token_count=corpus_token_count,
                    question_id=question.question_id,
                )
                try:
                    row = path_query(question, corpus_load)
                except Exception as exc:  # noqa: BLE001 — one bad cell becomes an error row, never halts
                    logger.error(
                        "sweep_cell_failed",
                        cell_key=cell_key,
                        size=size,
                        system=system.value,
                        question_id=question.question_id,
                        error_message=str(exc),
                        fix_suggestion="Path raised for this cell; recorded as an error ResultRow so the sweep continues",
                    )
                    row = ResultRow(
                        system=system,
                        corpus_size=corpus_size,
                        corpus_token_count=corpus_token_count,
                        measured_or_extrapolated="measured",
                        question_id=question.question_id,
                        tier=question.tier,
                        latency_seconds=0.0,
                        cost_usd=0.0,
                        accuracy=0.0,
                        answer_text="",
                        error=f"{type(exc).__name__}: {exc}",
                    )

                _append_completed_cell(store_path, cell_key, row)
                rows.append(row)
                progress.completed_cell_keys.add(cell_key)
                newly_run_cells += 1
                logger.info(
                    "sweep_cell_completed",
                    cell_key=cell_key,
                    system=system.value,
                    corpus_token_count=row.corpus_token_count,
                    measured_or_extrapolated=row.measured_or_extrapolated,
                    accuracy=row.accuracy,
                    skipped_reason=row.skipped_reason,
                    error=row.error,
                )

    csv_path, json_path = _write_results(out_dir, rows)
    logger.info(
        "sweep_completed",
        total_rows=len(rows),
        newly_run_cells=newly_run_cells,
        resumed_cells=resumed_cells,
        results_csv_path=str(csv_path),
        results_json_path=str(json_path),
    )
    return SweepResult(
        results_csv_path=str(csv_path),
        results_json_path=str(json_path),
        progress_store_path=str(store_path),
        total_rows=len(rows),
        newly_run_cells=newly_run_cells,
        resumed_cells=resumed_cells,
    )


def build_default_paths(
    *,
    index_corpus: Callable[..., object] | None = None,
    run_rag_query: Callable[..., ResultRow] | None = None,
    run_plain_llm_query: Callable[..., ResultRow] | None = None,
    ingest_corpus: Callable[..., object] | None = None,
    run_wiki_query: Callable[..., ResultRow] | None = None,
    measured_token_cap: int | None = None,
) -> dict[SystemName, PathQuery]:
    """Wire the real path seams into the uniform per-cell callables ``run_sweep`` injects.

    Encapsulates the one-time-per-size setup the pure ``run_sweep`` loop deliberately does not know
    about: RAG indexes each size once (memoized by token count) before querying it, and the wiki
    ingests each within-cap size once (an incomplete ingest yields an ``error`` row, never an answer
    against a partial wiki). Plain-LLM has no per-size setup — it just answers from the loaded corpus.
    The underlying seams default to the real module functions and are injectable so this wiring's
    once-per-size memoization is testable with fakes (CLAUDE.md §6).

    Args:
        index_corpus / run_rag_query: RAG seams (default ``rag.index_corpus`` / ``rag.run_rag_query``).
        run_plain_llm_query: Plain-LLM seam (default ``plain_llm.run_plain_llm_query``).
        ingest_corpus / run_wiki_query: Wiki seams (default ``wiki.ingest_corpus`` / ``wiki.run_wiki_query``).
        measured_token_cap: The wiki measured cap (default ``wiki.MEASURED_TOKEN_CAP``); past it the
            wiki path emits an extrapolated row without ingesting.

    Returns:
        A ``paths`` dict ready for ``run_sweep``.
    """
    import rag
    import wiki
    from plain_llm import run_plain_llm_query as _default_plain_llm_query

    index_corpus = index_corpus or rag.index_corpus
    run_rag_query = run_rag_query or rag.run_rag_query
    run_plain_llm_query = run_plain_llm_query or _default_plain_llm_query
    ingest_corpus = ingest_corpus or wiki.ingest_corpus
    run_wiki_query = run_wiki_query or wiki.run_wiki_query
    measured_token_cap = measured_token_cap if measured_token_cap is not None else wiki.MEASURED_TOKEN_CAP

    indexed_token_counts: set[int] = set()
    ingested_token_counts: set[int] = set()

    def _rag_path(question: BenchmarkQuestion, corpus_load: CorpusLoad) -> ResultRow:
        corpus_token_count = corpus_load.total_token_count
        if corpus_token_count not in indexed_token_counts:
            index_corpus(corpus_load)  # once per size — embed+upsert, not per question
            indexed_token_counts.add(corpus_token_count)
        return run_rag_query(
            question, corpus_token_count=corpus_token_count, corpus_size=len(corpus_load.sources)
        )

    def _plain_llm_path(question: BenchmarkQuestion, corpus_load: CorpusLoad) -> ResultRow:
        # The corpus is already loaded; feed it straight through instead of re-loading by token target.
        return run_plain_llm_query(
            question, corpus_load.total_token_count, corpus_loader=lambda _size: corpus_load
        )

    def _wiki_path(question: BenchmarkQuestion, corpus_load: CorpusLoad) -> ResultRow:
        corpus_token_count = corpus_load.total_token_count
        corpus_size = len(corpus_load.sources)
        if corpus_token_count <= measured_token_cap and corpus_token_count not in ingested_token_counts:
            ingest = ingest_corpus(corpus_load)  # once per size — resumable, one-time
            if not getattr(ingest, "complete", False):
                # Reason: a usage-window cutoff left the wiki partial — surface an error row telling the
                # caller to resume, never answer against a partial wiki (Rule 12, no silent partial).
                return ResultRow(
                    system=SystemName.WIKI,
                    corpus_size=corpus_size,
                    corpus_token_count=corpus_token_count,
                    measured_or_extrapolated="measured",
                    question_id=question.question_id,
                    tier=question.tier,
                    latency_seconds=0.0,
                    cost_usd=0.0,
                    accuracy=0.0,
                    answer_text="",
                    error="wiki_ingest_incomplete: rate-limit cutoff left the wiki partial; resume the ingest loop",
                )
            ingested_token_counts.add(corpus_token_count)
        return run_wiki_query(
            question, corpus_token_count=corpus_token_count, corpus_size=corpus_size
        )

    return {
        SystemName.RAG: _rag_path,
        SystemName.PLAIN_LLM: _plain_llm_path,
        SystemName.WIKI: _wiki_path,
    }
