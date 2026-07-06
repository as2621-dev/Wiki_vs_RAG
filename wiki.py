"""Wiki path: drive LLMWiki through the Claude Code CLI (``claude -p``) to ingest the corpus and
answer a question (issue #8, PRD stories #12/#13/#14).

This is the agent-built-wiki side of the thesis. For one sweep size it ingests the loaded corpus into
LLMWiki by invoking ``claude -p`` (MCP config from ``./llmwiki mcp-config``, wired via settings) once
per source, then answers a question by invoking ``claude -p`` again. Both invocations use
``--output-format json`` so the run's **API-equivalent** cost is read straight from the CLI's
``total_cost_usd`` field — the author pays $0 on the Max plan, but that field still reports what the
same work would have cost on the metered API, which IS the reported cost gap (the thesis).

Three properties make this path honest:

- **Resumable ingest (story #13).** Ingest is idempotent and checkpointed per source: a small JSON
  checkpoint keyed by corpus token count records which source keys are already in the wiki, written
  atomically (temp file + ``os.replace``) after each source. A rate-limit / usage-window cutoff
  mid-ingest (:class:`WikiRateLimitError`) stops the loop with the checkpoint intact, so the next
  off-hours run **resumes** from the last checkpoint rather than re-ingesting from scratch. Running a
  completed ingest again is a no-op.
- **Measured ceiling then extrapolate (PRD decision #4).** The wiki is measured only up to
  :data:`MEASURED_TOKEN_CAP` — building it past that is days of wall-clock under Max-plan rate limits.
  A size within the cap yields a ``measured`` row; a size past the cap yields an ``extrapolated`` row
  (emitted, marked, no CLI call), never silently dropped or mislabelled.
- **API-equivalent cost + wall-clock (story #14).** ``ResultRow.cost_usd`` for a query is the answer
  invocation's ``total_cost_usd`` read directly from the CLI JSON (already USD — so this path does NOT
  go through ``cost.py``'s token→USD pricing table, unlike RAG/plain-LLM which price raw token usage).
  Latency is the CLI-reported ``duration_ms`` (falling back to a measured ``perf_counter`` span).

Seams (the sweep runner #9 and tests consume these), mirroring ``rag.py``: ``ingest_corpus`` ingests a
size once (resumable, one-time — like ``rag.index_corpus``), and ``run_wiki_query`` answers one
question against an already-ingested size (per query — like ``rag.run_rag_query``). The ``claude -p``
subprocess is behind an injected ``cli_runner`` and the judge behind an injected ``judge_client``, so
tests mock at the boundary and NEVER invoke a real ``claude -p`` (CLAUDE.md §6). A ``ResultRow`` is
returned in every case — graded, extrapolated-past-cap, or ``error``-populated — so one bad cell never
halts the sweep (Rule 12).
"""

import json
import os
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import structlog

from judge import grade_answer
from models import BenchmarkQuestion, ResultRow, SystemName
from settings import BenchmarkSettings, get_settings

logger = structlog.get_logger()

# Reason: the wiki's measured ceiling (PRD decision #4: "Wiki is measured to a modest cap ~2–5M tokens
# then extrapolated"). Building the wiki over the cap is days of wall-clock under Max-plan rate limits,
# so sizes past it are projected by the aggregate step, not run here. 5M is the top of the documented
# 2–5M band — the most measured data we commit to collect before extrapolating. A size <= this is
# measured; strictly more is emitted as an extrapolated row (never silently dropped).
MEASURED_TOKEN_CAP = 5_000_000

# Reason: a hard per-invocation timeout so a wedged CLI/MCP call becomes an error row instead of
# hanging the off-hours loop forever (Rule 12). 30 min comfortably covers a single-source ingest or a
# single answer; a genuinely longer call is a problem worth surfacing, not waiting on.
CLAUDE_CLI_TIMEOUT_SECONDS = 1800

# Instruction handed to `claude -p` for ingest; the source TEXT is piped via stdin (never argv), so no
# corpus content is ever interpolated into the command line (no injection surface).
INGEST_INSTRUCTION = (
    "Ingest the document provided on standard input into the LLMWiki knowledge base using the wiki "
    "MCP tools. Extract entities, relationships, and facts. Do not answer any question — only ingest."
)

# Substrings that mark a rate-limit / usage-window cutoff in a CLI error message — the resumable signal.
_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "usage limit", "usage window", "usage_limit")


class WikiRateLimitError(Exception):
    """A ``claude -p`` rate-limit / usage-window cutoff — signals RESUME from checkpoint, not restart."""


class WikiCliError(Exception):
    """A ``claude -p`` non-zero exit / error result for a non-rate-limit reason (becomes an error row)."""


@dataclass(frozen=True)
class CliResult:
    """The raw outcome of one ``claude -p`` invocation (the injected ``cli_runner`` boundary shape)."""

    returncode: int
    stdout: str
    stderr: str


# A CLI runner runs an arg list and returns a CliResult. Injectable so tests mock the subprocess
# boundary (CLAUDE.md §6); the default below is the only place a real subprocess is spawned.
CliRunner = Callable[..., CliResult]


@dataclass(frozen=True)
class IngestResult:
    """Outcome of ingesting one corpus size into LLMWiki (resumable, one-time — like ``IndexResult``)."""

    corpus_token_count: int
    ingested_source_keys: tuple[str, ...]
    newly_ingested_count: int
    resumed: bool  # True if a prior checkpoint already had sources (this run continued, not started)
    complete: bool  # True if every source of this size is now ingested
    interrupted: bool  # True if a rate-limit cutoff stopped the loop with work remaining
    ingest_cost_usd: float  # cumulative API-equivalent ingest cost (one-time; NOT folded per-query)
    checkpoint_path: str


@dataclass
class WikiCheckpoint:
    """Resumable ingest state for one size: which source keys are in the wiki + the ingest cost so far.

    Persisted as JSON at ``wiki-checkpoint-<corpus_token_count>.json`` in the checkpoint dir. Keyed by
    corpus token count so sizes never share state, exactly as ``rag`` keys its Pinecone namespace.
    """

    corpus_token_count: int
    ingested_source_keys: list[str]
    ingest_cost_usd: float


def _run_claude_cli(
    args: list[str], *, input_text: str | None = None, timeout: float = CLAUDE_CLI_TIMEOUT_SECONDS
) -> CliResult:
    """Run the ``claude`` CLI with an explicit arg LIST — the only real subprocess spawn in this module.

    Security (Rule: no injection): ``shell=False`` and the corpus content travels via ``input`` (stdin),
    never as an argv element, so no untrusted text is interpolated into a command line. A bounded
    ``timeout`` keeps a wedged call from hanging the loop.

    Returns:
        A :class:`CliResult` capturing return code, stdout, and stderr.
    """
    completed = subprocess.run(  # noqa: S603 — fixed arg list, shell=False, corpus via stdin not argv
        args,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CliResult(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def _is_rate_limit_message(message: str) -> bool:
    """True if ``message`` reads like a rate-limit / usage-window cutoff (the resumable signal)."""
    lowered = message.lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


def _parse_cli_json(result: CliResult) -> dict:
    """Parse ``claude -p --output-format json`` stdout into a dict, classifying failures loudly.

    A non-zero exit or an ``is_error`` result raises :class:`WikiRateLimitError` when the message reads
    like a rate-limit cutoff (so the caller resumes), else :class:`WikiCliError` (so the caller records
    an error row). Nothing is swallowed and no failure is silently treated as a $0 success (Rule 12).

    Args:
        result: The raw CLI outcome.

    Returns:
        The parsed JSON object (``result`` / ``total_cost_usd`` / ``duration_ms`` fields).
    """
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        if _is_rate_limit_message(message):
            raise WikiRateLimitError(message[:200] or "rate-limit / usage-window cutoff")
        raise WikiCliError(f"claude -p exited {result.returncode}: {message[:200]}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise WikiCliError(f"claude -p produced unparseable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise WikiCliError("claude -p JSON output was not an object")

    if payload.get("is_error"):
        message = str(payload.get("result") or payload.get("error") or "")
        if _is_rate_limit_message(message):
            raise WikiRateLimitError(message[:200] or "rate-limit / usage-window cutoff")
        raise WikiCliError(f"claude -p reported an error: {message[:200]}")
    return payload


def _build_cli_args(prompt: str, settings: BenchmarkSettings) -> list[str]:
    """Build the ``claude -p`` arg list (JSON output, MCP config wired when configured).

    ``prompt`` is a fixed instruction or the graded question text (controlled, from the golden set) —
    never untrusted corpus content, which is piped via stdin instead.
    """
    args = [settings.claude_cli_binary, "-p", prompt, "--output-format", "json"]
    if settings.llmwiki_mcp_config_path:
        args += ["--mcp-config", settings.llmwiki_mcp_config_path]
    return args


def _build_answer_prompt(question_text: str) -> str:
    """Render the answer instruction: answer strictly from the LLMWiki knowledge base, concisely."""
    return (
        "Answer the following question using ONLY the LLMWiki knowledge base you have ingested "
        "(use the wiki MCP tools to look facts up). Do not use outside knowledge. If the wiki does "
        f"not contain the answer, say you cannot answer from the wiki. Answer concisely.\n\n"
        f"Question: {question_text}"
    )


def _checkpoint_path(checkpoint_dir: str, corpus_token_count: int) -> Path:
    """Return the checkpoint file path for a size (keyed by token count so sizes never share state)."""
    return Path(checkpoint_dir) / f"wiki-checkpoint-{corpus_token_count}.json"


def load_checkpoint(checkpoint_dir: str, corpus_token_count: int) -> WikiCheckpoint:
    """Load the resumable ingest checkpoint for a size, or an empty one if none exists yet.

    Example:
        >>> import tempfile
        >>> load_checkpoint(tempfile.mkdtemp(), 100).ingested_source_keys
        []
    """
    path = _checkpoint_path(checkpoint_dir, corpus_token_count)
    if not path.exists():
        return WikiCheckpoint(corpus_token_count=corpus_token_count, ingested_source_keys=[], ingest_cost_usd=0.0)
    data = json.loads(path.read_text())
    return WikiCheckpoint(
        corpus_token_count=int(data["corpus_token_count"]),
        ingested_source_keys=list(data["ingested_source_keys"]),
        ingest_cost_usd=float(data.get("ingest_cost_usd", 0.0)),
    )


def _write_checkpoint_atomic(checkpoint_dir: str, checkpoint: WikiCheckpoint) -> Path:
    """Write the checkpoint atomically (temp file + ``os.replace``) so a crash mid-write can't corrupt it.

    ``os.replace`` is atomic within a filesystem, so a resuming run always reads either the previous
    complete checkpoint or the new complete one — never a half-written file (data-integrity).

    Returns:
        The path the checkpoint was written to.
    """
    dir_path = Path(checkpoint_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, checkpoint.corpus_token_count)
    tmp_path = dir_path / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_text(
        json.dumps(
            {
                "corpus_token_count": checkpoint.corpus_token_count,
                "ingested_source_keys": checkpoint.ingested_source_keys,
                "ingest_cost_usd": checkpoint.ingest_cost_usd,
            }
        )
    )
    os.replace(tmp_path, path)
    return path


def _ingest_one_source(source: object, *, cli_runner: CliRunner, settings: BenchmarkSettings) -> float:
    """Ingest one whole source into LLMWiki via ``claude -p`` (text piped on stdin), return its cost.

    Raises :class:`WikiRateLimitError` on a usage cutoff (caller resumes) or :class:`WikiCliError`
    on any other CLI failure (caller records an error row).
    """
    args = _build_cli_args(INGEST_INSTRUCTION, settings)
    result = cli_runner(args, input_text=source.text, timeout=CLAUDE_CLI_TIMEOUT_SECONDS)
    payload = _parse_cli_json(result)
    return float(payload.get("total_cost_usd", 0.0) or 0.0)


def ingest_corpus(
    corpus_load: object,
    *,
    cli_runner: CliRunner = _run_claude_cli,
    checkpoint_dir: str | None = None,
    settings: BenchmarkSettings | None = None,
) -> IngestResult:
    """Ingest a corpus size into LLMWiki, resumably and idempotently (one-time per size).

    Loads the checkpoint for this size, then ingests each not-yet-ingested source via ``claude -p``,
    persisting the checkpoint atomically after each success. Already-ingested sources are skipped, so
    a re-run continues from where a prior run stopped (resume) and a re-run of a complete ingest is a
    no-op (idempotent). A rate-limit cutoff mid-loop stops with the checkpoint intact (``interrupted``)
    so the next off-hours run resumes rather than restarts (story #13).

    Args:
        corpus_load: A ``corpus_loader.CorpusLoad`` (``.total_token_count`` + ``.sources`` with
            ``.book_key`` / ``.text``).
        cli_runner: Injected in tests; the real subprocess runner otherwise (CLAUDE.md §6).
        checkpoint_dir: Where checkpoints live; ``settings.wiki_checkpoint_dir`` if omitted.
        settings: Benchmark settings (CLI binary, MCP config); ``get_settings()`` if omitted.

    Returns:
        An :class:`IngestResult` recording ingested keys, whether it resumed / completed / was
        interrupted, and the cumulative API-equivalent ingest cost.
    """
    settings = settings or get_settings()
    checkpoint_dir = checkpoint_dir or settings.wiki_checkpoint_dir
    corpus_token_count = corpus_load.total_token_count

    checkpoint = load_checkpoint(checkpoint_dir, corpus_token_count)
    already_ingested = set(checkpoint.ingested_source_keys)
    resumed = len(already_ingested) > 0
    all_source_keys = {source.book_key for source in corpus_load.sources}

    newly_ingested_count = 0
    interrupted = False
    checkpoint_path = _checkpoint_path(checkpoint_dir, corpus_token_count)
    for source in corpus_load.sources:
        if source.book_key in already_ingested:
            # Idempotent/resumable: this source is already in the wiki — never re-ingest it.
            continue
        try:
            call_cost_usd = _ingest_one_source(source, cli_runner=cli_runner, settings=settings)
        except WikiRateLimitError as exc:
            # Reason: a usage-window cutoff is not a failure — stop with the checkpoint intact so the
            # next off-hours run RESUMES from here. The current source is not marked done, so it retries.
            interrupted = True
            logger.warning(
                "wiki_ingest_rate_limited",
                corpus_token_count=corpus_token_count,
                ingested_so_far=len(already_ingested),
                remaining=len(all_source_keys - already_ingested),
                error_message=str(exc),
                fix_suggestion="Usage window hit; ingest checkpointed — re-run the off-hours loop to resume",
            )
            break

        checkpoint.ingested_source_keys.append(source.book_key)
        checkpoint.ingest_cost_usd += call_cost_usd
        # Persist AFTER each source (atomically) — this is what makes a mid-ingest cutoff resumable.
        checkpoint_path = _write_checkpoint_atomic(checkpoint_dir, checkpoint)
        already_ingested.add(source.book_key)
        newly_ingested_count += 1

    complete = all_source_keys.issubset(already_ingested)
    logger.info(
        "wiki_ingest_completed",
        corpus_token_count=corpus_token_count,
        newly_ingested=newly_ingested_count,
        total_ingested=len(already_ingested),
        resumed=resumed,
        complete=complete,
        interrupted=interrupted,
        ingest_cost_usd=checkpoint.ingest_cost_usd,
    )
    return IngestResult(
        corpus_token_count=corpus_token_count,
        ingested_source_keys=tuple(checkpoint.ingested_source_keys),
        newly_ingested_count=newly_ingested_count,
        resumed=resumed,
        complete=complete,
        interrupted=interrupted,
        ingest_cost_usd=checkpoint.ingest_cost_usd,
        checkpoint_path=str(checkpoint_path),
    )


def _answer_via_cli(
    question_text: str, *, cli_runner: CliRunner, settings: BenchmarkSettings
) -> tuple[str, float, float]:
    """Answer one question against the ingested wiki via ``claude -p``; parse answer, cost, latency.

    Returns:
        ``(answer_text, total_cost_usd, latency_seconds)`` — the answer, the API-equivalent cost read
        from the CLI's ``total_cost_usd``, and the wall-clock (CLI ``duration_ms``, else a measured span).
    """
    args = _build_cli_args(_build_answer_prompt(question_text), settings)
    started = time.perf_counter()
    result = cli_runner(args, input_text=None, timeout=CLAUDE_CLI_TIMEOUT_SECONDS)
    measured_elapsed = time.perf_counter() - started

    payload = _parse_cli_json(result)
    answer_text = str(payload.get("result", "") or "").strip()
    total_cost_usd = float(payload.get("total_cost_usd", 0.0) or 0.0)
    duration_ms = payload.get("duration_ms")
    latency_seconds = float(duration_ms) / 1000.0 if duration_ms is not None else measured_elapsed
    return answer_text, total_cost_usd, latency_seconds


def _extrapolated_row(
    question: BenchmarkQuestion, *, corpus_token_count: int, corpus_size: int, latency_seconds: float
) -> ResultRow:
    """Build the ``extrapolated`` row for a size past the measured cap (emitted + marked, no CLI call)."""
    logger.info(
        "wiki_extrapolated_past_cap",
        question_id=question.question_id,
        corpus_token_count=corpus_token_count,
        measured_token_cap=MEASURED_TOKEN_CAP,
        over_by=corpus_token_count - MEASURED_TOKEN_CAP,
    )
    return ResultRow(
        system=SystemName.WIKI,
        corpus_size=corpus_size,
        corpus_token_count=corpus_token_count,
        measured_or_extrapolated="extrapolated",
        question_id=question.question_id,
        tier=question.tier,
        latency_seconds=latency_seconds,
        cost_usd=0.0,
        accuracy=0.0,
        answer_text="",
    )


def run_wiki_query(
    question: BenchmarkQuestion,
    *,
    corpus_token_count: int,
    corpus_size: int,
    cli_runner: CliRunner = _run_claude_cli,
    judge_client: object | None = None,
    settings: BenchmarkSettings | None = None,
) -> ResultRow:
    """Answer one question against an already-ingested wiki size and return a graded ``ResultRow``.

    Within the measured cap: ``claude -p`` answers from the wiki → judge vs gold → a ``measured`` row
    whose ``cost_usd`` is the CLI's ``total_cost_usd`` (API-equivalent) and whose ``latency_seconds``
    is the CLI-reported duration. Past the cap: an ``extrapolated`` row (no CLI call), never silent. A
    CLI failure / partial wiki: an ``error`` row with ``error`` populated — the sweep records the bad
    cell and continues, it does not halt (Rule 12).

    Args:
        question: The graded question (gold answer, tier, id).
        corpus_token_count: The size's cumulative token count — the sweep axis and cap comparison.
        corpus_size: Source count at this size (ResultRow ordering detail).
        cli_runner: Injected in tests; the real subprocess runner otherwise (CLAUDE.md §6).
        judge_client: Anthropic client for the judge only; injected in tests, built by the judge if None.
        settings: Benchmark settings; ``get_settings()`` if omitted.

    Returns:
        A ``ResultRow`` for system WIKI — graded, extrapolated-past-cap, or ``error``-populated.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    if corpus_token_count > MEASURED_TOKEN_CAP:
        return _extrapolated_row(
            question,
            corpus_token_count=corpus_token_count,
            corpus_size=corpus_size,
            latency_seconds=time.perf_counter() - started,
        )

    try:
        answer_text, total_cost_usd, latency_seconds = _answer_via_cli(
            question.question_text, cli_runner=cli_runner, settings=settings
        )
        verdict = grade_answer(answer_text, question.gold_answer, client=judge_client)

        logger.info(
            "wiki_query_completed",
            question_id=question.question_id,
            corpus_token_count=corpus_token_count,
            score=verdict.score,
            cost_usd=total_cost_usd,
            latency_seconds=latency_seconds,
        )
        return ResultRow(
            system=SystemName.WIKI,
            corpus_size=corpus_size,
            corpus_token_count=corpus_token_count,
            measured_or_extrapolated="measured",
            question_id=question.question_id,
            tier=question.tier,
            latency_seconds=latency_seconds,
            cost_usd=total_cost_usd,
            accuracy=verdict.score,
            answer_text=answer_text,
            judge_rationale=verdict.rationale,
        )
    except Exception as exc:  # noqa: BLE001 — one bad cell becomes an error row, never halts the sweep
        latency_seconds = time.perf_counter() - started
        logger.error(
            "wiki_query_failed",
            question_id=question.question_id,
            corpus_token_count=corpus_token_count,
            error_message=str(exc),
            fix_suggestion="Inspect the claude -p wiki error (non-zero exit / partial wiki); cell recorded as an error ResultRow so the sweep continues",
        )
        return ResultRow(
            system=SystemName.WIKI,
            corpus_size=corpus_size,
            corpus_token_count=corpus_token_count,
            measured_or_extrapolated="measured",
            question_id=question.question_id,
            tier=question.tier,
            latency_seconds=latency_seconds,
            cost_usd=0.0,
            accuracy=0.0,
            answer_text="",
            error=f"{type(exc).__name__}: {exc}",
        )


def run_wiki_slice(
    question: BenchmarkQuestion,
    corpus_load: object,
    *,
    cli_runner: CliRunner = _run_claude_cli,
    checkpoint_dir: str | None = None,
    judge_client: object | None = None,
    settings: BenchmarkSettings | None = None,
) -> ResultRow:
    """Convenience (the demoable happy path): ingest the size then answer one question.

    Past the cap, emits the ``extrapolated`` row without ingesting. Within the cap, ingests (resumably)
    then answers via ``run_wiki_query``. An interrupted ingest (usage cutoff before the size is fully
    ingested) yields an ``error`` row directing the caller to resume — it does not answer against a
    partial wiki. The sweep runner (#9) calls ``ingest_corpus`` once per size and ``run_wiki_query`` per
    question instead of this wrapper.
    """
    settings = settings or get_settings()
    corpus_token_count = corpus_load.total_token_count
    corpus_size = len(corpus_load.sources)
    started = time.perf_counter()

    if corpus_token_count > MEASURED_TOKEN_CAP:
        return _extrapolated_row(
            question,
            corpus_token_count=corpus_token_count,
            corpus_size=corpus_size,
            latency_seconds=time.perf_counter() - started,
        )

    try:
        ingest = ingest_corpus(
            corpus_load, cli_runner=cli_runner, checkpoint_dir=checkpoint_dir, settings=settings
        )
    except Exception as exc:  # noqa: BLE001 — an ingest failure becomes an error row, never halts the sweep
        logger.error(
            "wiki_ingest_failed",
            question_id=question.question_id,
            corpus_token_count=corpus_token_count,
            error_message=str(exc),
            fix_suggestion="Inspect the claude -p ingest error; cell recorded as an error ResultRow so the sweep continues",
        )
        return ResultRow(
            system=SystemName.WIKI,
            corpus_size=corpus_size,
            corpus_token_count=corpus_token_count,
            measured_or_extrapolated="measured",
            question_id=question.question_id,
            tier=question.tier,
            latency_seconds=time.perf_counter() - started,
            cost_usd=0.0,
            accuracy=0.0,
            answer_text="",
            error=f"{type(exc).__name__}: {exc}",
        )

    if not ingest.complete:
        # Reason: a usage cutoff left the wiki partial — do NOT answer against it. Surface an error row
        # telling the caller to resume; the checkpoint holds the progress (Rule 12, no silent partial).
        return ResultRow(
            system=SystemName.WIKI,
            corpus_size=corpus_size,
            corpus_token_count=corpus_token_count,
            measured_or_extrapolated="measured",
            question_id=question.question_id,
            tier=question.tier,
            latency_seconds=time.perf_counter() - started,
            cost_usd=0.0,
            accuracy=0.0,
            answer_text="",
            error="wiki_ingest_incomplete: rate-limit cutoff left the wiki partial; resume the ingest loop before answering",
        )

    return run_wiki_query(
        question,
        corpus_token_count=corpus_token_count,
        corpus_size=corpus_size,
        cli_runner=cli_runner,
        judge_client=judge_client,
        settings=settings,
    )
