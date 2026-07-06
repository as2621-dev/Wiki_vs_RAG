"""Aggregate the measured sweep into fitted cost/accuracy curves extrapolated to 1B (issue #10).

Reads the ``results.csv`` the sweep runner (#9) wrote — one graded ``ResultRow`` per
(size × system × question) cell — groups it per system, fits a cost curve and an accuracy curve from
the MEASURED points only, extrapolates those fits onto a documented token grid up to 1B, and emits the
chart-ready ``results.json`` in the exact api-contracts shape (``series.accuracy`` / ``series.cost`` /
``answers``) plus additive fit metadata and headline summary stats.

Three honesty invariants make this defensible rather than decorative (Rule 12):

- **A projection is never presentable as a measurement.** Every series point carries ``kind`` —
  ``measured`` for points derived from run rows, ``extrapolated`` for curve projections, and
  ``skipped_context`` for the plain-LLM context-wall point (``accuracy: null``, never a zero score).
  The fit is computed from ``measured`` points only; the measured/extrapolated boundary is the last
  measured token count per system and is stated in ``meta``.
- **No row is silently dropped.** A skipped cell becomes a ``skipped_context`` point; a failed
  (error) cell is excluded from the accuracy mean (a failure is not a zero measurement) but is still
  surfaced in ``answers`` and logged — never vanished.
- **A weak fit is flagged, not faked.** Too few measured points → ``low_confidence`` on that system's
  meta (no confident-looking line from one point); a cost series dominated by one expensive point is
  called out in ``cost_domination_note`` so a reader sees the caveat.

Fit forms (from the PRD Technical Foundation): cost is ~linear in tokens
(``total_cost_usd = m*tokens + b``); accuracy is ~log / saturating (``accuracy = m*ln(tokens) + b``).
Both are ordinary least-squares over the measured points — a hand-rolled two-parameter fit in pure
Python, so no numpy dependency is pulled in for a job this small (Rule 2).
"""

import csv
import json
import math
import os
import uuid
from collections import defaultdict
from pathlib import Path

import structlog

from models import (
    AccuracySeriesPoint,
    AnswerCell,
    CostSeriesPoint,
    ResultRow,
    ResultsPayload,
    ResultsSeries,
    ResultsSummary,
    SeriesFitMeta,
    SeriesPointKind,
    SystemName,
)

logger = structlog.get_logger()

# Documented extrapolation grid: the token targets past the measured band we project the fits onto,
# up to the 1B target (api-contracts "100M–1B extrapolated"). Filtered to <= target at fit time.
DEFAULT_EXTRAPOLATION_GRID: tuple[int, ...] = (
    100_000_000,
    200_000_000,
    500_000_000,
    1_000_000_000,
)

DEFAULT_TARGET_TOKENS: int = 1_000_000_000

# Below this many measured points a curve is flagged low-confidence rather than fit as if trustworthy
# (a line through 1–2 points is not evidence of a trajectory). Documented threshold, not a magic number.
MIN_MEASURED_POINTS_FOR_FIT: int = 3

# A cost fit is "dominated" when one measured point is at least this fraction of the summed cost — the
# wiki's single expensive size can otherwise drive the whole slope; surfaced, never hidden.
COST_DOMINATION_FRACTION: float = 0.8

# Stable system order for deterministic series output, matching the sweep runner's enumeration order.
_SYSTEM_ORDER: tuple[SystemName, ...] = (SystemName.PLAIN_LLM, SystemName.RAG, SystemName.WIKI)


def _read_rows_from_csv(results_csv_path: Path) -> list[ResultRow]:
    """Read a sweep ``results.csv`` back into validated ``ResultRow`` objects.

    Round-trips the sweep's own serialization: each data line is zipped with ``ResultRow.csv_header``
    and coerced back through Pydantic (the exact inverse of ``ResultRow.to_csv_row``), so the aggregate
    reads precisely the contract the sweep wrote — no ad-hoc column parsing.

    Args:
        results_csv_path: Path to the sweep-produced ``results.csv``.

    Returns:
        One ``ResultRow`` per data line, in file order.

    Raises:
        FileNotFoundError: If the CSV does not exist (a missing sweep output is a loud failure).

    Example:
        >>> rows = _read_rows_from_csv(Path("results.csv"))  # doctest: +SKIP
        >>> rows[0].system  # doctest: +SKIP
        <SystemName.PLAIN_LLM: 'plain_llm'>
    """
    header = ResultRow.csv_header()
    rows: list[ResultRow] = []
    with results_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # discard the header line; column order is ResultRow.csv_header()
        for cells in reader:
            rows.append(ResultRow(**dict(zip(header, cells, strict=True))))
    return rows


def _least_squares(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Ordinary least-squares slope/intercept for ``y = slope*x + intercept``, or ``None`` if unfittable.

    Pure and dependency-free (Rule 2). Returns ``None`` when fewer than two points or when all x are
    identical (a vertical spread has no defined slope) — the caller treats that as "cannot fit", never
    a silent flat line masquerading as a trend.

    Args:
        xs: Independent-variable samples (already transformed, e.g. ``ln(tokens)`` for the accuracy fit).
        ys: Dependent-variable samples, aligned with ``xs``.

    Returns:
        ``(slope, intercept)`` or ``None`` if the points cannot define a line.

    Example:
        >>> _least_squares([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
        (2.0, 1.0)
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _humanize_tokens(token_count: int) -> str:
    """Render a token count as a compact human label (``1000000`` -> ``'1M'``) for ``generated_note``.

    Example:
        >>> _humanize_tokens(50_000_000)
        '50M'
        >>> _humanize_tokens(1_000_000_000)
        '1B'
    """
    for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if token_count >= divisor and token_count % divisor == 0:
            return f"{token_count // divisor}{suffix}"
    return str(token_count)


def _group_kind(rows: list[ResultRow]) -> SeriesPointKind:
    """Classify a group of run rows as measured vs extrapolated by their ``measured_or_extrapolated``.

    A single ``extrapolated`` row in the group (e.g. the wiki path past its ingest cap) makes the whole
    point extrapolated — the honest, conservative choice: never label a point measured if any of its
    underlying rows was projected.
    """
    if any(row.measured_or_extrapolated == "extrapolated" for row in rows):
        return SeriesPointKind.EXTRAPOLATED
    return SeriesPointKind.MEASURED


def _extrapolation_grid(target_tokens: int) -> list[int]:
    """Return the documented projection grid up to (and including) ``target_tokens``."""
    grid = sorted({token for token in DEFAULT_EXTRAPOLATION_GRID if token <= target_tokens} | {target_tokens})
    return grid


def _build_measured_and_skipped_points(
    rows: list[ResultRow],
) -> tuple[
    dict[SystemName, list[AccuracySeriesPoint]],
    dict[SystemName, list[CostSeriesPoint]],
    dict[SystemName, int],
]:
    """Derive per-system accuracy/cost points from actual rows and the max token seen per system.

    Groups rows by (system, token_count) and classifies each group: an all-skipped group becomes a
    ``skipped_context`` accuracy point (``accuracy=null``, no cost point — a context wall has no cost);
    a group with graded rows becomes a measured/extrapolated accuracy point (mean over graded rows,
    error rows excluded so a failure never counts as a zero) and a cost point (summed over attempted
    rows). An all-error group is logged and contributes no series point — surfaced, not silently zeroed.

    Returns:
        ``(accuracy_points_by_system, cost_points_by_system, max_token_by_system)`` — the max token is
        the largest token count seen at all (any kind), the ceiling above which projection begins.
    """
    grouped: dict[tuple[SystemName, int], list[ResultRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.system, row.corpus_token_count)].append(row)

    accuracy_points: dict[SystemName, list[AccuracySeriesPoint]] = defaultdict(list)
    cost_points: dict[SystemName, list[CostSeriesPoint]] = defaultdict(list)
    max_token: dict[SystemName, int] = defaultdict(int)

    for (system, token_count), group in grouped.items():
        max_token[system] = max(max_token[system], token_count)
        non_skipped = [row for row in group if not row.skipped_reason]
        gradable = [row for row in non_skipped if not row.error]
        skipped = [row for row in group if row.skipped_reason]

        if gradable:
            kind = _group_kind(gradable)
            mean_accuracy = sum(row.accuracy for row in gradable) / len(gradable)
            accuracy_points[system].append(
                AccuracySeriesPoint(
                    corpus_token_count=token_count, system=system, accuracy=mean_accuracy, kind=kind
                )
            )
            # Cost is what was actually spent at this size across attempted (non-skipped) rows.
            total_cost = sum(row.cost_usd for row in non_skipped)
            cost_points[system].append(
                CostSeriesPoint(
                    corpus_token_count=token_count, system=system, total_cost_usd=total_cost, kind=kind
                )
            )
        elif skipped:
            accuracy_points[system].append(
                AccuracySeriesPoint(
                    corpus_token_count=token_count,
                    system=system,
                    accuracy=None,
                    kind=SeriesPointKind.SKIPPED_CONTEXT,
                )
            )
        else:
            # All rows in the group errored: no honest accuracy or cost to report, but do not vanish it.
            logger.warning(
                "aggregate_group_all_errored",
                system=system.value,
                corpus_token_count=token_count,
                row_count=len(group),
                fix_suggestion="Every cell at this size/system failed; inspect the error rows before trusting the curve",
            )

    return accuracy_points, cost_points, max_token


def _fit_and_extrapolate_system(
    system: SystemName,
    accuracy_points: list[AccuracySeriesPoint],
    cost_points: list[CostSeriesPoint],
    max_token: int,
    target_tokens: int,
) -> tuple[list[AccuracySeriesPoint], list[CostSeriesPoint], SeriesFitMeta]:
    """Fit one system's cost + accuracy curves from its measured points and project them to ``target``.

    A system that hit the context wall (any ``skipped_context`` point) is NOT extrapolated — the wall
    is the honest end of its data, so projecting past it would fabricate a run that cannot happen. For
    a non-walled system the accuracy fit is log (``accuracy = m*ln(tokens) + b``) and the cost fit is
    linear (``total_cost_usd = m*tokens + b``); projections land on the documented grid above
    ``max_token`` and are clamped to sane ranges (accuracy to ``[0, 1]``, cost to ``>= 0``).

    Returns:
        ``(extra_accuracy_points, extra_cost_points, meta)`` — the projected points plus the stated fit
        method, boundary, low-confidence flag, and any cost-domination note for this system.
    """
    measured_accuracy = [point for point in accuracy_points if point.kind == SeriesPointKind.MEASURED]
    measured_cost = [point for point in cost_points if point.kind == SeriesPointKind.MEASURED]
    is_walled = any(point.kind == SeriesPointKind.SKIPPED_CONTEXT for point in accuracy_points)

    measured_tokens = [point.corpus_token_count for point in measured_accuracy]
    boundary = max(measured_tokens) if measured_tokens else None
    measured_point_count = len(measured_accuracy)
    low_confidence = measured_point_count < MIN_MEASURED_POINTS_FOR_FIT

    cost_domination_note = _cost_domination_note(measured_cost)

    if is_walled:
        meta = SeriesFitMeta(
            system=system,
            accuracy_fit_method="not extrapolated: system hit the context wall (skipped_context is its honest end)",
            cost_fit_method="not extrapolated: system hit the context wall (skipped_context is its honest end)",
            measured_boundary_token_count=boundary,
            measured_point_count=measured_point_count,
            low_confidence=low_confidence,
            cost_domination_note=cost_domination_note,
        )
        return [], [], meta

    grid = [token for token in _extrapolation_grid(target_tokens) if token > max_token]

    # Measured points always carry a real accuracy (only skipped_context is null, and it is excluded
    # above), so xs and ys stay aligned; the log transform gives the saturating accuracy fit form.
    accuracy_fit = _least_squares(
        [math.log(point.corpus_token_count) for point in measured_accuracy],
        [point.accuracy for point in measured_accuracy],
    )
    cost_fit = _least_squares(
        [float(point.corpus_token_count) for point in measured_cost],
        [point.total_cost_usd for point in measured_cost],
    )

    extra_accuracy: list[AccuracySeriesPoint] = []
    if accuracy_fit is not None:
        slope, intercept = accuracy_fit
        for token in grid:
            projected = max(0.0, min(1.0, slope * math.log(token) + intercept))
            extra_accuracy.append(
                AccuracySeriesPoint(
                    corpus_token_count=token, system=system, accuracy=projected, kind=SeriesPointKind.EXTRAPOLATED
                )
            )

    extra_cost: list[CostSeriesPoint] = []
    if cost_fit is not None:
        slope, intercept = cost_fit
        for token in grid:
            projected = max(0.0, slope * token + intercept)
            extra_cost.append(
                CostSeriesPoint(
                    corpus_token_count=token, system=system, total_cost_usd=projected, kind=SeriesPointKind.EXTRAPOLATED
                )
            )

    meta = SeriesFitMeta(
        system=system,
        accuracy_fit_method=_fit_method_label("accuracy = m*ln(tokens) + b", accuracy_fit, low_confidence),
        cost_fit_method=_fit_method_label("total_cost_usd = m*tokens + b", cost_fit, low_confidence),
        measured_boundary_token_count=boundary,
        measured_point_count=measured_point_count,
        low_confidence=low_confidence,
        cost_domination_note=cost_domination_note,
    )
    return extra_accuracy, extra_cost, meta


def _fit_method_label(form: str, fit: tuple[float, float] | None, low_confidence: bool) -> str:
    """Describe a fit in plain prose, including the flag when the fit is untrustworthy."""
    if fit is None:
        return f"no fit ({form}): fewer than 2 distinct measured points"
    slope, intercept = fit
    label = f"least-squares: {form} with m={slope:.3g}, b={intercept:.3g}"
    if low_confidence:
        label += f" (LOW CONFIDENCE: < {MIN_MEASURED_POINTS_FOR_FIT} measured points)"
    return label


def _cost_domination_note(measured_cost: list[CostSeriesPoint]) -> str:
    """Return a caveat string when one measured cost point dominates the series, else empty.

    The wiki's single most expensive size can drive the entire linear fit; when the largest point is
    at least ``COST_DOMINATION_FRACTION`` of the summed cost the note names it so the reader treats the
    projected slope with suspicion rather than as a clean trend.
    """
    if len(measured_cost) < 2:
        return ""
    costs = [point.total_cost_usd for point in measured_cost]
    total = sum(costs)
    if total <= 0:
        return ""
    largest = max(costs)
    if largest >= COST_DOMINATION_FRACTION * total:
        dominant = max(measured_cost, key=lambda point: point.total_cost_usd)
        share = largest / total
        return (
            f"cost fit dominated by a single point at {_humanize_tokens(dominant.corpus_token_count)} "
            f"tokens ({share:.0%} of measured cost) — treat the slope as indicative, not precise"
        )
    return ""


def _build_answers(rows: list[ResultRow]) -> dict[str, dict[str, dict[str, AnswerCell]]]:
    """Build the per-question / per-size / per-system answer map (api-contracts ``answers``).

    Every row is represented — measured, extrapolated, skipped, and error alike — so a reader can
    inspect any cell's raw answer. A skipped cell reports ``accuracy=null`` + ``skipped_context`` (the
    UI shows "exceeded context window"); other cells carry their row's accuracy and measured kind.
    """
    answers: dict[str, dict[str, dict[str, AnswerCell]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.skipped_reason:
            kind = SeriesPointKind.SKIPPED_CONTEXT
            accuracy: float | None = None
        else:
            kind = SeriesPointKind.EXTRAPOLATED if row.measured_or_extrapolated == "extrapolated" else SeriesPointKind.MEASURED
            accuracy = row.accuracy
        cell = AnswerCell(answer_text=row.answer_text, accuracy=accuracy, kind=kind)
        answers[row.question_id][str(row.corpus_token_count)][row.system.value] = cell
    return {qid: {token: dict(systems) for token, systems in by_token.items()} for qid, by_token in answers.items()}


def _sorted_accuracy(points: list[AccuracySeriesPoint]) -> list[AccuracySeriesPoint]:
    """Sort accuracy points deterministically by system order then token count."""
    return sorted(points, key=lambda point: (_SYSTEM_ORDER.index(point.system), point.corpus_token_count))


def _sorted_cost(points: list[CostSeriesPoint]) -> list[CostSeriesPoint]:
    """Sort cost points deterministically by system order then token count."""
    return sorted(points, key=lambda point: (_SYSTEM_ORDER.index(point.system), point.corpus_token_count))


def _cost_at_token(points: list[CostSeriesPoint], system: SystemName, token: int) -> float | None:
    """Return a system's total cost at a token count, or ``None`` if there is no such point."""
    for point in points:
        if point.system == system and point.corpus_token_count == token:
            return point.total_cost_usd
    return None


def _write_atomic(path: Path, content: str) -> None:
    """Write ``content`` atomically (temp file + ``os.replace``), reusing the sweep's durable pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def build_results_json(
    results_csv_path: str | Path | None = None,
    *,
    rows: list[ResultRow] | None = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    out_path: str | Path | None = None,
) -> ResultsPayload:
    """Fit + extrapolate the sweep's measured curves to ``target_tokens`` and emit the UI ``results.json``.

    Reads rows either from ``results_csv_path`` (the sweep output) or a passed ``rows`` list, groups
    them per system, fits cost (linear) and accuracy (log) curves from the MEASURED points only,
    projects them onto the documented grid up to ``target_tokens``, and assembles the api-contracts
    payload — ``series.accuracy`` / ``series.cost`` (each point tagged ``kind``), ``answers``, plus
    additive ``meta`` (fit method + measured/extrapolated boundary per system) and ``summary`` stats.

    Args:
        results_csv_path: Path to a sweep ``results.csv`` to read rows from. Mutually exclusive with
            ``rows`` — exactly one must be provided.
        rows: Pre-loaded ``ResultRow`` objects (tests inject these to stay hermetic).
        target_tokens: The token target the curves extrapolate to (default 1B).
        out_path: If given, the validated payload is written there atomically as pretty JSON.

    Returns:
        The validated ``ResultsPayload`` — serialize with ``model_dump(mode="json")`` for the file.

    Raises:
        ValueError: If neither or both of ``results_csv_path`` / ``rows`` are provided.

    Example:
        >>> payload = build_results_json(rows=some_rows)  # doctest: +SKIP
        >>> payload.series.accuracy[0].kind  # doctest: +SKIP
        <SeriesPointKind.MEASURED: 'measured'>
    """
    if (results_csv_path is None) == (rows is None):
        raise ValueError("provide exactly one of results_csv_path or rows")
    if rows is None:
        rows = _read_rows_from_csv(Path(results_csv_path))

    logger.info("aggregate_started", row_count=len(rows), target_tokens=target_tokens)

    accuracy_by_system, cost_by_system, max_token_by_system = _build_measured_and_skipped_points(rows)

    all_accuracy: list[AccuracySeriesPoint] = []
    all_cost: list[CostSeriesPoint] = []
    metas: list[SeriesFitMeta] = []

    for system in _SYSTEM_ORDER:
        measured_accuracy = accuracy_by_system.get(system, [])
        measured_cost = cost_by_system.get(system, [])
        if not measured_accuracy and not measured_cost:
            continue
        extra_accuracy, extra_cost, meta = _fit_and_extrapolate_system(
            system,
            measured_accuracy,
            measured_cost,
            max_token_by_system.get(system, 0),
            target_tokens,
        )
        all_accuracy.extend(measured_accuracy + extra_accuracy)
        all_cost.extend(measured_cost + extra_cost)
        metas.append(meta)
        logger.info(
            "aggregate_system_fitted",
            system=system.value,
            measured_points=meta.measured_point_count,
            boundary_token_count=meta.measured_boundary_token_count,
            low_confidence=meta.low_confidence,
            extrapolated_points=len(extra_accuracy) + len(extra_cost),
        )

    sorted_accuracy = _sorted_accuracy(all_accuracy)
    sorted_cost = _sorted_cost(all_cost)

    measured_tokens = [point.corpus_token_count for point in all_accuracy if point.kind == SeriesPointKind.MEASURED]
    max_measured = max(measured_tokens) if measured_tokens else 0
    projected_tokens = [
        point.corpus_token_count for point in all_accuracy + all_cost if point.kind == SeriesPointKind.EXTRAPOLATED
    ]
    if projected_tokens:
        generated_note = (
            f"measured to {_humanize_tokens(max_measured)} tokens; "
            f"{_humanize_tokens(min(projected_tokens))}–{_humanize_tokens(target_tokens)} extrapolated"
        )
    else:
        generated_note = f"measured to {_humanize_tokens(max_measured)} tokens; no extrapolation"

    wiki_cost = _cost_at_token(sorted_cost, SystemName.WIKI, target_tokens)
    rag_cost = _cost_at_token(sorted_cost, SystemName.RAG, target_tokens)
    ratio = wiki_cost / rag_cost if wiki_cost is not None and rag_cost not in (None, 0) else None

    payload = ResultsPayload(
        generated_note=generated_note,
        series=ResultsSeries(accuracy=sorted_accuracy, cost=sorted_cost),
        answers=_build_answers(rows),
        meta=metas,
        summary=ResultsSummary(
            max_measured_token_count=max_measured,
            extrapolated_to_token_count=target_tokens,
            wiki_to_rag_cost_ratio_at_target=ratio,
        ),
    )

    if out_path is not None:
        _write_atomic(Path(out_path), json.dumps(payload.model_dump(mode="json"), indent=2))
        logger.info("aggregate_completed", out_path=str(out_path), accuracy_points=len(sorted_accuracy),
                    cost_points=len(sorted_cost))

    return payload
