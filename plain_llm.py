"""Plain-LLM baseline: stuff the WHOLE loaded corpus into Sonnet 5's 1M context, answer, judge.

This is the long-context baseline that defines the crossover (issue #7, PRD stories #10/#11). For one
sweep size it loads the whole corpus (``corpus_loader``), and:

- **corpus <= the ~1M context wall** — stuffs the entire corpus into a single Sonnet 5 call (with the
  1M-context beta header wired), grades the answer with the shared judge, and emits a *measured*
  ``ResultRow`` with the input-token cost priced by the shared ``cost.py``.
- **corpus > the context wall** — returns a *skipped* ``ResultRow`` (``skipped_reason="exceeds_context_window"``)
  with **no API call and no raise**, so the wall is explicit in the data rather than an error (Rule 12,
  conventions.md "the plain-LLM path returns a skipped row with a reason past the context wall").

The crossover is the headline of the whole benchmark, so it must be **exact and logged**: the skip
decision compares ``corpus_load.total_token_count`` (the same tiktoken axis ``corpus_loader`` owns)
against :data:`CONTEXT_WINDOW_TOKEN_LIMIT`, and every decision is emitted as a structured log field
(never inferred). A generation failure *below* the wall becomes an error ``ResultRow`` (populated
``error``), not a skip and not a halt.

Seam (the sweep runner #9 and tests consume it): ``run_plain_llm_query(question, size, *,
anthropic_client=None, corpus_loader=..., settings=None) -> ResultRow`` — the Anthropic client and the
corpus loader are injected so tests mock at the boundary and never hit a live API (CLAUDE.md §6). This
mirrors ``rag.run_rag_query``'s shape (measured ``latency_seconds`` + ``cost.py`` cost on a
``ResultRow``) so #9 can treat all three paths uniformly.
"""

import time
from collections.abc import Callable

import structlog

import cost
from corpus_loader import CorpusLoad, load_for_token_target
from judge import grade_answer
from models import BenchmarkQuestion, ResultRow, SystemName
from settings import BenchmarkSettings, get_settings

logger = structlog.get_logger()

# Reason: the crossover wall. Sonnet 5's context window is ~1M tokens (reference/integrations.md, PRD
# decision #4: "plain-LLM is measured only while the corpus <= ~1M tokens"). The comparison is against
# the corpus token count on the SAME tiktoken axis corpus_loader uses, so the crossover is exact: a
# corpus of exactly this many tokens is answered; strictly more is skipped. Prompt-wrapper/answer
# tokens are negligible against the coarse sweep targets (100k, 500k, 1M, 2M, ...) and the PRD frames
# the wall as "corpus > context limit", so the corpus-token count is the documented comparison axis.
CONTEXT_WINDOW_TOKEN_LIMIT = 1_000_000

# The documented ``skipped_reason`` for the context-wall row (models.py / api-contracts.md). The UI
# renders this as "exceeded context window", not a zero score.
CONTEXT_WALL_SKIPPED_REASON = "exceeds_context_window"

# Reason: the 1M-token context beta header, wired per reference/integrations.md ("set the `context-1m`
# beta header"). NOTE (surfaced conflict, Rule 7): current-generation Claude models — including
# claude-sonnet-5 — expose the 1M window *natively* at standard pricing with no beta header (the
# `context-1m-*` beta gated the older Sonnet 4 / 4.5 line). On Sonnet 5 this header is therefore a
# harmless no-op; it is wired anyway to honour the reference doc and make the intent explicit. If the
# header ever starts 400-ing on a future model, drop it rather than "fixing" the value.
CONTEXT_1M_BETA_HEADER = "context-1m-2025-08-07"

GENERATION_MAX_TOKENS = 1024
GENERATION_SYSTEM_PROMPT = (
    "You are a question-answering system. Answer the user's question using ONLY the provided text. "
    "Do not use outside knowledge. If the text does not contain the answer, say you cannot answer "
    "from the provided text. Answer concisely."
)


def _build_anthropic_client() -> object:
    """Construct the Anthropic client keyed from settings (never a hardcoded secret; SDK imported lazily)."""
    import anthropic  # lazy: tests inject a fake and need neither the SDK nor a real key

    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key)


def _generate_answer(
    anthropic_client: object, question_text: str, corpus_text: str, *, model: str
) -> tuple[str, int, int]:
    """Answer ``question_text`` from the WHOLE ``corpus_text`` with Sonnet 5 (1M-context beta wired).

    The entire loaded corpus is stuffed into a single call — this is what makes the path the honest
    long-context baseline. The 1M-context beta header is attached via ``extra_headers`` so the same
    ``messages.create`` seam ``rag.py`` uses stays mockable.

    Returns:
        ``(answer_text, input_tokens, output_tokens)`` — the answer and usage for cost accounting.
    """
    user_prompt = (
        f"Text:\n{corpus_text}\n\nQuestion: {question_text}\n\nAnswer using only the text above."
    )
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=GENERATION_MAX_TOKENS,
        system=GENERATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        extra_headers={"anthropic-beta": CONTEXT_1M_BETA_HEADER},
    )
    answer_text = "".join(
        getattr(block, "text", "")
        for block in (getattr(response, "content", None) or [])
        if getattr(block, "type", None) == "text"
    ).strip()
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return answer_text, input_tokens, output_tokens


def run_plain_llm_query(
    question: BenchmarkQuestion,
    size: int,
    *,
    anthropic_client: object | None = None,
    corpus_loader: Callable[[int], CorpusLoad] = load_for_token_target,
    settings: BenchmarkSettings | None = None,
) -> ResultRow:
    """Run one question through the plain-LLM baseline at sweep ``size`` and return a ``ResultRow``.

    Loads the whole corpus for ``size`` via ``corpus_loader``, then either answers it with Sonnet 5
    (corpus within the context wall) or emits a skipped row (corpus past the wall). The three outcomes:

    - **within the wall** — Sonnet 5 answers from the whole corpus -> judge vs gold -> a ``measured``
      row with ``latency_seconds`` and ``cost_usd`` (generation input/output tokens priced by ``cost.py``).
    - **past the wall** (``corpus_token_count > CONTEXT_WINDOW_TOKEN_LIMIT``) — a ``measured`` row with
      ``skipped_reason="exceeds_context_window"``, **no API call**, no raise; cost/accuracy stay at
      their unmeasured defaults. The crossover value is logged, not inferred.
    - **generation failure below the wall** — caught and returned as a ``measured`` error row
      (populated ``error``); the sweep records the bad cell and continues (Rule 12), it does not halt.

    Args:
        question: The graded question (carries gold answer, tier, id).
        size: The sweep size — the token target passed to ``corpus_loader`` (the sweep axis point).
        anthropic_client: Injected in tests; a real one is built from settings if omitted.
        corpus_loader: ``(size) -> CorpusLoad`` — defaults to ``corpus_loader.load_for_token_target``;
            tests inject a fake returning a small ``CorpusLoad`` to stay hermetic.
        settings: Benchmark settings (generation model id); ``get_settings()`` if omitted.

    Returns:
        A ``ResultRow`` for system ``PLAIN_LLM`` — graded, skipped-past-wall, or error-populated.
    """
    settings = settings or get_settings()

    # Tracked outside the try so a failure *after* the corpus loaded still reports the real axis values
    # on the error row (a generation failure below the wall is honest about the size it happened at).
    corpus_token_count = 0
    corpus_size = 0

    started = time.perf_counter()
    try:
        corpus_load = corpus_loader(size)
        corpus_token_count = corpus_load.total_token_count
        corpus_size = len(corpus_load.sources)

        if corpus_token_count > CONTEXT_WINDOW_TOKEN_LIMIT:
            # Reason: the corpus no longer fits Sonnet 5's context — emit a skipped row (not an error,
            # not a raise) so the wall is explicit in the data. Log the exact crossover so the boundary
            # is a recorded field, never inferred downstream.
            logger.info(
                "plain_llm_context_wall_skipped",
                question_id=question.question_id,
                corpus_token_count=corpus_token_count,
                context_window_limit=CONTEXT_WINDOW_TOKEN_LIMIT,
                over_by=corpus_token_count - CONTEXT_WINDOW_TOKEN_LIMIT,
                skipped_reason=CONTEXT_WALL_SKIPPED_REASON,
            )
            return ResultRow(
                system=SystemName.PLAIN_LLM,
                corpus_size=corpus_size,
                corpus_token_count=corpus_token_count,
                measured_or_extrapolated="measured",
                question_id=question.question_id,
                tier=question.tier,
                latency_seconds=time.perf_counter() - started,
                cost_usd=0.0,
                accuracy=0.0,
                answer_text="",
                skipped_reason=CONTEXT_WALL_SKIPPED_REASON,
            )

        anthropic_client = anthropic_client or _build_anthropic_client()
        answer_text, gen_input_tokens, gen_output_tokens = _generate_answer(
            anthropic_client, question.question_text, corpus_load.text, model=settings.rag_generation_model
        )
        verdict = grade_answer(answer_text, question.gold_answer, client=anthropic_client)

        cost_usd = cost.token_cost_usd(
            settings.rag_generation_model, input_tokens=gen_input_tokens, output_tokens=gen_output_tokens
        )
        latency_seconds = time.perf_counter() - started

        logger.info(
            "plain_llm_query_completed",
            question_id=question.question_id,
            corpus_token_count=corpus_token_count,
            context_window_limit=CONTEXT_WINDOW_TOKEN_LIMIT,
            score=verdict.score,
            cost_usd=cost_usd,
        )
        return ResultRow(
            system=SystemName.PLAIN_LLM,
            corpus_size=corpus_size,
            corpus_token_count=corpus_token_count,
            measured_or_extrapolated="measured",
            question_id=question.question_id,
            tier=question.tier,
            latency_seconds=latency_seconds,
            cost_usd=cost_usd,
            accuracy=verdict.score,
            answer_text=answer_text,
            judge_rationale=verdict.rationale,
        )
    except Exception as exc:  # noqa: BLE001 — one bad cell becomes an error row, never halts the sweep
        latency_seconds = time.perf_counter() - started
        logger.error(
            "plain_llm_query_failed",
            question_id=question.question_id,
            size=size,
            error_message=str(exc),
            fix_suggestion="Inspect the plain-LLM error; cell recorded as an error ResultRow so the sweep continues",
        )
        return ResultRow(
            system=SystemName.PLAIN_LLM,
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
