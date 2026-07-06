"""Opus-4.8 accuracy judge: grade one answer against its gold into a JudgeVerdict.

The judge is the accuracy seam of the sweep: the RAG path (#6) and the sweep
runner (#9) call :func:`grade_answer` with a system's answer and the question's
gold answer, and receive a validated ``JudgeVerdict`` (score ∈ {0.0, 0.5, 1.0} +
a one-sentence rationale). Structured output is *enforced* via a strict tool the
model is forced to call — the score is parsed and validated into the Pydantic
model rather than regexed out of free text.

Determinism: the prompt is a module-level constant and the model must answer
through a fixed strict tool schema. ``temperature`` is deliberately not sent —
it is not an accepted parameter on ``claude-opus-4-8`` (Anthropic removed the
sampling parameters on the Opus 4.7/4.8 line); determinism therefore rests on the
frozen prompt + enforced tool schema, not on a temperature knob.

Grading is strictly against the provided gold text, never the model's own world
knowledge — an answer that is true in reality but absent from / contradicting the
gold is scored per the gold.
"""

import anthropic
import structlog
from pydantic import ValidationError

from models import JudgeVerdict
from settings import get_settings

logger = structlog.get_logger()

# Reason: one retry (2 attempts total) is enough to ride out a transient refusal or
# malformed tool call without turning a hard outage into a slow hang.
MAX_ATTEMPTS = 2
JUDGE_MAX_TOKENS = 1024

JUDGE_TOOL_NAME = "submit_verdict"

# Enforced structured output: a strict tool the model is *forced* to call, so the
# verdict arrives as a validated {score, rationale} object rather than free text.
JUDGE_TOOL: dict = {
    "name": JUDGE_TOOL_NAME,
    "description": "Submit the accuracy verdict for the candidate answer against the gold answer.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "enum": [0.0, 0.5, 1.0],
                "description": "0.0 wrong, 0.5 partially correct, 1.0 fully correct — judged against the gold text.",
            },
            "rationale": {
                "type": "string",
                "description": "One sentence explaining the score, referencing the gold answer.",
            },
        },
        "required": ["score", "rationale"],
        "additionalProperties": False,
    },
}

JUDGE_SYSTEM_PROMPT = (
    "You are a strict accuracy judge for a question-answering benchmark. You are given a "
    "candidate answer and a reference GOLD answer. Grade the candidate ONLY against the provided "
    "gold answer text — do not use your own world knowledge, and do not reward an answer for being "
    "true in reality if it disagrees with or is absent from the gold answer.\n\n"
    "Apply this rubric and report it through the submit_verdict tool:\n"
    "- 1.0: the candidate conveys the same core fact(s) as the gold answer (wording may differ).\n"
    "- 0.5: the candidate is partially correct — captures some but not all of the gold, or is "
    "correct but materially incomplete.\n"
    "- 0.0: the candidate is wrong, empty, irrelevant, or contradicts the gold answer.\n\n"
    "Always call the submit_verdict tool with a score of exactly 0.0, 0.5, or 1.0 and a one-sentence "
    "rationale. Never answer in prose."
)


def _build_client() -> anthropic.Anthropic:
    """Construct the Anthropic client, keyed from settings (never a hardcoded secret)."""
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key)


def _build_user_prompt(answer: str, gold_answer: str) -> str:
    """Render the per-question grading prompt from the answer and its gold reference."""
    return (
        f"GOLD ANSWER (the reference to grade against):\n{gold_answer}\n\n"
        f"CANDIDATE ANSWER (to be graded):\n{answer}\n\n"
        "Grade the candidate answer against the gold answer using the rubric, then call submit_verdict."
    )


def _parse_verdict(response: object) -> JudgeVerdict | None:
    """Extract and validate a JudgeVerdict from the forced tool call.

    Returns ``None`` (rather than raising) when the model produced no usable
    ``submit_verdict`` tool call or its input fails JudgeVerdict validation, so the
    caller can retry or fall back deterministically.
    """
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == JUDGE_TOOL_NAME:
            try:
                return JudgeVerdict(**block.input)
            except (TypeError, ValidationError) as exc:
                logger.warning(
                    "judge_verdict_validation_failed",
                    error_message=str(exc),
                    fix_suggestion="Model tool input did not match the JudgeVerdict schema; retry then fall back to 0.0",
                )
                return None
    return None


def grade_answer(answer: str, gold_answer: str, *, client: anthropic.Anthropic | None = None) -> JudgeVerdict:
    """Grade ``answer`` against ``gold_answer`` into a validated JudgeVerdict.

    Empty/whitespace answers short-circuit to 0.0 with no API call. A refusal or an
    unparseable model response is retried once and then falls back to a rationalised
    0.0 (never crashes, never silently swallowed). Genuine transport/API errors are
    retried once and then re-raised so the caller can record an error row (fail loud).

    Args:
        answer: The system's raw answer text to grade.
        gold_answer: The reference gold answer text to grade strictly against.
        client: Anthropic client (injected in tests); a real one is built if omitted.

    Returns:
        A ``JudgeVerdict`` with ``score`` in {0.0, 0.5, 1.0} and a one-sentence rationale.

    Example:
        >>> from unittest.mock import Mock
        >>> verdict = grade_answer("", "A consulting detective.", client=Mock())
        >>> verdict.score
        0.0
    """
    logger.info("judge_grade_started", answer_len=len(answer or ""), gold_len=len(gold_answer or ""))

    if not answer or not answer.strip():
        verdict = JudgeVerdict(score=0.0, rationale="No answer was produced, so it cannot match the gold answer.")
        logger.info("judge_scored", score=0.0, short_circuited=True)
        return verdict

    client = client or _build_client()
    user_prompt = _build_user_prompt(answer, gold_answer)
    last_reason = "no_attempt"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.messages.create(
                model=get_settings().judge_model,
                max_tokens=JUDGE_MAX_TOKENS,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[JUDGE_TOOL],
                tool_choice={"type": "tool", "name": JUDGE_TOOL_NAME},
            )
        except Exception as exc:
            # Reason: a transport/API failure is an infra problem, not a 0.0 grade —
            # retry once, then re-raise so the caller records an error row (Rule 12).
            last_reason = f"api_error: {exc}"
            logger.error(
                "judge_call_failed",
                attempt=attempt,
                error_message=str(exc),
                fix_suggestion="Check ANTHROPIC_API_KEY and network connectivity",
            )
            if attempt == MAX_ATTEMPTS:
                raise
            continue

        if getattr(response, "stop_reason", None) == "refusal":
            last_reason = "model_refused"
            logger.warning(
                "judge_refused",
                attempt=attempt,
                fix_suggestion="Answer/gold text may trip safety filters; retrying then falling back to 0.0",
            )
            continue

        verdict = _parse_verdict(response)
        if verdict is not None:
            logger.info("judge_scored", score=verdict.score, attempt=attempt)
            return verdict

        last_reason = "malformed_output"
        logger.warning(
            "judge_malformed_output",
            attempt=attempt,
            fix_suggestion="Model did not return a valid submit_verdict call; retrying then falling back to 0.0",
        )

    fallback = JudgeVerdict(
        score=0.0,
        rationale=f"Judge could not produce a valid verdict ({last_reason}); scored 0.0 by fallback.",
    )
    logger.error(
        "judge_failed",
        reason=last_reason,
        fix_suggestion="Inspect the model response; verdict defaulted to 0.0 so the cell is graded, not dropped",
    )
    return fallback
