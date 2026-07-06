"""Pydantic data models shared across the benchmark harness.

All structured data crossing a module boundary (questions, per-row results) is a
validated model rather than a raw dict, per the project conventions.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class QuestionTier(str, Enum):
    """Difficulty tiers of the fixed question bank."""

    LOOKUP = "lookup"        # single-fact recall ("What is X's profession?")
    RELATIONAL = "relational"  # multi-entity links ("How is X related to Y?")
    TIMELINE = "timeline"    # ordering/causality across the narrative


class BenchmarkQuestion(BaseModel):
    """One graded question with its gold answer."""

    question_id: str = Field(..., description="Stable identifier, e.g. 'holmes_lookup_1'")
    book_key: str = Field(..., description="Corpus book this question targets")
    tier: QuestionTier = Field(..., description="Difficulty tier")
    question_text: str = Field(..., description="The question posed to each system")
    gold_answer: str = Field(..., description="Reference answer used by the LLM judge")


class SystemName(str, Enum):
    """The three architectures under comparison on the token sweep axis."""

    PLAIN_LLM = "plain_llm"
    RAG = "rag"
    WIKI = "wiki"


class ResultRow(BaseModel):
    """One graded row of the scaling sweep — one CSV line.

    The true sweep axis is ``corpus_token_count``; ``corpus_size`` (book count) is
    retained only as an internal fixed-ordering detail. Every graded cell is a
    ``ResultRow``, including failures (populated ``error``) and the plain-LLM
    context-wall skip (populated ``skipped_reason``).
    """

    system: SystemName
    corpus_size: int = Field(..., description="Number of books loaded (1..5); internal ordering detail")
    corpus_token_count: int = Field(..., description="Cumulative corpus tokens — the true sweep axis")
    measured_or_extrapolated: Literal["measured", "extrapolated"] = Field(
        ..., description="Whether this row was run (measured) or projected (extrapolated) — never blurred"
    )
    question_id: str
    tier: QuestionTier
    latency_seconds: float = Field(..., description="Wall-clock query latency")
    cost_usd: float = Field(..., description="Query cost in USD (0.0 if unmeasured)")
    accuracy: float = Field(..., description="Judge score: 0.0 | 0.5 | 1.0")
    answer_text: str = Field(..., description="The system's raw answer")
    judge_rationale: str = Field(default="", description="One-line judge justification")
    error: str = Field(default="", description="Populated if the row failed to run")
    skipped_reason: str = Field(
        default="",
        description=(
            "Non-empty when the cell was not run, e.g. 'exceeds_context_window' for the plain-LLM "
            "wall row. A skipped row has no measured accuracy or cost (both stay at their unmeasured "
            "defaults); the UI renders it as 'exceeded context window', not as a zero score."
        ),
    )

    @classmethod
    def csv_header(cls) -> list[str]:
        """Return the CSV column names in stable field order.

        Example:
            >>> ResultRow.csv_header()[0]
            'system'
        """
        return list(cls.model_fields)

    def to_csv_row(self) -> list[str]:
        """Serialize this row to CSV cell strings aligned with ``csv_header``.

        Round-trips losslessly: ``ResultRow(**dict(zip(csv_header(), to_csv_row())))``
        reconstructs an equal row (Pydantic coerces the strings back).

        Example:
            >>> row = ResultRow(system=SystemName.RAG, corpus_size=1, corpus_token_count=100_000,
            ...                 measured_or_extrapolated="measured", question_id="q1",
            ...                 tier=QuestionTier.LOOKUP, latency_seconds=1.0, cost_usd=0.0,
            ...                 accuracy=1.0, answer_text="a")
            >>> row.to_csv_row()[0]
            'rag'
        """
        return [str(value) for value in self.model_dump(mode="json").values()]


class JudgeVerdict(BaseModel):
    """Structured-output schema returned by the accuracy judge."""

    score: Literal[0.0, 0.5, 1.0] = Field(
        ..., description="0 wrong, 0.5 partially correct, 1 fully correct"
    )
    rationale: str = Field(..., description="One sentence explaining the score")
