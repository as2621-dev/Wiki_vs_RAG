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
    """The two architectures under comparison."""

    RAG = "rag"
    WIKI = "wiki"


class ResultRow(BaseModel):
    """One graded row of the scaling sweep — one CSV line."""

    system: SystemName
    corpus_size: int = Field(..., description="Number of books loaded (1..5)")
    question_id: str
    tier: QuestionTier
    latency_seconds: float = Field(..., description="Wall-clock query latency")
    cost_usd: float = Field(..., description="Query cost in USD (0.0 if unmeasured)")
    accuracy: float = Field(..., description="Judge score: 0.0 | 0.5 | 1.0")
    answer_text: str = Field(..., description="The system's raw answer")
    judge_rationale: str = Field(default="", description="One-line judge justification")
    error: str = Field(default="", description="Populated if the row failed to run")


class JudgeVerdict(BaseModel):
    """Structured-output schema returned by the accuracy judge."""

    score: Literal[0.0, 0.5, 1.0] = Field(
        ..., description="0 wrong, 0.5 partially correct, 1 fully correct"
    )
    rationale: str = Field(..., description="One sentence explaining the score")
