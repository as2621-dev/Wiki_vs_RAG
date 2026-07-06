"""Tests for the RAG path in rag.py.

ALL THREE external services — Voyage (embeddings), Pinecone (vector store), Anthropic
(Sonnet 5 generation + judge) — are mocked at their boundaries; no test ever reaches a
live API (CLAUDE.md §6). Each test encodes *why* the behaviour matters (Rule 9): the RAG
path must produce a trustworthy measured ResultRow, keep sizes isolated by namespace so the
scaling sweep is honest, price cost via the shared cost.py, and degrade edge/error cases into
graded-or-error rows rather than crashing or halting the sweep.
"""

from types import SimpleNamespace

import pytest

import cost
import judge
import rag
from corpus import CorpusBook
from corpus_loader import load_for_token_target
from models import BenchmarkQuestion, JudgeVerdict, QuestionTier, ResultRow, SystemName


# ─── Boundary fakes (mirror the SDK shapes rag.py calls) ──────────────────────


class FakeVoyage:
    """Voyage boundary: ``.embed(texts, model, input_type)`` -> embeddings + total_tokens."""

    def __init__(self, *, tokens_per_call: int = 10) -> None:
        self.tokens_per_call = tokens_per_call
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def embed(self, texts: list[str], model: str, input_type: str) -> SimpleNamespace:
        self.calls.append((tuple(texts), input_type))
        return SimpleNamespace(embeddings=[[0.1, 0.2, 0.3, 0.4] for _ in texts], total_tokens=self.tokens_per_call)


class FakePinecone:
    """Pinecone boundary: upsert stores vectors per namespace; query returns only that namespace's."""

    def __init__(self) -> None:
        self.store: dict[str, list[dict]] = {}

    def upsert(self, *, vectors: list[dict], namespace: str) -> None:
        self.store.setdefault(namespace, []).extend(vectors)

    def query(self, *, vector, top_k: int, namespace: str, include_metadata: bool) -> SimpleNamespace:
        items = self.store.get(namespace, [])[:top_k]
        matches = [SimpleNamespace(score=1.0, metadata=item["metadata"]) for item in items]
        return SimpleNamespace(matches=matches)


class RecordingAnthropic:
    """Anthropic boundary: ``.messages.create(**kwargs)`` returns queued responses, records calls."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs) -> object:
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _generation_response(text: str, *, input_tokens: int = 100, output_tokens: int = 20) -> SimpleNamespace:
    """A canned Sonnet 5 generation response (text block + usage), mirroring the SDK shape."""
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens))


def _judge_tool_response(score: float, rationale: str) -> SimpleNamespace:
    """A canned judge response whose forced tool_use block carries the verdict (real judge parses it)."""
    block = SimpleNamespace(type="tool_use", name=judge.JUDGE_TOOL_NAME, input={"score": score, "rationale": rationale}, id="toolu_1")
    return SimpleNamespace(stop_reason="tool_use", content=[block])


def _settings(**overrides) -> SimpleNamespace:
    """Fake settings so tests need no .env / real keys (rag reads only these fields)."""
    base = {
        "voyage_embedding_model": "voyage-3",
        "rag_generation_model": "claude-sonnet-5",
        "retrieval_top_k": 8,
        "chunk_size_chars": 10,
        "chunk_overlap_chars": 2,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _question(**overrides) -> BenchmarkQuestion:
    fields = {
        "question_id": "holmes_lookup_1",
        "book_key": "holmes",
        "tier": QuestionTier.LOOKUP,
        "question_text": "What is Holmes's profession?",
        "gold_answer": "A consulting detective.",
    }
    fields.update(overrides)
    return BenchmarkQuestion(**fields)


def _corpus_load(text: str, *, total_token_count: int, sources: int = 1) -> SimpleNamespace:
    """A minimal CorpusLoad stand-in exposing the three attributes rag reads."""
    return SimpleNamespace(text=text, total_token_count=total_token_count, sources=list(range(sources)))


# ─── Happy path ───────────────────────────────────────────────────────────────


def test_happy_path_produces_measured_resultrow_with_latency_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 9): the whole point of the slice — one question at the smallest size runs the full
    # embed->upsert->retrieve->answer->judge pipeline and yields a fully-populated MEASURED ResultRow
    # with latency_seconds and cost_usd set, so it's a trustworthy cell of the scaling sweep.
    monkeypatch.setattr(rag, "grade_answer", lambda answer, gold, client=None: JudgeVerdict(score=1.0, rationale="matches gold"))
    anthropic = RecordingAnthropic([_generation_response("A consulting detective.")])

    row = rag.run_rag_slice(
        _question(),
        _corpus_load("Holmes is a consulting detective in London.", total_token_count=100_000),
        voyage_client=FakeVoyage(),
        pinecone_index=FakePinecone(),
        anthropic_client=anthropic,
        settings=_settings(),
    )

    assert isinstance(row, ResultRow)
    assert row.system is SystemName.RAG
    assert row.measured_or_extrapolated == "measured"
    assert row.corpus_token_count == 100_000
    assert row.accuracy == 1.0
    assert row.answer_text == "A consulting detective."
    assert row.latency_seconds >= 0.0
    assert row.cost_usd > 0.0  # embedding + generation priced, not left at the unmeasured default
    assert row.error == ""


# ─── Namespacing / no cross-contamination (PRD story #9) ──────────────────────


def test_two_sizes_use_separate_namespaces_with_no_cross_contamination(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 9): the sweep's honesty depends on sizes being isolated — a chunk indexed at size A
    # must NEVER be retrieved when querying size B, or the accuracy-vs-tokens curve is corrupted by
    # leakage. Index two sizes with distinguishable content, query A, and prove only A's text reaches
    # the generator.
    monkeypatch.setattr(rag, "grade_answer", lambda answer, gold, client=None: JudgeVerdict(score=1.0, rationale="ok"))
    voyage, pinecone = FakeVoyage(), FakePinecone()

    rag.index_corpus(_corpus_load("AAAAAAAAAA AAAAAAAAAA", total_token_count=100_000),
                     voyage_client=voyage, pinecone_index=pinecone, settings=_settings())
    rag.index_corpus(_corpus_load("BBBBBBBBBB BBBBBBBBBB", total_token_count=500_000),
                     voyage_client=voyage, pinecone_index=pinecone, settings=_settings())

    anthropic = RecordingAnthropic([_generation_response("answer from A")])
    rag.run_rag_query(
        _question(), corpus_token_count=100_000, corpus_size=1,
        voyage_client=voyage, pinecone_index=pinecone, anthropic_client=anthropic, settings=_settings(),
    )

    generation_prompt = anthropic.calls[0]["messages"][0]["content"]
    assert "A" in generation_prompt
    assert "B" not in generation_prompt  # size-500k chunks never leaked into the size-100k query
    assert set(pinecone.store) == {"corpus-100000", "corpus-500000"}


# ─── Cost computed by the shared cost.py (not inline) ─────────────────────────


def test_cost_is_computed_from_usage_by_shared_cost_module(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 9): cost is the thesis. The per-query cost must be exactly what cost.py computes from
    # the query-embedding tokens + generation usage, proving it flows through the shared contract
    # (reused by #7/#8) rather than being re-derived inline in rag.py.
    monkeypatch.setattr(rag, "grade_answer", lambda answer, gold, client=None: JudgeVerdict(score=0.5, rationale="partial"))
    voyage = FakeVoyage(tokens_per_call=42)
    anthropic = RecordingAnthropic([_generation_response("partial answer", input_tokens=300, output_tokens=50)])

    row = rag.run_rag_slice(
        _question(), _corpus_load("some corpus text here", total_token_count=100_000),
        voyage_client=voyage, pinecone_index=FakePinecone(), anthropic_client=anthropic, settings=_settings(),
    )

    expected = cost.token_cost_usd("voyage-3", input_tokens=42) + cost.token_cost_usd(
        "claude-sonnet-5", input_tokens=300, output_tokens=50
    )
    assert row.cost_usd == pytest.approx(expected)
    assert row.latency_seconds >= 0.0


# ─── Edge: zero-hit retrieval ─────────────────────────────────────────────────


def test_zero_hit_retrieval_yields_graded_row_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 9): a question whose size was never indexed (or whose retrieval returns nothing) must
    # still produce a graded row (the model answers "can't answer" from empty context -> likely low
    # score), never crash — the sweep must fill every cell.
    monkeypatch.setattr(rag, "grade_answer", lambda answer, gold, client=None: JudgeVerdict(score=0.0, rationale="no context"))
    anthropic = RecordingAnthropic([_generation_response("I cannot answer from the provided context.")])

    row = rag.run_rag_query(
        _question(), corpus_token_count=999_999, corpus_size=1,  # never indexed -> empty namespace
        voyage_client=FakeVoyage(), pinecone_index=FakePinecone(), anthropic_client=anthropic, settings=_settings(),
    )

    assert row.error == ""
    assert row.accuracy == 0.0
    assert row.answer_text  # a real (low-scoring) answer was produced and graded


# ─── Edge: Pinecone free-tier vector cap exceeded (surfaced, not silent) ───────


def test_vector_cap_exceeded_is_flagged_and_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 12): the free Starter tier has a hard vector ceiling; exceeding it must be a loud,
    # inspectable signal (flag + log), never a silent degrade that quietly drops vectors and skews
    # the sweep. Shrink the cap so a small corpus trips it.
    monkeypatch.setattr(rag, "PINECONE_STARTER_VECTOR_CAP", 2)
    result = rag.index_corpus(
        _corpus_load("abcdefghijklmnopqrstuvwxyz0123456789", total_token_count=100_000),
        voyage_client=FakeVoyage(), pinecone_index=FakePinecone(), settings=_settings(),
    )

    assert result.chunk_count > 2
    assert result.vector_cap_exceeded is True


# ─── Error/boundary: embedding rate-limit backoff, and per-query failure -> error row ─


def test_embedding_retries_on_transient_failure_then_succeeds() -> None:
    # Why (Rule 9): a transient Voyage rate limit must not fail the run — a bounded backoff retries
    # and succeeds. Prove the second attempt is made (and backoff is bounded, no infinite loop).
    class FlakyVoyage:
        def __init__(self) -> None:
            self.attempts = 0

        def embed(self, texts, model, input_type):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("429 rate limit exceeded")
            return SimpleNamespace(embeddings=[[0.0]], total_tokens=7)

    flaky = FlakyVoyage()
    embeddings, tokens = rag._embed_with_retry(
        flaky, ["q"], model="voyage-3", input_type="query", sleep=lambda _seconds: None
    )
    assert flaky.attempts == 2
    assert tokens == 7 and embeddings == [[0.0]]


def test_embedding_gives_up_after_bounded_attempts() -> None:
    # Why (Rule 12): a persistent failure must not loop forever — the retry is bounded and then
    # re-raises so the caller can record an error row.
    class DeadVoyage:
        def __init__(self) -> None:
            self.attempts = 0

        def embed(self, texts, model, input_type):
            self.attempts += 1
            raise RuntimeError("persistent outage")

    dead = DeadVoyage()
    with pytest.raises(RuntimeError, match="persistent outage"):
        rag._embed_with_retry(dead, ["q"], model="voyage-3", input_type="query", sleep=lambda _seconds: None)
    assert dead.attempts == rag.MAX_EMBED_ATTEMPTS


def test_per_query_failure_becomes_error_row_not_a_raised_halt(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 12): a single failing cell (here generation raises) must be recorded as an error
    # ResultRow so the sweep continues over the other (size, question) cells — it must NOT propagate
    # and halt the whole run.
    class ExplodingAnthropic:
        def __init__(self) -> None:
            self.messages = self

        def create(self, **kwargs):
            raise RuntimeError("anthropic 500")

    voyage, pinecone = FakeVoyage(), FakePinecone()
    rag.index_corpus(_corpus_load("indexed corpus text", total_token_count=100_000),
                     voyage_client=voyage, pinecone_index=pinecone, settings=_settings())

    row = rag.run_rag_query(
        _question(), corpus_token_count=100_000, corpus_size=1,
        voyage_client=voyage, pinecone_index=pinecone, anthropic_client=ExplodingAnthropic(), settings=_settings(),
    )

    assert isinstance(row, ResultRow)
    assert row.error  # populated
    assert "anthropic 500" in row.error
    assert row.measured_or_extrapolated == "measured"
    assert row.accuracy == 0.0


# ─── Integration: real corpus_loader + real judge + real cost.py wiring ───────


def test_integration_real_loader_real_judge_real_cost_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    # Why (Rule 9 + B3.5): exercise the REAL pipeline wiring the sweep runner (#9) will consume —
    # corpus_loader assembles the corpus, rag chunks/embeds/upserts/retrieves (mocked boundaries),
    # Sonnet 5 answers (mocked), the REAL judge parses a real tool_use verdict, and the REAL cost.py
    # prices it — all landing in a real ResultRow. Not a fully-stubbed happy path.
    monkeypatch.setattr(judge, "get_settings", lambda: SimpleNamespace(judge_model="claude-opus-4-8"))

    books = [CorpusBook("holmes", "A Study in Scarlet", "Sherlock Holmes is a consulting detective who lives at Baker Street.")]
    load = load_for_token_target(3, books=books, count_tokens=lambda t: len(t.split()))

    anthropic = RecordingAnthropic([
        _generation_response("A consulting detective.", input_tokens=250, output_tokens=8),
        _judge_tool_response(1.0, "The answer conveys the same profession as the gold."),
    ])

    row = rag.run_rag_slice(
        _question(),
        load,
        voyage_client=FakeVoyage(tokens_per_call=15),
        pinecone_index=FakePinecone(),
        anthropic_client=anthropic,
        settings=_settings(),
    )

    expected_cost = cost.token_cost_usd("voyage-3", input_tokens=15) + cost.token_cost_usd(
        "claude-sonnet-5", input_tokens=250, output_tokens=8
    )
    assert row.system is SystemName.RAG
    assert row.measured_or_extrapolated == "measured"
    assert row.accuracy == 1.0  # real judge parsed the real tool_use verdict
    assert row.judge_rationale
    assert row.cost_usd == pytest.approx(expected_cost)
    assert row.corpus_token_count == load.total_token_count
    assert len(anthropic.calls) == 2  # one generation call, one judge call
