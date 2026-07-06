"""RAG path: embed (Voyage) -> upsert (Pinecone) -> retrieve top-k -> answer (Sonnet 5) -> judge.

This is the first full end-to-end path of the sweep (issue #6, PRD stories #8/#9/#16/#26).
For one corpus size it embeds the loaded chunks with Voyage, upserts them into a Pinecone
namespace **keyed by corpus token count** so sizes never cross-contaminate retrieval, then per
question embeds the query, retrieves the top-k chunks, has Sonnet 5 answer from ONLY those chunks,
grades the answer with the shared judge, and emits a ``ResultRow`` with measured latency + cost.

Seams (the sweep runner #9 and tests consume these): ``index_corpus`` embeds+upserts a size once
(embed-the-corpus-once-per-size, not per question), and ``run_rag_query`` runs one question against
an already-indexed size. All three external services are injected (``voyage_client`` /
``pinecone_index`` / ``anthropic_client``) so tests mock at the boundary and never hit a live API
(CLAUDE.md §6); a real client is built from settings only when a seam is called with none.

Cost is API-equivalent and computed by the shared ``cost.py`` (never inline): per query =
query-embedding cost + generation cost. Corpus embedding is a one-time indexing cost, not a
per-query charge, so it is not folded into ``ResultRow.cost_usd``.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import structlog

import cost
from judge import grade_answer
from models import BenchmarkQuestion, ResultRow, SystemName
from settings import BenchmarkSettings, get_settings

logger = structlog.get_logger()

# Reason: Pinecone free Starter tier caps at ~300K vectors (reference/integrations.md). Past it the
# $50/mo Standard plan applies — we surface the ceiling (log + flag), never silently degrade.
PINECONE_STARTER_VECTOR_CAP = 300_000

# Reason: a bounded retry (not an infinite loop) rides out a transient Voyage rate limit without
# turning a hard outage into a hang; exhausting it re-raises so the query becomes an error row.
MAX_EMBED_ATTEMPTS = 3
_EMBED_BACKOFF_BASE_SECONDS = 0.5

# Generation is grounded ONLY in the retrieved chunks — this is what makes the path honestly RAG.
GENERATION_MAX_TOKENS = 1024
GENERATION_SYSTEM_PROMPT = (
    "You are a question-answering system. Answer the user's question using ONLY the provided "
    "context passages. Do not use outside knowledge. If the context does not contain the answer, "
    "say you cannot answer from the provided context. Answer concisely."
)


@dataclass(frozen=True)
class IndexResult:
    """Outcome of embedding+upserting one corpus size into its Pinecone namespace.

    ``vector_cap_exceeded`` is the surfaced free-tier ceiling signal (Rule 12): True when the
    number of chunks for this size would push the index past the Starter cap — logged and flagged,
    never silently degraded.
    """

    namespace: str
    chunk_count: int
    embedding_tokens: int
    vector_cap_exceeded: bool


def _namespace_for(corpus_token_count: int) -> str:
    """Return the Pinecone namespace keying a corpus size, so sizes don't cross-contaminate.

    Example:
        >>> _namespace_for(100_000)
        'corpus-100000'
    """
    return f"corpus-{corpus_token_count}"


def chunk_text(text: str, *, chunk_size_chars: int, chunk_overlap_chars: int) -> list[str]:
    """Split ``text`` into fixed-size overlapping character windows for embedding.

    Deterministic and offset-based so the same corpus always chunks identically. Overlap preserves
    context that would otherwise be cut at a window boundary. A window shorter than the step is the
    final chunk; empty text yields no chunks.

    Args:
        text: The assembled corpus text to chunk.
        chunk_size_chars: Character length of each chunk (must be positive).
        chunk_overlap_chars: Characters shared between consecutive chunks (0 <= overlap < size).

    Returns:
        The ordered list of chunk strings.

    Example:
        >>> chunk_text("abcdef", chunk_size_chars=4, chunk_overlap_chars=1)
        ['abcd', 'def']
    """
    if chunk_size_chars <= 0:
        raise ValueError(f"chunk_size_chars must be positive, got {chunk_size_chars}")
    if not 0 <= chunk_overlap_chars < chunk_size_chars:
        raise ValueError(f"chunk_overlap_chars must be in [0, chunk_size_chars), got {chunk_overlap_chars}")
    if not text:
        return []

    step = chunk_size_chars - chunk_overlap_chars
    chunks = [text[start : start + chunk_size_chars] for start in range(0, len(text), step)]
    # Reason: the trailing window can be fully covered by its predecessor's overlap; drop an empty tail.
    return [chunk for chunk in chunks if chunk]


def _build_voyage_client() -> object:
    """Construct a Voyage client keyed from settings (SDK imported lazily; tests inject a mock)."""
    import voyageai  # lazy: tests mock the boundary and need not install the SDK

    return voyageai.Client(api_key=get_settings().voyage_api_key)


def _build_pinecone_index(settings: BenchmarkSettings) -> object:
    """Construct the Pinecone index handle keyed from settings (SDK imported lazily)."""
    from pinecone import Pinecone  # lazy: tests mock the boundary and need not install the SDK

    client = Pinecone(api_key=settings.pinecone_api_key)
    return client.Index(settings.pinecone_index_name)


def _build_anthropic_client() -> object:
    """Construct the Anthropic client keyed from settings (never a hardcoded secret)."""
    import anthropic

    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key)


def _embed_with_retry(
    voyage_client: object,
    texts: list[str],
    *,
    model: str,
    input_type: str,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[list[float]], int]:
    """Embed ``texts`` via Voyage with bounded exponential backoff on transient failures.

    Retries up to :data:`MAX_EMBED_ATTEMPTS` (a rate limit or blip should not fail the whole run),
    then re-raises so the caller records an error row rather than looping forever (Rule 12).

    Returns:
        ``(embeddings, total_tokens)`` — the vectors and the input tokens Voyage billed.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_EMBED_ATTEMPTS + 1):
        try:
            result = voyage_client.embed(texts, model=model, input_type=input_type)
            return list(result.embeddings), int(result.total_tokens)
        except Exception as exc:  # noqa: BLE001 — bounded retry then re-raise; nothing swallowed
            last_exc = exc
            logger.warning(
                "rag_embed_retry",
                attempt=attempt,
                max_attempts=MAX_EMBED_ATTEMPTS,
                error_message=str(exc),
                fix_suggestion="Transient Voyage failure (e.g. rate limit); backing off then retrying",
            )
            if attempt == MAX_EMBED_ATTEMPTS:
                break
            sleep(_EMBED_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))

    logger.error(
        "rag_embed_failed",
        max_attempts=MAX_EMBED_ATTEMPTS,
        error_message=str(last_exc),
        fix_suggestion="Check VOYAGE_API_KEY, network, and Voyage rate limits; query recorded as an error row",
    )
    assert last_exc is not None  # loop runs at least once, so a failure path always set this
    raise last_exc


def index_corpus(
    corpus_load: object,
    *,
    voyage_client: object | None = None,
    pinecone_index: object | None = None,
    settings: BenchmarkSettings | None = None,
) -> IndexResult:
    """Embed a corpus size's chunks and upsert them into its own Pinecone namespace.

    Runs once per size (not per question) so the corpus is embedded once and reused across all its
    questions. The namespace is keyed by ``corpus_load.total_token_count`` so a chunk from size A is
    never retrievable when querying size B. When the chunk count would exceed the free-tier vector
    cap, the ceiling is logged and flagged on the result (surfaced, not silently degraded).

    Args:
        corpus_load: A ``corpus_loader.CorpusLoad`` (``.text`` + ``.total_token_count``).
        voyage_client / pinecone_index: Injected in tests; real ones built from settings if omitted.
        settings: Benchmark settings (model id, chunk sizes); ``get_settings()`` if omitted.

    Returns:
        An ``IndexResult`` with the namespace, chunk/embedding-token counts, and the cap flag.
    """
    settings = settings or get_settings()
    voyage_client = voyage_client or _build_voyage_client()
    pinecone_index = pinecone_index or _build_pinecone_index(settings)

    corpus_token_count = corpus_load.total_token_count
    namespace = _namespace_for(corpus_token_count)
    chunks = chunk_text(
        corpus_load.text,
        chunk_size_chars=settings.chunk_size_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
    )

    vector_cap_exceeded = len(chunks) > PINECONE_STARTER_VECTOR_CAP
    if vector_cap_exceeded:
        logger.error(
            "rag_pinecone_vector_cap_exceeded",
            namespace=namespace,
            chunk_count=len(chunks),
            cap=PINECONE_STARTER_VECTOR_CAP,
            fix_suggestion="Corpus exceeds the free Starter cap (~300K vectors); upgrade to Standard or shrink size",
        )

    if not chunks:
        logger.info("rag_index_empty_corpus", namespace=namespace)
        return IndexResult(namespace=namespace, chunk_count=0, embedding_tokens=0, vector_cap_exceeded=False)

    embeddings, embedding_tokens = _embed_with_retry(
        voyage_client, chunks, model=settings.voyage_embedding_model, input_type="document"
    )
    vectors = [
        {"id": f"{namespace}-chunk-{index}", "values": vector, "metadata": {"text": chunk}}
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True))
    ]
    pinecone_index.upsert(vectors=vectors, namespace=namespace)

    logger.info(
        "rag_index_completed",
        namespace=namespace,
        chunk_count=len(chunks),
        embedding_tokens=embedding_tokens,
        vector_cap_exceeded=vector_cap_exceeded,
    )
    return IndexResult(
        namespace=namespace,
        chunk_count=len(chunks),
        embedding_tokens=embedding_tokens,
        vector_cap_exceeded=vector_cap_exceeded,
    )


def _matches(query_response: object) -> list[object]:
    """Extract the match list from a Pinecone query response (object- or dict-shaped)."""
    if hasattr(query_response, "matches"):
        return list(query_response.matches or [])
    if isinstance(query_response, dict):
        return list(query_response.get("matches") or [])
    return []


def _match_text(match: object) -> str:
    """Pull the stored chunk text out of one Pinecone match's metadata."""
    metadata = getattr(match, "metadata", None)
    if metadata is None and isinstance(match, dict):
        metadata = match.get("metadata")
    if not metadata:
        return ""
    return str(metadata.get("text", ""))


def retrieve_context(
    query: str,
    *,
    corpus_token_count: int,
    voyage_client: object,
    pinecone_index: object,
    settings: BenchmarkSettings,
) -> tuple[list[str], int]:
    """Embed the query and retrieve the top-k chunk texts from this size's namespace.

    Only the query is embedded here (the corpus was embedded once by ``index_corpus``), so cost
    scales with questions, not corpus size. A zero-hit retrieval returns an empty context — the
    caller still answers (and is graded), it is not a crash.

    Returns:
        ``(chunk_texts, query_embedding_tokens)`` — retrieved passages (top-k order) and the tokens
        Voyage billed for the query embedding (for the per-query cost).
    """
    namespace = _namespace_for(corpus_token_count)
    embeddings, query_tokens = _embed_with_retry(
        voyage_client, [query], model=settings.voyage_embedding_model, input_type="query"
    )
    response = pinecone_index.query(
        vector=embeddings[0],
        top_k=settings.retrieval_top_k,
        namespace=namespace,
        include_metadata=True,
    )
    chunk_texts = [text for text in (_match_text(match) for match in _matches(response)) if text]
    logger.info("rag_retrieved", namespace=namespace, hits=len(chunk_texts), query_tokens=query_tokens)
    return chunk_texts, query_tokens


def _generate_answer(
    anthropic_client: object, question_text: str, context_chunks: list[str], *, model: str
) -> tuple[str, int, int]:
    """Answer ``question_text`` from ONLY ``context_chunks`` with Sonnet 5.

    Returns:
        ``(answer_text, input_tokens, output_tokens)`` — the answer and usage for cost accounting.
    """
    context_block = (
        "\n\n".join(f"[Passage {index}]\n{chunk}" for index, chunk in enumerate(context_chunks, start=1))
        if context_chunks
        else "(no passages were retrieved)"
    )
    user_prompt = f"Context passages:\n{context_block}\n\nQuestion: {question_text}\n\nAnswer using only the passages above."
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=GENERATION_MAX_TOKENS,
        system=GENERATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    answer_text = "".join(
        getattr(block, "text", "") for block in (getattr(response, "content", None) or []) if getattr(block, "type", None) == "text"
    ).strip()
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return answer_text, input_tokens, output_tokens


def run_rag_query(
    question: BenchmarkQuestion,
    *,
    corpus_token_count: int,
    corpus_size: int,
    voyage_client: object | None = None,
    pinecone_index: object | None = None,
    anthropic_client: object | None = None,
    settings: BenchmarkSettings | None = None,
) -> ResultRow:
    """Run one question against an already-indexed corpus size and return a graded ``ResultRow``.

    Full per-query pipeline: embed query -> retrieve top-k from this size's namespace -> Sonnet 5
    answers from only those chunks -> judge vs gold -> assemble a ``measured`` row with
    ``latency_seconds`` and ``cost_usd`` (query-embedding + generation, priced by ``cost.py``).

    A query-level failure (embedding exhausted its retries, a transport error, …) is caught and
    returned as an error ``ResultRow`` with ``error`` populated — the sweep records the bad cell and
    continues; it does not halt (Rule 12). The corpus size must have been indexed first via
    ``index_corpus`` (same ``corpus_token_count``).

    Args:
        question: The graded question (carries gold answer, tier, id).
        corpus_token_count: The size's cumulative token count — the namespace key and sweep axis.
        corpus_size: Book/script count loaded at this size (ResultRow ordering detail).
        voyage_client / pinecone_index / anthropic_client: Injected in tests; built if omitted.
        settings: Benchmark settings; ``get_settings()`` if omitted.

    Returns:
        A ``ResultRow`` for system RAG at this size — graded on success, ``error``-populated on failure.
    """
    settings = settings or get_settings()

    started = time.perf_counter()
    try:
        voyage_client = voyage_client or _build_voyage_client()
        pinecone_index = pinecone_index or _build_pinecone_index(settings)
        anthropic_client = anthropic_client or _build_anthropic_client()

        context_chunks, query_tokens = retrieve_context(
            question.question_text,
            corpus_token_count=corpus_token_count,
            voyage_client=voyage_client,
            pinecone_index=pinecone_index,
            settings=settings,
        )
        answer_text, gen_input_tokens, gen_output_tokens = _generate_answer(
            anthropic_client, question.question_text, context_chunks, model=settings.rag_generation_model
        )
        verdict = grade_answer(answer_text, question.gold_answer, client=anthropic_client)

        cost_usd = cost.token_cost_usd(
            settings.voyage_embedding_model, input_tokens=query_tokens
        ) + cost.token_cost_usd(
            settings.rag_generation_model, input_tokens=gen_input_tokens, output_tokens=gen_output_tokens
        )
        latency_seconds = time.perf_counter() - started

        logger.info(
            "rag_query_completed",
            question_id=question.question_id,
            corpus_token_count=corpus_token_count,
            hits=len(context_chunks),
            score=verdict.score,
            cost_usd=cost_usd,
        )
        return ResultRow(
            system=SystemName.RAG,
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
            "rag_query_failed",
            question_id=question.question_id,
            corpus_token_count=corpus_token_count,
            error_message=str(exc),
            fix_suggestion="Inspect the RAG query error; cell recorded as an error ResultRow so the sweep continues",
        )
        return ResultRow(
            system=SystemName.RAG,
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


def run_rag_slice(
    question: BenchmarkQuestion,
    corpus_load: object,
    *,
    voyage_client: object | None = None,
    pinecone_index: object | None = None,
    anthropic_client: object | None = None,
    settings: BenchmarkSettings | None = None,
) -> ResultRow:
    """Convenience: index the corpus size then run one question (the demoable happy path).

    Wires ``index_corpus`` + ``run_rag_query`` for a single (size, question) — the sweep runner (#9)
    calls ``index_corpus`` once per size and ``run_rag_query`` per question instead.
    """
    settings = settings or get_settings()
    voyage_client = voyage_client or _build_voyage_client()
    pinecone_index = pinecone_index or _build_pinecone_index(settings)
    anthropic_client = anthropic_client or _build_anthropic_client()

    index_corpus(corpus_load, voyage_client=voyage_client, pinecone_index=pinecone_index, settings=settings)
    return run_rag_query(
        question,
        corpus_token_count=corpus_load.total_token_count,
        corpus_size=len(corpus_load.sources),
        voyage_client=voyage_client,
        pinecone_index=pinecone_index,
        anthropic_client=anthropic_client,
        settings=settings,
    )
