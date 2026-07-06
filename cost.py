"""Shared, path-agnostic API-equivalent cost model for the benchmark harness.

Every path (RAG #6, plain-LLM #7, wiki #8) reports **API-equivalent** cost so the
cost gap — the thesis — is computed identically in one place, never re-derived
inline per path (PRD "Cost accounting lives in one place"). This module is a pure
pricing table plus pure functions over usage/token counts: no network, no clients,
no path-specific logic. A path assembles its ``ResultRow.cost_usd`` by summing the
per-call costs this module returns (e.g. RAG = query-embedding + generation).

Pricing sources (USD per 1M tokens), captured 2026-07-05 — update when a provider
changes list pricing:
- Anthropic ``claude-sonnet-5`` (RAG/baseline generation): $3.00 in / $15.00 out —
  standard list price (an intro $2.00/$10.00 runs through 2026-08-31; we report the
  durable sticker so the reported cost does not silently drop when the intro ends).
- Anthropic ``claude-opus-4-8`` (accuracy judge): $5.00 in / $25.00 out.
- Voyage ``voyage-3`` (RAG embeddings): $0.06 per 1M tokens (embedding-only; no
  output tokens) — Voyage AI list pricing; matches the "~$30 one-time" sweep estimate
  in ``reference/integrations.md``.
"""

from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

# Reason: one place converts the human-readable "$/1M tokens" list price into a
# per-token multiplier, so the sticker prices above read exactly as documented.
_TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """List price for one model, in USD per 1M tokens (output 0.0 for embedding models)."""

    input_usd_per_million: float
    output_usd_per_million: float


# The single pricing table. Keys are the exact model ids used across the harness
# (``settings.py`` defaults). Path-agnostic: any path prices any of these models.
PRICING: dict[str, ModelPrice] = {
    "claude-sonnet-5": ModelPrice(input_usd_per_million=3.00, output_usd_per_million=15.00),
    "claude-opus-4-8": ModelPrice(input_usd_per_million=5.00, output_usd_per_million=25.00),
    "voyage-3": ModelPrice(input_usd_per_million=0.06, output_usd_per_million=0.0),
}


def token_cost_usd(model: str, *, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Compute API-equivalent cost in USD for one call's token usage.

    Pure and path-agnostic: looks the model up in :data:`PRICING` and applies its
    per-token rates. Embedding calls pass only ``input_tokens`` (their model prices
    output at $0.0). An unknown model is a configuration bug, not a silent $0.0 —
    it raises loudly (Rule 12) so a mispriced cell surfaces rather than under-reporting
    the cost gap.

    Args:
        model: Exact model id (e.g. ``"claude-sonnet-5"``, ``"voyage-3"``).
        input_tokens: Prompt/embedding input tokens billed at the model's input rate.
        output_tokens: Generated output tokens billed at the model's output rate.

    Returns:
        The call cost in USD (``0.0`` only when both token counts are 0).

    Raises:
        ValueError: If ``model`` is not in :data:`PRICING`, or a token count is negative.

    Example:
        >>> round(token_cost_usd("claude-sonnet-5", input_tokens=1_000_000, output_tokens=0), 2)
        3.0
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(f"token counts must be non-negative, got input={input_tokens}, output={output_tokens}")

    price = PRICING.get(model)
    if price is None:
        logger.error(
            "cost_unknown_model",
            model=model,
            known_models=sorted(PRICING),
            fix_suggestion="Add the model's list price to cost.PRICING before pricing calls against it",
        )
        raise ValueError(f"no pricing for model {model!r}; add it to cost.PRICING")

    input_cost = input_tokens / _TOKENS_PER_MILLION * price.input_usd_per_million
    output_cost = output_tokens / _TOKENS_PER_MILLION * price.output_usd_per_million
    return input_cost + output_cost
