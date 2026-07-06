"""Tests for the shared, path-agnostic cost model in cost.py.

Each test encodes *why* the behaviour matters (Rule 9): the cost gap is the whole
thesis, so a mispriced or silently-zeroed cell would corrupt the finding. These tests
pin the documented list prices, prove the function is pure arithmetic over token
counts (reusable by every path), and prove an unknown model fails loud rather than
under-reporting cost. No network, no clients — cost.py has no external boundary.
"""

import pytest

from cost import PRICING, token_cost_usd


def test_generation_cost_matches_documented_sonnet_5_list_price() -> None:
    # Why (Rule 9): the reported RAG/baseline cost must equal the documented Sonnet 5
    # list price ($3/1M in, $15/1M out) exactly — a drift here silently mis-states the
    # cost gap the whole project exists to measure.
    cost = token_cost_usd("claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(3.00 + 15.00)


def test_embedding_cost_prices_input_only_at_voyage_rate() -> None:
    # Why: embeddings have no output tokens; the RAG per-query cost must price the
    # query-embedding at Voyage's $0.06/1M input rate and add $0 for output.
    cost = token_cost_usd("voyage-3", input_tokens=2_000_000)
    assert cost == pytest.approx(2 * 0.06)


def test_zero_tokens_is_zero_cost() -> None:
    # Why: an unmeasured/empty call must cost exactly 0.0 (matches ResultRow's cost_usd
    # default) — not a spurious minimum charge.
    assert token_cost_usd("claude-sonnet-5") == 0.0


def test_cost_is_linear_and_path_agnostic_across_models() -> None:
    # Why: cost.py is a SHARED contract — the same function must price the judge model
    # (Opus) the same way it prices generation, so #7/#8 can reuse it unchanged. Prove
    # linearity so summing per-call costs in a path is valid.
    single = token_cost_usd("claude-opus-4-8", input_tokens=500, output_tokens=100)
    double = token_cost_usd("claude-opus-4-8", input_tokens=1000, output_tokens=200)
    assert double == pytest.approx(2 * single)


def test_unknown_model_raises_rather_than_returning_zero() -> None:
    # Why (Rule 12): a missing price is a config bug. Returning $0.0 would silently
    # under-report the cost gap; the function must surface it loudly instead.
    with pytest.raises(ValueError, match="no pricing for model"):
        token_cost_usd("gpt-4o", input_tokens=100)


def test_negative_token_count_is_rejected() -> None:
    # Why: a negative usage count is never legitimate; accepting it would produce a
    # negative cost that quietly offsets real spend in an aggregate.
    with pytest.raises(ValueError, match="non-negative"):
        token_cost_usd("claude-sonnet-5", input_tokens=-1)


def test_pricing_table_covers_the_models_the_rag_path_uses() -> None:
    # Why: the RAG path prices voyage-3 (query embed) and claude-sonnet-5 (generation);
    # a regression removing either from the table would break every RAG cost cell.
    assert {"voyage-3", "claude-sonnet-5", "claude-opus-4-8"} <= set(PRICING)
