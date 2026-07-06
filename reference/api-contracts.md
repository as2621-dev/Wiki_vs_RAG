# Data Contracts

**Why this doc exists:** the engine's output CSV/JSON is the seam between Python and the Next.js UI,
and between one sweep run and the next. Both sides must agree on the shape.
**Update when:** `ResultRow`, `JudgeVerdict`, or the UI results JSON changes.

## Sweep axis

The true axis is **cumulative token count**, targets:
`100k → 500k → 1M → 2M → 5M → 10M → 20M → 50M` (measured), extrapolated to **1B**.
`corpus_size` (book/script count) is retained only as an internal fixed-ordering detail.

## `ResultRow` (one graded cell = one CSV line)

Existing fields (`models.py`): `system`, `corpus_size`, `question_id`, `tier`, `latency_seconds`,
`cost_usd`, `accuracy` (0.0|0.5|1.0), `answer_text`, `judge_rationale`, `error`.

**Add for this PRD:**
- `corpus_token_count: int` — the real sweep axis for this row.
- `measured_or_extrapolated: Literal["measured", "extrapolated"]` — never hide the boundary (Rule 12).
- `skipped_reason: str = ""` — e.g. `"exceeds_context_window"` for the plain-LLM wall row.

`SystemName` gains `PLAIN_LLM` alongside `RAG`, `WIKI`.

## `JudgeVerdict` (structured judge output)

`score: Literal[0.0, 0.5, 1.0]` + `rationale: str` (one sentence). Graded strictly against the gold
answer, not the judge's own knowledge.

## UI results JSON (pre-baked, read-only)

The UI reads a single `results.json` derived from the sweep + `analysis.py`. Shape:

```json
{
  "generated_note": "measured to 50M tokens; 100M–1B extrapolated",
  "series": {
    "accuracy": [
      {"corpus_token_count": 1000000, "system": "plain_llm", "accuracy": 0.94, "kind": "measured"},
      {"corpus_token_count": 2000000, "system": "plain_llm", "accuracy": null,  "kind": "skipped_context"},
      {"corpus_token_count": 50000000, "system": "rag", "accuracy": 0.61, "kind": "measured"},
      {"corpus_token_count": 1000000000, "system": "wiki", "accuracy": 0.80, "kind": "extrapolated"}
    ],
    "cost": [
      {"corpus_token_count": 50000000, "system": "rag",  "total_cost_usd": 0.75, "kind": "measured"},
      {"corpus_token_count": 50000000, "system": "wiki", "total_cost_usd": 100.0, "kind": "extrapolated"}
    ]
  },
  "answers": {
    "<question_id>": {
      "<corpus_token_count>": {
        "plain_llm": {"answer_text": "...", "accuracy": 1.0, "kind": "measured"},
        "rag":       {"answer_text": "...", "accuracy": 0.5, "kind": "measured"},
        "wiki":      {"answer_text": "...", "accuracy": 1.0, "kind": "measured"}
      }
    }
  }
}
```

- `kind` drives the visual measured-vs-extrapolated distinction (solid vs dashed).
- `accuracy: null` + `kind: "skipped_context"` → UI shows "exceeded context window", not a blank.

## Error shape (logs)

Structured JSON, `snake_case` event, `fix_suggestion` on every error. A failed cell is an error
`ResultRow` (populated `error`), not a dropped row.
