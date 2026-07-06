---
title: Enforced structured output from Claude + hermetic boundary mocking (LLM judge/generation)
tags: [anthropic, claude, structured-output, tool-use, pytest, mocking, pydantic, settings]
problem_type: convention
symptoms: An LLM call must return a validated Pydantic model (not regexed free text), and its tests must never hit the real API or require real API keys in the env.
date: 2026-07-05
---

For any module that calls Claude and must return a validated shape (`judge.py`;
the same pattern fits the RAG/plain-LLM/wiki generation slices):

## Enforce structured output — don't parse free text
- Define a **strict** tool and **force** it: `tool={"name":..., "strict": True,
  "input_schema": {..., "additionalProperties": False, "required": [...]}}` plus
  `tool_choice={"type": "tool", "name": <tool-name>}`. The verdict/answer arrives
  as a `tool_use` block whose `.input` is already-parsed JSON.
- Validate at the boundary: `Model(**block.input)`. On no-tool-use / validation
  failure, return `None` and let the caller retry-then-fall-back — never regex.
- `Literal[0.0, 0.5, 1.0]` in a Pydantic model: the SDK delivers `input` as parsed
  JSON numbers, and Pydantic coerces int `1` → `1.0`. Only strings are rejected, so
  there is no latent false-fallback from an integer score.

## Determinism on `claude-opus-4-8`
- **Do not send `temperature`** — the sampling params are removed on the Opus
  4.7/4.8 line and return a 400. Determinism rests on a module-level constant
  prompt + the enforced strict tool schema, not a temperature knob.
- Keep the model id in `settings` (`judge_model` default `claude-opus-4-8`); read
  the key via `get_settings().anthropic_api_key` — never hardcode or log it.

## Fail-loud vs fall-back (grade/answer functions)
- Empty/whitespace input → short-circuit to the zero verdict with **no API call**.
- Refusal (`stop_reason == "refusal"`) or unparseable output → retry once, then
  fall back to a rationalised zero (the judge couldn't decide; conservative).
- Genuine transport/API `Exception` → retry once, then **re-raise** so the caller
  records an error row (Rule 12). Do not turn an outage into accuracy 0.0.

## Hermetic tests (mock at the boundary)
- Inject the client: `grade_answer(..., *, client=None)`; tests pass a `Mock()` and
  assert on `client.messages.create.call_args` / `.call_count`.
- Fake responses with `types.SimpleNamespace` mirroring the SDK: a response has
  `.stop_reason` and `.content` (list of blocks); a tool_use block has
  `.type == "tool_use"`, `.name`, `.input` (dict).
- The module reads only `judge_model` from settings, but `BenchmarkSettings`
  **requires all three provider keys**. Stub the config boundary so tests need no
  `.env`: an autouse fixture `monkeypatch.setattr(mod, "get_settings", lambda:
  SimpleNamespace(judge_model="claude-opus-4-8"))`. Do not instantiate real settings
  in tests.
