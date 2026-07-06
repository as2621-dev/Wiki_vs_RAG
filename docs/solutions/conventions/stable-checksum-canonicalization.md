---
title: Stable checksum over a frozen data set (golden question bank)
tags: [checksum, hashlib, determinism, golden-set, questions]
problem_type: convention
symptoms: A stored checksum over in-code data must pass on the untouched set and fail on any edit, and stay identical across runs / Python versions / machines.
date: 2026-07-05
---

For the frozen golden set in `questions.py` (`QUESTION_BANK_CHECKSUM`,
`compute_question_bank_checksum`, `verify_question_bank_checksum`):

- Use `hashlib.sha256`, **never** Python's builtin `hash()` — it is salted per process
  (`PYTHONHASHSEED`), so a stored digest would not reproduce.
- Canonicalize deterministically: build one dict per record with an **explicit** field tuple
  (`question_id, book_key, tier, question_text, gold_answer`), **sort records by a stable key**
  (`question_id`) so list reordering is a no-op, then `json.dumps(..., sort_keys=True,
  ensure_ascii=True, separators=(",", ":"))`. `ensure_ascii=True` matters — gold answers contain
  non-ASCII (Dantès, Misérables); escaping keeps the digest identical across locales.
- Serialize enums by `.value`, not `repr`.
- Let `compute_*` take the bank as a parameter (default the module constant) so the tamper test
  hashes a modified copy without mutating the frozen bank.
- Regenerate the constant deliberately and review the diff; a mismatch is the alarm, not a bug.

Composition note: `questions_for_token_target(loaded_source_keys)` takes the corpus's loaded
anchor keys **injected** (from `corpus.GUTENBERG_BOOKS` order), never hard-coded to a book count —
the corpus loader owns the token→sources decision.

Env note: the `.venv` (py3.14) ships only pydantic/pytest/ruff. Any test importing `corpus.py`
pulls `httpx`/`structlog`/`pydantic-settings` (module-top imports) — install them before running.
