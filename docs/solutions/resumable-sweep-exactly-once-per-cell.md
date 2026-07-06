# Resumable sweep: exactly-once-per-cell via append-only JSONL + explicit cell keys

**Context:** the sweep runner (`sweep.py`, issue #9) orchestrates every `(size × system × question)`
cell into a `ResultRow` over a multi-day headless run that WILL be killed mid-way. It must resume from
the last completed cell, never re-run (re-bill) a done cell, never double-count a row, and never halt
on one flaky cell.

## The pattern

1. **Store the cell key explicitly next to the row** — each progress line is
   `{"cell_key": "<size>:<system>:<question_id>", "row": {<ResultRow model_dump>}}`, appended the
   instant a cell finishes (`write` + `flush` + `os.fsync`). Append-only ⇒ every prior line is durable;
   a kill leaves at most a torn *final* line.
2. **Derive done-cells on replay, don't recompute them.** On startup replay the JSONL into a
   `set[str]` of completed cell keys + the rows in store order. A cell whose key is in the set is
   **skipped — the path callable is never called again**. A torn final line is skipped *loudly*
   (`sweep_progress_line_skipped`) instead of aborting the whole resume; a duplicate cell key keeps the
   first occurrence (dedup guard ⇒ no double-count).
3. **Key on the INPUT, not a derived value.** The cell key uses the sweep *token target* `size`, not
   the corpus's derived `corpus_token_count` — because two targets can cap to the same available-corpus
   token count (shortfall), and a derived key would let one cell overwrite another's resume slot.
   Storing the key explicitly (step 1) also means replay never has to re-derive it from the row.
4. **Per-cell failure isolation is at the runner, not just the path.** Wrap each injected path call in
   `try/except Exception`: a raise becomes an `error` `ResultRow` (real axis values from the loaded
   corpus) and the loop continues. `BaseException` (e.g. `KeyboardInterrupt` = a real kill) is *not*
   caught, so it propagates and the run stops cleanly with the store intact to resume from.

## Why not reuse #8's checkpoint verbatim

#8 (`wiki.py`) checkpoints a **rewritten** JSON blob of completed *source keys* per size (temp +
`os.replace`). That fits a small, bounded, mutated set. A sweep produces thousands of independent rows
streamed over days — an **append-only log** is the better fit: O(1) crash-safe append per cell, no
read-modify-rewrite, and the rows themselves are the payload. Both share the same durability spine
(fsync / atomic) — reuse the *spine*, pick the *shape* per workload. Final `results.csv`/`results.json`
still use #8's temp-+-`os.replace` atomic write (whole-file, written once at the end).

## Testing (all paths injected — CLAUDE.md §6)

Simulate a kill by having a fake path raise `KeyboardInterrupt` mid-sweep, assert the store holds the
pre-crash cells, then call `run_sweep` again on the same `out_dir` and assert the path fn is called
**only** for the not-yet-done cells (`second_run_calls == ["q2", "q3"]`). Hand-craft a torn final line
and a duplicated line to test the two replay guards directly.
