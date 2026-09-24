# gdn-rec — the GDN recurrence kernel in every phase (sparkle task D.4)

Qwen3.8-Flash-Next's gated-delta-net recurrence on the DGX Spark: 36 layers, each carrying a 3 MiB fp32 state `[48, 128, 128]` through one sequential recurrence per row. The serving engine runs one kernel and one arithmetic in every phase — a decode step (T=1), the served verify pass (T=5, the state after every row written for rollback), the prefill chunk (T=8192) and a prompt's final chunk with the residency pin (one extra state write mid-chunk) — so a row's bytes do not depend on which phase computed it. The kernel is sparkle's own CUDA (`load_inline`), one warp per (V head, 16-row slice), the state in registers.

- `gdn_rec.py` — the op: `gdn_recurrence(q, k, v, g, beta, h0, *, states=False, pin_at=None, out=None) -> RecurrenceOut`. The baseline is the shipped kernel; only this file changes.
- `reference.py` — the shipped kernel, verbatim (the extension name changed so the two compile side by side). Read-only; the bytes every candidate must reproduce on `o`, `ht`, `states` and `pin`.
- `shapes.py`, `data.py` — the served geometry, seeded inputs with the served distributions, and the served model's captured kernel inputs from sparkle's `vectors/gdn-recurrence-v1`.
- `test_rec.py` — bitwise equality against the reference at the four served launches (three seeds at the chunk), a small-T volume over T=1..8 × n=1..4 with and without per-row states, pins at several positions, the captured vectors, T rows in one launch against T single-row launches, determinism, CUDA-graph capture and replay.
- `bench.py` — `rec_score`: the geometric mean over the four shapes of per-call µs against the shipped kernel's; 1.0 is baseline, lower is better.

The contract is exact and the arithmetic is fixed: per row, per state element, the same fp32 operation sequence, the same reduction tree. What is free is everything around it — the launch geometry, what is staged where, how the state is read and written, how the per-row writes overlap the next row. See the hints in `configs/gdn_rec.toml` for what was measured before the run.
