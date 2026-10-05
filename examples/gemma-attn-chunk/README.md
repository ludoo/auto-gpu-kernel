# gemma-attn-chunk — gemma's attention kernel at the prefill chunk (sparkle task next.130)

The second loop on gemma's paged attention. The first ([gemma-attn](../gemma-attn/README.md), sparkle next.120) made the kernel one class — a row's bytes a function of the row alone, bitwise across decode, verify and the chunk — at the wheel's price on a 12-cell geomean. Sparkle's per-chunk profile then found the global kind's chunk is the only bucket of a prefill chunk that grows with context: 284 ms of a 4096-token chunk at 4k context, 2155 ms at 57k, ~31 TFLOPS against the part's 126 dense bf16 peak, 40% of the mean chunk and 59% at the end of a 65k prompt. This target is that cell alone, with every other cell held.

- `gemma_attn.py` — the op, sparkle's `kernels/gemma_attention.py` verbatim (its docstring is the design: the two-level split-invariant fold in absolute position space, the exact no-ops, the image block mask). The start, at 1.0. Only this file changes.
- `shapes.py`, `data.py` — the two geometries, the metadata contract, seeded sequences on a permuted paged cache.
- `reference.py` — the fp32 reference (the correctness side) and the wheel (the correctness comparator, printed beside each cell for the record; not the contract, not the score).
- `test_attn.py` — sparkle's own gate `tests/test_gemma_attention_gpu.py` with the imports pointed here: invariance bitwise over every arm the engine runs, correctness within twice the wheel's rel-L2 from fp32, the image-span contract on the sliding kind, determinism, CUDA-graph replay with rewritten metadata, no torch attention on the path. 45 tests, 7 skips.
- `bench.py` — `chunk_score`: the geomean over the three scored cells (the global chunk at kv 8k / 32k / 60k) of µs over the start's, times the geomean over the nine held cells of `max(1, µs over the start's)`. The start is 1.0; a held cell can only hurt.

The start on this box (2026-10-06, cold rotated sets, `sparkle.service` stopped), the wheel beside it:

| cell | start µs | wheel µs | FLOP floor at 126 TFLOPS | start TFLOPS |
|---|---|---|---|---|
| global chunk 8k | 25704 | 22660 | 6.5 ms | 32 |
| global chunk 32k | 128255 | 122787 | 32.7 ms | 32 |
| global chunk 60k | 250726 | 236685 | 63.3 ms | 32 |
| global decode 4k / 32k / 128k | 50.6 / 325.6 / 1323 | 51.0 / 308.1 / 1155 | — | at the DRAM rate |
| global verify 4k / 32k / 128k | 53.0 / 587.6 / 2394 | 74.8 / 600.0 / 2284 | — | |
| sliding decode / verify / chunk 32k | 45.1 / 45.5 / 1569 | 46.7 / 48.3 / 1126 | — | |

Bench noise on the scored cells is ~1% run to run.
