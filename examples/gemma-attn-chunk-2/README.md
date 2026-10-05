# gemma-attn-chunk-2 — gemma's attention at the prefill chunk, round 2 (sparkle task next.130)

Round 1 ([gemma-attn-chunk](../gemma-attn-chunk/README.md), DeepSeek flash, 12 iterations) took the global chunk −4.85% on the paired A/B — a fixed block-grain fold on the global kind and an L2-residency rule for the pipeline depth — and then measured every tuning axis of the kernel's structure closed. Its diagnostics located where the remaining 3× to the tensor pipe sits: one resident CTA per SM, pinned by registers (248 per thread) and shared memory (100 KB) at once, ~2 warps per scheduler, 15% of issue slots used; the same kernel with every K/V address forced L1-resident runs the 8k cell in 8.3 ms against 24. This round asks for a structure, not a tuning.

- `gemma_attn.py` — the op, round 1's survivor verbatim (`work/gemma-attn-chunk-dsflash/repo` at `d5a4aed`). The start, at 1.0. Only this file changes.
- `shapes.py`, `data.py`, `reference.py`, `test_attn.py`, `bench.py` — as round 1's, the baselines re-derived on the survivor. `test_attn.py` is sparkle's own gate with the imports pointed here; `chunk_score` is the geomean over the global chunk at kv 8k / 32k / 60k over the start's, times the geomean over the nine held cells of `max(1, ratio)`.

The start on this box (2026-10-06, cold rotated sets, `sparkle.service` stopped), the wheel beside it:

| cell | start µs | wheel µs | FLOP floor at 126 TFLOPS | start TFLOPS |
|---|---|---|---|---|
| global chunk 8k | 23822 | 22660 | 6.5 ms | 35 |
| global chunk 32k | 125959 | 122787 | 32.7 ms | 33 |
| global chunk 60k | 244556 | 236685 | 63.3 ms | 33 |
| global decode 4k / 32k / 128k | 49.3 / 325.4 / 1329 | 51.0 / 308.1 / 1155 | — | at the DRAM rate |
| global verify 4k / 32k / 128k | 51.7 / 591.2 / 2395 | 74.8 / 600.0 / 2284 | — | |
| sliding decode / verify / chunk 32k | 45.3 / 45.4 / 1579 | 46.7 / 48.3 / 1126 | — | |

Bench noise on the scored cells is ~1–2% run to run; the held 4k cells swing ±5%.
