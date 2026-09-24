# qsa-attn — the QSA sparse attention kernel, M-pinned (sparkle task next.75)

Qwen3.8-Flash-Next's query-sparse attention on the DGX Spark: 12 layers, each attending 24 query heads over 2 KV heads at head_dim 256 across 2051 selected tokens gathered from a paged bf16 cache. The serving engine pins every phase to one kernel configuration (BLOCK_N 32, one split, one warp) so a token's attention bytes do not depend on whether decode, verify or prefill computed them. That configuration is a 2-program grid at decode (M=1) and a 10-program grid at verify (M=5) on a 48-SM part: ~107 and ~130 µs a layer where the same kernel's split-k rows run 30 and 41 — the whole price of the pin, ~1 ms per served step.

- `qsa_attn.py` — the op: `qsa_attention(q, k_cache, v_cache, packed, block_table, token_to_req, out)`. The baseline is the shipped kernel; only this file changes.
- `reference.py` — the shipped kernel, verbatim, in the served configuration. Read-only; the bytes every candidate must reproduce.
- `shapes.py`, `data.py` — the served geometry and seeded inputs on a 131k paged cache with a permuted block table.
- `test_attn.py` — bitwise equality against the reference at M=1, 5 and 8192 (three seeds at the chunk: a wrong kernel shows in ~1 element per 250,000 of bf16 output, so small shapes prove nothing), short rows, four requests, determinism, CUDA-graph capture and replay.
- `bench.py` — `attn_score`: the geometric mean over the three shapes of per-call µs against the shipped kernel's; 1.0 is baseline, lower is better.

The contract is exact and the arithmetic is fixed: per output element, the same mma accumulate chain in tile order, the same exp2, the same reductions. What is free is everything around it — how many programs and warps share a row's work, how the gather is issued, what is resident where. See the hints in `configs/qsa_attn.toml` for what was measured before the run.
