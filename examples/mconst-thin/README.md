# mconst-thin — the M-invariant linear's thin tile at decode (sparkle task next.89)

Qwen3.8-Flash-Next's hyper-connection `down_inject` linear, `[336, 10240]` bf16, runs 96 times per decode or verify step through sparkle's own `mconst_linear` kernel rather than cuBLAS: cuBLAS picks its bf16 GEMM algorithm by the row count M, the algorithms round differently, and a row's bytes then depend on which phase computed it. The kernel's reduction is a `tl.dot` chain over 64-wide K chunks in order into one fp32 accumulator, so a row's bytes are a function of (N, K) alone; the tile around that chain is free and chosen by phase. The `thin` tile (BM 16, BN 16, 4 warps, 4 stages) is the decode one.

Cold — 96 distinct weights streamed from DRAM once each, as the step does it — the thin tile runs at 47.4 µs per call (145 GB/s), cuBLAS at 36–37 (190 GB/s), the weight's bytes at the DRAM rate 29.0. The gap is ~1.1 ms per ~100 ms step.

- `mconst_thin.py` — the candidate: `linear(x, w, *, tile='wide'|'thin')`. The baseline is the shipped kernel; only this file changes.
- `reference.py` — the shipped kernel, verbatim. Read-only; the bytes every candidate must reproduce.
- `shapes.py` — the two served shapes and the row counts.
- `test_thin.py` — bitwise equality against the reference on both shapes at M=1..8, 16, 63, 4096, 8192, under both tile names; row invariance; determinism; CUDA-graph capture and replay.
- `bench.py` — `thin_score`: the geometric mean over M=1 and M=5 of the cold 96-call chain's per-call µs against the shipped kernel's; 1.0 is baseline, lower is better.
