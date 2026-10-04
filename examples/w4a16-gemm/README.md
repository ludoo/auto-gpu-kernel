# w4a16-gemm — gemma's W4A16 linear made one class (sparkle task next.120)

Gemma 4 12B's six linear shapes on the DGX Spark: `y = x @ dequant(W).T`, x bf16 `[M, K]`, W the checkpoint's packed int4 `[N, K/8]` int32 (eight bias-8 nibbles per word, LSB-first along K) with bf16 group-32 scales `[N, K/32]`, y bf16 `[M, N]`. The engine runs each at M = 1 (decode), 4 (verify), 1–3 (eager verify) and up to 4096 (a prefill chunk), and requires a row's output bytes to be a function of the row and the weight alone. The wheel's Marlin is not: it rounds a row by the call's M (one class at M in [17, 64] ∪ [81, 127], others elsewhere). This target replaces it with a kernel that holds the invariant at Marlin's price at both ends — the DRAM rate at M ≤ 4, the tensor pipe at the chunk.

- `w4a16_gemm.py` — the op: `w4a16_gemm(x, packed, scale, out)`. The starting point is the slowest kernel that holds the contract by construction (one program per 16×64 tile, 32-wide K leaves in ascending order, the dequant in registers, one fp32 accumulator); only this file changes.
- `shapes.py`, `data.py` — the six shapes, the row counts, seeded weights and activations, the fp32 dequant.
- `reference.py` — the fp32 reference (the correctness side) and the wheel through sparkle's shim (the bench baseline and the correctness comparator; not the contract).
- `test_w4a16.py` — invariance bitwise over the engine's row counts and the wheel's class edges at three offsets, the whole in 17- and 64-row pieces, correctness within twice the wheel's rel-L2 from fp32, determinism, CUDA-graph replay with x rewritten, no torch matmul on the path.
- `bench.py` — `gemm_score`: the geomean over 18 cells (six shapes × M in {1, 4, 4096}) of per-call µs over the wheel's, cold rotated weights; 1.0 is par, lower is better.
