"""Gemma 4 12B's six linear shapes on the DGX Spark, one call each
(sparkle task next.120, the Marlin target).

`y = x @ dequant(W).T`: x bf16 `[M, K]`, W packed int4 `[N, K/8]` int32
(eight bias-8 nibbles per word, LSB-first along K — the checkpoint's
compressed-tensors pack, consumed as-is), bf16 per-group scales
`[N, K/32]` (`GROUP = 32`), y bf16 `[M, N]`. The engine runs every
linear at M = 1 (decode), 4 (verify), 1-3 (eager verify) and up to
4096 (a prefill chunk), and requires a row's output bytes to be a
function of the row and the weight alone — never of M or of the row's
position in the call.
"""

GROUP = 32

# name -> (N, K, calls per layer)
SHAPES = {
    'q_proj': (4096, 3840, 1),
    'k_proj_sliding': (2048, 3840, 2),
    'k_proj_global': (512, 3840, 1),
    'o_proj': (3840, 4096, 1),
    'gate_up': (15360, 3840, 2),
    'down': (3840, 15360, 1),
}

VALIDATE_ROWS = 4096  # the whole: every row computed in one call
ARM_MS = (1, 2, 3, 4, 17, 64, 65, 128, 1536)  # the engine's and the wheel's class edges
ARM_OFFSETS = (0, 700, 1025)  # where a piece starts inside the whole
BENCH_MS = (1, 4)  # this round measures the decode end only; the chunk tile is frozen
QUICK_MS = (1, 4)
