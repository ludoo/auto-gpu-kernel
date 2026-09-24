"""The served shape: Qwen3.8-Flash-Next's query-sparse attention on the
DGX Spark, one layer's call.

Per QSA layer: 24 query heads over 2 KV heads (group 12, BLOCK_M 16),
head_dim 256, bf16 throughout. The selection is a packed int32 buffer
`[rows, WIDTH + 1]`: WIDTH = budget 2048 + compress_ratio 4 - 1 = 2051
sequence-relative token indices (`-1` past a row's valid count) plus a
trailing count column — never a token index — that the kernel reads as
its tile-loop bound. The K/V caches are paged `[pages, 16, 2, 256]`
bf16; `block_table` int32 `[requests, n_pages]` maps a row's logical
page to a physical one, `token_to_req` int32 `[rows]` names the row's
request.

The three row counts the engine runs the kernel at, all in the same
configuration (the M pin): M=1 (a plain decode step), M=5 (the served
verify pass, draft depth 4), M=8192 (the prefill chunk). The measured
context is 131072 tokens: 8192 pages, a permuted block table.
"""

HEADS = 24
KV_HEADS = 2
GROUP = HEADS // KV_HEADS  # 12
HEAD_DIM = 256
PAGE = 16
BUDGET = 2048
RATIO = 4
WIDTH = BUDGET + RATIO - 1  # 2051 selection columns; column WIDTH is the count
TILE = 32  # the served BLOCK_N: cdiv(2051, 32) = 65 tiles per row
CONTEXT = 131072
PAGES = CONTEXT // PAGE  # 8192

# (name, rows): the served row counts.
SHAPES = [
    ("m1", 1),
    ("m5", 5),
    ("chunk", 8192),
]
