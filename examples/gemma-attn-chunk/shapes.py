"""Gemma 4 12B's two attention geometries on the DGX Spark, one layer's
call (sparkle task next.120).

Both kinds: bf16 q `[rows, 16, head_dim]`, bf16 paged K/V
`[pages, 2, 16, kv_heads, head_dim]` (page 16 tokens, index 0 K, 1 V),
`sm_scale = 1.0`, causal, bf16 out `[rows, 16, head_dim]`. Positions
are absolute and contiguous within one sequence's rows; a call carries
the LAST `rows` positions of each sequence.

  sliding (40 layers): 16 q heads over 8 kv heads (group 2), head_dim
    256, a window of 1024 keys including self (`WINDOW_LEFT = 1023`,
    inclusive: a key at distance <= 1023 is attended). The K/V the
    call sees is the gathered window — pages from
    `first_page = max(0, total - rows - 1024) // 16` to the current
    page — so key index 0 of the gather is absolute position
    `first_page * 16` (`pos_base`), a page boundary at or before the
    window start.
  global (8 layers): 16 q heads over 1 kv head (group 16), head_dim
    512, no window; the K/V is the sequence from position 0
    (`pos_base = 0`).

The engine runs each kind's kernel at rows = 1 (decode), 4 (verify,
draft depth 3), 1-3 (eager verify) and up to 4096 (a prefill chunk),
with n >= 1 sequences per launch (FlashInfer-shaped metadata).

The call's metadata, all int32 device tensors (CUDA-graph replay
rewrites them between replays, so the kernel reads them on device):
  qo_indptr        [n + 1]  row spans per sequence
  kv_indptr        [n + 1]  page spans per sequence into kv_indices
  kv_indices       [>= kv_indptr[n]]  physical page ids
  kv_last_page_len [n]      tokens used in each sequence's last page
  pos_base         [n]      absolute position of each gather's key 0
A sequence's key count is `(pages - 1) * 16 + last_page_len`; row `i`
of its span sits at absolute position `pos_base + kv_len - rows + i`.
"""

PAGE = 16
HEADS = 16
WINDOW = 1024

KINDS = {
    'sliding': dict(kv_heads=8, head_dim=256, window_left=WINDOW - 1),
    'global': dict(kv_heads=1, head_dim=512, window_left=-1),
}

# The served row counts and contexts the bench prices, per kind.
# (phase, rows, context): the call is the last `rows` positions of a
# `context`-long sequence. The sliding kind's bytes are
# context-independent, so it is priced at one context.
#
# SCORED cells: the global kind's 4096-row chunk at context 8k (the
# served chunk 1: 4096 of prompt behind it), 32k (the mean chunk of a
# 65k prompt) and 60k (its last full chunk). HELD cells: every other
# cell enters the score only when it is slower than the starting
# kernel — decode, verify and the sliding kind keep their price or
# better, they cannot buy anything.
BENCH_CELLS = {
    'sliding': [('decode', 1, 32768), ('verify', 4, 32768),
                ('chunk', 4096, 32768)],
    'global': [('decode', 1, 4096), ('decode', 1, 32768),
               ('decode', 1, 131072), ('verify', 4, 4096),
               ('verify', 4, 32768), ('verify', 4, 131072),
               ('chunk', 4096, 8192), ('chunk', 4096, 32768),
               ('chunk', 4096, 61440)],
}
SCORED = ('global/chunk/8192', 'global/chunk/32768', 'global/chunk/61440')
QUICK_PHASES = ('decode', 'verify')

# The validate sequence: 1600 positions per kind, through the window
# and past it; the long-context samples check the last 64 rows.
VALIDATE_LEN = 1600
LONG_CONTEXTS = (32768, 131072)
LONG_TAIL = 64
