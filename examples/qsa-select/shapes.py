"""The real shapes: one prefill chunk of T=4096 query rows against the
compressed-column widths of the serving contexts of record.

cols = the score matrix width: cdiv(context/4, 64) * 64 (4 tokens per
compressed group, width rounded up to 64). k = 512 selected groups
(the 2048-token budget / 4). `visible(base, rows)` is each query row's
visible-group count: row i at absolute position base+i sees
(base+i)//4 complete groups — a per-row prefix bound, monotonically
non-decreasing down the chunk.
"""

import torch

K = 512
T = 4096

# (name, base_position, cols): the last chunk of a 66k prefill and of a
# 180k prefill — the two contexts the sparkle task measures TTFT at.
SHAPES = [
    ("ctx66k", 61440, 16512),
    ("ctx180k", 176128, 45056),
]


def visible(base: int, rows: int, device="cuda") -> torch.Tensor:
    pos = torch.arange(base, base + rows, device=device)
    return (pos // 4).to(torch.int32)
