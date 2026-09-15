"""Decode-time bf16 linear: y = x @ W.T for M <= 4 rows.

This is the function to optimize. It is called ~670 times per decode
step of a 48-layer model, once per weight matrix, with every weight
cold in DRAM (the step streams 7.2 GB of bf16 weights and nothing is
reused between calls). The metric is therefore cold-weight bandwidth,
not hot-loop latency.

Contract (see test_gemv.py):
  x: bf16 [M, K], 1 <= M <= 4, contiguous
  w: bf16 [N, K], row-major contiguous (torch.nn.Linear layout)
  returns bf16 [M, N], fp32 accumulation over K
"""

from __future__ import annotations

import torch


def gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x @ w.T
