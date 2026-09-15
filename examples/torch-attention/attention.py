"""Scaled dot-product attention in PyTorch — the function under optimization.

Layout: q is [B, H_q, L, D]; k and v are [B, H_kv, S, D] with H_q % H_kv == 0
(grouped-query attention). Output is [B, H_q, L, D] in q's dtype.

This baseline is deliberately naive: it materializes the repeated K/V heads, the full
[L, S] score matrix, and the mask. Semantics match
``torch.nn.functional.scaled_dot_product_attention`` — which is the reference in the
tests and is off-limits inside this function, as is any other prebuilt attention op.
"""

from __future__ import annotations

import torch


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    B, H_q, L, D = q.shape
    _, H_kv, S, _ = k.shape
    assert H_q % H_kv == 0, "query heads must be a multiple of key/value heads"
    scale = D**-0.5 if scale is None else scale

    if H_kv != H_q:
        k = k.repeat_interleave(H_q // H_kv, dim=1)
        v = v.repeat_interleave(H_q // H_kv, dim=1)

    scores = (q * scale) @ k.transpose(-1, -2)
    if causal:
        # Query i (aligned to the end of the key sequence) may see keys 0..S-L+i.
        mask = torch.ones((L, S), dtype=torch.bool, device=q.device).tril(diagonal=S - L)
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return probs @ v
