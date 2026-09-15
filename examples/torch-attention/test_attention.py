"""Correctness contract: `attention` must match PyTorch's fused SDPA."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from attention import attention

# (B, H_q, H_kv, L, S, D, causal)
CASES = [
    pytest.param(1, 8, 8, 1, 512, 64, False, id="decode-mha"),
    pytest.param(2, 8, 2, 1, 1024, 128, False, id="decode-gqa"),
    pytest.param(1, 8, 8, 256, 256, 64, True, id="prefill-causal"),
    pytest.param(2, 16, 4, 128, 128, 128, True, id="prefill-gqa-causal"),
    pytest.param(1, 4, 4, 32, 512, 64, True, id="chunked-prefill-causal"),
    pytest.param(1, 4, 4, 7, 7, 96, False, id="odd-shapes"),
]
DTYPES = [pytest.param(torch.float32, id="f32"), pytest.param(torch.float16, id="f16")]
TOL = {torch.float32: 1e-4, torch.float16: 2e-3}


def reference(q, k, v, *, scale, causal):
    """SDPA with an explicit mask so chunked prefill (L < S) aligns queries to the end
    of the key sequence, matching `attention`'s contract."""
    B, H_q, L, D = q.shape
    _, H_kv, S, _ = k.shape
    mask = None
    if causal:
        mask = torch.ones((L, S), dtype=torch.bool, device=q.device).tril(diagonal=S - L)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, enable_gqa=H_kv != H_q
    )


@pytest.mark.parametrize("B,H_q,H_kv,L,S,D,causal", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_fused_sdpa(B, H_q, H_kv, L, S, D, causal, dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn((B, H_q, L, D), generator=g, device="cuda", dtype=dtype)
    k = torch.randn((B, H_kv, S, D), generator=g, device="cuda", dtype=dtype)
    v = torch.randn((B, H_kv, S, D), generator=g, device="cuda", dtype=dtype)

    out = attention(q, k, v, causal=causal)
    ref = reference(q, k, v, scale=D**-0.5, causal=causal)
    torch.cuda.synchronize()

    assert out.shape == ref.shape
    assert out.dtype == dtype
    err = (out.float() - ref.float()).abs().max().item()
    assert err < TOL[dtype], f"max abs error {err}"


def test_explicit_scale():
    q = torch.randn((1, 2, 4, 16), device="cuda")
    k = torch.randn((1, 2, 8, 16), device="cuda")
    v = torch.randn((1, 2, 8, 16), device="cuda")
    out = attention(q, k, v, scale=0.1)
    ref = F.scaled_dot_product_attention(q, k, v, scale=0.1)
    assert torch.allclose(out, ref, atol=1e-5)
