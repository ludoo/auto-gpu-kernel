"""Correctness contract: gemv matches an fp32 reference within tolerance.

Bit-exactness against cuBLAS is not required (its reduction order is
opaque); fp32 accumulation is. The tolerance is the bf16 output
rounding plus a small allowance for reduction-order differences in
fp32; a kernel that accumulates in bf16 fails it.
"""

from __future__ import annotations

import pytest
import torch
from gemv import gemv
from shapes import SHAPES

CASES = [pytest.param(n, k, id=f"{name}[{n}x{k}]") for n, k, _, name in SHAPES]
MS = [1, 2, 3, 4]


def reference(x, w):
    return (x.float() @ w.float().T)


def check(x, w):
    y = gemv(x, w)
    assert y.dtype == torch.bfloat16 and y.shape == (x.shape[0], w.shape[0])
    ref = reference(x, w)
    # bf16 has 8 bits of mantissa: 1 ulp at |ref| is ~2^-8 * |ref|. Allow 1 ulp
    # of the element plus 1 ulp of the row's largest output, so elements that
    # cancel to near zero are judged at the row's scale. cuBLAS passes this
    # (its M=4 split-K path rounds partials to bf16 and lands at ~2 ulp of
    # scale); a bf16 accumulator over K=2560 does not.
    ulp = 2.0 ** -8
    tol = ulp * ref.abs() + ulp * ref.abs().amax(dim=1, keepdim=True) + 1e-3
    err = (y.float() - ref).abs()
    assert bool((err <= tol).all()), (
        f"max err {err.max().item():.4g} at |ref| {ref.abs().max().item():.4g}; "
        f"{int((err > tol).sum())} of {err.numel()} out of tolerance")


@pytest.mark.parametrize("N,K", CASES)
@pytest.mark.parametrize("M", MS)
def test_matches_fp32_reference(N, K, M):
    g = torch.Generator(device="cuda").manual_seed(N * 7919 + K * 31 + M)
    x = torch.randn((M, K), generator=g, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), generator=g, device="cuda", dtype=torch.bfloat16)
    check(x, w)


def test_large_magnitude_accumulation():
    """Catches bf16 accumulation: many same-sign terms overflow bf16 precision."""
    M, N, K = 4, 640, 10240
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.full((M, K), 1.0, device="cuda", dtype=torch.bfloat16)
    w = (0.5 + torch.rand((N, K), generator=g, device="cuda")).to(torch.bfloat16)
    check(x, w)


def test_does_not_write_inputs():
    M, N, K = 3, 512, 2560
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    x0, w0 = x.clone(), w.clone()
    gemv(x, w)
    torch.cuda.synchronize()
    assert torch.equal(x, x0) and torch.equal(w, w0)
