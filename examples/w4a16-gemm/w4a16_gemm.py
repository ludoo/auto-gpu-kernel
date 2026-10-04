"""The op: `w4a16_gemm(x, packed, scale, out)` — y = x @ dequant(W).T.

x bf16 [M, K]; packed int32 [N, K/8] (eight bias-8 nibbles per word,
LSB-first along K); scale bf16 [N, K/32]; out bf16 [M, N]. M is a
runtime int (the prefill row count varies chunk to chunk; decode and
verify replay under CUDA graphs).

The contract (test_w4a16.py): a row's bytes are a function of the row
and the weight alone — the same at every M and at every position in
the call — and within bf16 rounding of the fp32 reference.

This starting kernel holds the contract by construction and is slow:
one program per (16-row tile, 64-column tile), the K walk in 32-wide
leaves (one scale group) in ascending order, each leaf's weight
dequantized in registers to bf16 ((nibble - 8) * scale, one rounding),
`tl.dot` with an fp32 accumulator threaded through the leaves, one
rounding to bf16 at the end. No split-K, no pipelining to speak of. The
leaf width and the accumulate chain are the fold; the tile is not.
Only this file changes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from shapes import GROUP

BM = 16
BN = 64
BK = GROUP


@triton.jit(do_not_specialize=['M'])
def _w4a16_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GROUP: tl.constexpr):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  offs_m = pid_m * BM + tl.arange(0, BM)
  offs_n = pid_n * BN + tl.arange(0, BN)
  offs_k = tl.arange(0, BK)
  offs_w = tl.arange(0, BK // 8)
  shifts = (tl.arange(0, 8) * 4).to(tl.int32)
  Kw = K // 8
  Kg = K // GROUP
  row_ok = offs_m < M
  acc = tl.zeros((BM, BN), dtype=tl.float32)
  for k0 in range(0, K, BK):
    xb = tl.load(x_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :],
                 mask=row_ok[:, None], other=0.0)  # [BM, BK] bf16
    words = tl.load(w_ptr + offs_n[:, None] * Kw + (k0 // 8 + offs_w)[None, :])
    nib = (words[:, :, None] >> shifts[None, None, :]) & 0xF  # [BN, BK/8, 8]
    nib = tl.reshape(nib, (BN, BK))
    sc = tl.load(s_ptr + offs_n * Kg + k0 // GROUP)  # [BN] bf16
    wb = ((nib.to(tl.float32) - 8.0) * sc.to(tl.float32)[:, None]).to(tl.bfloat16)
    acc = tl.dot(xb, tl.trans(wb), acc)
  tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),
           mask=row_ok[:, None])


def w4a16_gemm(x, packed, scale, out):
  if x.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
    raise ValueError('x and out must be bf16')
  M, K = x.shape
  N, Kw = packed.shape
  if Kw * 8 != K or scale.shape != (N, K // GROUP) or out.shape != (M, N):
    raise ValueError(f'shape mismatch: x {tuple(x.shape)} packed '
                     f'{tuple(packed.shape)} scale {tuple(scale.shape)} '
                     f'out {tuple(out.shape)}')
  if not (x.is_contiguous() and packed.is_contiguous() and
          scale.is_contiguous() and out.is_contiguous()):
    raise ValueError('operands must be contiguous')
  grid = (triton.cdiv(M, BM), N // BN)
  _w4a16_kernel[grid](x, packed, scale, out, M, N, K, BM=BM, BN=BN, BK=BK,
                      GROUP=GROUP, num_warps=4, num_stages=3)
  return out
