"""The M-invariant bf16 linear: `y = x @ w.T`, bf16 in and out, fp32
accumulation, whose per-row reduction order does not depend on how
many rows are in the call (DOC#DIRECTION-DESIGN, layer 1; DOC#KERNELS).

Authored because cuBLAS chooses its bf16 GEMM algorithm by M on the
served model's tiny shapes ([48, 2560] in seven classes to M=3937,
[1, 2560] alternating with period 8 above 6185) and the algorithms
round differently, so a row's bytes depended on its chunk. One Triton
kernel, `M` unspecialised, the reduction a `tl.dot` chain over 64-wide
K chunks in order: each output element's MMA sequence is a function of
(N, K) alone. The tile (BM, BN, warps, stages) does not enter a row's
bytes (every variant of it is byte-equal), so it is chosen by phase:
`wide` for prefill's M, `thin` for decode's and verify's, where M is
fixed by the composition (tests/test_mconst_linear_gpu.py); and the
chain's bytes are cuBLAS class A's on the wide pinned shapes, so the
thin tile at decode's M agrees with the padded prefill row for row
(tests/test_m_census_gpu.py). Not tuned and not meant to be — at the served tiny
shapes it is at cost parity with cuBLAS at every M; on wide shapes it
is 4–6x slower at M=8192, which is why those sites keep cuBLAS behind
a padded M instead (`nn/mpin`). Its contract is invariance plus
tolerance, not a bit-exact torch reference: no torch path reproduces
a `tl.dot` accumulation order (tests/test_mconst_linear_gpu.py).
"""

import torch

BM, BK = 64, 64

# (BM, BN or None for the static rule, num_warps, num_stages).
TILES = {
    'wide': (64, None, 4, 3),
    'thin': (16, 16, 4, 4),
}


def _kernel():
  global _K
  try:
    return _K
  except NameError:
    pass
  import triton  # noqa: PLC0415
  import triton.language as tl  # noqa: PLC0415

  @triton.jit(do_not_specialize=['M'])
  def _mconst(x_ptr, w_ptr, y_ptr, M, N: tl.constexpr, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
      kk = k0 + rk
      xm = (rm < M)[:, None] & (kk < K)[None, :]
      wm = (rn < N)[:, None] & (kk < K)[None, :]
      xt = tl.load(x_ptr + rm[:, None] * K + kk[None, :], mask=xm, other=0.0)
      wt = tl.load(w_ptr + rn[:, None] * K + kk[None, :], mask=wm, other=0.0)
      acc = tl.dot(xt, tl.trans(wt), acc)
    ym = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
             mask=ym)

  _K = _mconst
  return _K


def block_n(n: int) -> int:
  """The static tile rule: BN from N alone."""
  return 32 if n <= 32 else 64


def linear(x: torch.Tensor, w: torch.Tensor, *,
           tile: str = 'wide') -> torch.Tensor:
  """bf16 `[M, K]` x bf16 `[N, K]` -> bf16 `[M, N]`, both contiguous
  CUDA tensors; a row's bytes are the same at every M and under every
  `tile` (a `TILES` name)."""
  if tile not in TILES:
    raise ValueError(f'unknown mconst tile {tile!r}; one of {sorted(TILES)}')
  if x.dim() != 2 or w.dim() != 2 or x.shape[1] != w.shape[1]:
    raise ValueError(f'mconst linear needs [M, K] x [N, K], got '
                     f'{tuple(x.shape)} x {tuple(w.shape)}')
  if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
    raise ValueError(f'mconst linear is bf16 only, got {x.dtype} x {w.dtype}')
  if not (x.is_contiguous() and w.is_contiguous()):
    raise ValueError('mconst linear needs contiguous operands')
  M, K = x.shape
  N = w.shape[0]
  y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
  bm, bn, warps, stages = TILES[tile]
  bn = block_n(N) if bn is None else bn
  grid = (-(-M // bm), -(-N // bn))
  _kernel()[grid](x, w, y, M, N, K, bm, bn, BK, num_warps=warps,
                  num_stages=stages)
  return y
