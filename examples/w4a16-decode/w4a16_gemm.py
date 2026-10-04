"""The op: `w4a16_gemm(x, packed, scale, out)` — y = x @ dequant(W).T.

x bf16 [M, K]; packed int32 [N, K/8] (eight bias-8 nibbles per word,
LSB-first along K); scale bf16 [N, K/32]; out bf16 [M, N]. M is a
runtime int (the prefill row count varies chunk to chunk; decode and
verify replay under CUDA graphs).

The contract (test_w4a16.py): a row's bytes are a function of the row
and the weight alone — the same at every M and at every position in
the call — and within bf16 rounding of the fp32 reference.

The K fold is one fp32 accumulator threaded through `tl.dot` in
ascending 32-wide leaf order, identical at every M; only the tile is
allowed to move. The tile is chosen by M: a thin decode tile while the
call is a decode/verify batch (M <= 16), a wide prefill tile above,
where a 16-row tile re-reads the weight N/BN times and the shape turns
into a weight-traffic problem. Two specializations compile once and
are then fixed; nothing is autotuned at runtime.

The weight is loaded row-major [BN, BK/8] so eight consecutive words —
one 32-byte sector — belong to one row. The K fold is one fp32
accumulator threaded through `tl.dot` in ascending leaf order at every
M, but the *unpack* differs between the two regimes because the layout
conversion it needs is sized by the tile: the decode tile builds [BN, BK]
and transposes the dequantized bf16, while the prefill tile transposes
the 8x smaller int32 word tile and builds [BK, BN] directly, leaving the
dot's B operand in place. `(nibble - 8)` is exact in bf16 and
`(nibble - 8) * scale` rounds once, the same single rounding the fp32
route produces, so the dequant stays in bf16 either way.
`evict_first` marks the weight as a stream read once.
Only this file changes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from shapes import GROUP

# M below (or equal to) this is the decode/verify tile; above it the prefill tile.
SPLIT_M = 16

# (BM, BN, BK, num_warps, num_stages).  The leaf width BK is the fold: it is a
# leaf of the same K sequence at every M, and a wider leaf only groups the
# following 32-wide leaves into one `tl.dot` in the same order, so it does not
# move the bytes.  Decode is bandwidth/latency bound and wants many narrow
# column tiles with a long, wide leaf: BK=256 is the widest leaf that divides
# every K here (3840/4096/15360), and it cuts the K walk to 60 steps of 32
# adjacent words (128 B) per row -- fewer stalls and a full-load-per-row walk.
#
# The decode wants a *short* M tile.  The mma needs M=16, but tl.dot accepts a
# smaller M and pads internally, so BM=4 stages only 4 rows of x per leaf while
# running the same 16-row tensor op.  The x stage is the pipeline's largest
# buffer (`BM*BK*2` vs `BN*BK/2` for the weight) and only 1-4 of its rows are
# ever real, so shrinking BM hands the shared budget to the weight stream the
# kernel is actually bound by: at `(4,16,256)` the x and weight stages are the
# same size, against 4:1 at BM=16.  M=4 is also the exact verify row count.
# Measured over the six shapes at M in {1,4}: 1.13 vs 1.18 at BM=16, and BM=8,
# 2, 1 and every wider BN or deeper stage are worse.
#
# The prefill wants a tall tile, because each of its N/BN column tiles re-reads
# the whole x slab for its m rows, and that x read is the dominant L2 traffic:
# total x bytes scale as (N/BN) * M * K, so the slab size is set by K.  The
# default tile keeps BN narrow enough to leave plenty of blocks; the deep-K
# tile pays for a wider BN (and the 8 warps its 256x128 accumulator needs) and
# wins once the x slab is large.  Both prefill tiles are measured, and both
# keep the same fold.
DECODE = (4, 16, 256, 4, 3)
PREFILL = (256, 64, 32, 4, 3)
PREFILL_DEEP = (256, 128, 32, 8, 3)
DEEP_K = 8192

# Per-(N, K) decode overrides, measured rather than derived: (N, K) is static, so
# this is a table of fixed constants, not a runtime decision.  The default tile
# is best for five of the six shapes; 4096x3840 (q_proj) wants a wider column
# tile instead and is 15% faster with BN=32 at its two decode row counts.  K
# alone does not decide it -- gate_up shares K=3840 and prefers the default --
# so the key is the shape.
DECODE_BY_SHAPE = {(4096, 3840): (4, 32, 256, 4, 3)}


@triton.jit(do_not_specialize=['M'])
def _w4a16_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GROUP: tl.constexpr, WT: tl.constexpr):
  pid_m = tl.program_id(1)
  pid_n = tl.program_id(0)
  offs_m = pid_m * BM + tl.arange(0, BM)
  offs_n = pid_n * BN + tl.arange(0, BN)
  offs_k = tl.arange(0, BK)
  offs_w = tl.arange(0, BK // 8)
  shifts = (tl.arange(0, 8) * 4).to(tl.int32)
  Kw = K // 8
  Kg = K // GROUP
  SB: tl.constexpr = BK // GROUP
  row_ok = offs_m < M
  acc = tl.zeros((BM, BN), dtype=tl.float32)
  for k0 in range(0, K, BK):
    xb = tl.load(x_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :],
                 mask=row_ok[:, None], other=0.0)  # [BM, BK] bf16
    words = tl.load(w_ptr + offs_n[:, None] * Kw + (k0 // 8 + offs_w)[None, :],
                    eviction_policy='evict_first')  # [BN, BK/8] int32
    if WT:
      # Transpose the int32 words, not the bf16 they expand to.  The word tile
      # is 8x smaller, and the result lands as [BK, BN] so the dot's B operand
      # needs no layout conversion at all.
      wt = tl.trans(words)  # [BK/8, BN]
      nib = tl.reshape((wt[:, None, :] >> shifts[None, :, None]) & 0xF,
                       (BK, BN))
      sc = tl.load(s_ptr + offs_n[None, :] * Kg + k0 // GROUP
                   + tl.arange(0, SB)[:, None])  # [BK/32, BN] bf16
      sc = tl.reshape(tl.broadcast_to(sc[:, None, :], (SB, GROUP, BN)), (BK, BN))
      wb = (nib.to(tl.bfloat16) - 8.0) * sc  # one rounding
      acc = tl.dot(xb, wb, acc)
    else:
      nib = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF,
                       (BN, BK))
      sc = tl.load(s_ptr + offs_n[:, None] * Kg + k0 // GROUP
                   + tl.arange(0, SB)[None, :])  # [BN, BK/32] bf16
      sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, SB, GROUP)), (BN, BK))
      wb = (nib.to(tl.bfloat16) - 8.0) * sc  # one rounding
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
  if M <= SPLIT_M:
    BM, BN, BK, warps, stages = DECODE_BY_SHAPE.get((N, K), DECODE)
  else:
    BM, BN, BK, warps, stages = PREFILL_DEEP if K >= DEEP_K else PREFILL
  # The column-tile axis varies fastest on purpose.  For the same tile, the
  # x slab a program reads is M/BM times smaller than the weight slab it reads
  # N/BN times, and only the small one can stay resident in L2: with the m axis
  # fastest the resident blocks share a weight slab and re-read every x slab
  # from DRAM within one n tile, while with the n axis fastest they share the
  # x slab and stream each weight slab once.  Same bytes, same fold.
  grid = (N // BN, triton.cdiv(M, BM))
  _w4a16_kernel[grid](x, packed, scale, out, M, N, K, BM=BM, BN=BN, BK=BK,
                      GROUP=GROUP, WT=M > SPLIT_M, num_warps=warps,
                      num_stages=stages)
  return out
