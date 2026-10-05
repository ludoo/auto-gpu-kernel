"""Paged causal attention for gemma's two kinds — the one attention
launch of the served gemma model in every phase (DOC#MODEL-GEMMA-12B,
DOC#task-next.120). Authored under the kernels bar (DOC#KERNELS): a
static grid, no autotuning, geometry from the tensors, and a contract
of invariance plus tolerance — a row's bytes are a function of the
row alone and sit within bf16 rounding of an fp32 reference
(tests/test_gemma_attention_gpu.py). Adopted from the kopt data
session's survivor: `auto-gpu-kernel` fork, `work/gemma-attn/repo/
gemma_attn.py` at `b3b9ff6`; the loop and its measurements are in
docs/history/archive/next120/KOPT-NOTES.md.

`gemma_attention(q, pages, qo_indptr, kv_indptr, kv_indices,
kv_last_page_len, pos_base, window_left, out, *, span_indptr=None,
span_bounds=None)`: bf16 in and out, n >= 1 sequences per launch, the
last `rows` positions of each. The metadata is FlashInfer's four int32
device tensors plus `pos_base` int32 [n], the absolute position of
each sequence's gather key 0 (`first_page * page_size` on the sliding
kind, 0 on the global; `sparkle.nn.kvmeta.window_pos_base`), and the
two optional keyword-only int32 span tensors `span_indptr` [n + 1] and
`span_bounds` [S, 2]: sequence `seq`'s image spans are rows
`span_indptr[seq] .. span_indptr[seq + 1]`, each an absolute `[lo, hi)`
image span (images only, spans never overlap). None means no spans. A
sequence's key count is
`(pages - 1) * page_size + last_page_len`; row `i` of its span sits
at absolute position `pos_base + kv_len - rows + i`. Under CUDA-graph
replay the metadata is read on device from fixed buffers: the split
path's block count and the wide-tile rule read `kv_indices.shape[0]`,
so a fixed buffer longer than the gather launches the max block
count (early-exit programs) and takes the wide tile at 4 rows — a
price, never a byte.

The fold
--------
Two levels, both fixed in ABSOLUTE position space (`pos_base` + gather
index), so neither moves with the row count, the chunk start or the
tile:

  leaf  LEAF keys   — a left fold in registers over the leaf's keys,
                      starting from a true zero (see below). On the
                      global kind that fold is a plain accumulation
                      against the block's scale; on the sliding kind it
                      is an online-softmax fold against a running max
  block BLOCK keys  — a zero-start partial per block, then a left fold
                      over blocks in ascending order through `_combine`

The block level is SPLIT-INVARIANT. One program walking every block
and folding each into a running state performs exactly the same
sequence of `_combine` calls as N programs each computing one block's
partial and a second kernel folding them in the same ascending order.
So the chunk folds in registers (its workspace would be terabytes)
while decode and verify split the kv range across programs, and the
two paths agree bitwise. The block partials are merged with ordinary
fp32 arithmetic, never through the mma's accumulate, and every block is a
zero-start fold.

The block scale
---------------
`_CFG[(1, 512)]`'s `fixed_base` picks a second leaf grain for the global
kind. A block's scale is the row max of its FIRST leaf, held for the whole
block, and each leaf then contributes `exp2(s - m)` straight into the
mma's C operand: no per-leaf row max, no `alpha`, no `[BM, DV]` rescale,
and one reduction for `l` per block instead of one per leaf. Measured
-4.1 / -4.4 / -4.5% on the three scored cells.

`m` is therefore not the block's true max but a per-row constant. The
two-level fold only needs a scale `_combine` can rescale from, which any
finite per-row value is, and this one is a lower bound of the block's max
so `p` never underflows. `BLOCK_CEIL` bounds it from above; `BLOCK_FLOOR`
keeps a row whose first leaf is fully masked finite, and on the global
kind a masked first leaf means the whole block is past that row, so
`p = 0` there is the exact no-op. The sliding kind keeps the online leaf
fold, because its window lets a row's first leaf be masked while a later
leaf in the same block is attended.

Exact no-ops
------------
A leaf a row may not attend must change nothing for that row, or a row
could not ride a tile (several rows, one key loop) or a block (whose
bounds are absolute, not the row's). It changes nothing because `m_i`
starts at a finite `-3.0e38` rather than `-inf`:

  * fully masked leaf: `m_new = max(-BIG, -inf) = -BIG`, so
    `alpha = exp2(0) = 1` and `p = exp2(-inf + BIG) = 0`; `l` and `acc`
    take `* 1.0 + 0.0`;
  * the row's first real leaf: `alpha = exp2(-BIG - m) = 0` exactly, so
    the fold starts from a true zero;
  * the identity state `(-BIG, 0, 0)` is a two-sided identity of
    `_combine`, which is what lets a block outside a row's span, or a
    tile wider than a row's span, cost it nothing.

Rows of a tile do not interact in the mma, so a row cannot see how
many rows rode with it.

The image block mask
--------------------
The sliding kind's one custom mask rides in as the span metadata: for
a query at absolute `pos_abs` and a key at `key_abs`, with the
query's own span `[row_lo, row_hi)` ((0, 0) when none),

    ok = (key_abs >= lo_abs) & ((key_abs <= pos_abs) |
                                ((key_abs >= row_lo) & (key_abs < row_hi)))

which, because a span is contiguous and `row_lo <= pos_abs`, is one
bound per row: `key_abs <= reach` with
`reach = max(pos_abs, row_hi - 1)` — the compare the text launch pays.
`lo_abs` is already the window start (0 on the global kind). The
predicate enters `ok` in `_fold_block` without touching any of the
fold above, so a text row's bytes do not depend on whether a span rode
in its launch and an empty-span launch is byte-identical to one with
no spans; `sparkle.nn.gemma_vision.image_block_mask_bool` is the
predicate reference. The tile's key span is extended to the last key
an in-span row can reach, so the image keys after a query are loaded.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)
NEG_BIG = tl.constexpr(-3.0e38)
# Bounds on a FIXED block's scale (module docstring), in log2 score units.
# The floor makes a fully masked first leaf finite; the ceiling bounds a
# block whose later leaf outruns its first leaf's max.
BLOCK_FLOOR = tl.constexpr(-100.0)
BLOCK_CEIL = tl.constexpr(100.0)

# Per-kind tiling, keyed by (kv_heads, head_dim). Every constant here
# enters the fold, so each is a property of the geometry alone and
# never of the row count.
# MEASURED, the hard way: `num_warps` and `num_stages` ARE part of the
# fold. Giving the split path its own warp layout breaks bitwise
# invariance against the fused path in about one element of 25k (one
# bf16 ulp) — the warp layout moves the mma's reduction order. Both
# paths must therefore share `num_warps` and `num_stages`, which is why
# there is no `split_warps`. LEAF and BLOCK set the fold's grain and
# are shared for the same reason. `num_stages` is the one exception, and
# only on the global kind: the previous loop measured (exp_011) that it
# does not move a byte there, which is what `split_stages` and the
# L2-residency rule in `gemma_attention` rely on. On the sliding kind it
# does move a byte and stays shared.
#
# The row tile is NOT in the fold (it only adds exact no-op leaves) and
# is free to differ, which matters because the two paths want opposite
# things.
# The fused path wants the widest tile it can hold, to amortise the key
# loop over rows; the split path has one tile either way and wants the
# narrowest legal one, so padding lanes cost no mma work.
#   sliding fused: group 2, d 256 -> 32 rows x 2 heads = M 64
#   sliding split: group 2, d 256 ->  8 rows x 2 heads = M 16
#   global  split: group 16, d 512 ->  1 row  x 16 heads = M 16
#   global  fused: group 16, d 512 ->  2 rows x 16 heads = M 32
# The sliding fused tile is held at M 64 / LEAF 64 by shared memory:
# q + K + V at [64, 256] bf16 is 96 KB of the part's 99 KB, so its key
# loop runs unpipelined (num_stages=1); M 128 or LEAF 128 will not fit.
# Global keeps rows_tile 2 on both paths: rows_tile 1 is worth 14 us at
# decode/4k but makes a 4-row verify four tiles, each re-reading the
# whole sequence (5057 us against 2695 at 128k).
# BLOCK is the kv-split grain: the state a split program writes is
# BM*D fp32 per block against BLOCK*D*4 bytes of K/V read, i.e. a
# round trip of 2*BM/BLOCK of the traffic it saves. It is in the fold.
# DCHUNK is the stage-2 reduction's column tile and is NOT in the fold
# (the fold over partials is elementwise in D), so it is free to be
# whatever parallelises the reduction best.
_CFG = {
    (8, 256):
        dict(rows_tile=32, leaf=64, block=256, num_warps=8, num_stages=1,
             split_stages=1, split_rows_tile=8, dchunk=64, wide_rows_tile=None,
             wide_dsplit=1, fixed_base=0),
    (1, 512):
        dict(rows_tile=2, leaf=32, block=256, num_warps=8, num_stages=2,
             split_stages=1, split_rows_tile=1, dchunk=16, wide_rows_tile=4,
             wide_dsplit=2, fixed_base=1),
}

# Split the kv range across programs at or below this many rows: the
# chunk has tiles to spare, decode and verify do not. The rule is on
# the row count, which is allowed, and it is safe because both paths
# run the same fold.
SPLIT_MAX_ROWS = 8

# Gather size above which a 4-row verify is worth one wide tile that
# reads K twice, rather than two narrow tiles that read K and V twice.
WIDE_MIN_KEYS = 8192

# The empty-span launch's metadata: an all-zero `span_indptr` and one
# zero `[lo, hi)` row, so the span loop in `_tile_geometry` never
# runs. Allocated once per device at a fixed size and NEVER moved: a
# captured decode or verify graph reads these addresses at every
# replay, so a reallocation would leave it reading freed memory. An
# all-zero indptr reads the same at any n, so one buffer serves every
# launch up to EMPTY_SPAN_MAX_SEQ sequences.
EMPTY_SPAN_MAX_SEQ = 1024
_EMPTY_SPANS: dict = {}

# This part's L2 size, read once per device. Used only to pick `num_stages`
# on the fused path (a static shape rule, module `gemma_attention`).
_L2_BYTES: dict = {}


def _l2_bytes(device):
  """L2 size in bytes, cached per device."""
  n = _L2_BYTES.get(device)
  if n is None:
    n = torch.cuda.get_device_properties(device).L2_cache_size
    _L2_BYTES[device] = n
  return n


def _empty_spans(n_seq, device):
  """`(span_indptr, span_bounds)` that mean no spans: the device's
  fixed zero buffers."""
  if n_seq + 1 > EMPTY_SPAN_MAX_SEQ + 1:
    raise ValueError(f'gemma_attention: {n_seq} sequences in one launch, '
                     f'the empty-span buffer holds {EMPTY_SPAN_MAX_SEQ}')
  bufs = _EMPTY_SPANS.get(device)
  if bufs is None:
    bufs = (torch.zeros(EMPTY_SPAN_MAX_SEQ + 1, dtype=torch.int32,
                        device=device),
            torch.zeros(1, 2, dtype=torch.int32, device=device))
    _EMPTY_SPANS[device] = bufs
  return bufs


@triton.jit
def _combine(m_a, l_a, acc_a, m_b, l_b, acc_b):
  """Fold block state b into state a. `(-BIG, 0, 0)` is a two-sided
  identity, exactly: `exp2(-BIG - m) = 0` and `exp2(0) = 1`."""
  m_n = tl.maximum(m_a, m_b)
  ca = tl.exp2(m_a - m_n)
  cb = tl.exp2(m_b - m_n)
  return m_n, l_a * ca + l_b * cb, acc_a * ca[:, None] + acc_b * cb[:, None]


@triton.jit
def _fold_block(q, leaf_lo, leaf_hi, k_ptr, v_ptr, kv_indices, kv_start, base,
                lo_abs, reach, lo_abs_tile, hi_abs_tile, stride_p_page,
                stride_p_tok, stride_p_head, kvh, GROUP: tl.constexpr,
                D: tl.constexpr, PAGE_: tl.constexpr, LEAF_: tl.constexpr,
                BM: tl.constexpr, DV: tl.constexpr, v_col,
                FIXED: tl.constexpr = 0):
  """Zero-start partial over leaves [leaf_lo, leaf_hi], for the DV output
  columns of V starting at `v_col`.

  The scores, and so `m` and `l`, do not depend on which output columns
  this call owns: they are computed from the whole of q and K either
  way. Only V's load and `acc` are narrowed. A caller that wants all of
  D passes DV = D and v_col = 0.

  FIXED selects the block-grain fold used on the global kind. The block's
  scale is the row max of its FIRST leaf, held for the whole block, so
  `acc` is accumulated straight through the mma's C operand: no per-leaf
  row max, no `alpha`, no [BM, DV] rescale. `m` is then not the block's
  true max but a per-row constant, which is what the two-level fold needs
  -- `_combine` is exact for any scale, and the identity holds because
  the scale is finite. The first leaf is peeled so the reduction that
  sets the scale runs once per block instead of once per leaf.

  BLOCK_FLOOR keeps a row whose first leaf is fully masked at a finite
  scale, so its whole block yields `p = 0` exactly rather than inf - inf;
  on the global kind a masked first leaf means the whole block is past
  the row (keys ascend, the causal bound is `key <= pos`), so that is the
  correct no-op. BLOCK_CEIL keeps a block whose later leaf outruns the
  first leaf's max finite; it engages only when one block's scores span
  more than 100 log2 units."""
  tm = tl.arange(0, LEAF_)
  dm = tl.arange(0, D)
  dv = v_col + tl.arange(0, DV)
  if FIXED:
    key_abs = leaf_lo * LEAF_ + tm
    gidx = key_abs - base
    load_ok = (key_abs >= lo_abs_tile) & (key_abs <= hi_abs_tile)
    page = tl.load(kv_indices + kv_start + gidx // PAGE_, mask=load_ok,
                   other=0)
    off = (page * stride_p_page + (gidx % PAGE_) * stride_p_tok +
           kvh * stride_p_head)
    k = tl.load(k_ptr + off[:, None] + dm[None, :], mask=load_ok[:, None],
                other=0.0)
    ok = (key_abs[None, :] >= lo_abs[:, None]) & \
         (key_abs[None, :] <= reach[:, None])
    s = tl.dot(q, tl.trans(k)) * LOG2E
    v = tl.load(v_ptr + off[:, None] + dv[None, :], mask=load_ok[:, None],
                other=0.0)
    s = tl.where(ok, s, float('-inf'))
    m_i = tl.maximum(tl.max(s, 1), BLOCK_FLOOR)
    mb = tl.zeros([BM, LEAF_], tl.float32) + m_i[:, None]
    p = tl.exp2(tl.minimum(s - mb, BLOCK_CEIL))
    pl = p
    acc = tl.dot(p.to(tl.bfloat16), v)
    for leaf in range(leaf_lo + 1, leaf_hi + 1):
      key_abs = leaf * LEAF_ + tm
      gidx = key_abs - base
      load_ok = (key_abs >= lo_abs_tile) & (key_abs <= hi_abs_tile)
      page = tl.load(kv_indices + kv_start + gidx // PAGE_, mask=load_ok,
                     other=0)
      off = (page * stride_p_page + (gidx % PAGE_) * stride_p_tok +
             kvh * stride_p_head)
      k = tl.load(k_ptr + off[:, None] + dm[None, :], mask=load_ok[:, None],
                  other=0.0)
      ok = (key_abs[None, :] >= lo_abs[:, None]) & \
           (key_abs[None, :] <= reach[:, None])
      s = tl.dot(q, tl.trans(k)) * LOG2E
      v = tl.load(v_ptr + off[:, None] + dv[None, :], mask=load_ok[:, None],
                  other=0.0)
      s = tl.where(ok, s, float('-inf'))
      p = tl.exp2(tl.minimum(s - mb, BLOCK_CEIL))
      pl += p
      acc = tl.dot(p.to(tl.bfloat16), v, acc)
    return m_i, tl.sum(pl, 1), acc
  m_i = tl.full([BM], NEG_BIG, tl.float32)
  l_i = tl.zeros([BM], tl.float32)
  acc = tl.zeros([BM, DV], tl.float32)
  for leaf in range(leaf_lo, leaf_hi + 1):
    key_abs = leaf * LEAF_ + tm
    gidx = key_abs - base
    load_ok = (key_abs >= lo_abs_tile) & (key_abs <= hi_abs_tile)
    page = tl.load(kv_indices + kv_start + gidx // PAGE_, mask=load_ok, other=0)
    off = (page * stride_p_page + (gidx % PAGE_) * stride_p_tok +
           kvh * stride_p_head)
    k = tl.load(k_ptr + off[:, None] + dm[None, :], mask=load_ok[:, None],
                other=0.0)  # [LEAF, D]
    ok = (key_abs[None, :] >= lo_abs[:, None]) & \
         (key_abs[None, :] <= reach[:, None])
    s = tl.dot(q, tl.trans(k)) * LOG2E  # [BM, LEAF]
    # V is loaded only after the QK dot has consumed K. Issuing both
    # loads up front costs shared memory and registers for a value not
    # needed until the second dot: on the sliding chunk tile that is
    # 24 spills against 8, and 1630 us against 1540.
    v = tl.load(v_ptr + off[:, None] + dv[None, :], mask=load_ok[:, None],
                other=0.0)
    s = tl.where(ok, s, float('-inf'))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    m_i = m_new
  return m_i, l_i, acc


@triton.jit
def _tile_geometry(qo_indptr, kv_indptr, kv_last, pos_base, span_indptr,
                   span_bounds, seq, r0, GROUP: tl.constexpr,
                   WINDOW_LEFT: tl.constexpr, PAGE_: tl.constexpr,
                   BM: tl.constexpr):
  """Everything a tile needs that lives on device: row validity, each
  M slot's absolute position, window start and reach (its position,
  or its image span's last key), and the tile's own absolute key span
  (the union of its rows')."""
  qo_start = tl.load(qo_indptr + seq)
  qo_len = tl.load(qo_indptr + seq + 1) - qo_start
  kv_start = tl.load(kv_indptr + seq)
  n_pages = tl.load(kv_indptr + seq + 1) - kv_start
  kv_len = (n_pages - 1) * PAGE_ + tl.load(kv_last + seq)
  base = tl.load(pos_base + seq)

  hm = tl.arange(0, BM)
  row = r0 + hm // GROUP
  head = hm % GROUP
  row_ok = row < qo_len
  pos_abs = base + kv_len - qo_len + row
  r_last = tl.minimum(r0 + BM // GROUP - 1, qo_len - 1)
  hi_abs_tile = base + kv_len - qo_len + r_last
  # The image span each row lies in, (0, 0) when none: the rows of a
  # tile may sit in different spans, so this is per-row.
  row_lo = pos_abs - pos_abs
  row_hi = pos_abs - pos_abs
  s0 = tl.load(span_indptr + seq)
  s1 = tl.load(span_indptr + seq + 1)
  for s in range(s0, s1):
    lo = tl.load(span_bounds + s * 2)
    hi = tl.load(span_bounds + s * 2 + 1)
    inside = (pos_abs >= lo) & (pos_abs < hi)
    row_lo = tl.where(inside, lo, row_lo)
    row_hi = tl.where(inside, hi, row_hi)
  # OR(causal, same span) for a row inside a contiguous span is
  # `key <= hi - 1`, since `lo <= pos_abs` already admits every key
  # below the span causally; for a text row it is `key <= pos_abs`.
  reach = tl.maximum(pos_abs, row_hi - 1)
  hi_abs_tile = tl.maximum(hi_abs_tile,
                           tl.minimum(tl.max(reach), base + kv_len - 1))
  if WINDOW_LEFT >= 0:
    lo_abs = tl.maximum(pos_abs - WINDOW_LEFT, 0)
    lo_abs_tile = tl.maximum(base + kv_len - qo_len + r0 - WINDOW_LEFT, 0)
  else:
    lo_abs = pos_abs - pos_abs  # 0, int32 [BM]
    lo_abs_tile = base - base  # 0, int32 scalar
  return (qo_start, qo_len, kv_start, base, row, head, row_ok, pos_abs, lo_abs,
          reach, lo_abs_tile, hi_abs_tile)


@triton.jit
def _attn_fused(
    q_ptr,
    out_ptr,
    k_ptr,
    v_ptr,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last,
    pos_base,
    span_indptr,
    span_bounds,
    stride_q_row,
    stride_q_head,
    stride_p_page,
    stride_p_tok,
    stride_p_head,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    PAGE_: tl.constexpr,
    LEAF_: tl.constexpr,
    BLOCK_: tl.constexpr,
    ROWS_TILE: tl.constexpr,
    BM: tl.constexpr,
    FIXED: tl.constexpr,
):
  """One program per (row tile, sequence, kv head): every block, folded
  in registers in ascending order."""
  tile = tl.program_id(0)
  seq = tl.program_id(1)
  kvh = tl.program_id(2)
  r0 = tile * ROWS_TILE
  if r0 >= tl.load(qo_indptr + seq + 1) - tl.load(qo_indptr + seq):
    return
  (qo_start, qo_len, kv_start, base, row, head, row_ok, pos_abs, lo_abs, reach,
   lo_abs_tile, hi_abs_tile) = _tile_geometry(qo_indptr, kv_indptr, kv_last,
                                              pos_base, span_indptr,
                                              span_bounds, seq, r0, GROUP,
                                              WINDOW_LEFT, PAGE_, BM)

  dm = tl.arange(0, D)
  q = tl.load(
      q_ptr + (qo_start + row)[:, None] * stride_q_row +
      (kvh * GROUP + head)[:, None] * stride_q_head + dm[None, :],
      mask=row_ok[:, None], other=0.0)

  m_i = tl.full([BM], NEG_BIG, tl.float32)
  l_i = tl.zeros([BM], tl.float32)
  acc = tl.zeros([BM, D], tl.float32)
  for blk in range(lo_abs_tile // BLOCK_, hi_abs_tile // BLOCK_ + 1):
    leaf_lo = tl.maximum(blk * BLOCK_, lo_abs_tile) // LEAF_
    leaf_hi = tl.minimum(blk * BLOCK_ + BLOCK_ - 1, hi_abs_tile) // LEAF_
    m_b, l_b, acc_b = _fold_block(q, leaf_lo, leaf_hi, k_ptr, v_ptr, kv_indices,
                                  kv_start, base, lo_abs, reach, lo_abs_tile,
                                  hi_abs_tile, stride_p_page, stride_p_tok,
                                  stride_p_head, kvh, GROUP, D, PAGE_, LEAF_,
                                  BM, D, 0, FIXED)
    m_i, l_i, acc = _combine(m_i, l_i, acc, m_b, l_b, acc_b)

  o = acc / l_i[:, None]
  tl.store(
      out_ptr + (qo_start + row)[:, None] * stride_q_row +
      (kvh * GROUP + head)[:, None] * stride_q_head + dm[None, :],
      o.to(tl.bfloat16), mask=row_ok[:, None])


@triton.jit
def _attn_split(
    q_ptr,
    ws_m,
    ws_l,
    ws_acc,
    k_ptr,
    v_ptr,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last,
    pos_base,
    span_indptr,
    span_bounds,
    stride_q_row,
    stride_q_head,
    stride_p_page,
    stride_p_tok,
    stride_p_head,
    n_tiles,
    n_kv,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    PAGE_: tl.constexpr,
    LEAF_: tl.constexpr,
    BLOCK_: tl.constexpr,
    ROWS_TILE: tl.constexpr,
    BM: tl.constexpr,
    DSPLIT: tl.constexpr,
    FIXED: tl.constexpr,
):
  """Stage 1: one program per (row tile, sequence, kv head, block,
  column slice). Writes that block's zero-start partial; folds nothing.

  DSPLIT > 1 buys a wider row tile. `acc` at [BM, D] fp32 is what caps
  ROWS_TILE on the global kind -- [64, 512] is 128 KB, the whole
  register file -- so a 4-row verify had to run as two tiles, each
  re-reading the entire sequence. With the output columns cut DSPLIT
  ways each program holds [BM, D/DSPLIT] and re-reads K instead: for
  two tiles of two rows that is 2*(K+V), for one tile of four rows in
  two column halves it is 2*K + V, three quarters of the traffic."""
  z = tl.program_id(0)
  tile = tl.program_id(1) // DSPLIT
  cs = tl.program_id(1) % DSPLIT
  sq = tl.program_id(2)
  seq = sq // n_kv
  kvh = sq % n_kv
  r0 = tile * ROWS_TILE
  if r0 >= tl.load(qo_indptr + seq + 1) - tl.load(qo_indptr + seq):
    return
  (qo_start, qo_len, kv_start, base, row, head, row_ok, pos_abs, lo_abs, reach,
   lo_abs_tile, hi_abs_tile) = _tile_geometry(qo_indptr, kv_indptr, kv_last,
                                              pos_base, span_indptr,
                                              span_bounds, seq, r0, GROUP,
                                              WINDOW_LEFT, PAGE_, BM)

  blk = lo_abs_tile // BLOCK_ + z
  if blk > hi_abs_tile // BLOCK_:
    return

  dm = tl.arange(0, D)
  q = tl.load(
      q_ptr + (qo_start + row)[:, None] * stride_q_row +
      (kvh * GROUP + head)[:, None] * stride_q_head + dm[None, :],
      mask=row_ok[:, None], other=0.0)
  leaf_lo = tl.maximum(blk * BLOCK_, lo_abs_tile) // LEAF_
  leaf_hi = tl.minimum(blk * BLOCK_ + BLOCK_ - 1, hi_abs_tile) // LEAF_
  DV: tl.constexpr = D // DSPLIT
  v_col = cs * DV
  m_b, l_b, acc_b = _fold_block(q, leaf_lo, leaf_hi, k_ptr, v_ptr, kv_indices,
                                kv_start, base, lo_abs, reach, lo_abs_tile,
                                hi_abs_tile, stride_p_page, stride_p_tok,
                                stride_p_head, kvh, GROUP, D, PAGE_, LEAF_, BM,
                                DV, v_col, FIXED)

  slot = (sq * n_tiles + tile) * tl.num_programs(0) + z
  hm = tl.arange(0, BM)
  if cs == 0:  # m and l are the same in every column slice
    tl.store(ws_m + slot * BM + hm, m_b, mask=row_ok)
    tl.store(ws_l + slot * BM + hm, l_b, mask=row_ok)
  tl.store(
      ws_acc + slot * BM * D + hm[:, None] * D +
      (v_col + tl.arange(0, DV))[None, :], acc_b, mask=row_ok[:, None])


@triton.jit
def _attn_reduce(
    out_ptr,
    ws_m,
    ws_l,
    ws_acc,
    qo_indptr,
    kv_indptr,
    kv_last,
    pos_base,
    span_indptr,
    span_bounds,
    stride_q_row,
    stride_q_head,
    n_tiles,
    n_kv,
    n_blk,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    PAGE_: tl.constexpr,
    BLOCK_: tl.constexpr,
    ROWS_TILE: tl.constexpr,
    BM: tl.constexpr,
    DCHUNK: tl.constexpr,
):
  """Stage 2: the same left fold over blocks in ascending order that
  `_attn_fused` performs in registers, over the stage-1 partials.

  Split over the head dimension: `acc`'s fold is elementwise in D, so a
  program owning DCHUNK columns folds exactly the bytes it will store.
  `m` and `l` are [BM] and are folded redundantly in every column
  program — the same inputs in the same order, hence the same bytes,
  for a few hundred flops against the serial reduction this removes.
  Without it stage 2 is one CTA per (tile, kv head) dragging the whole
  partial workspace through itself: 8 MB at 128k, and the measured
  reason a fine split grain stopped paying at long context."""
  tile = tl.program_id(0)
  sq = tl.program_id(1)
  ds = tl.program_id(2) * DCHUNK
  seq = sq // n_kv
  kvh = sq % n_kv
  r0 = tile * ROWS_TILE
  if r0 >= tl.load(qo_indptr + seq + 1) - tl.load(qo_indptr + seq):
    return
  (qo_start, qo_len, kv_start, base, row, head, row_ok, pos_abs, lo_abs, _reach,
   lo_abs_tile, hi_abs_tile) = _tile_geometry(qo_indptr, kv_indptr, kv_last,
                                              pos_base, span_indptr,
                                              span_bounds, seq, r0, GROUP,
                                              WINDOW_LEFT, PAGE_, BM)

  dm = ds + tl.arange(0, DCHUNK)
  hm = tl.arange(0, BM)
  m_i = tl.full([BM], NEG_BIG, tl.float32)
  l_i = tl.zeros([BM], tl.float32)
  acc = tl.zeros([BM, DCHUNK], tl.float32)
  n_z = hi_abs_tile // BLOCK_ - lo_abs_tile // BLOCK_ + 1
  base_slot = (sq * n_tiles + tile) * n_blk
  for z in range(0, n_z):
    slot = base_slot + z
    m_b = tl.load(ws_m + slot * BM + hm, mask=row_ok, other=NEG_BIG)
    l_b = tl.load(ws_l + slot * BM + hm, mask=row_ok, other=0.0)
    acc_b = tl.load(ws_acc + slot * BM * D + hm[:, None] * D + dm[None, :],
                    mask=row_ok[:, None], other=0.0)
    m_i, l_i, acc = _combine(m_i, l_i, acc, m_b, l_b, acc_b)

  o = acc / l_i[:, None]
  tl.store(
      out_ptr + (qo_start + row)[:, None] * stride_q_row +
      (kvh * GROUP + head)[:, None] * stride_q_head + dm[None, :],
      o.to(tl.bfloat16), mask=row_ok[:, None])


def gemma_attention(q, pages, qo_indptr, kv_indptr, kv_indices,
                    kv_last_page_len, pos_base, window_left, out, *,
                    span_indptr=None, span_bounds=None):
  """Paged causal attention over `pages` for the rows of `q`, written
  to `out` (bf16 `[rows, heads, head_dim]`, returned). `pages` is
  `[n_phys, 2, page_size, kv_heads, head_dim]` bf16, index 0 K and 1
  V; the five int32 metadata tensors are the launch's (module
  docstring); `window_left` is the inclusive window (-1 for none),
  static per kind. Geometry is read from the tensors: heads and
  head_dim from `q`, kv heads and the page size from `pages`.

  `span_indptr` [n + 1] and `span_bounds` [S, 2] are the optional
  absolute image spans the sliding kind reads (module docstring);
  both or neither, and None is the no-span launch."""
  rows, heads, d = q.shape
  assert q.dtype == torch.bfloat16 and pages.dtype == torch.bfloat16
  nkv = pages.shape[3]
  group = heads // nkv
  PAGE = pages.shape[2]
  n_seq = qo_indptr.shape[0] - 1
  if span_indptr is None and span_bounds is None:
    span_indptr, span_bounds = _empty_spans(n_seq, q.device)
  elif span_indptr is None or span_bounds is None:
    raise ValueError('span_indptr and span_bounds come together')
  elif span_indptr.shape[0] < n_seq + 1 or span_bounds.dim() != 2 or \
      span_bounds.shape[1] != 2:
    # Shapes only, no device read: a short indptr is an out-of-bounds
    # load in every program.
    raise ValueError(
        f'gemma_attention: span_indptr {tuple(span_indptr.shape)} for '
        f'{n_seq} sequences (needs >= {n_seq + 1}), span_bounds '
        f'{tuple(span_bounds.shape)} (needs [S, 2])')
  cfg = _CFG[(nkv, d)]
  split = rows <= SPLIT_MAX_ROWS
  # A row count above 2 is worth a wider split tile paid for by cutting
  # the output columns: it trades a re-read of K for a re-read of K+V.
  # At 1-2 rows there is one tile either way, so the trade is all cost.
  # The wide tile pays only where the sequence is long enough that
  # halving the re-reads beats the extra pass over K. At 4k the gather
  # is parallelism-starved, not bandwidth-starved, and it costs 19%.
  # Both tests are on shapes, and neither row tile nor DSPLIT is in the
  # fold, so this cannot move a single output byte.
  wide = (split and rows > 2 and cfg['wide_rows_tile'] is not None and
          kv_indices.shape[0] * PAGE > WIDE_MIN_KEYS)
  dsplit = cfg['wide_dsplit'] if wide else 1
  rt = (cfg['wide_rows_tile']
        if wide else cfg['split_rows_tile'] if split else cfg['rows_tile'])
  bm = rt * group
  k_ptr = pages[:, 0]
  v_ptr = pages[:, 1]
  n_tiles = triton.cdiv(rows, rt)
  # The fused path's `num_stages` is not in the fold. When the whole gather's
  # K and V fit in L2 the async-copy pipeline only costs: its prefetch is
  # never rewarded, and its commit/wait/barrier bookkeeping is paid per leaf.
  # Measured on the scored cells: -4.0% at kv 8192 (16.8 MB of K/V against a
  # 24 MiB L2) but +22% at 32k and 60k, which stream. `kv_indices.shape[0]`
  # is a shape, so this is a static rule and never reads device metadata.
  kv_bytes = 2 * kv_indices.shape[0] * PAGE * d * 2
  fused_stages = cfg['num_stages'] if kv_bytes > _l2_bytes(q.device) else 1
  common = dict(GROUP=group, D=d, WINDOW_LEFT=window_left, PAGE_=PAGE,
                BLOCK_=cfg['block'], ROWS_TILE=rt, BM=bm,
                num_warps=cfg['num_warps'],
                num_stages=cfg['split_stages'] if split else fused_stages)

  if not split:
    _attn_fused[(n_tiles, n_seq,
                 nkv)](q, out, k_ptr, v_ptr, qo_indptr, kv_indptr, kv_indices,
                       kv_last_page_len, pos_base, span_indptr, span_bounds,
                       q.stride(0),
                       q.stride(1), pages.stride(0), pages.stride(2),
                       pages.stride(3), LEAF_=cfg['leaf'],
                       FIXED=cfg['fixed_base'], **common)
    return out

  # Split the kv range. The block count is bounded by the gather's own
  # size, a shape — no device metadata is read on the host. A program
  # whose block is past its tile's span exits.
  n_blk = triton.cdiv(kv_indices.shape[0] * PAGE, cfg['block']) + 1
  n_slot = n_seq * nkv * n_tiles * n_blk
  ws_acc = torch.empty(n_slot, bm, d, dtype=torch.float32, device=q.device)
  ws_m = torch.empty(n_slot, bm, dtype=torch.float32, device=q.device)
  ws_l = torch.empty(n_slot, bm, dtype=torch.float32, device=q.device)
  _attn_split[(n_blk, n_tiles * dsplit,
               n_seq * nkv)](q, ws_m, ws_l, ws_acc, k_ptr, v_ptr, qo_indptr,
                             kv_indptr, kv_indices, kv_last_page_len,
                             pos_base, span_indptr, span_bounds, q.stride(0),
                             q.stride(1), pages.stride(0), pages.stride(2),
                             pages.stride(3), n_tiles, nkv, LEAF_=cfg['leaf'],
                             DSPLIT=dsplit, FIXED=cfg['fixed_base'], **common)
  dchunk = cfg['dchunk']
  _attn_reduce[(n_tiles, n_seq * nkv,
                d // dchunk)](out, ws_m, ws_l, ws_acc, qo_indptr, kv_indptr,
                              kv_last_page_len,
                              pos_base, span_indptr, span_bounds, q.stride(0),
                              q.stride(1), n_tiles, nkv, n_blk, DCHUNK=dchunk,
                              **common)
  return out
