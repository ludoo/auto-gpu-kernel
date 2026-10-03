"""The op under optimization: `gemma_attention(q, pages, qo_indptr,
kv_indptr, kv_indices, kv_last_page_len, pos_base, window_left, out)`
— paged causal attention on gemma's two geometries (shapes.py), bf16
in and out, n >= 1 sequences per launch, the last `rows` positions of
each.

The contract (test_attn.py): a row's output bytes are a function of
the row alone — never of how many rows were in the call, where the
call started, which tile the row rode in, or which other sequences
shared the launch — and within bf16 rounding of the fp32 reference.

This starting point is the slowest kernel that holds the contract by
construction: one program per (row, kv head), the row's group heads
as the mma's M (2 of 16 on the sliding kind, all 16 on the global),
walking key leaves of LEAF keys on a grid anchored at absolute
position 0 (`pos_base` + gather index), in order, with an fp32 online
softmax folded in registers. No kv split, so decode on the global kind
walks 128k keys in one program. Only this file (and helper modules it
imports) may change.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from shapes import HEADS, PAGE

LEAF = 32
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _attn_kernel(
    q_ptr, out_ptr, k_ptr, v_ptr,
    qo_indptr, kv_indptr, kv_indices, kv_last, pos_base,
    stride_q_row, stride_q_head,
    stride_p_page, stride_p_tok, stride_p_head,
    N_SEQ: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    WINDOW_LEFT: tl.constexpr, PAGE_: tl.constexpr, LEAF_: tl.constexpr,
    BM: tl.constexpr,
):
  row = tl.program_id(0)
  kvh = tl.program_id(1)

  # the row's sequence (n is small and static)
  seq = 0
  for i in range(1, N_SEQ):
    seq += (row >= tl.load(qo_indptr + i)).to(tl.int32)
  qo_start = tl.load(qo_indptr + seq)
  qo_len = tl.load(qo_indptr + seq + 1) - qo_start
  kv_start = tl.load(kv_indptr + seq)
  n_pages = tl.load(kv_indptr + seq + 1) - kv_start
  kv_len = (n_pages - 1) * PAGE_ + tl.load(kv_last + seq)
  base = tl.load(pos_base + seq)
  pos_rel = kv_len - qo_len + (row - qo_start)  # gather index of self
  pos_abs = base + pos_rel
  if WINDOW_LEFT >= 0:
    lo_abs = tl.maximum(pos_abs - WINDOW_LEFT, 0)
  else:
    lo_abs = 0
  leaf_lo = lo_abs // LEAF_
  leaf_hi = pos_abs // LEAF_

  hm = tl.arange(0, BM)
  dm = tl.arange(0, D)
  head_ok = hm < GROUP
  q = tl.load(q_ptr + row * stride_q_row + (kvh * GROUP + hm)[:, None] *
              stride_q_head + dm[None, :], mask=head_ok[:, None], other=0.0)

  m_i = tl.full([BM], float('-inf'), tl.float32)
  l_i = tl.zeros([BM], tl.float32)
  acc = tl.zeros([BM, D], tl.float32)
  tm = tl.arange(0, LEAF_)
  for leaf in range(leaf_lo, leaf_hi + 1):
    key_abs = leaf * LEAF_ + tm
    gidx = key_abs - base
    ok = (key_abs >= lo_abs) & (key_abs <= pos_abs) & (gidx >= 0)
    page = tl.load(kv_indices + kv_start + gidx // PAGE_, mask=ok, other=0)
    off = (page * stride_p_page + (gidx % PAGE_) * stride_p_tok +
           kvh * stride_p_head)
    k = tl.load(k_ptr + off[:, None] + dm[None, :], mask=ok[:, None],
                other=0.0)  # [LEAF, D]
    v = tl.load(v_ptr + off[:, None] + dm[None, :], mask=ok[:, None],
                other=0.0)
    s = tl.dot(q, tl.trans(k)) * LOG2E  # [BM, LEAF]
    s = tl.where(ok[None, :], s, float('-inf'))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    m_i = m_new
  o = acc / l_i[:, None]
  tl.store(out_ptr + row * stride_q_row + (kvh * GROUP + hm)[:, None] *
           stride_q_head + dm[None, :], o.to(tl.bfloat16),
           mask=head_ok[:, None])


def gemma_attention(q, pages, qo_indptr, kv_indptr, kv_indices,
                    kv_last_page_len, pos_base, window_left, out):
  rows, heads, d = q.shape
  assert heads == HEADS and q.dtype == torch.bfloat16
  nkv = pages.shape[3]
  group = HEADS // nkv
  n_seq = qo_indptr.shape[0] - 1
  k_ptr = pages[:, 0]
  v_ptr = pages[:, 1]
  grid = (rows, nkv)
  _attn_kernel[grid](
      q, out, k_ptr, v_ptr, qo_indptr, kv_indptr, kv_indices,
      kv_last_page_len, pos_base, q.stride(0), q.stride(1),
      pages.stride(0), pages.stride(2), pages.stride(3),
      N_SEQ=n_seq, GROUP=group, D=d, WINDOW_LEFT=window_left, PAGE_=PAGE,
      LEAF_=LEAF, BM=16, num_warps=4, num_stages=1)
  return out
