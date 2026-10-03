"""Seeded sequences on a paged bf16 cache in gemma's two geometries,
shared by the tests, the reference, the wheel and the bench.

`make_cache(kind, seed, specs)` builds one cache holding several
sequences: `specs` is a list of `(total, keep_from[, q_from])` — a sequence of
`total` positions whose K/V are materialised from position
`keep_from` (a multiple of 16) on and whose q from `q_from` (default
`keep_from`), so a 128k sequence costs only the rows a call touches. Physical pages are a random permutation across
the whole cache, so every call's `kv_indices` is a real gather. Each
sequence's K/V are stored at their absolute positions, pages
`table[pos // 16]`, slot `pos % 16`.

`Cache.meta(calls)` builds one launch's metadata for a list of
`(seq_index, rows, end)` calls — the last `rows` positions up to `end`
of each sequence — in the engine's FlashInfer shape plus `pos_base`
(shapes.py). `Cache.q(calls)` stacks the matching query rows.
"""

from __future__ import annotations

import torch

from shapes import HEADS, KINDS, PAGE, WINDOW

SCALE = 0.3


class Sequence:

  def __init__(self, total, keep_from, q_from, q_all, table):
    assert keep_from % PAGE == 0
    self.total = total
    self.keep_from = keep_from
    self.q_from = q_from
    self.q_all = q_all  # [total - q_from, HEADS, D]
    self.table = table  # int32 [ceil(total / PAGE)], valid from keep_from

  def q(self, start, end):
    assert start >= self.q_from and end <= self.total, (start, end)
    return self.q_all[start - self.q_from:end - self.q_from]


class Cache:

  def __init__(self, kind, pages, seqs):
    self.kind = kind
    self.g = KINDS[kind]
    self.pages = pages  # [n_phys, 2, PAGE, kv_heads, D]
    self.seqs = seqs

  def gather_range(self, seq_index, rows, end):
    """(first_page, used_pages, last_len) of the call's K/V."""
    s = self.seqs[seq_index]
    if self.kind == 'sliding':
      first = max(0, end - rows - WINDOW) // PAGE
    else:
      first = 0
    assert first * PAGE >= s.keep_from, (first, s.keep_from)
    used = (end + PAGE - 1) // PAGE
    return first, used, end - (used - 1) * PAGE

  def meta(self, calls, device='cuda'):
    qo = [0]
    kvp = [0]
    idx = []
    last = []
    base = []
    for si, rows, end in calls:
      first, used, last_len = self.gather_range(si, rows, end)
      qo.append(qo[-1] + rows)
      idx.append(self.seqs[si].table[first:used])
      kvp.append(kvp[-1] + used - first)
      last.append(last_len)
      base.append(first * PAGE)
    i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=device)  # noqa: E731
    return (i32(qo), i32(kvp), torch.cat(idx).to(torch.int32), i32(last),
            i32(base))

  def q(self, calls):
    return torch.cat([self.seqs[si].q(end - rows, end) for si, rows, end in calls])

  def dense_kv(self, seq_index, start, end):
    """K and V at absolute positions [start, end) as dense
    [end - start, kv_heads, D] tensors (for the reference)."""
    s = self.seqs[seq_index]
    pos = torch.arange(start, end, device=self.pages.device)
    phys = s.table[pos // PAGE].long()
    slot = pos % PAGE
    return self.pages[phys, 0, slot], self.pages[phys, 1, slot]


def make_cache(kind, seed, specs):
  g = KINDS[kind]
  gen = torch.Generator(device='cuda').manual_seed(seed * 7919 + len(specs))
  specs = [s if len(s) == 3 else (s[0], s[1], s[1]) for s in specs]
  n_pages = [(t + PAGE - 1) // PAGE - k // PAGE for t, k, _ in specs]
  n_phys = sum(n_pages)
  perm = torch.randperm(n_phys, device='cuda', generator=gen).to(torch.int32)
  # q and k are RMS-normed in the engine; SCALE puts the scores at
  # sm_scale 1.0 around std 1.4 (d 256) / 2.0 (d 512): a softmax with a
  # real dynamic range, not a flat one.
  pages = (torch.randn(n_phys, 2, PAGE, g['kv_heads'], g['head_dim'],
                       device='cuda', generator=gen) * SCALE).bfloat16()
  seqs = []
  off = 0
  for (total, keep_from, q_from), n in zip(specs, n_pages):
    q_all = (torch.randn(total - q_from, HEADS, g['head_dim'],
                         device='cuda', generator=gen) * SCALE).bfloat16()
    table = torch.full(((total + PAGE - 1) // PAGE,), -1, dtype=torch.int32,
                       device='cuda')
    table[keep_from // PAGE:] = perm[off:off + n]
    off += n
    seqs.append(Sequence(total, keep_from, q_from, q_all, table))
  return Cache(kind, pages, seqs)
