"""The fp32 reference and the wheel.

`reference_rows(cache, seq_index, start, end)`: plain fp32 softmax
attention for the rows at absolute positions [start, end) of one
sequence — every key from the window start (or 0) through the row's
own position, scores in fp32 at sm_scale 1.0, fp32 softmax, fp32 P·V,
rounded once to bf16. The correctness side of the contract: a
candidate is held to within bf16 rounding of this, measured as the
wheel's own distance (test_attn.py).

`wheel_attention(cache, q, meta, out)`: FlashInfer's paged prefill
kernel on the engine's exact plan (causal, sm_scale 1.0, the sliding
kind's inclusive `window_left`). The bench's baseline and the
correctness comparator; not the contract — its bytes are the thing
being left (its kv-tile grid and CTA_TILE_Q move with the call).
"""

from __future__ import annotations

import torch

from shapes import HEADS, PAGE


def reference_rows(cache, seq_index, start, end):
  g = cache.g
  nkv, d, wl = g['kv_heads'], g['head_dim'], g['window_left']
  group = HEADS // nkv
  lo = 0 if wl < 0 else max(0, start - wl)
  k, v = cache.dense_kv(seq_index, lo, end)  # [L, nkv, d]
  k = k.float()
  v = v.float()
  q = cache.seqs[seq_index].q(start, end).float()  # [R, HEADS, d]
  R, L = end - start, end - lo
  # [R, nkv, group, L]
  s = torch.einsum('rhgd,lhd->rhgl', q.view(R, nkv, group, d), k)
  pos = torch.arange(start, end, device=q.device)[:, None]
  key = torch.arange(lo, end, device=q.device)[None, :]
  allowed = key <= pos
  if wl >= 0:
    allowed &= key >= pos - wl
  s = s.masked_fill(~allowed[:, None, None, :], float('-inf'))
  p = torch.softmax(s, dim=-1)
  o = torch.einsum('rhgl,lhd->rhgd', p, v)
  return o.reshape(R, HEADS, d).to(torch.bfloat16)


_WRAPPER = None


def _new_wrapper():
  import flashinfer  # noqa: PLC0415
  ws = torch.empty(128 << 20, device='cuda', dtype=torch.uint8)
  return flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, 'NHD')


def _plan(wr, cache, meta):
  g = cache.g
  qo, kvp, idx, last, _ = meta
  wr.plan(qo, kvp, idx, last, HEADS, g['kv_heads'], g['head_dim'], PAGE,
          causal=True, sm_scale=1.0, window_left=g['window_left'],
          q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
  wr.pages = cache.pages
  return wr


def wheel_wrapper(cache, meta):
  """A fresh wrapper planned for one call (the bench's rotated sets)."""
  return _plan(_new_wrapper(), cache, meta)


def wheel_attention(cache, q, meta, out):
  global _WRAPPER
  if _WRAPPER is None:
    _WRAPPER = _new_wrapper()
  wr = _plan(_WRAPPER, cache, meta)
  out.copy_(wr.run(q, cache.pages))
  return out
