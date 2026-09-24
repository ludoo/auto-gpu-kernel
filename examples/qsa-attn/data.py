"""Seeded inputs at the served shape: a 131k paged bf16 K/V cache with a
permuted block table, and per-row packed selections drawn at random
from the context. Shared by the tests and the bench; the reference and
the candidate see identical tensors.

`make_cache(seed)` builds the cache once (2 x 268 MB). `make_rows(seed,
rows, ...)` draws one call's inputs: the query, the packed selection
(`holes=True` gives some rows a short count and `-1` past it, as the
engine does near a sequence start), and `token_to_req` over `requests`
block-table entries.
"""

from __future__ import annotations

import torch

from shapes import BUDGET, CONTEXT, HEAD_DIM, HEADS, KV_HEADS, PAGE, PAGES, WIDTH


def make_cache(seed: int, requests: int = 1):
  g = torch.Generator(device="cuda").manual_seed(seed)
  k = (torch.randn(PAGES, PAGE, KV_HEADS, HEAD_DIM, device="cuda",
                   generator=g) * 0.5).bfloat16()
  v = (torch.randn(PAGES, PAGE, KV_HEADS, HEAD_DIM, device="cuda",
                   generator=g) * 0.5).bfloat16()
  # Each request owns a slice of the physical pages, in a permuted order.
  perm = torch.randperm(PAGES, device="cuda", generator=g).to(torch.int32)
  per = PAGES // requests
  table = perm[:per * requests].view(requests, per).contiguous()
  return k, v, table


def make_rows(seed: int, rows: int, requests: int = 1, holes: bool = False):
  g = torch.Generator(device="cuda").manual_seed(seed + 1_000_003 * rows)
  q = (torch.randn(rows, HEADS, HEAD_DIM, device="cuda", generator=g) *
       0.5).bfloat16()
  per = PAGES // requests
  tokens = per * PAGE  # tokens addressable through one request's table
  packed = torch.full((rows, WIDTH + 1), -1, dtype=torch.int32, device="cuda")
  token_to_req = (torch.arange(rows, device="cuda") % requests).to(torch.int32)
  # A random subset of BUDGET distinct tokens per row, cheaply: one
  # permutation of the context per seed, each row a random window of it,
  # sorted (the engine's selection is ascending).
  perm = torch.randperm(tokens, device="cuda", generator=g)
  start = torch.randint(0, tokens - BUDGET, (rows, 1), device="cuda",
                        generator=g)
  window = start + torch.arange(BUDGET, device="cuda")[None, :]
  sel = perm[window].sort(dim=1).values.to(torch.int32)
  packed[:, :BUDGET] = sel
  counts = torch.full((rows,), BUDGET, dtype=torch.int32, device="cuda")
  if holes:
    # Every eighth row is short: a count below the budget, -1 past it,
    # including one row with a single tile and one with zero.
    short = torch.arange(rows, device="cuda") % 8 == 3
    short_counts = torch.randint(1, BUDGET, (rows,), device="cuda",
                                 generator=g).to(torch.int32)
    counts = torch.where(short, short_counts, counts)
    if rows > 8:
      counts[3] = 17
      counts[11 % rows] = 0
    cols = torch.arange(WIDTH, device="cuda")[None, :]
    packed[:, :WIDTH] = torch.where(cols < counts[:, None], packed[:, :WIDTH],
                                    torch.full_like(packed[:, :WIDTH], -1))
  packed[:, WIDTH] = counts
  return q, packed, token_to_req
