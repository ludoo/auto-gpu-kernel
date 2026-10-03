"""One layer's attention call per kind at the served row counts and
contexts, cold rotated working set, against FlashInfer's plain plan.

    python bench.py              # every cell, prints a table
    python bench.py --quick      # decode and verify cells only
    python bench.py --json out.json
    python bench.py --wheel      # time the wheel instead (re-derive BASELINE_US)

Per cell (shapes.py BENCH_CELLS) the inputs are S rotated sets whose
bytes exceed 4x the part's L2 (one input set per shape measures the
cache, not the engine), the call is timed by CUDA events around a batch
of back-to-back calls cycling the sets (20 at rows < 4096, 1 at the
chunk — the engine replays decode and verify under CUDA graphs with no
launch gaps), and the figure is the median per-call µs over the rounds.

The metric of record is `attn_score`: the geometric mean over every
cell of (candidate µs / the wheel's µs on this box, BASELINE_US), so
1.0 is par with the wheel and lower is better. Both kinds and all three
phases weigh equally: a kernel that wins decode and loses the chunk
does not win. Quick mode's score is the geomean over the decode and
verify cells only.

The wheel's figures assume one launch at a time, steady state (the
floor's assumption, named). They are per layer; the step pays 40
sliding + 8 global decode calls, the chunk's TTFT the same per chunk.
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from data import make_cache
from gemma_attn import gemma_attention
from reference import wheel_wrapper
from shapes import BENCH_CELLS, KINDS, PAGE, QUICK_PHASES, WINDOW

# FlashInfer's plain plans through this bench (`--wheel`, permuted
# pages, rotated sets) on this box, 2026-10-03, median per-call µs. The
# same plans on an `arange` page table run 2-15% faster at decode and
# verify (sparkle docs/history/archive/next120/attention_floor.log).
BASELINE_US = {
    'sliding/decode/32768': 46.6,
    'sliding/verify/32768': 47.9,
    'sliding/chunk/32768': 1116.9,
    'global/decode/4096': 50.1,
    'global/decode/32768': 303.9,
    'global/decode/131072': 1137.9,
    'global/verify/4096': 74.3,
    'global/verify/32768': 590.5,
    'global/verify/131072': 2246.4,
    'global/chunk/4096': 7842.0,
    'global/chunk/32768': 121890.8,
    'global/chunk/131072': 516090.5,
}


def cell_bytes(kind, rows, ctx):
  g = KINDS[kind]
  first = 0 if kind == 'global' else max(0, ctx - rows - WINDOW) // PAGE
  pages = (ctx + PAGE - 1) // PAGE - first
  return (pages * 2 * PAGE * g['kv_heads'] * g['head_dim'] * 2 +
          rows * 16 * g['head_dim'] * 2)


def time_cell(kind, rows, ctx, wheel, rounds):
  l2 = torch.cuda.get_device_properties(0).L2_cache_size
  sets = min(16, max(2, math.ceil(4 * l2 / cell_bytes(kind, rows, ctx))))
  keep = 0 if kind == 'global' else max(0, ctx - rows - WINDOW - 16) // 16 * 16
  caches = [make_cache(kind, 100 + s, [(ctx, keep, ctx - rows)]) for s in range(sets)]
  calls = [(0, rows, ctx)]
  inputs = [(c, c.q(calls), c.meta(calls), torch.empty(rows, 16, KINDS[kind]['head_dim'],
                                                       dtype=torch.bfloat16, device='cuda'))
            for c in caches]
  wl = KINDS[kind]['window_left']
  if wheel:
    # one planned wrapper per set: the plan carries the set's page ids
    wrappers = [wheel_wrapper(c, m) for c, _, m, _ in inputs]
    inputs = [(w, q, m, o) for w, (_, q, m, o) in zip(wrappers, inputs)]
    run = lambda w, q, m, o: w.run(q, w.pages)  # noqa: E731
  else:
    run = lambda c, q, m, o: gemma_attention(q, c.pages, *m, wl, o)  # noqa: E731
  for c, q, m, o in inputs:
    run(c, q, m, o)
  torch.cuda.synchronize()
  batch = 20 if rows < 4096 else 1
  samples = []
  i = 0
  for _ in range(rounds):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(batch):
      c, q, m, o = inputs[i % sets]
      i += 1
      run(c, q, m, o)
    e.record()
    e.synchronize()
    samples.append(s.elapsed_time(e) * 1000.0 / batch)
  del inputs, caches
  torch.cuda.empty_cache()
  return sorted(samples)[len(samples) // 2]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--quick', action='store_true')
  ap.add_argument('--json')
  ap.add_argument('--wheel', action='store_true')
  args = ap.parse_args()
  result = {}
  ratios = []
  for kind, cells in BENCH_CELLS.items():
    for phase, rows, ctx in cells:
      if args.quick and phase not in QUICK_PHASES:
        continue
      key = f'{kind}/{phase}/{ctx}'
      rounds = (10 if args.quick else 30) if rows < 4096 else 5
      us = time_cell(kind, rows, ctx, args.wheel, rounds)
      result[f'{key}_us'] = us
      r = us / BASELINE_US[key]
      ratios.append(r)
      print(f'{key:24s} {us:12.1f} us   x{r:.3f} of the wheel')
  result['attn_score'] = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
  print(f"attn_score {result['attn_score']:.4f}")
  if args.json:
    with open(args.json, 'w') as f:
      json.dump(result, f, indent=2)


if __name__ == '__main__':
  main()
