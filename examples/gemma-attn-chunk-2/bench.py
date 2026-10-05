"""One layer's attention call per kind at the served row counts and
contexts, cold rotated working set, against the starting kernel.

    python bench.py              # every cell, prints a table
    python bench.py --quick      # the three scored chunk cells only
    python bench.py --json out.json
    python bench.py --wheel      # time FlashInfer's plan instead
    python bench.py --start      # re-derive BASELINE_US (the starting kernel)

Per cell (shapes.py BENCH_CELLS) the inputs are S rotated sets whose
bytes exceed 4x the part's L2 (one input set per shape measures the
cache, not the engine), the call is timed by CUDA events around a batch
of back-to-back calls cycling the sets (20 at rows < 4096, 1 at the
chunk), and the figure is the median per-call µs over the rounds.

The metric of record is `chunk_score`, lower is better:

    geomean over the SCORED cells (the global chunk at kv 8k / 32k /
    60k) of candidate µs / BASELINE_US
  × geomean over the HELD cells (every other cell) of
    max(1.0, candidate µs / BASELINE_US)

BASELINE_US is the STARTING kernel's own figure on this box, so the
start scores 1.0. A held cell cannot lower the score: decode, verify
and the sliding kind keep their price or better and buy nothing.
WHEEL_US, FlashInfer's plan on the same bench, is printed beside each
cell for the record and enters nothing.

The figures assume one launch at a time, steady state (the floor's
assumption, named). They are per layer; a prefill chunk pays 8 global
and 40 sliding calls.
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from data import make_cache
from gemma_attn import gemma_attention
from reference import wheel_wrapper
from shapes import BENCH_CELLS, KINDS, PAGE, SCORED, WINDOW

# The starting kernel — round 1's survivor (work/gemma-attn-chunk-dsflash,
# repo `d5a4aed`: the fixed block-grain fold and the L2-residency stages
# rule on the global kind) — through this bench on this box, median
# per-call µs. 2026-10-06, `--start`, sparkle.service stopped.
BASELINE_US = {
    'sliding/decode/32768': 45.3,
    'sliding/verify/32768': 45.4,
    'sliding/chunk/32768': 1579.4,
    'global/decode/4096': 49.3,
    'global/decode/32768': 325.4,
    'global/decode/131072': 1328.9,
    'global/verify/4096': 51.7,
    'global/verify/32768': 591.2,
    'global/verify/131072': 2394.5,
    'global/chunk/8192': 23821.7,
    'global/chunk/32768': 125959.1,
    'global/chunk/61440': 244555.9,
}

# FlashInfer's plan through this bench (`--wheel`), for the record.
WHEEL_US = {
    'sliding/decode/32768': 46.7,
    'sliding/verify/32768': 48.3,
    'sliding/chunk/32768': 1126.0,
    'global/decode/4096': 51.0,
    'global/decode/32768': 308.1,
    'global/decode/131072': 1155.4,
    'global/verify/4096': 74.8,
    'global/verify/32768': 600.0,
    'global/verify/131072': 2284.0,
    'global/chunk/8192': 22659.6,
    'global/chunk/32768': 122787.1,
    'global/chunk/61440': 236684.5,
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
  caches = [
      make_cache(kind, 100 + s, [(ctx, keep, ctx - rows)]) for s in range(sets)
  ]
  calls = [(0, rows, ctx)]
  inputs = [(c, c.q(calls), c.meta(calls),
             torch.empty(rows, 16, KINDS[kind]['head_dim'],
                         dtype=torch.bfloat16, device='cuda')) for c in caches]
  wl = KINDS[kind]['window_left']
  if wheel:
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


def geomean(xs):
  return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 1.0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--quick', action='store_true')
  ap.add_argument('--json')
  ap.add_argument('--wheel', action='store_true')
  ap.add_argument('--start', action='store_true',
                  help='print the table only (re-deriving BASELINE_US)')
  args = ap.parse_args()
  result = {}
  scored, held = [], []
  for kind, cells in BENCH_CELLS.items():
    for phase, rows, ctx in cells:
      key = f'{kind}/{phase}/{ctx}'
      if args.quick and key not in SCORED:
        continue
      rounds = (10 if args.quick else 30) if rows < 4096 else 5
      us = time_cell(kind, rows, ctx, args.wheel, rounds)
      result[f'{key}_us'] = us
      base = BASELINE_US[key]
      w = WHEEL_US[key]
      wtxt = f'   wheel {w:10.1f} us' if w else ''
      if args.start or args.wheel or base is None:
        print(f'{key:24s} {us:12.1f} us{wtxt}')
        continue
      r = us / base
      if key in SCORED:
        scored.append(r)
        tag = 'scored'
      else:
        held.append(max(1.0, r))
        tag = 'held' if r <= 1.0 else 'HELD, SLOWER'
      print(f'{key:24s} {us:12.1f} us   x{r:.3f} of the start  {tag}{wtxt}')
  if not (args.start or args.wheel):
    result['chunk_score'] = geomean(scored) * geomean(held)
    print(f"chunk_score {result['chunk_score']:.4f}  "
          f'(scored {geomean(scored):.4f} x held {geomean(held):.4f})')
  if args.json:
    with open(args.json, 'w') as f:
      json.dump(result, f, indent=2)


if __name__ == '__main__':
  main()
