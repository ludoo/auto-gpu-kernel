"""One layer's attention call at the three served row counts on a 131k
paged cache with random selections.

    python bench.py            # all three shapes, prints a table
    python bench.py --quick    # M=1 and M=5 only, fewer rounds
    python bench.py --json out.json

For each shape the inputs are built once (seed 0), then `qsa_attention`
is timed by CUDA events around a batch of back-to-back calls (20 at
M=1 and M=5, 1 at the chunk — the engine replays decode under CUDA
graphs with no launch gaps, and a single event pair around a 100 µs
launch would time the gap); the per-shape number is the median
per-call time over the rounds. The metric of record is `attn_score`: the
geometric mean of the three per-shape times, each divided by its
baseline (the shipped kernel measured on this box, BASELINE_US below),
so 1.0 is the baseline and lower is better. The three shapes weigh
equally: a candidate that wins decode and loses the chunk does not win.
Quick mode's `attn_score` is the geomean over M=1 and M=5 only.

The floors on this box, for reference (the table's split rows of the
same kernel, which do not reproduce the bytes): M=1 29.9 µs (64 splits,
4 warps), M=5 41.2 µs (8 splits, 1 warp). At the chunk the reference on
a contiguous selection with an identity page table runs 28.4 ms against
78.0 random: the gather is the cost there, the arithmetic is not.
"""

from __future__ import annotations

import argparse
import json

import torch

from data import make_cache, make_rows
from qsa_attn import qsa_attention
from shapes import HEAD_DIM, HEADS, SHAPES

# The shipped kernel on this box, seed 0, medians of three bench runs (2026-09-24).
BASELINE_US = {"m1": 107.3, "m5": 133.4, "chunk": 78000.0}


def time_shape(cache, rows, rounds):
  k, v, table = cache
  q, packed, t2r = make_rows(0, rows)
  out = torch.zeros(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
  for _ in range(3):  # warmup and compile
    qsa_attention(q, k, v, packed, table, t2r, out)
  torch.cuda.synchronize()
  batch = 20 if rows < 8192 else 1
  samples = []
  for _ in range(rounds):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(batch):
      qsa_attention(q, k, v, packed, table, t2r, out)
    e.record()
    e.synchronize()
    samples.append(s.elapsed_time(e) * 1000.0 / batch)  # µs per call
  return sorted(samples)[len(samples) // 2]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--quick", action="store_true")
  ap.add_argument("--json")
  args = ap.parse_args()
  cache = make_cache(0)
  shapes = [s for s in SHAPES if not (args.quick and s[1] == 8192)]
  result = {}
  for name, rows in shapes:
    rounds = (20 if args.quick else 50) if rows < 8192 else 10
    result[f"{name}_us"] = time_shape(cache, rows, rounds)
  ratios = [result[f"{name}_us"] / BASELINE_US[name] for name, _ in shapes]
  score = 1.0
  for r in ratios:
    score *= r
  result["attn_score"] = score**(1.0 / len(ratios))
  for name, _ in shapes:
    print(f"{name:6s} {result[f'{name}_us']:12.1f} us   "
          f"x{result[f'{name}_us'] / BASELINE_US[name]:.3f} of baseline")
  print(f"attn_score {result['attn_score']:.4f}")
  if args.json:
    with open(args.json, "w") as f:
      json.dump(result, f, indent=2)


if __name__ == "__main__":
  main()
