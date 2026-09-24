"""One layer's recurrence call at the four served launches.

    python bench.py            # all four shapes, prints a table
    python bench.py --quick    # t1 and t5 only, fewer rounds
    python bench.py --json out.json

For each shape the inputs are built once (seed 0), then `gdn_recurrence`
is timed by CUDA events around a batch of back-to-back calls (20 at
T=1 and T=5, 1 at the chunk — the engine replays decode and verify
under CUDA graphs with no launch gaps, and a single event pair around
a 20 µs launch would time the gap); the per-shape number is the median
per-call time over the rounds. The metric of record is `rec_score`: the
geometric mean of the four per-shape times, each divided by its
baseline (the shipped kernel measured on this box, BASELINE_US below),
so 1.0 is the baseline and lower is better. The four shapes weigh
equally: a candidate that wins the verify pass and loses the chunk does
not win. Quick mode's `rec_score` is the geomean over t1 and t5 only.

Floors on this box, for reference: writing 15 MiB with `fill_` takes
71.9 µs (t5 writes five 3 MiB states, the kernel's five writes cost
75.8 µs over the no-states launch's 23.1); a 3 MiB copy runs at L2
speed, 6.3 µs. The chunk runs 0.56 µs per row-step on a serial chain
against an FMA-pipe floor of ~0.25. The pinned chunk at a grid that fits
one wave (HV=42) runs 5.05 ms against 4.76 plain.
"""

from __future__ import annotations

import argparse
import json

import torch

from data import make_inputs
from gdn_rec import gdn_recurrence
from shapes import SHAPES

# The shipped kernel on this box, seed 0, medians of three bench runs (2026-09-24).
BASELINE_US = {"t1": 20.6, "t5": 97.1, "chunk": 4536.0, "pinchunk": 7108.0}


def time_shape(shape, rounds):
  T, n, states, pin_at = shape
  q, k, v, g, b, h0 = make_inputs(0, T, n)
  for _ in range(3):  # warmup and compile
    gdn_recurrence(q, k, v, g, b, h0, states=states, pin_at=pin_at)
  torch.cuda.synchronize()
  batch = 20 if T < 8192 else 1
  samples = []
  for _ in range(rounds):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(batch):
      gdn_recurrence(q, k, v, g, b, h0, states=states, pin_at=pin_at)
    e.record()
    e.synchronize()
    samples.append(s.elapsed_time(e) * 1000.0 / batch)  # µs per call
  return sorted(samples)[len(samples) // 2]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--quick", action="store_true")
  ap.add_argument("--json")
  args = ap.parse_args()
  shapes = [s for s in SHAPES if not (args.quick and s[1][0] == 8192)]
  result = {}
  for name, shape in shapes:
    rounds = (20 if args.quick else 50) if shape[0] < 8192 else 7
    result[f"{name}_us"] = time_shape(shape, rounds)
  ratios = [result[f"{name}_us"] / BASELINE_US[name] for name, _ in shapes]
  score = 1.0
  for r in ratios:
    score *= r
  result["rec_score"] = score**(1.0 / len(ratios))
  for name, _ in shapes:
    print(f"{name:9s} {result[f'{name}_us']:12.1f} us   "
          f"x{result[f'{name}_us'] / BASELINE_US[name]:.3f} of baseline")
  print(f"rec_score {result['rec_score']:.4f}")
  if args.json:
    with open(args.json, "w") as f:
      json.dump(result, f, indent=2)


if __name__ == "__main__":
  main()
