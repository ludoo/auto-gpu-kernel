"""One linear call per shape at M = 1 and 4 (decode and verify), cold rotated
weights, against the wheel's Marlin. The decode round: the chunk cells are
not scored here (the prefill tile is frozen; the full 18-cell bench in
../w4a16-gemm is re-run on the survivor).

    python bench.py              # every cell, prints a table
    python bench.py --json out.json
    python bench.py --wheel      # time the wheel instead (re-derive BASELINE_US)

Per cell (shape x M in {1, 4, 4096}) the inputs are S rotated sets —
a distinct weight per set, since at M <= 4 the weight is the working
set and one weight per shape measures L2, not the step — whose bytes
exceed 4x the part's L2; the call is timed by CUDA events around a
batch of back-to-back calls cycling the sets (20 at M <= 4, 1 at the
chunk), median per-call µs over the rounds.

The metric of record is `gemm_score`: the geometric mean over every
cell of (candidate µs / the wheel's µs on this box, BASELINE_US); 1.0
is par, lower is better. Every cell weighs equally: decode's bandwidth
end and the chunk's tensor-pipe end both count. Quick mode's score is
the geomean over the M=1 and M=4 cells only.

The wheel's figures assume one launch at a time, steady state (the
floor's assumption, named). They are per call; a sliding layer pays
q + 2k + o + 2 gate/up + down per step.
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from data import make_weight, make_x
from reference import wheel
from shapes import BENCH_MS, GROUP, QUICK_MS, SHAPES
from w4a16_gemm import w4a16_gemm

# The wheel's Marlin through this bench on this box, 2026-10-04, median
# per-call µs (sparkle docs/history/archive/next120/marlin_floor.log).
BASELINE_US = {
    'q_proj/M1': 41.1, 'q_proj/M4': 40.4, 'q_proj/M4096': 1388.1,
    'k_proj_sliding/M1': 23.8, 'k_proj_sliding/M4': 23.9, 'k_proj_sliding/M4096': 708.4,
    'k_proj_global/M1': 15.5, 'k_proj_global/M4': 15.6, 'k_proj_global/M4096': 231.4,
    'o_proj/M1': 39.7, 'o_proj/M4': 41.7, 'o_proj/M4096': 1376.6,
    'gate_up/M1': 136.3, 'gate_up/M4': 137.6, 'gate_up/M4096': 8474.1,
    'down/M1': 137.3, 'down/M4': 135.2, 'down/M4096': 5037.9,
}


def time_cell(n, k, m, use_wheel, rounds):
  l2 = torch.cuda.get_device_properties(0).L2_cache_size
  per_set = n * k // 2 + n * (k // GROUP) * 2 + m * k * 2 + m * n * 2
  sets = min(16, max(2, math.ceil(4 * l2 / per_set)))
  inputs = []
  for s in range(sets):
    packed, scale = make_weight(n, k, 100 + s)
    x = make_x(m, k, 200 + s)
    out = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
    inputs.append((x, packed, scale, out))
  if use_wheel:
    run = lambda x, p, s, o: wheel(x, p, s)  # noqa: E731
  else:
    run = lambda x, p, s, o: w4a16_gemm(x, p, s, o)  # noqa: E731
  for x, p, s, o in inputs:
    run(x, p, s, o)
  torch.cuda.synchronize()
  batch = 20 if m < 4096 else 1
  samples = []
  i = 0
  for _ in range(rounds):
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(batch):
      x, p, s, o = inputs[i % sets]
      i += 1
      run(x, p, s, o)
    en.record()
    en.synchronize()
    samples.append(st.elapsed_time(en) * 1000.0 / batch)
  del inputs
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
  for name, (n, k, _) in SHAPES.items():
    for m in BENCH_MS:
      if args.quick and m not in QUICK_MS:
        continue
      key = f'{name}/M{m}'
      rounds = (10 if args.quick else 30) if m < 4096 else 10
      us = time_cell(n, k, m, args.wheel, rounds)
      result[f'{key}_us'] = us
      r = us / BASELINE_US[key]
      ratios.append(r)
      print(f'{key:24s} {us:12.1f} us   x{r:.3f} of the wheel')
  result['gemm_score'] = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
  print(f"gemm_score {result['gemm_score']:.4f}")
  if args.json:
    with open(args.json, 'w') as f:
      json.dump(result, f, indent=2)


if __name__ == '__main__':
  main()
