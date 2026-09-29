"""The 96-call `down_inject` chain at decode's and verify's M with a
cold working set.

    python bench.py            # M=1 and M=5, prints a table
    python bench.py --quick    # fewer rounds
    python bench.py --json out.json

ROT distinct weights (ROT x 6.9 MB, past L2) are allocated once; a
round issues 96 back-to-back calls over 96 distinct weights, timed by
CUDA events, and the per-call number is the median over rounds of the
round's time / 96. The step runs the 96 layers' weights once each, so
every call streams its weight from DRAM; timing one weight in a loop
measures L2 instead (D.3 read 17.3 µs hot for a call that is 49 µs
cold). The metric of record is `thin_score`: the geometric mean over
M=1 and M=5 of per-call µs / the shipped kernel's µs on this box
(BASELINE_US); 1.0 is the baseline, lower is better.
"""
import argparse
import json

import torch

from mconst_thin import linear
from shapes import SERVED_M, SHAPES

N, K, CALLS = SHAPES['down_inject']
ROT = 128
# The shipped thin tile on this box, median of three runs (2026-09-30, three bench runs 47.3-47.6).
BASELINE_US = {1: 47.4, 5: 47.4}
# cuBLAS on the same cold chain, for reference only: 37.1 (M=1), 36.2 (M=5).
FLOOR_US = N * K * 2 / 237e3  # the weight's bytes at the DRAM read rate


def time_chain(M, rounds):
  g = torch.Generator(device='cuda').manual_seed(0)
  ws = [(torch.randn(N, K, device='cuda', generator=g) * 0.02).bfloat16()
        for _ in range(ROT)]
  x = (torch.randn(M, K, device='cuda', generator=g) * 0.5).bfloat16()
  for w in ws:  # warm the compile, and run every weight once
    linear(x, w, tile='thin')
  torch.cuda.synchronize()
  samples = []
  off = 0
  for _ in range(rounds):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for j in range(CALLS):
      linear(x, ws[(off + j) % ROT], tile='thin')
    e.record()
    e.synchronize()
    samples.append(s.elapsed_time(e) * 1000.0 / CALLS)
    off += CALLS
  return sorted(samples)[len(samples) // 2]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--quick', action='store_true')
  ap.add_argument('--json')
  args = ap.parse_args()
  rounds = 15 if args.quick else 40
  result = {}
  score = 1.0
  for M in SERVED_M:
    t = time_chain(M, rounds)
    result[f'm{M}_us'] = t
    score *= t / BASELINE_US[M]
    print(f'M={M}  {t:6.1f} us/call  x{t / BASELINE_US[M]:.3f} of baseline  '
          f'({N * K * 2 / t / 1e3:.0f} GB/s; floor {FLOOR_US:.1f} us)')
  result['thin_score'] = score ** (1.0 / len(SERVED_M))
  print(f"thin_score {result['thin_score']:.4f}")
  if args.json:
    with open(args.json, 'w') as f:
      json.dump(result, f, indent=2)


if __name__ == '__main__':
  main()
