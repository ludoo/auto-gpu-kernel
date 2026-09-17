"""Selection time per prefill chunk at the real widths.

    python bench.py            # both contexts, prints a table
    python bench.py --quick    # ctx66k only, 2 rounds
    python bench.py --json out.json

For each context the score matrix [T=4096, cols] fp32 is produced once
(realistic content: sparse non-negative scores, NaN scratch past each
row's visible bound), then `masked_topk` is timed by CUDA events over
the whole call. The metric of record is `select_ms_mean`: the mean over
the two contexts of the median per-call milliseconds (lower is better).

Baseline (the four-pass torch glue) is ~13 ms at ctx66k and ~30+ ms at
ctx180k per production traces. One read of the matrix at the box's
~237 GB/s practical DRAM bandwidth is ~1.1 ms (270 MB) and ~3.1 ms
(738 MB) respectively — the floor for a one-pass kernel.
"""

from __future__ import annotations

import argparse
import json

import torch

from qsa_select import masked_topk
from shapes import K, SHAPES, T, visible


def make_logits(base, cols):
    g = torch.Generator(device="cuda").manual_seed(base)
    x = torch.randn(T, cols, device="cuda", generator=g).abs()
    keep = torch.rand(T, cols, device="cuda", generator=g) < 0.05
    logits = torch.where(keep, x, torch.zeros_like(x))
    vis = visible(base, T)
    columns = torch.arange(cols, device="cuda")[None, :]
    return logits.masked_fill(columns >= vis[:, None], float("nan")), vis


def time_ctx(base, cols, rounds):
    logits, vis = make_logits(base, cols)
    out = torch.empty(T, K, dtype=torch.int32, device="cuda")
    for _ in range(3):  # warmup and compile
        masked_topk(logits, vis, out)
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        masked_topk(logits, vis, out)
        e.record()
        e.synchronize()
        samples.append(s.elapsed_time(e))
    return sorted(samples)[len(samples) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--json")
    args = ap.parse_args()
    shapes = SHAPES[:1] if args.quick else SHAPES
    rounds = 5 if args.quick else args.rounds
    per = {}
    for name, base, cols in shapes:
        per[name] = time_ctx(base, cols, rounds)
        print(f"{name:8s} cols={cols:6d}  {per[name]:8.3f} ms")
    metric = sum(per.values()) / len(per)
    print(f"select_ms_mean {metric:.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"select_ms_mean": metric, "per_context": per}, f, indent=2)


if __name__ == "__main__":
    main()
