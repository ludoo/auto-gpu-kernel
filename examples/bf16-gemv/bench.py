"""Cold-weight benchmark: milliseconds per full decode pass.

    python bench.py            # M=1 and M=4, all shapes, prints a table
    python bench.py --quick    # M=4 only, 2 rounds
    python bench.py --json out.json

For each shape, `count` distinct weights are allocated (as in the real
model) and called in sequence after a 64 MiB scrub of L2, so every
weight streams from DRAM exactly as it does in serving. The metric of
record is `pass_ms_mean`: the mean over M of the summed per-shape
times (lower is better). Timing is by CUDA events around the whole
per-shape sequence; per-call host overhead is included, as it is in
serving (the model runs these under CUDA graphs, so a kernel that
needs host-side work per call is at a disadvantage in production too).

Practical DRAM read bandwidth on this box is ~237 GB/s (torch.sum over
512 MiB), so the floor for the 7.2 GB pass is about 30.4 ms.
"""

from __future__ import annotations

import argparse
import json

import torch
from gemv import gemv
from shapes import SHAPES

SCRUB = 64 << 20


def time_shape(N, K, count, M, rounds):
    ws = [torch.randn((N, K), device="cuda", dtype=torch.bfloat16) for _ in range(count)]
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    scrub = torch.empty(SCRUB, device="cuda", dtype=torch.uint8)
    for w in ws:  # warmup and compile
        gemv(x, w)
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        scrub.fill_(1)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for w in ws:
            gemv(x, w)
        e.record()
        e.synchronize()
        samples.append(s.elapsed_time(e))
    del ws
    return sorted(samples)[len(samples) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--json")
    args = ap.parse_args()
    ms_list = [4] if args.quick else [1, 4]
    rounds = 2 if args.quick else args.rounds
    out = {"per_m": {}}
    for M in ms_list:
        total = 0.0
        rows = {}
        print(f"M={M}\n{'ms/pass':>8} {'GB/s':>6}  shape")
        for N, K, count, name in SHAPES:
            ms = time_shape(N, K, count, M, rounds)
            gbs = N * K * 2 * count / ms / 1e6
            rows[name] = {"N": N, "K": K, "count": count, "ms": ms, "gbs": gbs}
            total += ms
            print(f"{ms:8.2f} {gbs:6.0f}  [{N},{K}] x{count} {name}")
        print(f"pass total M={M}: {total:.2f} ms\n")
        out["per_m"][M] = {"pass_ms": total, "shapes": rows}
    mean = sum(v["pass_ms"] for v in out["per_m"].values()) / len(out["per_m"])
    out["pass_ms_mean"] = mean
    print(f"pass_ms_mean: {mean:.2f} ms  (lower is better; floor ~30.4)")
    print(f"device: {torch.cuda.get_device_name()}  sm_{''.join(map(str, torch.cuda.get_device_capability()))}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
