"""Time `attention` on representative decode and prefill shapes.

    python bench.py            # all workloads, prints a table
    python bench.py --quick    # one decode + one prefill workload
    python bench.py --json out.json

The metric of record is the geometric mean of per-workload median latency in
milliseconds (lower is better). Compilation and the first calls are excluded by warmup;
every timed call is bracketed by CUDA events on the current stream, so only GPU time is
measured and Python launch overhead outside the call is excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch
from attention import attention

# name -> (B, H_q, H_kv, L, S, D, causal)
WORKLOADS = {
    "decode-s1k": (1, 32, 8, 1, 1024, 128, False),
    "decode-s8k": (1, 32, 8, 1, 8192, 128, False),
    "decode-b8-s4k": (8, 32, 8, 1, 4096, 128, False),
    "prefill-512": (1, 32, 8, 512, 512, 128, True),
    "prefill-2k": (1, 32, 8, 2048, 2048, 128, True),
}
QUICK = ("decode-s1k", "prefill-512")


def make_inputs(B, H_q, H_kv, L, S, D, dtype=torch.float16, device="cuda"):
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn((B, H_q, L, D), generator=g, device=device, dtype=dtype)
    k = torch.randn((B, H_kv, S, D), generator=g, device=device, dtype=dtype)
    v = torch.randn((B, H_kv, S, D), generator=g, device=device, dtype=dtype)
    return q, k, v


def time_workload(spec, *, warmup: int, iters: int) -> list[float]:
    *dims, causal = spec
    q, k, v = make_inputs(*dims)
    for _ in range(warmup):
        attention(q, k, v, causal=causal)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        attention(q, k, v, causal=causal)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--json")
    args = ap.parse_args()

    names = QUICK if args.quick else tuple(WORKLOADS)
    per = {}
    for name in names:
        samples = time_workload(WORKLOADS[name], warmup=args.warmup, iters=args.iters)
        per[name] = {
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "samples_ms": samples,
        }
        print(f"{name:>16}  median {per[name]['median_ms']:8.3f} ms"
              f"  min {per[name]['min_ms']:8.3f}  max {per[name]['max_ms']:8.3f}")

    geomean = math.exp(sum(math.log(w["median_ms"]) for w in per.values()) / len(per))
    print(f"\ngeomean median latency: {geomean:.3f} ms  (lower is better)")
    print(f"device: {torch.cuda.get_device_name()}  sm_{''.join(map(str, torch.cuda.get_device_capability()))}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"geomean_median_ms": geomean, "workloads": per}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
