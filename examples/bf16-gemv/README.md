# bf16-gemv

The decode-time bf16 linear of a 48-layer hybrid model: `y = x @ W.T` with 1 to 4 rows, called ~670 times per step over 7.2 GB of cold weights. The stock path (cuBLAS through torch) averages ~170 GB/s on a box that reads at 237 GB/s.

```bash
python -m pytest -q          # tolerance against an fp32 reference, all real shapes, M=1..4
python bench.py --quick      # M=4, cold weights, ms per pass
python bench.py              # M=1 and M=4; pass_ms_mean is the metric
```

- `gemv.py` — the function to optimize.
- `shapes.py` — the real per-pass shape mix.
- `test_gemv.py` — the contract: fp32 accumulation, 2 bf16 ulp tolerance, inputs untouched.
- `bench.py` — cold-weight timing with an L2 scrub between rounds.
