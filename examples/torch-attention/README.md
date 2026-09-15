# torch-attention

A deliberately naive scaled dot-product attention in PyTorch, used as the development
example for `kopt init-task` on a local NVIDIA GPU. The CUDA twin of `mlx-attention`.

```bash
pip install torch triton pytest    # a venv with a CUDA torch
python -m pytest -q                # correctness vs. F.scaled_dot_product_attention
python bench.py --quick            # latency on one decode + one prefill shape
python bench.py                    # all workloads; geomean median ms is the metric
```

- `attention.py` — the function to optimize (`attention(q, k, v, scale=, causal=)`).
- `test_attention.py` — the contract it must keep.
- `bench.py` — decode (L=1) and causal prefill workloads in float16, timed with CUDA events.
