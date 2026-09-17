# qsa-select — the QSA indexer's masked top-k (sparkle task next.40)

The only bucket in a serving prefill chunk that grows with context: the indexer scores a `[4096, ctx/4]` fp32 matrix per QSA layer and the baseline selection walks it ~4 more times (masked_fill, two radix passes plus a gather in torch.topk, the -1 fill) — ~13 ms per layer at 66k context, ~470 ms of a ~2.2 s chunk at 180k across 14 layers. One read of the matrix is ~1.1/~3.1 ms at the box's ~237 GB/s.

- `qsa_select.py` — the op: `masked_topk(logits, visible, out)`. Baseline is the production four-pass torch glue.
- `shapes.py` — the real widths (66k and 180k context) and the per-row visible bound.
- `test_select.py` — exact set equality against a stable-sort reference: the tie rule (lowest index among tied scores), the bound rule (NaN scratch past `visible` never read), determinism over 8 calls, CUDA-graph capture.
- `bench.py` — `select_ms_mean`, the mean over the two contexts of median per-call ms.

The contract is exact: any kernel returning a different valid top-k set fails validation. There is no tolerance to widen.
