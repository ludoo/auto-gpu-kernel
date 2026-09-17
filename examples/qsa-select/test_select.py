"""Exact validation: the selected set equals a stable-sort reference —
no tolerance, pass or fail. The tie rule (lowest index first among tied
scores) and the bound rule (scratch past `visible` is never read as a
score) are the contract; a kernel that returns any other valid top-k
set FAILS.
"""

import pytest
import torch

from qsa_select import masked_topk
from shapes import K

torch.manual_seed(0)


def tied_logits(rows, cols):
    """The indexer's real score shape: fp32 sum_heads relu(q.k), most
    columns exact zero — the rank-k boundary sits inside a tie."""
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", generator=g).abs()
    keep = torch.rand(rows, cols, device="cuda", generator=g) < 0.05
    return torch.where(keep, x, torch.zeros_like(x))


def stable_topk(logits, visible, k):
    """The canonical selection: descending score, lowest index first
    among ties, -1 past the visible count; returned ascending per row."""
    cols = torch.arange(logits.shape[1], device=logits.device)[None, :]
    masked = logits.masked_fill(cols >= visible[:, None], float("-inf"))
    idx = torch.sort(masked, dim=1, descending=True, stable=True).indices[:, :k].int()
    ranks = torch.arange(k, device=logits.device)[None, :]
    idx = torch.where(ranks < visible[:, None], idx, torch.full_like(idx, -1))
    return idx.sort(1).values


def run(logits, visible):
    out = torch.empty(logits.shape[0], K, dtype=torch.int32, device="cuda")
    masked_topk(logits, visible, out)
    return out.sort(1).values


@pytest.mark.parametrize("cols", (64, 384, 16512, 45056))
def test_canonical_at_a_tie_and_masks_by_bound(cols):
    rows = 32
    logits = tied_logits(rows, cols)
    visible = torch.full((rows,), cols, dtype=torch.int32, device="cuda")
    visible[0] = 0
    visible[1] = min(cols, K)
    visible[2] = max(1, min(cols, K) - 7)
    visible[3] = max(1, cols // 2)
    expect = stable_topk(logits, visible, K)
    # scratch past the bound is NaN: masking by value instead of by the
    # bound, or reading past it at all, poisons the result
    columns = torch.arange(cols, device="cuda")[None, :]
    poisoned = logits.masked_fill(columns >= visible[:, None], float("nan"))
    for _ in range(8):  # determinism: the same set on every call
        assert torch.equal(run(poisoned, visible), expect)


def test_all_zero_rows_select_lowest_indices():
    """A fully tied row (all scores equal) must select columns 0..k-1."""
    rows, cols = 8, 4096
    logits = torch.zeros(rows, cols, device="cuda")
    visible = torch.full((rows,), cols, dtype=torch.int32, device="cuda")
    expect = torch.arange(K, dtype=torch.int32, device="cuda").expand(rows, K)
    assert torch.equal(run(logits, visible), expect)


def test_graph_capturable():
    """Production replays this under CUDA graphs: grid from shapes only,
    no host read of tensor values, no allocation surprises on replay."""
    logits = tied_logits(16, 6400)
    visible = torch.full((16,), 6400, dtype=torch.int32, device="cuda")
    visible[1] = 3000
    out = torch.empty(16, K, dtype=torch.int32, device="cuda")
    masked_topk(logits, visible, out)  # warmup/compile
    expect = stable_topk(logits, visible, K)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        masked_topk(logits, visible, out)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        masked_topk(logits, visible, out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.sort(1).values, expect)
