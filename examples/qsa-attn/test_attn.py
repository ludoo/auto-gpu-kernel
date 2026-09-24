"""The contract: `qsa_attention` is BITWISE equal to the reference kernel
in its served configuration, on every input. There is no tolerance and
none may be introduced; a single differing element fails.

Coverage is deliberately large. A kernel whose fp32 arithmetic differs
from the reference's shows in the bf16 output only where an element
lands on a rounding boundary: measured on this box, the reference at
4 warps instead of 1 differs in ~1 element per 250,000 (202 of 50M at
8192 rows) and is bitwise equal at M=1 and M=5. So the chunk shape runs
three seeds (150M elements) and every shape runs holes and multiple
requests. Passing M=1 and M=5 alone means nothing.
"""

from __future__ import annotations

import pytest
import torch

from data import make_cache, make_rows
from qsa_attn import qsa_attention
from reference import reference_attention
from shapes import HEAD_DIM, HEADS, SHAPES


def _run(fn, cache, rows_in):
  k, v, table = cache
  q, packed, t2r = rows_in
  out = torch.zeros(q.shape[0], HEADS, HEAD_DIM, dtype=torch.bfloat16,
                    device="cuda")
  fn(q, k, v, packed, table, t2r, out)
  torch.cuda.synchronize()
  return out


def _assert_bitwise(out, ref, label):
  if torch.equal(out, ref):
    return
  diff = out != ref
  rows = int(diff.any(dim=(1, 2)).sum())
  raise AssertionError(
      f"{label}: {int(diff.sum())} elements differ in {rows} rows "
      f"(max |delta| {(out.float() - ref.float()).abs().max().item():.3e})")


@pytest.fixture(scope="module")
def cache():
  return make_cache(0)


@pytest.fixture(scope="module")
def cache_4req():
  return make_cache(7, requests=4)


@pytest.mark.parametrize("name,rows", SHAPES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bitwise_at_served_rows(cache, name, rows, seed):
  if rows < 8192 and seed:
    pytest.skip("small shapes: one seed; the chunk carries the coverage")
  rows_in = make_rows(seed, rows)
  _assert_bitwise(_run(qsa_attention, cache, rows_in),
                  _run(reference_attention, cache, rows_in),
                  f"{name} seed {seed}")


@pytest.mark.parametrize("name,rows", SHAPES)
def test_bitwise_with_holes(cache, name, rows):
  """Short counts, -1 past them, a one-tile row, a zero-count row."""
  rows_in = make_rows(11, rows, holes=True)
  _assert_bitwise(_run(qsa_attention, cache, rows_in),
                  _run(reference_attention, cache, rows_in),
                  f"{name} holes")


@pytest.mark.parametrize("rows", [5, 8192])
def test_bitwise_multi_request(cache_4req, rows):
  rows_in = make_rows(5, rows, requests=4, holes=True)
  _assert_bitwise(_run(qsa_attention, cache_4req, rows_in),
                  _run(reference_attention, cache_4req, rows_in),
                  f"rows {rows} 4 requests")


def test_zero_count_row_writes_zeros(cache):
  q, packed, t2r = make_rows(3, 16, holes=True)
  packed[:, -1] = 0
  packed[:, :-1] = -1
  out = _run(qsa_attention, cache, (q, packed, t2r))
  assert torch.equal(out, torch.zeros_like(out))


def test_deterministic_over_calls(cache):
  rows_in = make_rows(21, 8192)
  first = _run(qsa_attention, cache, rows_in)
  for i in range(7):
    _assert_bitwise(_run(qsa_attention, cache, rows_in), first,
                    f"call {i + 2} vs call 1")


@pytest.mark.parametrize("rows", [1, 5])
def test_cuda_graph_capture_and_replay(cache, rows):
  """Decode and verify run under CUDA graphs: the call must capture and
  replay bitwise, with no host work that depends on data."""
  k, v, table = cache
  q, packed, t2r = make_rows(31, rows)
  out = torch.zeros(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
  ref = _run(reference_attention, cache, (q, packed, t2r))
  stream = torch.cuda.Stream()
  with torch.cuda.stream(stream):
    for _ in range(3):
      qsa_attention(q, k, v, packed, table, t2r, out)
  torch.cuda.current_stream().wait_stream(stream)
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    qsa_attention(q, k, v, packed, table, t2r, out)
  out.zero_()
  graph.replay()
  torch.cuda.synchronize()
  _assert_bitwise(out, ref, "graph replay")
  # A second selection through the same graph: the inputs are read at
  # replay, not captured.
  q2, packed2, _ = make_rows(32, rows)
  q.copy_(q2)
  packed.copy_(packed2)
  ref2 = _run(reference_attention, cache, (q, packed, t2r))
  graph.replay()
  torch.cuda.synchronize()
  _assert_bitwise(out, ref2, "graph replay, new inputs")


def test_no_torch_matmul_or_compile_on_the_path(monkeypatch, cache):
  """The kernel is the candidate's own: torch's attention, matmul and
  compile must not be on the main path."""
  import torch.nn.functional as F  # noqa: PLC0415

  def boom(*a, **k):
    raise AssertionError("torch op on the main path")

  for mod, name in ((torch, "matmul"), (torch, "bmm"), (torch, "einsum"),
                    (F, "scaled_dot_product_attention"), (torch, "compile"),
                    (torch, "softmax")):
    monkeypatch.setattr(mod, name, boom)
  rows_in = make_rows(41, 5)
  _run(qsa_attention, cache, rows_in)
