"""The contract: `gdn_recurrence` in gdn_rec.py is BITWISE equal to the
reference kernel on every output — `o`, `ht`, `states`, `pin` — on
every input. There is no tolerance and none may be introduced; a single
differing element fails.

The reference's construction makes T rows in one launch equal to T
single-row launches threading the state; the candidate inherits that
contract through equality with the reference at every T tested, not by
its own construction. Coverage is deliberately wide: the served four
shapes, three seeds at the chunk, a small-T volume over T=1..8 and
n=1..4 with fresh seeds (a row-count rule in the candidate sends the
served rows down code the chunk never runs), the served model's
captured inputs, pins at positions other than the midpoint, determinism,
and CUDA-graph capture and replay (decode and verify replay under
graphs: no host sync, no data-dependent host work).
"""

from __future__ import annotations

import pytest
import torch

from data import make_inputs, real_vectors
from gdn_rec import gdn_recurrence
from reference import gdn_recurrence as reference_recurrence
from shapes import HEAD_DIM, HV, SHAPES


def _run(fn, inp, states=False, pin_at=None):
  q, k, v, g, b, h0 = inp
  out = fn(q, k, v, g, b, h0, states=states, pin_at=pin_at)
  torch.cuda.synchronize()
  return out


def _assert_bitwise(a, b, label):
  if a is None and b is None:
    return
  assert (a is None) == (b is None), f"{label}: one side is None"
  if torch.equal(a, b):
    return
  diff = a != b
  raise AssertionError(
      f"{label}: {int(diff.sum())} of {a.numel()} elements differ "
      f"(max |delta| {(a.float() - b.float()).abs().max().item():.3e})")


def _assert_out(out, ref, label):
  _assert_bitwise(out.o, ref.o, f"{label} o")
  _assert_bitwise(out.ht, ref.ht, f"{label} ht")
  _assert_bitwise(out.states, ref.states, f"{label} states")
  _assert_bitwise(out.pin, ref.pin, f"{label} pin")


@pytest.mark.parametrize("name,shape", SHAPES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bitwise_at_served_shapes(name, shape, seed):
  T, n, states, pin_at = shape
  if T < 8192 and seed:
    pytest.skip("small shapes: one seed here; the volume test covers them")
  inp = make_inputs(seed, T, n)
  _assert_out(_run(gdn_recurrence, inp, states, pin_at),
              _run(reference_recurrence, inp, states, pin_at),
              f"{name} seed {seed}")


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_small_t_volume(T, n):
  """Fresh seeds at every small (T, n), with per-row states: the rows
  the served step runs, at the batch sizes the engine can run them."""
  inp = make_inputs(100 + 17 * T + n, T, n)
  _assert_out(_run(gdn_recurrence, inp, True),
              _run(reference_recurrence, inp, True), f"T={T} n={n} states")
  _assert_out(_run(gdn_recurrence, inp, False),
              _run(reference_recurrence, inp, False), f"T={T} n={n}")


@pytest.mark.parametrize("T,pin_at", [(64, 1), (64, 63), (4096, 1000),
                                      (4096, 4095), (8192, 64)])
def test_pin_positions(T, pin_at):
  inp = make_inputs(200 + T + pin_at, T, 1)
  _assert_out(_run(gdn_recurrence, inp, False, pin_at),
              _run(reference_recurrence, inp, False, pin_at),
              f"T={T} pin_at={pin_at}")


@pytest.mark.parametrize("T", [2, 5])
def test_states_and_pin_together(T):
  inp = make_inputs(300 + T, T, 1)
  _assert_out(_run(gdn_recurrence, inp, True, 1),
              _run(reference_recurrence, inp, True, 1),
              f"T={T} states+pin")


def test_real_vectors():
  """The served model's captured kernel inputs: equal to the stored
  outputs, and the state after rows 2 and 4 of call2 equal to the
  stored ones through `states` and `pin_at` (row indices: `states[2]`
  is the state after row 2, the pin_at=3 target)."""
  vecs = real_vectors()
  if not vecs:
    pytest.skip("captured vectors not on this box")
  for d in vecs:
    inp = (d["q"], d["k"], d["v"], d["g"], d["beta"], d["h0"][None])
    out = _run(gdn_recurrence, inp, states=True)
    _assert_bitwise(out.o, d["o"], f"{d['name']} o")
    _assert_bitwise(out.ht[0], d["ht"], f"{d['name']} ht")
    if "h_after_row2" in d:
      _assert_bitwise(out.states[2], d["h_after_row2"], f"{d['name']} row2")
      _assert_bitwise(out.states[4], d["h_after_row4"], f"{d['name']} row4")
      pinned = _run(gdn_recurrence, inp, pin_at=3)
      _assert_bitwise(pinned.pin, d["h_after_row2"], f"{d['name']} pin@3")


def test_stepped_equals_one_launch():
  """T rows in one launch against T single-row launches threading the
  state, on the candidate alone: the phase pin's own property."""
  T = 16
  q, k, v, g, b, h0 = make_inputs(400, T, 1)
  whole = _run(gdn_recurrence, (q, k, v, g, b, h0), states=True)
  h = h0
  for t in range(T):
    step = _run(gdn_recurrence, (q[t:t + 1], k[t:t + 1], v[t:t + 1],
                                 g[t:t + 1], b[t:t + 1], h))
    _assert_bitwise(step.o, whole.o[t:t + 1], f"row {t} o")
    _assert_bitwise(step.ht[0], whole.states[t], f"row {t} state")
    h = step.ht
  _assert_bitwise(h, whole.ht, "final state")


def test_deterministic_over_calls():
  inp = make_inputs(500, 8192, 1)
  first = _run(gdn_recurrence, inp, False, 4096)
  for i in range(7):
    _assert_out(_run(gdn_recurrence, inp, False, 4096), first,
                f"call {i + 2} vs call 1")


@pytest.mark.parametrize("T,states", [(1, False), (5, True)])
def test_cuda_graph_capture_and_replay(T, states):
  """Decode and verify run under CUDA graphs: the call must capture and
  replay bitwise, with no host work that depends on data. The outputs
  are read back from the captured call's return, so the candidate's
  allocations happen inside the capture (torch's graph pool) exactly
  as the engine does it."""
  q, k, v, g, b, h0 = make_inputs(600 + T, T, 1)
  ref = _run(reference_recurrence, (q, k, v, g, b, h0), states)
  stream = torch.cuda.Stream()
  with torch.cuda.stream(stream):
    for _ in range(3):
      gdn_recurrence(q, k, v, g, b, h0, states=states)
  torch.cuda.current_stream().wait_stream(stream)
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    out = gdn_recurrence(q, k, v, g, b, h0, states=states)
  out.o.zero_()
  out.ht.zero_()
  graph.replay()
  torch.cuda.synchronize()
  _assert_out(out, ref, "graph replay")
  # New inputs through the same graph: read at replay, not captured.
  q2, k2, v2, g2, b2, h02 = make_inputs(700 + T, T, 1)
  for dst, src in ((q, q2), (k, k2), (v, v2), (g, g2), (b, b2), (h0, h02)):
    dst.copy_(src)
  ref2 = _run(reference_recurrence, (q, k, v, g, b, h0), states)
  graph.replay()
  torch.cuda.synchronize()
  _assert_out(out, ref2, "graph replay, new inputs")


def test_no_torch_compute_on_the_path(monkeypatch):
  """The kernel is the candidate's own: torch matmul, einsum, compile
  and the like must not be on the main path."""

  def boom(*a, **k):
    raise AssertionError("torch compute op on the main path")

  for name in ("matmul", "bmm", "einsum", "compile", "baddbmm", "mm"):
    monkeypatch.setattr(torch, name, boom)
  inp = make_inputs(800, 5, 1)
  _run(gdn_recurrence, inp, True)


def test_output_shapes_and_dtypes():
  q, k, v, g, b, h0 = make_inputs(900, 5, 2)
  out = _run(gdn_recurrence, (q, k, v, g, b, h0), states=True)
  assert out.o.shape == (10, HV, HEAD_DIM) and out.o.dtype == torch.bfloat16
  assert out.ht.shape == (2, HV, HEAD_DIM, HEAD_DIM) and out.ht.dtype == torch.float32
  assert out.states.shape == (10, HV, HEAD_DIM, HEAD_DIM)
  assert out.pin is None
