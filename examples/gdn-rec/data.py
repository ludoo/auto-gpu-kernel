"""Seeded inputs at the served shape, and the served model's captured
kernel inputs. Shared by the tests and the bench; the reference and the
candidate see identical tensors.

`make_inputs(seed, T, n)` draws one call's inputs with the served
distributions (measured on the captured vectors: q ~ 0.003, k ~ 0.05,
v ~ 0.05 with a tail to 5, g in [-258, 0] mostly small, beta in
(0, 1], state ~ 0.004 with a tail to 4). `real_vectors()` loads the two
captured calls from sparkle's `vectors/gdn-recurrence-v1` (64 rows from
a zero state; 16 rows from that state, with the state after rows 2 and
4) when the files are present.
"""

from __future__ import annotations

import os

import torch

from shapes import H, HEAD_DIM, HV

VECTORS = os.environ.get(
    "GDN_REC_VECTORS",
    "/home/ludo/dev/kb-vault/code/sparkle/vectors/gdn-recurrence-v1")


def make_inputs(seed: int, T: int, n: int = 1):
  g = torch.Generator(device="cuda").manual_seed(seed + 1_000_003 * T + 7 * n)
  rows = n * T
  K = HEAD_DIM
  q = torch.randn(rows, H, K, device="cuda", generator=g) * 0.004
  k = torch.randn(rows, H, K, device="cuda", generator=g) * 0.06
  v = torch.randn(rows, HV, K, device="cuda", generator=g) * 0.06
  # a heavy tail on v, as the captured rows have
  v = v * torch.where(torch.rand(rows, HV, 1, device="cuda", generator=g) < 0.02,
                      torch.tensor(40.0, device="cuda"),
                      torch.tensor(1.0, device="cuda"))
  v = v.bfloat16()
  gg = -torch.rand(rows, HV, device="cuda", generator=g) ** 3 * 20.0
  beta = torch.rand(rows, HV, device="cuda", generator=g).clamp_min(0.02)
  h0 = torch.randn(n, HV, K, K, device="cuda", generator=g) * 0.004
  h0 = h0 * torch.where(torch.rand(n, HV, K, 1, device="cuda", generator=g) < 0.01,
                        torch.tensor(200.0, device="cuda"),
                        torch.tensor(1.0, device="cuda"))
  return q.contiguous(), k.contiguous(), v.contiguous(), gg.contiguous(), \
      beta.contiguous(), h0.contiguous()


def real_vectors():
  """The captured calls as a list of dicts, or an empty list when the
  vector files are not on this box."""
  if not os.path.isdir(VECTORS):
    return []
  from safetensors.torch import load_file  # noqa: PLC0415
  out = []
  for name in ("call1", "call2"):
    p = os.path.join(VECTORS, f"{name}.safetensors")
    if os.path.exists(p):
      d = {k: t.cuda().contiguous() for k, t in load_file(p).items()}
      d["name"] = name
      out.append(d)
  return out
