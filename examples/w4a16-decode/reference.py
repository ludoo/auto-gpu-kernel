"""The fp32 reference and the wheel.

`reference(x, packed, scale)`: `x.float() @ dequant(W).float().T`,
rounded once to bf16. The correctness side of the contract: a
candidate is held to within bf16 rounding of this, measured as the
wheel's own distance (test_w4a16.py).

`wheel(x, packed, scale)`: sparkle's `shim.w4a16_linear` — the wheel's
Marlin `marlin_gemm` on the engine's exact call (uint4b8, fp32 reduce,
no zero points), on a record `shim.prep_w4a16_linear` builds (the Marlin
repack and scale permutation, cached per weight here). The bench's
baseline and the correctness comparator; not the contract — its bytes
move with the call's M (one class at M in [17, 64] ∪ [81, 127], others
elsewhere) and are the thing being left.
"""

from __future__ import annotations

import torch

from data import dequant
from shapes import GROUP

_PREPPED = {}


def reference(x, packed, scale):
  return (x.float() @ dequant(packed, scale).T).to(torch.bfloat16)


def _rec(packed, scale):
  from sparkle import shim  # noqa: PLC0415
  key = (packed.data_ptr(), scale.data_ptr())
  rec = _PREPPED.get(key)
  if rec is None:
    rec = shim.prep_w4a16_linear(packed, scale, group_size=GROUP)
    _PREPPED[key] = rec
  return rec


def wheel(x, packed, scale):
  from sparkle import shim  # noqa: PLC0415
  return shim.w4a16_linear(x, _rec(packed, scale))
