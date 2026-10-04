"""Seeded weights and activations, and the fp32 dequant."""

from __future__ import annotations

import torch

from shapes import GROUP


def make_weight(n, k, seed):
  g = torch.Generator(device='cuda').manual_seed(seed)
  packed = torch.randint(0, 1 << 31, (n, k // 8), device='cuda',
                         dtype=torch.int32, generator=g)
  scale = (torch.rand(n, k // GROUP, device='cuda', generator=g) + 0.5) * 0.01
  return packed, scale.to(torch.bfloat16)


def make_x(m, k, seed):
  g = torch.Generator(device='cuda').manual_seed(seed)
  return torch.randn(m, k, device='cuda', generator=g).to(torch.bfloat16)


def dequant(packed, scale):
  """fp32 `[N, K]`: (nibble - 8) * scale, nibbles LSB-first along K."""
  N, Kw = packed.shape
  nib = torch.stack([(packed >> (4 * i)) & 0xF for i in range(8)],
                    dim=-1).reshape(N, Kw * 8)
  return (nib.float() - 8.0) * scale.float().repeat_interleave(GROUP, dim=1)
