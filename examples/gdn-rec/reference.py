"""The GDN recurrence in every phase: one CUDA kernel for prefill,
verify and decode (DOC#DIRECTION-DESIGN, § The GDN recurrence kernel).

Why this kernel exists (DOC#DIRECTION, layer 2): the wheel runs the
FLA chunk kernel at prefill and its recurrent kernel at decode and
verify, and a row's bytes differ between the two — the chunk kernel's
numerics for a row depend on where the row sits in its 64-token grid.
This kernel gives every row the same bytes whether it arrives alone, in
a verify pass of five or in a chunk of 8192: the loop body is one
sequential recurrence per (value head, row slice), the reduction order
is fixed, T is a runtime int, and nothing dispatches on the row count.
At the served shape it runs the 8192-row chunk in 0.70× the chunk
kernel's time (service down; DOC#task-next.74).

Structure: one warp per (V head, 16-row slice of V), grid
(8, HV, entries). Each lane owns two V rows × a quarter of K, the fp32
state in registers; a row's two dot products are in-thread FMA chains
with four accumulators finished by two `shfl_xor` levels. The head's k
and q rows, the warp's v rows, g and beta are staged per step through
a warp-private three-deep shared-memory ring by `cp.async`, with no
block barrier. Per row per step, in this order: `h *= exp(g)`;
`u = (v − h·k) · beta`; `h += u ⊗ k`; `o = h·q`. The template
constants (KSPLIT 4, ring depth 3, two rows per lane) are the measured
choice, not autotuned (DOC#KERNELS, authored-kernel bar).

Numerics contract: byte-equal to the next.74 reference kernel, pinned
by `tests/test_gdn_recurrence_gpu.py` on the served model's captured
kernel inputs; a change to the arithmetic order is a design change.
Inputs: q and k fp32 `[rows, H, K]` with the attention scale already
folded into q; v bf16 `[rows, HV, V]`; g and beta fp32 `[rows, HV]`;
h0 fp32 `[n, HV, V, K]`. `rows = n * T`, entry-major, T rows per
entry. K = V = 128 is the compiled shape; H and HV come from the
tensors, so any GQA ratio inherits the kernel.

State writes beyond the final state, both optional: the state after
every row (`states`, verify's retention — cheap at verify's T, never
asked for at prefill's), and the state after row `pin_at − 1` (`pin`,
the mid-chunk pin; DOC#CONTRACTS). Either one selects the kernel's
second instantiation, whose loop carries the writes and whose state
reads and writes go through a warp-private staging tile so the
global side is coalesced rows: a lane's register slice is a quarter
row, and written directly it wastes half of every sector — nothing
at T=8192, where the state is written once, and the whole cost at
verify's five writes per layer per step. The plain instantiation is
the reference's loop, register for register. Neither touches the
arithmetic: one body, one order, both pinned to the same vectors.

Importing this module stays CPU-safe: the CUDA source is compiled by
`torch.utils.cpp_extension.load_inline` on first use, into the torch
extensions cache (`TORCH_EXTENSIONS_DIR`).
"""

import dataclasses
from typing import Any

_NAME = 'gdn_rec_reference'
HEAD_DIM = 128  # K = V, the compiled shape
_KSPLIT, _RING, _RPL = 4, 3, 2
_ROWS_PER_WARP = (32 // _KSPLIT) * _RPL  # 16
_WARPS_PER_HEAD = HEAD_DIM // _ROWS_PER_WARP  # 8

_SRC = r'''
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
  unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// One warp per (V head, WROWS-row slice), one entry per blockIdx.z.
// KSPLIT lanes share a row's K range; each lane owns RPL rows. WRITES
// selects the loop with the per-row and pin state writes (verify, the
// mid-chunk pin); without it the loop is the reference's shape and
// register count. The arithmetic is one body either way.
template <int KSPLIT, int D, int RPL, bool WRITES>
__global__ void __launch_bounds__(32)
gdn_recurrence_kernel(const float* __restrict__ q, const float* __restrict__ k,
                      const __nv_bfloat16* __restrict__ v,
                      const float* __restrict__ g, const float* __restrict__ beta,
                      __nv_bfloat16* __restrict__ o, const float* __restrict__ h0,
                      float* __restrict__ ht, float* __restrict__ states,
                      float* __restrict__ pin, int pin_row, int T, int H, int HV) {
  constexpr int K = 128, V = 128;
  constexpr int KL = K / KSPLIT;
  constexpr int ROWS = 32 / KSPLIT;
  constexpr int NF = KL / 4;
  constexpr int WROWS = ROWS * RPL;
  constexpr int PART = KL * 4 + 16;
  constexpr int ROWB = KSPLIT * PART;
  constexpr int OFF_Q = ROWB, OFF_V = 2 * ROWB, OFF_G = OFF_V + ((WROWS * 2 + 15) / 16) * 16;
  constexpr int SLOT = OFF_G + 16;
  constexpr int SROW = K + 4;  // the staging row stride: K padded to spread banks
  __shared__ __align__(16) char ring[D * SLOT];
  // The warp's 16 state rows are one contiguous 8 KB block in global
  // memory, but a lane's register slice is a quarter row. In the WRITES
  // instantiation every state read and write goes through this staging
  // tile so the global side is 512-byte coalesced rows rather than
  // 16-byte pieces at 128-byte stride (the per-row writes are that
  // instantiation's whole cost). Without WRITES the state is read once
  // and written once and the tile's presence alone costs 45 registers
  // in the hot loop, so that instantiation keeps the direct layout.
  __shared__ __align__(16) float stage[WRITES ? WROWS * SROW : 4];

  const int lane = threadIdx.x;
  const int kpart = lane / ROWS, row = lane % ROWS;
  const int i_hv = blockIdx.y, warp = blockIdx.x, entry = blockIdx.z;
  const int i_h = i_hv / (HV / H);
  const int r0 = warp * WROWS + row;
  const size_t row0 = (size_t)entry * T;  // this entry's first row
  const size_t hstride = (size_t)HV * V * K;  // one state

  const char* kg = reinterpret_cast<const char*>(k + (row0 * H + i_h) * K);
  const char* qg = reinterpret_cast<const char*>(q + (row0 * H + i_h) * K);
  const char* vg = reinterpret_cast<const char*>(v + (row0 * HV + i_hv) * V + warp * WROWS);
  const char* gg = reinterpret_cast<const char*>(g + row0 * HV + i_hv);
  const char* bg = reinterpret_cast<const char*>(beta + row0 * HV + i_hv);
  const size_t sqb = (size_t)H * K * 4, svb = (size_t)HV * V * 2, sgb = (size_t)HV * 4;

  auto issue = [&](int t) {
    char* slot = ring + (t % D) * SLOT;
    const int part = lane / (KL / 4), c = lane % (KL / 4);
    cp_async16(slot + part * PART + c * 16, kg + (size_t)t * sqb + lane * 16);
    cp_async16(slot + OFF_Q + part * PART + c * 16, qg + (size_t)t * sqb + lane * 16);
    if (lane < WROWS * 2 / 16)
      cp_async16(slot + OFF_V + lane * 16, vg + (size_t)t * svb + lane * 16);
    else if (lane == 30) cp_async4(slot + OFF_G, gg + (size_t)t * sgb);
    else if (lane == 31) cp_async4(slot + OFF_G + 4, bg + (size_t)t * sgb);
  };
  static_assert(WROWS * 2 / 16 <= 30 && WROWS * 2 % 16 == 0, "v row staging");
  static_assert(K / 4 == 32, "one state row is one float4 per lane");

  // The warp's rows in a state tensor: rows warp * WROWS .. + WROWS - 1
  // of head i_hv, contiguous, K floats each.
  const size_t wrows_off = ((size_t)i_hv * V + warp * WROWS) * K;
  float h[RPL][KL];
  {
    const float* hbase = h0 + entry * hstride;
    if constexpr (WRITES) {
      const float4* src = reinterpret_cast<const float4*>(hbase + wrows_off);
#pragma unroll 1
      for (int r = 0; r < WROWS; ++r)
        *reinterpret_cast<float4*>(stage + r * SROW + lane * 4) = src[r * (K / 4) + lane];
      __syncwarp();
    }
#pragma unroll
    for (int i = 0; i < RPL; ++i) {
      const float* hp = WRITES
          ? stage + (row + i * ROWS) * SROW + kpart * KL
          : hbase + ((size_t)i_hv * V + r0 + i * ROWS) * K + kpart * KL;
#pragma unroll
      for (int j = 0; j < KL; j += 4) {
        float4 t4 = *reinterpret_cast<const float4*>(hp + j);
        h[i][j] = t4.x; h[i][j + 1] = t4.y; h[i][j + 2] = t4.z; h[i][j + 3] = t4.w;
      }
    }
  }
  // A state write: this lane's slice of its RPL rows — through the
  // staging tile and out as coalesced rows under WRITES, direct otherwise.
  auto store_state = [&](float* base) {
    if constexpr (WRITES) __syncwarp();
#pragma unroll
    for (int i = 0; i < RPL; ++i) {
      float* hq = WRITES
          ? stage + (row + i * ROWS) * SROW + kpart * KL
          : base + ((size_t)i_hv * V + r0 + i * ROWS) * K + kpart * KL;
#pragma unroll
      for (int j = 0; j < KL; j += 4)
        *reinterpret_cast<float4*>(hq + j) = make_float4(h[i][j], h[i][j + 1], h[i][j + 2], h[i][j + 3]);
    }
    if constexpr (WRITES) {
      __syncwarp();
      float4* dst = reinterpret_cast<float4*>(base + wrows_off);
#pragma unroll 1
      for (int r = 0; r < WROWS; ++r)
        dst[r * (K / 4) + lane] = *reinterpret_cast<const float4*>(stage + r * SROW + lane * 4);
    }
  };
  __nv_bfloat16* op = o + (row0 * HV + i_hv) * V + r0;
  const size_t sv = (size_t)HV * V;

#pragma unroll
  for (int s = 0; s < D - 1; ++s) {
    if (s < T) issue(s);
    cp_async_commit();
  }

  for (int t = 0; t < T; ++t) {
    cp_async_wait<D - 2>();
    __syncwarp();
    if (t + D - 1 < T) issue(t + D - 1);
    cp_async_commit();

    const char* slot = ring + (t % D) * SLOT;
    const float4* ks = reinterpret_cast<const float4*>(slot + kpart * PART);
    const float4* qs = reinterpret_cast<const float4*>(slot + OFF_Q + kpart * PART);
    const __nv_bfloat16* vs = reinterpret_cast<const __nv_bfloat16*>(slot + OFF_V);
    const float gv = *reinterpret_cast<const float*>(slot + OFF_G);
    const float bb = *reinterpret_cast<const float*>(slot + OFF_G + 4);
    const float e = expf(gv);

    float a[RPL][4];
#pragma unroll
    for (int i = 0; i < RPL; ++i) { a[i][0] = 0.f; a[i][1] = 0.f; a[i][2] = 0.f; a[i][3] = 0.f; }
#pragma unroll
    for (int j = 0; j < NF; ++j) {
      const float4 kk = ks[j];
#pragma unroll
      for (int i = 0; i < RPL; ++i) {
        h[i][4 * j] *= e; h[i][4 * j + 1] *= e; h[i][4 * j + 2] *= e; h[i][4 * j + 3] *= e;
        a[i][0] = fmaf(h[i][4 * j], kk.x, a[i][0]);
        a[i][1] = fmaf(h[i][4 * j + 1], kk.y, a[i][1]);
        a[i][2] = fmaf(h[i][4 * j + 2], kk.z, a[i][2]);
        a[i][3] = fmaf(h[i][4 * j + 3], kk.w, a[i][3]);
      }
    }
    float u[RPL];
#pragma unroll
    for (int i = 0; i < RPL; ++i) {
      float dot = (a[i][0] + a[i][1]) + (a[i][2] + a[i][3]);
#pragma unroll
      for (int off = ROWS; off < 32; off <<= 1)
        dot += __shfl_xor_sync(0xffffffffu, dot, off);
      const float vv = __bfloat162float(vs[row + i * ROWS]);
      u[i] = (vv - dot) * bb;
    }
    float ob[RPL][4];
#pragma unroll
    for (int i = 0; i < RPL; ++i) { ob[i][0] = 0.f; ob[i][1] = 0.f; ob[i][2] = 0.f; ob[i][3] = 0.f; }
#pragma unroll
    for (int j = 0; j < NF; ++j) {
      const float4 kk = ks[j];
      const float4 qq = qs[j];
#pragma unroll
      for (int i = 0; i < RPL; ++i) {
        h[i][4 * j] = fmaf(u[i], kk.x, h[i][4 * j]);
        h[i][4 * j + 1] = fmaf(u[i], kk.y, h[i][4 * j + 1]);
        h[i][4 * j + 2] = fmaf(u[i], kk.z, h[i][4 * j + 2]);
        h[i][4 * j + 3] = fmaf(u[i], kk.w, h[i][4 * j + 3]);
        ob[i][0] = fmaf(h[i][4 * j], qq.x, ob[i][0]);
        ob[i][1] = fmaf(h[i][4 * j + 1], qq.y, ob[i][1]);
        ob[i][2] = fmaf(h[i][4 * j + 2], qq.z, ob[i][2]);
        ob[i][3] = fmaf(h[i][4 * j + 3], qq.w, ob[i][3]);
      }
    }
#pragma unroll
    for (int i = 0; i < RPL; ++i) {
      float out = (ob[i][0] + ob[i][1]) + (ob[i][2] + ob[i][3]);
#pragma unroll
      for (int off = ROWS; off < 32; off <<= 1)
        out += __shfl_xor_sync(0xffffffffu, out, off);
      if (kpart == 0) op[i * ROWS] = __float2bfloat16(out);
    }
    op += sv;
    if (WRITES) {
      if (states != nullptr) store_state(states + (row0 + t) * hstride);
      if (t == pin_row) store_state(pin);
    }
  }
  cp_async_wait<0>();
  store_state(ht + entry * hstride);
}

void gdn_recurrence(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                    torch::Tensor g, torch::Tensor beta, torch::Tensor o,
                    torch::Tensor h0, torch::Tensor ht, torch::Tensor states,
                    torch::Tensor pin, int64_t pin_row, int64_t T) {
  const int H = q.size(1), HV = v.size(1), n = h0.size(0);
  constexpr int KSPLIT = 4, D = 3, RPL = 2;
  constexpr int NW = 128 / ((32 / KSPLIT) * RPL);
  dim3 grid(NW, HV, n);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool writes = states.numel() > 0 || pin_row >= 0;
  auto launch = [&](auto kern) {
    kern<<<grid, 32, 0, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        g.data_ptr<float>(), beta.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr<at::BFloat16>()),
        h0.data_ptr<float>(), ht.data_ptr<float>(),
        states.numel() ? states.data_ptr<float>() : nullptr,
        pin.numel() ? pin.data_ptr<float>() : nullptr,
        static_cast<int>(pin_row), static_cast<int>(T), H, HV);
  };
  if (writes) launch(gdn_recurrence_kernel<KSPLIT, D, RPL, true>);
  else launch(gdn_recurrence_kernel<KSPLIT, D, RPL, false>);
}
'''

_mod: Any = None


def _module():
  """The compiled extension, built on first use."""
  global _mod
  if _mod is None:
    from torch.utils.cpp_extension import load_inline  # noqa: PLC0415
    _mod = load_inline(
        name=_NAME,
        cpp_sources='void gdn_recurrence(torch::Tensor, torch::Tensor, '
        'torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, '
        'torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, '
        'int64_t, int64_t);',
        cuda_sources=_SRC,
        functions=['gdn_recurrence'],
        extra_cuda_cflags=['-O3'],
        verbose=False,
    )
  return _mod


@dataclasses.dataclass(frozen=True)
class RecurrenceOut:
  """One launch's outputs. `o` bf16 `[rows, HV, V]`; `ht` fp32
  `[n, HV, V, K]`, each entry's state after its last row; `states`
  fp32 `[rows, HV, V, K]`, the state after every row, or None;
  `pin` fp32 `[HV, V, K]`, the state after row `pin_at - 1`, or None."""
  o: Any
  ht: Any
  states: Any = None
  pin: Any = None


def gdn_recurrence(q: Any, k: Any, v: Any, g: Any, beta: Any, h0: Any, *,
                   states: bool = False, pin_at: int | None = None,
                   out: Any = None) -> RecurrenceOut:
  """Run the recurrence over `rows = n * T` entry-major rows from the
  n initial states in `h0`. q/k fp32 `[rows, H, K]` (q pre-scaled),
  v bf16 `[rows, HV, V]`, g/beta fp32 `[rows, HV]`, h0 fp32
  `[n, HV, V, K]`, all contiguous on one CUDA device; K = V =
  `HEAD_DIM`. `states=True` also returns the state after every row;
  `pin_at` (n = 1, strictly inside the rows) also returns the state
  after row `pin_at - 1`. Geometry is read from the tensors; a shape
  the kernel is not compiled for is a ValueError."""
  import torch  # noqa: PLC0415

  rows, H, K = q.shape
  n, HV, V, K2 = h0.shape
  if K != HEAD_DIM or V != HEAD_DIM or K2 != HEAD_DIM:
    raise ValueError(f'gdn_recurrence is compiled for K = V = {HEAD_DIM}; '
                     f'got K={K}, V={V}, state K={K2}')
  if k.shape != (rows, H, K) or v.shape != (rows, HV, V) or \
      g.shape != (rows, HV) or beta.shape != (rows, HV):
    raise ValueError(f'gdn_recurrence: inconsistent shapes q{tuple(q.shape)} '
                     f'k{tuple(k.shape)} v{tuple(v.shape)} g{tuple(g.shape)} '
                     f'beta{tuple(beta.shape)} h0{tuple(h0.shape)}')
  if HV % H:
    raise ValueError(f'gdn_recurrence: HV={HV} is not a multiple of H={H}')
  if n == 0 or rows % n:
    raise ValueError(f'gdn_recurrence: {rows} rows do not split into {n} '
                     'equal entries')
  T = rows // n
  if q.dtype != torch.float32 or k.dtype != torch.float32 or \
      v.dtype != torch.bfloat16 or g.dtype != torch.float32 or \
      beta.dtype != torch.float32 or h0.dtype != torch.float32:
    raise ValueError('gdn_recurrence: q/k/g/beta/h0 fp32, v bf16')
  for name, t in (('q', q), ('k', k), ('v', v), ('g', g), ('beta', beta), ('h0',
                                                                           h0)):
    if not t.is_contiguous() or not t.is_cuda:
      raise ValueError(f'gdn_recurrence: {name} must be contiguous on CUDA')
  if pin_at is not None:
    if n != 1:
      raise ValueError('gdn_recurrence: pin_at needs a single entry')
    if not 0 < pin_at < T:
      raise ValueError(f'gdn_recurrence: pin_at {pin_at} is not strictly '
                       f'inside {T} rows')
  o = v.new_empty(v.shape) if out is None else out
  ht = torch.empty_like(h0)
  st = (torch.empty(
      (rows, HV, V,
       K), dtype=torch.float32, device=q.device) if states else q.new_empty(0))
  pin = (torch.empty((HV, V, K), dtype=torch.float32, device=q.device)
         if pin_at is not None else q.new_empty(0))
  _module().gdn_recurrence(q, k, v, g, beta, o, h0, ht, st, pin,
                           -1 if pin_at is None else pin_at - 1, T)
  return RecurrenceOut(o, ht, st if states else None,
                       pin if pin_at is not None else None)
