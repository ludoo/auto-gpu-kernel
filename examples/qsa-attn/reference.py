"""The reference: sparkle's shipped QSA sparse paged attention kernel,
verbatim (sparkle/kernels/qsa.py at main 2026-09-24, itself a port of
the vLLM `[R]` tree's qsa.py), launched in the served configuration —
BLOCK_N 32, one split, one warp, num_stages 2 — for every row count.

This file is READ-ONLY for the optimization loop. `test_attn.py`
compares the candidate in `qsa_attn.py` against `reference_attention`
bitwise; `bench.py` also times the table's split rows through
`reference_attention_config` for the floor. Do not import this module
from the candidate's main path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from shapes import HEAD_DIM, HEADS, KV_HEADS, PAGE, TILE, WIDTH

@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
  row = tl.program_id(0)
  kv_head = tl.program_id(1)
  split_id = tl.program_id(2)
  request = tl.load(token_to_req_ptr + row)
  safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

  # The packed selection buffer carries one TRAILING COUNT COLUMN per row
  # (column TOPK of a TOPK+1-wide buffer): the row's valid-entry count,
  # written by the expand kernel. It is never a token index — the tile loop
  # and the index load below only ever cover columns [0, TOPK).
  valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)

  head_offsets = tl.arange(0, BLOCK_M)
  dim_offsets = tl.arange(0, HEAD_DIM)
  column_offsets = tl.arange(0, BLOCK_N)
  first_head = kv_head * GROUP_SIZE
  query = tl.load(
      q_ptr + row * stride_q_row +
      (first_head + head_offsets[:, None]) * stride_q_head +
      dim_offsets[None, :],
      mask=head_offsets[:, None] < GROUP_SIZE,
      other=0.0,
  )

  max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
  normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
  accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
  softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

  tile_end = tl.minimum(NUM_TILES,
                        tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

  for tile in range(split_id, tile_end, NUM_SPLITS):
    columns = tile * BLOCK_N + column_offsets
    logical_token = tl.load(
        indices_ptr + row * stride_indices_row + columns,
        mask=columns < TOPK,
        other=-1,
    )
    safe_token = tl.maximum(logical_token, 0)
    logical_page = safe_token // PAGE_SIZE
    page_offset = safe_token % PAGE_SIZE
    valid = ((request >= 0) & (request < num_requests) & (logical_token >= 0) &
             (logical_page < PAGE_TABLE_WIDTH))
    physical_page = tl.load(
        block_table_ptr + safe_request * stride_table_req +
        tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    # physical_page * block stride can overflow int32 for large caches.
    safe_page = tl.maximum(physical_page, 0).to(tl.int64)
    keys = tl.load(
        k_cache_ptr + safe_page[None, :] * stride_k_block +
        page_offset[None, :] * stride_k_token + kv_head * stride_k_head +
        dim_offsets[:, None],
        mask=valid[None, :],
        other=0.0,
    )
    values = tl.load(
        v_cache_ptr + safe_page[:, None] * stride_v_block +
        page_offset[:, None] * stride_v_token + kv_head * stride_v_head +
        dim_offsets[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    scores = tl.dot(query, keys)
    # Scaling scores avoids re-quantizing a scaled query to BF16.
    scores *= softmax_scale_log2
    scores = tl.where(valid[None, :], scores, -1.0e20)
    next_max = tl.maximum(max_value, tl.max(scores, axis=1))
    alpha = tl.math.exp2(max_value - next_max)
    probabilities = tl.where(valid[None, :],
                             tl.math.exp2(scores - next_max[:, None]), 0.0)
    accumulator = tl.dot(
        probabilities.to(values.dtype),
        values,
        acc=accumulator * alpha[:, None],
    )
    normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
    max_value = next_max

  has_values = normalizer > 0
  normalized_output = tl.where(
      has_values[:, None],
      accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
      0.0,
  )
  output_mask = head_offsets[:, None] < GROUP_SIZE
  if NUM_SPLITS == 1:
    tl.store(
        output_ptr + row * stride_output_row +
        (first_head + head_offsets[:, None]) * stride_output_head +
        dim_offsets[None, :],
        normalized_output,
        mask=output_mask,
    )
  else:
    partial_lse = tl.where(
        has_values,
        max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
        -float("inf"),
    )
    partial_row = (split_id * num_rows + row).to(tl.int64)
    tl.store(
        partial_output_ptr +
        (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None]) *
        HEAD_DIM + dim_offsets[None, :],
        normalized_output,
        mask=output_mask,
    )
    tl.store(
        partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head +
        head_offsets,
        partial_lse,
        mask=head_offsets < GROUP_SIZE,
    )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
  row = tl.program_id(0)
  head = tl.program_id(1)
  split_offsets = tl.arange(0, BLOCK_SPLITS)
  dim_offsets = tl.arange(0, HEAD_DIM)
  split_mask = split_offsets < NUM_SPLITS
  lse = tl.load(
      partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS +
      head,
      mask=split_mask,
      other=-float("inf"),
  )
  lse_max = tl.max(lse, axis=0)
  has_values = lse_max > -float("inf")
  shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
  weights = tl.math.exp2(shifted)
  denominator = tl.sum(weights, axis=0)
  split_rows = split_offsets.to(tl.int64) * num_rows + row
  partial_output = tl.load(
      partial_output_ptr +
      (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM +
      dim_offsets[None, :],
      mask=split_mask[:, None],
      other=0.0,
  )
  merged = tl.sum(partial_output * weights[:, None], axis=0)
  merged = tl.where(denominator > 0, merged / denominator, 0.0)
  tl.store(
      output_ptr + row * stride_output_row + head * stride_output_head +
      dim_offsets,
      merged,
  )


def reference_attention_config(q, k_cache, v_cache, packed, block_table,
                               token_to_req, out, *, block_n, splits, warps,
                               stages):
  """The shipped kernel at an explicit (block_n, splits, warps, stages)."""
  rows = q.shape[0]
  group = HEADS // KV_HEADS
  num_tiles = triton.cdiv(WIDTH, block_n)
  num_splits = min(splits, num_tiles)
  if num_splits == 1:
    partial_output, partial_lse = out, out
  else:
    partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32,
                                 device=q.device)
    partial_lse = torch.empty((num_splits, rows, HEADS), dtype=torch.float32,
                              device=q.device)
  _qsa_sparse_paged_gqa_splitk_kernel[(rows, KV_HEADS, num_splits)](
      q, k_cache, v_cache, packed, block_table, token_to_req, partial_output,
      partial_lse, out, q.stride(0), q.stride(1), k_cache.stride(0),
      k_cache.stride(1), k_cache.stride(2), v_cache.stride(0),
      v_cache.stride(1), v_cache.stride(2), packed.stride(0),
      block_table.stride(0), out.stride(0), out.stride(1), rows,
      k_cache.shape[0], block_table.shape[0], TOPK=WIDTH, PAGE_SIZE=PAGE,
      PAGE_TABLE_WIDTH=block_table.shape[1], GROUP_SIZE=group,
      HEAD_DIM=HEAD_DIM, NUM_QUERY_HEADS=HEADS, NUM_SPLITS=num_splits,
      NUM_TILES=num_tiles, BLOCK_M=triton.next_power_of_2(group),
      BLOCK_N=block_n, num_warps=warps, num_stages=stages)
  if num_splits == 1:
    return out
  _qsa_merge_splitk_kernel[(rows, HEADS)](
      partial_output, partial_lse, out, out.stride(0), out.stride(1), rows,
      HEAD_DIM=HEAD_DIM, NUM_QUERY_HEADS=HEADS, NUM_SPLITS=num_splits,
      BLOCK_SPLITS=triton.next_power_of_2(num_splits), num_warps=2,
      num_stages=1)
  return out


def reference_attention(q, k_cache, v_cache, packed, block_table, token_to_req,
                        out):
  """The served configuration: the bytes every candidate must reproduce."""
  return reference_attention_config(q, k_cache, v_cache, packed, block_table,
                                    token_to_req, out, block_n=TILE,
                                    splits=1, warps=1, stages=2)
