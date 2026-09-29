"""The two served shapes that run through the M-invariant linear at
decode and verify (Qwen3.8-Flash-Next on the DGX Spark), and the row
counts the engine calls them with.

  down_inject  [N=336, K=10240]   96 calls per step (one per hyper-connection
                                  layer), the loop's target
  kv_proj      [N=12800, K=2560]  1 call per step

M=1 is a plain decode step, M=5 the served verify pass (draft depth 4);
the prefill chunk runs the same kernel at M up to 8192 through the
'wide' tile and its bytes must not move either.
"""

SHAPES = {
    'down_inject': (336, 10240, 96),
    'kv_proj': (12800, 2560, 1),
}
SERVED_M = (1, 5)
