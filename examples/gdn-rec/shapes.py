"""The served shape: Qwen3.8-Flash-Next's gated-delta-net recurrence on
the DGX Spark, one layer's call.

Per GDN layer (36 of them): H=16 QK heads, HV=48 V heads (GQA ratio 3),
K=V=128. The recurrent state is fp32 `[HV, V, K]` — 3 MiB per layer per
sequence. Inputs per row: q and k fp32 `[H, K]` (the attention scale
already folded into q), v bf16 `[HV, V]`, g and beta fp32 `[HV]` (g is
a log-decay, ≤ 0; beta in (0, 1]). Output o bf16 `[HV, V]`.

The four launches the engine runs, all through one kernel and one
arithmetic (the phase pin):

  t1       T=1, n=1                 a plain decode step (drafter off)
  t5       T=5, n=1, states=True    the served verify pass (draft depth
                                    4): the state after every row is
                                    written for rollback's selection
  chunk    T=8192, n=1              the prefill chunk
  pinchunk T=8192, n=1, pin_at=4096 a prompt's final chunk with the
                                    residency pin: one extra state write
                                    at row pin_at-1
"""

H = 16
HV = 48
HEAD_DIM = 128
LAYERS = 36

# name -> (T, n, states, pin_at)
SHAPES = [
    ("t1", (1, 1, False, None)),
    ("t5", (5, 1, True, None)),
    ("chunk", (8192, 1, False, None)),
    ("pinchunk", (8192, 1, False, 4096)),
]
