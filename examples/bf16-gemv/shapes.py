"""The real per-pass shape mix: (N, K, count per forward pass, name).

One decode step of the target model runs every entry `count` times
with a distinct weight each time. N and K are the torch.nn.Linear
[out, in] dimensions.
"""

SHAPES = [
    (10240, 2560, 36, "gdn in_proj_qkv"),
    (6144, 2560, 36, "gdn in_proj_z"),
    (2560, 6144, 36, "gdn out_proj"),
    (48, 2560, 72, "gdn in_proj_a/b"),
    (12288, 2560, 12, "attn q_proj"),
    (512, 2560, 24, "attn k/v_proj"),
    (2560, 6144, 12, "attn o_proj"),
    (640, 2560, 12, "indexer qk_proj"),
    (320, 10240, 96, "hc mix down"),
    (10240, 320, 96, "hc mix up"),
    (640, 2560, 96, "shared gate/up"),
    (2560, 640, 48, "shared down"),
    (512, 2560, 48, "moe gate"),
    (1, 2560, 48, "shared_expert_gate"),
]
