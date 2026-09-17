"""The op under optimization: the QSA indexer's masked top-k selection.

`masked_topk(logits, visible, out)` selects, per row, the k = out.shape[1]
highest-scoring visible columns of a fp32 score matrix and writes their
column indices as int32, `-1` where a row has fewer than k visible
columns. Columns at or past a row's visible count are unwritten scratch
(the producer allocates with torch.empty) and may hold anything,
including NaN — they must be excluded by the bound, never by value.

The selected SET must be canonical: descending score, and among tied
scores the lowest column index wins. Output order within a row does not
matter (the consumer canonicalizes), but the set must be identical on
every call — the serving engine's run-to-run bit stability rests on it.

This baseline is the production code as it stands: four passes over the
row (an out-of-place masked_fill = copy + fill, two radix passes plus a
gather inside torch.topk, a gather + where + copy for the -1 fill).
"""

import torch


def masked_topk(logits: torch.Tensor, visible: torch.Tensor,
                out: torch.Tensor) -> None:
    """logits: [rows, cols] fp32 cuda; visible: [rows] int32; out: [rows, k] int32."""
    k = out.shape[1]
    columns = torch.arange(logits.shape[1], device=logits.device)[None, :]
    masked = logits.masked_fill(columns >= visible[:, None], float("-inf"))
    if masked.shape[1] < k:
        pad = masked.new_full((masked.shape[0], k - masked.shape[1]), float("-inf"))
        masked = torch.cat([masked, pad], dim=1)
    picked = torch.topk(masked, k, dim=1, sorted=False).indices
    hidden = masked.gather(1, picked) == float("-inf")
    out.copy_(torch.where(hidden, torch.full_like(picked, -1), picked).int())
