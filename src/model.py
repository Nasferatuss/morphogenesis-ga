from __future__ import annotations

from typing import Optional

import torch
from torch import nn

NUM_CELL_TYPES = 4
NUM_ACTIONS = 9  # 5 legacy + 4 directional divides (N/S/E/W)
NUM_NEIGHBORS = 8


class LittleLM(nn.Module):
    """Compact transformer controller with optional spatial-position awareness.

    When ``use_position=True`` (default), the forward pass accepts a normalized
    ``[y, x]`` coordinate for each cell in [0, 1] and mixes a learned positional
    projection into the attention-pooled representation. This lets cells know
    *where* they are in the grid in addition to *what* surrounds them.

    Backwards compatible: if callers don't pass ``positions``, the position
    contribution is zero and behavior matches the legacy model.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 4,
        use_position: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_position = use_position
        self.token_embed = nn.Embedding(NUM_CELL_TYPES, embed_dim)
        self.pos_embed = nn.Embedding(NUM_NEIGHBORS, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        if use_position:
            # Projects normalized (y, x) coords into the embedding space.
            self.spatial_proj = nn.Linear(2, embed_dim)
        self.head = nn.Linear(embed_dim, NUM_ACTIONS)
        self.register_buffer(
            "position_ids", torch.arange(NUM_NEIGHBORS, dtype=torch.long), persistent=False
        )

    def forward(
        self,
        neighbor_tokens: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if neighbor_tokens.numel() == 0:
            return torch.zeros((0, NUM_ACTIONS), device=neighbor_tokens.device)
        pos_ids = self.position_ids.to(neighbor_tokens.device)
        pos_ids = pos_ids.unsqueeze(0).expand(neighbor_tokens.size(0), -1)
        token_embed = self.token_embed(neighbor_tokens)
        x = token_embed + self.pos_embed(pos_ids)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        pooled = attn_out.mean(dim=1)
        if self.use_position and positions is not None and positions.numel() > 0:
            # positions: (N, 2) with normalized [y, x] in [0, 1]
            spatial = self.spatial_proj(positions.to(pooled.dtype))
            pooled = pooled + spatial
        pooled = self.norm(pooled)
        logits = self.head(pooled)
        return logits
