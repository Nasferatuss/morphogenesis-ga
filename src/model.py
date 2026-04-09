from __future__ import annotations

import torch
from torch import nn

NUM_CELL_TYPES = 4
NUM_ACTIONS = 5
NUM_NEIGHBORS = 8


class LittleLM(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.token_embed = nn.Embedding(NUM_CELL_TYPES, embed_dim)
        self.pos_embed = nn.Embedding(NUM_NEIGHBORS, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, NUM_ACTIONS)
        self.register_buffer(
            "position_ids", torch.arange(NUM_NEIGHBORS, dtype=torch.long), persistent=False
        )

    def forward(self, neighbor_tokens: torch.Tensor) -> torch.Tensor:
        if neighbor_tokens.numel() == 0:
            return torch.zeros((0, NUM_ACTIONS), device=neighbor_tokens.device)
        pos_ids = self.position_ids.to(neighbor_tokens.device)
        pos_ids = pos_ids.unsqueeze(0).expand(neighbor_tokens.size(0), -1)
        token_embed = self.token_embed(neighbor_tokens)
        x = token_embed + self.pos_embed(pos_ids)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        pooled = attn_out.mean(dim=1)
        pooled = self.norm(pooled)
        logits = self.head(pooled)
        return logits
