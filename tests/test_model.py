from __future__ import annotations

import torch

from src.model import NUM_ACTIONS, NUM_CELL_TYPES, NUM_NEIGHBORS, LittleLM
from src.utils import set_seed


class TestLittleLM:
    def test_output_shape_single(self):
        model = LittleLM(embed_dim=16, num_heads=2)
        model.eval()
        x = torch.randint(0, NUM_CELL_TYPES, (1, NUM_NEIGHBORS))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, NUM_ACTIONS)

    def test_output_shape_batch(self):
        model = LittleLM(embed_dim=32, num_heads=4)
        model.eval()
        x = torch.randint(0, NUM_CELL_TYPES, (10, NUM_NEIGHBORS))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (10, NUM_ACTIONS)

    def test_empty_input(self):
        model = LittleLM()
        model.eval()
        x = torch.zeros((0, NUM_NEIGHBORS), dtype=torch.long)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (0, NUM_ACTIONS)

    def test_deterministic_with_seed(self):
        set_seed(42)
        m1 = LittleLM(embed_dim=16, num_heads=2)
        m1.eval()
        x = torch.randint(0, NUM_CELL_TYPES, (5, NUM_NEIGHBORS))

        set_seed(42)
        m2 = LittleLM(embed_dim=16, num_heads=2)
        m2.eval()

        with torch.no_grad():
            out1 = m1(x)
            out2 = m2(x)
        assert torch.allclose(out1, out2)

    def test_default_config(self):
        model = LittleLM()
        assert model.embed_dim == 32
        assert model.num_heads == 4
