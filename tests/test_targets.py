from __future__ import annotations

import pytest
import torch

from src.targets import make_target, make_target_cross, make_target_T


class TestMakeTargetT:
    def test_shape(self):
        mask = make_target_T(15, 15)
        assert mask.shape == (15, 15)
        assert mask.dtype == torch.bool

    def test_has_cells(self):
        mask = make_target_T(15, 15)
        assert mask.sum().item() > 0

    def test_t_shape_structure(self):
        """T-shape should have a horizontal bar near top and vertical stem."""
        mask = make_target_T(15, 15)
        # Top half should have a wide bar
        top_filled = mask[:5, :].sum().item()
        assert top_filled > 0
        # Center column should be filled
        center = 15 // 2
        center_col_filled = mask[:, center].sum().item()
        assert center_col_filled > 0

    def test_small_grid(self):
        mask = make_target_T(5, 5)
        assert mask.shape == (5, 5)
        assert mask.sum().item() > 0


class TestMakeTargetCross:
    def test_shape(self):
        mask = make_target_cross(15, 15)
        assert mask.shape == (15, 15)
        assert mask.dtype == torch.bool

    def test_has_horizontal_and_vertical_arms(self):
        mask = make_target_cross(15, 15)
        center_row = mask[7, :].sum().item()
        center_col = mask[:, 7].sum().item()
        # Both center row and center col should be mostly filled
        assert center_row >= 10
        assert center_col >= 10

    def test_small_grid(self):
        mask = make_target_cross(5, 5)
        assert mask.sum().item() > 0


class TestMakeTarget:
    def test_t_by_name(self):
        mask = make_target("T", 15, 15)
        expected = make_target_T(15, 15)
        assert torch.equal(mask, expected)

    def test_cross_by_name(self):
        mask = make_target("cross", 15, 15)
        expected = make_target_cross(15, 15)
        assert torch.equal(mask, expected)

    def test_case_insensitive(self):
        mask1 = make_target("T", 15, 15)
        mask2 = make_target("t", 15, 15)
        assert torch.equal(mask1, mask2)

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            make_target("nonexistent", 15, 15)

    def test_tee_alias(self):
        mask1 = make_target("tee", 15, 15)
        mask2 = make_target_T(15, 15)
        assert torch.equal(mask1, mask2)
