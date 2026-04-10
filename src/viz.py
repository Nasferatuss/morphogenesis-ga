from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None

EMPTY_COLOR = (0, 0, 0)
STEM_COLOR = (170, 170, 170)
OVERLAP_COLOR = (0, 255, 80)
TARGET_ONLY_COLOR = (20, 60, 180)
PREDICTION_ONLY_COLOR = (255, 190, 0)
_DEFAULT_RENDER_SIZE = 256
_MAX_RENDER_SIZE = 512


def _build_color_canvas(grid_np: np.ndarray, target_bool: Optional[np.ndarray]) -> np.ndarray:
    height, width = grid_np.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if target_bool is None:
        target_mask = np.zeros((height, width), dtype=bool)
    else:
        target_arr = np.asarray(target_bool, dtype=bool)
        if target_arr.ndim > 2:
            target_arr = np.squeeze(target_arr)
        if target_arr.shape != (height, width):
            resized = np.zeros((height, width), dtype=bool)
            common_h = min(height, target_arr.shape[0])
            common_w = min(width, target_arr.shape[-1])
            resized[:common_h, :common_w] = target_arr[:common_h, :common_w]
            target_mask = resized
        else:
            target_mask = target_arr
    stem_mask = grid_np == 1
    prediction_mask = np.logical_or(grid_np == 2, grid_np == 3)
    overlap_mask = np.logical_and(prediction_mask, target_mask)
    target_only_mask = np.logical_and(target_mask, np.logical_not(prediction_mask))
    prediction_only_mask = np.logical_and(prediction_mask, np.logical_not(target_mask))
    canvas[target_only_mask] = TARGET_ONLY_COLOR
    canvas[prediction_only_mask] = PREDICTION_ONLY_COLOR
    canvas[overlap_mask] = OVERLAP_COLOR
    canvas[stem_mask] = STEM_COLOR
    return canvas




def render_grid_to_rgb(
    grid: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
    target_size: int = _DEFAULT_RENDER_SIZE,
) -> np.ndarray:
    if grid.numel() == 0:
        base = np.zeros((1, 1, 3), dtype=np.uint8)
    else:
        grid_np = grid.to(torch.long).cpu().numpy()
        target_np = None
        if target_mask is not None:
            target_np = target_mask.to(dtype=torch.bool).cpu().numpy()
        base = _build_color_canvas(grid_np, target_np)

    max_dim = max(base.shape[0], base.shape[1])
    preferred = int(target_size) if target_size else _DEFAULT_RENDER_SIZE
    preferred = min(_MAX_RENDER_SIZE, max(_DEFAULT_RENDER_SIZE, preferred))
    chosen_size = preferred if max_dim <= preferred else _MAX_RENDER_SIZE
    scale = max(1, math.ceil(chosen_size / max(1, max_dim)))
    scaled = np.repeat(np.repeat(base, scale, axis=0), scale, axis=1)
    if scaled.shape[0] < chosen_size or scaled.shape[1] < chosen_size:
        canvas = np.zeros((chosen_size, chosen_size, 3), dtype=np.uint8)
        canvas[: scaled.shape[0], : scaled.shape[1]] = scaled
        resized = canvas
    else:
        resized = scaled[:chosen_size, :chosen_size]
    return resized



class Visualizer:
    def __init__(
        self,
        width: int,
        height: int,
        cell_size: int,
        target_mask: torch.Tensor,
        fps: int = 30,
    ) -> None:
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.target_mask = target_mask.to(dtype=torch.bool).cpu()
        self._target_bool = self.target_mask.numpy()
        self.fps = max(0, int(fps))
        self._enabled = pygame is not None
        self.surface: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self._capture_active = False
        self._capture_stride = 1
        self._capture_frames: List[np.ndarray] = []

        if not self._enabled:
            print("[WARN] Pygame is not available. Visualization disabled.")
            return

        pygame.init()
        size = (self.width * self.cell_size, self.height * self.cell_size)
        self.surface = pygame.display.set_mode(size)
        pygame.display.set_caption("Morphogenesis World")
        self.clock = pygame.time.Clock()

    def render(self, grid: torch.Tensor, target_mask: torch.Tensor, step: int, iou: float) -> None:
        if not self._enabled or self.surface is None:
            return
        incoming_mask = target_mask.to(dtype=torch.bool).cpu()
        if incoming_mask.shape != self.target_mask.shape or not torch.equal(incoming_mask, self.target_mask):
            self.target_mask = incoming_mask.clone()
            self._target_bool = self.target_mask.numpy()

        self._pump_events()
        numpy_grid = grid.to(torch.long).cpu().numpy()
        color_canvas = _build_color_canvas(numpy_grid, self._target_bool)
        for y in range(self.height):
            for x in range(self.width):
                color = tuple(int(value) for value in color_canvas[y, x])
                rect = pygame.Rect(
                    x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size
                )
                pygame.draw.rect(self.surface, color, rect)
        pygame.display.set_caption(f"Morphogenesis :: step {step} :: IoU {iou:.3f}")
        pygame.display.flip()
        self._maybe_capture(step)
        self._throttle()


    def _pump_events(self) -> None:
        if not self._enabled:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._enabled = False
                print("[WARN] Visualization window closed by user.")
                break

    def _throttle(self) -> None:
        if not self._enabled or self.clock is None or self.fps <= 0:
            return
        self.clock.tick(self.fps)

    def start_capture(self, frame_stride: int = 1) -> None:
        if not self._enabled:
            return
        self._capture_frames = []
        self._capture_active = True
        self._capture_stride = max(1, int(frame_stride))

    def stop_capture(self) -> None:
        self._capture_active = False

    def capture_frame(self, surface: Optional[pygame.Surface] = None) -> None:
        if not self._enabled or not self._capture_active:
            return
        if surface is None:
            surface = self.surface
        if surface is None:
            return
        array = pygame.surfarray.array3d(surface)
        frame = np.transpose(array, (1, 0, 2)).copy()
        self._capture_frames.append(frame)

    def _maybe_capture(self, step: int) -> None:
        if not self._capture_active:
            return
        if step % self._capture_stride == 0:
            self.capture_frame(self.surface)

    def save_gif(self, path: str, fps: int, max_frames: int) -> None:
        if not self._capture_frames:
            print("[WARN] No captured frames available for GIF export.")
            return
        try:
            from imageio import v2 as imageio
        except ImportError:
            print("[WARN] imageio is required to export GIFs. Install imageio to enable this feature.")
            return

        frames = self._capture_frames
        if max_frames > 0 and len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
            frames = [frames[idx] for idx in indices]
        duration = 1.0 / max(1, int(fps))
        imageio.mimsave(str(path), frames, duration=duration)
        self.stop_capture()

    def close(self) -> None:
        if not self._enabled:
            return
        pygame.display.quit()
        pygame.quit()
        self._enabled = False
        self.stop_capture()


class GifRecorder:
    def __init__(self) -> None:
        self._frames: List[np.ndarray] = []
        self._max_frames = 0

    def start_capture(self, max_frames: int) -> None:
        self._frames = []
        try:
            value = int(max_frames)
        except (TypeError, ValueError):
            value = 0
        self._max_frames = max(0, value)

    def capture_frame(self, rgb_frame: np.ndarray) -> None:
        if rgb_frame is None:
            return
        frame = np.asarray(rgb_frame, dtype=np.uint8)
        self._frames.append(frame.copy())
        if self._max_frames > 0 and len(self._frames) > self._max_frames:
            self._frames = self._frames[-self._max_frames :]

    def save_gif(self, path: Union[str, Path], fps: int) -> None:
        if not self._frames:
            print("[WARN] No captured frames available for GIF export.")
            return
        imageio = _require_imageio()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        duration = 1.0 / max(1, int(fps))
        imageio.mimsave(str(target), self._frames, duration=duration)
        self.stop_capture()

    def stop_capture(self) -> None:
        self._frames = []


def _require_imageio():
    try:
        from imageio import v2 as imageio
    except ImportError as err:  # pragma: no cover - optional dep
        message = "imageio is required for GIF export. Install it via 'pip install imageio pillow'."
        print(f"[ERROR] {message}")
        raise RuntimeError(message) from err
    return imageio
