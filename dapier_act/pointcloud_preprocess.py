from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class TaskFrameConfig:
    """
    Task frame T defined inside RoboTwin world frame W.

    origin_world:
        World-coordinate position of the Task-frame origin, ^W t_T.

    rpy_deg:
        Roll/Pitch/Yaw of the Task-frame axes relative to World.
        R_WT = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    A world point is converted by:
        p_T = R_WT.T @ (p_W - t_WT)
    """
    origin_world: Tuple[float, float, float] = (0.0, -0.35, 0.74)
    rpy_deg: Tuple[float, float, float] = (0.0, 0.0, 90.0)
    crop_min: Tuple[float, float, float] = (0.00, -0.25, 0.005)
    crop_max: Tuple[float, float, float] = (0.40, 0.25, 0.30)


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def task_rotation_in_world(rpy_deg: Iterable[float]) -> np.ndarray:
    roll_deg, pitch_deg, yaw_deg = [float(v) for v in rpy_deg]
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    return _rz(yaw) @ _ry(pitch) @ _rx(roll)


def world_to_task(xyz_world: np.ndarray, cfg: TaskFrameConfig) -> np.ndarray:
    xyz_world = np.asarray(xyz_world, dtype=np.float32)
    if xyz_world.ndim != 2 or xyz_world.shape[1] != 3:
        raise ValueError(f"xyz_world must have shape [N,3], got {xyz_world.shape}")

    origin = np.asarray(cfg.origin_world, dtype=np.float32).reshape(1, 3)
    r_wt = task_rotation_in_world(cfg.rpy_deg)
    return (xyz_world - origin) @ r_wt


def crop_task_points(xyz_task: np.ndarray, cfg: TaskFrameConfig):
    xyz_task = np.asarray(xyz_task, dtype=np.float32)
    lo = np.asarray(cfg.crop_min, dtype=np.float32)
    hi = np.asarray(cfg.crop_max, dtype=np.float32)

    if np.any(lo >= hi):
        raise ValueError(f"crop_min must be < crop_max: min={lo}, max={hi}")

    finite = np.isfinite(xyz_task).all(axis=1)
    inside = ((xyz_task >= lo) & (xyz_task <= hi)).all(axis=1)
    mask = finite & inside
    return xyz_task[mask], mask


def transform_and_crop(xyz_world: np.ndarray, cfg: TaskFrameConfig) -> Dict[str, np.ndarray]:
    xyz_world = np.asarray(xyz_world, dtype=np.float32)
    finite_world = np.isfinite(xyz_world).all(axis=1)
    xyz_world_finite = xyz_world[finite_world]

    xyz_task = world_to_task(xyz_world_finite, cfg)
    xyz_crop, crop_mask_on_finite = crop_task_points(xyz_task, cfg)

    return {
        "world": xyz_world_finite,
        "task": xyz_task,
        "cropped": xyz_crop,
        "finite_world_mask": finite_world,
        "crop_mask_on_finite": crop_mask_on_finite,
    }


def linspace_candidates(xyz: torch.Tensor, candidate_points: int) -> torch.Tensor:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N,3], got {tuple(xyz.shape)}")

    n = int(xyz.shape[0])
    if n == 0:
        return xyz
    if candidate_points <= 0 or n <= candidate_points:
        return xyz

    idx = torch.linspace(
        0,
        n - 1,
        steps=int(candidate_points),
        device=xyz.device,
    ).round().long()
    return xyz.index_select(0, idx)


@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, num_points: int) -> torch.Tensor:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N,3], got {tuple(xyz.shape)}")

    n = int(xyz.shape[0])
    k = int(num_points)

    if n == 0:
        return xyz.new_zeros((0, 3))
    if k <= 0:
        return xyz
    if n <= k:
        if n == k:
            return xyz
        repeat_idx = torch.arange(k, device=xyz.device) % n
        return xyz.index_select(0, repeat_idx)

    centroids = torch.empty(k, dtype=torch.long, device=xyz.device)
    min_dist = torch.full((n,), float("inf"), dtype=xyz.dtype, device=xyz.device)

    center = xyz.mean(dim=0, keepdim=True)
    farthest = torch.sum((xyz - center) ** 2, dim=1).argmax()

    for i in range(k):
        centroids[i] = farthest
        centroid = xyz[farthest].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        min_dist = torch.minimum(min_dist, dist)
        farthest = min_dist.argmax()

    return xyz.index_select(0, centroids)


@torch.no_grad()
def sample_task_points(
    xyz_task_cropped: np.ndarray,
    candidate_points: int = 8192,
    num_points: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """Deterministic metric-space sampling used by DAPIER-ACT point encoders.

    The order intentionally stays:
        cropped Task-frame XYZ (meters)
        -> candidate reduction
        -> farthest-point sampling

    Normalization is applied only after FPS so Euclidean distances are measured
    in the physical Task-frame metric space.
    """
    xyz = np.asarray(xyz_task_cropped, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz_task_cropped must have shape [N,3], got {xyz.shape}")

    if xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    torch_device = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    t = torch.from_numpy(xyz).to(torch_device)
    t = linspace_candidates(t, candidate_points)
    t = farthest_point_sample(t, num_points)
    return t.detach().cpu().numpy().astype(np.float32)


def limits_normalize(
    xyz: np.ndarray,
    point_min: np.ndarray,
    point_max: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """DP3-style per-axis limits normalization to approximately [-1, 1].

    No clipping is applied. Validation/real points outside the train limits are
    therefore allowed to produce values outside [-1, 1], matching the usual
    linear-normalizer behavior.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    lo = np.asarray(point_min, dtype=np.float32).reshape(1, 3)
    hi = np.asarray(point_max, dtype=np.float32).reshape(1, 3)

    scale = hi - lo
    scale = np.where(scale < eps, 1.0, scale).astype(np.float32)
    return (2.0 * (xyz - lo) / scale - 1.0).astype(np.float32)


@torch.no_grad()
def sample_for_spatial256(
    xyz_task_cropped: np.ndarray,
    candidate_points: int = 8192,
    num_point_tokens: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """Backward-compatible Spatial256 helper."""
    return sample_task_points(
        xyz_task_cropped,
        candidate_points=candidate_points,
        num_points=num_point_tokens,
        device=device,
    )
