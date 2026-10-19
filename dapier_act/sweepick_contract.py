"""Source-bound Sweepick 2026-10-19 Task-frame contract (pure NumPy)."""
import numpy as np
from .pointcloud_preprocess import TaskFrameConfig, transform_and_crop, sample_task_points, limits_normalize

SWEEPICK_TASK = TaskFrameConfig(
    origin_world=(0.0, -0.2539, 0.0),
    rpy_deg=(0.0, 0.0, 90.0),
    crop_min=(0.15, -0.26, 0.005),
    crop_max=(0.32, 0.26, 0.40),
)


def normalized_sweepick_pointcloud(xyz_world, *, sample_device="cpu"):
    """World XYZ(m) -> fixed Task crop -> FPS512(m) -> fixed [-1,1].

    The production pretraining DataLoader MUST explicitly call/route to this
    contract; creating this helper does not switch legacy model inputs.
    """
    xyz_task = transform_and_crop(xyz_world, SWEEPICK_TASK)["cropped"]
    if len(xyz_task) == 0:
        raise ValueError("No Sweepick task points remain after crop; reject episode")
    xyz = sample_task_points(xyz_task, candidate_points=8192, num_points=512, device=sample_device)
    return limits_normalize(xyz, SWEEPICK_TASK.crop_min, SWEEPICK_TASK.crop_max)
