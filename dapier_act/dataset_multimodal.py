import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dapier_act.dataset import (
    concat_joint_group,
    resolve_episode_files,
    get_pointcloud_length,
    read_pointcloud_frame,
)

from dapier_act.pointcloud_preprocess import (
    TaskFrameConfig,
    sample_task_points,
    transform_and_crop,
)


def decode_rgb_frame(
    root,
    camera_name,
    timestep,
):
    """
    RoboTwin HDF5 JPEG bytes
        -> OpenCV BGR
        -> RGB uint8 [H, W, 3]
    """
    key = f"vision/{camera_name}/colors"

    if key not in root:
        raise KeyError(
            f"Camera RGB dataset not found: {key}"
        )

    raw = root[key][timestep]

    buffer = np.frombuffer(
        raw,
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        buffer,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise ValueError(
            f"Failed to decode {camera_name} "
            f"at timestep {timestep}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return np.ascontiguousarray(
        image_rgb
    )


def rgb_to_act_tensor(image_rgb):
    """
    Original ACT-style preprocessing.

    uint8 RGB [H, W, 3]
        -> float32 [3, H, W]
        -> /255.0

    No resize.
    No ImageNet mean/std normalization.
    """
    tensor = torch.from_numpy(
        image_rgb
    ).permute(
        2,
        0,
        1,
    ).float()

    tensor = tensor / 255.0

    return tensor


def precompute_dp3_pointclouds(
    dataset_dir,
    episode_files=None,
    task_frame_cfg=None,
    candidate_points=8192,
    num_point_samples=512,
    device="cuda",
    fit_limits=False,
):
    """Precompute Task-crop + FPS point clouds once per frame.

    DP3-style policies normally consume an already fixed-size point cloud.
    Keeping this cache outside the model avoids repeating expensive FPS on
    every epoch while preserving the raw HDF5 dataset unchanged.

    Returns:
        cache: {(str(path), timestep): float32 [num_point_samples, 3]}
        point_min, point_max: train-only sampled XYZ limits when fit_limits=True
    """
    files = resolve_episode_files(
        dataset_dir,
        episode_files,
    )

    cfg = (
        task_frame_cfg
        if task_frame_cfg is not None
        else TaskFrameConfig()
    )

    cache = {}
    point_min = np.full(3, np.inf, dtype=np.float64)
    point_max = np.full(3, -np.inf, dtype=np.float64)
    sampled_count = 0
    frame_count = 0

    for episode_index, path in enumerate(files, start=1):
        with h5py.File(path, "r") as root:
            length = get_pointcloud_length(root)

            for timestep in range(length):
                xyz_world = read_pointcloud_frame(
                    root,
                    timestep,
                )

                processed = transform_and_crop(
                    xyz_world[:, :3],
                    cfg,
                )
                xyz_crop = processed["cropped"]

                if xyz_crop.shape[0] == 0:
                    raise RuntimeError(
                        "Task crop produced zero points while precomputing "
                        f"DP3 PointCloud: path={path}, timestep={timestep}"
                    )

                sampled = sample_task_points(
                    xyz_crop,
                    candidate_points=candidate_points,
                    num_points=num_point_samples,
                    device=device,
                )

                if sampled.shape[0] == 0:
                    raise RuntimeError(
                        "Point sampling produced zero points while precomputing "
                        f"DP3 PointCloud: path={path}, timestep={timestep}"
                    )

                sampled = np.ascontiguousarray(
                    sampled,
                    dtype=np.float32,
                )

                cache[(str(path), int(timestep))] = sampled

                if fit_limits:
                    point_min = np.minimum(
                        point_min,
                        sampled.min(axis=0),
                    )
                    point_max = np.maximum(
                        point_max,
                        sampled.max(axis=0),
                    )
                    sampled_count += sampled.shape[0]

                frame_count += 1

        print(
            "DP3 PointCloud cache: "
            f"episode {episode_index}/{len(files)} done "
            f"({path.name})"
        )

    if frame_count == 0:
        raise ValueError(
            "No PointCloud frames found while building DP3 cache"
        )

    if not fit_limits:
        return cache, None, None

    if sampled_count == 0:
        raise ValueError(
            "No sampled PointCloud points found while fitting DP3 limits"
        )

    if not np.isfinite(point_min).all() or not np.isfinite(point_max).all():
        raise ValueError(
            f"Non-finite DP3 point limits: min={point_min}, max={point_max}"
        )

    if np.any(point_max - point_min < 1e-4):
        raise ValueError(
            "Degenerate DP3 point range: "
            f"min={point_min}, max={point_max}"
        )

    return (
        cache,
        point_min.astype(np.float32),
        point_max.astype(np.float32),
    )


def compute_dp3_point_limits(
    dataset_dir,
    episode_files=None,
    task_frame_cfg=None,
    candidate_points=8192,
    num_point_samples=512,
    device="cuda",
):
    """Compatibility helper returning only train-set sampled XYZ limits."""
    _, point_min, point_max = precompute_dp3_pointclouds(
        dataset_dir,
        episode_files=episode_files,
        task_frame_cfg=task_frame_cfg,
        candidate_points=candidate_points,
        num_point_samples=num_point_samples,
        device=device,
        fit_limits=True,
    )
    return point_min, point_max


class DapierMultimodalDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        stats,
        chunk_size=20,
        episode_files=None,
        normalize_pointcloud=False,
        use_task_crop=True,
        task_frame_cfg=None,
        pointcloud_cache=None,
    ):
        self.files = resolve_episode_files(
            dataset_dir,
            episode_files,
        )

        self.stats = stats
        self.chunk_size = chunk_size

        self.normalize_pointcloud = (
            normalize_pointcloud
        )

        self.use_task_crop = bool(
            use_task_crop
        )

        self.task_frame_cfg = (
            task_frame_cfg
            if task_frame_cfg is not None
            else TaskFrameConfig()
        )

        self.pointcloud_cache = (
            pointcloud_cache
            if pointcloud_cache is not None
            else {}
        )

        self.samples = []

        # ----------------------------------------------------
        # Build (episode, timestep) index.
        #
        # Require all modalities in an episode to have
        # matching temporal length.
        # ----------------------------------------------------
        for path in self.files:
            with h5py.File(
                path,
                "r",
            ) as root:

                pointcloud_length = (
                    get_pointcloud_length(
                        root
                    )
                )

                state_length = len(
                    root[
                        "state/"
                        "left_arm_joint_states"
                    ]
                )

                action_length = len(
                    root[
                        "action/"
                        "left_arm_joint_states"
                    ]
                )

                left_length = len(
                    root[
                        "vision/"
                        "cam_left_wrist/"
                        "colors"
                    ]
                )

                right_length = len(
                    root[
                        "vision/"
                        "cam_right_wrist/"
                        "colors"
                    ]
                )

                lengths = {
                    "pointcloud":
                        pointcloud_length,
                    "state":
                        state_length,
                    "action":
                        action_length,
                    "left_wrist":
                        left_length,
                    "right_wrist":
                        right_length,
                }

                if len(
                    set(lengths.values())
                ) != 1:
                    raise ValueError(
                        f"Temporal length mismatch "
                        f"in {path}: {lengths}"
                    )

                length = pointcloud_length

            for timestep in range(
                length
            ):
                self.samples.append(
                    (
                        path,
                        timestep,
                    )
                )

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):
        path, timestep = (
            self.samples[index]
        )

        # ----------------------------------------------------
        # Read observation + future actions
        # ----------------------------------------------------
        with h5py.File(
            path,
            "r",
        ) as root:

            qpos_all = concat_joint_group(
                root,
                "state",
            )

            action_all = concat_joint_group(
                root,
                "action",
            )

            cache_key = (
                str(path),
                int(timestep),
            )

            pointcloud_is_precomputed = (
                cache_key in self.pointcloud_cache
            )

            if pointcloud_is_precomputed:
                pointcloud = self.pointcloud_cache[
                    cache_key
                ]
            else:
                pointcloud = (
                    read_pointcloud_frame(
                        root,
                        timestep,
                    )
                )

            left_rgb = decode_rgb_frame(
                root,
                "cam_left_wrist",
                timestep,
            )

            right_rgb = decode_rgb_frame(
                root,
                "cam_right_wrist",
                timestep,
            )

        # ----------------------------------------------------
        # Head PointCloud geometry preprocessing
        #
        # RoboTwin HDF5:
        #     World-coordinate XYZ
        #
        # DAPIER-ACT:
        #     World -> Task -> Workspace Crop
        # ----------------------------------------------------
        pointcloud = np.asarray(
            pointcloud,
            dtype=np.float32,
        )

        if pointcloud_is_precomputed:
            # Cache already contains fixed-size sampled Task-frame XYZ.
            pointcloud = pointcloud[:, :3]
        elif self.use_task_crop:
            pcd_result = transform_and_crop(
                pointcloud[:, :3],
                self.task_frame_cfg,
            )

            pointcloud = pcd_result[
                "cropped"
            ]

            if pointcloud.shape[0] == 0:
                raise RuntimeError(
                    "Task crop produced zero points: "
                    f"path={path}, "
                    f"timestep={timestep}"
                )
        else:
            pointcloud = pointcloud[:, :3]

        # ----------------------------------------------------
        # Current robot state
        # ----------------------------------------------------
        qpos = qpos_all[
            timestep
        ]

        # ----------------------------------------------------
        # ACT future action chunk
        # ----------------------------------------------------
        actions = action_all[
            timestep:
            min(
                timestep
                + self.chunk_size,
                len(action_all),
            )
        ]

        # ----------------------------------------------------
        # Optional legacy mean/std PointCloud normalization.
        # DP3-head training keeps this False because limits normalization
        # is applied after FPS inside DP3PointGlobalEncoder.
        # ----------------------------------------------------
        qpos = (
            qpos
            - self.stats[
                "qpos_mean"
            ]
        ) / self.stats[
            "qpos_std"
        ]

        if self.normalize_pointcloud:
            pointcloud = (
                pointcloud
                - self.stats[
                    "point_mean"
                ]
            ) / self.stats[
                "point_std"
            ]

        actions = (
            actions
            - self.stats[
                "action_mean"
            ]
        ) / self.stats[
            "action_std"
        ]

        # ----------------------------------------------------
        # Pad action chunk to fixed length
        # ----------------------------------------------------
        action_dim = (
            action_all.shape[-1]
        )

        action_chunk = np.zeros(
            (
                self.chunk_size,
                action_dim,
            ),
            dtype=np.float32,
        )

        is_pad = np.ones(
            self.chunk_size,
            dtype=bool,
        )

        valid_length = len(
            actions
        )

        action_chunk[
            :valid_length
        ] = actions

        is_pad[
            :valid_length
        ] = False

        # ----------------------------------------------------
        # Original ACT-style RGB preprocessing
        #
        # 240x320 remains 240x320.
        # RGB float32 [0, 1].
        # ----------------------------------------------------
        left_wrist = rgb_to_act_tensor(
            left_rgb
        )

        right_wrist = rgb_to_act_tensor(
            right_rgb
        )

        return {
            "qpos":
                torch.from_numpy(
                    qpos.astype(
                        np.float32
                    )
                ),

            "pointcloud":
                torch.from_numpy(
                    pointcloud.astype(
                        np.float32
                    )
                ),

            "left_wrist":
                left_wrist,

            "right_wrist":
                right_wrist,

            "actions":
                torch.from_numpy(
                    action_chunk
                ),

            "is_pad":
                torch.from_numpy(
                    is_pad
                ),
        }


def multimodal_collate_fn(
    batch,
):
    """
    PointCloud stays as a Python list because
    the number of points may vary between samples.

    RGB / qpos / action use normal tensor stacking.
    """
    qpos = torch.stack(
        [
            sample["qpos"]
            for sample in batch
        ],
        dim=0,
    )

    pointcloud = [
        sample["pointcloud"]
        for sample in batch
    ]

    left_wrist = torch.stack(
        [
            sample["left_wrist"]
            for sample in batch
        ],
        dim=0,
    )

    right_wrist = torch.stack(
        [
            sample["right_wrist"]
            for sample in batch
        ],
        dim=0,
    )

    actions = torch.stack(
        [
            sample["actions"]
            for sample in batch
        ],
        dim=0,
    )

    is_pad = torch.stack(
        [
            sample["is_pad"]
            for sample in batch
        ],
        dim=0,
    )

    return {
        "qpos": qpos,
        "pointcloud": pointcloud,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "actions": actions,
        "is_pad": is_pad,
    }
