from pathlib import Path
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


JOINT_KEYS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)


def concat_joint_group(root, group_name):
    arrays = []

    for key in JOINT_KEYS:
        arrays.append(
            np.asarray(
                root[f"{group_name}/{key}"],
                dtype=np.float32,
            )
        )

    return np.concatenate(
        arrays,
        axis=-1,
    )


def find_episode_files(dataset_dir):
    files = sorted(
        Path(dataset_dir).glob(
            "episode_*.hdf5"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No HDF5 episodes found in {dataset_dir}"
        )

    return files


def resolve_episode_files(
    dataset_dir,
    episode_files=None,
):
    if episode_files is None:
        return find_episode_files(
            dataset_dir
        )

    files = [
        Path(path)
        for path in episode_files
    ]

    if not files:
        raise ValueError(
            "episode_files is empty"
        )

    return files


def get_pointcloud_length(root):
    pointcloud = root[
        "vision/pointcloud"
    ]

    if isinstance(
        pointcloud,
        h5py.Dataset,
    ):
        return pointcloud.shape[0]

    if isinstance(
        pointcloud,
        h5py.Group,
    ):
        return len(pointcloud)

    raise TypeError(
        "Unsupported pointcloud storage type"
    )


def read_pointcloud_frame(
    root,
    timestep,
):
    pointcloud = root[
        "vision/pointcloud"
    ]

    if isinstance(
        pointcloud,
        h5py.Dataset,
    ):
        frame = np.asarray(
            pointcloud[
                timestep,
                :,
                :3,
            ],
            dtype=np.float32,
        )

    elif isinstance(
        pointcloud,
        h5py.Group,
    ):
        key = f"{timestep:06d}"

        if key not in pointcloud:
            keys = sorted(
                pointcloud.keys()
            )

            if timestep >= len(keys):
                raise IndexError(
                    f"Pointcloud timestep {timestep} "
                    f"out of range"
                )

            key = keys[timestep]

        frame = np.asarray(
            pointcloud[key],
            dtype=np.float32,
        )

        if (
            frame.ndim != 2
            or frame.shape[1] < 3
        ):
            raise ValueError(
                f"Unexpected pointcloud frame shape: "
                f"{frame.shape}"
            )

        frame = frame[:, :3]

    else:
        raise TypeError(
            "Unsupported pointcloud storage type"
        )

    if (
        frame.ndim != 2
        or frame.shape[1] != 3
    ):
        raise ValueError(
            f"Expected pointcloud [N, 3], "
            f"got {frame.shape}"
        )

    return frame



def compute_qpos_action_stats(
    dataset_dir,
    episode_files=None,
):
    """Compute only robot-state/action statistics.

    This is used by the multimodal DP3-head path, where PointCloud
    normalization is handled separately with train-set min/max limits.
    The existing compute_stats() function is intentionally left unchanged
    for baseline compatibility.
    """
    files = resolve_episode_files(
        dataset_dir,
        episode_files,
    )

    qpos_list = []
    action_list = []

    for path in files:
        with h5py.File(path, "r") as root:
            qpos_list.append(
                concat_joint_group(root, "state")
            )
            action_list.append(
                concat_joint_group(root, "action")
            )

    qpos = np.concatenate(qpos_list, axis=0)
    actions = np.concatenate(action_list, axis=0)

    return {
        "qpos_mean": qpos.mean(axis=0).astype(np.float32),
        "qpos_std": np.maximum(
            qpos.std(axis=0),
            1e-2,
        ).astype(np.float32),
        "action_mean": actions.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(
            actions.std(axis=0),
            1e-2,
        ).astype(np.float32),
    }

def compute_stats(
    dataset_dir,
    episode_files=None,
):
    files = resolve_episode_files(
        dataset_dir,
        episode_files,
    )

    qpos_list = []
    action_list = []

    point_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    point_sq_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    point_count = 0

    for path in files:
        with h5py.File(
            path,
            "r",
        ) as root:
            qpos = concat_joint_group(
                root,
                "state",
            )

            actions = concat_joint_group(
                root,
                "action",
            )

            pointcloud_length = (
                get_pointcloud_length(
                    root
                )
            )

            for timestep in range(
                pointcloud_length
            ):
                points = (
                    read_pointcloud_frame(
                        root,
                        timestep,
                    )
                )

                points64 = points.astype(
                    np.float64,
                    copy=False,
                )

                point_sum += points64.sum(
                    axis=0
                )

                point_sq_sum += (
                    points64
                    * points64
                ).sum(
                    axis=0
                )

                point_count += (
                    points64.shape[0]
                )

        qpos_list.append(
            qpos
        )

        action_list.append(
            actions
        )

    qpos = np.concatenate(
        qpos_list,
        axis=0,
    )

    actions = np.concatenate(
        action_list,
        axis=0,
    )

    if point_count == 0:
        raise ValueError(
            "No pointcloud points found"
        )

    point_mean = (
        point_sum
        / point_count
    )

    point_var = (
        point_sq_sum
        / point_count
        - point_mean ** 2
    )

    point_var = np.maximum(
        point_var,
        0.0,
    )

    point_std = np.sqrt(
        point_var
    )

    stats = {
        "qpos_mean":
            qpos.mean(
                axis=0
            ).astype(
                np.float32
            ),

        "qpos_std":
            np.maximum(
                qpos.std(
                    axis=0
                ),
                1e-2,
            ).astype(
                np.float32
            ),

        "action_mean":
            actions.mean(
                axis=0
            ).astype(
                np.float32
            ),

        "action_std":
            np.maximum(
                actions.std(
                    axis=0
                ),
                1e-2,
            ).astype(
                np.float32
            ),

        "point_mean":
            point_mean.astype(
                np.float32
            ),

        "point_std":
            np.maximum(
                point_std,
                1e-4,
            ).astype(
                np.float32
            ),
    }

    return stats


def save_stats(
    stats,
    path,
):
    with open(
        path,
        "wb",
    ) as file:
        pickle.dump(
            stats,
            file,
        )


class DapierDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        stats,
        chunk_size=20,
        episode_files=None,
        normalize_pointcloud=True,
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

        self.samples = []

        for path in self.files:
            with h5py.File(
                path,
                "r",
            ) as root:
                length = (
                    get_pointcloud_length(
                        root
                    )
                )

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

            pointcloud = (
                read_pointcloud_frame(
                    root,
                    timestep,
                )
            )

        qpos = qpos_all[
            timestep
        ]

        actions = action_all[
            timestep:
            min(
                timestep
                + self.chunk_size,
                len(action_all),
            )
        ]

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

            "actions":
                torch.from_numpy(
                    action_chunk
                ),

            "is_pad":
                torch.from_numpy(
                    is_pad
                ),
        }


def variable_pointcloud_collate_fn(
    batch,
):
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
        "actions": actions,
        "is_pad": is_pad,
    }
