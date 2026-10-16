import argparse
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DapierDataset
from model import DapierACT


ACTION_NAMES = [
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_gripper",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_gripper",
]


@torch.no_grad()
def evaluate_dataset(
    model,
    dataset,
    device,
    action_mean,
    action_std,
    batch_size,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    total_abs_error = 0.0
    total_valid_values = 0

    dim_abs_error = torch.zeros(
        12,
        device=device,
    )

    dim_valid_count = 0

    for batch in loader:
        qpos = batch["qpos"].to(device)
        pointcloud = batch["pointcloud"].to(device)
        target = batch["actions"].to(device)
        is_pad = batch["is_pad"].to(device)

        prediction, _, _ = model(
            qpos,
            pointcloud,
            actions=None,
            is_pad=None,
        )

        prediction_real = (
            prediction * action_std
            + action_mean
        )

        target_real = (
            target * action_std
            + action_mean
        )

        valid = (
            ~is_pad
        ).unsqueeze(-1)

        error = torch.abs(
            prediction_real
            - target_real
        )

        masked_error = (
            error
            * valid
        )

        valid_steps = (
            valid.sum().item()
        )

        total_abs_error += (
            masked_error.sum().item()
        )

        total_valid_values += (
            valid_steps
            * prediction.shape[-1]
        )

        dim_abs_error += (
            masked_error.sum(
                dim=(0, 1)
            )
        )

        dim_valid_count += (
            valid_steps
        )

    overall_mae = (
        total_abs_error
        / total_valid_values
    )

    per_dim_mae = (
        dim_abs_error
        / dim_valid_count
    ).cpu()

    return (
        overall_mae,
        per_dim_mae,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
    )

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        default="policy_best.pt",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    if (
        args.device.startswith("cuda")
        and torch.cuda.is_available()
    ):
        device = torch.device(
            args.device
        )
    else:
        device = torch.device(
            "cpu"
        )

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    with open(
        checkpoint_dir
        / "dataset_stats.pkl",
        "rb",
    ) as file:
        stats = pickle.load(file)

    with open(
        checkpoint_dir
        / "episode_split.json",
        "r",
        encoding="utf-8",
    ) as file:
        split_info = json.load(file)

    checkpoint = torch.load(
        checkpoint_dir
        / args.checkpoint,
        map_location=device,
    )

    chunk_size = checkpoint[
        "chunk_size"
    ]

    val_files = [
        Path(args.dataset_dir)
        / name
        for name
        in split_info[
            "validation"
        ]
    ]

    print(
        f"device: {device}"
    )

    print(
        f"checkpoint: {args.checkpoint}"
    )

    print(
        f"checkpoint epoch: {checkpoint['epoch']}"
    )

    print(
        f"validation episodes: {len(val_files)}"
    )

    for path in val_files:
        print(
            f"  {path.name}"
        )

    model = DapierACT(
        qpos_dim=12,
        action_dim=12,
        chunk_size=chunk_size,
        hidden_dim=256,
        latent_dim=32,
        num_point_tokens=checkpoint[
            "num_point_tokens"
        ],
        nheads=8,
        encoder_layers=4,
        decoder_layers=4,
        ff_dim=1024,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    model.eval()

    action_mean = torch.tensor(
        stats[
            "action_mean"
        ],
        dtype=torch.float32,
        device=device,
    )

    action_std = torch.tensor(
        stats[
            "action_std"
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        "\nPer-episode z=0 inference:"
    )

    episode_maes = []

    for path in val_files:
        dataset = DapierDataset(
            args.dataset_dir,
            stats,
            chunk_size=chunk_size,
            episode_files=[
                path
            ],
        )

        mae, _ = evaluate_dataset(
            model,
            dataset,
            device,
            action_mean,
            action_std,
            args.batch_size,
        )

        episode_maes.append(
            mae
        )

        print(
            f"{path.name}: "
            f"samples={len(dataset):3d} "
            f"MAE={mae:.6f}"
        )

    val_dataset = DapierDataset(
        args.dataset_dir,
        stats,
        chunk_size=chunk_size,
        episode_files=val_files,
    )

    overall_mae, per_dim_mae = (
        evaluate_dataset(
            model,
            val_dataset,
            device,
            action_mean,
            action_std,
            args.batch_size,
        )
    )

    print(
        "\nOverall held-out z=0 inference"
    )

    print(
        f"samples: {len(val_dataset)}"
    )

    print(
        f"MAE: {overall_mae:.6f}"
    )

    print(
        "\nPer-action-dimension MAE:"
    )

    for name, value in zip(
        ACTION_NAMES,
        per_dim_mae.tolist(),
    ):
        print(
            f"{name:15s}: "
            f"{value:.6f}"
        )


if __name__ == "__main__":
    main()
