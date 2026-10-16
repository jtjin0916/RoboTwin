import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import (
    compute_qpos_action_stats,
    find_episode_files,
    save_stats,
)

from dataset_multimodal import (
    DapierMultimodalDataset,
    precompute_dp3_pointclouds,
    multimodal_collate_fn,
)

from pointcloud_preprocess import TaskFrameConfig

from model import dapier_act_loss
from model_multimodal import DapierACTMultimodal


def move_batch_to_device(
    batch,
    device,
):
    qpos = batch["qpos"].to(
        device,
        non_blocking=True,
    )

    pointcloud = [
        cloud.to(
            device,
            non_blocking=True,
        )
        for cloud in batch["pointcloud"]
    ]

    left_wrist = batch["left_wrist"].to(
        device,
        non_blocking=True,
    )

    right_wrist = batch["right_wrist"].to(
        device,
        non_blocking=True,
    )

    actions = batch["actions"].to(
        device,
        non_blocking=True,
    )

    is_pad = batch["is_pad"].to(
        device,
        non_blocking=True,
    )

    return (
        qpos,
        pointcloud,
        left_wrist,
        right_wrist,
        actions,
        is_pad,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    kl_weight,
):
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_kl = 0.0
    step_count = 0

    for batch in loader:
        (
            qpos,
            pointcloud,
            left_wrist,
            right_wrist,
            actions,
            is_pad,
        ) = move_batch_to_device(
            batch,
            device,
        )

        prediction, mu, logvar = model(
            qpos=qpos,
            pointcloud=pointcloud,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            actions=actions,
            is_pad=is_pad,
        )

        losses = dapier_act_loss(
            prediction,
            actions,
            is_pad,
            mu,
            logvar,
            kl_weight=kl_weight,
        )

        total_loss += losses["loss"].item()
        total_l1 += losses["l1"].item()
        total_kl += losses["kl"].item()

        step_count += 1

    if step_count == 0:
        raise RuntimeError(
            "Validation loader contains zero batches"
        )

    return {
        "loss":
            total_loss
            / step_count,

        "l1":
            total_l1
            / step_count,

        "kl":
            total_kl
            / step_count,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "./checkpoints/"
            "dapier_act_multimodal_dp3head256_os30a_30"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--kl-weight",
        type=float,
        default=0.01,
    )

    # Current Spatial256 experiment used
    # 24 train / 6 validation episodes.
    parser.add_argument(
        "--val-episodes",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-point-samples",
        type=int,
        default=512,
        help=(
            "Number of metric-space FPS points fed to the "
            "DP3-inspired point encoder. Head ACT tokens stay fixed at 1."
        ),
    )

    parser.add_argument(
        "--candidate-points",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--no-pretrained-wrist-backbone",
        action="store_true",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    random.seed(
        args.split_seed
    )

    torch.manual_seed(
        args.split_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.split_seed
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
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

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"device: {device}"
    )

    # --------------------------------------------------------
    # Episode-level train / validation split
    # --------------------------------------------------------
    all_files = find_episode_files(
        args.dataset_dir
    )

    if (
        args.val_episodes <= 0
        or args.val_episodes >= len(all_files)
    ):
        raise ValueError(
            "val_episodes must be greater than 0 "
            "and smaller than total episodes"
        )

    shuffled_files = list(
        all_files
    )

    rng = random.Random(
        args.split_seed
    )

    rng.shuffle(
        shuffled_files
    )

    val_files = sorted(
        shuffled_files[
            :args.val_episodes
        ]
    )

    train_files = sorted(
        shuffled_files[
            args.val_episodes:
        ]
    )

    split_info = {
        "split_seed":
            args.split_seed,

        "train": [
            path.name
            for path in train_files
        ],

        "validation": [
            path.name
            for path in val_files
        ],
    }

    with open(
        output_dir / "episode_split.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_info,
            file,
            indent=2,
        )

    print(
        f"train episodes: {len(train_files)}"
    )

    print(
        f"val episodes:   {len(val_files)}"
    )

    # --------------------------------------------------------
    # Statistics
    #
    # qpos/action: standard mean/std normalization.
    # PointCloud: DP3-style train-only per-axis min/max limits.
    # The PointCloud limits are fit AFTER Task crop + FPS so the
    # statistics match the observations entering the point encoder.
    # --------------------------------------------------------
    task_frame_cfg = TaskFrameConfig()

    print(
        "computing qpos/action statistics..."
    )

    stats = compute_qpos_action_stats(
        args.dataset_dir,
        episode_files=train_files,
    )

    print(
        "precomputing train DP3 PointClouds "
        "(Task crop -> FPS) and fitting train min/max..."
    )

    train_pointcloud_cache, point_min, point_max = (
        precompute_dp3_pointclouds(
            args.dataset_dir,
            episode_files=train_files,
            task_frame_cfg=task_frame_cfg,
            candidate_points=args.candidate_points,
            num_point_samples=args.num_point_samples,
            device=str(device),
            fit_limits=True,
        )
    )

    print(
        "precomputing validation DP3 PointClouds "
        "(Task crop -> FPS; train limits are reused)..."
    )

    val_pointcloud_cache, _, _ = (
        precompute_dp3_pointclouds(
            args.dataset_dir,
            episode_files=val_files,
            task_frame_cfg=task_frame_cfg,
            candidate_points=args.candidate_points,
            num_point_samples=args.num_point_samples,
            device=str(device),
            fit_limits=False,
        )
    )

    stats["point_min"] = point_min
    stats["point_max"] = point_max
    stats["point_normalization"] = "dp3_limits_-1_1"

    save_stats(
        stats,
        output_dir / "dataset_stats.pkl",
    )

    print(
        f"point min: {point_min.tolist()}"
    )
    print(
        f"point max: {point_max.tolist()}"
    )

    # --------------------------------------------------------
    # Dataset
    #
    # PointCloud is returned as metric Task-frame cropped XYZ.
    # FPS + DP3 limits normalization happen inside the point encoder,
    # preserving metric geometry for FPS.
    # --------------------------------------------------------
    train_dataset = (
        DapierMultimodalDataset(
            args.dataset_dir,
            stats,
            chunk_size=args.chunk_size,
            episode_files=train_files,
            normalize_pointcloud=False,
            use_task_crop=True,
            task_frame_cfg=task_frame_cfg,
            pointcloud_cache=train_pointcloud_cache,
        )
    )

    val_dataset = (
        DapierMultimodalDataset(
            args.dataset_dir,
            stats,
            chunk_size=args.chunk_size,
            episode_files=val_files,
            normalize_pointcloud=False,
            use_task_crop=True,
            task_frame_cfg=task_frame_cfg,
            pointcloud_cache=val_pointcloud_cache,
        )
    )

    print(
        f"cached train PointCloud frames: {len(train_pointcloud_cache)}"
    )
    print(
        f"cached val PointCloud frames:   {len(val_pointcloud_cache)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        collate_fn=multimodal_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        collate_fn=multimodal_collate_fn,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    pretrained_wrist_backbone = (
        not args.no_pretrained_wrist_backbone
    )

    model = DapierACTMultimodal(
        qpos_dim=12,
        action_dim=12,
        chunk_size=args.chunk_size,
        hidden_dim=256,
        latent_dim=32,
        nheads=8,
        encoder_layers=4,
        decoder_layers=4,
        ff_dim=1024,
        dropout=0.0,
        num_point_samples=args.num_point_samples,
        candidate_points=args.candidate_points,
        point_min=stats["point_min"],
        point_max=stats["point_max"],
        pretrained_wrist_backbone=(
            pretrained_wrist_backbone
        ),
    ).to(
        device
    )

    # --------------------------------------------------------
    # Optimizer
    #
    # Same as Spatial256 baseline.
    # --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    left_parameter_count = sum(
        parameter.numel()
        for parameter
        in model.left_wrist_encoder.parameters()
        if parameter.requires_grad
    )

    right_parameter_count = sum(
        parameter.numel()
        for parameter
        in model.right_wrist_encoder.parameters()
        if parameter.requires_grad
    )

    print(
        f"train samples: {len(train_dataset)}"
    )

    print(
        f"val samples:   {len(val_dataset)}"
    )

    print(
        f"parameters:    {parameter_count:,}"
    )

    print(
        f"left ResNet:   {left_parameter_count:,}"
    )

    print(
        f"right ResNet:  {right_parameter_count:,}"
    )

    print(
        f"point samples: {args.num_point_samples}"
    )

    print(
        "head tokens:   1"
    )

    print(
        "wrist input:   native 240x320 RGB"
    )

    print(
        "wrist tokens:  80 + 80"
    )

    print(
        "visual tokens:  "
        f"{1 + 80 + 80}"
    )

    print(
        "total context tokens: "
        f"{1 + 80 + 80 + 2}"
    )

    print(
        "pretrained wrist backbones: "
        f"{pretrained_wrist_backbone}"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    best_val_loss = float(
        "inf"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        total_loss = 0.0
        total_l1 = 0.0
        total_kl = 0.0
        step_count = 0

        for batch in train_loader:
            (
                qpos,
                pointcloud,
                left_wrist,
                right_wrist,
                actions,
                is_pad,
            ) = move_batch_to_device(
                batch,
                device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction, mu, logvar = model(
                qpos=qpos,
                pointcloud=pointcloud,
                left_wrist=left_wrist,
                right_wrist=right_wrist,
                actions=actions,
                is_pad=is_pad,
            )

            losses = dapier_act_loss(
                prediction,
                actions,
                is_pad,
                mu,
                logvar,
                kl_weight=args.kl_weight,
            )

            losses["loss"].backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += (
                losses["loss"].item()
            )

            total_l1 += (
                losses["l1"].item()
            )

            total_kl += (
                losses["kl"].item()
            )

            step_count += 1

        if step_count == 0:
            raise RuntimeError(
                "Training loader contains zero batches"
            )

        train_metrics = {
            "loss":
                total_loss
                / step_count,

            "l1":
                total_l1
                / step_count,

            "kl":
                total_kl
                / step_count,
        }

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.kl_weight,
        )

        print(
            f"epoch {epoch:04d} | "
            f"train loss "
            f"{train_metrics['loss']:.6f} | "
            f"train l1 "
            f"{train_metrics['l1']:.6f} | "
            f"train kl "
            f"{train_metrics['kl']:.6f} | "
            f"val loss "
            f"{val_metrics['loss']:.6f} | "
            f"val l1 "
            f"{val_metrics['l1']:.6f} | "
            f"val kl "
            f"{val_metrics['kl']:.6f}"
        )

        checkpoint = {
            "model":
                model.state_dict(),

            "epoch":
                epoch,

            "model_type":
                "multimodal_dp3head256_resnet18",

            "qpos_dim":
                12,

            "action_dim":
                12,

            "chunk_size":
                args.chunk_size,

            "hidden_dim":
                256,

            "latent_dim":
                32,

            "num_point_samples":
                args.num_point_samples,

            "head_point_tokens":
                1,

            "candidate_points":
                args.candidate_points,

            "point_min":
                stats["point_min"].tolist(),

            "point_max":
                stats["point_max"].tolist(),

            "point_normalization":
                "dp3_limits_-1_1",

            "task_frame": {
                "origin_world": list(task_frame_cfg.origin_world),
                "rpy_deg": list(task_frame_cfg.rpy_deg),
                "crop_min": list(task_frame_cfg.crop_min),
                "crop_max": list(task_frame_cfg.crop_max),
            },

            "left_wrist_tokens":
                80,

            "right_wrist_tokens":
                80,

            "wrist_image_height":
                240,

            "wrist_image_width":
                320,

            "pretrained_wrist_backbone":
                pretrained_wrist_backbone,

            "model_config": {
                "qpos_dim": 12,
                "action_dim": 12,
                "chunk_size": args.chunk_size,
                "hidden_dim": 256,
                "latent_dim": 32,
                "num_point_samples": args.num_point_samples,
                "candidate_points": args.candidate_points,
                "point_min": stats["point_min"].tolist(),
                "point_max": stats["point_max"].tolist(),
                "nheads": 8,
                "encoder_layers": 4,
                "decoder_layers": 4,
                "ff_dim": 1024,
                "dropout": 0.0,
                "pretrained_wrist_backbone": False,
            },

            "train_loss":
                train_metrics["loss"],

            "train_l1":
                train_metrics["l1"],

            "train_kl":
                train_metrics["kl"],

            "val_loss":
                val_metrics["loss"],

            "val_l1":
                val_metrics["l1"],

            "val_kl":
                val_metrics["kl"],

            "split_seed":
                args.split_seed,

            "val_episodes":
                args.val_episodes,
        }

        torch.save(
            checkpoint,
            output_dir / "policy_last.pt",
        )

        if (
            val_metrics["loss"]
            < best_val_loss
        ):
            best_val_loss = (
                val_metrics["loss"]
            )

            torch.save(
                checkpoint,
                output_dir / "policy_best.pt",
            )

            print(
                f"saved best checkpoint: "
                f"val_loss={best_val_loss:.6f}"
            )


if __name__ == "__main__":
    main()
