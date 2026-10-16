import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import (
    DapierDataset,
    compute_stats,
    find_episode_files,
    save_stats,
)

from model import (
    DapierACT,
    dapier_act_loss,
)


def move_batch_to_device(
    batch,
    device,
):
    qpos = batch[
        "qpos"
    ].to(
        device,
        non_blocking=True,
    )

    pointcloud = batch[
        "pointcloud"
    ].to(
        device,
        non_blocking=True,
    )

    actions = batch[
        "actions"
    ].to(
        device,
        non_blocking=True,
    )

    is_pad = batch[
        "is_pad"
    ].to(
        device,
        non_blocking=True,
    )

    return (
        qpos,
        pointcloud,
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
            actions,
            is_pad,
        ) = move_batch_to_device(
            batch,
            device,
        )

        prediction, mu, logvar = model(
            qpos,
            pointcloud,
            actions,
            is_pad,
        )

        losses = dapier_act_loss(
            prediction,
            actions,
            is_pad,
            mu,
            logvar,
            kl_weight=kl_weight,
        )

        total_loss += (
            losses[
                "loss"
            ].item()
        )

        total_l1 += (
            losses[
                "l1"
            ].item()
        )

        total_kl += (
            losses[
                "kl"
            ].item()
        )

        step_count += 1

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
        default="./checkpoints/dapier_act_v0",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
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

    parser.add_argument(
        "--num-point-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--val-episodes",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    if (
        args.device.startswith(
            "cuda"
        )
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

    all_files = find_episode_files(
        args.dataset_dir
    )

    if (
        args.val_episodes <= 0
        or args.val_episodes
        >= len(all_files)
    ):
        raise ValueError(
            "val_episodes must be greater than 0 "
            "and smaller than the total number of episodes"
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

        "train":
            [
                path.name
                for path
                in train_files
            ],

        "validation":
            [
                path.name
                for path
                in val_files
            ],
    }

    with open(
        output_dir
        / "episode_split.json",
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

    print(
        "train files:"
    )

    for path in train_files:
        print(
            f"  {path.name}"
        )

    print(
        "val files:"
    )

    for path in val_files:
        print(
            f"  {path.name}"
        )

    stats = compute_stats(
        args.dataset_dir,
        episode_files=train_files,
    )

    save_stats(
        stats,
        output_dir
        / "dataset_stats.pkl",
    )

    train_dataset = DapierDataset(
        args.dataset_dir,
        stats,
        chunk_size=args.chunk_size,
        episode_files=train_files,
    )

    val_dataset = DapierDataset(
        args.dataset_dir,
        stats,
        chunk_size=args.chunk_size,
        episode_files=val_files,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    model = DapierACT(
        qpos_dim=12,
        action_dim=12,
        chunk_size=args.chunk_size,
        hidden_dim=256,
        latent_dim=32,
        num_point_tokens=args.num_point_tokens,
        nheads=8,
        encoder_layers=4,
        decoder_layers=4,
        ff_dim=1024,
        dropout=0.0,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"train samples: {len(train_dataset)}"
    )

    print(
        f"val samples:   {len(val_dataset)}"
    )

    print(
        f"parameters: {parameter_count:,}"
    )

    batch = next(
        iter(
            train_loader
        )
    )

    (
        qpos,
        pointcloud,
        actions,
        is_pad,
    ) = move_batch_to_device(
        batch,
        device,
    )

    with torch.no_grad():
        prediction, _, _ = model(
            qpos,
            pointcloud,
            actions,
            is_pad,
        )

    print(
        f"qpos:       {tuple(qpos.shape)}"
    )

    print(
        f"pointcloud: {tuple(pointcloud.shape)}"
    )

    print(
        f"actions:    {tuple(actions.shape)}"
    )

    print(
        f"prediction: {tuple(prediction.shape)}"
    )

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
                qpos,
                pointcloud,
                actions,
                is_pad,
            )

            losses = dapier_act_loss(
                prediction,
                actions,
                is_pad,
                mu,
                logvar,
                kl_weight=args.kl_weight,
            )

            losses[
                "loss"
            ].backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += (
                losses[
                    "loss"
                ].item()
            )

            total_l1 += (
                losses[
                    "l1"
                ].item()
            )

            total_kl += (
                losses[
                    "kl"
                ].item()
            )

            step_count += 1

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
            f"train loss {train_metrics['loss']:.6f} | "
            f"train l1 {train_metrics['l1']:.6f} | "
            f"train kl {train_metrics['kl']:.6f} | "
            f"val loss {val_metrics['loss']:.6f} | "
            f"val l1 {val_metrics['l1']:.6f} | "
            f"val kl {val_metrics['kl']:.6f}"
        )

        checkpoint = {
            "model":
                model.state_dict(),

            "epoch":
                epoch,

            "chunk_size":
                args.chunk_size,

            "num_point_tokens":
                args.num_point_tokens,

            "train_loss":
                train_metrics[
                    "loss"
                ],

            "val_loss":
                val_metrics[
                    "loss"
                ],

            "split_seed":
                args.split_seed,
        }

        torch.save(
            checkpoint,
            output_dir
            / "policy_last.pt",
        )

        if (
            val_metrics[
                "loss"
            ]
            < best_val_loss
        ):
            best_val_loss = (
                val_metrics[
                    "loss"
                ]
            )

            torch.save(
                checkpoint,
                output_dir
                / "policy_best.pt",
            )

            print(
                f"saved best checkpoint: "
                f"val_loss={best_val_loss:.6f}"
            )


if __name__ == "__main__":
    main()
