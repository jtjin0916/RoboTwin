#!/usr/bin/env python3
"""Full-coverage Grid100 diagnostic of measured Sweepick arms.

Each (cell, arm, repeat) gets a CSV record even on failure. No seed search,
success-only replacement, demonstrations, or cached HDF5 reuse occurs.

Use --list-only to see the candidate positions without loading RoboTwin.
"""
import argparse
import csv
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sweepick_workspace_grid100 import cells, cell, NUM_CELLS

FIELDS = (
    "grid_size", "config", "grid_id", "x_base_m", "y_base_m", "arm", "repeat", "seed",
    "setup_ok", "near_cube_points_proxy",
    "pregrasp_pose_found", "pregrasp_plan_ok", "grasp_plan_ok",
    "gripper_close_ok", "lift_plan_ok", "lift_delta_z_m",
    "lift_success", "failure_stage", "error",
)


def prepare_args(config_name):
    """Match scripts/collect_data.py's embodiment configuration semantics."""
    import yaml
    from scripts.collect_data import get_embodiment_config
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    config_path = Path(CONFIGS_PATH) / f"{config_name}.yml"
    with config_path.open(encoding="utf-8") as f:
        args = yaml.safe_load(f)
    with (Path(CONFIGS_PATH) / "_embodiment_config.yml").open(encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    left, right, distance = args["embodiment"]
    args.update(
        left_robot_file=registry[left]["file_path"],
        right_robot_file=registry[right]["file_path"],
        left_embodiment_config=get_embodiment_config(registry[left]["file_path"]),
        right_embodiment_config=get_embodiment_config(registry[right]["file_path"]),
        embodiment_dis=float(distance),
        dual_arm_embodied=False,
        embodiment_name=f"{left}+{right}",
        task_name="sweepick_block40_grid100",
        task_config=config_name,
        need_plan=True,
        render_freq=0,
        save_data=False,
        collect_data=False,
        save_path="./data_sweepick_grid100_diagnostic_unused",
    )
    return args


def count_near_cube_points(task):
    """Coarse PCD spatial occupancy proxy, not semantic object visibility.

    Head camera points are converted World->base_footprint by measured task.
    Count near ground-level 40mm block in base-local XY (not pixel RGB).
    Surface/robot points may cause false positives; keep visual inspection.
    """
    import numpy as np
    obs = task.get_obs()
    raw = obs.get("pointcloud")
    if raw is None:
        return None
    pts = np.asarray(raw)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return None
    xyz = pts[:, :3]
    x, y = task.BLOCK_ROBOT_XY
    valid = (
        np.isfinite(xyz).all(axis=1)
        & (np.abs(xyz[:, 0] - x) <= .023)
        & (np.abs(xyz[:, 1] - y) <= .023)
        & (xyz[:, 2] > .005)
        & (xyz[:, 2] < .045)
    )
    return int(valid.sum())


def existing_rows(path, config_name):
    """Only resume complete records from this grid and config.

    In particular a Grid50 ID is NOT a Grid100 ID: Y varies fastest.
    Reject legacy/mismatched/truncated CSVs before opening for append.
    Failed attempts are completed keys too, and are never retried implicitly.
    """
    import math

    completed = set()
    if not path.exists() or path.stat().st_size == 0:
        return completed
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Output CSV schema mismatch; use a new --out path")
        for line, item in enumerate(reader, start=2):
            try:
                if None in item or any(v is None for v in item.values()):
                    raise ValueError("incomplete CSV record")
                j, repeat = int(item["grid_id"]), int(item["repeat"])
                x, y = cell(j)
                key = (j, item["arm"], repeat)
                if (int(item["grid_size"]) != NUM_CELLS
                        or item["config"] != config_name
                        or item["arm"] not in ("left", "right")
                        or repeat < 0
                        or int(item["seed"]) != j + NUM_CELLS * repeat
                        or not math.isclose(float(item["x_base_m"]), x, abs_tol=1e-9)
                        or not math.isclose(float(item["y_base_m"]), y, abs_tol=1e-9)
                        or item["setup_ok"] not in ("True", "False")
                        or item["lift_success"] not in ("True", "False")
                        or key in completed):
                    raise ValueError("grid/config/seed mismatch or duplicate key")
                completed.add(key)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid Grid100 CSV at line {line}: {e}; use a new --out path"
                ) from e
    return completed


def run_scan(opts, cfg, task_factory):
    """Run diagnostics with an injectable task for CPU coverage tests."""
    path = opts.out
    completed = existing_rows(path, opts.config)
    path.parent.mkdir(parents=True, exist_ok=True)
    end = min(NUM_CELLS, opts.start + (opts.limit or NUM_CELLS))
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            writer.writeheader()
            f.flush()
        for j in range(opts.start, end):
            x, y = cell(j)
            for arm in dict.fromkeys(opts.arms):
                for repeat in range(opts.repeats):
                    key = (j, arm, repeat)
                    if key in completed:
                        print(f"[GRID100] skip already recorded {key}")
                        continue
                    seed = j + NUM_CELLS * repeat
                    row = {
                        "grid_size": NUM_CELLS, "config": opts.config,
                        "grid_id": j, "x_base_m": x, "y_base_m": y,
                        "arm": arm, "repeat": repeat, "seed": seed,
                        "setup_ok": False, "lift_success": False,
                        "failure_stage": "setup", "error": "",
                    }
                    task = None
                    was_initialized = False
                    try:
                        task = task_factory()
                        task.setup_demo(
                            now_ep_num=j, seed=seed,
                            grid_index=j, scan_arm=arm, **cfg
                        )
                        was_initialized = True
                        row["setup_ok"] = True
                        try:
                            row["near_cube_points_proxy"] = count_near_cube_points(task)
                        except Exception as e:
                            row["error"] = f"visibility_probe: {type(e).__name__}: {e}"
                        task.play_once()
                        row.update(task.scan_metrics)
                    except Exception as e:
                        # Preserve the stage and successful steps before an exception.
                        if task is not None:
                            row.update(getattr(task, "scan_metrics", {}))
                        row["error"] = (
                            (str(row["error"]) + " | ") if row["error"] else ""
                        ) + f"{type(e).__name__}: {e}"
                        row["lift_success"] = False
                        if not was_initialized:
                            row["failure_stage"] = "setup"
                        print(f"[GRID100] ERROR cell={j} arm={arm} seed={seed}")
                        traceback.print_exc()
                    finally:
                        if task is not None:
                            try:
                                task.close_env(clear_cache=True)
                            except Exception as e:
                                print("[GRID100] close warning:", repr(e))
                    writer.writerow({k: row.get(k) for k in FIELDS})
                    f.flush()
                    completed.add(key)
                    print(f"[GRID100] saved {key} lift={row['lift_success']} "
                          f"stage={row['failure_stage']} -> {path}")
    print(f"[GRID100] diagnostic CSV: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--arms", nargs="+", default=["left", "right"],
                        choices=("left", "right"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--config",
        default="demo_so101_pgripper_sweepick_workspace_grid100"
    )
    parser.add_argument("--out", type=Path, default=Path("data_sweepick_workspace_grid100/attempts.csv"))
    opts = parser.parse_args()

    if opts.list_only:
        for c in cells():
            print(f"{c['grid_id']:02d}: x={c['x_base_m']:.5f}, "
                  f"y={c['y_base_m']:+.5f}, z={c['z_base_m']:.3f}")
        return
    if not 0 <= opts.start < NUM_CELLS:
        parser.error(f"--start must be 0..{NUM_CELLS - 1}")
    if opts.limit is not None and opts.limit <= 0:
        parser.error("--limit must be positive")
    if opts.repeats <= 0:
        parser.error("--repeats must be positive")

    # Validate resume data before loading SAPIEN/CuRobo or changing the CSV.
    try:
        existing_rows(opts.out, opts.config)
    except ValueError as e:
        parser.error(str(e))
    from envs.sweepick_block40_grid100 import sweepick_block40_grid100
    cfg = prepare_args(opts.config)
    # Fixed geometry/camera and no DR by default, as in Grid50.
    run_scan(opts, cfg, sweepick_block40_grid100)


if __name__ == "__main__":
    main()
