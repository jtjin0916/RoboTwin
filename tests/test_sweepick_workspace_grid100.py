"""CPU-only Grid100 geometry, full-attempt CSV and resume contracts."""
import csv
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from sweepick_workspace_grid100 import (
    X_CENTERS, Y_CENTERS, BLOCK_HALF_SIZE, CROP_MIN, CROP_MAX,
    NUM_CELLS, cell, cells,
)
from sweepick_table_frame import base_to_world, world_to_base
from scripts import scan_sweepick_workspace_grid100 as scan

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "demo_so101_pgripper_sweepick_workspace_grid100"


def test_all_100_cells_unique_symmetric_and_y_fastest():
    assert NUM_CELLS == 100
    assert len(X_CENTERS) == len(Y_CENTERS) == 10
    assert np.allclose(Y_CENTERS, np.linspace(-.15, .15, 10))
    assert np.allclose(Y_CENTERS, -Y_CENTERS[::-1])
    assert not np.any(np.isclose(Y_CENTERS, 0))  # Even number of Y samples.
    assert np.allclose(X_CENTERS, np.linspace(.18, .29, 10))
    positions = list(cells())
    assert [c["grid_id"] for c in positions] == list(range(100))
    assert len({(c["x_base_m"], c["y_base_m"]) for c in positions}) == 100
    for j, c in enumerate(positions):
        assert cell(j) == (X_CENTERS[j // 10], Y_CENTERS[j % 10])
        assert c["z_base_m"] == .021
    assert np.allclose(cell(0), [.18, -.15])
    assert np.allclose(cell(9), [.18, .15])
    assert np.allclose(cell(10), [X_CENTERS[1], -.15])
    assert np.allclose(cell(99), [.29, .15])
    for j in (-1, 100):
        with pytest.raises(ValueError, match="0..99"):
            cell(j)


def test_cube_and_table_frame_unchanged():
    for c in cells():
        x, y = c["x_base_m"], c["y_base_m"]
        assert x - BLOCK_HALF_SIZE >= CROP_MIN[0] + .01 - 1e-9
        assert x + BLOCK_HALF_SIZE <= CROP_MAX[0] - .01 + 1e-9
        assert y - BLOCK_HALF_SIZE >= CROP_MIN[1]
        assert y + BLOCK_HALF_SIZE <= CROP_MAX[1]
        p = [x, y, c["z_base_m"]]
        assert np.allclose(world_to_base(base_to_world(p)), p)
        assert np.allclose(base_to_world(p), [-y, x - .2539, .521])


def test_config_changes_only_output_path():
    folder = ROOT / "env_cfg/task_config"
    old = yaml.safe_load((folder / CONFIG.replace("100", "50")).with_suffix(".yml").read_text())
    new = yaml.safe_load((folder / CONFIG).with_suffix(".yml").read_text())
    assert new.pop("save_path") == "./data_sweepick_workspace_grid100"
    assert old.pop("save_path") == "./data_sweepick_workspace_grid50"
    assert new == old


def options(tmp_path, **overrides):
    values = dict(out=tmp_path / "grid100/attempts.csv", config=CONFIG,
                  start=0, limit=None, arms=["left", "right"], repeats=1)
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeTask:
    """Setup failure, planning exception, failed lift and successful lift."""
    created = []

    def __init__(self):
        self.closed = False
        self.created.append(self)

    def setup_demo(self, **kwargs):
        self.args = kwargs
        self.BLOCK_ROBOT_XY = cell(kwargs["grid_index"])
        self.scan_metrics = {"failure_stage": "not_started", "lift_success": False}
        if kwargs["grid_index"] == 0 and kwargs["scan_arm"] == "left":
            raise RuntimeError("setup failed")

    def get_obs(self):
        x, y = self.BLOCK_ROBOT_XY
        return {"pointcloud": np.array([[x, y, .021], [x + 1, y, .021]])}

    def play_once(self):
        j = self.args["grid_index"]
        self.scan_metrics.update(pregrasp_pose_found=True, pregrasp_plan_ok=True,
                                 failure_stage="grasp_plan")
        if j == 1 and self.args["scan_arm"] == "left":
            raise RuntimeError("planner failed")
        success = self.args["scan_arm"] == "right"
        self.scan_metrics.update(lift_success=success,
                                 failure_stage="" if success else "lift_under_30mm")

    def close_env(self, **kwargs):
        self.closed = True


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_full_two_arm_coverage_failures_and_repeats_resume(tmp_path):
    opts = options(tmp_path)
    FakeTask.created = []
    scan.run_scan(opts, {}, FakeTask)
    rows = read_rows(opts.out)
    assert len(rows) == len(FakeTask.created) == 200
    assert all(t.closed for t in FakeTask.created)
    assert {(int(r["grid_id"]), r["arm"]) for r in rows} == {
        (j, a) for j in range(100) for a in ("left", "right")}
    assert all(int(r["seed"]) == int(r["grid_id"]) for r in rows)
    assert rows[0]["setup_ok"] == "False" and rows[0]["failure_stage"] == "setup"
    assert rows[1]["lift_success"] == "True"
    assert rows[2]["failure_stage"] == "grasp_plan"
    assert rows[2]["pregrasp_plan_ok"] == "True"
    assert "planner failed" in rows[2]["error"]
    assert rows[3]["near_cube_points_proxy"] == "1"
    before = opts.out.read_bytes()
    scan.run_scan(opts, {}, FakeTask)
    assert opts.out.read_bytes() == before
    assert len(FakeTask.created) == 200  # Failed attempts also skipped.
    opts.repeats = 3
    scan.run_scan(opts, {}, FakeTask)
    rows = read_rows(opts.out)
    assert len(rows) == 600
    assert len(scan.existing_rows(opts.out, CONFIG)) == 600
    assert all(int(r["seed"]) == int(r["grid_id"]) + 100 * int(r["repeat"]) for r in rows)
    assert {int(r["seed"]) for r in rows} == set(range(300))


def test_partial_resume_last_cell_and_duplicate_arm(tmp_path):
    opts = options(tmp_path, start=99, limit=5, arms=["left", "left"])
    scan.run_scan(opts, {}, FakeTask)
    opts.arms = ["left", "right"]
    scan.run_scan(opts, {}, FakeTask)
    rows = read_rows(opts.out)
    assert len(rows) == 2
    assert all(r["grid_id"] == "99" for r in rows)
    assert {r["arm"] for r in rows} == {"left", "right"}


def test_interruption_retries_only_unrecorded_attempt(tmp_path):
    opts = options(tmp_path, limit=2)

    class Interrupted(FakeTask):
        def play_once(self):
            if self.args["grid_index"] == 1:
                raise KeyboardInterrupt()
            super().play_once()

    FakeTask.created = []
    with pytest.raises(KeyboardInterrupt):
        scan.run_scan(opts, {}, Interrupted)
    assert len(read_rows(opts.out)) == 2
    assert all(t.closed for t in FakeTask.created)
    scan.run_scan(opts, {}, FakeTask)
    assert len(read_rows(opts.out)) == 4


@pytest.mark.parametrize("field,value", [
    ("grid_size", "50"), ("grid_id", "100"), ("seed", "100"),
    ("repeat", "-1"), ("x_base_m", ".29"), ("y_base_m", "-.075"),
    ("config", "other"), ("arm", "both"), ("lift_success", ""),
])
def test_resume_rejects_incompatible_rows_without_writing(tmp_path, field, value):
    opts = options(tmp_path, limit=1, arms=["right"])
    scan.run_scan(opts, {}, FakeTask)
    row = read_rows(opts.out)[0]
    row[field] = value
    with opts.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scan.FIELDS)
        writer.writeheader()
        writer.writerow(row)
    before = opts.out.read_bytes()
    with pytest.raises(ValueError):
        scan.run_scan(opts, {}, FakeTask)
    assert opts.out.read_bytes() == before


def test_resume_rejects_legacy_and_truncated_csv(tmp_path):
    opts = options(tmp_path)
    opts.out.parent.mkdir()
    for text in ("grid_id,arm,repeat\n0,left,0\n",
                 ",".join(scan.FIELDS) + "\n100," + CONFIG + ",0\n"):
        opts.out.write_text(text)
        with pytest.raises(ValueError):
            scan.existing_rows(opts.out, CONFIG)
        assert opts.out.read_text() == text


def test_list_only_and_invalid_cli_work_without_simulator():
    script = str(ROOT / "scripts/scan_sweepick_workspace_grid100.py")
    result = subprocess.run([sys.executable, script, "--list-only"], capture_output=True, text=True)
    assert result.returncode == 0
    assert len(result.stdout.splitlines()) == 100
    assert result.stdout.startswith("00: x=0.18000, y=-0.15000")
    assert "99: x=0.29000, y=+0.15000" in result.stdout
    for args in (["--start", "100"], ["--start", "-1"],
                 ["--limit", "0"], ["--repeats", "0"]):
        result = subprocess.run([sys.executable, script, *args], capture_output=True, text=True)
        assert result.returncode == 2
        assert "Traceback" not in result.stderr


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("fail_stage", [None, "select_grasp_pose", "pregrasp_plan",
                                        "grasp_plan", "close_gripper", "lift_plan",
                                        "lift_under_30mm"])
def test_real_task_both_arms_and_failure_stages(monkeypatch, arm, fail_stage):
    """Exercise the real task control flow with a simulated planner/object."""
    import importlib.util
    import types

    class Parent:
        def setup_demo(self, **kwargs):
            self.info = {}
            self.plan_success = True

    # Load the real task file without importing SAPIEN or envs/__init__.py.
    package = types.ModuleType("envs")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "envs", package)
    monkeypatch.setitem(sys.modules, "envs.sweepick_block40_measured",
                        SimpleNamespace(sweepick_block40_measured=Parent))
    monkeypatch.setitem(sys.modules, "envs.utils", SimpleNamespace(ArmTag=lambda a: a))
    spec = importlib.util.spec_from_file_location(
        "envs.sweepick_block40_grid100_cpu_test", ROOT / "envs/sweepick_block40_grid100.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task = module.sweepick_block40_grid100()
    task.setup_demo(grid_index=99, scan_arm=arm)
    assert np.allclose(task.BLOCK_ROBOT_XY, [.29, .15])
    height = [.521]
    task.object = SimpleNamespace(get_pose=lambda: SimpleNamespace(p=[0, 0, height[0]]))
    calls = []

    def grasp_pose(obj, **kwargs):
        assert kwargs["arm_tag"] == arm
        return (None, None) if fail_stage == "select_grasp_pose" else ("pre", "grasp")

    def move(action):
        calls.append(action)
        task.plan_success = task.scan_metrics["failure_stage"] != fail_stage
        if action == "lift" and task.plan_success:
            height[0] += .01 if fail_stage == "lift_under_30mm" else .05

    def action(name, **kwargs):
        assert kwargs["arm_tag"] == arm
        return name

    task.choose_grasp_pose = grasp_pose
    task.move_to_pose = lambda **kw: action(kw["target_pose"], **kw)
    task.close_gripper = lambda **kw: action("close", **kw)
    task.move_by_displacement = lambda **kw: action("lift", **kw)
    task.move = move
    result = task.play_once()["metrics"]
    assert result["grid_id"] == 99 and result["arm"] == arm
    assert result["lift_success"] is (fail_stage is None)
    assert result["failure_stage"] == (fail_stage or "")
    if fail_stage is None:
        assert calls == ["pre", "grasp", "close", "lift"]
        assert result["lift_delta_z_m"] == pytest.approx(.05)
