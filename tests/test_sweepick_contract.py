"""CPU-only checks. SAPIEN/PhysX and CuRobo execution are NOT tested here."""
import numpy as np
from dapier_act.sweepick_contract import SWEEPICK_TASK, normalized_sweepick_pointcloud
from dapier_act.pointcloud_preprocess import world_to_task


def test_handoff_mount_world_mapping():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    origin = np.array([0.0, -0.2539, 0.0])
    arm = np.array([[-0.0961, 0.15, 0.14], [-0.0961, -0.15, 0.14]])
    actual = (R @ arm.T).T + origin
    assert np.allclose(actual, [[-0.15, -0.35, 0.14], [0.15, -0.35, 0.14]])
    assert np.allclose(world_to_task(actual.astype(np.float32), SWEEPICK_TASK), arm, atol=1e-6)


def test_handoff_block_and_crop_normalization():
    # 40mm block centre: robot XYZ=(.18,.14,.02), world=(-.14,-.0739,.02).
    p = np.array([[-0.14, -0.0739, 0.021]], dtype=np.float32)
    task = world_to_task(p, SWEEPICK_TASK)
    assert np.allclose(task, [[0.18, 0.14, 0.021]], atol=1e-6)
    points = np.repeat(p, 512, axis=0)
    z = normalized_sweepick_pointcloud(points, sample_device="cpu")
    assert z.shape == (512, 3)
    expected = 2*(task[0]-SWEEPICK_TASK.crop_min)/(np.array(SWEEPICK_TASK.crop_max)-np.array(SWEEPICK_TASK.crop_min))-1
    assert np.allclose(z[0], expected, atol=1e-5)


def test_sweepick_embodiment_and_floor_metadata():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]
    embodiment_registry = yaml.safe_load(
        (root / "env_cfg/task_config/_embodiment_config.yml").read_text()
    )
    assert embodiment_registry["so101-pgripper-sweepick"]["file_path"] == (
        "./assets/embodiments/so101-pgripper-sweepick"
    )
    yml = yaml.safe_load(
        (root / "assets/embodiments/so101-pgripper-sweepick/curobo.yml").read_text()
    )
    floor = yml["planner"]["support_cuboid_base_frame"]
    assert np.isclose(floor["pose"][2] + floor["dims"][2] / 2, -0.14)
    assert np.isclose(yml["planner"]["support_top_world_z"], 0.0)
    cfg = yaml.safe_load(
        (root / "env_cfg/task_config/demo_so101_pgripper_sweepick_block40_smoke.yml").read_text()
    )
    assert cfg["max_seed_attempts"] == 3
