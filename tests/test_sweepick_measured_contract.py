"""CPU-only measured-config contract. NO CuRobo/SAPIEN runtime test."""
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_measured_poses_and_cam():
    cfg = yaml.safe_load((ROOT/"assets/embodiments/so101-pgripper-sweepick-measured/config.yml").read_text())
    a, b = cfg["robot_pose"]
    assert np.allclose(a, [-.060,.150,.123,1,0,0,0])
    assert np.allclose(b, [-.060,-.150,.123,1,0,0,0])
    cam = cfg["static_camera_list"][0]
    assert np.allclose(cam["position"],[-.0330105670803,.0220671492958,.5129377312368])


def test_floor_and_no_extra_arm_separation():
    data = yaml.safe_load((ROOT/"env_cfg/task_config/demo_so101_pgripper_sweepick_measured_smoke.yml").read_text())
    assert data["embodiment"] == ["so101-pgripper-sweepick-measured"]*2+[0.]
    assert data["max_seed_attempts"] == 3
    assert data["episode_num"] == 1
    cfg = yaml.safe_load((ROOT/"assets/embodiments/so101-pgripper-sweepick-measured/curobo.yml").read_text())
    floor = cfg["planner"]["support_cuboid_base_frame"]
    assert abs(floor["pose"][2]+floor["dims"][2]/2 + .123)<1e-9


def test_measured_registry():
    cfg = yaml.safe_load((ROOT/"env_cfg/task_config/_embodiment_config.yml").read_text())
    assert cfg["so101-pgripper-sweepick-measured"]["file_path"]=="./assets/embodiments/so101-pgripper-sweepick-measured"
