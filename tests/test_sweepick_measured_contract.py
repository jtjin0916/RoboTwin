"""CPU-only contracts for Sweepick virtual-floor tabletop smoke (NO GPU sim)."""
from pathlib import Path
import numpy as np
import yaml
from sweepick_table_frame import base_to_world, world_to_base

ROOT=Path(__file__).resolve().parents[1]


def _yaml(path):
    return yaml.safe_load((ROOT/path).read_text())


def test_world_yaw_and_measured_so101_positions():
    a,b=_yaml("assets/embodiments/so101-pgripper-sweepick-measured/config.yml")["robot_pose"]
    assert np.allclose(a[:3],[-.15,-.3139,.623])
    assert np.allclose(b[:3],[.15,-.3139,.623])
    assert all(np.allclose(p[3:],[2**-.5,0,0,2**-.5]) for p in (a,b))
    assert np.allclose(base_to_world([-.06,.15,.123]),a[:3])
    assert np.allclose(base_to_world([-.06,-.15,.123]),b[:3])


def test_base_footprint_frame_roundtrip_at_multiple_table_heights():
    pts=np.array([[.18,.14,.021],[-.06,.15,.123],[-.06,-.15,.123]],float)
    for h in (.50,.48,.25):
        converted=base_to_world(pts,tabletop_world_z=h)
        assert np.allclose(world_to_base(converted,tabletop_world_z=h),pts)
        assert np.isclose(converted[0,2],h+.021)


def test_tabletop_yaw_and_floor_collision_height_contract():
    c=_yaml("assets/embodiments/so101-pgripper-sweepick-measured/curobo.yml")
    assert np.isclose(c["planner"]["support_top_world_z"],.5)
    floor=c["planner"]["support_cuboid_base_frame"]
    assert np.isclose(floor["pose"][2]+floor["dims"][2]/2,-.123)
    t=_yaml("env_cfg/task_config/demo_so101_pgripper_sweepick_measured_smoke.yml")
    assert t["embodiment"]==["so101-pgripper-sweepick-measured"]*2+[0.]
    assert t["max_seed_attempts"]==3
    assert t["episode_num"]==1
    assert t["pcd_crop"] is False
    assert t["sweepick_crop_base"] is False
    cfg=_yaml("assets/embodiments/so101-pgripper-sweepick-measured/config.yml")
    assert np.isclose(cfg["static_camera_list"][0]["position"][2],1.0129377312368)


def test_base_and_world_crop_consistency():
    p_b=np.array([[.18,.14,.021],[-.50,.40,.09]],float)
    p_w=base_to_world(p_b)
    back=world_to_base(p_w)
    cropped=(back>=np.array([.15,-.26,.005])).all(1)&(back<=np.array([.32,.26,.40])).all(1)
    assert cropped.tolist()==[True,False]


def test_registry_and_runtime_wiring():
    reg=_yaml("env_cfg/task_config/_embodiment_config.yml")
    assert reg["so101-pgripper-sweepick-measured"]["file_path"]=="./assets/embodiments/so101-pgripper-sweepick-measured"
    scene=(ROOT/"envs/sweepick_block40_measured.py").read_text()
    base_task=(ROOT/"envs/_base_task.py").read_text()
    assert "world_to_base(" in scene
    assert "self.now_obs[\"pointcloud\"]" in scene
    assert "length=getattr(self, \"table_length\", 1.2)" in base_task
