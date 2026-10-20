"""CPU-only contract for 13cm headcal crop and 15cm near-field Grid100."""
import csv,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from sweepick_workspace_near_grid100 import X_CENTERS,Y_CENTERS,CROP_MIN,CROP_MAX,NUM_CELLS,cell,cells,BLOCK_HALF_SIZE
from sweepick_table_frame import base_to_world,world_to_base
from scripts import scan_sweepick_headcal_near_grid100 as scan
ROOT=Path(__file__).resolve().parents[1]
CFG="demo_so101_pgripper_sweepick_headcal_near_grid100"

def test_100_grid_and_nonnegative_cube_crop_clearance():
    assert NUM_CELLS==100
    assert np.allclose(X_CENTERS,np.linspace(.15,.29,10))
    assert np.allclose(Y_CENTERS,np.linspace(-.15,.15,10))
    assert np.allclose(CROP_MIN,[.13,-.26,.005])
    assert np.allclose(CROP_MAX,[.32,.26,.40])
    assert cell(0)==(.15,-.15) and np.allclose(cell(99),(.29,.15))
    for q in cells():
        x,y=q["x_base_m"],q["y_base_m"]
        assert x-BLOCK_HALF_SIZE>=CROP_MIN[0]-1e-9
        assert x+BLOCK_HALF_SIZE<=CROP_MAX[0]
        assert y-BLOCK_HALF_SIZE>=CROP_MIN[1]
        assert y+BLOCK_HALF_SIZE<=CROP_MAX[1]
        p=[x,y,q["z_base_m"]]
        assert np.allclose(world_to_base(base_to_world(p)),p)

def test_separate_headcal_crop_config_and_originals():
    new=yaml.safe_load((ROOT/"env_cfg/task_config"/(CFG+".yml")).read_text())
    old=yaml.safe_load((ROOT/"env_cfg/task_config/demo_so101_pgripper_sweepick_headcal.yml").read_text())
    assert new["sweepick_crop_base"] and not old["sweepick_crop_base"]
    assert not new["data_type"]["rgb"]
    assert not new["domain_randomization"]["random_light"]
    assert new["domain_randomization"]["random_head_camera_dis"]==0
    assert new["sweepick_head_translation_dr_m"]==0
    assert new["sweepick_head_rotation_dr_deg"]==0
    assert new["save_path"]!=old["save_path"]
    assert "np.linspace(.18, .29, 10" in (ROOT/"sweepick_workspace_grid100.py").read_text()
    task=(ROOT/"envs/sweepick_block40_headcal_near_grid100.py").read_text()
    assert "class sweepick_block40_headcal_near_grid100(sweepick_block40_headcal)" in task

class Stub:
    def setup_demo(self,**kwargs):
        self.data=kwargs
        self.BLOCK_ROBOT_XY=cell(kwargs["grid_index"])
        self.scan_metrics={"failure_stage":"", "lift_success":True}
    def get_obs(self):
        x,y=self.BLOCK_ROBOT_XY
        return {"pointcloud":np.array([[x,y,.021]])}
    def play_once(self):return {}
    def close_env(self,**kw):return None

def test_assigned_policy_and_resume(tmp_path):
    path=tmp_path/"near/attempts.csv"
    opts=SimpleNamespace(out=path,config=CFG,start=0,limit=100,
                         arms=["left","right"],arm_mode="assigned",repeats=1)
    scan.run_scan(opts,{},Stub)
    with path.open(newline="") as f:rows=list(csv.DictReader(f))
    assert len(rows)==100
    assert all(r["arm"]==("right" if float(r["y_base_m"])<0 else "left") for r in rows)
    assert sum(r["arm"]=="left" for r in rows)==50
    assert sum(r["arm"]=="right" for r in rows)==50
    original=path.read_bytes()
    scan.run_scan(opts,{},Stub)
    assert path.read_bytes()==original

def test_list_only_without_sapien_profile():
    p=ROOT/"scripts/scan_sweepick_headcal_near_grid100.py"
    z=subprocess.run([sys.executable,str(p),"--list-only"],capture_output=True,text=True)
    assert z.returncode==0,z.stderr
    lines=z.stdout.splitlines()
    assert len(lines)==100
    assert lines[0].startswith("00: x=0.15000, y=-0.15000")
    assert lines[-1].startswith("99: x=0.29000, y=+0.15000")
