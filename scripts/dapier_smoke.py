#!/usr/bin/env python3
"""Validate the preserved embodiment/config and optional CPU jaw physics.
Camera rendering / CuRobo planning must be checked separately on an NVIDIA host.
"""
import argparse,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import yaml
p=argparse.ArgumentParser();p.add_argument('--physics',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];base=root/'assets/embodiments/so101-pgripper'
cfg=yaml.safe_load((base/'config.yml').read_text());assert cfg['urdf_path']=='./so101_pgripper.urdf'
assert cfg['gripper_scale']==[2.2028,0.0] and cfg['gripper_bias']==.08224
assert cfg['move_group']==['Gripper_Base_v1_1']*2
reg=json.loads((root/'env_cfg/robot/_robot_info.json').read_text())['dual_so101_pgripper'];assert reg=={'arm_dim':[5,5],'ee_dim':[1,1]}
xreg=json.loads((root/'XPolicyLab/utils/robot/_robot_info.json').read_text())['so101_pgripper'];assert xreg==reg
cam=yaml.safe_load((root/'env_cfg/task_config/_camera_config.yml').read_text());assert (cam['D435']['h'],cam['D435']['w'])==(240,320)
task=yaml.safe_load((root/'env_cfg/task_config/demo_so101_pgripper_calib.yml').read_text());assert task['embodiment'][:2]==['so101-pgripper']*2
assert task['camera']['collect_head_camera'] and task['camera']['collect_wrist_camera']
kin=[]
for name in ['so101_pgripper.urdf','so101_pgripper_calib.urdf']:
 u=ET.parse(base/name); joints={j.attrib['name']:j for j in u.findall('joint')}
 kin.append({n:{tag:joints[n].find(tag).attrib for tag in ('parent','child','origin','axis')} for n in cfg['arm_joints_name'][0]})
 for mesh in u.findall('.//mesh'):assert (base/mesh.attrib['filename']).is_file(),mesh.attrib
 if name=='so101_pgripper.urdf':assert len(u.findall(".//collision[@name='jaw_box_collision']/geometry/box"))==2
 for j,m in [('rev_motor_08_1',-.0115),('rev_motor_08_2',.0115)]:
  assert joints[j].attrib['type']=='prismatic'
  assert [j,m,0.0] in cfg['gripper_name'][0]['mimic']
for n in kin[0]:assert kin[0][n]==kin[1][n],n
for entry in json.loads((base/'ASSET_PROVENANCE.json').read_text())['files']:
 assert hashlib.sha256((base/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
print('PASS config, two-arm action_dim=12, head/wrist config, all mesh hashes, jaw boxes, runtime/planner arm joints')
if a.physics:
 import sapien,numpy as np
 from sapien.physx import PhysxCpuSystem
 scene=sapien.Scene([PhysxCpuSystem()]);scene.set_timestep(1/250)
 robots=[]
 for arm in range(2):
  loader=scene.create_urdf_loader();loader.fix_root_link=True
  robot=loader.load(str(base/'so101_pgripper.urdf'));assert robot is not None
  robot.set_root_pose(sapien.Pose([arm*.5,0,0]));robots.append(robot)
 for robot in robots:
  names=[j.name for j in robot.get_active_joints()];q=robot.get_qpos()
  for angle in [0.,2.2028,0.]:
   q[names.index('rev_motor_08')]=angle
   q[names.index('rev_motor_08_1')]=-.0115*angle
   q[names.index('rev_motor_08_2')]=.0115*angle
   robot.set_qpos(q);assert np.isfinite(robot.get_qpos()).all()
   np.testing.assert_allclose(robot.get_qpos(),q,atol=1e-6)
 print('PASS CPU dual articulation load and open/close kinematic mapping; dynamic/contact hold NOT tested')
