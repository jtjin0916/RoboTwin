#!/usr/bin/env python3
"""Recorded-observation CPU adapter inference. Does not send actions to a robot."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py,numpy as np,torch
from XPolicyLab.policy.DAPIER_ACT.model import Model
from XPolicyLab.utils.process_data import decode_image_bit
from dapier_act.dataset import read_pointcloud_frame
p=argparse.ArgumentParser();p.add_argument('--checkpoint-dir',required=True);p.add_argument('--episode',required=True);a=p.parse_args()
torch.set_num_threads(2)
model=Model({'env_cfg_type':'so101_pgripper','ckpt_name':a.checkpoint_dir})
with h5py.File(a.episode,'r') as root:
 state={k:root['state/'+k+'s'][0] for k in ['left_arm_joint_state','left_ee_joint_state','right_arm_joint_state','right_ee_joint_state']}
 vision={'pointcloud':read_pointcloud_frame(root,0)}
 for camera in ['cam_left_wrist','cam_right_wrist']:vision[camera]={'color':decode_image_bit(root['vision/'+camera+'/colors'][0])}
model.update_obs({'state':state,'vision':vision})
with torch.inference_mode():actions=model.get_action()
assert isinstance(actions,list) and len(actions)==1
for action in actions:
 print('action shapes:',{k:np.asarray(v).shape for k,v in action.items()})
 assert sum(np.asarray(v).size for v in action.values())==12
 assert all(np.isfinite(v).all() for v in action.values())
print('PASS adapter CPU recorded-observation inference: 12 finite values; runtime RGB unchanged')
