"""Streaming aligned q(t), command(t), sensors(t); excludes legacy joint_action."""
import json
from pathlib import Path
import cv2
import h5py
import numpy as np
from .contract import SCHEMA

JOINT_KEYS = ('left_arm_joint_states', 'left_ee_joint_states',
              'right_arm_joint_states', 'right_ee_joint_states')
SLICES = (slice(0,5), slice(5,6), slice(6,11), slice(11,12))


def robot_vectors(robot):
    observed, command, requested = [], [], []
    for arm in ('left', 'right'):
        entity = getattr(robot, arm+'_entity')
        active = list(entity.get_active_joints())
        joints = list(getattr(robot, arm+'_arm_joints')) + [getattr(robot, arm+'_gripper')[0][0]]
        q = entity.get_qpos()
        observed.extend(float(q[active.index(j)]) for j in joints)
        command.extend(float(j.get_drive_target()[0]) for j in joints)
        requested.append(float(getattr(robot, 'get_'+arm+'_gripper_val')()))
    return np.asarray(observed, np.float32), np.asarray(command, np.float32), requested


class EpisodeWriter:
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.partial = self.path.with_suffix('.partial.hdf5')
        if self.path.exists(): raise FileExistsError(self.path)
        self.root = h5py.File(self.partial, 'x')
        self.root.attrs.update(schema=SCHEMA, metadata=json.dumps(metadata,sort_keys=True),
            pointcloud_frame='base_footprint', head_modality='depth_only',
            state_dim=12, action_dim=12, gripper_units='motor_joint_rad',
            alignment='sensors_and_observed_q_before_physics_step; current_drive_target_is_action',
            task_success=False, complete=False)
        self.index = 0
        self.control_index = 0

    def append(self, key, value, dtype=None):
        a = np.asarray(value)
        if key not in self.root:
            self.root.create_dataset(key, shape=(0,)+a.shape, maxshape=(None,)+a.shape,
                                     dtype=dtype or a.dtype, chunks=True)
        ds = self.root[key]; ds.resize(self.index+1, axis=0); ds[self.index]=value

    def control_frame(self, timestamp, phase, q, action, requested):
        # Full physics-rate trace prevents hidden downsampling of Expert commands.
        for key,value,dtype in [('timestamp',timestamp,None),('q',q,None),('action',action,None),
                                ('gripper_requested',requested,None),('phase',phase,h5py.string_dtype())]:
            a=np.asarray(value);name='control/'+key
            if name not in self.root:
                self.root.create_dataset(name,shape=(0,)+a.shape,maxshape=(None,)+a.shape,
                                         dtype=dtype or a.dtype,chunks=True)
            ds=self.root[name];ds.resize(self.control_index+1,axis=0);ds[self.control_index]=value
        self.control_index+=1

    def frame(self, timestamp, phase, obs, q, action, requested, block_xyz):
        if np.shape(q)!=(12,) or np.shape(action)!=(12,) or not np.isfinite([q,action]).all():
            raise ValueError('Invalid measured/command vectors')
        self.append('timestamp', float(timestamp)); self.append('phase', phase, h5py.string_dtype())
        self.append('gripper/requested_robotwin_normalized', requested)
        self.append('block_world_xyz', block_xyz)
        for key, sl in zip(JOINT_KEYS, SLICES):
            self.append('state/'+key, q[sl]); self.append('action/'+key, action[sl])
        xyz = np.asarray(obs['pointcloud'],np.float32)
        if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all():
            raise ValueError('Missing/nonfinite head XYZ in base frame')
        self.root.create_dataset(f'vision/pointcloud/{self.index:06}', data=xyz, compression='gzip')
        head = obs['observation']['head_camera']
        if 'rgb' in head or 'rgba' in head: raise ValueError('Head RGB is forbidden')
        depth=np.asarray(head['depth'],np.float32)/1000.
        valid=np.isfinite(depth)&(depth>0)&(depth<16.384)
        self.append('vision/head_depth_m',np.where(valid,depth,0.))
        self.append('vision/head_depth_valid',valid)
        for source, dest in [('left_camera','cam_left_wrist'), ('right_camera','cam_right_wrist')]:
            rgb=np.asarray(obs['observation'][source]['rgb'])
            if rgb.shape!=(240,320,3) or rgb.dtype!=np.uint8:
                raise ValueError(f'Expected native 240x320 uint8 RGB: {source} {rgb.shape}')
            ok, jpg=cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not ok: raise ValueError('JPEG encode failed')
            key=f'vision/{dest}/colors'
            if key not in self.root:
                self.root.create_dataset(key, shape=(0,),maxshape=(None,),dtype=h5py.vlen_dtype(np.dtype('uint8')))
            ds=self.root[key]; ds.resize(self.index+1,axis=0); ds[self.index]=jpg.ravel()
        self.index+=1

    def close(self, success, metrics):
        self.root.attrs['task_success']=bool(success)
        self.root.attrs['metrics']=json.dumps(metrics,allow_nan=False)
        self.root.attrs['complete']=True
        self.root.flush();self.root.close()
        # Every attempt retained; loader explicitly filters task_success.
        self.partial.rename(self.path)
