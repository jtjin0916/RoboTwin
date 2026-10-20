"""Selected private head calibration on measured Sweepick. Existing tasks preserved.
Optical pose and K replace the old camera AFTER creation; table bias applied once.
"""
import os
from pathlib import Path
import numpy as np
import sapien
from .sweepick_block40_measured import sweepick_block40_measured
from sweepick_headcal import load_profile,world_camera,apply_camera,randomized_profile


class sweepick_block40_headcal(sweepick_block40_measured):
    def setup_demo(self,**kwargs):
        options=dict(kwargs)
        path=options.get('sweepick_head_profile') or os.environ.get('SWEEPICK_HEAD_PROFILE')
        if not path:raise ValueError('Set SWEEPICK_HEAD_PROFILE to the locally prepared head_profile.json')
        self.headcal_source=load_profile(Path(path))
        self.headcal=randomized_profile(self.headcal_source,np.random.default_rng(int(options.get('seed',0))),
            float(options.get('sweepick_head_translation_dr_m',0)),float(options.get('sweepick_head_rotation_dr_deg',0)))
        # One explicit randomization path; do not add the legacy camera perturbation.
        dr=dict(options.get('domain_randomization',{}));dr['random_head_camera_dis']=0
        options['domain_randomization']=dr
        super().setup_demo(**options)

    def load_camera(self,**kwargs):
        # Never create the old head pose, even for the parent's initial render step.
        from copy import deepcopy
        options=deepcopy(kwargs);c=world_camera(self.headcal,self.FLOOR_TOP_WORLD_Z)
        for entry in options['left_embodiment_config']['static_camera_list']:
            if entry['name']=='head_camera':
                entry.update({key:c[key] for key in ('position','forward','left')})
        super().load_camera(**options)
        heads=[cam for name,cam in zip(self.cameras.static_camera_name,self.cameras.static_camera_list) if name=='head_camera']
        if len(heads)!=1:raise ValueError('Expected enabled head_camera')
        apply_camera(heads[0],self.headcal,self.tabletop_world_z,sapien.Pose)
        self.scene.update_render()

    def get_obs(self):
        # Parent has already converted pointcloud coordinates World -> base.
        result=super().get_obs()
        head=result.get('observation',{}).get('head_camera',{})
        head.pop('rgb',None);head.pop('rgba',None)
        if isinstance(result.get('pointcloud'),np.ndarray):
            points=result['pointcloud']
            points=points[np.isfinite(points[:,:3]).all(1),:3].copy()
            result['pointcloud']=points;self.now_obs['pointcloud']=points.copy()
        result['sweepick_headcal']={'profile_sha256':self.headcal_source['_profile_file_sha256'],
            'source_extrinsic_sha256':self.headcal_source['source_extrinsic_sha256'],
            'real_depth_multiplier':1.,'pointcloud_frame':'base_footprint',
            'episode_perturbation':self.headcal.get('episode_perturbation',{})}
        return result
