"""Explicit local head calibration for Sweepick. Base-relative optical TF -> World.
No real-device access. Private calibration files are supplied outside source control.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from sweepick_table_frame import ROT_WORLD_FROM_BASE,BASE_ORIGIN_WORLD_XY

SCHEMA='sweepick.headcal-profile.v1'


def validate(profile):
    if profile.get('schema')!=SCHEMA:raise ValueError('Unsupported headcal schema')
    if profile.get('parent')!='base_footprint' or profile.get('child')!='camera_optical_frame':
        raise ValueError('Expected base_footprint <- camera_optical_frame')
    t=np.asarray(profile['T_base_from_optical'],float);k=np.asarray(profile['depth_K'],float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):raise ValueError('Invalid optical transform')
    r=t[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(r),1):raise ValueError('Invalid optical rotation')
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0 or not np.allclose(k[2],[0,0,1]):raise ValueError('Invalid K')
    if profile['resolution']!=[640,460]:raise ValueError('Expected 640x460 depth')
    if float(profile['real_depth_multiplier'])!=1.:raise ValueError('Latest headcal contract requires real depth multiplier 1.0')
    return t,k


def load_profile(path):
    path=Path(path).expanduser().resolve();raw=path.read_bytes();p=json.loads(raw)
    validate(p)
    p['_profile_file_sha256']=hashlib.sha256(raw).hexdigest()
    return p


def world_camera(profile,tabletop_world_z):
    t,k=validate(profile);w=np.eye(4);w[:3,:3]=ROT_WORLD_FROM_BASE
    w[:3,3]=[*BASE_ORIGIN_WORLD_XY,float(tabletop_world_z)];opt=w@t
    return {'position':opt[:3,3].tolist(),'forward':opt[:3,2].tolist(),
            'left':(-opt[:3,0]).tolist(),'K':k.tolist(),'T_world_from_optical':opt.tolist()}


def apply_camera(camera,profile,tabletop_world_z,pose_factory):
    c=world_camera(profile,tabletop_world_z);k=np.array(c['K'])
    if (camera.width,camera.height)!=(640,460):raise ValueError('Head camera resolution differs from calibration')
    camera.set_perspective_parameters(.2,2.5,k[0,0],k[1,1],k[0,2],k[1,2],k[0,1])
    f,l=np.array(c['forward']),np.array(c['left']);t=np.eye(4)
    t[:3,:3]=np.column_stack([f,l,np.cross(f,l)]);t[:3,3]=c['position']
    camera.entity.set_pose(pose_factory(t))


def real_depth_metres(raw_mm):
    """Latest real contract: no historical 1/1.016. Invalid if 0 or >=16384."""
    d=np.asarray(raw_mm)
    if d.shape!=(460,640):raise ValueError('Expected 460x640 real ZD depth')
    valid=np.isfinite(d)&(d>0)&(d<16384)
    z=np.full(d.shape,np.nan,dtype=float);z[valid]=d[valid]*.001
    return z,valid


def randomized_profile(profile,rng,translation_m=0.,rotation_deg=0.):
    """Per-episode optical mounting perturbation, not sensor noise/miscalibration.
    Bounds are per translation axis and total rotation angle. Both default to zero.
    Caller supplies empirically justified bounds; broad DR does not repair bias.
    """
    if min(translation_m,rotation_deg)<0 or not np.isfinite([translation_m,rotation_deg]).all():raise ValueError('Invalid perturbation bounds')
    out=copy.deepcopy(profile);t,_=validate(profile);t=t.copy()
    if translation_m:t[:3,3]+=rng.uniform(-translation_m,translation_m,3)
    if rotation_deg:
        v=rng.normal(size=3);v/=np.linalg.norm(v);a=rng.uniform(-1,1)*math.radians(rotation_deg)
        cross=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        r=np.eye(3)+math.sin(a)*cross+(1-math.cos(a))*(cross@cross)
        t[:3,:3]=t[:3,:3]@r
    out['T_base_from_optical']=t.tolist()
    out['episode_perturbation']={'translation_bound_per_axis_m':translation_m,'rotation_bound_deg':rotation_deg}
    validate(out);return out
