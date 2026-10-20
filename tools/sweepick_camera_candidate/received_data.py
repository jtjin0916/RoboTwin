"""Adapters for received SDK data. Explicit profile selection; no live activation.
ICP extrinsics are estimates tied to the arm model, not body calibration measurements.
"""
import math
import numpy as np
from model import transform,MOUNT_XYZ,TOP_Z,ATTACH_XYZ,PITCH


def validate_profile(profile):
    t=np.asarray(profile['T_base_from_optical'],float)
    k=np.asarray(profile['depth_intrinsic_K'],float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):
        raise ValueError('Expected finite homogeneous transform')
    r=t[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(r),1):
        raise ValueError('Extrinsic must have a proper orthonormal rotation')
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0 or not np.allclose(k[2],[0,0,1]):
        raise ValueError('Invalid depth intrinsic K')
    if profile['depth_resolution']!=[640,460]:raise ValueError('This received data contract is 640x460 only')
    return t,k


def body_from_optical(profile):
    t,_=validate_profile(profile)
    body=transform(MOUNT_XYZ)@transform([0,0,TOP_Z])@transform(ATTACH_XYZ,[0,PITCH,0])
    return np.linalg.inv(body)@t


def rpy_from_rotation(r):
    # URDF R = Rz(yaw) Ry(pitch) Rx(roll); received rotations are not at gimbal lock.
    r=np.asarray(r,float);pitch=math.atan2(-r[2,0],math.hypot(r[0,0],r[1,0]))
    if abs(math.cos(pitch))<1e-8:raise ValueError('Gimbal lock needs explicit matrix handling')
    return [math.atan2(r[2,1],r[2,2]),pitch,math.atan2(r[1,0],r[0,0])]


def camera_config(profile,tabletop_world_z=.5):
    t,k=validate_profile(profile)
    w=transform([0,-.2539,tabletop_world_z],[0,0,math.pi/2])@t
    return dict(name='head_camera',position=w[:3,3].tolist(),forward=w[:3,2].tolist(),
                left=(-w[:3,0]).tolist(),intrinsic_K=k.tolist(),w=640,h=460,near=.2,far=2.5,
                status=profile['status'])


def apply_to_sapien_camera(camera,profile,tabletop_world_z=.5):
    """Opt-in on a head camera in a separately cloned scene, after camera creation.
    No depth bias is applied to ideal simulated geometry. No existing files mutated.
    """
    import sapien
    c=camera_config(profile,tabletop_world_z);k=c['intrinsic_K']
    if camera.width!=640 or camera.height!=460:raise ValueError('Camera must be 640x460')
    camera.set_perspective_parameters(.2,2.5,k[0][0],k[1][1],k[0][2],k[1][2],k[0][1])
    f,l=np.array(c['forward']),np.array(c['left']);t=np.eye(4)
    t[:3,:3]=np.column_stack([f,l,np.cross(f,l)]);t[:3,3]=c['position']
    camera.entity.set_pose(sapien.Pose(t))


def depth_to_m(depth_mm,*,apply_empirical_scale):
    """Received real ZD data only: mask first, then optional team 1/1.016 correction.
    This correction is an empirical operating assumption, not a factory calibration.
    Never use it to rescale ideal SAPIEN geometry, or apply twice to corrected depth.
    Range filtering is a separate caller choice; 16384's SDK meaning is unresolved.
    """
    d=np.asarray(depth_mm)
    if d.shape!=(460,640):raise ValueError('Expected received depth shape 460x640')
    valid=np.isfinite(d)&(d>0)&(d!=16384)
    z=np.full(d.shape,np.nan,dtype=np.float64)
    z[valid]=d[valid]*.001/(1.016 if apply_empirical_scale else 1.)
    return z,valid


def unproject_depth(depth_mm,profile,*,apply_empirical_scale):
    t,k=validate_profile(profile)
    z,valid=depth_to_m(depth_mm,apply_empirical_scale=apply_empirical_scale)
    v,u=np.indices(z.shape,dtype=float)
    y=(v-k[1,2])/k[1,1]
    x=(u-k[0,2]-k[0,1]*y)/k[0,0]
    optical=np.stack([x*z,y*z,z],axis=-1)
    base=optical@t[:3,:3].T+t[:3,3]
    return optical,base,valid


def apply_to_measured_task(task,profile):
    """Call once AFTER measured task super().load_camera(), in an isolated preview.
    Uses the task's final table height, so no second table bias is added.
    """
    pairs=list(zip(task.cameras.static_camera_name,task.cameras.static_camera_list))
    heads=[camera for name,camera in pairs if name=='head_camera']
    if len(heads)!=1:raise ValueError('Expected exactly one enabled static head_camera')
    apply_to_sapien_camera(heads[0],profile,tabletop_world_z=task.tabletop_world_z)
