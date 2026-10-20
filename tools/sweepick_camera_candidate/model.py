"""CPU geometry/TF helpers. Camera body datum != depth optical origin."""
import math
import numpy as np

MOUNT_XYZ = [-.060, 0., .123]
TOP_Z = .223712455749511
# CAD face 1187 centre, after top normalization. Direct rear M3 mounting ASSUMED.
ATTACH_XYZ = [.02369738224671755, .0001870002746582, .16817842666893002]
PITCH = math.radians(65.)
OPTICAL_RPY = [-math.pi/2, 0., -math.pi/2]


def rotation(rpy):
    if len(rpy)!=3 or not np.isfinite(np.asarray(rpy,float)).all():
        raise ValueError("RPY must contain three finite radians")
    r,p,y=map(float,rpy)
    cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def transform(xyz,rpy=(0,0,0)):
    if len(xyz)!=3 or not np.isfinite(np.asarray(xyz,float)).all():
        raise ValueError("XYZ must contain three finite metres")
    t=np.eye(4);t[:3,:3]=rotation(rpy);t[:3,3]=xyz;return t


def camera_tf(calibration):
    """Require measured translation; never invent a default optical centre."""
    xyz=calibration.get('optical_xyz_in_camera_link_m')
    if xyz is None or len(xyz)!=3 or not np.isfinite(np.asarray(xyz,float)).all():
        raise ValueError('Measure optical_xyz_in_camera_link_m; body/front-face centres are not depth origins')
    a=calibration.get('assembly',{})
    return (transform(MOUNT_XYZ,[0,0,a.get('mount_yaw_rad',0)]) @
            transform([0,0,TOP_Z+a.get('seating_delta_m',0)],[0,0,a.get('top_yaw_rad',0)]) @
            transform(ATTACH_XYZ,[0,PITCH,0]) @
            transform(xyz,calibration.get('optical_rpy_in_camera_link_rad',OPTICAL_RPY)))


def sapien_camera(calibration,tabletop_world_z=.5,allow_unverified=False):
    if calibration.get('status')!='measured_and_reviewed' and not allow_unverified:
        raise ValueError('Unverified camera calibration; preview requires explicit allow_unverified=True')
    t=camera_tf(calibration)
    w=transform([0,-.2539,tabletop_world_z],[0,0,math.pi/2])@t
    return dict(name='head_camera',position=w[:3,3].tolist(),
                forward=w[:3,2].tolist(),left=(-w[:3,0]).tolist())


def obstacle_in_arm_frame(center_base,dims,side):
    """CuRobo cuboid wxyz pose; measured arms have identity R in base_footprint."""
    if side not in ('left','right'): raise ValueError(side)
    root=np.array([-.060,.150 if side=='left' else -.150,.123])
    return dict(dims=list(dims),pose=[*(np.asarray(center_base)-root).tolist(),1,0,0,0])
