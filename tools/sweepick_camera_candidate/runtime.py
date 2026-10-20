"""Explicit opt-in SAPIEN/ROS-frame adapters. Never starts arm motion."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from model import transform, obstacle_in_arm_frame


def fixed_transforms(urdf):
    root=ET.parse(urdf).getroot(); poses={'base_footprint':np.eye(4)}
    pending=list(root.findall('joint'))
    while pending:
        progressed=False
        for j in pending[:]:
            if j.get('type')!='fixed':raise ValueError('Preview adapter accepts mount-only fixed URDF')
            p=j.find('parent').get('link');c=j.find('child').get('link')
            if p not in poses:continue
            o=j.find('origin')
            xyz=[float(v) for v in o.get('xyz','0 0 0').split()] if o is not None else [0]*3
            rpy=[float(v) for v in o.get('rpy','0 0 0').split()] if o is not None else [0]*3
            poses[c]=poses[p]@transform(xyz,rpy);pending.remove(j);progressed=True
        if not progressed:raise ValueError('Disconnected or cyclic tree')
    return root,poses


def conservative_obstacles(urdf,padding_m=.005):
    """Return metre AABBs in each arm base, wxyz quaternion; no auto-install.
    Boxes contain STL vertices and camera box. Slots/holes intentionally filled.
    Chassis, other arm, cables and rail uncertainty still need separate review.
    """
    if padding_m<0:raise ValueError('padding_m must be nonnegative')
    import trimesh
    root,poses=fixed_transforms(urdf);items={}
    for link in root.findall('link'):
        points=[]
        for col in link.findall('collision'):
            o=col.find('origin');c=np.eye(4)
            if o is not None:c=transform([float(v) for v in o.get('xyz','0 0 0').split()],[float(v) for v in o.get('rpy','0 0 0').split()])
            geo=col.find('geometry');mesh=geo.find('mesh');box=geo.find('box')
            if mesh is not None:
                v=np.asarray(trimesh.load(str(Path(urdf).parent/mesh.get('filename'))).vertices)
                v=v*np.array([float(x) for x in mesh.get('scale','1 1 1').split()])
            elif box is not None:
                h=np.array([float(x) for x in box.get('size').split()])/2
                v=np.array([[x,y,z] for x in (-h[0],h[0]) for y in (-h[1],h[1]) for z in (-h[2],h[2])])
            else:raise ValueError('Unsupported geometry')
            t=poses[link.get('name')]@c
            points.append(v@t[:3,:3].T+t[:3,3])
        if points:
            v=np.concatenate(points);lo=v.min(0)-padding_m;hi=v.max(0)+padding_m
            items[link.get('name')]=((lo+hi)/2,hi-lo)
    return {side:{'cuboid':{name:obstacle_in_arm_frame(c,d,side) for name,(c,d) in items.items()}}
            for side in ('left','right')}


def attach_mount_preview(scene,urdf,tabletop_world_z=.5):
    """Add fixed mount preview to existing measured SAPIEN scene; no planner changes.
    URDF loader collision treatment is engine dependent. This is visual inspection.
    """
    import sapien
    loader=scene.create_urdf_loader();loader.fix_root_link=True
    actor=loader.load(str(Path(urdf).resolve()))
    if actor is None:raise RuntimeError('SAPIEN mount load failed')
    actor.set_root_pose(sapien.Pose(p=[0,-.2539,tabletop_world_z],q=[2**-.5,0,0,2**-.5]))
    return actor


def depth_only_observation(observation):
    """For HEAD-ONLY observations: discard synthetic colour and nonfinite point coordinates.
    Do not call on mixed head+wrist pointclouds where sensor provenance is lost.
    """
    from copy import deepcopy
    out=deepcopy(observation)
    head=out.get('observation',{}).get('head_camera',{})
    head.pop('rgb',None);head.pop('rgba',None)
    if 'pointcloud' in out:
        p=np.asarray(out['pointcloud']);out['pointcloud']=p[np.isfinite(p[:,:3]).all(1),:3].copy()
    return out
