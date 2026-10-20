#!/usr/bin/env python3
"""Stationary geometry-only SAPIEN viewer. No arms, planning or data collection.
Run on the team's SAPIEN 3 workstation; CPU contracts do not test rendering.
"""
import argparse,json
from pathlib import Path
import numpy as np
from model import sapien_camera
from runtime import attach_mount_preview

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--allow-unverified-camera',action='store_true');a=p.parse_args()
    import sapien
    from sapien.utils import Viewer
    scene=sapien.Scene();scene.set_timestep(1/100)
    scene.set_ambient_light([.5,.5,.5]);scene.add_directional_light([1,1,-1],[1,1,1])
    scene.add_ground(.5)
    attach_mount_preview(scene,a.model,.5)
    if a.calibration:
        c=sapien_camera(json.loads(a.calibration.read_text()),allow_unverified=a.allow_unverified_camera)
        camera=scene.add_camera('head_camera',640,460,np.deg2rad(60),.2,2.5)
        f,l=np.array(c['forward']),np.array(c['left']);t=np.eye(4)
        t[:3,:3]=np.column_stack([f,l,np.cross(f,l)]);t[:3,3]=c['position']
        camera.entity.set_pose(sapien.Pose(t))
    viewer=Viewer();viewer.set_scene(scene)
    viewer.set_camera_xyz(.8,-1.,.9);viewer.set_camera_rpy(0,-.25,2.3)
    print('CANDIDATE preview only; no optical camera unless calibration supplied')
    while not viewer.closed:
        scene.update_render();viewer.render()

if __name__=='__main__':main()
