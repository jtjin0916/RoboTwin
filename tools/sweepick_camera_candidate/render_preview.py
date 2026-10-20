#!/usr/bin/env python3
"""Orthographic CAD preview, no inferred optical origin. Requires matplotlib/trimesh."""
import argparse
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from model import MOUNT_XYZ,TOP_Z,ATTACH_XYZ,PITCH,rotation

ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    fig,axes=plt.subplots(1,2,figsize=(12,8))
    for ax,xy in zip(axes,[(0,2),(1,2)]):
        for name,z,color in [('MOUNTBOTTM2_0_m.stl',0,'#617b91'),('cam_mount_top2_0_m.stl',TOP_Z,'#45a39b'),('cam_mount_top2_1_m.stl',TOP_Z,'#d8a546')]:
            mesh=trimesh.load(str(ROOT/'assets/sweepick_camera_candidate/meshes'/name));v=mesh.vertices+MOUNT_XYZ+[0,0,z]
            ax.add_collection(PolyCollection(v[mesh.faces][:,:,xy]*1000,facecolors=color,edgecolors=color,linewidths=.15,alpha=.6))
        box=trimesh.creation.box(extents=[.0255,.09,.025]);v=(box.vertices+[.01275,0,0])@rotation([0,PITCH,0]).T+np.array(MOUNT_XYZ)+[0,0,TOP_Z]+ATTACH_XYZ
        ax.add_collection(PolyCollection(v[box.faces][:,:,xy]*1000,facecolors='#885eab',edgecolors='#885eab',linewidths=.4,alpha=.6))
        ax.autoscale();ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel(('X forward' if xy[0]==0 else 'Y left')+' [mm]');ax.set_ylabel('base_footprint Z [mm]')
        ax.axhline(123,color='gray',ls='--',lw=.7);ax.axhline(524.481878142041,color='gray',ls='--',lw=.7)
        ax.set_ylim(100,550);ax.set_xlim((-170,70) if xy[0]==0 else (-120,120))
    axes[0].set_title('Side: X-Z; body normal 65 deg down');axes[1].set_title('Front: Y-Z')
    axes[0].annotate('mount origin (-60, 0, 123) mm',(-60,123),xytext=(-160,180),fontsize=9,arrowprops={'arrowstyle':'->'})
    c=np.array(MOUNT_XYZ)+[0,0,TOP_Z]+ATTACH_XYZ
    d=np.array([np.cos(PITCH),-np.sin(PITCH)])*65
    axes[0].annotate('',xy=c[[0,2]]*1000+d,xytext=c[[0,2]]*1000,arrowprops={'arrowstyle':'->','color':'#ab3860','lw':2})
    axes[0].text(-165,540,'Optical origin unresolved',fontsize=9,color='#ab3860')
    fig.suptitle('SWEEPICK CAD assembly CANDIDATE\nShoulder seating; yaw and camera rear mounting require confirmation',fontsize=13)
    fig.text(.5,.025,'Mount-only height 401.482 mm; top Z in base_footprint 524.482 mm. Purple: approximate OS30A body.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.93]);a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=160)
if __name__=='__main__':main()
