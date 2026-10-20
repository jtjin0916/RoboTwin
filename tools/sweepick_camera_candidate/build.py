#!/usr/bin/env python3
"""Write a NEW portable preview directory. Does not modify a ROS checkout.
Optional team expanded URDF and measured arm URDF produce a combined model.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from model import MOUNT_XYZ,TOP_Z,ATTACH_XYZ,PITCH,OPTICAL_RPY,sapien_camera

ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'assets/sweepick_camera_candidate'

def fmt(v): return ' '.join(f'{x:.15g}' for x in v)

def joint(root,name,parent,child,xyz,rpy=(0,0,0)):
    j=ET.SubElement(root,'joint',name=name,type='fixed')
    ET.SubElement(j,'parent',link=parent);ET.SubElement(j,'child',link=child)
    ET.SubElement(j,'origin',xyz=fmt(xyz),rpy=fmt(rpy))

def mesh_link(root,name,filenames):
    link=ET.SubElement(root,'link',name=name)
    for filename in filenames:
        for tag in ('visual','collision'):
            element=ET.SubElement(link,tag)
            ET.SubElement(ET.SubElement(element,'geometry'),'mesh',filename=filename,scale='1 1 1')
    return link

def append_mount(root,cal):
    links={e.get('name') for e in root.findall('link')}
    required={'mount_bottom','mount_top','camera_link','camera_optical_frame'}
    if required&links: raise ValueError(f'Frame conflict: {required&links}')
    if 'base_footprint' not in links: raise ValueError('Input must contain base_footprint')
    a=cal.get('assembly',{})
    mesh_link(root,'mount_bottom',['meshes/MOUNTBOTTM2_0_m.stl'])
    mesh_link(root,'mount_top',[f'meshes/cam_mount_top2_{i}_m.stl' for i in (0,1)])
    joint(root,'mount_bottom_joint','base_footprint','mount_bottom',MOUNT_XYZ,[0,0,a.get('mount_yaw_rad',0)])
    joint(root,'mount_top_joint','mount_bottom','mount_top',[0,0,TOP_Z+a.get('seating_delta_m',0)],[0,0,a.get('top_yaw_rad',0)])
    link=ET.SubElement(root,'link',name='camera_link')
    for tag in ('visual','collision'):
        e=ET.SubElement(link,tag);ET.SubElement(e,'origin',xyz='.01275 0 0')
        ET.SubElement(ET.SubElement(e,'geometry'),'box',size='.0255 .09 .025')
    joint(root,'camera_body_joint','mount_top','camera_link',ATTACH_XYZ,[0,PITCH,0])
    if cal.get('optical_xyz_in_camera_link_m') is not None:
        sapien_camera(cal,allow_unverified=True) # validates values
        ET.SubElement(root,'link',name='camera_optical_frame')
        joint(root,'camera_optical_joint','camera_link','camera_optical_frame',
              cal['optical_xyz_in_camera_link_m'],cal.get('optical_rpy_in_camera_link_rad',OPTICAL_RPY))

def portable_meshes(root,source,out,packages):
    for mesh in root.findall('.//mesh'):
        uri=mesh.get('filename')
        if uri.startswith('package://'):
            package,rel=uri[10:].split('/',1)
            if package not in packages: raise ValueError(f'Pass --package-dir {package}=/absolute/package/path')
            p=packages[package]/rel
        elif uri.startswith('file://'): p=Path(uri[7:])
        else: p=Path(uri) if Path(uri).is_absolute() else source.parent/uri
        p=p.resolve()
        if not p.is_file(): raise FileNotFoundError(p)
        digest=hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        dst=out/'meshes'/f'{digest}_{p.name}'
        shutil.copyfile(p,dst);mesh.set('filename','meshes/'+dst.name)

def append_arms(root,path,out,packages):
    arm=ET.parse(path).getroot()
    if any(e.tag not in ('link','joint','material','transmission') for e in arm):
        raise ValueError('Arm input supports links/joints/materials/transmissions; review other extensions locally')
    portable_meshes(arm,path,out,packages)
    roots={x.get('name') for x in arm.findall('link')}-{j.find('child').get('link') for j in arm.findall('joint')}
    if len(roots)!=1: raise ValueError('Expected one arm root')
    for side,y in [('left',.15),('right',-.15)]:
        clone=copy.deepcopy(arm)
        for e in clone.iter():
            if e.tag in ('link','joint','material','transmission','actuator') and e.get('name'):e.set('name',side+'_'+e.get('name'))
            if e.tag in ('parent','child'):e.set('link',side+'_'+e.get('link'))
            if e.tag=='mimic':e.set('joint',side+'_'+e.get('joint'))
        root.extend(clone)
        joint(root,side+'_arm_mount','base_footprint',side+'_'+next(iter(roots)),[-.06,y,.123])

def validate_tree(root,out):
    names=[e.get('name') for e in root.findall('link')]
    joints=[e.get('name') for e in root.findall('joint')]
    if len(set(names))!=len(names) or len(set(joints))!=len(joints):raise ValueError('Duplicate names')
    parent={}
    for j in root.findall('joint'):
        p,c=j.find('parent').get('link'),j.find('child').get('link')
        if p not in names or c not in names or c in parent:raise ValueError('Invalid URDF tree')
        parent[c]=p
    if len(set(names)-set(parent))!=1:raise ValueError('Expected single root')
    for n in names:
        seen=set()
        while n in parent:
            if n in seen:raise ValueError('Cycle')
            seen.add(n);n=parent[n]
    for m in root.findall('.//mesh'):
        if not (out/m.get('filename')).is_file():raise FileNotFoundError(m.get('filename'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--robot-urdf',type=Path)
    p.add_argument('--arm-urdf',type=Path,help='Both measured arms added at confirmed roots; omit if robot already has arms')
    p.add_argument('--package-dir',action='append',default=[])
    p.add_argument('--calibration',type=Path,default=ASSET/'calibration.template.json')
    a=p.parse_args();out=a.out.resolve()
    if out.exists():raise FileExistsError('Use a NEW output directory; existing outputs are never overwritten')
    cal=json.loads(a.calibration.read_text())
    packages={k:Path(v).resolve() for k,v in (x.split('=',1) for x in a.package_dir)}
    # Stage beside destination so errors leave no partially generated deliverable.
    import tempfile
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage=Path(temp);(stage/'meshes').mkdir()
        root=ET.parse(a.robot_urdf).getroot() if a.robot_urdf else ET.Element('robot',name='sweepick_camera_CANDIDATE')
        if a.robot_urdf:portable_meshes(root,a.robot_urdf,stage,packages)
        else:ET.SubElement(root,'link',name='base_footprint')
        if a.arm_urdf:append_arms(root,a.arm_urdf,stage,packages)
        for mesh in (ASSET/'meshes').glob('*.stl'):shutil.copyfile(mesh,stage/'meshes'/mesh.name)
        append_mount(root,cal);validate_tree(root,stage)
        ET.indent(root);ET.ElementTree(root).write(stage/'sweepick_candidate.urdf',encoding='utf-8',xml_declaration=True)
        (stage/'calibration.json').write_text(json.dumps(cal,indent=2)+'\n')
        if cal.get('optical_xyz_in_camera_link_m') is not None:
            import yaml
            config={'status':cal['status'],'static_camera_list':[sapien_camera(cal,allow_unverified=cal['status']!='measured_and_reviewed')],
                    'enable_color_stream':False,'rgb':False,'collect_data':False}
            (stage/'sapien_camera_candidate.yml').write_text(yaml.safe_dump(config))
        (stage/'STATUS.txt').write_text('CANDIDATE ONLY. Assembly, optical extrinsic, full robot collisions require review.\nNo optical TF or SAPIEN extrinsic is emitted without supplied optical calibration.\n')
        shutil.copytree(stage,out)
    print(out/'sweepick_candidate.urdf')

if __name__=='__main__':main()
