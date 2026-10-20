#!/usr/bin/env python3
"""Audit an OS30A handoff ZIP as data; write NEW opt-in comparison models.
No SDK code is executed and no current robot/task configuration is modified.
"""
import argparse,hashlib,io,json,re,shutil,tempfile
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
import yaml
import xml.etree.ElementTree as ET
from build import append_mount,validate_tree,ROOT,ASSET,joint
from received_data import validate_profile,body_from_optical,rpy_from_rotation,camera_config,unproject_depth


def audit(archive):
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())>200_000_000:raise ValueError('Unexpected archive size')
        def read(suffix):
            names=[n for n in z.namelist() if n.endswith('/'+suffix)]
            if len(names)!=1:raise ValueError('Expected unique '+suffix)
            return z.read(names[0])
        scene=json.loads(read('SCENE.json'));raw=read('depth_000006.png')
        depth=np.asarray(Image.open(io.BytesIO(raw)))
        merged=np.load(io.BytesIO(read('depth_mm.npy')),allow_pickle=False)
        if depth.shape!=(460,640) or depth.dtype!=np.uint16:raise ValueError('Unexpected raw depth')
        if merged.shape!=(460,640) or merged.dtype!=np.float32:raise ValueError('Unexpected merged depth')
        trace=read('sdk.trace.log').decode('utf-8');lines=trace.splitlines()
        h=14695981039346656037
        for byte in depth.astype('<u2').tobytes():h=((h^byte)*1099511628211)&((1<<64)-1)
        matches=[i for i,l in enumerate(lines) if f'depth_zd={h:016x}' in l]
        if len(matches)!=1:raise ValueError('Raw depth does not uniquely match SDK trace')
        k=np.array(scene['K']).reshape(3,3);kd=np.diag([.5,.5,1])@k
        profiles={}
        report={'archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                'evidence_sha256':{i.filename:hashlib.sha256(z.read(i)).hexdigest() for i in z.infolist() if not i.is_dir()},
                'raw_depth':{'shape':list(depth.shape),'dtype':str(depth.dtype),'zeros':int((depth==0).sum()),
                  'value_16384_pixels':int((depth==16384).sum()),'fnv1a64':f'{h:016x}',
                  'matching_trace_context':lines[matches[0]-1:matches[0]+2]},
                'median_depth':{'shape':list(merged.shape),'dtype':str(merged.dtype),
                  'fractional_pixels':int((np.isfinite(merged)&(merged%1!=0)).sum()),
                  'note':'30-frame per-pixel median, not same acquisition as raw frame 6'},
                'depth_K':kd.tolist(),'K_mapping':'u_reference=2*u_depth; v_reference=2*v_depth; no half-pixel correction (team convention, full factory log unavailable)',
                'images':{}}
        for name in ['top.png','top_000006.jpg']:
            im=np.asarray(Image.open(io.BytesIO(read(name))))
            report['images'][name]={'shape':list(im.shape),'equal_rgb_channel_share':float(np.mean((im[...,0]==im[...,1])&(im[...,1]==im[...,2])))}
        records=re.findall(r'received sn=(\d+),(\d+),(\d+) timestamps_us=(\d+),(\d+),(\d+),(\d+)',trace)
        stamps=np.array([int(q[4]) for q in records],float)/1e6
        report['frameset_delivery']={'record_count':len(records),'depth_sdk_span_s':float(stamps[-1]-stamps[0]),
            'average_depth_timestamp_rate_hz':float((len(stamps)-1)/(stamps[-1]-stamps[0])),
            'note':'Delivered FrameSets by SDK depth stamps, not independent color exposure FPS or host acquisition latency'}
        report['profiles']={}
        for key,filename in [('in_use','T_base_footprint_from_os30a_optical_IN_USE_20261018.json'),('candidate','T_base_footprint_from_os30a_optical_CANDIDATE_20261020.json')]:
            source=json.loads(read(filename));t=np.array(source['T_parent_child'])
            profile={'id':key,'status':source['status'],'method':source['method'],
              'T_base_from_optical':t.tolist(),'depth_resolution':[640,460],
              'depth_intrinsic_K':kd.tolist(),'reference_resolution':[1280,920],
              'reference_intrinsic_K':k.tolist(),'depth_unit':'mm','empirical_depth_multiplier':1/1.016,
              'raw_invalid_codes':[0,16384],'depth_axis_definition':'Z-depth assumed by team; SDK pointcloud equivalence not yet checked',
              'source_extrinsic':source,'intrinsic_provenance':'SCENE.json K; SDK NewCamMat1 according to team README; full factory log absent',
              'body_from_optical_provenance':'Computed inverse(CAD candidate body TF) @ empirical base optical TF; NOT an independently measured body optical calibration',
              'requested_downward_angle_from_horizontal_deg':25.,'automatically_activated':False}
            validate_profile(profile);profiles[key]=profile
            derived=body_from_optical(profile)
            optical,base,valid=unproject_depth(depth,profile,apply_empirical_scale=True)
            crop=valid&np.all(base>=[.15,-.26,.005],axis=-1)&np.all(base<=[.32,.26,.40],axis=-1)
            f=t[:3,2];report['profiles'][key]={'down_deg':float(np.degrees(np.arctan2(-f[2],np.linalg.norm(f[:2])))),
                'R_determinant':float(np.linalg.det(t[:3,:3])),'T_body_from_optical_derived':derived.tolist(),
                'raw_sample_valid_points':int(valid.sum()),'raw_sample_points_in_task_crop':int(crop.sum()),
                'crop_note':'Same raw frame reprojected with each empirical TF; no independent ground truth or P5 refit in this archive'}
        delta=np.array(profiles['candidate']['T_base_from_optical'])@np.linalg.inv(np.array(profiles['in_use']['T_base_from_optical']))
        point=np.r_[scene['object']['centre_world'],1.]
        a,b,c=scene['desk_z']['floor']['coef']
        oldplane=np.array([-a,-b,1.,-c]);newplane=np.linalg.inv(delta).T@oldplane;newplane/=newplane[2]
        report['scene_diagnostic_reexpression']={'old_reported_object_center_m':point[:3].tolist(),'same_point_under_candidate_m':(delta@point)[:3].tolist(),
            'old_reported_floor_z_equals_ax_by_c':[a,b,c], 'same_plane_under_candidate_z_equals_ax_by_c':[-newplane[0],-newplane[1],-newplane[3]],
            'note':'Algebraic re-expression of SCENE median-observation diagnostics, NOT independent truth or re-detection on raw sample 6'}
        report['scene_uses_in_use_transform']=bool(np.allclose(scene['T_world_camera'],profiles['in_use']['T_base_from_optical'],atol=1e-12))
        report['scene_frame_warning']='SCENE T_world_camera is numerically the base optical transform. RoboTwin World requires Rz(+90deg) and (0,-.2539,tabletop_z). Do not use SCENE world name as RoboTwin World.'
        report['not_verified']=['Full factory calibration R/T/D, firmware, SDK pointcloud mapping and Z-depth definition',
          'Independent physical body optical origin and absolute robot extrinsic',
          'P5 fit metrics (reported by provider only; fitting poses and pointclouds absent)',
          'Scale 1/1.016 (reported from separate board experiment; board evidence absent)',
          'Cause of reported tool-minus-block XY residual (+27..31,+15)mm',
          'GPU SAPIEN/CuRobo runtime or physical collision safety']
        return profiles,report


def write_models(profiles,report,out):
    for key,p in profiles.items():
        dest=out/key;dest.mkdir();(dest/'meshes').mkdir()
        for m in (ASSET/'meshes').glob('*.stl'):shutil.copyfile(m,dest/'meshes'/m.name)
        root=ET.Element('robot',name='sweepick_received_'+key+'_ICP_ESTIMATE');ET.SubElement(root,'link',name='base_footprint')
        append_mount(root,{'assembly':{}},include_optical=False)
        local=body_from_optical(p);ET.SubElement(root,'link',name='camera_optical_frame')
        joint(root,'camera_optical_joint','camera_link','camera_optical_frame',local[:3,3],rpy_from_rotation(local[:3,:3]))
        root.insert(0,ET.Comment(' Optical TF reproduces an empirical base-to-optical ICP profile. Body-relative transform is algebraically derived from candidate CAD, NOT independently measured. '))
        validate_tree(root,dest);ET.indent(root);ET.ElementTree(root).write(dest/'sweepick_received.urdf',encoding='utf-8',xml_declaration=True)
        (dest/'profile.json').write_text(json.dumps(p,indent=2)+'\n')
        (dest/'sapien_camera_preview.yml').write_text(yaml.safe_dump({'profile_status':p['status'],
            'automatically_activated':False,'static_camera_list':[camera_config(p)]},sort_keys=False))
        # Do not copy embodiment as an active config: its relative arm paths would change.
        # Only emit an explicit head-camera override to merge beside existing measured config.
        (dest/'measured_head_override.yml').write_text(yaml.safe_dump({'profile_status':p['status'],
            'static_camera_list':[camera_config(p)],'note':'Manual opt-in only. Apply exact K using received_data.apply_to_sapien_camera; fovy-only cameras ignore the principal-point offset.'},sort_keys=False))
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve()
    if out.exists():raise FileExistsError('Use a new output directory')
    profiles,report=audit(a.archive);out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage=Path(temp);write_models(profiles,report,stage);shutil.copytree(stage,out)
    print(json.dumps({'output':str(out),'raw_depth':report['raw_depth'],'profiles':report['profiles']},indent=2))

if __name__=='__main__':main()
