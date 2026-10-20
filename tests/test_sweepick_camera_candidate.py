"""CPU-only CAD/mesh/frame/portable integration checks, no SAPIEN or robot motion."""
from pathlib import Path
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
TOOL=ROOT/'tools/sweepick_camera_candidate'
sys.path.insert(0,str(TOOL))
from model import camera_tf,sapien_camera,rotation,OPTICAL_RPY,PITCH
from runtime import fixed_transforms,conservative_obstacles,depth_only_observation
A=ROOT/'assets/sweepick_camera_candidate'

@pytest.fixture
def model(tmp_path):
    out=tmp_path/'preview'
    subprocess.run([sys.executable,str(TOOL/'build.py'),'--out',str(out)],check=True)
    return out/'sweepick_candidate.urdf'


def test_step_and_exported_meshes_preserve_units_and_volume():
    import hashlib,trimesh
    audit=json.loads((A/'cad_audit.json').read_text())
    assert audit['proposed_assembly']['interference_volume_mm3']<1e-5
    assert np.isclose(audit['proposed_assembly']['bounds_mm'][2][1],401.481878142041)
    for name,d in audit['parts'].items():
        assert hashlib.sha256((A/'cad'/f'{name}.step').read_bytes()).hexdigest()==d['sha256']
        meshes=[trimesh.load(str(A/'meshes'/f'{name}_{i}_m.stl')) for i in range(d['solids'])]
        assert all(m.is_watertight for m in meshes)
        assert abs(sum(m.volume for m in meshes)/1e-9/d['volume_mm3']-1)<.001
        bounds=np.array(d['normalized_bounds_mm']).T/1000
        v=np.concatenate([m.vertices for m in meshes])
        assert np.allclose([v.min(0),v.max(0)],bounds,atol=1e-5)


def test_plate_holes_and_25_degree_plane():
    faces=json.loads((A/'cam_mount_top2_faces.json').read_text())
    holes=[f for f in faces if f['type']=='CYLINDER' and abs(f['radius_mm']-1.5)<1e-8]
    assert len(holes)==2
    assert np.isclose(np.linalg.norm(np.array(holes[0]['axis_point_mm'])-holes[1]['axis_point_mm']),45)
    assert np.allclose(holes[0]['axis'],[np.sin(np.deg2rad(25)),-np.cos(np.deg2rad(25)),0])


def test_optical_origin_is_never_defaulted_and_forward_sign_is_correct():
    cal=json.loads((A/'calibration.template.json').read_text())
    with pytest.raises(ValueError,match='optical'):camera_tf(cal)
    # Synthetic offsets verify math ONLY, not a hardware calibration.
    cal['optical_xyz_in_camera_link_m']=[.021,.004,.002]
    with pytest.raises(ValueError,match='Unverified'):sapien_camera(cal)
    R=rotation(OPTICAL_RPY)
    assert np.allclose(R[:,2],[1,0,0])
    assert np.allclose(R[:,0],[0,-1,0])
    assert np.allclose(R[:,1],[0,0,-1])
    t=camera_tf(cal)
    assert np.allclose(t[:3,2],[np.cos(PITCH),0,-np.sin(PITCH)])
    info=sapien_camera(cal,allow_unverified=True,tabletop_world_z=.5)
    info2=sapien_camera(cal,allow_unverified=True,tabletop_world_z=.3)
    assert np.allclose(np.array(info['position'])-info2['position'],[0,0,.2])
    assert np.allclose(info['forward'],[0,np.cos(PITCH),-np.sin(PITCH)])
    assert np.isclose(np.dot(info['forward'],info['left']),0)


def test_urdf_datums_portability_and_missing_optical(model):
    root,poses=fixed_transforms(model)
    assert np.allclose(poses['mount_bottom'][:3,3],[-.06,0,.123])
    assert np.isclose(poses['mount_top'][2,3],.346712455749511)
    assert 'camera_optical_frame' not in poses
    assert not (model.parent/'sapien_camera_candidate.yml').exists()
    assert all((model.parent/m.get('filename')).is_file() for m in root.findall('.//mesh'))
    assert all(m.get('scale')=='1 1 1' for m in root.findall('.//mesh'))
    before=model.read_bytes()
    result=subprocess.run([sys.executable,str(TOOL/'build.py'),'--out',str(model.parent)],capture_output=True)
    assert result.returncode!=0 and model.read_bytes()==before


def test_obstacles_have_correct_per_arm_frames_and_cover_all_vertices(model):
    import trimesh
    root,poses=fixed_transforms(model)
    obs=conservative_obstacles(model)
    for side,y in [('left',.15),('right',-.15)]:
        for name,item in obs[side]['cuboid'].items():
            center=np.array(item['pose'][:3])+[-.06,y,.123];half=np.array(item['dims'])/2
            link=root.find(f"link[@name='{name}']")
            for m in link.findall('.//mesh'):
                v=trimesh.load(str(model.parent/m.get('filename'))).vertices
                t=poses[name];v=v@t[:3,:3].T+t[:3,3]
                assert (v>=center-half).all() and (v<=center+half).all()
    assert np.allclose(np.array(obs['left']['cuboid']['mount_top']['pose'][:3])-obs['right']['cuboid']['mount_top']['pose'][:3],[0,-.3,0])


def test_portable_combined_robot_arm_tree(tmp_path):
    body=tmp_path/'body.urdf';body.write_text('<robot name="team"><link name="base_footprint"/></robot>')
    # Original tracked arm used only for compatibility test; real measured STL is external.
    arm=ROOT/'assets/embodiments/so101-pgripper/so101_pgripper.urdf'
    out=tmp_path/'combined'
    subprocess.run([sys.executable,str(TOOL/'build.py'),'--out',str(out),'--robot-urdf',str(body),'--arm-urdf',str(arm)],check=True)
    r=ET.parse(out/'sweepick_candidate.urdf').getroot()
    for side,y in [('left',.15),('right',-.15)]:
        j=r.find(f"joint[@name='{side}_arm_mount']")
        assert np.allclose(np.fromstring(j.find('origin').get('xyz'),sep=' '),[-.06,y,.123])
    assert all((out/m.get('filename')).is_file() for m in r.findall('.//mesh'))


def test_depth_observation_has_no_synthetic_rgb():
    original={'observation':{'head_camera':{'rgb':np.zeros((2,2,3)),'depth':np.ones((2,2))}},
              'pointcloud':np.array([[1,2,3,1,0,0],[float('nan'),0,1,0,0,0]])}
    out=depth_only_observation(original)
    assert 'rgb' not in out['observation']['head_camera']
    assert out['pointcloud'].shape==(1,3)
    assert 'rgb' in original['observation']['head_camera']


def test_xacro_and_generated_urdf_match(model,tmp_path):
    import xacro
    wrapper=tmp_path/'preview.xacro'
    wrapper.write_text(f'''<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="preview">
    <xacro:include filename="{TOOL/'camera_mount.urdf.xacro'}"/>
    <link name="base_footprint"/>
    <xacro:sweepick_camera_mount_candidate parent="base_footprint" prefix="" mesh_dir="{A/'meshes'}"/>
    </robot>''')
    expanded=tmp_path/'expanded.urdf';expanded.write_text(xacro.process_file(str(wrapper)).toxml())
    _,a=fixed_transforms(model);_,b=fixed_transforms(expanded)
    assert a.keys()==b.keys()
    for name in a:assert np.allclose(a[name],b[name])


def test_reviewed_calibration_rejects_user_angle_conflict():
    cal=json.loads((A/'calibration.template.json').read_text())
    assert cal['user_confirmed']['bottom_long_axis_robot']=='X'
    assert cal['user_confirmed']['camera_rear_m3_direct_attachment'] is True
    cal['optical_xyz_in_camera_link_m']=[.02,0,0] # synthetic, math test only
    cal['status']='measured_and_reviewed'
    with pytest.raises(ValueError,match='Assembly angle unresolved'):
        sapien_camera(cal)
    # Explicit preview may inspect the inconsistent candidate, never relabel it as confirmed.
    assert sapien_camera(cal,allow_unverified=True)['name']=='head_camera'
