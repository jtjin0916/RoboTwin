"""Received OS30A contract checks, CPU only. No live driver or GPU required."""
import copy,json,sys,types
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];TOOL=ROOT/'tools/sweepick_camera_candidate';sys.path.insert(0,str(TOOL))
from received_data import validate_profile,depth_to_m,unproject_depth,camera_config,apply_to_sapien_camera
from runtime import fixed_transforms
from sweepick_table_frame import base_to_world
A=ROOT/'assets/sweepick_camera_candidate/received_20261020'

def profile(name):return json.loads((A/name/'profile.json').read_text())

@pytest.mark.parametrize('name',['in_use','candidate'])
def test_urdf_chain_reproduces_source_optical_matrix(name):
    p=profile(name);_,poses=fixed_transforms(A/name/'sweepick_received.urdf')
    assert np.allclose(poses['camera_optical_frame'],p['source_extrinsic']['T_parent_child'],atol=1e-12)
    assert p['automatically_activated'] is False
    assert 'NOT APPROVED' in profile('candidate')['status']


def test_intrinsic_scaling_preserves_team_pixel_convention_and_offset():
    p=profile('candidate');k=np.array(p['depth_intrinsic_K']);ref=np.array(p['reference_intrinsic_K'])
    assert np.allclose(k,[[409.29998779296875,0,313.14874267578125],[0,409.29998779296875,231.207763671875],[0,0,1]])
    assert k[0,2]!=320 and k[1,2]!=230
    uv=np.array([107.,213.,1.]);assert np.allclose(np.linalg.inv(k)@uv,np.linalg.inv(ref)@(uv*[2,2,1]))


def test_invalid_depth_mask_scale_and_roundtrip_projection():
    d=np.zeros((460,640),np.uint16);d[10,20]=16384;d[21,32]=1016
    z,valid=depth_to_m(d,apply_empirical_scale=True)
    assert valid.sum()==1 and z[21,32]==pytest.approx(1.)
    assert np.isnan(z[10,20])
    raw,_=depth_to_m(d,apply_empirical_scale=False);assert raw[21,32]==pytest.approx(1.016)
    p=profile('candidate');opt,base,v=unproject_depth(d,p,apply_empirical_scale=True)
    k=np.array(p['depth_intrinsic_K']);pixel=k@opt[21,32];assert np.allclose(pixel/pixel[2],[32,21,1])
    t=np.array(p['T_base_from_optical']);assert np.allclose(base[21,32],t[:3,:3]@opt[21,32]+t[:3,3])
    with pytest.raises(TypeError):depth_to_m(d) # caller must choose scale explicitly


@pytest.mark.parametrize('name',['in_use','candidate'])
def test_sapien_world_conversion_and_table_height(name):
    p=profile(name);t=np.array(p['T_base_from_optical']);c=camera_config(p,.5);higher=camera_config(p,.7)
    assert np.allclose(c['position'],base_to_world(t[:3,3]))
    assert np.allclose(np.array(higher['position'])-c['position'],[0,0,.2])
    forward=np.array(c['forward']);left=np.array(c['left'])
    assert np.isclose(forward@left,0) and np.isclose(np.linalg.norm(forward),1)


def test_actual_sapien_adapter_passes_full_intrinsics_without_principal_point_loss(monkeypatch):
    class Camera:
        width=640;height=460
        def set_perspective_parameters(self,*args):self.parameters=args
        def set_pose(self,pose):self.pose=pose
    cam=Camera();cam.entity=cam
    monkeypatch.setitem(sys.modules,'sapien',types.SimpleNamespace(Pose=lambda m:m))
    p=profile('candidate');apply_to_sapien_camera(cam,p)
    assert np.allclose(cam.parameters,[.2,2.5,409.29998779296875,409.29998779296875,313.14874267578125,231.207763671875,0])
    assert np.allclose(cam.pose[:3,3],camera_config(p)['position'])
    cam.width=320
    with pytest.raises(ValueError):apply_to_sapien_camera(cam,p)


def test_rejects_reflections_and_wrong_depth_shape():
    p=copy.deepcopy(profile('candidate'));p['T_base_from_optical'][0][:3]=(-np.array(p['T_base_from_optical'][0][:3])).tolist()
    with pytest.raises(ValueError):validate_profile(p)
    with pytest.raises(ValueError):depth_to_m(np.ones((920,1280)),apply_empirical_scale=True)


def test_received_sample_audit_separates_single_from_median_and_fitted_from_ground_truth():
    report=json.loads((A/'audit.json').read_text())
    assert report['raw_depth']['zeros']==109963
    assert report['raw_depth']['value_16384_pixels']==7
    assert report['raw_depth']['fnv1a64']=='1b804e8ead724ba8'
    assert report['median_depth']['fractional_pixels']==17146
    assert report['scene_uses_in_use_transform']
    assert report['images']['top.png']['equal_rgb_channel_share']==1.


def test_task_adapter_targets_head_only_and_adds_table_height_once(monkeypatch):
    import received_data
    head=object();other=object();calls=[]
    monkeypatch.setattr(received_data,'apply_to_sapien_camera',lambda cam,p,tabletop_world_z:calls.append((cam,tabletop_world_z)))
    task=types.SimpleNamespace(cameras=types.SimpleNamespace(static_camera_name=['other','head_camera'],static_camera_list=[other,head]),tabletop_world_z=.57)
    received_data.apply_to_measured_task(task,profile('candidate'))
    assert calls==[(head,.57)]
