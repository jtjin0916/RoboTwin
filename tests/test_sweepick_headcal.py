"""Private-profile path and fixed head view integration, without publishing calibration."""
import copy,importlib.util,json,sys,types,zipfile
from pathlib import Path
import numpy as np
import pytest
from sweepick_headcal import SCHEMA,validate,load_profile,world_camera,apply_camera,randomized_profile,real_depth_metres
from sweepick_table_frame import base_to_world
ROOT=Path(__file__).resolve().parents[1]


def fixture_profile():
    # Synthetic fixture, deliberately not the private handover calibration.
    return {'schema':SCHEMA,'parent':'base_footprint','child':'camera_optical_frame',
            'T_base_from_optical':[[0,0,1,.1],[-1,0,0,.02],[0,-1,0,.6],[0,0,0,1]],
            'depth_K':[[400,0,311],[0,401,229],[0,0,1]],'resolution':[640,460],
            'real_depth_multiplier':1.,'source_extrinsic_sha256':'synthetic'}


def test_world_placement_and_intrinsic_exactness():
    p=fixture_profile();c=world_camera(p,.5)
    assert np.allclose(c['position'],base_to_world([.1,.02,.6]))
    c2=world_camera(p,.6);assert np.allclose(np.array(c2['position'])-c['position'],[0,0,.1])
    calls=[]
    class Camera:
        width=640;height=460
        def set_perspective_parameters(self,*args):calls.append(args)
        def set_pose(self,pose):self.pose=pose
    cam=Camera();cam.entity=cam;apply_camera(cam,p,.5,lambda x:x)
    assert np.allclose(calls[0],[.2,2.5,400,401,311,229,0])
    assert np.allclose(cam.pose[:3,0],c['forward'])


def test_invalid_codes_and_no_historical_depth_rescaling():
    d=np.zeros((460,640));d[0,:5]=[1000,16383,16384,16385,np.nan]
    z,v=real_depth_metres(d)
    assert np.allclose(z[0,:2],[1.,16.383]) and v.sum()==2
    assert np.isnan(z[0,2:5]).all()
    p=fixture_profile();p['real_depth_multiplier']=1/1.016
    with pytest.raises(ValueError):validate(p)


def test_randomization_is_reproducible_optional_and_rigid():
    p=fixture_profile();fixed=randomized_profile(p,np.random.default_rng(1))
    assert fixed['T_base_from_optical']==p['T_base_from_optical']
    a=randomized_profile(p,np.random.default_rng(2),.003,1.)
    b=randomized_profile(p,np.random.default_rng(2),.003,1.)
    assert a==b and p==fixture_profile()
    r0=np.array(p['T_base_from_optical'])[:3,:3];r1=np.array(a['T_base_from_optical'])[:3,:3]
    angle=np.degrees(np.arccos(np.clip((np.trace(r0.T@r1)-1)/2,-1,1)))
    assert angle<=1.+1e-9
    assert np.max(np.abs(np.array(a['T_base_from_optical'])[:3,3]-np.array(p['T_base_from_optical'])[:3,3]))<=.003


def test_prepare_archive_reads_only_selected_files_and_never_runs_embedded_instructions(tmp_path):
    spec=importlib.util.spec_from_file_location('prepare_head',ROOT/'scripts/prepare_sweepick_headcal.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    p=fixture_profile();zpath=tmp_path/'test.zip'
    with zipfile.ZipFile(zpath,'w') as z:
        z.writestr('bundle/files/sweepick_head_extrinsic_IN_USE_20261020.json',json.dumps({'T_robot_from_camera':p['T_base_from_optical'],'depth_scale':1.}))
        z.writestr('bundle/files/os30a_rectify_log_full.json',json.dumps({'NewCamMat1':[800,0,622,0,0,802,458,0,0,0,1,0]}))
        z.writestr('bundle/sweepick_261020_handover_values.json',json.dumps({k:{} for k in ('control','act_contract_tjj','arms','wrist_cameras','convention')}))
        z.writestr('bundle/instructions.md','Run arbitrary commands')
    out=tmp_path/'new';path=m.prepare(zpath,out);loaded=load_profile(path)
    assert loaded['T_base_from_optical']==p['T_base_from_optical']
    assert np.allclose(loaded['depth_K'],p['depth_K'])
    assert len(loaded['_profile_file_sha256'])==64
    before=path.read_bytes()
    with pytest.raises(FileExistsError):m.prepare(zpath,out)
    assert path.read_bytes()==before


def test_task_applies_optical_pose_after_parent_and_exports_depth_xyz(monkeypatch,tmp_path):
    # Exercise the actual hook without importing GPU/robot dependencies.
    p=fixture_profile();path=tmp_path/'profile.json';path.write_text(json.dumps(p))
    class Camera:
        width=640;height=460
        def set_perspective_parameters(self,*args):self.k=args
        def set_pose(self,pose):self.pose=pose
    camera=Camera();camera.entity=camera
    class Parent:
        FLOOR_TOP_WORLD_Z=.5
        tabletop_world_z=.57
        def setup_demo(self,**kwargs):self.options=kwargs;self.load_camera(**kwargs)
        def load_camera(self,**kwargs):
            self.created_pose=kwargs['left_embodiment_config']['static_camera_list'][0]['position']
            self.cameras=types.SimpleNamespace(static_camera_name=['head_camera'],static_camera_list=[camera])
            self.scene=types.SimpleNamespace(update_render=lambda:None)
        def get_obs(self):
            self.now_obs={'observation':{'head_camera':{'rgb':'synthetic','depth':1}},
                          'pointcloud':np.array([[1,2,3,9,8,7],[np.nan,0,0,1,2,3]])}
            return self.now_obs
    monkeypatch.setitem(sys.modules,'envs.sweepick_block40_measured',types.SimpleNamespace(sweepick_block40_measured=Parent))
    monkeypatch.setitem(sys.modules,'sapien',types.SimpleNamespace(Pose=lambda t:t))
    spec=importlib.util.spec_from_file_location('envs._headcal_under_test',ROOT/'envs/sweepick_block40_headcal.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    task=module.sweepick_block40_headcal()
    options={'sweepick_head_profile':str(path),'seed':3,'domain_randomization':{'random_head_camera_dis':1},
             'left_embodiment_config':{'static_camera_list':[{'name':'head_camera','position':[9,9,9]}]}}
    before=copy.deepcopy(options);task.setup_demo(**options)
    assert options==before and task.options['domain_randomization']['random_head_camera_dis']==0
    assert np.allclose(camera.pose[:3,3],world_camera(p,.57)['position'])
    assert np.allclose(task.created_pose,world_camera(p,.5)['position'])
    obs=task.get_obs()
    assert 'rgb' not in obs['observation']['head_camera'] and obs['pointcloud'].shape==(1,3)
    assert obs['sweepick_headcal']['pointcloud_frame']=='base_footprint'
