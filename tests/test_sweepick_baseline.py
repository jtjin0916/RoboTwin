"""CPU integration tests. Synthetic fixtures are never grasp-success evidence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
import torch
from sweepick_baseline.contract import STOW, PHASES, validate_path, validate_manifest, canonical_hash
from sweepick_baseline.capture import EpisodeWriter, robot_vectors
from sweepick_baseline.expert import ExpertMixin, StepObserver
from sweepick_baseline.data import episode_splits
from scripts.plan_sweepick_baseline import plan
from scripts.train_sweepick_baseline import prepare_data, train

ROOT=Path(__file__).resolve().parents[1]


def manifest():return json.loads((ROOT/'docs/dapier/SWEEPICK_BASELINE_SPLITS_20261020.json').read_text())


def test_accepted_plan_and_leakage():
    m=validate_manifest(manifest())
    assert m['source_summary']==dict(attempts=300,lift=186,success_cells=62,failure_cells=38)
    assert len(m['splits']['train'])==62
    assert all(p['xy'][0]<=.26 for ps in m['splits'].values() for p in ps)
    assert min(p['xy'][0] for s in ('val','test') for p in m['splits'][s])>.15
    bad=copy.deepcopy(m);bad['splits']['val'][0]['xy']=bad['splits']['train'][0]['xy']
    with pytest.raises(ValueError,match='leakage'):validate_manifest(bad)


@pytest.mark.parametrize('arm,cap',[('left',.957),('right',.994)])
def test_entire_wrist_path_not_just_endpoint(arm,cap):
    path=np.tile(STOW[arm],(3,1));path[1,4]=cap+.001
    with pytest.raises(ValueError,match='asymmetric'):validate_path(path,arm)
    path[1,4]=cap;validate_path(path,arm)
    path[1,4]=-2.745
    with pytest.raises(ValueError):validate_path(path,arm)


def test_reject_short_diagnostic(tmp_path):
    p=tmp_path/'attempts.csv';p.write_text('grid_id\n0\n')
    with pytest.raises(ValueError,match='300 attempts'):plan(p)


def test_observed_q_is_not_drive_or_requested_gripper():
    class Joint:
        def __init__(self,value):self.value=value
        def get_drive_target(self):return [self.value+10]
    class Entity:
        def __init__(self,joints):self.joints=joints
        def get_active_joints(self):return self.joints
        def get_qpos(self):return np.arange(6,dtype=float)
    robot=SimpleNamespace()
    for arm in ('left','right'):
        js=[Joint(i) for i in range(6)]
        setattr(robot,arm+'_entity',Entity(js));setattr(robot,arm+'_arm_joints',js[:5])
        setattr(robot,arm+'_gripper',[(js[5],1,0)]);setattr(robot,'get_'+arm+'_gripper_val',lambda:0.)
    q,a,g=robot_vectors(robot)
    assert q[5]==5 and a[5]==15 and g[0]==0
    assert not np.array_equal(q,a)


def test_hold_advances_physics_and_rejects_drop():
    class Scene:
        def __init__(self):self.steps=0
        def get_timestep(self):return .004
        def step(self):self.steps+=1
    class Task(ExpertMixin):
        def __init__(self,drop):
            self.phase='stow';self.metrics=dict(lift_success=True,stow_reached=True,hold_steps=0,hold_valid=True)
            self.initial_z=.02;self.robot=SimpleNamespace(_entity_qf=lambda e:None,left_entity=None,right_entity=None)
            self.raw=Scene();self.scene=StepObserver(self.raw,self);self.drop=drop
            self.object=SimpleNamespace(get_pose=lambda:SimpleNamespace(p=[0,0,.02 if self.drop and self.raw.steps>250 else .08]))
        def guard_current_commands(self):pass
        def in_stow(self):return True
        def grasp_contact(self):return True
    t=Task(False);assert t.finish_hold()['success'];assert t.raw.steps==500
    t=Task(True);assert not t.finish_hold()['success'];assert t.raw.steps==500


def fixture_data(tmp_path):
    m=manifest();root=tmp_path/'episodes';rng=np.random.default_rng(20)
    for split,shift in [('train',0.),('val',.1)]:
        point=m['splits'][split][0];directory=root/split;directory.mkdir(parents=True)
        meta=dict(point=point,split=split,seed=1,manifest_sha256=canonical_hash(m),fingerprint='test-only',
                  head_profile_sha256='synthetic',source='synthetic_fixture',dt=.004)
        w=EpisodeWriter(directory/'episode_0000.hdf5',meta)
        for i,phase in enumerate(PHASES):
            xyz=rng.uniform([.16,-.1,.01],[.23,.1,.12],(33+i,3)).astype(np.float32)
            q=np.full(12,.1+shift,np.float32)+i*.001;a=q+.02
            obs=dict(pointcloud=xyz,observation=dict(head_camera=dict(depth=np.full((460,640),500.)),
                    left_camera=dict(rgb=np.broadcast_to(np.array([220,30,10],np.uint8),(240,320,3)).copy()),
                    right_camera=dict(rgb=np.full((240,320,3),100,dtype=np.uint8))))
            for k in range(25):w.control_frame((i*25+k)*.004,phase,q,a,[0,1])
            w.frame(i*.1,phase,obs,q,a,[0,1],[0,0,.06])
        w.close(True,dict(success=True,lift_success=True,stow_reached=True,hold_valid=True,hold_seconds=2.))
    # Deliberately invalid Test bytes: training must never open Test trajectories.
    (root/'test').mkdir();(root/'test'/'episode_9999.hdf5').write_bytes(b'DO NOT READ FOR TRAINING')
    return root,m


def test_data_contract_chunk_padding_and_train_only_stats(tmp_path):
    root,m=fixture_data(tmp_path)
    with pytest.raises(ValueError,match='provenance'):episode_splits(root,m)
    ds,stats,splits=prepare_data(root,m,20,16,64,'cpu',True)
    assert np.allclose(stats['qpos_mean'],.103,atol=1e-6)
    last=ds['train'][6]
    assert last['qpos'].shape==(12,) and last['actions'].shape==(20,12)
    assert last['is_pad'].sum()==19
    assert last['pointcloud'].shape==(16,3)
    assert last['left_wrist'].shape==(3,240,320)
    assert 0<=last['left_wrist'].min()<=last['left_wrist'].max()<=1
    assert last['left_wrist'][0].mean()>last['left_wrist'][2].mean()+.5
    assert splits['test']==[]
    # No second World->base transform: sampled metric points retain X>.13.
    assert last['pointcloud'][:,0].min()>=.13
    f=splits['val'][0]
    with h5py.File(f,'r+') as h:h['timestamp'][2]=.9
    with pytest.raises(ValueError,match='timestamps'):prepare_data(root,m,20,16,64,'cpu',True)


def test_real_model_train_validation_and_prior_smoke(tmp_path):
    root,m=fixture_data(tmp_path);mp=tmp_path/'manifest.json';mp.write_text(json.dumps(m))
    a=SimpleNamespace(data=root,manifest=mp,out=tmp_path/'smoke',device='cpu',epochs=1,steps=1,
                      chunk_size=20,compact_smoke=True,synthetic_fixture=True)
    result=train(a)
    assert result['synthetic_fixture'] and not result['closed_loop_evaluated']
    assert result['history'][0]['optimizer_steps']==1
    assert np.isfinite(result['history'][0]['val_prior_l1'])
    assert (a.out/'policy_smoke.pt').is_file()
    with pytest.raises(FileExistsError):train(a)


def test_closed_loop_uses_feedback_and_50hz_executor_without_expert(monkeypatch):
    import scripts.eval_sweepick_baseline as ev
    calls={'policy':0,'command':0}
    def infer(*args):calls['policy']+=1;return np.zeros(12),np.zeros(12)
    def apply(*args):calls['command']+=1
    monkeypatch.setattr(ev,'policy_action',infer);monkeypatch.setattr(ev,'apply_action',apply)
    class Task:
        initial_z=0.;success=False
        object=SimpleNamespace(get_pose=lambda:SimpleNamespace(p=np.array([0,0,.04])))
        def __init__(self):
            self.scene=SimpleNamespace(dt=.004,step=self.step)
            self.steps=0
        def begin_baseline(self):self.metrics=dict(lift_success=False)
        def step(self):self.steps+=1;self.after_physics_step()
        def in_stow(self):return True
        def grasp_contact(self):return True
        def play_once(self):raise AssertionError('Expert called during ACT evaluation')
    t=Task();metrics,trace=ev.rollout(t,None,None,None,'cpu',3)
    assert metrics['success'] and t.steps==500
    assert calls==dict(policy=20,command=100) and len(trace)==20


def test_midpath_measured_joint_rejection():
    path=np.tile(STOW['left'],(3,1));path[1,2]=2.
    with pytest.raises(ValueError,match='measured/model'):validate_path(path,'left')


def test_failed_episodes_preserved_but_never_trained(tmp_path):
    root,m=fixture_data(tmp_path)
    path=root/'train'/'episode_0001.hdf5'
    w=EpisodeWriter(path,{'not_training':'failure audit'})
    w.close(False,{'error':'fixture failure'})
    splits=episode_splits(root,m,allow_synthetic=True)
    assert path.exists() and path not in splits['train']
    with pytest.raises(FileExistsError):EpisodeWriter(path,{})
