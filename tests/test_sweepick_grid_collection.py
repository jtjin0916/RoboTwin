"""Coverage, split leakage, failure-only logging and resume without a GPU."""
import copy
import json
from pathlib import Path
import h5py
import pytest
from scripts.collect_sweepick_grid_dataset import (open_state, collect, coverage, attempt_seed,
    inspect_episode, reconcile, write_reports)
from sweepick_baseline.contract import canonical_hash

ROOT=Path(__file__).resolve().parents[1]


def manifest():
    return json.loads((ROOT/'docs/dapier/SWEEPICK_GRID_DATASET_SPLITS_20261020.json').read_text())


def fresh(tmp_path, m=None):
    out=tmp_path/'dataset'
    state=open_state(out,m or manifest(),'signature','profile')
    return out,state


def fake_success(split,point,seed,path,state):
    metrics=dict(success=True,lift_success=True,lift_delta_z_m=.05,stow_reached=True,
                 hold_seconds=2.,hold_valid=True,failure_stage='')
    if path is not None:
        path.parent.mkdir(exist_ok=True)
        with h5py.File(path,'x') as f:
            f.attrs.update(complete=True,task_success=True,metrics=json.dumps(metrics),
                metadata=json.dumps(dict(fingerprint=state['fingerprint'],point=point,seed=seed,split=split)))
    return metrics


def test_rectangle_includes_all_historical_failures_and_midpoints_are_disjoint():
    m=manifest();train=m['splits']['train']
    assert [p['grid_id'] for p in train]==list(range(80))
    assert sum(p['prior_lift_successes']==0 for p in train)==18
    assert len(m['splits']['val'])==len(m['splits']['test'])==17
    seen=set()
    for split,ps in m['splits'].items():
        for p in ps:
            xy=tuple(p['xy']);assert xy not in seen;seen.add(xy)
            if split!='train':
                assert len(p['neighbors'])==4
                assert all(train[i]['prior_lift_successes']==3 for i in p['neighbors'])
                assert .15<xy[0]<.259 and -.15<xy[1]<.15
                assert xy==pytest.approx(tuple(sum(train[i]['xy'][j] for i in p['neighbors'])/4 for j in (0,1)))


def test_all_114_points_and_success_only_dataset(tmp_path):
    out,state=fresh(tmp_path)
    summary=collect(out,state,fake_success)
    assert summary['all_points_attempted'] and summary['all_targets_met']
    assert len(list(out.glob('*/episode_*.hdf5')))==114
    assert len(state['attempts'])==80+34*2
    assert len({a['seed'] for a in state['attempts']})==len(state['attempts'])
    assert all(a['stage']=='capture' for a in state['attempts'] if a['split']=='train')
    assert (out/'attempts.csv').read_bytes().startswith(b'\xef\xbb\xbf')
    assert 'grid04' in (out/'dashboard.html').read_text()


def test_resume_skips_completed_and_increases_explicit_target(tmp_path):
    out,state=fresh(tmp_path)
    collect(out,state,fake_success,limit_points=1)
    first=out/'train/episode_grid00_a0000.hdf5';before=first.read_bytes()
    resumed=open_state(out,manifest(),'signature','profile',True)
    collect(out,resumed,fake_success,limit_points=1)
    assert len(list(out.glob('train/*.hdf5')))==2 and first.read_bytes()==before
    assert resumed['attempts'][-1]['point_id']=='grid01'
    expanded=open_state(out,manifest(),'signature','profile',True,target=2)
    collect(out,expanded,fake_success,point_ids=['grid00'])
    assert len(list(out.glob('train/episode_grid00*.hdf5')))==2
    with pytest.raises(ValueError,match='reduce'):
        open_state(out,manifest(),'signature','profile',True,target=1)
    with pytest.raises(ValueError,match='mismatch'):
        open_state(out,manifest(),'different','profile',True)


def test_failures_logged_not_saved_and_retry_uses_new_seed(tmp_path):
    out,state=fresh(tmp_path)
    def fail(split,point,seed,path,state):
        path.parent.mkdir(exist_ok=True);path.write_bytes(b'not successful trajectory')
        return dict(success=False,failure_stage='lift',lift_delta_z_m=0.)
    summary=collect(out,state,fail,point_ids=['grid04'])
    assert not list(out.glob('train/*.hdf5'))
    assert state['attempts'][0]['failure_stage']=='lift'
    assert 'grid04' in (out/'attempts.csv').read_text(encoding='utf-8-sig')
    assert not summary['all_points_attempted']
    old_seed=state['attempts'][0]['seed']
    state=open_state(out,manifest(),'signature','profile',True,max_attempts=2)
    collect(out,state,fake_success,point_ids=['grid04'])
    assert state['attempts'][-1]['seed']!=old_seed
    assert len(list(out.glob('train/*.hdf5')))==1
    assert len(state['attempts'])==2


def test_failed_eval_preflight_never_creates_trajectory(tmp_path):
    out,state=fresh(tmp_path)
    point=manifest()['splits']['val'][0]['id'];calls=[]
    def fail(split,p,seed,path,state):
        calls.append(path);return dict(success=False,failure_stage='grasp')
    collect(out,state,fail,point_ids=[point])
    assert calls==[None] and not list(out.glob('val/*.hdf5'))
    c=next(c for c in coverage(state) if c['point_id']==point)
    assert c['status']=='missing_success' and not c['preflight_ok']


def test_recover_completed_file_after_journal_interruption(tmp_path):
    out,state=fresh(tmp_path)
    split='train';point=manifest()['splits'][split][0];seed=attempt_seed(split,0,'capture',0)
    row=dict(point_id=point['id'],split=split,arm=point['arm'],seed=seed,stage='capture',attempt=0,
             episode='train/episode_grid00_a0000.hdf5',status='running',success=False)
    state['attempts'].append(row)
    fake_success(split,point,seed,out/row['episode'],state)
    reconcile(out,state,inspect_episode)
    assert row['status']=='finished' and row['success']
    collect(out,state,fake_success,point_ids=['grid00'])
    assert len(state['attempts'])==1


def test_unjournaled_existing_file_is_never_deleted(tmp_path):
    out,state=fresh(tmp_path)
    p=out/'train/episode_grid00_a0000.hdf5';p.parent.mkdir();p.write_bytes(b'KEEP')
    with pytest.raises(FileExistsError):collect(out,state,fake_success,point_ids=['grid00'])
    assert p.read_bytes()==b'KEEP'


def test_partial_interrupt_logged_removed_and_new_attempt_id(tmp_path):
    out,state=fresh(tmp_path)
    row=dict(point_id='grid00',split='train',arm='right',seed=110000000,stage='capture',attempt=0,
        episode='train/episode_grid00_a0000.hdf5',status='running',success=False)
    state['attempts'].append(row)
    p=out/row['episode'];p.parent.mkdir();partial=p.with_suffix('.partial.hdf5');partial.write_bytes(b'partial')
    reconcile(out,state,inspect_episode)
    assert row['status']=='interrupted' and not partial.exists()
    collect(out,state,fake_success,point_ids=['grid00'])
    assert state['attempts'][-1]['attempt']==1 and state['attempts'][-1]['seed']!=row['seed']


def test_finished_sweep_can_have_failed_cells_without_claiming_quota_met(tmp_path):
    m=manifest();m['splits']={s:ps[:1] for s,ps in m['splits'].items()}
    out,state=fresh(tmp_path,m)
    def fail(split,p,seed,path,state):
        return dict(success=False,failure_stage='lift')
    summary=collect(out,state,fail)
    assert summary['all_points_attempted'] and summary['sweep_finished']
    assert not summary['all_targets_met'] and not list(out.glob('*/*.hdf5'))
