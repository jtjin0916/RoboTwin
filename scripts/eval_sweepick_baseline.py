#!/usr/bin/env python3
"""Independent simulated ACT feedback execution on fixed unseen Test XY."""
import argparse
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from dapier_act.model_multimodal import DapierACTMultimodal
from dapier_act.pointcloud_preprocess import sample_task_points
from sweepick_baseline.capture import robot_vectors
from sweepick_baseline.contract import SEEDS, canonical_hash, digest, validate_manifest, validate_path
from scripts.collect_sweepick_baseline import fingerprint, runtime_config


def apply_action(task, action):
    a=np.asarray(action,float)
    if a.shape!=(12,) or not np.isfinite(a).all():raise ValueError('Invalid ACT action12')
    # Reject all arms before applying any command; no clipping/wrapping wrist.
    for arm,off in [('left',0),('right',6)]:
        validate_path(a[None,off:off+5],arm,task.arm_limits(arm))
        scale=getattr(task.robot,arm+'_gripper_scale')
        if not min(scale)-1e-6<=a[off+5]<=max(scale)+1e-6:raise ValueError('Gripper command exceeds model scale')
    for arm,off in [('left',0),('right',6)]:
        task.robot.set_arm_joints(a[off:off+5],np.zeros(5),arm)
        lo,hi=getattr(task.robot,arm+'_gripper_scale')
        task.robot.set_gripper(float((a[off+5]-lo)/(hi-lo)),arm,gripper_eps=0.)


def policy_action(model, stats, config, task, device):
    q,_,_=robot_vectors(task.robot);obs=task.get_obs()
    xyz=sample_task_points(obs['pointcloud'],candidate_points=config['candidate_points'],
                          num_points=config['num_point_samples'],device=device)
    inputs=dict(qpos=torch.tensor((q-stats['qpos_mean'])/stats['qpos_std'],device=device)[None],
        pointcloud=[torch.tensor(xyz,device=device)])
    for source,dest in [('left_camera','left_wrist'),('right_camera','right_wrist')]:
        rgb=obs['observation'][source]['rgb']
        if rgb.shape!=(240,320,3):raise ValueError('Wrong wrist image shape')
        inputs[dest]=torch.tensor(np.ascontiguousarray(rgb),device=device).permute(2,0,1).float()[None]/255.
    # No target actions, no expert state/poses: prior z=0, receding horizon.
    with torch.no_grad():prediction,_,_=model(**inputs)
    action=prediction[0,0].cpu().numpy()*stats['action_std']+stats['action_mean']
    return action,q


def rollout(task, model, stats, config, device, seconds):
    task.begin_baseline()
    streak=0;trace=[]
    def monitor():
        nonlocal streak
        dz=float(task.object.get_pose().p[2])-task.initial_z
        task.metrics['lift_success'] |= dz>=.03
        valid=task.metrics['lift_success'] and task.in_stow() and dz>=.03 and task.grasp_contact()
        streak=streak+1 if valid else 0
        task.metrics.update(stow_reached=bool(task.in_stow()),hold_steps=streak,
                            hold_seconds=streak*task.scene.dt,hold_valid=bool(valid))
    task.after_physics_step=monitor
    physics_per_executor=round(.02/task.scene.dt)
    if not math.isclose(physics_per_executor*task.scene.dt,.02,abs_tol=1e-8):
        raise ValueError('50Hz executor not divisible by physics dt')
    for tick in range(math.ceil(seconds*10)):
        a,q=policy_action(model,stats,config,task,device)
        trace.append(dict(t=tick*.1,q=q.tolist(),action=a.tolist(),block_world_xyz=task.object.get_pose().p.tolist()))
        # Re-infer at 10Hz, update command at 50Hz, step physics at actual dt.
        for _ in range(5):
            apply_action(task,a)
            for _ in range(physics_per_executor):task.scene.step()
        if streak*task.scene.dt>=2.:
            task.success=True;break
    task.metrics.update(success=task.success,control_hz=10,executor_hz=50,
                        failure_stage='' if task.success else 'closed_loop_timeout')
    return task.metrics,trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','profile','checkpoint','preflight','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seconds',type=float,default=30)
    a=p.parse_args()
    if a.seconds<=2:p.error('seconds must exceed Hold duration')
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    a.profile=a.profile.expanduser().resolve()
    m=validate_manifest(json.loads(a.manifest.read_text()));evidence=json.loads(a.preflight.read_text())
    sig=fingerprint(m,a.profile)
    if evidence['mode']!='preflight' or evidence['fingerprint']!=sig:raise ValueError('Stale preflight')
    ok={r['point_id'] for r in evidence['attempts'] if r.get('success')}
    if any(p['id'] not in ok for p in m['splits']['test']):raise ValueError('Test points require Expert feasibility evidence')
    # User-generated local checkpoint only; torch checkpoints can contain pickle.
    ckpt=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    if ckpt.get('synthetic_fixture'):raise ValueError('Synthetic checkpoint cannot be used as experiment evidence')
    if canonical_hash(ckpt['manifest'])!=canonical_hash(m):raise ValueError('Checkpoint split manifest mismatch')
    if ckpt['fingerprint']!=sig or ckpt['head_profile_sha256']!=digest(a.profile):
        raise ValueError('Checkpoint calibration/code mismatch')
    model=DapierACTMultimodal(**ckpt['model_config']).to(a.device);model.load_state_dict(ckpt['model']);model.eval()
    a.out.mkdir(parents=True,exist_ok=False)
    from envs.sweepick_block40_act_baseline import sweepick_block40_act_baseline
    cfg=runtime_config(a.profile);report=dict(policy_sha256=digest(a.checkpoint),fingerprint=sig,
                                            inference='ACT prior z=0, first chunk action; closed loop 10Hz',attempts=[])
    for index,point in enumerate(m['splits']['test']):
        task=None;trace=[];row=dict(point_id=point['id'],arm=point['arm'],xy=point['xy'],
                                  seed=SEEDS['test']+2000000+index,success=False)
        try:
            task=sweepick_block40_act_baseline()
            task.setup_demo(now_ep_num=index,seed=row['seed'],baseline_point=point,**cfg)
            metrics,trace=rollout(task,model,ckpt['stats'],ckpt['model_config'],a.device,a.seconds)
            row.update(metrics)
        except Exception as exc:row.update(success=False,error=f'{type(exc).__name__}: {exc}')
        finally:
            if task is not None:
                if hasattr(task,'raw_scene'):task.scene=task.raw_scene
                try:task.close_env(clear_cache=True)
                except Exception as exc:row['close_error']=str(exc)
        (a.out/f'trace_{index:03}.json').write_text(json.dumps(trace,allow_nan=False))
        report['attempts'].append(row)
        (a.out/'closed_loop.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    report.update(successes=sum(r['success'] for r in report['attempts']),total=len(report['attempts']))
    report['success_rate']=report['successes']/report['total']
    report['per_arm']={arm:dict(successes=sum(r['success'] for r in report['attempts'] if r['arm']==arm),
        total=sum(r['arm']==arm for r in report['attempts'])) for arm in ('left','right')}
    (a.out/'closed_loop.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
