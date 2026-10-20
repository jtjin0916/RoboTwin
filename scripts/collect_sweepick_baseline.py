#!/usr/bin/env python3
"""Bounded Expert preflight/collection. Every attempted point is reported."""
import argparse
import json
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sweepick_baseline.contract import validate_manifest, canonical_hash, digest, SEEDS


def selected_points(manifest, per_arm):
    result=[]
    for split in SEEDS:
        for arm in ('left','right'):
            points=[p for p in manifest['splits'][split] if p['arm']==arm]
            count=per_arm if split=='train' else len(points)
            if len(points)<count:raise ValueError('Insufficient reserved train points')
            # Spread selected train examples over the available success grid.
            indices=[round(i*(len(points)-1)/max(count-1,1)) for i in range(count)]
            result.extend((split,points[i]) for i in indices)
    return result


def fingerprint(manifest, profile):
    files=sorted([*ROOT.glob('sweepick_baseline/*.py'),
        ROOT/'envs/sweepick_block40_act_baseline.py',
        ROOT/'env_cfg/task_config/demo_so101_pgripper_sweepick_act_baseline.yml',
        ROOT/'scripts/collect_sweepick_baseline.py'])
    return canonical_hash(dict(manifest=manifest,profile_sha256=digest(profile),
                               code={str(p.relative_to(ROOT)):digest(p) for p in files}))


def runtime_config(profile):
    from scripts.scan_sweepick_headcal_near_grid100 import prepare_args
    cfg=prepare_args('demo_so101_pgripper_sweepick_act_baseline')
    cfg.update(task_name='sweepick_block40_act_baseline',sweepick_head_profile=str(profile),save_freq=None)
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--per-arm',type=int,choices=range(5,11),default=5)
    p.add_argument('--mode',choices=('preflight','collect'),required=True)
    p.add_argument('--point-id',action='append',help='Bounded diagnostic subset; missing points remain unvalidated')
    p.add_argument('--preflight',type=Path);a=p.parse_args()
    m=validate_manifest(json.loads(a.manifest.read_text()));profile=a.profile.expanduser().resolve()
    from sweepick_headcal import load_profile
    load_profile(profile)
    signature=fingerprint(m,profile);points=selected_points(m,a.per_arm)
    # Stable original indices preserve seeds when running a diagnostic subset.
    indexed_points=list(enumerate(points))
    if a.point_id:
        indexed_points=[(i,item) for i,item in indexed_points if item[1]['id'] in a.point_id]
        if not indexed_points:p.error('No selected point matches --point-id')
    if a.mode=='collect':
        if a.preflight is None:p.error('--preflight report required')
        evidence=json.loads(a.preflight.read_text())
        ok={r['point_id'] for r in evidence['attempts'] if r.get('success')}
        if evidence['fingerprint']!=signature or evidence['mode']!='preflight':
            p.error('Preflight provenance mismatch: rerun Expert preflight')
        if any(point['id'] not in ok for _,point in points):
            p.error('Selected points lack full Expert Stow/Hold preflight success')
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'manifest.json').write_text(json.dumps(m,indent=2))
    report=dict(mode=a.mode,fingerprint=signature,attempts=[],completed=False)
    from envs.sweepick_block40_act_baseline import sweepick_block40_act_baseline
    from sweepick_baseline.capture import EpisodeWriter
    cfg=runtime_config(profile)
    for index,(split,point) in indexed_points:
        task=None;writer=None
        seed=SEEDS[split]+index+(1000000 if a.mode=='preflight' else 0)
        row=dict(point_id=point['id'],split=split,arm=point['arm'],xy=point['xy'],seed=seed,success=False)
        try:
            task=sweepick_block40_act_baseline()
            task.setup_demo(now_ep_num=index,seed=seed,baseline_point=point,**cfg)
            if a.mode=='collect':
                directory=a.out/split;directory.mkdir(exist_ok=True)
                metadata=dict(point=point,split=split,seed=seed,manifest_sha256=canonical_hash(m),
                    fingerprint=signature,head_profile_sha256=digest(profile),headcal=task.headcal,
                    visual_colors=task.visual_colors,dt=float(task.scene.get_timestep()),policy_hz=10,
                    source='simulated_expert',pointcloud_frame='base_footprint')
                writer=EpisodeWriter(directory/f'episode_{index:04}.hdf5',metadata)
            task.begin_baseline(writer);row.update(task.play_once())
        except Exception as exc:
            if task is not None:row.update(getattr(task,'metrics',{}))
            row.update(success=False,error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc()
        finally:
            if writer is not None:writer.close(row['success'],row)
            if task is not None:
                if hasattr(task,'raw_scene'):task.scene=task.raw_scene
                try:task.close_env(clear_cache=True)
                except Exception as exc:row['close_error']=str(exc)
            report['attempts'].append(row)
            (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            print(json.dumps(row,allow_nan=False))
    report['completed']=True
    report['all_success']=all(r['success'] for r in report['attempts'])
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    if not report['all_success']:raise SystemExit(1)

if __name__=='__main__':main()
