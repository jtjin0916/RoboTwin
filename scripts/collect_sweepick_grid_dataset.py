#!/usr/bin/env python3
"""Resumable, success-only full-grid Expert collection with a complete attempt log.

Train attempts every rectangle cell, including historic failures. Val/Test use
separate fixed interstitial XY and require successful full-sequence preflight.
No ACT inference or training is involved in this collection command.
"""
import argparse
import csv
import html
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sweepick_baseline.contract import canonical_hash, digest, validate_manifest, SEEDS

SCHEMA = 'sweepick.grid-collection.v1'
FIELDS = ('point_id', 'split', 'arm', 'x_m', 'y_m', 'stage', 'attempt', 'seed',
          'status', 'success', 'failure_stage', 'error', 'lift_delta_z_m',
          'stow_reached', 'hold_seconds', 'hold_valid', 'episode', 'started_at', 'finished_at')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def points(manifest):
    return [(split, point) for split in SEEDS for point in manifest['splits'][split]]


def runtime_signature(manifest, profile, renderer="raster"):
    # The physical/capture implementation is pinned independently of ACT training.
    names = ['sweepick_baseline/contract.py', 'sweepick_baseline/capture.py',
             'sweepick_baseline/expert.py', 'envs/sweepick_block40_act_baseline.py',
             'envs/sweepick_block40_headcal.py', 'envs/sweepick_block40_measured.py',
             'envs/sweepick_block40.py', 'envs/_base_task.py', 'envs/robot/robot.py',
             'envs/robot/planner.py', 'envs/camera/camera.py', 'sweepick_headcal.py',
             'sweepick_table_frame.py', 'scripts/collect_sweepick_grid_dataset.py',
             'env_cfg/task_config/demo_so101_pgripper_sweepick_act_baseline.yml',
             'assets/embodiments/so101-pgripper-sweepick-measured/config.yml',
             'assets/embodiments/so101-pgripper-sweepick-measured/curobo.yml']
    # Runtime assets matter too. A profile/URDF change cannot silently resume.
    urdf = os.environ.get('SWEEPICK_TURTLEBOT_URDF')
    model = ROOT/'assets/embodiments/so101-pgripper/so101_pgripper_sweepick_unit.urdf'
    return canonical_hash(dict(manifest=manifest, profile=digest(profile), renderer=renderer,
        files={name: digest(ROOT/name) for name in names},
        measured_model=digest(model) if model.is_file() else 'MISSING',
        turtlebot=digest(Path(urdf).expanduser()) if urdf and Path(urdf).expanduser().is_file() else 'MISSING'))


def attempt_seed(split, ordinal, stage, attempt):
    return SEEDS[split]*1000 + ordinal*10000 + attempt + (1000000000 if stage == 'preflight' else 0)


def coverage(state):
    result = []
    for split, point in points(state['manifest']):
        rows = [a for a in state['attempts'] if a['point_id'] == point['id']]
        captures = [a for a in rows if a['stage'] == 'capture']
        successes = sum(a['status'] == 'finished' and a.get('success', False) for a in captures)
        preflight_ok = split == 'train' or any(a['stage'] == 'preflight' and a.get('success') for a in rows)
        tried = any(a['status'] == 'finished' for a in rows)
        finished_captures=sum(a['status']=='finished' for a in captures)
        finished_preflights=sum(a['status']=='finished' and a['stage']=='preflight' for a in rows)
        exhausted=(not preflight_ok and finished_preflights>=state['max_attempts_per_point']) or (
            preflight_ok and finished_captures>=state['target_per_point']*state['max_attempts_per_point'])
        if successes >= state['target_per_point']:
            status = 'complete'
        elif any(a['status'] == 'running' for a in rows):
            status = 'running'
        elif tried:
            status = 'missing_success'
        else:
            status = 'not_attempted'
        result.append(dict(point_id=point['id'], split=split, arm=point['arm'], xy=point['xy'],
                           prior_lift_successes=point.get('prior_lift_successes'),
                           attempted=tried, successes=successes, target=state['target_per_point'],
                           preflight_ok=preflight_ok, attempts=len(rows), budget_exhausted=exhausted, status=status))
    return result


def write_reports(out, state):
    cells = coverage(state)
    summary = dict(schema=SCHEMA, fingerprint=state['fingerprint'], updated_at=utc_now(),
                   target_per_point=state['target_per_point'], points=cells, splits={},
                   all_points_attempted=all(p['attempted'] for p in cells),
                   all_targets_met=all(p['status'] == 'complete' for p in cells),
                   sweep_finished=all(p['status']=='complete' or p['budget_exhausted'] for p in cells))
    for split in SEEDS:
        subset = [c for c in cells if c['split'] == split]
        summary['splits'][split] = dict(points=len(subset), attempted=sum(c['attempted'] for c in subset),
            covered=sum(c['status'] == 'complete' for c in subset),
            successful_episodes=sum(c['successes'] for c in subset),
            missing=[c['point_id'] for c in subset if c['status'] != 'complete'])
    atomic_json(out/'state.json', state)
    atomic_json(out/'coverage.json', summary)
    with (out/'attempts.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(state['attempts'])
    with (out/'coverage.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['split','point_id','arm','x_m','y_m','attempted','successes','target','status','preflight_ok'])
        for c in cells:
            writer.writerow([c['split'],c['point_id'],c['arm'],*c['xy'],c['attempted'],c['successes'],
                             c['target'],c['status'],c['preflight_ok']])
    render_dashboard(out, state, summary)
    return summary


def event(out, kind, row):
    with (out/'attempts.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(event=kind, logged_at=utc_now(), **row), ensure_ascii=False, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def render_dashboard(out, state, summary):
    esc = html.escape
    colors = {'complete':'#bde6d1','running':'#ffedb0','not_attempted':'#f4f6f9','missing_success':'#f4caca'}
    lookup = {c['point_id']:c for c in summary['points']}
    grid = '<table><tr><th>X cm / Y cm</th>'+''.join(f'<th>{15-j*30/9:+.1f}</th>' for j in range(10))+'</tr>'
    for i in range(7,-1,-1):
        grid += f'<tr><th>{15+i*14/9:.1f}</th>'
        for j in range(9,-1,-1):
            c = lookup.get(f'grid{i*10+j:02}')
            if c:
                grid += f'<td style="background:{colors[c["status"]]}">{c["point_id"]}<br>{c["successes"]}/{c["target"]}<br>{esc(c["status"])}</td>'
            else:
                grid += '<td>—</td>'
        grid += '</tr>'
    grid += '</table>'
    rows = ''
    for a in reversed(state['attempts']):
        rows += '<tr>'+''.join(f'<td>{esc(str(a.get(k,"")))}</td>' for k in
            ('point_id','split','arm','stage','seed','status','success','failure_stage','error'))+'</tr>'
    evaluations = '<table><tr><th>split</th><th>point</th><th>XY m</th><th>success/target</th><th>status</th></tr>'
    for c in summary['points']:
        if c['split'] != 'train':
            evaluations += f'<tr style="background:{colors[c["status"]]}"><td>{c["split"]}</td><td>{c["point_id"]}</td><td>{esc(str(c["xy"]))}</td><td>{c["successes"]}/{c["target"]}</td><td>{c["status"]}</td></tr>'
    evaluations += '</table>'
    document = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="refresh" content="15">
<title>SWEEPICK 전체 그리드 수집</title><style>body{{font:15px system-ui;margin:24px;color:#243244}}table{{border-collapse:collapse;margin:20px 0}}td,th{{border:1px solid #d7dfe5;padding:7px;font-size:12px}}th{{background:#eef2f6}}h1{{font-size:25px}}.summary{{white-space:pre-wrap}}a{{color:#1766ad}}</style>
<h1>SWEEPICK 전체 그리드 Expert 수집</h1><p>성공 궤적만 HDF5 저장 · 실패는 로그만 보존 · ACT 학습/평가는 수행하지 않음</p>
<p>녹색: 목표 충족 / 노랑: 진행 중 / 빨강: 시도했으나 성공 부족 / 회색: 미시도. 15초마다 갱신.</p>
<pre class="summary">{esc(json.dumps(summary['splits'],ensure_ascii=False,indent=2))}</pre>
<p>순회 완료: {summary['sweep_finished']} / 전 지점 시도: {summary['all_points_attempted']} / 전 지점 성공 목표 충족: {summary['all_targets_met']}</p>
<h2>Train 사각영역 — 로봇 전방 +X ↑</h2>{grid}<h2>그리드 사이 Val / Test</h2>{evaluations}
<h2>시도 로그 (최신순)</h2><table><tr><th>지점</th><th>split</th><th>팔</th><th>단계</th><th>seed</th><th>상태</th><th>성공</th><th>실패 단계</th><th>오류</th></tr>{rows}</table>
<p><a href="attempts.csv">시도 CSV</a> · <a href="coverage.csv">지점별 CSV</a> · <a href="coverage.json">현재 집계 JSON</a> · <a href="attempts.jsonl">전체 이벤트 JSONL</a></p></html>'''
    temp = out/'dashboard.html.tmp'
    temp.write_text(document, encoding='utf-8')
    temp.replace(out/'dashboard.html')


def open_state(out, manifest, signature, profile_sha, resume=False, target=None, max_attempts=None):
    if resume:
        state = json.loads((out/'state.json').read_text())
        if state['schema'] != SCHEMA or state['fingerprint'] != signature or state['manifest'] != manifest:
            raise ValueError('Resume manifest/runtime/calibration mismatch; use a new output directory')
        new_target = state['target_per_point'] if target is None else target
        new_max = state['max_attempts_per_point'] if max_attempts is None else max_attempts
        if new_target < state['target_per_point'] or new_max < state['max_attempts_per_point']:
            raise ValueError('Cannot reduce an existing target or attempt budget')
        if (new_target, new_max) != (state['target_per_point'], state['max_attempts_per_point']):
            state['settings_history'].append(dict(at=utc_now(),target=new_target,max_attempts=new_max))
        state.update(target_per_point=new_target,max_attempts_per_point=new_max)
    else:
        out.mkdir(parents=True, exist_ok=False)
        state = dict(schema=SCHEMA, fingerprint=signature, head_profile_sha256=profile_sha,
            manifest=manifest, target_per_point=target or 1, max_attempts_per_point=max_attempts or 1,
            attempts=[], settings_history=[], created_at=utc_now())
        atomic_json(out/'manifest.json', manifest)
    return state


def reconcile(out, state, inspect_episode):
    for row in state['attempts']:
        if row['status'] != 'running':
            if row.get('success') and row.get('episode'):
                if not (out/row['episode']).is_file():
                    raise ValueError('A recorded successful episode is missing; do not silently count/recollect it')
                if not inspect_episode(out/row['episode'], state, row).get('success'):
                    raise ValueError('A recorded success is no longer valid')
            continue
        path = out/row['episode'] if row.get('episode') else None
        if path and path.is_file():
            metrics = inspect_episode(path, state, row)
            row.update(metrics, status='interrupted' if metrics.get('failure_stage')=='interrupted' else 'finished', finished_at=utc_now())
            if not row.get('success'):
                path.unlink()
                row['episode'] = ''
        else:
            row.update(status='interrupted', success=False, failure_stage='interrupted', finished_at=utc_now())
            if path:
                partial = path.with_suffix('.partial.hdf5')
                if partial.is_file():
                    partial.unlink()  # Only the file reserved by this new cohort's own journal.
            row['episode'] = ''
        event(out, 'reconciled', row)


def inspect_episode(path, state, row):
    import h5py
    with h5py.File(path, 'r') as f:
        meta = json.loads(f.attrs['metadata'])
        if (not f.attrs.get('complete') or meta['fingerprint'] != state['fingerprint'] or
            meta['point']['id'] != row['point_id'] or meta['seed'] != row['seed'] or meta['split'] != row['split']):
            raise ValueError('Recovered episode provenance mismatch')
        metrics = json.loads(f.attrs['metrics'])
        if bool(f.attrs['task_success']) != bool(metrics.get('success')):
            raise ValueError('Recovered episode success mismatch')
        return metrics


def collect(out, state, attempt_fn, selected_splits=('train','val','test'), point_ids=None,
            limit_points=None, inspect_fn=inspect_episode):
    reconcile(out, state, inspect_fn)
    write_reports(out, state)
    processed = 0
    for ordinal, (split, point) in enumerate(points(state['manifest'])):
        if split not in selected_splits or (point_ids and point['id'] not in point_ids):
            continue
        own = lambda stage: [a for a in state['attempts'] if a['point_id'] == point['id'] and a['stage'] == stage]
        count = lambda: sum(a.get('success',False) and a['status']=='finished' for a in own('capture'))
        if count() >= state['target_per_point']:
            continue
        if limit_points is not None and processed >= limit_points:
            break
        processed += 1
        # Each Train cell is attempted directly, regardless of historical result.
        # Val/Test feasibility is checked independently before recording training-ready data.
        stages = ['capture'] if split == 'train' else ['preflight','capture']
        for stage in stages:
            if stage == 'preflight' and any(a.get('success') for a in own(stage)):
                continue
            if stage == 'capture' and split != 'train' and not any(a.get('success') for a in own('preflight')):
                break
            budget = state['max_attempts_per_point'] * (state['target_per_point'] if stage == 'capture' else 1)
            while sum(a['status']=='finished' for a in own(stage)) < budget:
                if stage == 'capture' and count() >= state['target_per_point']:
                    break
                number = len(own(stage))
                seed = attempt_seed(split, ordinal, stage, number)
                relative = f'{split}/episode_{point["id"]}_a{number:04}.hdf5' if stage == 'capture' else ''
                if relative and ((out/relative).exists() or (out/relative).with_suffix('.partial.hdf5').exists()):
                    raise FileExistsError('Unjournaled episode path already exists; refusing to overwrite/delete it')
                row = dict(point_id=point['id'], split=split, arm=point['arm'], x_m=point['xy'][0],y_m=point['xy'][1],
                    stage=stage, attempt=number, seed=seed, episode=relative, status='running', success=False,
                    started_at=utc_now())
                state['attempts'].append(row)
                write_reports(out, state)
                event(out, 'started', row)
                try:
                    metrics = attempt_fn(split, point, seed, out/relative if relative else None, state)
                    row.update(metrics)
                    row['status'] = 'finished'
                    if relative and row.get('success'):
                        recovered = inspect_fn(out/relative, state, row)
                        if not recovered.get('success'):
                            raise ValueError('Attempt returned success without a successful episode')
                    if relative and not row.get('success'):
                        path = out/relative
                        if path.exists(): path.unlink()
                        partial = path.with_suffix('.partial.hdf5')
                        if partial.exists(): partial.unlink()
                        row['episode'] = ''
                except Exception as exc:
                    row.update(status='finished',success=False,error=f'{type(exc).__name__}: {exc}',
                               failure_stage=row.get('failure_stage') or 'runtime')
                    if relative:
                        for p in (out/relative,(out/relative).with_suffix('.partial.hdf5')):
                            if p.exists(): p.unlink()
                    row['episode'] = ''
                row['finished_at'] = utc_now()
                event(out, 'finished', row)
                write_reports(out, state)
                print(json.dumps(row,ensure_ascii=False,allow_nan=False), flush=True)
                if stage == 'preflight' and row.get('success'):
                    break
    return write_reports(out, state)


def real_attempt(profile, renderer="raster"):
    from scripts.collect_sweepick_baseline import runtime_config
    from envs.sweepick_block40_act_baseline import sweepick_block40_act_baseline
    from sweepick_baseline.capture import EpisodeWriter
    cfg = runtime_config(profile)
    class GridTask(sweepick_block40_act_baseline):
        def setup_scene(self, **kwargs):
            super().setup_scene(**kwargs)
            if renderer == 'raster':
                import sapien
                # Future cameras use raster shaders; physics/calibration unchanged.
                sapien.render.set_camera_shader_dir('default')
    def run(split, point, seed, path, state):
        task = None
        writer = None
        metrics = dict(success=False,failure_stage='setup')
        try:
            task = GridTask()
            task.setup_demo(now_ep_num=seed,seed=seed,baseline_point=point,**cfg)
            if path is not None:
                path.parent.mkdir(exist_ok=True)
                metadata = dict(point=point,split=split,seed=seed,
                    manifest_sha256=canonical_hash(state['manifest']),fingerprint=state['fingerprint'],
                    head_profile_sha256=state['head_profile_sha256'],headcal=task.headcal,
                    visual_colors=task.visual_colors,dt=float(task.scene.get_timestep()),policy_hz=10,
                    source='simulated_expert',pointcloud_frame='base_footprint',collection_schema=SCHEMA,render_backend=renderer)
                writer = EpisodeWriter(path,metadata)
            task.begin_baseline(writer)
            metrics.update(task.play_once())
        except KeyboardInterrupt:
            metrics.update(success=False,failure_stage='interrupted',error='KeyboardInterrupt')
            raise
        except Exception as exc:
            if task is not None: metrics.update(getattr(task,'metrics',{}))
            metrics.update(success=False,error=f'{type(exc).__name__}: {exc}')
        finally:
            if writer is not None: writer.close(bool(metrics.get('success')), metrics)
            if task is not None:
                if hasattr(task,'raw_scene'): task.scene = task.raw_scene
                try: task.close_env(clear_cache=True)
                except Exception as exc: metrics['close_error'] = str(exc)
        return metrics
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--profile',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--renderer',choices=('raster','raytrace'),default='raster')
    p.add_argument('--target-per-point',type=int)
    p.add_argument('--max-attempts-per-point',type=int)
    p.add_argument('--splits',nargs='+',choices=tuple(SEEDS),default=list(SEEDS))
    p.add_argument('--point-id',action='append')
    p.add_argument('--limit-points',type=int)
    p.add_argument('--plan-only',action='store_true')
    args = p.parse_args()
    m = validate_manifest(json.loads(args.manifest.read_text()))
    if m.get('collection_policy') != 'full_rectangle_all_cells_success_only':
        p.error('Use the full-rectangle manifest, not the historical 5-per-arm smoke plan')
    valid = {point['id'] for _,point in points(m)}
    if any(not re.fullmatch(r'[A-Za-z0-9_]+', key) for key in valid): p.error('Unsafe point ID')
    if args.point_id and not set(args.point_id) <= valid: p.error('Unknown --point-id')
    for value,name,maximum in [(args.target_per_point,'target-per-point',100),
                               (args.max_attempts_per_point,'max-attempts-per-point',20)]:
        if value is not None and not 1 <= value <= maximum: p.error(f'{name} must be 1..{maximum}')
    if args.limit_points is not None and args.limit_points<1: p.error('limit-points must be positive')
    if args.plan_only:
        print(json.dumps(dict(counts={s:len(v) for s,v in m['splits'].items()},
            points=[dict(split=s,**point) for s,point in points(m)]),indent=2))
        return
    if not args.profile or not args.out: p.error('--profile and --out are required to collect')
    profile = args.profile.expanduser().resolve()
    from sweepick_headcal import load_profile
    load_profile(profile)
    signature = runtime_signature(m,profile,args.renderer)
    args.out = args.out.expanduser().resolve()
    # Prevent two processes writing the same cohort. flock releases on crash.
    import fcntl
    args.out.parent.mkdir(parents=True,exist_ok=True)
    lock_path = args.out.parent/(args.out.name+'.lock')
    with lock_path.open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: p.error('Another collector is already using this output directory')
        state = open_state(args.out,m,signature,digest(profile),args.resume,args.target_per_point,args.max_attempts_per_point)
        summary = collect(args.out,state,real_attempt(profile,args.renderer),args.splits,args.point_id,args.limit_points)
    if not summary['all_targets_met']:
        print('Some locations have no successful trajectory; inspect coverage.csv/dashboard.html. Failures were not saved as data.',flush=True)
    if not summary['sweep_finished']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
