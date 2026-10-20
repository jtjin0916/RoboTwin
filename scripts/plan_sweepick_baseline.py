#!/usr/bin/env python3
"""Freeze spatial splits from diagnostic 100x3 CSV, never trajectory data."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sweepick_baseline.contract import SCHEMA, SEEDS, digest, validate_manifest
from sweepick_workspace_near_grid100 import cell


def plan(path):
    groups = defaultdict(list)
    rows = list(csv.DictReader(path.open(newline='', encoding='utf-8-sig')))
    if len(rows) != 300:
        raise ValueError(f'Require original 300 attempts, found {len(rows)}; do not substitute a single scan')
    for r in rows:
        j = int(r['grid_id']); xy = cell(j)
        if (r['arm'] != ('right' if xy[1] < 0 else 'left') or int(r['grid_size']) != 100
                or any(abs(float(r[k]) - v) > 1e-8 for k, v in zip(('x_base_m','y_base_m'), xy))
                or r['lift_success'] not in ('True', 'False')):
            raise ValueError('CSV grid/arm/boolean mismatch')
        groups[j].append(r)
    if set(groups) != set(range(100)):
        raise ValueError('Missing grid IDs')
    for j, rs in groups.items():
        if sorted(int(r['repeat']) for r in rs) != [0, 1, 2]:
            raise ValueError('Require three independent recorded repeats per cell')
        for r in rs:
            if int(r['seed']) != j + 100 * int(r['repeat']):
                raise ValueError('Diagnostic seed mismatch')
            if r['lift_success'] == 'True' and (r['setup_ok'] != 'True' or float(r['lift_delta_z_m']) < .03):
                raise ValueError('Inconsistent lift evidence')
    wins = {j for j, rs in groups.items() if all(r['lift_success']=='True' for r in rs)}
    successes = sum(r['lift_success']=='True' for r in rows)
    zeros = sum(all(r['lift_success']=='False' for r in rs) for rs in groups.values())
    if (successes, len(wins), zeros) != (186, 62, 38):
        raise ValueError('Source does not match accepted 186/300, 62 full-success, 38 full-failure result')
    train = [dict(id=f'grid{j:02}', grid_id=j, xy=list(cell(j)), arm=groups[j][0]['arm'])
             for j in sorted(wins) if cell(j)[0] <= .26]
    # Same-arm neighboring success grid centres -> reserved interstitial XY.
    # Candidates only: require preflight Expert evidence before collection.
    mids = []
    for a in train:
        for b in train:
            if b['grid_id'] - a['grid_id'] != 10 or a['arm'] != b['arm']:
                continue
            xy = [(x+y)/2 for x,y in zip(a['xy'], b['xy'])]
            if any(all(abs(v-w)<1e-8 for v,w in zip(xy,p['xy'])) for p in train+mids):
                continue
            mids.append(dict(id=f"between{a['grid_id']:02}_{b['grid_id']:02}", xy=xy, arm=a['arm'],
                             neighbors=[a['grid_id'], b['grid_id']]))
    val, test = [], []
    for arm in ('left','right'):
        points = [p for p in mids if p['arm']==arm]
        if len(points)<4: raise ValueError('Insufficient interstitial candidates')
        val += points[::2][:2]; test += points[1::2][:2]
    m = dict(schema=SCHEMA, source_csv_sha256=digest(path), source_summary=dict(attempts=300,lift=186,
             success_cells=62,failure_cells=38), yaw_rad=0, table_world_xy=[0,-.25],
             seeds=SEEDS, splits=dict(train=train,val=val,test=test),
             status='XY_RESERVED_EXPERT_PREFLIGHT_REQUIRED')
    return validate_manifest(m)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempts',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    m=plan(a.attempts);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf-8') as f:json.dump(m,f,indent=2,allow_nan=False)
