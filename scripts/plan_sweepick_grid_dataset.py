#!/usr/bin/env python3
"""Freeze the user's full rectangle and interstitial Val/Test positions."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plan_sweepick_baseline import plan
from sweepick_workspace_near_grid100 import cell
from sweepick_baseline.contract import validate_manifest


def rectangle_plan(attempts):
    previous = plan(attempts)  # Verifies the actual 300-row, 186/300 source.
    winners = {p['grid_id'] for p in previous['splits']['train']}
    train = []
    for grid_id in range(80):
        x, y = cell(grid_id)
        train.append(dict(id=f'grid{grid_id:02}', grid_id=grid_id, xy=[x, y],
                          arm='left' if y >= 0 else 'right',
                          prior_lift_successes=3 if grid_id in winners else 0))
    evaluation = []
    for i in range(7):
        for j in range(9):
            corners = [i*10+j, i*10+j+1, (i+1)*10+j, (i+1)*10+j+1]
            if not all(g in winners for g in corners):
                continue
            a, b = cell(corners[0]), cell(corners[-1])
            xy = [(a[0]+b[0])/2, (a[1]+b[1])/2]
            evaluation.append(dict(id=f'mid{i:02}_{j:02}', xy=xy, neighbors=corners,
                                   arm='left' if xy[1] >= 0 else 'right'))
    val, test = [], []
    for arm in ('left', 'right'):
        candidates = [p for p in evaluation if p['arm'] == arm]
        val.extend(candidates[::2])
        test.extend(candidates[1::2])
    result = dict(previous, splits=dict(train=train, val=val, test=test),
                  collection_policy='full_rectangle_all_cells_success_only',
                  status='ALL_RECTANGLE_CELLS_TO_ATTEMPT_EVAL_REQUIRES_PREFLIGHT',
                  evaluation_rule='2D cell centers; all four corners historical 3/3; disjoint fixed Val/Test')
    return validate_manifest(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = rectangle_plan(args.attempts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    print({s: len(points) for s, points in manifest['splits'].items()})
