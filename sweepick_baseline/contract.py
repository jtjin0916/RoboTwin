"""Experiment contract; all angles in model radians, points in base_footprint."""
import hashlib
import json
import numpy as np

PHASES = ('home', 'pregrasp', 'grasp', 'close', 'lift', 'stow', 'hold')
STOW = {'left': [-.0629, -1.6122, 1.5616, -.0184, -1.6210],
        'right': [-.0453, -1.6153, 1.5585, -.0967, -1.5106]}
WRIST_LIMIT = {'left': (-2.744, .957), 'right': (-2.744, .994)}
# Servo mapping from sweepick_261020_start_carry_pose.md. Five-tick end margin.
SERVO = {
 'left': [(726,3406,0.),(772,3321,-.01266),(941,3153,-.1216),(866,3160,-.04603)],
 'right': [(766,3489,0.),(776,3318,-.04872),(932,3162,-.10932),(862,3197,-.06521)]}
REAL_LIMITS = {arm: np.array([[(lo+5-(lo+hi)/2)*2*np.pi/4095+offset,
                             (hi-5-(lo+hi)/2)*2*np.pi/4095+offset]
                            for lo,hi,offset in values]+[list(WRIST_LIMIT[arm])])
               for arm,values in SERVO.items()}
SEEDS = {'train': 110000, 'val': 220000, 'test': 330000}
CROP_MIN = [.13, -.26, .005]
CROP_MAX = [.32, .26, .40]
SCHEMA = 'sweepick-baseline-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_path(positions, arm, limits=None):
    q = np.asarray(positions, dtype=float)
    if q.ndim != 2 or q.shape[1] != 5 or len(q) == 0 or not np.isfinite(q).all():
        raise ValueError('Expected finite nonempty [T,5] joint path')
    lo, hi = WRIST_LIMIT[arm]
    if (q[:, 4] < lo).any() or (q[:, 4] > hi).any():
        raise ValueError(f'{arm} wrist_roll path exceeds real asymmetric command limits')
    bounds = REAL_LIMITS[arm].copy()
    if limits is not None:
        model_bounds = np.asarray(limits, float)
        if model_bounds.shape!=(5,2):raise ValueError('Wrong model limits shape')
        bounds[:,0]=np.maximum(bounds[:,0],model_bounds[:,0])
        bounds[:,1]=np.minimum(bounds[:,1],model_bounds[:,1])
    if bounds is not None:
        if bounds.shape != (5, 2) or (q < bounds[:, 0] - 1e-6).any() or (q > bounds[:, 1] + 1e-6).any():
            raise ValueError(f'{arm} joint path exceeds measured/model limits')
    return q


def validate_manifest(m):
    if m['seeds'] != SEEDS:raise ValueError('Split seeds are fixed independently')
    if m['schema'] != SCHEMA or m['yaw_rad'] != 0 or m['table_world_xy'] != [0, -.25]:
        raise ValueError('Wrong experiment geometry/schema')
    seen, ids = set(), set()
    for split in SEEDS:
        if not m['splits'][split]:
            raise ValueError(f'Empty {split}')
        for p in m['splits'][split]:
            xy = tuple(round(float(v), 8) for v in p['xy'])
            if len(xy) != 2 or not np.isfinite(xy).all() or not .15 <= xy[0] <= .26 or abs(xy[1]) > .15:
                raise ValueError('Point outside baseline workspace')
            if xy in seen or p['id'] in ids:
                raise ValueError('Duplicate XY/ID: spatial leakage')
            if p['arm'] not in STOW:
                raise ValueError('Invalid arm')
            seen.add(xy); ids.add(p['id'])
    return m
