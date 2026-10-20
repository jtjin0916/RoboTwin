"""Strict episode provenance gate before reuse of the existing ACT loader."""
import json
from pathlib import Path
import h5py
import numpy as np
from .contract import SCHEMA, PHASES, canonical_hash, validate_manifest, CROP_MIN, CROP_MAX
from dapier_act.dataset import concat_joint_group, get_pointcloud_length, read_pointcloud_frame
from dapier_act.dataset_multimodal import decode_rgb_frame


def episode_splits(directory, manifest, allow_synthetic=False):
    validate_manifest(manifest)
    result={s:[] for s in ('train','val','test')}
    provenance=set()
    train_counts={'left':0,'right':0}
    for split in ('train','val'):
        allowed={p['id']:p for p in manifest['splits'][split]}
        for path in sorted((Path(directory)/split).glob('episode_*.hdf5')):
            if '.partial.' in path.name:continue
            with h5py.File(path,'r') as f:
                if f.attrs.get('schema')!=SCHEMA or not f.attrs.get('complete',False):
                    raise ValueError(f'Incomplete/wrong schema: {path}')
                if not f.attrs.get('task_success',False):continue
                meta=json.loads(f.attrs['metadata']);point=meta['point']
                if (meta['split']!=split or point!=allowed.get(point['id']) or
                    meta['manifest_sha256']!=canonical_hash(manifest) or
                    meta['source'] not in (('simulated_expert','synthetic_fixture') if allow_synthetic else ('simulated_expert',))):
                    raise ValueError('Episode provenance/split/XY mismatch')
                provenance.add((meta['fingerprint'],meta['head_profile_sha256']))
                if f.attrs['pointcloud_frame']!='base_footprint' or f.attrs['gripper_units']!='motor_joint_rad':
                    raise ValueError('Wrong geometry/action units')
                q=concat_joint_group(f,'state');a=concat_joint_group(f,'action')
                n=len(q);pcn=get_pointcloud_length(f)
                keys=['timestamp','phase','vision/head_depth_m','gripper/requested_robotwin_normalized',
                      'vision/cam_left_wrist/colors','vision/cam_right_wrist/colors','block_world_xyz','vision/head_depth_valid']
                if f['vision/head_depth_m'].shape[1:]!=(460,640):raise ValueError('Wrong head depth resolution')
                if n<2 or q.shape!=(n,12) or a.shape!=(n,12) or pcn!=n or any(len(f[k])!=n for k in keys):
                    raise ValueError('Temporal alignment/action-state dimension mismatch')
                if not np.isfinite([q,a]).all():raise ValueError('Nonfinite joint data')
                if not np.allclose(np.diff(f['timestamp'][:]),.1,atol=1e-6):
                    raise ValueError('Expected uniform simulated 10Hz sample timestamps')
                ct=np.asarray(f['control/timestamp']);cq=np.asarray(f['control/q']);ca=np.asarray(f['control/action'])
                if meta['dt']<=0 or not np.allclose(np.diff(ct),meta['dt'],atol=1e-7):
                    raise ValueError('Invalid physics-rate command timestamps')
                stride=round(.1/meta['dt'])
                if not np.allclose(ct[::stride],f['timestamp'][:],atol=1e-6):
                    raise ValueError('Control/sensor timestamp alignment mismatch')
                if not np.array_equal(cq[::stride],q) or not np.array_equal(ca[::stride],a):
                    raise ValueError('Control/sensor state-action alignment mismatch')
                if not np.isfinite([cq,ca]).all():raise ValueError('Nonfinite dense command trace')
                phases=[v.decode() if isinstance(v,bytes) else v for v in f['phase'][:]]
                if (any(p not in PHASES for p in phases) or list(dict.fromkeys(phases))!=list(PHASES)
                    or any(PHASES.index(a)>PHASES.index(b) for a,b in zip(phases,phases[1:]))):
                    raise ValueError('Missing or out-of-order Expert phases')
                metrics=json.loads(f.attrs['metrics'])
                if not (metrics['success'] and metrics['hold_valid'] and metrics['stow_reached']
                        and metrics['lift_success'] and metrics['hold_seconds']>=2.):
                    raise ValueError('Insufficient Expert Stow/Hold evidence')
                if any('head' in k and k not in ('head_depth_m','head_depth_valid') for k in f['vision']):
                    raise ValueError('Unexpected head modality')
                for t in range(n):
                    xyz=read_pointcloud_frame(f,t)
                    if not len(xyz) or not np.isfinite(xyz).all():raise ValueError('Empty/nonfinite PCD')
                    if (xyz<np.array(CROP_MIN)-1e-6).any() or (xyz>np.array(CROP_MAX)+1e-6).any():
                        raise ValueError('PCD outside explicit base crop')
                    for cam in ('cam_left_wrist','cam_right_wrist'):
                        if decode_rgb_frame(f,cam,t).shape!=(240,320,3):raise ValueError('Wrong wrist resolution')
            result[split].append(path)
            if split=='train':train_counts[point['arm']]+=1
    if not allow_synthetic and any(n<5 or n>10 for n in train_counts.values()):
        raise ValueError('Require 5..10 successful Train trajectories per arm for this baseline')
    if len(provenance)>1:raise ValueError('Mixed calibration/code fingerprints')
    if not result['train'] or not result['val']:raise ValueError('Need successful independent Train and Val episodes')
    return result
