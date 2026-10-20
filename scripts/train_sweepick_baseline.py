#!/usr/bin/env python3
"""Explicit spatial splits; actual multimodal ACT one/multi-step smoke training."""
import argparse
import json
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from dapier_act.dataset import compute_qpos_action_stats, read_pointcloud_frame
from dapier_act.dataset_multimodal import DapierMultimodalDataset, multimodal_collate_fn
from dapier_act.pointcloud_preprocess import sample_task_points
from dapier_act.model_multimodal import DapierACTMultimodal
from dapier_act.model import dapier_act_loss
from sweepick_baseline.data import episode_splits
from sweepick_baseline.contract import canonical_hash
import h5py


def prepare_data(directory, manifest, chunk, points, candidates, device, allow_synthetic=False):
    splits=episode_splits(directory,manifest,allow_synthetic)
    stats=compute_qpos_action_stats(directory,episode_files=splits['train'])
    cache={};mins=[];maxs=[]
    # Test trajectories are excluded from normalization, cache and loaders.
    for split in ('train','val'):
        for path in splits[split]:
            with h5py.File(path,'r') as f:
                for t in range(len(f['timestamp'])):
                    xyz=sample_task_points(read_pointcloud_frame(f,t),candidate_points=candidates,
                                           num_points=points,device=device)
                    cache[(str(path),t)]=np.asarray(xyz,np.float32)
                    if split=='train':mins.append(xyz.min(0));maxs.append(xyz.max(0))
    stats['point_min']=np.min(mins,axis=0);stats['point_max']=np.max(maxs,axis=0)
    if np.any(stats['point_max']-stats['point_min']<1e-4):raise ValueError('Degenerate train PCD limits')
    datasets={s:DapierMultimodalDataset(directory,stats,chunk_size=chunk,episode_files=splits[s],
        normalize_pointcloud=False,use_task_crop=False,pointcloud_cache=cache) for s in ('train','val')}
    return datasets,stats,splits


def inputs(batch,device):
    return {k:([v.to(device) for v in value] if k=='pointcloud' else value.to(device))
            for k,value in batch.items()}


def losses_for(model,batch):
    pred,mu,logvar=model(**{k:batch[k] for k in ('qpos','pointcloud','left_wrist','right_wrist','actions','is_pad')})
    return dapier_act_loss(pred,batch['actions'],batch['is_pad'],mu,logvar,kl_weight=.01)


def train(args):
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(2)
    device=args.device
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
    manifest=json.loads(args.manifest.read_text())
    npoints=16 if args.compact_smoke else 512;candidates=64 if args.compact_smoke else 8192
    datasets,stats,splits=prepare_data(args.data,manifest,args.chunk_size,npoints,candidates,device,args.synthetic_fixture)
    loaders={s:DataLoader(datasets[s],batch_size=1,shuffle=(s=='train'),num_workers=0,
                         collate_fn=multimodal_collate_fn) for s in datasets}
    config=dict(qpos_dim=12,action_dim=12,chunk_size=args.chunk_size,hidden_dim=32 if args.compact_smoke else 256,
        latent_dim=32,num_point_samples=npoints,candidate_points=candidates,point_min=stats['point_min'].tolist(),
        point_max=stats['point_max'].tolist(),nheads=4 if args.compact_smoke else 8,
        encoder_layers=1 if args.compact_smoke else 4,decoder_layers=1 if args.compact_smoke else 4,
        ff_dim=128 if args.compact_smoke else 1024,dropout=0.,pretrained_wrist_backbone=False)
    model=DapierACTMultimodal(**config).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    args.out.mkdir(parents=True,exist_ok=False)
    history=[]
    for epoch in range(args.epochs):
        model.train();train_rows=[]
        for i,b in enumerate(loaders['train']):
            b=inputs(b,device);opt.zero_grad(set_to_none=True)
            loss=losses_for(model,b)
            if not all(torch.isfinite(v) for v in loss.values()):raise ValueError('Nonfinite train loss')
            loss['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            train_rows.append({k:float(v.detach()) for k,v in loss.items()})
            if i+1>=args.steps:break
        model.eval();val_rows=[];prior_rows=[]
        # Verify buffers and parameters remain identical across validation.
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        with torch.no_grad():
            for b in loaders['val']:
                b=inputs(b,device);loss=losses_for(model,b)
                pred,_,_=model(**{k:b[k] for k in ('qpos','pointcloud','left_wrist','right_wrist')})
                prior=dapier_act_loss(pred,b['actions'],b['is_pad'],None,None)
                if not all(torch.isfinite(v) for v in [*loss.values(),*prior.values()]):
                    raise ValueError('Nonfinite validation/inference loss')
                val_rows.append({k:float(v) for k,v in loss.items()});prior_rows.append(float(prior['l1']))
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):
            raise AssertionError('Validation mutated model')
        row=dict(epoch=epoch+1,train={k:float(np.mean([r[k] for r in train_rows])) for k in train_rows[0]},
            val={k:float(np.mean([r[k] for r in val_rows])) for k in val_rows[0]},
            val_prior_l1=float(np.mean(prior_rows)),optimizer_steps=len(train_rows))
        history.append(row);print(json.dumps(row))
    summary=dict(synthetic_fixture=args.synthetic_fixture,compact_smoke=args.compact_smoke,history=history,
        manifest_sha256=canonical_hash(manifest),model_config=config,closed_loop_evaluated=False,
        note='Posterior CVAE loss and prior(z=0) L1 are offline checks, not task success.',
        episode_splits={s:[str(p) for p in paths] for s,paths in splits.items()})
    (args.out/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with h5py.File(splits['train'][0],'r') as source:
        provenance=json.loads(source.attrs['metadata'])
    torch.save(dict(fingerprint=provenance['fingerprint'],head_profile_sha256=provenance['head_profile_sha256'],
                    model=model.state_dict(),model_config=config,stats=stats,manifest=manifest,
                    synthetic_fixture=args.synthetic_fixture),args.out/'policy_smoke.pt')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--steps',type=int,default=2)
    p.add_argument('--chunk-size',type=int,default=20);p.add_argument('--compact-smoke',action='store_true')
    p.add_argument('--synthetic-fixture',action='store_true',help='Only for generated test fixtures; never evidence of grasp success')
    a=p.parse_args()
    if min(a.epochs,a.steps,a.chunk_size)<=0:p.error('epochs/steps/chunk-size must be positive')
    train(a)
