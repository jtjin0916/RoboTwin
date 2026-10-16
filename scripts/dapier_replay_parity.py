#!/usr/bin/env python3
"""CPU regression against an exact source tree; no training or motor I/O.
Run once per source (working, frozen reference, public), then compare NPZs.
Only use trusted local checkpoint/stats files, which contain pickle data.
"""
import argparse,sys,types,json,pickle,hashlib
from pathlib import Path
import numpy as np
import torch,h5py
p=argparse.ArgumentParser()
p.add_argument('--source',required=True,type=Path,help='Directory containing model_multimodal.py')
p.add_argument('--checkpoint',required=True,type=Path)
p.add_argument('--episode',required=True,type=Path)
p.add_argument('--stats',required=True,type=Path)
p.add_argument('--timestep',type=int,default=0)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
# Namespace the selected source independently of any editable install.
pkg=types.ModuleType('dapier_act');pkg.__path__=[str(a.source.resolve())];sys.modules['dapier_act']=pkg
from dapier_act.model_multimodal import DapierACTMultimodal
from dapier_act.dataset_multimodal import DapierMultimodalDataset
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
with a.stats.open('rb') as f:stats=pickle.load(f)
cfg=ck.get('model_config') or dict(qpos_dim=12,action_dim=12,chunk_size=ck['chunk_size'],hidden_dim=256,latent_dim=32,num_point_samples=ck.get('num_point_samples',512),candidate_points=ck.get('candidate_points',8192),point_min=ck['point_min'],point_max=ck['point_max'],nheads=8,encoder_layers=4,decoder_layers=4,ff_dim=1024,dropout=0.)
cfg=dict(cfg);cfg['pretrained_wrist_backbone']=False
model=DapierACTMultimodal(**cfg).eval();r=model.load_state_dict(ck['model'],strict=True)
dataset=DapierMultimodalDataset(a.episode.parent,stats,chunk_size=20,episode_files=[a.episode])
idx=next(i for i,(_,t) in enumerate(dataset.samples) if t==a.timestep);sample=dataset[idx]
with torch.inference_mode():
 action,mu,logvar,features=model(qpos=sample['qpos'][None],pointcloud=[sample['pointcloud']],left_wrist=sample['left_wrist'][None],right_wrist=sample['right_wrist'][None],actions=None,is_pad=None,return_features=True)
assert action.shape==(1,20,12) and mu is None and logvar is None and torch.isfinite(action).all()
arrays={'input_'+k:v.numpy() for k,v in sample.items() if isinstance(v,torch.Tensor)}
arrays.update({k:v.numpy() for k,v in features.items()});arrays['action']=action.numpy()
a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**arrays)
with h5py.File(a.episode,'r') as f:
 schema={}
 def visit(k,v):
  if isinstance(v,h5py.Dataset):schema[k]={'shape':list(v.shape),'dtype':str(v.dtype)}
 f.visititems(visit)
result={'source':str(a.source.resolve()),'episode':str(a.episode.resolve()),'episode_sha256':hashlib.file_digest(a.episode.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(a.episode.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'stats_sha256':hashlib.sha256(a.stats.read_bytes()).hexdigest(),'timestep':a.timestep,'model_parameters':sum(p.numel() for p in model.parameters()),'strict_missing':r.missing_keys,'strict_unexpected':r.unexpected_keys,'schema':schema,'arrays':{k:list(v.shape) for k,v in arrays.items()},'inference':'z=0; historical RGB loader; no temporal ensemble'}
a.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['schema','arrays']}))
