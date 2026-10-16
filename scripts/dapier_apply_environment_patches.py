#!/usr/bin/env python3
"""Opt-in reproduction of two observed upstream-installer package patches.
Use only in a fresh clone's dedicated venv. Does not alter URDF or ACT behavior.
"""
import argparse,sys,importlib.util
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
if sys.prefix==sys.base_prefix:raise SystemExit('Use the reproduction virtualenv, not system Python')
changes=[('sapien','wrapper/urdf_loader.py','with open(urdf_file, "r") as f:','with open(urdf_file, "r", encoding="utf-8") as f:'),('mplib','planner.py','if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:','if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:')]
for package,rel,old,new in changes:
 target=Path(importlib.util.find_spec(package).origin).parent/rel;s=target.read_text()
 if new in s:print('already matches observed environment:',target);continue
 if s.count(old)!=1:raise SystemExit(f'Unexpected package source, refusing to patch {target}')
 print('patch candidate:',target,'apply' if a.apply else 'dry-run')
 if a.apply:
  backup=target.with_name(target.name+'.dapier-original')
  if backup.exists():raise SystemExit(f'Backup exists; inspect {backup}')
  backup.write_text(s);target.write_text(s.replace(old,new))
