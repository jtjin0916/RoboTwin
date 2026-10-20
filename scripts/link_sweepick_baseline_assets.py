#!/usr/bin/env python3
"""Reuse missing local simulation assets by symlink; never overwrite any path."""
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def link_missing(source, destination):
    source=source.expanduser().resolve();destination=destination.resolve()
    if not source.is_dir() or source==destination or source in destination.parents or destination in source.parents:
        raise ValueError('Require distinct, non-nested asset directories')
    destination.mkdir(parents=True,exist_ok=True)
    def visit(src,dst):
        for child in src.iterdir():
            target=dst/child.name
            if target.exists() or target.is_symlink():
                if child.is_dir() and not child.is_symlink() and target.is_dir() and not target.is_symlink():
                    visit(child,target)
                continue
            target.symlink_to(child,target_is_directory=child.is_dir())
    visit(source,destination)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();link_missing(a.source,ROOT/'assets')
