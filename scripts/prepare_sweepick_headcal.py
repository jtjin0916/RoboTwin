#!/usr/bin/env python3
"""Prepare a NEW local profile from the team's handover ZIP. No embedded tasks run."""
import argparse,hashlib,json,sys,zipfile,tempfile,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sweepick_headcal import SCHEMA,validate


def prepare(archive,out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Use a new output directory')
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())>100_000_000:raise ValueError('Unexpected archive size')
        def read(suffix):
            names=[n for n in z.namelist() if n.endswith('/'+suffix)]
            if len(names)!=1:raise ValueError('Expected unique '+suffix)
            return z.read(names[0])
        exraw=read('sweepick_head_extrinsic_IN_USE_20261020.json');extrinsic=json.loads(exraw)
        rect=json.loads(read('os30a_rectify_log_full.json'));values=json.loads(read('sweepick_261020_handover_values.json'))
        k=np.array(rect['NewCamMat1'],float).reshape(3,4)[:,:3];k=np.diag([.5,.5,1])@k
        profile={'schema':SCHEMA,'parent':'base_footprint','child':'camera_optical_frame',
          'T_base_from_optical':extrinsic['T_robot_from_camera'],'depth_K':k.tolist(),'resolution':[640,460],
          'real_depth_multiplier':extrinsic['depth_scale'],'status':'USER_SELECTED_HANDOVER_BASELINE',
          'source_status':extrinsic.get('status'),'source_extrinsic_sha256':hashlib.sha256(exraw).hexdigest(),
          'archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
          'provenance':'Board PnP plus arm-model translation; selected as simulation baseline, not independent absolute calibration',
          'invalid_depth_rule':'0 or >=16384; nonfinite invalid','pixel_mapping':'u_ref=2*u_depth, v_ref=2*v_depth'}
        validate(profile)
        contract={k:values[k] for k in ('control','act_contract_tjj','arms','wrist_cameras','convention')}
        manifest={'profile_source_sha256':profile['source_extrinsic_sha256'],'archive_sha256':profile['archive_sha256'],
          'note':'Local selected baseline. Real recordings and checkpoints were not modified. No archive instructions executed.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage=Path(temp)
        for name,obj in [('head_profile.json',profile),('real_contract.json',contract),('MANIFEST.json',manifest)]:
            (stage/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
        shutil.copytree(stage,out)
    return out/'head_profile.json'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--handover',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(prepare(a.handover,a.out))
