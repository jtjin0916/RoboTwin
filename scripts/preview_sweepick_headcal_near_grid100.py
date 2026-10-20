#!/usr/bin/env python3
"""Fixed head-view inspection, no play_once or imitation dataset generation.
--check-only validates the private profile and TF without GPU/robot assets.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sweepick_headcal import load_profile,world_camera
from sweepick_workspace_near_grid100 import cell,NUM_CELLS,CROP_MIN,CROP_MAX
from sweepick_table_frame import world_to_base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--check-only',action='store_true');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--grid-id',type=int,default=0)
    a=p.parse_args();out=a.out.resolve()
    if not 0<=a.grid_id<NUM_CELLS:p.error('grid-id must be 0..99')
    if out.exists():raise FileExistsError('Use a new output directory')
    profile=load_profile(a.profile);c=world_camera(profile,.5)
    report={'mode':'CPU_PROFILE_CHECK_ONLY' if a.check_only else 'FIXED_GPU_HEAD_VIEW',
            'profile_file_sha256':profile['_profile_file_sha256'],'source_extrinsic_sha256':profile['source_extrinsic_sha256'],
            'T_base_from_optical':profile['T_base_from_optical'],'world_camera_at_nominal_tabletop':c,
            'real_depth_multiplier':1.,'simulated_depth_scaling':1.,'physical_calibration_or_collision_safety_verified':False}
    if a.check_only:
        out.mkdir(parents=True);(out/'CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(out/'CHECK.json');return
    from scripts.scan_sweepick_headcal_near_grid100 import prepare_args
    from envs.sweepick_block40_headcal_near_grid100 import sweepick_block40_headcal_near_grid100
    args=prepare_args('demo_so101_pgripper_sweepick_headcal_near_grid100')
    args.update(task_name='sweepick_block40_headcal_near_grid100',task_config='demo_so101_pgripper_sweepick_headcal_near_grid100',
                sweepick_head_profile=str(a.profile.resolve()),seed=a.seed,now_ep_num=0,save_data=False,collect_data=False,
                grid_index=a.grid_id,scan_arm=('right' if cell(a.grid_id)[1]<0 else 'left'))
    task=sweepick_block40_headcal_near_grid100()
    try:
        task.setup_demo(**args) # Simulation-only initialization and home pose; no pick/lift.
        camera=next(cam for name,cam in zip(task.cameras.static_camera_name,task.cameras.static_camera_list) if name=='head_camera')
        task.scene.update_render();camera.take_picture();position=camera.get_picture('Position')
        depth=-position[...,2];valid=(position[...,3]<1)&np.isfinite(depth)&(depth>0)&(depth<2.5)
        model=camera.get_model_matrix();world=position[valid,:3]@model[:3,:3].T+model[:3,3]
        base=world_to_base(world,tabletop_world_z=task.tabletop_world_z)
        # Diagnostic occupancy is not actor segmentation; visually inspect too.
        inside_crop=np.isfinite(base).all(1)&(base>=CROP_MIN).all(1)&(base<=CROP_MAX).all(1)
        cropped=base[inside_crop]
        bx,by=cell(a.grid_id)
        vicinity=(np.abs(cropped[:,0]-bx)<.023)&(np.abs(cropped[:,1]-by)<.023)&(cropped[:,2]>.010)&(cropped[:,2]<.045)
        near_object_points=int(vicinity.sum())
        measured_k=np.asarray(camera.get_intrinsic_matrix())
        if not np.allclose(measured_k,np.array(profile['depth_K']),atol=1e-3):raise RuntimeError('Renderer K does not match selected K')
        report.update(tabletop_world_z=task.tabletop_world_z,actual_renderer_K=measured_k.tolist(),valid_points=len(base),
            grid_id=a.grid_id,block_xy_base_m=list(cell(a.grid_id)),candidate_crop_min=CROP_MIN.tolist(),candidate_crop_max=CROP_MAX.tolist(),
            cropped_base_points=int(len(cropped)),near_object_points_proxy=near_object_points,
            note='Synthetic geometry only. Real validation scene was not supplied here; no 1cm sim-real match is claimed.')
        out.mkdir(parents=True)
        np.savez_compressed(out/'HEAD_VIEW.npz',depth_m=np.where(valid,depth,np.nan),valid=valid,xyz_base=base,xyz_crop_base=cropped,K=measured_k)
        from PIL import Image
        Image.fromarray(np.where(valid,depth*1000,0).clip(0,65535).astype(np.uint16)).save(out/'HEAD_DEPTH_MM.png')
        (out/'CAPTURE.json').write_text(json.dumps(report,indent=2)+'\n');print(out)
    finally:
        if hasattr(task,'close_env'):task.close_env()

if __name__=='__main__':main()
