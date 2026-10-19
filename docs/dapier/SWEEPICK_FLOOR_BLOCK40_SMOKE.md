# Sweepick 2026-10-19: floor Block40 smoke (DRAFT, NOT physically validated)

## Sources
- Team handoff: `sweepick_261019_sim_handoff.zip`, especially `SWEEPICK_POINTCLOUD_FRAME_AND_SIM_HANDOFF.md`, `sweepick_head_extrinsic_measured.json`.
- TurtleBot3 + mounting source models: [Alpenj/DAPIER, branch pro/sweepick-auto-teacher-20261018](https://github.com/Alpenj/DAPIER/tree/pro/sweepick-auto-teacher-20261018/2ARM_ROBOT/sim).
- Runtime SO-101 P-gripper / CuRobo stay on the existing, tested RoboTwin URDFs; teammate's stock SO-101 URDF is **not** substituted.

## Stage 1 only
1. Base task table top is selectable; legacy defaults 0.74m, Sweepick passes 0.0m. When set to 0, **the existing SAPIEN ground is the sole support surface**; do not create a 0-height table with negative-size legs.
2. CuRobo floor cuboid is configured separately in the Sweepick CuRobo robot-base frame, with centre z=-.16 and top z=-.14 relative to the 140mm raised base; defaults for legacy tabletop remain unchanged. Includes no TurtleBot/mount obstacle.
3. New embodiment has L/R bases at World=(-.15,-.35,.14), (+.15,-.35,.14), TurtleBot origin=(0,-.2539,0) yaw90; head position/direction from measured OpenCV `T_robot_from_camera`.
4. New 40mm red floor block centre in tb3 `base_footprint` is (.18,.14,.021), matching the handoff's left-arm recommended point (.18,.14). Existing left-arm single-block grasp/lift code is reused for **initial smoke only**.
5. Static TurtleBot model is loaded from the teammate's existing Waffle Pi URDF checkout (mesh files remain in their repository), plus three collision-only provisional mount/housing box proxies. The platform's actual SAPIEN loading is **unverified** until running the smoke test on the target computer.
6. A pure-NumPy fixed crop/FPS512 normalization reference is provided, but **the production training DataLoader is not yet switched**. Pretraining with old loader would be wrong.

## Reusing large ignored asset packs in a Git worktree\n\nUpstream RoboTwin ignores `assets/*`; an isolated worktree will therefore **not** include the local `objects` asset pack that another checkout may have downloaded. `envs.utils` eagerly reads both `assets/objects/objaverse/list.json` and `assets/objects/same.json`, even with `cluttered_table: false`. If the original checkout owns these assets, share without modifying the originals:\n\n```bash\ncd ~/RoboTwin-sweepick\nls -l ~/RoboTwin/assets/objects/{objaverse/list.json,same.json}\nif [ ! -e assets/objects ] && [ ! -L assets/objects ]; then\n  ln -s ~/RoboTwin/assets/objects assets/objects\nfi\nls -l assets/objects/{objaverse/list.json,same.json}\n```\n\nIf the source files are absent, obtain the licensed RoboTwin object assets rather than inventing dummy JSON. The `XPolicyLab` submodule may also need initialization on this worktree.\n\n## Run on the target GPU/Ubuntu machine
```bash
# Provide the complete external model+meshes via a checked-out teammate repository.
git clone --branch pro/sweepick-auto-teacher-20261018 --single-branch https://github.com/Alpenj/DAPIER.git ~/Alpenj-DAPIER
export SWEEPICK_TURTLEBOT_URDF="$HOME/Alpenj-DAPIER/2ARM_ROBOT/sim/turtlebot3_waffle_pi/turtlebot3_waffle_pi_mujoco.urdf"
# From the RoboTwin root on this feature branch:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/RoboTwin/venv/bin/python -m pytest tests/test_sweepick_contract.py -q
bash collect_data.sh sweepick_block40 demo_so101_pgripper_sweepick_block40_smoke 0\n# max_seed_attempts: 3 prevents infinite seed retries in this experimental smoke task.
```

## Mandatory gates before large-scale collection
- [ ] Confirm TurtleBot URDF with original meshes really loads in SAPIEN; wheel/link geometry must render; no silent proxy substitution.
- [ ] Inspect physical floor contact and SO-101 base world transforms; table and CuRobo world obstacle must coincide with z=0. Confirm CuRobo robot-local obstacle pose convention.
- [ ] Verify head intrinsics: measured depth K=(fx,fy,cx,cy)=(409.3,409.3,313.15,231.21); SAPIEN fovy centred camera is **only an approximation** (principal point differs).
- [ ] Confirm head and both wrist camera images against real images; wrist FOV has been changed, but measured `T_tool_camera` is **not yet applied** because tool-site vs `Gripper_Base_v1_1` is unresolved.
- [ ] Add validated full mount/stand meshes and CuRobo collision obstacles; current three box proxies do not certify collision-free plans.
- [ ] Compare measured wrist-roll joint limits (+.957 rad L, +.993 rad R) against current URDF/Planner range before expert collection.
- [ ] Confirm real action_dt=0.1s, H=20; `save_freq:15` is **not yet proven** equivalent to 10Hz labels.
- [ ] Wire the pure `SWEEPICK_TASK` crop/FPS512/fixed normalization into the simulation trainer **and** real data converter. Preserve legacy checkpoints via explicit preprocessing contract.
- [ ] Integrate previously fixed RGB decode mode from [DAPIER-ACT](https://github.com/jtjin0916/DAPIER-ACT/blob/main/docs/RGB_CONTRACT.md), do not silently modify legacy v4.
- [ ] Implement inference-mode (`z=0`) offline validation and closed-loop block grasp/lift/stow evaluation; **not verified** in this PR.
- [ ] Collect new Clean Sim expert trajectories, then Real Fine-tuning. No synthetic depth noise in initial Pretraining.

## Source accuracy notes
The handoff's measured head extrinsic is a **measured estimate** (ICP fit, operator approved), not a guarantee of pixel-exact alignment. The nominal simulated OS30A profile does not reproduce depth dropout or IR artifacts. The team model's 140mm arm height is measured, but chassis mount collision geometry remains unverified. Do not operate real motors using this PR.
