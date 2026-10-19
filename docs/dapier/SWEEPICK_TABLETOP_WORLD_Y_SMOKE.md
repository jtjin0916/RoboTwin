# Sweepick tabletop virtual-floor SAPIEN smoke (Oct 19, 2026)

## Latest frame contract
- Keep RoboTwin World heading **+Y**: TurtleBot `base_footprint` World position (0,-0.2539,0.500)m; yaw +90°.
- 2 × 2m tabletop, top World Z=0.500m, center World XY=(0,-0.7)m. Wall starts at World Y=0.4m, leaving 0.1m clearance to tabletop's positive Y edge (Y=0.3m).
- Arms measured in `base_footprint`: L=(-.060,+.150,+.123)m; R=(-.060,-.150,+.123)m. Thus new World arm origins L=(-.15,-.3139,.623), R=(+.15,-.3139,.623)m, each yaw+90°.
- Head camera pose is a provisional transformed estimate, **not** validated against hardware.
- Starting 40mm cube center in `base_footprint` = (.18,.14,.021)m; World = (-.14,-.0739,.521)m.
- `Camera.get_pcd()` first produces World points. This isolated task overrides `get_obs()` to save the point cloud XYZ in `base_footprint` instead, retaining RGB columns. **Legacy datasets should not be mixed without frame conversion.**
- `random_table_height` still defaults 0. If enabled, it shifts tabletop, TurtleBot, SO101 bases, Head camera and target cube jointly. The fixed base-relative CuRobo floor cuboid height then remains consistent (only the constant world metadata `support_top_world_z` would no longer describe the sampled height; the planner in this embodiment uses the local cuboid).
- WARNING: CuRobo currently represents a local support plane patch **only**, not the chassis/rails/wall. Full collision clearance requires a separate check before Workspace publication or bulk collection.

## Worktree and runtime prerequisites

Do **not** modify dirty `~/RoboTwin` or clean/reset its uncommitted files. In the separate `~/RoboTwin-sweepick` worktree:

```bash
cd ~/RoboTwin-sweepick
git fetch https://github.com/jtjin0916/RoboTwin.git refs/heads/feature/sweepick-tabletop-world-y-smoke-20261019
git worktree add --detach ~/RoboTwin-tabletop FETCH_HEAD
cd ~/RoboTwin-tabletop
```

Generate isolated SAPIEN P-gripper runtime URDF with the user-modified STL, preserving source joint kinematics:

```bash
~/RoboTwin/venv/bin/python scripts/build_sweepick_measured_urdf.py \
  --base-stl ~/sweepick_ros2_ws/src/sweepick_description/resources/Base_SO101__.stl
```

If this input path differs, replace it with the actual `Base_SO101__.stl` path. The generated URDF and copied STL remain in the new worktree only.

If objects do not exist in the isolated checkout, link the original external object assets only if that directory actually exists:

```bash
test -e assets/objects || ln -s ~/RoboTwin/assets/objects assets/objects
```

Run CPU-only coordinate/setup checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/RoboTwin/venv/bin/python -m pytest \
  tests/test_sweepick_measured_contract.py -v
```

The SAPIEN 40mm block smoke (requires ROS-independent local renderer, GPU and RoboTwin's SAPIEN/CuRobo dependencies):

```bash
export SWEEPICK_TURTLEBOT_URDF="$HOME/Alpenj-DAPIER/2ARM_ROBOT/sim/turtlebot3_waffle_pi/turtlebot3_waffle_pi_mujoco.urdf"
test -f "$SWEEPICK_TURTLEBOT_URDF" || echo "Update path to your actual teammate TurtleBot URDF"
bash collect_data.sh sweepick_block40_measured demo_so101_pgripper_sweepick_measured_smoke 0
```

Verify correct SAPIEN camera view, table contact and 40mm lift >30mm. Save the logs. None of the changes in this branch is proof of a real simulator run. Do not collect randomized/grid demonstrations until Grasp/Lift and collision geometry are verified.

## Next experimental steps
1. Fix any loader/mesh/collision failure found in the smoke run, including SO101 base STL origin differences.
2. Calibrate OS30A optical extrinsic; check object points are in expected `base_footprint` XYZ and enable `sweepick_crop_base` only after inspecting raw points.
3. Synchronize CuRobo collision obstacles for actual TurtleBot chassis and 20x20x400mm rails; confirm the tabletop width/shape and edge clearance.
4. Re-measure L/R reachability and actual SAPIEN grasp success with a robot-relative XY grid; keep seeds and failures.
5. Hold out spatial cells and test objects; then progressively randomize validated positions, block yaw/shapes, tabletop textures, lighting and calibrated camera noise/extrinsics.

The original top-level measured development branch and earlier Sweepick smoke branch are preserved.
