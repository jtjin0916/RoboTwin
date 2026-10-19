# Sweepick measured 40mm floor block smoke — experimental 2026-10-19

Keep the original RoboTwin and older Sweepick files unchanged. This experiment uses a distinct task, embodiment and data folder.
Measured L=(-.060,+.150,.123)m, R=(-.060,-.150,.123)m; World=base_footprint. Rails 20x20x400mm at X=-.014/-.106m, Z=.113m, symmetric placement not physically certified.
WARNING: CuRobo has a floor cuboid ONLY, not the actual TurtleBot/profiles. Do not claim collision-safe planning or collect bulk demonstrations.

## Create isolated local worktree

    cd ~/RoboTwin-sweepick
    git fetch https://github.com/jtjin0916/RoboTwin.git refs/heads/feature/sweepick-measured-block40-smoke-20261019
    git worktree add --detach ~/RoboTwin-measured FETCH_HEAD
    cd ~/RoboTwin-measured

## Generate only the measured runtime URDF

    ~/RoboTwin/venv/bin/python scripts/build_sweepick_measured_urdf.py --base-stl ~/sweepick_ros2_ws/src/sweepick_description/resources/Base_SO101__.stl

The generator copies user geometry into a NEW STL filename and derives a NEW URDF, with joint origin/limits and all non-base meshes preserved. It does NOT overwrite original URDFs.
Expect two untracked generated files in the isolated worktree.

## Run CPU contract, then SAPIEN/CuRobo on target GPU

    if [ ! -e assets/objects ]; then ln -s ~/RoboTwin/assets/objects assets/objects; fi
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/RoboTwin/venv/bin/python -m pytest tests/test_sweepick_measured_contract.py -v
    export SWEEPICK_TURTLEBOT_URDF="$HOME/Alpenj-DAPIER/2ARM_ROBOT/sim/turtlebot3_waffle_pi/turtlebot3_waffle_pi_mujoco.urdf"
    bash collect_data.sh sweepick_block40_measured demo_so101_pgripper_sweepick_measured_smoke 0

Interpret success only after verifying load, left-arm FK/TCP and CuRobo paths, collision margins, actual SAPIEN contact and 30mm block lift. Original block centre is (0.18,0.14,0.021) in the NEW world.
Do not launch workspace Grid/randomization until measured smoke and full collision model are validated.

Notion: Project / DAPIER / SWEEPICK measured URDF→RoboTwin verification, Grid, Domain Randomization plan.