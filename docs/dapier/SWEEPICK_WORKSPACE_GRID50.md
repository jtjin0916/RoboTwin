# Sweepick measured Workspace Grid50 diagnostic

**Status: code authored; actual two-arm SAPIEN tests not yet run.** Prior
successful 40mm block smoke was only left arm, base (X,Y)=(.18,+.14)m,
delta_z=.049857m. Do not infer entire workspace or right arm success.

## Candidate grid contract (metres, TurtleBot base_footprint)
- Robot local +X=forward, +Y=left, +Z=up. SAPIEN World yaw+90° unchanged.
- Head PointCloud task crop: x=.15..32, y=-.26..+.26, z=.005..40.
- 40mm cube half-side=.02m and 10mm extra X margin.
- **X 10 positions:** np.linspace(.18,.29,10), step ≈.01222m.
- **Y 5 positions:** [-.15,-.075,0,+.075,+.15]m, step .075m.
- **50 total unique XY cells**; cube center base z=.021m.
- This region is a nominal crop-contained **candidate** only; actual OS30A
  visibility can be blocked by grippers/profiles or invalid pixels. A
  local PointCloud proximity count is recorded as a coarse proxy, but not
  definitive object segmentation / camera visibility.
- Each cell is attempted for **both arms** by default, for 100 trials if
  repetitions=1. Previous Grid50 used nearest arm, which cannot establish
  the full independent left/right reachable regions.

## Run in isolated worktree

```bash
cd ~/RoboTwin-tabletop
git status --short
git fetch https://github.com/jtjin0916/RoboTwin.git \
  refs/heads/feature/sweepick-tabletop-world-y-smoke-20261019
git switch --detach FETCH_HEAD

# Pure NumPy grid inspection, works before loading SAPIEN:
python scripts/scan_sweepick_workspace_grid50.py --list-only
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_sweepick_workspace_grid50.py -v

# Needs existing measured-unit URDF, source TurtleBot URDF assets and GPU.
# First run only 5 XY positions with left arm to diagnose unknown failures:
python scripts/scan_sweepick_workspace_grid50.py --arms left --start 0 --limit 5

# Then scan complete two-arm 50-grid; completed keys are skipped:
python scripts/scan_sweepick_workspace_grid50.py --arms left right

# Optional: 3 repeats per arm per cell (already completed repeat=0 skipped):
python scripts/scan_sweepick_workspace_grid50.py --arms left right --repeats 3
```

Ensure `SWEEPICK_TURTLEBOT_URDF` is exported and the generated
`assets/embodiments/so101-pgripper/so101_pgripper_sweepick_unit.urdf`
exists in the worktree; otherwise generate using
`scripts/build_sweepick_measured_unit_urdf.py --base-stl ...`.

All attempts (success/failure) append to
`data_sweepick_workspace_grid50/attempts.csv`. Grid index maps directly
to XY and does not change when a trial fails. Each CSV row includes:
cell, arm, repeat, seed, setup status, approximate near-cube PCD count,
candidate grasp availability, Pregrasp/Grasp/Close/Lift planning,
object lift delta_z, actual LiftSuccess and failure stage. CSV writes
immediately after each attempt, so failed and interrupted scans are
resumable with no loss of already completed cells.

This scan is **not** a training dataset: it has no all-success-only
replacements and doesn't save HDF5/video by default. Capture a few
actual scene images separately for visibility confirmation.

**Important:** Existing CuRobo obstacles do not yet include all
TurtleBot/mount-profile collisions. Thus a successful planner result
does not prove geometric safety. Build/validate collision geometry,
then repeat the scan before declaring the measured physical workspace.
Grid boundaries should be refined after seeing the actual per-cell
success heatmap; +/-15cm can be narrowed or expanded based on measured
visibility and bilateral grasp success.
