# Sweepick measured Workspace Grid100 diagnostic

**Status: code authored; actual two-arm SAPIEN tests not yet run.** Prior
successful 40mm block smoke was only left arm, base (X,Y)=(.18,+.14)m,
delta_z=.049857m. Do not infer entire workspace or right arm success.

## Candidate grid contract (metres, TurtleBot base_footprint)
- Robot local +X=forward, +Y=left, +Z=up. SAPIEN World yaw+90° unchanged.
- Head PointCloud task crop: x=.15..32, y=-.26..+.26, z=.005..40.
- 40mm cube half-side=.02m and 10mm extra X margin.
- **X 10 positions:** np.linspace(.18,.29,10), step ≈.01222m.
- **Y 10 positions:** np.linspace(-.15,+.15,10), step ≈.03333m.
  Values in mm: -150, -116.667, -83.333, -50, -16.667, +16.667,
  +50, +83.333, +116.667, +150. Even sampling includes no Y=0 point.
- **100 total unique XY cells**; cube center base z=.021m.
- This region is a nominal crop-contained **candidate** only; actual OS30A
  visibility can be blocked by grippers/profiles or invalid pixels. A
  local PointCloud proximity count is recorded as a coarse proxy, but not
  definitive object segmentation / camera visibility.
- Each cell is attempted for **both arms** by default, for **200 trials** if
  repetitions=1, or 600 trials for 3 repeats. Both arms use the same seed
  for the same cell/repeat to allow a paired comparison.

Grid ID is `10 * x_index + y_index` (Y varies fastest), with indices 0..9.
ID 0=(.18,-.15), ID 9=(.18,+.15), ID 10=(.192222…,-.15), and
ID 99=(.29,+.15). The seed is `grid_id + 100 * repeat` (repeat starts at 0).
Grid50 IDs and seeds must not be interpreted as Grid100 positions.

The measured parent task, 2×2m table at World Z=.5m (centre=(0,-.7)),
base origin=(0,-.2539,.5), 40mm cube, arm calibration, camera and crop
settings are inherited unchanged. The Grid100 YAML differs from Grid50
only in its output path; background/light/camera/table randomization
remain disabled. Existing randomization profiles remain available via
`--config`; use a separate `--out` path for each selected profile.

Grid50 code/config/results are preserved. Use the new Grid100 entry
points below; do not rename or copy an existing Grid50 CSV into Grid100.

## Local execution

```bash
cd ~/RoboTwin-tabletop
git status --short
git fetch https://github.com/jtjin0916/RoboTwin.git \
  refs/heads/codex/sweepick-workspace-grid100-20261019
git switch --detach FETCH_HEAD

# Pure NumPy grid inspection, works before loading SAPIEN:
python scripts/scan_sweepick_workspace_grid100.py --list-only
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_sweepick_workspace_grid100.py -v

# Needs existing measured-unit URDF, source TurtleBot URDF assets and GPU.
# First run only 5 XY positions with left arm to diagnose unknown failures:
python scripts/scan_sweepick_workspace_grid100.py --arms left --start 0 --limit 5

# Then scan complete two-arm Grid100 (200 attempts); completed keys are skipped:
python scripts/scan_sweepick_workspace_grid100.py --arms left right

# Optional: 3 repeats per arm per cell (already completed repeat=0 skipped):
python scripts/scan_sweepick_workspace_grid100.py --arms left right --repeats 3
```

Ensure `SWEEPICK_TURTLEBOT_URDF` is exported and the generated
`assets/embodiments/so101-pgripper/so101_pgripper_sweepick_unit.urdf`
exists in the worktree; otherwise generate using
`scripts/build_sweepick_measured_unit_urdf.py --base-stl ...`.

All attempts (success/failure) append to
`data_sweepick_workspace_grid100/attempts.csv`. Grid index maps directly
to XY and does not change when a trial fails. Each CSV row includes:
grid size, config name, cell, arm, repeat, seed, setup status, approximate near-cube PCD count,
candidate grasp availability, Pregrasp/Grasp/Close/Lift planning,
object lift delta_z, actual LiftSuccess and failure stage. CSV writes
immediately after each attempt. Resume uses `(grid_id, arm, repeat)`;
**both successful and failed recorded attempts are skipped**. Rerun the
same command to continue. Increasing `--repeats` adds only missing repeat
keys. Ctrl-C cleans up the active environment; the unrecorded active
attempt is retried on the next run. This is not a success-only seed search.

Resume checks the complete header, grid size, config name, coordinates,
seed, key uniqueness and required status fields before appending. A
Grid50/mismatched/truncated CSV is rejected without modification. If a
crash leaves a partial final line, preserve that file and use a fresh
`--out` path or repair that final line explicitly. The CSV stores the config
name, not a snapshot of config contents; after editing a YAML/calibration,
use a new output path. Concurrent writers to the same CSV are unsupported.
To intentionally retry failures, use a new output path, for example
`--out data_sweepick_workspace_grid100_retry/attempts.csv`.

## CPU verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
python -m compileall -q sweepick_workspace_grid100.py \
  scripts/scan_sweepick_workspace_grid100.py envs/sweepick_block40_grid100.py
```

Validation in the editing environment: **48 tests passed**, Python compilation
and whitespace checks passed. NVIDIA driver access was unavailable.

CPU tests check all 100 coordinates, crop and World/base conversion,
unchanged YAML settings, 200 two-arm records including setup/planning/lift
failures, 600 records over three repeats, last-cell limits, per-arm resume,
interruption, invalid resume rejection and simulator-free CLI listing.
Fake tasks test scanner bookkeeping; they do not establish physical grasp
success. Actual Grid100 SAPIEN/CuRobo execution requires the local GPU and
assets and has not been performed here.

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
