# SWEEPICK selected OS30A headcal: Crop X=0.13m, Grid X=0.15..0.29m (candidate)

**Not verified in GPU or physical comparison.** New files are isolated and do not change the
original `codex/sweepick-camera-cad-candidate-20261020` headcal path, old Grid100 CSV,
dataset, or ACT normalizer/checkpoints.

## Robot-front exclusion review

The team's TurtleBot3 Waffle Pi URDF has a base collision box centered at
base_link X=-0.064m, size X=.266m, so its forward edge is about **+0.069m**
in base_footprint. The stock camera collision box on the TurtleBot extends
to X≈.0855m. The proposed crop X min=.13m begins ~44.5mm past this
URDF-based forward collision edge. This does **not** prove visibility,
physical clearance or detailed CAD / arm collision safety: some mounted
pieces and dynamic arms have separate or incomplete collision models.

A 40mm cube centered at X=.15m has its near face X=.13m exactly on the
crop boundary. This is **zero margin**; near-face points may be filtered.
Treat X=.15m cells as exploratory, not automatically valid for ACT data.
For a 10mm margin keep cube centres >=.16m, or adjust the crop only after
an actual depth/collision study. X max stays .29m.
Y is linspace(-.15,+.15,10) and Y<0 right arm, Y>0 left arm (50 per arm).
The cube centre z=.021m; fixed selected optical K/TF profile used.
Crop min=[.13,-.26,.005]; max=[.32,.26,.40] in base_footprint.

## Run: do not overwrite existing files

```bash
cd ~/RoboTwin-tabletop
git status --short
git fetch origin codex/sweepick-headcal-near-grid100-20261020
git switch --detach FETCH_HEAD

# Your local IN_USE profile must already exist; otherwise run prepare script
# with sweepick_261020_handover.zip into a NEW local_sweepick_headcal* directory.
export SWEEPICK_HEAD_PROFILE="$PWD/local_sweepick_headcal_20261020/head_profile.json"
export SWEEPICK_TURTLEBOT_URDF="$HOME/Alpenj-DAPIER/2ARM_ROBOT/sim/turtlebot3_waffle_pi/turtlebot3_waffle_pi_mujoco.urdf"

python scripts/scan_sweepick_headcal_near_grid100.py --list-only
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_sweepick_headcal_near_grid100.py -v

# First fixed camera-only captures at the near X=.15m rightmost/leftmost Y cells:
python scripts/preview_sweepick_headcal_near_grid100.py \
  --profile "$SWEEPICK_HEAD_PROFILE" --grid-id 0 --out /tmp/head-near-00
python scripts/preview_sweepick_headcal_near_grid100.py \
  --profile "$SWEEPICK_HEAD_PROFILE" --grid-id 9 --out /tmp/head-near-09

# Only after inspecting actual depth/PointCloud/occlusion:
python scripts/scan_sweepick_headcal_near_grid100.py \
  --profile "$SWEEPICK_HEAD_PROFILE" --arm-mode assigned --limit 10
python scripts/scan_sweepick_headcal_near_grid100.py \
  --profile "$SWEEPICK_HEAD_PROFILE" --arm-mode assigned --repeats 3
```

Each attempted (grid, assigned arm, repeat) is appended, including failures,
to `data_sweepick_headcal_near_grid100/attempts.csv`. Never append the old
X=.18..29 Grid100 results because same grid IDs now map to different X.
The GPU code has **not** run in this editing environment; repeated success
and independent real depth are needed to establish a usable workspace.
Count-near-block proxy is not semantic object segmentation.

## Data-collection gates

1. Verify head images: depth 640x460, actual K, full selected optical pose,
   base-frame XYZ and static body mask at X crop .13m.
2. Verify that cube at X .15m is fully observable, and make an explicit keep/
   reject decision for the *zero-margin* row. At X>=.16m margin >=10mm.
3. Inspect the selected-arm Grasp/Lift outcomes for near X=.15-.18m,
   and repeat seed trials. Successful SAPIEN Lift alone does NOT certify
   collision safety; CAD mount/chassis/other arm/rails missing in CuRobo.
4. Implement and validate Pick -> Lift -> Return-to-Stow -> ObjectHeld
   expert on actual measured home joint limits and both arms.
5. Compare an independently captured real OS30A scene, inspect the
   handover contract's joint rates, RGB wrist inputs, action/state clocks,
   head cloud FPS512 and 8192 candidates, new *versioned* crop normalizer.
   Legacy ACT dataset uses the old crop and must not be silently mixed.
6. Only then enable physically justified background/light/pose/noise
   randomization and capture a small expert HDF5 QA set. Large-scale
   pretraining / real deployment is **not** authorized by this geometry change.
