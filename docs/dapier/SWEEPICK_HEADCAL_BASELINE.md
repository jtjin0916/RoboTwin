# Sweepick selected head calibration

This is an isolated simulation path for the user-selected 2026-10-20 handover.
Existing measured tasks/configurations, recordings and checkpoints are preserved.
The supplied optical transform is relative to `base_footprint`, in metres:
`p_base = T_base_from_optical @ p_optical` (homogeneous coordinates).
It includes the mount height; do not add the arm-base Z offset again.
`camera_link` remains a CAD body datum and is not the depth optical origin.

## Prepare a local profile

From the RoboTwin repository root, using its normal Python environment:

```sh
python scripts/prepare_sweepick_headcal.py --handover /absolute/path/sweepick_261020_handover.zip --out local_sweepick_headcal_20261020
python scripts/preview_sweepick_headcal.py --profile local_sweepick_headcal_20261020/head_profile.json --check-only --out /tmp/sweepick-head-check-new
```

Outputs must be new directories. No embedded archive instructions run. Profile
hashes record provenance. `/local_sweepick_headcal*/` is ignored by Git. Keep
profiles and actual recordings local; source tests use synthetic calibration.
`real_contract.json` exports the supplied arm/control/ACT/wrist contracts for
review; it does not enforce those limits in the old planner or convert datasets.

## Render a fixed view

Install the repository's SAPIEN/CuRobo/GPU dependencies and provide the team's
actual TurtleBot URDF/meshes and generated measured SO101 URDF assets described
in the existing measured-robot handoff. The parent measured task still requires
these original assets; placeholders do not establish collision safety.

```sh
export SWEEPICK_TURTLEBOT_URDF=/absolute/path/to/actual_turtlebot.urdf
python scripts/preview_sweepick_headcal.py --profile local_sweepick_headcal_20261020/head_profile.json --out /tmp/sweepick-head-view-new
```

This runner initializes simulation at home and captures a fixed view; it does
not call `play_once`, produce demonstrations or command a real robot. It writes
`HEAD_DEPTH_MM.png`, `HEAD_VIEW.npz` (depth metres, mask, base-frame XYZ, K), and
`CAPTURE.json`. Preview depth excludes nonpositive/background and values >=2.5m.
CPU checks do not validate GPU rendering or physical alignment.

The new task is `sweepick_block40_headcal`, configured by
`env_cfg/task_config/demo_so101_pgripper_sweepick_headcal.yml`. Other runners can
pass `sweepick_head_profile` or set `SWEEPICK_HEAD_PROFILE` to the local JSON.
The view uses the full 640x460 rectified K, including principal point, after
camera creation. World yaw/translation and randomized table height are applied
once. It exports depth and XYZ; head RGB is removed. Wrist RGB is a separate
modality and is disabled in this inspection config.

## Baseline and randomization

The selected handover combines board PnP and a translation fitted against arm
FK. It is the accepted simulation baseline, not an independent absolute optical
calibration or a held-out sim-to-real accuracy guarantee. It supersedes the
older profile on this new path only. Real ZD depth is millimetres with multiplier
1.0; do not apply the historic 1/1.016 correction. Zero, >=16384 and nonfinite
real depth are invalid. Simulated depth remains metric with no empirical scaling.

First inspect the fixed view against real depth in the same scene. Camera DR
bounds default to zero. `sweepick_head_translation_dr_m` bounds each base-axis
translation; `sweepick_head_rotation_dr_deg` bounds total local optical rotation.
Both are sampled deterministically per episode seed. Legacy camera DR is forced
to zero to prevent a second perturbation. Choose bounds from measured variation;
pose DR does not simulate depth holes/noise or calibration bias. Add those models
separately and validate on real held-out runs before claiming robust ACT transfer.

The existing mechanical candidate, rail and housing proxies, and CuRobo safety
limitations are unchanged. This head-view update does not certify those models
or automatically attach the CAD mount to the measured runtime scene. The older
CAD body's optical offset must not be presented as an independently measured
physical camera datum merely because this accepted optical TF is available.
