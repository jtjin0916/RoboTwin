# SO-101 P-gripper + Custom ACT reproduction

Source: Notion handoff 2026-10-16, sections 6–8. This branch preserves the local
working code, not a new ACT implementation. Original ~/RoboTwin and ~/DAPIER-ACT
were left untouched. Upstream RoboTwin commit and source hashes are in
source-manifest.json. XPolicyLab fork commit is pinned by the Git submodule.

## Clone and CPU contracts

```bash
# LFS is optional for unrelated policy figures, not DAPIER code or gripper meshes.
GIT_LFS_SKIP_SMUDGE=1 git clone --branch so101-pgripper-custom-act \
  https://github.com/jtjin0916/RoboTwin.git
cd RoboTwin
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r docs/dapier/requirements-runtime.txt
python -m pip install -e ./XPolicyLab --no-deps
python scripts/dapier_smoke.py
python -m dapier_act.train_multimodal --help
```

The requirements capture the observed working versions of direct runtime
packages, not a complete platform/CUDA lock. No clean-machine dependency install
was performed on this host; the commands require a Python 3.10 installation.
GPU simulation additionally needs a compatible NVIDIA driver, CUDA toolkit,
Vulkan support and CuRobo. Observed environment details are in environment.json.
Do not use scripts/_install.sh for this branch without reviewing it: that script
updates XPolicyLab to a moving branch. Keep the committed submodule SHA.

## GPU simulator setup and smoke

```bash
# In the new clone's environment, after NVIDIA/CUDA/Vulkan are working:
git clone https://github.com/NVlabs/curobo.git envs/curobo
git -C envs/curobo checkout d64c4b005459db10c5dd867d8b30a87d5bda9bdb
python -m pip install -e ./envs/curobo --no-build-isolation
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" --no-build-isolation
# Reproduce the two observed upstream installer patches in this new venv:
python scripts/dapier_apply_environment_patches.py
python scripts/dapier_apply_environment_patches.py --apply
python scripts/dapier_smoke.py --physics
# Collection writes only to this clone's ignored data/ directory:
bash collect_data.sh move_woodenblock_fixed demo_so101_pgripper_calib 0
```

CuRobo source at that SHA accepts a robot configuration dict. This migration
resolves `./` URDF/asset paths relative to curobo.yml and passes that dict, removing
the old /home/user/RoboTwin dependency. Other absolute paths retain their values.
The observed local CuRobo source was clean; CUDA execution is not validated here.
The observed SAPIEN loader has the UTF-8 patch and MPLib screw planner omits
the `or collide` failure condition, matching the upstream installer modifications.
The opt-in patch script reproduces exactly these observed changes in the new
venv and keeps backups. It does not add the installer comment’s suggested SRDF
dot change (the actual local loader still uses `+ "srdf"`). Dynamic collision
behavior on a new GPU host remains unverified.

The preserved wooden-block/marker/slim-can tasks construct primitive geometry;
their no-random-background configs do not require the whole external asset pack.
Other upstream tasks may require TianxingChen/RoboTwin2.0 assets, whose terms
must be reviewed separately. Do not run bulk asset path-rewrite scripts over the
committed pgripper config. The actual active URDFs and all 22 referenced meshes
are included and hash-checked. Runtime uses meter gripper meshes and two jaw
boxes. Planner uses calibration URDF with continuous arm joints and an extra
legacy gripper frame; arm parent/child, origins and axes match exactly.
Mimic is applied by robot.py from config.yml, not URDF mimic tags.

## ACT training and checkpoint replay

```bash
# Training uses the preserved historical loader. Read the RGB caveat below.
python -m dapier_act.train_multimodal \
  --dataset-dir /path/to/episode_hdf5_directory \
  --output-dir checkpoints/new_run --no-pretrained-wrist-backbone
# Replay once for each source dir: this clone/dapier_act, the public repository's
# dapier_act, and its frozen reference/robotwin_dapier_act.
python scripts/dapier_adapter_smoke.py --checkpoint-dir /path/to/checkpoint_directory \
  --episode /path/to/episode_0000000.hdf5
python scripts/dapier_replay_parity.py --source ./dapier_act \
  --checkpoint /path/to/policy_best.pt --stats /path/to/dataset_stats.pkl \
  --episode /path/to/episode_0000000.hdf5 --timestep 0 --output replay.npz
```

Public ACT baseline: jtjin0916/DAPIER-ACT commit afd3808 (full SHA in manifest).
Its tests were rerun with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2
python -m pytest -q -p no:cacheprovider tests`: 9 passed, 1 nested-tensor warning.
Replay strict-load: 31,637,708 parameters, no missing/unexpected keys.
Reference/public/migrated working implementation match exactly for 15 arrays,
including cropped PCD, normalized qpos/action chunk, two wrist RGB tensors,
FPS512, PointNet/head/wrist tokens, feature maps, context [1,163,256] and raw
z=0 action [1,20,12], on episode 0000000 timestep 0. This is one-frame numerical
regression, not a whole-corpus or policy success-rate evaluation.

The episode, checkpoint and stats stay outside Git. Their SHA256, local storage
location, full HDF5 schema and replay timestamp index are recorded in
validation/reference.json. Dataset: 72 steps, head depth 240x320, both wrist
JPEG buffers, state/action [72,5+1+5+1], raw world PCD [76800,6] per frame,
camera intrinsics/extrinsics. The manifest points to the external SSD; it is
**not a public download URL**. Team artifact sharing is still required before
another machine can replay these exact files. No dataset rights are assumed.

## Results and remaining gates

- PASS static embodiment/task/camera config, action_dim=12 and mesh hashes.
- PASS runtime/planner arm FK metadata; wrist-roll zero is the same.
- PASS CPU checkpoint strict load and same-episode 15-array regression.
- PASS actual pinned XPolicyLab adapter import, observation packing, checkpoint
  load and recorded-observation inference on CPU: 12 finite action values.
- BLOCKED full SAPIEN articulation rendering: Vulkan ErrorExtensionNotPresent;
  NVIDIA driver unavailable (`nvidia-smi` failed, torch.cuda.is_available false).
- NOT VERIFIED GPU planning, dynamic jaw closure/contact, live camera RGB/depth
  observations, full expert collection and ACT closed-loop metrics.
- NOT VERIFIED clean NVIDIA-host install and full external-artifact replay.

Historical v4 RGB loader swaps channels; XPolicyLab runtime does not. Existing
public RGB-contract color-marker unit tests passed, but a physical marker/color
reference in this episode and 50-grid raw/swapped RGB A/B remain unverified.
Temporal ensemble is oldest-heavy and unchanged. Expert teacher success is not
ACT success. OS30A live intrinsics/extrinsics, object-ROI depth validity, real
mount heights (140 vs 124 mm), signs/zero/FK/gripper motor mapping are unresolved.
The simulation task transform origin (0,-0.35,0.74) must not be used as a real
camera calibration. Real motor operation is outside this migration.

## To match the real robot

Before real motion, record each arm's mounting transform, joint order, units,
signs, zero offsets and limits; compare measured FK/TCP with this URDF at several
safe poses. Measure gripper opening versus motor command; simulation 0=close,
1=open is not a Feetech conversion. Calibrate head and wrist camera intrinsics,
Camera→Task extrinsics and timestamps with the team's physical setup. Verify
per-object depth coverage and units (stored simulated depth is millimetres)
before changing augmentation. Run offline recorded-observation replay and bounded
dry-run joint checks; physical closed-loop deployment still needs user approval.
