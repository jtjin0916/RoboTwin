# Migration changes — 2026-10-16

Working source directories remain untouched. Two independent local clones retain
RoboTwin and XPolicyLab history; forks use feature branches, no force push.

Preserved working changes: task base, SO-101 planner fallback and robot/gripper
handling, camera, HDF5 pointcloud/depth packing, evaluation path, embodiment and
camera registry, 6 task implementations and selected existing task configurations,
custom ACT model/dataset/training core, XPolicyLab adapter and venv launcher.
`source-manifest.json` maps exact local copies and public/reference counterparts.

Packaging changes: relative CuRobo paths resolved beside YAML; SO-101 entry added
to XPolicyLab shell training registry; portable FK diagnostic; clarified adapter
README (train/process scripts are historical stubs); pinned fork submodule;
static/simulator-smoke and replay scripts; observed dependency and artifact
manifests; licenses and exclusion rules. The upstream README is retained with a
branch-specific entry point. The local marker README is saved separately.

Active gripper/URDF and ACT core behavior are unchanged. No new augmentation,
channel correction, ensemble changes, physical extrinsics or motor mapping is
introduced. Runtime/planner arm FK metadata matches, but planner continuous joint
types/legacy gripper frame are intentionally preserved and reported.

Excluded: local backups/debug variants, venv, 16 GB complete asset tree,
checkpoints (242 MB), local data (1.8 GB plus external SSD symlinks), logs/MP4,
PointTransformerV3, CuRobo source, SDKs, unreferenced meshes. No binary artifact
rights or public hosting are inferred. Only active 22 audited STL meshes ship.

Initial status evidence, strict-load, parity, syntax/static contract results and
GPU/Vulkan failure evidence are in validation/. Final commit SHA is available
from `git rev-parse HEAD`; XPolicyLab pin is in the Git tree and source manifest.
