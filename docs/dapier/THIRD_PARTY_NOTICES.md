# Attribution and release scope

RoboTwin upstream history and MIT LICENSE are retained. XPolicyLab history and
Apache-2.0 LICENSE remain in the forked pinned submodule. DAPIER ACT files are
behavior-preserving copies of the user's working implementation, corresponding
to jtjin0916/DAPIER-ACT afd3808; its MIT license is retained in licenses/.
No PointTransformerV3 tree, external ACT/DP3 repository, pretrained weights,
camera SDK or NVIDIA CuRobo source is vendored in this change. The DP3-style
point encoder name alone does not establish external code provenance.

SO-101 meshes: TheRobotStudio/SO-ARM100 at a758567c3978dfeefe282ede0500085a48fe8f78,
Simulation/SO101/assets, Apache-2.0. Exact Git blob matches verified.
NormaCore meshes: norma-core/norma-core at 0d17bb300d712eeb5437d41ddb95ce380b5384e6,
hardware/elrobot (including simulation/assets and STL/gripper_camera_mounts),
Apache-2.0 under hardware/elrobot/LICENSE. The hardware/pgripper Apache-2.0
license is also retained. Do not infer asset terms from NormaCore root MIT.
Exact filenames and hashes are in the embodiment ASSET_PROVENANCE.json.
Converted meter meshes have the same triangle layout and coordinates scaled
by 0.001; numeric comparison passed. DAPIER modifications are described in the
adjacent NOTICE.md. URDF structure follows SO-101; local gripper, collision,
calibration and camera edits are retained without claiming ownership of sources.

Only the 22 mesh files referenced by active runtime/planner URDFs are included.
Unreferenced meshes, legacy/debug URDFs, SDKs, PointTransformerV3, virtualenv,
datasets, checkpoint/stat pickle, logs, videos and backups are excluded.
The complete 16 GB local asset collection is not redistributed. These checks
cover selected sources; they are not a new license grant for external artifacts.
