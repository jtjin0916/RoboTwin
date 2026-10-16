# SO101-PGripper zero-radian migration

## Active RoboTwin files
- `so101_pgripper.urdf`: SAPIEN/RoboTwin runtime articulation model.
- `so101_pgripper_calib.urdf`: CuRobo planning model referenced by `curobo.yml`.
- `config.yml`: RoboTwin embodiment configuration.
- `curobo.yml`: CuRobo configuration.

## Wrist-roll zero update
Both active URDFs use the same wrist-roll kinematic origin:

```xml
<origin xyz="5.55112e-17 -0.0611 0.0181"
        rpy="3.141592654 1.570796327 0" />
```

The joint axis remains unchanged:

```xml
<axis xyz="0 -1 0" />
```

Updating both runtime and planner URDFs is required so SAPIEN and CuRobo use the same kinematic zero convention.

## Camera chain
The active RoboTwin runtime URDF keeps the existing camera chain unchanged:

`Gripper_Base_v1_1 -> camera_mount_link -> wrist_camera_link -> wrist_camera_optical_frame -> camera`

## Legacy directory
Historical debug snapshots and the old RViz-specific URDF were moved to `legacy/`.
Exact pre-zero active URDFs are preserved as:
- `legacy/so101_pgripper_pre_zero.urdf`
- `legacy/so101_pgripper_calib_pre_zero.urdf`

Do not use the old RViz URDF as the ROS/MoveIt source. Build the ROS/RViz/MoveIt model separately after RoboTwin regression passes.
