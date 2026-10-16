# Sim-to-real alignment handoff

The GitHub migration preserves a simulation baseline. It does not certify the
real robot's kinematics, sensors or closed-loop policy.

| Contract | Simulation evidence | Real-machine measurement needed |
|---|---|---|
| Joint/action order | simulation-joint-contract.json; left 5+1 then right 5+1 | Physical motor IDs and order, radian mapping, signs, zero offsets, limits |
| Arm mounts | config.yml robot_pose plus task embodiment spacing | Measured base→left/right transforms; left height 140 vs 124 mm unresolved |
| Wrist roll zero | active runtime/planner origin rpy 3.141592654,1.570796327,0 | Multiple-pose FK/TCP comparison using measured markers |
| Gripper | rev_motor_08 0 open, 2.2028 close; ±.0115 prismatic mimic | Measured jaw aperture vs Feetech motor position, mechanical travel/limits |
| Head camera | current task configs select D435 or OS30A; YAML is simulated sensor geometry | Actual OS30A intrinsics, distortion, depth unit, bias and Camera→Task matrix |
| Wrist cameras | URDF camera chain, 240x320 tensors in v4 corpus | Real camera mount poses, intrinsics, RGB channel marker and synchronized timestamps |
| Task frame | World origin (0,-.35,.74), yaw +90°, fixed simulation crop | Measured real task origin/axes and independent extrinsic calibration |
| Depth coverage | saved raw head depth/PCD schema in validation/reference.json | Foreground/object ROI depth-valid fraction; whole-image 9–12% is insufficient |

`dual_arm: false` in one embodiment describes one articulation. The task config
instantiates that embodiment twice. It is not evidence of a single-arm policy.
URDF jaw mimic tags are absent: runtime robot.py applies config.yml mimic rules.

For the next team run, save robot serial/firmware, calibration date, motor
mapping, measured mount matrices and head/wrist calibration with the recordings.
Compare FK at several safe configurations; use recorded observations to check
PCD transforms/crop, wrist RGB convention and normalized 12D actions. Only then
perform bounded dry-run commands and separately approved real closed-loop tests.
Unresolved RGB A/B, temporal ensemble weighting and v4 policy success remain
separate from the numerical compatibility demonstrated in this migration.
