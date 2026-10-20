# Validation evidence, 2026-10-20

Source branch head: 0920801b210293f44b32570b48e6047111ee9b9a.

CPU command (Python 3.12, CadQuery 2.8.0 / OCP 7.9.3.1.1):

```bash
python -m pytest -q \
 tests/test_sweepick_camera_candidate.py \
 tests/test_sweepick_measured_contract.py \
 tests/test_sweepick_camera_toggle.py \
 tests/test_sweepick_unit_stl.py \
 tests/test_sweepick_urdf_handoff.py \
 tests/test_sweepick_workspace_grid100.py \
 tests/test_sweepick_workspace_grid50.py
```

Result: **58 passed** (10 new + 48 existing). BRep audit completed,
all three exported mesh surfaces watertight. Mesh extents agree with STEP
within 0.01mm and summed mesh volume within 0.1%. Export linear deflection
0.00005m, angular deflection 0.05rad; normalized STL and URDF scale=1.
`cad_audit.json` and `seating_probe_results.json` record exact CAD results.

`python -m compileall -q tools/sweepick_camera_candidate` passed.
Generated Xacro and URDF transforms compared in tests. Portable combined
URDF tested using a minimal chassis fixture plus the tracked original arm
for compatibility; the actual external chassis/measured arm were unavailable.
This fixture is not a substitute model and is not shipped as the robot.

An attempted broader run including `tests/test_sweepick_contract.py`
failed at collection because `torch` is not installed. That module is
excluded from the 58 passing tests; no claim of complete repository CI.

Not executed: ROS2 check_urdf/RViz, GPU renderer SAPIEN, CuRobo planning,
real ROS chassis merge, hardware optical calibration or collision safety.
The optical centre remains null, so the shipped candidate omits optical TF
and a simulated optical extrinsic rather than fabricating either.

Follow-up: user confirmed X long-axis, rear M3 direct mount and requested 25deg downward from horizontal. New test rejects reviewed extrinsic output while the CAD candidate remains at 65deg. Assembly photograph and optical calibration remain pending.

Mechanical-only mode verified with synthetic optical calibration present: optical TF and SAPIEN extrinsic remain omitted, body CAD transform preserved, inputs unchanged and mesh paths portable.

Received SDK-data follow-up: 68 CPU tests passed (10 new received-camera tests + 58 previous). Real depth PNG FNV matches trace record 6. Received profile URDF chains reproduce both supplied optical matrices; full K adapters tested with a stub. Physical optical calibration and GPU runtime remain unverified.
