# Marker v3 instruction hotfix

Fixes collection crash after a successful trajectory.

Cause:
- description/task_instruction/move_marker_randomized.json was missing.
- numeric diagnostics had also been placed inside `info`, which is consumed by
  RoboTwin's language-instruction generator and expects string-valued parameters.

Fix:
- adds move_marker_randomized.json
- keeps instruction-generation `info` minimal/string-only
- moves grid/yaw/success diagnostics to sibling `metrics` in scene_info.json
