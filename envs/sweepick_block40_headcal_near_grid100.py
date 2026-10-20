"""Diagnostic per-cell Sweepick Grasp/Lift workspace scanner task.

Unlike RoboTwin's collect_data success-only seed selection, caller
runs each cell and arm once and logs failures as well as successes.
This is NOT a production imitation-learning demonstration dataset.
"""
import numpy as np

from .sweepick_block40_headcal import sweepick_block40_headcal
from .utils import ArmTag
from sweepick_workspace_near_grid100 import cell, NUM_CELLS, CROP_MIN, CROP_MAX


class sweepick_block40_headcal_near_grid100(sweepick_block40_headcal):
    # Only this diagnostic task uses the new crop: no legacy data/ACT changes.
    CROP_MIN = CROP_MIN.copy()
    CROP_MAX = CROP_MAX.copy()
    def setup_demo(self, **kwargs):
        self.grid_id = int(kwargs.get("grid_index", 0))
        self.scan_arm = kwargs.get("scan_arm", "left")
        if self.scan_arm not in ("left", "right"):
            raise ValueError(f"Invalid scan arm: {self.scan_arm}")
        self.BLOCK_ROBOT_XY = np.asarray(cell(self.grid_id), dtype=np.float64)
        self.scan_metrics = {
            "grid_id": self.grid_id,
            "arm": self.scan_arm,
            "pregrasp_pose_found": False,
            "pregrasp_plan_ok": False,
            "grasp_plan_ok": False,
            "gripper_close_ok": False,
            "lift_plan_ok": False,
            "lift_delta_z_m": None,
            "lift_success": False,
            "failure_stage": "not_started",
        }
        super().setup_demo(**kwargs)

    def load_actors(self):
        super().load_actors()
        print(
            f"[NEAR GRID100] id={self.grid_id:02d}/{NUM_CELLS - 1} "
            f"arm={self.scan_arm} base_xy={self.BLOCK_ROBOT_XY.tolist()} "
            f"world_xyz={np.round(self.start_pose.p, 5).tolist()}"
        )

    def _finish(self):
        m = self.scan_metrics
        self.info["info"] = {"{A}": "Sweepick 40mm cube Workspace grid test"}
        self.info["metrics"] = dict(m)
        print(f"[NEAR GRID100] id={self.grid_id:02d} arm={self.scan_arm} "
              f"lift={m['lift_success']} failed_at={m['failure_stage']}")
        return self.info

    def play_once(self):
        arm = ArmTag(self.scan_arm)
        m = self.scan_metrics
        m["failure_stage"] = "select_grasp_pose"
        pre, grasp = self.choose_grasp_pose(
            self.object,
            arm_tag=arm,
            pre_dis=.10,
            target_dis=0.0,
            contact_point_id=None,
        )
        if pre is None or grasp is None:
            return self._finish()
        m["pregrasp_pose_found"] = True

        m["failure_stage"] = "pregrasp_plan"
        self.move(self.move_to_pose(arm_tag=arm, target_pose=pre))
        if not self.plan_success:
            return self._finish()
        m["pregrasp_plan_ok"] = True

        m["failure_stage"] = "grasp_plan"
        self.move(self.move_to_pose(arm_tag=arm, target_pose=grasp))
        if not self.plan_success:
            return self._finish()
        m["grasp_plan_ok"] = True

        m["failure_stage"] = "close_gripper"
        self.move(self.close_gripper(arm_tag=arm, pos=0.0))
        if not self.plan_success:
            return self._finish()
        m["gripper_close_ok"] = True

        before_z = float(self.object.get_pose().p[2])
        m["failure_stage"] = "lift_plan"
        self.move(self.move_by_displacement(arm_tag=arm, z=.05, move_axis="world"))
        if not self.plan_success:
            return self._finish()
        m["lift_plan_ok"] = True
        dz = float(self.object.get_pose().p[2]) - before_z
        m["lift_delta_z_m"] = dz
        m["lift_success"] = bool(dz > .03)
        m["failure_stage"] = "" if m["lift_success"] else "lift_under_30mm"
        self.lift_success = bool(m["lift_success"])
        return self._finish()
