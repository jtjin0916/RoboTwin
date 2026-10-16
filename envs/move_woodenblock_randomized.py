import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *


class move_woodenblock_randomized(Base_Task):

    # =========================================================
    # Stratified 50-position dataset
    #
    # Task frame:
    #   +X = robot forward
    #   +Y = robot left
    #
    # 10 forward positions x 5 lateral positions = 50 cells
    # =========================================================

    TASK_X_CENTERS = np.linspace(
        0.18,
        0.26,
        10,
    )

    # Interleaved ordering:
    # right / left / right / left / center
    # so both arms are exercised from early episodes.
    TASK_Y_CENTERS = np.array(
        [
            -0.08,
            +0.08,
            -0.04,
            +0.04,
            0.0,
        ],
        dtype=np.float32,
    )

    JITTER_X = 0.0
    JITTER_Y = 0.0

    # Current dual-SO101 base positions in world frame.
    LEFT_BASE_WORLD_XY = np.array(
        [-0.15, -0.35],
        dtype=np.float32,
    )

    RIGHT_BASE_WORLD_XY = np.array(
        [+0.15, -0.35],
        dtype=np.float32,
    )

    NUM_GRID_CELLS = (
        len(TASK_X_CENTERS)
        * len(TASK_Y_CENTERS)
    )

    def setup_demo(self, **kwargs):
        # RoboTwin supplies these values for both seed search
        # and final data collection.
        self.collection_index = int(
            kwargs.get("now_ep_num", 0)
        )

        self.collection_seed = int(
            kwargs.get("seed", 0)
        )

        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.block_size = 0.04
        self.block_half_size = (
            self.block_size / 2.0
        )

        # -------------------------------------------------
        # Grid cell
        # -------------------------------------------------

        # Use seed as candidate index.
        # If a position cannot be planned, RoboTwin increments
        # the seed and automatically moves to the next grid cell.
        self.grid_index = (
            self.collection_seed
            % self.NUM_GRID_CELLS
        )

        num_lateral = len(self.TASK_Y_CENTERS)
        forward_idx = self.grid_index // num_lateral
        lateral_idx = self.grid_index % num_lateral

        task_x_center = float(
            self.TASK_X_CENTERS[
                forward_idx
            ]
        )

        task_y_center = float(
            self.TASK_Y_CENTERS[
                lateral_idx
            ]
        )

        # Use a local RNG derived only from the RoboTwin seed.
        # Therefore seed-search and final replay generate
        # exactly the same jittered position.
        rng = np.random.default_rng(
            self.collection_seed
        )

        jitter_x = float(
            rng.uniform(
                -self.JITTER_X,
                +self.JITTER_X,
            )
        )

        jitter_y = float(
            rng.uniform(
                -self.JITTER_Y,
                +self.JITTER_Y,
            )
        )

        self.task_x = (
            task_x_center + jitter_x
        )

        self.task_y = (
            task_y_center + jitter_y
        )

        # -------------------------------------------------
        # Task -> World
        #
        # X_T = Y_W + 0.35
        # Y_T = -X_W
        #
        # therefore:
        # X_W = -Y_T
        # Y_W = X_T - 0.35
        # -------------------------------------------------

        start_x = -self.task_y
        start_y = self.task_x - 0.35

        self.start_pose = sapien.Pose(
            p=[
                start_x,
                start_y,
                0.741 + self.block_half_size,
            ],
            q=[1.0, 0.0, 0.0, 0.0],
        )

        self.object = create_box(
            scene=self,
            pose=self.start_pose,
            half_size=(
                self.block_half_size,
                self.block_half_size,
                self.block_half_size,
            ),
            color=(0.72, 0.45, 0.20),
            name="wooden_block_4cm",
        )

        # -------------------------------------------------
        # Nearest arm selection
        # -------------------------------------------------

        object_xy = np.array(
            [start_x, start_y],
            dtype=np.float32,
        )

        left_distance = float(
            np.linalg.norm(
                object_xy
                - self.LEFT_BASE_WORLD_XY
            )
        )

        right_distance = float(
            np.linalg.norm(
                object_xy
                - self.RIGHT_BASE_WORLD_XY
            )
        )

        if left_distance <= right_distance:
            self.active_arm_name = "left"
        else:
            self.active_arm_name = "right"

        self.active_arm_tag = ArmTag(
            self.active_arm_name
        )

        self.lift_success = False

        print("[GRID TEST] ===========================")
        print(
            "[GRID TEST] episode/grid =",
            f"{self.collection_index}/"
            f"{self.grid_index}",
        )
        print(
            "[GRID TEST] seed =",
            self.collection_seed,
        )
        print(
            "[GRID TEST] Task center =",
            np.round(
                [
                    task_x_center,
                    task_y_center,
                ],
                4,
            ),
        )
        print(
            "[GRID TEST] jitter =",
            np.round(
                [
                    jitter_x,
                    jitter_y,
                ],
                4,
            ),
        )
        print(
            "[GRID TEST] Task XY =",
            np.round(
                [
                    self.task_x,
                    self.task_y,
                ],
                4,
            ),
        )
        print(
            "[GRID TEST] World XYZ =",
            np.round(
                self.object.get_pose().p,
                4,
            ),
        )
        print(
            "[GRID TEST] arm =",
            self.active_arm_name,
        )
        print(
            "[GRID TEST] arm distance L/R =",
            round(left_distance, 4),
            "/",
            round(right_distance, 4),
        )
        print("[GRID TEST] ===========================")

    def print_gripper_state(self, label):
        if self.active_arm_name == "left":
            entity = self.robot.left_entity
            gripper_value = (
                self.robot.get_left_gripper_val()
            )
            gripper_cfg = self.robot.left_gripper
        else:
            entity = self.robot.right_entity
            gripper_value = (
                self.robot.get_right_gripper_val()
            )
            gripper_cfg = self.robot.right_gripper

        active_joints = entity.get_active_joints()
        qpos = entity.get_qpos()

        print()
        print("[GRIPPER DEBUG] =========================")
        print("[GRIPPER DEBUG]", label)
        print(
            "[GRIPPER DEBUG] normalized_value =",
            gripper_value,
        )

        for joint_info in gripper_cfg:
            joint = joint_info[0]
            multiplier = joint_info[1]
            offset = joint_info[2]

            joint_index = active_joints.index(joint)

            target = float(
                np.asarray(
                    joint.get_drive_target()
                ).reshape(-1)[0]
            )

            print(
                "[GRIPPER DEBUG]",
                joint.get_name(),
                "qpos=",
                round(
                    float(qpos[joint_index]),
                    6,
                ),
                "target=",
                round(target, 6),
                "multiplier=",
                multiplier,
                "offset=",
                offset,
            )

        print("[GRIPPER DEBUG] =========================")
        print()

    def play_once(self):
        arm_tag = self.active_arm_tag
        self.return_success = False

        pre_grasp_pose, grasp_pose = self.choose_grasp_pose(
            self.object,
            arm_tag=arm_tag,
            pre_dis=0.10,
            target_dis=0.0,
            contact_point_id=None,
        )

        print("[GRASP TEST] Move to pre-grasp")

        self.move(
            (
                arm_tag,
                [
                    Action(
                        arm_tag,
                        "move",
                        target_pose=pre_grasp_pose,
                    )
                ],
            )
        )

        if not self.plan_success:
            return self.info

        print("[GRASP TEST] Move to grasp")

        self.move(
            (
                arm_tag,
                [
                    Action(
                        arm_tag,
                        "move",
                        target_pose=grasp_pose,
                    )
                ],
            )
        )

        if not self.plan_success:
            return self.info

        self.print_gripper_state("BEFORE CLOSE")

        print("[GRASP TEST] Close gripper")

        self.move(
            (
                arm_tag,
                [
                    Action(
                        arm_tag,
                        "close",
                        target_gripper_pos=0.0,
                    )
                ],
            )
        )

        if not self.plan_success:
            return self.info

        self.print_gripper_state("AFTER CLOSE")

        object_z_before_lift = float(
            self.object.get_pose().p[2]
        )

        print(
            "[LIFT TEST] object_z before =",
            round(
                object_z_before_lift,
                6,
            ),
        )

        print("[LIFT TEST] Lift 5 cm")

        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=0.05,
                move_axis="world",
            )
        )

        if not self.plan_success:
            print("[LIFT TEST] Planning failed")
            return self.info

        object_z_after_lift = float(
            self.object.get_pose().p[2]
        )

        delta_z = (
            object_z_after_lift
            - object_z_before_lift
        )

        self.lift_success = (
            delta_z > 0.03
        )

        print(
            "[LIFT TEST] object_z after =",
            round(
                object_z_after_lift,
                6,
            ),
        )

        print(
            "[LIFT TEST] delta_z =",
            round(
                delta_z,
                6,
            ),
        )

        print(
            "[LIFT TEST] success =",
            self.lift_success,
        )

        self.print_gripper_state("AFTER LIFT")

        if not self.lift_success:
            print(
                "[RETURN TEST] Skip return because lift failed"
            )
            return self.info

        # -------------------------------------------------
        # Return to the configured Stow joint pose while
        # keeping the gripper closed.
        # -------------------------------------------------
        if arm_tag == "left":
            stow_q = list(
                self.robot.left_homestate
            )
        else:
            stow_q = list(
                self.robot.right_homestate
            )

        print("[RETURN TEST] Return to Stow")
        print("[RETURN TEST] arm =", arm_tag)
        print(
            "[RETURN TEST] target q =",
            np.round(stow_q, 4),
        )

        self.return_success = self.move_to_joint_state(
            arm_tag=arm_tag,
            target_joint_pos=stow_q,
        )

        if not self.return_success:
            print(
                "[RETURN TEST] Joint planning failed"
            )
            return self.info

        print(
            "[RETURN TEST] Stow trajectory complete"
        )
        self.print_gripper_state("AFTER RETURN")

        object_position = (
            self.object.get_pose().p
        )

        self.info["info"] = {
            "{A}": "Stratified grid grasp, lift, and return-to-stow test",
            "grid_index": int(self.grid_index),
            "seed": int(self.collection_seed),
            "active_arm": self.active_arm_name,
            "task_x": float(self.task_x),
            "task_y": float(self.task_y),
            "start_x": float(self.start_pose.p[0]),
            "start_y": float(self.start_pose.p[1]),
            "final_x": float(object_position[0]),
            "final_y": float(object_position[1]),
            "final_z": float(object_position[2]),
            "lift_success": bool(self.lift_success),
            "return_success": bool(self.return_success),
        }

        return self.info

    def check_success(self):
        return bool(
            getattr(self, "lift_success", False)
            and getattr(self, "return_success", False)
        )
