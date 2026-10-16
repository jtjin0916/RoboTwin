import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *


class move_woodenblock_fixed(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.block_size = 0.04
        self.block_half_size = self.block_size / 2.0

        self.start_pose = sapien.Pose(
            p=[
                -0.15,
                -0.10,
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

        self.target_position = np.array(
            [
                -0.02,
                -0.10,
                0.741
                + self.block_half_size
                + self.table_z_bias,
            ],
            dtype=np.float64,
        )

        self.lift_success = False

        print("[FIXED TEST] =========================")
        print("[FIXED TEST] block = 4 cm cube")
        print(
            "[FIXED TEST] size =",
            [self.block_size] * 3,
            "m",
        )
        print(
            "[FIXED TEST] start =",
            np.round(self.object.get_pose().p, 4),
        )
        print(
            "[FIXED TEST] target =",
            np.round(self.target_position, 4),
        )
        print("[FIXED TEST] =========================")

    def print_gripper_state(self, label):
        active_joints = self.robot.left_entity.get_active_joints()
        qpos = self.robot.left_entity.get_qpos()

        print()
        print("[GRIPPER DEBUG] =========================")
        print("[GRIPPER DEBUG]", label)
        print(
            "[GRIPPER DEBUG] normalized_value =",
            self.robot.get_left_gripper_val(),
        )

        for joint_info in self.robot.left_gripper:
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
                round(float(qpos[joint_index]), 6),
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
        arm_tag = ArmTag("left")

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
            round(object_z_before_lift, 6),
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

        self.lift_success = delta_z > 0.03

        print(
            "[LIFT TEST] object_z after =",
            round(object_z_after_lift, 6),
        )

        print(
            "[LIFT TEST] delta_z =",
            round(delta_z, 6),
        )

        print(
            "[LIFT TEST] success =",
            self.lift_success,
        )

        self.print_gripper_state("AFTER LIFT")

        self.info["info"] = {
            "{A}": "Grasp and lift test"
        }

        return self.info

    def check_success(self):
        print(
            "[LIFT TEST] final success =",
            self.lift_success,
        )

        return self.lift_success
