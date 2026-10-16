from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from ._GLOBAL_CONFIGS import *


class handover_block_so101_fixed(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.block_start_pose = sapien.Pose(
            p=[-0.15, 0.20, 0.841],
            q=[0.981, 0.0, 0.0, 0.195],
        )

        self.box = create_box(
            scene=self,
            pose=self.block_start_pose,
            half_size=(0.03, 0.03, 0.10),
            color=(1, 0, 0),
            name="box",
            boxtype="long",
        )

        self.target_pose = sapien.Pose(
            p=[0.15, 0.10, 0.746],
            q=[1.0, 0.0, 0.0, 0.0],
        )

        self.target_box = create_box(
            scene=self,
            pose=self.target_pose,
            half_size=(0.05, 0.05, 0.005),
            color=(0, 0, 1),
            name="target_box",
            is_static=True,
        )

        self.add_prohibit_area(self.box, padding=0.10)
        self.add_prohibit_area(self.target_box, padding=0.10)

        self.block_middle_pose = [
            0.0,
            -0.02,
            0.90,
            0.0,
            1.0,
            0.0,
            0.0,
        ]

    def play_once(self):
        grasp_arm_tag = ArmTag("left")
        handover_arm_tag = ArmTag("right")

        print()
        print("========================================")
        print("[HANDOVER TEST] SO101 fixed handover")
        print("========================================")

        print("[HANDOVER TEST] 1. Left grasp")

        self.move(
            self.grasp_actor(
                self.box,
                arm_tag=grasp_arm_tag,
                pre_grasp_dis=0.07,
                grasp_dis=0.0,
                contact_point_id=[0, 1, 2, 3],
            )
        )

        print("[HANDOVER TEST] 2. Left lift")

        self.move(
            self.move_by_displacement(
                arm_tag=grasp_arm_tag,
                z=0.10,
                move_axis="world",
            )
        )

        print("[HANDOVER TEST] 3. Move to handover pose")

        self.move(
            self.place_actor(
                self.box,
                target_pose=self.block_middle_pose,
                arm_tag=grasp_arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
                constrain="free",
            )
        )

        print("[HANDOVER TEST] 4. Right grasp")

        self.move(
            self.grasp_actor(
                self.box,
                arm_tag=handover_arm_tag,
                pre_grasp_dis=0.07,
                grasp_dis=0.0,
                contact_point_id=[4, 5, 6, 7],
            )
        )

        print("[HANDOVER TEST] 5. Left release")

        self.move(
            self.open_gripper(grasp_arm_tag)
        )

        print("[HANDOVER TEST] 6. Left retreat")

        self.move(
            self.move_by_displacement(
                arm_tag=grasp_arm_tag,
                z=0.08,
                move_axis="arm",
            )
        )

        right_before = self.box.get_pose().p.copy()

        print("[HANDOVER TEST] 7. Right lift validation")

        self.move(
            self.move_by_displacement(
                arm_tag=handover_arm_tag,
                z=0.05,
                move_axis="world",
            )
        )

        right_after = self.box.get_pose().p.copy()

        self.handover_delta_z = float(
            right_after[2] - right_before[2]
        )

        self.handover_success = self.handover_delta_z > 0.03

        print(
            "[HANDOVER TEST] object_z before =",
            f"{right_before[2]:.6f}",
        )
        print(
            "[HANDOVER TEST] object_z after =",
            f"{right_after[2]:.6f}",
        )
        print(
            "[HANDOVER TEST] delta_z =",
            f"{self.handover_delta_z:.6f}",
        )
        print(
            "[HANDOVER TEST] success =",
            self.handover_success,
        )

        return self.info

    def get_info(self):
        return self.info["info"]

    def check_success(self):
        return bool(
            getattr(
                self,
                "handover_success",
                False,
            )
        )
