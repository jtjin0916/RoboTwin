import numpy as np
import sapien
import transforms3d as t3d

from .move_woodenblock_randomized import (
    move_woodenblock_randomized,
)
from .utils import *
from .utils.actor_utils import Actor


class move_slim_can_randomized(
    move_woodenblock_randomized
):
    """
    Slim-can Pick -> Lift -> Stow task for dual SO-101.

    Design basis:
      - 50-position Task-frame grid inherited from
        move_woodenblock_randomized.
      - RoboTwin-style Actor contact-point grasp annotations.
      - Side-grasp-only contact poses for a cylindrical object.
      - Official move_can_pot-style grasp_actor() flow.
      - Existing DAPIER joint-space return-to-Stow motion.

    Can dimensions are intentionally smaller than the roughly
    50 mm SO-101 gripper opening used in this project.
    """

    CAN_DIAMETER = 0.040
    CAN_RADIUS = CAN_DIAMETER / 2.0
    CAN_HEIGHT = 0.110
    CAN_HALF_HEIGHT = CAN_HEIGHT / 2.0

    # Official move_can_pot uses a 5 cm pre-grasp distance.
    PRE_GRASP_DIS = 0.05
    LIFT_HEIGHT = 0.05

    # Slightly above the geometric center to increase table clearance.
    SIDE_GRASP_HEIGHT = 0.010

    @staticmethod
    def _side_contact_matrix(rotation, z_offset):
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.asarray(
            rotation,
            dtype=np.float32,
        )
        mat[:3, 3] = np.array(
            [0.0, 0.0, z_offset],
            dtype=np.float32,
        )
        return mat.tolist()

    def _create_slim_can(self, pose):
        """
        Create an upright procedural cylinder and wrap it in
        RoboTwin's Actor class so choose_grasp_pose/grasp_actor
        can use explicit side-grasp contact annotations.

        PhysX cylinders are aligned with local +X. The cylinder
        collision/visual is therefore rotated +90 deg around Y
        inside an otherwise world-aligned actor frame, making the
        cylinder axis vertical (+/-Z) while keeping the Actor frame
        convenient for grasp annotations.
        """

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")

        cylinder_local_pose = sapien.Pose(
            p=[0.0, 0.0, 0.0],
            q=t3d.euler.euler2quat(
                0.0,
                np.pi / 2.0,
                0.0,
            ),
        )

        builder.add_cylinder_collision(
            pose=cylinder_local_pose,
            radius=self.CAN_RADIUS,
            half_length=self.CAN_HALF_HEIGHT,
            material=self.scene.default_physical_material,
            density=500.0,
        )

        builder.add_cylinder_visual(
            pose=cylinder_local_pose,
            radius=self.CAN_RADIUS,
            half_length=self.CAN_HALF_HEIGHT,
            material=(0.72, 0.16, 0.10),
            name="slim_can_visual",
        )

        builder.set_initial_pose(pose)
        entity = builder.build(
            name="slim_can_40mm"
        )

        # These side-contact rotations are the same orientation
        # convention used by RoboTwin's box side-grasp annotations.
        # No top-down grasp annotation is provided, so the expert
        # selects a side grasp closest to the arm's preferred side
        # direction.
        side_contacts = [
            self._side_contact_matrix(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                self.SIDE_GRASP_HEIGHT,
            ),
            self._side_contact_matrix(
                [
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                ],
                self.SIDE_GRASP_HEIGHT,
            ),
            self._side_contact_matrix(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ],
                self.SIDE_GRASP_HEIGHT,
            ),
            self._side_contact_matrix(
                [
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                self.SIDE_GRASP_HEIGHT,
            ),
        ]

        actor_data = {
            "center": [0.0, 0.0, 0.0],
            # Full XYZ dimensions. add_prohibit_area() divides by 2.
            "extents": [
                self.CAN_DIAMETER,
                self.CAN_DIAMETER,
                self.CAN_HEIGHT,
            ],
            # Contact-point translations are already in meters.
            "scale": 1.0,
            "contact_points_pose": side_contacts,
            "contact_points_description": [
                "side grasp 0",
                "side grasp 1",
                "side grasp 2",
                "side grasp 3",
            ],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True],
            "target_pose": [np.eye(4).tolist()],
            "functional_matrix": [],
            "transform_matrix": np.eye(4).tolist(),
        }

        return Actor(
            entity,
            actor_data,
            mass=0.03,
        )

    def load_actors(self):
        # -------------------------------------------------
        # 50-position grid inherited from cube task
        # -------------------------------------------------
        self.grid_index = (
            self.collection_seed
            % self.NUM_GRID_CELLS
        )

        num_lateral = len(
            self.TASK_Y_CENTERS
        )
        forward_idx = (
            self.grid_index // num_lateral
        )
        lateral_idx = (
            self.grid_index % num_lateral
        )

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

        # Task -> World
        # X_T = Y_W + 0.35
        # Y_T = -X_W
        # => X_W = -Y_T
        # => Y_W = X_T - 0.35
        start_x = -self.task_y
        start_y = self.task_x - 0.35

        self.start_pose = sapien.Pose(
            p=[
                start_x,
                start_y,
                0.741 + self.CAN_HALF_HEIGHT,
            ],
            q=[1.0, 0.0, 0.0, 0.0],
        )

        self.object = self._create_slim_can(
            self.start_pose
        )
        self.can = self.object

        self.add_prohibit_area(
            self.object,
            padding=0.04,
        )

        # -------------------------------------------------
        # Nearest-arm selection, same policy as cube task
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
        self.return_success = False
        self.hold_success = False

        print(
            "[SLIM CAN] ==========================="
        )
        print(
            "[SLIM CAN] episode/grid =",
            f"{self.collection_index}/"
            f"{self.grid_index}",
        )
        print(
            "[SLIM CAN] seed =",
            self.collection_seed,
        )
        print(
            "[SLIM CAN] diameter/height =",
            f"{self.CAN_DIAMETER:.3f} / "
            f"{self.CAN_HEIGHT:.3f} m",
        )
        print(
            "[SLIM CAN] Task XY =",
            np.round(
                [self.task_x, self.task_y],
                4,
            ),
        )
        print(
            "[SLIM CAN] World XYZ =",
            np.round(
                self.object.get_pose().p,
                4,
            ),
        )
        print(
            "[SLIM CAN] arm =",
            self.active_arm_name,
        )
        print(
            "[SLIM CAN] arm distance L/R =",
            round(left_distance, 4),
            "/",
            round(right_distance, 4),
        )
        print(
            "[SLIM CAN] ==========================="
        )

    def play_once(self):
        arm_tag = self.active_arm_tag

        # -------------------------------------------------
        # Side grasp
        # -------------------------------------------------
        print(
            "[CAN GRASP] Side grasp "
            f"(pre={self.PRE_GRASP_DIS:.3f} m)"
        )

        self.move(
            self.grasp_actor(
                self.object,
                arm_tag=arm_tag,
                pre_grasp_dis=(
                    self.PRE_GRASP_DIS
                ),
                grasp_dis=0.0,
                gripper_pos=0.0,
            )
        )

        if not self.plan_success:
            print(
                "[CAN GRASP] Planning failed"
            )
            return self.info

        self.print_gripper_state(
            "CAN AFTER CLOSE"
        )

        object_z_before_lift = float(
            self.object.get_pose().p[2]
        )

        # -------------------------------------------------
        # Lift
        # -------------------------------------------------
        print(
            "[CAN LIFT] Lift =",
            self.LIFT_HEIGHT,
            "m",
        )

        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=self.LIFT_HEIGHT,
                move_axis="world",
            )
        )

        if not self.plan_success:
            print(
                "[CAN LIFT] Planning failed"
            )
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
            "[CAN LIFT] delta_z =",
            round(delta_z, 6),
        )
        print(
            "[CAN LIFT] success =",
            self.lift_success,
        )

        if not self.lift_success:
            return self.info

        # -------------------------------------------------
        # Joint-space return to configured Stow pose.
        # No gripper command is sent here, so the grasp
        # command remains closed while the arm returns.
        # -------------------------------------------------
        if arm_tag == "left":
            stow_q = list(
                self.robot.left_homestate
            )
        else:
            stow_q = list(
                self.robot.right_homestate
            )

        print(
            "[CAN RETURN] Return to Stow"
        )
        print(
            "[CAN RETURN] target q =",
            np.round(stow_q, 4),
        )

        self.return_success = (
            self.move_to_joint_state(
                arm_tag=arm_tag,
                target_joint_pos=stow_q,
            )
        )

        if not self.return_success:
            print(
                "[CAN RETURN] Joint planning failed"
            )
            return self.info

        final_position = (
            self.object.get_pose().p
        )

        # If the can falls during the return, it should end
        # close to its original table height. Require it to
        # remain clearly elevated after Stow.
        self.hold_success = (
            float(final_position[2])
            - float(self.start_pose.p[2])
            > 0.03
        )

        print(
            "[CAN RETURN] Stow complete"
        )
        print(
            "[CAN RETURN] final can XYZ =",
            np.round(final_position, 4),
        )
        print(
            "[CAN RETURN] hold_success =",
            self.hold_success,
        )

        self.print_gripper_state(
            "CAN AFTER RETURN"
        )

        self.info["info"] = {
            "{A}": (
                "Slim can side-grasp, lift, "
                "and return-to-stow test"
            ),
            "grid_index": int(
                self.grid_index
            ),
            "seed": int(
                self.collection_seed
            ),
            "active_arm": (
                self.active_arm_name
            ),
            "task_x": float(self.task_x),
            "task_y": float(self.task_y),
            "can_diameter": float(
                self.CAN_DIAMETER
            ),
            "can_height": float(
                self.CAN_HEIGHT
            ),
            "start_x": float(
                self.start_pose.p[0]
            ),
            "start_y": float(
                self.start_pose.p[1]
            ),
            "final_x": float(
                final_position[0]
            ),
            "final_y": float(
                final_position[1]
            ),
            "final_z": float(
                final_position[2]
            ),
            "lift_success": bool(
                self.lift_success
            ),
            "return_success": bool(
                self.return_success
            ),
            "hold_success": bool(
                self.hold_success
            ),
        }

        return self.info

    def check_success(self):
        return bool(
            getattr(
                self,
                "lift_success",
                False,
            )
            and getattr(
                self,
                "return_success",
                False,
            )
            and getattr(
                self,
                "hold_success",
                False,
            )
        )
