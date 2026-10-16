import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *


class move_marker_randomized(Base_Task):

    # =========================================================
    # 50 positions x 10 deterministic yaw angles = 500 cases
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

    # Deterministic orientation sweep:
    # every one of the 50 XY positions gets all 10 yaw angles.
    MARKER_YAW_DEG = np.arange(
        0.0,
        91.0,
        10.0,
        dtype=np.float32,
    )

    NUM_YAWS = len(MARKER_YAW_DEG)
    NUM_CASES = NUM_GRID_CELLS * NUM_YAWS

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

    def _create_marker_actor(self, pose):
        """Create a horizontal cylindrical marker Actor.

        Geometry:
          local +X = marker long axis
          radius   = 9 mm
          length   = 100 mm

        This keeps the visual/collision shape marker-like instead of using
        a rectangular box, while preserving RoboTwin-style grasp metadata.
        """
        scene = self.scene

        entity = sapien.Entity()
        entity.set_name("marker_pen_cylinder_100x18mm")
        entity.set_pose(pose)

        rigid_component = sapien.physx.PhysxRigidDynamicComponent()
        rigid_component.attach(
            sapien.physx.PhysxCollisionShapeCylinder(
                radius=self.marker_radius,
                half_length=self.marker_length / 2.0,
                material=scene.default_physical_material,
            )
        )

        material = sapien.render.RenderMaterial(
            base_color=[0.08, 0.08, 0.08, 1.0]
        )
        material.metallic = 0.05
        material.roughness = 0.45

        render_component = sapien.render.RenderBodyComponent()
        render_component.attach(
            sapien.render.RenderShapeCylinder(
                radius=self.marker_radius,
                half_length=self.marker_length / 2.0,
                material=material,
            )
        )

        entity.add_component(rigid_component)
        entity.add_component(render_component)
        scene.add_entity(entity)
        entity.set_pose(pose)

        # Same top-down orientation convention used by RoboTwin create_box().
        # The cylinder's long axis is local X. Contact ids 1/2 are kept so
        # the gripper pinches across the short local-Y direction.
        contact_points_pose = [
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0.0],
                [0, 0, 0, 1],
            ],
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0.0],
                [0, 0, 0, 1],
            ],
            [
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0.0],
                [0, 0, 0, 1],
            ],
            [
                [0, 0, -1, 0],
                [-1, 0, 0, 0],
                [0, 1, 0, 0.0],
                [0, 0, 0, 1],
            ],
        ]

        marker_data = {
            "center": [0, 0, 0],
            "extents": [
                self.marker_length / 2.0,
                self.marker_radius,
                self.marker_radius,
            ],
            "scale": [1.0, 1.0, 1.0],
            "target_pose": [
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1],
                ]
            ],
            "contact_points_pose": contact_points_pose,
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [],
            "contact_points_description": [
                "top-down marker grasp 0",
                "top-down marker grasp 1",
                "top-down marker grasp 2",
                "top-down marker grasp 3",
            ],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True],
            "target_point_description": [],
        }

        marker = Actor(entity, marker_data)
        marker.set_mass(0.02)
        return marker

    def load_actors(self):
        # -------------------------------------------------
        # Marker geometry
        #
        # Marker-like cylindrical rigid body:
        #   length   = 100 mm
        #   diameter = 18 mm
        #   local +X = long axis
        # -------------------------------------------------
        self.marker_length = 0.100
        self.marker_diameter = 0.018
        self.marker_radius = self.marker_diameter / 2.0


        # -------------------------------------------------
        # Grid cell
        # -------------------------------------------------

        # Deterministic case mapping.
        #
        # seed 0..9   : grid 0, yaw 0..90 deg
        # seed 10..19 : grid 1, yaw 0..90 deg
        # ...
        # seed 490..499: grid 49, yaw 0..90 deg
        #
        # After 500 seeds this schedule repeats. The log records
        # grid/yaw so failed combinations can be identified later.
        self.case_index = (
            self.collection_seed
            % self.NUM_CASES
        )
        self.grid_index = (
            self.case_index
            // self.NUM_YAWS
        )
        self.yaw_index = (
            self.case_index
            % self.NUM_YAWS
        )
        self.marker_yaw_deg = float(
            self.MARKER_YAW_DEG[
                self.yaw_index
            ]
        )
        self.marker_yaw = np.deg2rad(
            self.marker_yaw_deg
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

        marker_q = t3d.euler.euler2quat(
            0.0,
            0.0,
            self.marker_yaw,
        )

        self.start_pose = sapien.Pose(
            p=[
                start_x,
                start_y,
                0.741 + self.marker_radius,
            ],
            q=marker_q,
        )

        self.object = self._create_marker_actor(
            self.start_pose
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

        print("[MARKER GRID] ===========================")
        print(
            "[MARKER GRID] episode/grid =",
            f"{self.collection_index}/"
            f"{self.grid_index}",
        )
        print(
            "[MARKER GRID] seed =",
            self.collection_seed,
        )
        print(
            "[MARKER GRID] case =",
            f"{self.case_index}/{self.NUM_CASES - 1}",
        )
        print(
            "[MARKER GRID] size L/D =",
            f"{self.marker_length:.3f} / "
            f"{self.marker_diameter:.3f} m",
        )
        print(
            "[MARKER GRID] yaw_deg =",
            self.marker_yaw_deg,
        )
        print(
            "[MARKER GRID] Task center =",
            np.round(
                [
                    task_x_center,
                    task_y_center,
                ],
                4,
            ),
        )
        print(
            "[MARKER GRID] jitter =",
            np.round(
                [
                    jitter_x,
                    jitter_y,
                ],
                4,
            ),
        )
        print(
            "[MARKER GRID] Task XY =",
            np.round(
                [
                    self.task_x,
                    self.task_y,
                ],
                4,
            ),
        )
        print(
            "[MARKER GRID] World XYZ =",
            np.round(
                self.object.get_pose().p,
                4,
            ),
        )
        print(
            "[MARKER GRID] arm =",
            self.active_arm_name,
        )
        print(
            "[MARKER GRID] arm distance L/R =",
            round(left_distance, 4),
            "/",
            round(right_distance, 4),
        )
        print("[MARKER GRID] ===========================")

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
        self.hold_success = False

        # -------------------------------------------------
        # Top-down pinch at the marker center.
        #
        # RoboTwin create_box() provides four top-down
        # contact orientations. With the marker long axis
        # along local X, ids 1 and 2 orient the gripper
        # across the short local-Y axis.
        # -------------------------------------------------
        contact_ids = [1, 2]

        print(
            "[MARKER GRASP] top-down short-axis pinch"
        )
        print(
            "[MARKER GRASP] contact ids =",
            contact_ids,
        )

        pre_grasp_pose, grasp_pose = self.choose_grasp_pose(
            self.object,
            arm_tag=arm_tag,
            pre_dis=0.10,
            target_dis=0.0,
            contact_point_id=contact_ids,
        )

        print(
            "[MARKER GRASP] pre_grasp_pose =",
            None
            if pre_grasp_pose is None
            else np.round(pre_grasp_pose, 4),
        )
        print(
            "[MARKER GRASP] grasp_pose =",
            None
            if grasp_pose is None
            else np.round(grasp_pose, 4),
        )

        if pre_grasp_pose is None or grasp_pose is None:
            print(
                "[MARKER GRASP] No reachable grasp pose; "
                "reject seed."
            )
            self.plan_success = False
            return self.info

        print("[MARKER GRASP] Move to pre-grasp")
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
            print("[MARKER GRASP] Pre-grasp planning failed")
            return self.info

        print("[MARKER GRASP] Move to grasp")
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
            print("[MARKER GRASP] Grasp planning failed")
            return self.info

        self.print_gripper_state("MARKER BEFORE CLOSE")

        print("[MARKER GRASP] Close gripper")
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

        self.print_gripper_state("MARKER AFTER CLOSE")

        object_z_before_lift = float(
            self.object.get_pose().p[2]
        )

        print(
            "[MARKER LIFT] object_z before =",
            round(object_z_before_lift, 6),
        )
        print("[MARKER LIFT] Lift 5 cm")

        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=0.05,
                move_axis="world",
            )
        )

        if not self.plan_success:
            print("[MARKER LIFT] Planning failed")
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
            "[MARKER LIFT] object_z after =",
            round(object_z_after_lift, 6),
        )
        print(
            "[MARKER LIFT] delta_z =",
            round(delta_z, 6),
        )
        print(
            "[MARKER LIFT] success =",
            self.lift_success,
        )

        self.print_gripper_state("MARKER AFTER LIFT")

        if not self.lift_success:
            print(
                "[MARKER RETURN] Skip return because lift failed"
            )
            return self.info

        # -------------------------------------------------
        # Return to Stow while keeping the gripper closed.
        # -------------------------------------------------
        if arm_tag == "left":
            stow_q = list(
                self.robot.left_homestate
            )
        else:
            stow_q = list(
                self.robot.right_homestate
            )

        print("[MARKER RETURN] Return to Stow")
        print("[MARKER RETURN] arm =", arm_tag)
        print(
            "[MARKER RETURN] target q =",
            np.round(stow_q, 4),
        )

        self.return_success = self.move_to_joint_state(
            arm_tag=arm_tag,
            target_joint_pos=stow_q,
        )

        if not self.return_success:
            print(
                "[MARKER RETURN] Joint planning failed"
            )
            return self.info

        print(
            "[MARKER RETURN] Stow trajectory complete"
        )

        final_position = self.object.get_pose().p

        self.hold_success = (
            float(final_position[2])
            - float(self.start_pose.p[2])
            > 0.03
        )

        print(
            "[MARKER RETURN] final marker XYZ =",
            np.round(final_position, 4),
        )
        print(
            "[MARKER RETURN] hold_success =",
            self.hold_success,
        )

        self.print_gripper_state("MARKER AFTER RETURN")

        # `info` is consumed by RoboTwin's instruction generator.
        # Keep it string-only and minimal so task-instruction filtering
        # and placeholder replacement remain compatible.
        self.info["info"] = {
            "{a}": str(self.active_arm_name),
        }

        # Keep numeric diagnostics as a sibling field. collect_data.py
        # still stores the complete returned dictionary in scene_info.json,
        # while build_episode_instructions() reads only self.info["info"].
        self.info["metrics"] = {
            "grid_index": int(self.grid_index),
            "seed": int(self.collection_seed),
            "active_arm": str(self.active_arm_name),
            "task_x": float(self.task_x),
            "task_y": float(self.task_y),
            "case_index": int(self.case_index),
            "yaw_deg": float(self.marker_yaw_deg),
            "marker_length": float(self.marker_length),
            "marker_diameter": float(self.marker_diameter),
            "start_x": float(self.start_pose.p[0]),
            "start_y": float(self.start_pose.p[1]),
            "final_x": float(final_position[0]),
            "final_y": float(final_position[1]),
            "final_z": float(final_position[2]),
            "lift_success": bool(self.lift_success),
            "return_success": bool(self.return_success),
            "hold_success": bool(self.hold_success),
        }

        return self.info

    def check_success(self):
        return bool(
            getattr(self, "lift_success", False)
            and getattr(self, "return_success", False)
            and getattr(self, "hold_success", False)
        )
