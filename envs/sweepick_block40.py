"""Sweepick floor Block40 single-left-arm smoke test (NOT an expert dataset).

Handoff source: sweepick_261019_sim_handoff.zip (2026-10-19).
The TurtleBot Waffle Pi URDF and meshes are loaded from an external checkout to
avoid silently substituting an arbitrary chassis model. Original RoboTwin tasks
and URDFs are not modified. CuRobo must independently verify obstacles.
"""

import os
from pathlib import Path

import numpy as np
import sapien

from .move_woodenblock_fixed import move_woodenblock_fixed
from .utils import create_box, create_entity_box


class sweepick_block40(move_woodenblock_fixed):
    # Chosen from the left-arm recommended grid in the 2026-10-19 handoff.
    # Coordinates are in TurtleBot base_footprint: X forward, Y left, Z up.
    ROBOT_WORLD_ORIGIN = np.array([0.0, -0.2539, 0.0])
    BLOCK_ROBOT_XY = np.array([0.18, 0.14])
    FLOOR_TOP_WORLD_Z = 0.0
    BLOCK_HALF_SIZE = 0.02

    def setup_demo(self, **kwargs):
        # Base_Task's historical default is tabletop z=0.74; this task is floor.
        options = dict(kwargs)
        options["table_height"] = self.FLOOR_TOP_WORLD_Z
        super().setup_demo(**options)

    @staticmethod
    def _robot_to_world(xyz):
        """TurtleBot base_footprint yaw +90deg at (0,-.2539,0)."""
        x, y, z = map(float, xyz)
        return [-y, float(sweepick_block40.ROBOT_WORLD_ORIGIN[1]) + x, z]

    def _load_turtlebot(self):
        path = os.environ.get("SWEEPICK_TURTLEBOT_URDF", "")
        if not path or not Path(path).is_file():
            raise RuntimeError(
                "SWEEPICK_TURTLEBOT_URDF must point to the handoff's checked-out "
                "2ARM_ROBOT/sim/turtlebot3_waffle_pi/turtlebot3_waffle_pi_mujoco.urdf "
                "with its upstream/ mesh directory. No proxy fallback is allowed."
            )
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.turtlebot = loader.load(str(Path(path).resolve()))
        if self.turtlebot is None:
            raise RuntimeError("SAPIEN failed to load TurtleBot3 Waffle Pi URDF")
        self.turtlebot.set_root_pose(
            sapien.Pose(
                p=self.ROBOT_WORLD_ORIGIN.tolist(),
                q=[0.7071068, 0.0, 0.0, 0.7071068],
            )
        )
        # The above URDF is a MuJoCo derivative of ROBOTIS's source.
        # Its wheel joints are not controlled in this task; base is fixed.

    def _load_mount_collision_proxies(self):
        """Provisional rails/upper OS30A housing from team MODEL_FOOTPRINT.

        Real assembly CAD is NOT represented with complete geometry here.
        These proxies only add obvious collision volumes to SAPIEN; CuRobo
        still needs a separately validated obstacle configuration.
        """
        proxies = (
            ("sweepick_rail_front_UNVERIFIED", [-0.11, 0.0, 0.114], [0.010, 0.20, 0.010]),
            ("sweepick_rail_back_UNVERIFIED", [-0.018, 0.0, 0.114], [0.010, 0.20, 0.010]),
            ("sweepick_os30a_housing_UNVERIFIED", [-0.035, 0.0, 0.4905], [0.017, 0.045, 0.0175]),
        )
        self.mount_proxies = []
        for name, center, half_size in proxies:
            self.mount_proxies.append(
                create_entity_box(
                    scene=self.scene,
                    pose=sapien.Pose(
                        p=self._robot_to_world(center),
                        q=[0.7071068, 0.0, 0.0, 0.7071068],
                    ),
                    half_size=half_size,
                    color=(0.30, 0.34, 0.39),
                    is_static=True,
                    name=name,
                )
            )

    def load_actors(self):
        self._load_turtlebot()
        self._load_mount_collision_proxies()
        self.block_size = 2.0 * self.BLOCK_HALF_SIZE
        self.block_half_size = self.BLOCK_HALF_SIZE
        bx, by = map(float, self.BLOCK_ROBOT_XY)
        self.start_pose = sapien.Pose(
            p=self._robot_to_world([bx, by, self.BLOCK_HALF_SIZE + 0.001]),
            q=[1.0, 0.0, 0.0, 0.0],
        )
        self.object = create_box(
            scene=self,
            pose=self.start_pose,
            half_size=(self.BLOCK_HALF_SIZE,) * 3,
            color=(0.78, 0.10, 0.10),
            name="sweepick_floor_red_block_40mm",
        )
        self.target_position = np.asarray(self.start_pose.p, dtype=np.float64)
        self.lift_success = False
        print(
            "[SWEEPICK BLOCK40 SMOKE] robot_xy=", self.BLOCK_ROBOT_XY,
            "world_xyz=", np.round(self.start_pose.p, 4),
            "floor_world_z=0.0; left-arm grasp/lift only; NOT validated",
        )
