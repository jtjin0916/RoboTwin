"""Measured Sweepick floor-block smoke; original smoke test remains unchanged."""
import os
from pathlib import Path
import numpy as np
import sapien
from .sweepick_block40 import sweepick_block40
from .utils import create_entity_box


class sweepick_block40_measured(sweepick_block40):
    ROBOT_WORLD_ORIGIN = np.array([0.0, 0.0, 0.0])

    @staticmethod
    def _robot_to_world(xyz):
        """The SAPIEN World equals TurtleBot base_footprint."""
        return list(map(float, xyz))

    def _load_turtlebot(self):
        path = os.environ.get("SWEEPICK_TURTLEBOT_URDF", "")
        if not path or not Path(path).is_file():
            raise RuntimeError("Missing SWEEPICK_TURTLEBOT_URDF with meshes")
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.turtlebot = loader.load(str(Path(path).resolve()))
        if self.turtlebot is None:
            raise RuntimeError("TurtleBot URDF load failed")
        self.turtlebot.set_root_pose(sapien.Pose(
            p=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0]
        ))

    def _load_mount_collision_proxies(self):
        # User dimensions: two 20 x 20 x 400mm rails along Y,
        # X centres are ASSUMED symmetric about the arm base at X=-60mm.
        # SAPIEN objects exist but CuRobo does not yet have their obstacles.
        profiles = (
            ("sweepick_profile_front_ASSUMED", [-0.014, 0.0, 0.113],
             [0.010, 0.200, 0.010]),
            ("sweepick_profile_rear_ASSUMED", [-0.106, 0.0, 0.113],
             [0.010, 0.200, 0.010]),
            ("sweepick_os30a_housing_UNVERIFIED", [-0.035, 0.0, 0.4905],
             [0.017, 0.045, 0.0175]),
        )
        self.mount_proxies = [
            create_entity_box(
                scene=self.scene,
                pose=sapien.Pose(p=p, q=[1.0, 0.0, 0.0, 0.0]),
                half_size=half, color=(0.30, 0.34, 0.39),
                is_static=True, name=name
            )
            for name, p, half in profiles
        ]

    def load_actors(self):
        super().load_actors()
        print("[SWEEPICK MEASURED] world=base_footprint; "
              "arm mounts left=(-.060,+.150,.123), right=(-.060,-.150,.123); "
              "profiles estimated; camera extrinsic provisional; "
              "CuRobo chassis/profile collisions NOT certified")
