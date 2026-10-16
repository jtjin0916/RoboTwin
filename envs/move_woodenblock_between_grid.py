import numpy as np
import sapien

from .move_woodenblock_randomized import (
    move_woodenblock_randomized,
)
from .utils import *
from ._GLOBAL_CONFIGS import *


class move_woodenblock_between_grid(
    move_woodenblock_randomized
):
    """
    Evaluation-only task.

    Uses deterministic midpoint positions BETWEEN the
    fixed training grid lines. With the current 10 x 5
    training grid this produces 9 x 4 = 36 unseen points.

    No evaluation position overlaps a training position.
    """

    def load_actors(self):
        self.block_size = 0.04
        self.block_half_size = (
            self.block_size / 2.0
        )

        # -------------------------------------------------
        # Midpoints between training grid lines
        # -------------------------------------------------
        train_x = np.sort(
            np.asarray(
                self.TASK_X_CENTERS,
                dtype=np.float32,
            )
        )

        train_y = np.sort(
            np.asarray(
                self.TASK_Y_CENTERS,
                dtype=np.float32,
            )
        )

        eval_x = (
            train_x[:-1] + train_x[1:]
        ) / 2.0

        eval_y = (
            train_y[:-1] + train_y[1:]
        ) / 2.0

        num_x = len(eval_x)
        num_y = len(eval_y)
        self.num_eval_cells = num_x * num_y

        seed = int(
            getattr(
                self,
                "collection_seed",
                0,
            )
        )

        # Deterministic seed -> unseen midpoint cell.
        self.grid_index = (
            seed % self.num_eval_cells
        )

        x_idx = self.grid_index // num_y
        y_idx = self.grid_index % num_y

        self.task_x = float(eval_x[x_idx])
        self.task_y = float(eval_y[y_idx])

        # Task -> World
        #
        # X_T = Y_W + 0.35
        # Y_T = -X_W
        #
        # therefore:
        # X_W = -Y_T
        # Y_W = X_T - 0.35

        start_x = -self.task_y
        start_y = self.task_x - 0.35

        self.start_pose = sapien.Pose(
            p=[
                start_x,
                start_y,
                0.741 + self.block_half_size,
            ],
            q=[
                1.0,
                0.0,
                0.0,
                0.0,
            ],
        )

        self.object = create_box(
            scene=self,
            pose=self.start_pose,
            half_size=(
                self.block_half_size,
                self.block_half_size,
                self.block_half_size,
            ),
            color=(
                0.72,
                0.45,
                0.20,
            ),
            name="wooden_block_4cm",
        )

        # Nearest-arm selection.
        object_xy = np.array(
            [
                start_x,
                start_y,
            ],
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

        print(
            "[BETWEEN GRID] ====================="
        )
        print(
            "[BETWEEN GRID] seed/cell =",
            f"{seed}/{self.grid_index}",
        )
        print(
            "[BETWEEN GRID] total cells =",
            self.num_eval_cells,
        )
        print(
            "[BETWEEN GRID] Task XY midpoint =",
            np.round(
                [
                    self.task_x,
                    self.task_y,
                ],
                6,
            ),
        )
        print(
            "[BETWEEN GRID] World XYZ =",
            np.round(
                self.object.get_pose().p,
                6,
            ),
        )
        print(
            "[BETWEEN GRID] arm =",
            self.active_arm_name,
        )
        print(
            "[BETWEEN GRID] arm distance L/R =",
            round(left_distance, 4),
            "/",
            round(right_distance, 4),
        )
        print(
            "[BETWEEN GRID] ====================="
        )
