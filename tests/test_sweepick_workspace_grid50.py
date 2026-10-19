"""CPU-only geometry and scan coverage contract; no SAPIEN required."""
from pathlib import Path
import numpy as np
import yaml

from sweepick_workspace_grid50 import (
    X_CENTERS, Y_CENTERS, BLOCK_HALF_SIZE, CROP_MIN, CROP_MAX,
    NUM_CELLS, cell, cells,
)
from sweepick_table_frame import base_to_world, world_to_base

ROOT = Path(__file__).resolve().parents[1]


def test_all_50_cells_unique_and_symmetric_y():
    assert NUM_CELLS == 50
    assert len(X_CENTERS) == 10 and len(Y_CENTERS) == 5
    assert np.allclose(Y_CENTERS, [-.15,-.075,0,.075,.15])
    assert np.isclose(X_CENTERS[0], .18)
    assert np.isclose(X_CENTERS[-1], .29)
    assert len(set((round(x["x_base_m"],8),round(x["y_base_m"],8))
                   for x in cells())) == 50
    assert np.allclose(cell(0), [.18,-.15])
    assert np.allclose(cell(49), [.29,.15])


def test_cube_40mm_wholly_inside_crop_with_margin():
    for x in X_CENTERS:
        for y in Y_CENTERS:
            assert x-BLOCK_HALF_SIZE >= CROP_MIN[0] + .01-1e-7
            assert x+BLOCK_HALF_SIZE <= CROP_MAX[0] - .01+1e-7
            assert y-BLOCK_HALF_SIZE >= CROP_MIN[1]
            assert y+BLOCK_HALF_SIZE <= CROP_MAX[1]
            p=np.array([x,y,.021])
            assert np.allclose(world_to_base(base_to_world(p)),p)


def test_grid_diagnostic_has_no_success_only_replacement():
    config=yaml.safe_load((
        ROOT/"env_cfg/task_config/demo_so101_pgripper_sweepick_workspace_grid50.yml"
    ).read_text())
    dr=config["domain_randomization"]
    assert not dr["random_background"] and not dr["random_light"]
    assert dr["random_table_height"] == 0
    assert dr["random_head_camera_dis"] == 0
    assert config["collect_data"] is False
    source=(ROOT/"scripts/scan_sweepick_workspace_grid50.py").read_text()
    assert "csv.DictWriter" in source
    assert "lift_success" in source
    assert "existing_rows" in source
    assert "near_cube_points_proxy" in source
