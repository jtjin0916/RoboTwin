"""World (+Y) and TurtleBot base_footprint coordinate contract for Sweepick.

Pure NumPy to allow CPU validation without SAPIEN/Isaac/CuRobo.
All 3D dimensions are in metres. R_WB is a +90deg yaw around World Z.
"""
import numpy as np

TABLE_NOMINAL_WORLD_Z = 0.5
BASE_ORIGIN_WORLD_XY = (0.0, -0.2539)
ROT_WORLD_FROM_BASE = np.array([
    [0., -1., 0.],
    [1.,  0., 0.],
    [0.,  0., 1.],
], dtype=np.float64)


def base_to_world(xyz, *, tabletop_world_z=TABLE_NOMINAL_WORLD_Z):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[-1] != 3:
        raise ValueError("Expected (..., 3) XYZ array")
    t = np.array([*BASE_ORIGIN_WORLD_XY, tabletop_world_z])
    return xyz @ ROT_WORLD_FROM_BASE.T + t


def world_to_base(xyz, *, tabletop_world_z=TABLE_NOMINAL_WORLD_Z):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[-1] != 3:
        raise ValueError("Expected (..., 3) XYZ array")
    t = np.array([*BASE_ORIGIN_WORLD_XY, tabletop_world_z])
    return (xyz - t) @ ROT_WORLD_FROM_BASE
