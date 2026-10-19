"""Unvalidated Sweepick 100-grid candidate workspace in base_footprint metres.

The measured task's local PointCloud crop is x=[.15,.32],
y=[-.26,+.26], z=[.005,.40]. For a 40mm axis-aligned cube and
10mm additional X margin, x centres may be [.18,.29].
Y +/-15cm is a candidate field, NOT a proven OS30A-visible workspace.
"""
import numpy as np

X_CENTERS = np.linspace(.18, .29, 10, dtype=np.float64)
Y_CENTERS = np.linspace(-.15, .15, 10, dtype=np.float64)
CROP_MIN = np.array([.15, -.26, .005])
CROP_MAX = np.array([.32, +.26, .40])
BLOCK_HALF_SIZE = .02
EXTRA_X_MARGIN = .01
BLOCK_CENTER_Z = BLOCK_HALF_SIZE + .001
NUM_CELLS = len(X_CENTERS) * len(Y_CENTERS)


def cell(index):
    index = int(index)
    if not 0 <= index < NUM_CELLS:
        raise ValueError(f"Grid ID must be 0..{NUM_CELLS-1}: {index}")
    j = index % len(Y_CENTERS)
    i = index // len(Y_CENTERS)
    return float(X_CENTERS[i]), float(Y_CENTERS[j])


def cells():
    for i in range(NUM_CELLS):
        x, y = cell(i)
        yield {"grid_id": i, "x_base_m": x, "y_base_m": y,
               "z_base_m": BLOCK_CENTER_Z}


def validate():
    assert NUM_CELLS == 100
    assert X_CENTERS.min() >= CROP_MIN[0] + BLOCK_HALF_SIZE + EXTRA_X_MARGIN - 1e-9
    assert X_CENTERS.max() <= CROP_MAX[0] - BLOCK_HALF_SIZE - EXTRA_X_MARGIN + 1e-9
    assert Y_CENTERS.min() - BLOCK_HALF_SIZE >= CROP_MIN[1]
    assert Y_CENTERS.max() + BLOCK_HALF_SIZE <= CROP_MAX[1]
    assert CROP_MIN[2] < BLOCK_CENTER_Z < CROP_MAX[2]


validate()
