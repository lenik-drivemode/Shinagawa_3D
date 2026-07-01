"""Vehicle marker geometry: simple cuboid in marker-local space (PRD §12.5).

Marker local coordinate convention:
  - +Z is forward (direction of travel / heading)
  - +X is right
  - +Y is up; y=0 is the bottom face (placed at marker_y_m in world space)

The caller applies translate(x, marker_y_m, z) @ rotate_y(heading) as model matrix.
"""
from typing import Tuple

import numpy as np


def make_marker_mesh(cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices float32 [8,3], indices uint32 [36]) for the marker box.

    Vertices:
        8 corners of the cuboid (no per-face duplication needed for flat color).
    Indices:
        36 values forming 12 triangles (2 per face × 6 faces).
    """
    r = cfg["render"]
    length = float(r["marker_length_m"])   # z extent
    width  = float(r["marker_width_m"])    # x extent
    height = float(r["marker_height_m"])   # y extent

    hl = length * 0.5
    hw = width  * 0.5

    # 8 corners: bottom-4 (y=0), top-4 (y=height)
    # Indices: 0-3 bottom, 4-7 top
    verts = np.array([
        #  x    y       z
        [-hw,  0.0,  -hl],   # 0 bottom-left-back
        [ hw,  0.0,  -hl],   # 1 bottom-right-back
        [ hw,  0.0,   hl],   # 2 bottom-right-front
        [-hw,  0.0,   hl],   # 3 bottom-left-front
        [-hw,  height, -hl],  # 4 top-left-back
        [ hw,  height, -hl],  # 5 top-right-back
        [ hw,  height,  hl],  # 6 top-right-front
        [-hw,  height,  hl],  # 7 top-left-front
    ], dtype=np.float32)

    idxs = np.array([
        0, 2, 1,   0, 3, 2,   # bottom (y=0), winding CW from below
        4, 5, 6,   4, 6, 7,   # top    (y=height)
        0, 1, 5,   0, 5, 4,   # back   (z=-hl)
        2, 3, 7,   2, 7, 6,   # front  (z=+hl)
        0, 4, 7,   0, 7, 3,   # left   (x=-hw)
        1, 2, 6,   1, 6, 5,   # right  (x=+hw)
    ], dtype=np.uint32)

    return verts, idxs
