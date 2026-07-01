"""Centralised WGS84 → local metric coordinate projection (NFR-MAINT-003).

Coordinate convention (PRD §4.3):
    X  east,  positive east   (= projected easting  − origin easting)
    Y  up,    positive up     (caller supplies height; ground = 0)
    Z  north, positive north  (= projected northing − origin northing)

Origin is the bbox centre from config, projected into the target CRS.
All public functions use always_xy=True so the argument order is always
(longitude, latitude) for geographic CRS and (easting, northing) for
projected CRS — matching NumPy "x before y" conventions.
"""
from typing import Tuple, Union

import numpy as np
from pyproj import Transformer


# ---------------------------------------------------------------------------
# Low-level helpers (used by Projector and by metadata writers)
# ---------------------------------------------------------------------------

def make_transformer(cfg: dict) -> Transformer:
    """Build a pyproj Transformer from the config EPSG codes."""
    src = f"EPSG:{cfg['projection']['source_epsg']}"
    dst = f"EPSG:{cfg['projection']['target_epsg']}"
    return Transformer.from_crs(src, dst, always_xy=True)


def project_origin(cfg: dict, transformer: Transformer) -> Tuple[float, float]:
    """Return (origin_easting, origin_northing) for the configured bbox centre."""
    lon = cfg["area"]["center"]["lon"]
    lat = cfg["area"]["center"]["lat"]
    easting, northing = transformer.transform(lon, lat)
    return float(easting), float(northing)


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

class Projector:
    """Converts WGS84 lat/lon to local world (X, Z) metre coordinates.

    Instantiate once per run and pass to preprocessing stages.
    """

    def __init__(self, cfg: dict) -> None:
        self._transformer = make_transformer(cfg)
        self.origin_easting, self.origin_northing = project_origin(cfg, self._transformer)

    # ------------------------------------------------------------------
    # Single-point
    # ------------------------------------------------------------------

    def to_xz(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert one lat/lon point to local (world_x, world_z) in metres."""
        easting, northing = self._transformer.transform(lon, lat)
        return float(easting - self.origin_easting), float(northing - self.origin_northing)

    # ------------------------------------------------------------------
    # Batch (NumPy) — use for all geometry arrays
    # ------------------------------------------------------------------

    def batch_to_xz(
        self,
        lats: Union[np.ndarray, list],
        lons: Union[np.ndarray, list],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert arrays of lat/lon to local (world_x[], world_z[]) in metres.

        Returns float64 arrays; callers downcast to float32 when building GPU
        buffers (after subtracting the origin, float32 precision is adequate).
        """
        eastings, northings = self._transformer.transform(
            np.asarray(lons, dtype=np.float64),
            np.asarray(lats, dtype=np.float64),
        )
        xs = np.asarray(eastings,  dtype=np.float64) - self.origin_easting
        zs = np.asarray(northings, dtype=np.float64) - self.origin_northing
        return xs, zs

    def polygon_to_xz(self, polygon) -> Tuple[np.ndarray, np.ndarray]:
        """Project a Shapely polygon's exterior ring to (xs, zs) arrays."""
        coords = np.array(polygon.exterior.coords)  # shape (N, 2): lon, lat
        return self.batch_to_xz(lats=coords[:, 1], lons=coords[:, 0])
