"""Building mesh generation: footprint polygons → extruded 3D blocks (PRD §7.3, §13.1).

Vertex format: float32 (x, y, z, nx, ny, nz) — position + face normal.
The renderer can use abs(dot(normal, light)) for two-sided diffuse shading
since polygon winding varies in raw OSM data.

MVP limitations (FR-BLDG-008):
  - Only exterior ring is used; polygon holes are ignored.
  - Flat roofs only (FR-BLDG-007).
  - MultiPolygon parts each get their own extrusion.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Height clamping (PRD §13.1)
_HEIGHT_MIN_M  =   3.0
_HEIGHT_MAX_M  = 250.0


# ---------------------------------------------------------------------------
# Height helpers (PRD §7.3 FR-BLDG-005, §13.1)
# ---------------------------------------------------------------------------

def parse_height(value) -> Optional[float]:
    """Parse an OSM height tag to float metres.

    Handles: '12', '12.5', '12 m', '12.5m'.
    Returns None on failure (feet, complex strings, etc.).
    """
    if value is None:
        return None
    s = str(value).strip()
    # Bare number or number with optional 'm' suffix
    import re
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*m?", s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def get_building_height(row, cfg: dict) -> float:
    """Return building height in metres using priority: height tag > levels > default.

    Clamps explicit values to [3, 250] m for visual sanity (PRD §13.1).
    """
    floor_h   = cfg["preprocess"]["floor_height_m"]
    default_h = cfg["preprocess"]["default_building_height_m"]

    h = parse_height(_row_tag(row, "height"))
    if h is not None:
        return float(np.clip(h, _HEIGHT_MIN_M, _HEIGHT_MAX_M))

    levels = parse_height(_row_tag(row, "building:levels"))
    if levels is not None and levels >= 1:
        return float(np.clip(levels * floor_h, _HEIGHT_MIN_M, _HEIGHT_MAX_M))

    return default_h


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _row_tag(row, key: str) -> Optional[str]:
    """Safely retrieve a tag from a GeoDataFrame row, treating NaN as missing."""
    val = row.get(key)
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return str(val)


def _get_polygons(geom) -> list:
    """Return a flat list of Shapely Polygon parts from any geometry type."""
    from shapely.geometry import Polygon, MultiPolygon
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def _clean_polygon(poly):
    """Fix invalid polygon; return None if unrecoverable."""
    from shapely.geometry import Polygon, MultiPolygon
    if poly.is_valid and not poly.is_empty:
        return poly
    try:
        import shapely
        fixed = shapely.make_valid(poly)
        if isinstance(fixed, Polygon) and not fixed.is_empty:
            return fixed
        if isinstance(fixed, MultiPolygon):
            return max(fixed.geoms, key=lambda p: p.area)
    except Exception as exc:
        log.debug("make_valid failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Extrusion (FR-BLDG-004, FR-BLDG-006)
# ---------------------------------------------------------------------------

def _triangulate_roof(xz: np.ndarray) -> np.ndarray:
    """Return flat uint32 index array for the roof polygon via earcut.

    Falls back to a naive fan if earcut fails (convex polygons only).
    """
    import mapbox_earcut as earcut
    n = len(xz)
    try:
        rings = np.array([n], dtype=np.uint32)
        idxs = earcut.triangulate_float64(xz.astype(np.float64), rings)
        if len(idxs) > 0:
            return idxs.astype(np.uint32)
    except Exception as exc:
        log.debug("earcut failed: %s", exc)
    # Fan fallback (correct for convex polygons)
    fan = []
    for i in range(1, n - 1):
        fan.extend([0, i, i + 1])
    return np.array(fan, dtype=np.uint32)


def _extrude_building(
    xz_ring: np.ndarray,
    height: float,
    base_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrude a 2-D polygon exterior ring into walls + flat roof.

    Args:
        xz_ring: (N, 2) array of (x, z) vertices.  Closing vertex included or
                 not — both are handled.
        height:  Building height above base_y in metres.
        base_y:  Ground Y coordinate (0 for flat terrain MVP).

    Returns:
        vertices: float32 (M, 6) — x, y, z, nx, ny, nz
        indices:  uint32  (K,)
    """
    from shapely.geometry.polygon import orient
    from shapely.geometry import Polygon

    # Drop closing vertex if present
    ring = xz_ring[:-1] if np.allclose(xz_ring[0], xz_ring[-1]) else xz_ring
    n = len(ring)
    if n < 3:
        return np.empty((0, 6), dtype=np.float32), np.empty(0, dtype=np.uint32)

    # Ensure CCW winding so that outward normal = (dz, 0, −dx)
    # (orient expects (lon, lat) = (x, z) in our local space)
    shaped = Polygon(ring)
    shaped = orient(shaped, sign=1.0)   # CCW
    ring = np.array(shaped.exterior.coords[:-1], dtype=np.float64)  # drop closing
    n = len(ring)

    top_y = base_y + height
    all_verts: List[list] = []
    all_idxs:  List[int]  = []
    base_idx = 0

    # --- Walls ---
    for i in range(n):
        j = (i + 1) % n
        x0, z0 = ring[i]
        x1, z1 = ring[j]

        dx, dz = x1 - x0, z1 - z0
        seg_len = np.hypot(dx, dz)
        if seg_len < 1e-6:
            continue
        dx /= seg_len; dz /= seg_len
        # Outward wall normal for CCW polygon: right-perpendicular (CW rotation)
        nx, nz = dz, -dx

        # quad: bottom-start, bottom-end, top-end, top-start
        all_verts += [
            [x0, base_y, z0, nx, 0.0, nz],
            [x1, base_y, z1, nx, 0.0, nz],
            [x1, top_y,  z1, nx, 0.0, nz],
            [x0, top_y,  z0, nx, 0.0, nz],
        ]
        b = base_idx
        all_idxs += [b, b+1, b+2, b, b+2, b+3]
        base_idx += 4

    # --- Roof ---
    roof_idxs = _triangulate_roof(ring)
    if len(roof_idxs) > 0:
        roof_base = base_idx
        for x, z in ring:
            all_verts.append([x, top_y, z, 0.0, 1.0, 0.0])
        all_idxs += (roof_idxs + roof_base).tolist()

    if not all_verts:
        return np.empty((0, 6), dtype=np.float32), np.empty(0, dtype=np.uint32)

    return (
        np.array(all_verts, dtype=np.float32),
        np.array(all_idxs,  dtype=np.uint32),
    )


# ---------------------------------------------------------------------------
# Full-GeoDataFrame pipeline
# ---------------------------------------------------------------------------

def build_building_mesh(
    gdf,
    projector,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Project and extrude all building footprints into one combined mesh.

    Returns:
        vertices:  float32 (N, 6) — x y z nx ny nz
        indices:   uint32  (M,)
        bldg_meta: list of per-building dicts
    """
    all_verts:  List[np.ndarray] = []
    all_idxs:   List[np.ndarray] = []
    bldg_meta:  list             = []
    global_base = 0
    n_skipped   = 0

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            n_skipped += 1
            continue

        polys = _get_polygons(geom)
        if not polys:
            n_skipped += 1
            continue

        height = get_building_height(row, cfg)

        for poly in polys:
            cleaned = _clean_polygon(poly)
            if cleaned is None:
                n_skipped += 1
                continue

            xs, zs = projector.polygon_to_xz(cleaned)
            xz = np.column_stack([xs, zs])

            verts, idxs = _extrude_building(xz, height, base_y=0.0)
            if len(verts) == 0:
                n_skipped += 1
                continue

            all_verts.append(verts)
            all_idxs.append(idxs + global_base)
            global_base += len(verts)
            bldg_meta.append({
                "height_m":    float(height),
                "n_wall_verts": int(len(verts) - len(np.array(cleaned.exterior.coords[:-1]))),
                "n_roof_verts": int(len(np.array(cleaned.exterior.coords[:-1]))),
            })

    if not all_verts:
        log.warning("No building geometry was generated.")
        return (
            np.empty((0, 6), dtype=np.float32),
            np.empty(0, dtype=np.uint32),
            bldg_meta,
        )

    vertices = np.concatenate(all_verts, axis=0)
    indices  = np.concatenate(all_idxs)
    log.info(
        "Building mesh: %d buildings  %d vertices  %d triangles  (%d skipped)",
        len(bldg_meta), len(vertices), len(indices) // 3, n_skipped,
    )
    return vertices, indices, bldg_meta


# ---------------------------------------------------------------------------
# Serialisation (FR-BLDG-011)
# ---------------------------------------------------------------------------

def save_building_mesh(
    vertices:  np.ndarray,
    indices:   np.ndarray,
    bldg_meta: list,
    cfg: dict,
) -> Path:
    """Write buildings_mesh.npz and buildings_meta.json to processed output dir."""
    out_dir = Path(cfg["preprocess"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "buildings_mesh.npz"
    np.savez(
        npz_path,
        vertices=vertices.astype(np.float32),
        indices=indices.astype(np.uint32),
    )
    log.info(
        "Saved building mesh → %s  (%d verts, %d idx)",
        npz_path, len(vertices), len(indices),
    )

    meta_path = out_dir / "buildings_meta.json"
    with open(meta_path, "w") as f:
        json.dump(bldg_meta, f)

    return npz_path
