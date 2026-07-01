"""Road mesh generation: graph edges → flat strip meshes (PRD §7.2, §13.2)."""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Width fallback table (FR-ROAD-004)
_HIGHWAY_WIDTHS: Dict[str, float] = {
    "motorway":     12.0,
    "trunk":        12.0,
    "primary":      10.0,
    "secondary":     8.0,
    "tertiary":      7.0,
    "residential":   5.5,
    "service":       4.0,
    "unclassified":  5.0,
}
_DEFAULT_WIDTH = 5.0

# Matches "6", "6.5", "6 m", "6.5m" (PRD §13.2)
_WIDTH_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*m?\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Width helpers
# ---------------------------------------------------------------------------

def parse_width(value) -> Optional[float]:
    """Parse an OSM width tag to float metres.  Returns None on failure."""
    if value is None:
        return None
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = _WIDTH_RE.match(s)
    return float(m.group(1)) if m else None


def highway_width(highway_type) -> float:
    """Return default width in metres for a highway type string (or list)."""
    if isinstance(highway_type, list):
        for t in highway_type:
            if t in _HIGHWAY_WIDTHS:
                return _HIGHWAY_WIDTHS[t]
        return _DEFAULT_WIDTH
    return _HIGHWAY_WIDTHS.get(str(highway_type), _DEFAULT_WIDTH)


def get_edge_width(edge_data: dict) -> float:
    """Return road width: parsed OSM tag if sane, else highway-type fallback."""
    w = parse_width(edge_data.get("width"))
    if w is not None and 1.0 <= w <= 50.0:
        return w
    return highway_width(edge_data.get("highway", ""))


# ---------------------------------------------------------------------------
# Edge geometry extraction
# ---------------------------------------------------------------------------

def _edge_latlons(G, u: int, v: int, data: dict) -> np.ndarray:
    """Return (N, 2) array of [lat, lon] pairs along an edge.

    Uses the edge's Shapely LineString geometry when available, otherwise
    falls back to the straight line between the two endpoint nodes.
    """
    geom = data.get("geometry")
    if geom is not None:
        # osmnx stores edge geometry as LineString in geographic (lon, lat) order
        coords = np.array(geom.coords)   # (N, 2): [lon, lat]
        return coords[:, ::-1]           # flip → [lat, lon]
    n_u, n_v = G.nodes[u], G.nodes[v]
    return np.array([
        [n_u["y"], n_u["x"]],   # [lat, lon]
        [n_v["y"], n_v["x"]],
    ])


# ---------------------------------------------------------------------------
# Strip mesh generation (FR-ROAD-003, FR-ROAD-005)
# ---------------------------------------------------------------------------

def polyline_to_strip(
    xz: np.ndarray,
    width: float,
    y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 2-D polyline in local XZ to a flat road-strip mesh.

    For each consecutive segment pair (p_i, p_{i+1}):
      - compute the perpendicular (CCW rotation of the direction vector)
      - place four vertices at ±half_width offsets on each side
      - emit two triangles per quad

    MVP note (FR-ROAD-006): adjacent quads share no vertices, so sharp
    corners produce visible gaps/overlaps.  This is accepted for PoC.

    Args:
        xz:    (N, 2) float array of local (x, z) coordinates.
        width: road width in metres.
        y:     world Y for all vertices (road_y_m from config).

    Returns:
        vertices: float32 (4*(N-1), 3)
        indices:  uint32  (6*(N-1),)
    """
    n = len(xz)
    if n < 2:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint32)

    half = width * 0.5
    all_verts: List[np.ndarray] = []
    all_idxs:  List[np.ndarray] = []
    base = 0

    for i in range(n - 1):
        p0, p1 = xz[i], xz[i + 1]
        d = p1 - p0
        seg_len = np.hypot(d[0], d[1])
        if seg_len < 1e-6:
            continue                    # degenerate segment — skip
        d /= seg_len
        # Perpendicular: CCW rotation of (dx, dz) → (-dz, dx)
        perp = np.array([-d[1], d[0]])

        v0 = [p0[0] + perp[0] * half, y, p0[1] + perp[1] * half]
        v1 = [p0[0] - perp[0] * half, y, p0[1] - perp[1] * half]
        v2 = [p1[0] - perp[0] * half, y, p1[1] - perp[1] * half]
        v3 = [p1[0] + perp[0] * half, y, p1[1] + perp[1] * half]

        all_verts.extend([v0, v1, v2, v3])
        # CCW winding when viewed from +Y (camera above): v0,v2,v1 and v0,v3,v2
        all_idxs.append([base, base+2, base+1, base, base+3, base+2])
        base += 4

    if not all_verts:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint32)

    return (
        np.array(all_verts, dtype=np.float32),
        np.array(all_idxs, dtype=np.uint32).ravel(),
    )


# ---------------------------------------------------------------------------
# Full-graph mesh builder
# ---------------------------------------------------------------------------

def build_road_mesh(
    G,
    projector,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Project and mesh all edges in the road graph into one combined mesh.

    Returns:
        vertices:  float32 (N, 3)
        indices:   uint32  (M,)
        edge_meta: list of per-edge dicts (for metadata.json / debugging)
    """
    y = cfg["preprocess"]["road_y_m"]
    all_verts: List[np.ndarray] = []
    all_idxs:  List[np.ndarray] = []
    edge_meta: list = []
    global_base = 0
    skipped = 0

    for u, v, _key, data in G.edges(data=True, keys=True):
        latlons = _edge_latlons(G, u, v, data)
        xs, zs = projector.batch_to_xz(
            lats=latlons[:, 0],
            lons=latlons[:, 1],
        )
        xz = np.column_stack([xs, zs])

        width = get_edge_width(data)
        verts, idxs = polyline_to_strip(xz, width, y=y)

        if len(verts) == 0:
            skipped += 1
            continue

        all_verts.append(verts)
        all_idxs.append(idxs + global_base)
        global_base += len(verts)
        edge_meta.append({
            "u": int(u),
            "v": int(v),
            "highway": str(data.get("highway", "")),
            "width_m": float(width),
            "n_segments": len(verts) // 4,
        })

    if not all_verts:
        log.warning("No road segments were generated.")
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.uint32),
            edge_meta,
        )

    vertices = np.concatenate(all_verts, axis=0)
    indices  = np.concatenate(all_idxs)
    log.info(
        "Road mesh: %d edges  %d vertices  %d triangles  (%d degenerate skipped)",
        len(edge_meta), len(vertices), len(indices) // 3, skipped,
    )
    return vertices, indices, edge_meta


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_road_mesh(
    vertices: np.ndarray,
    indices: np.ndarray,
    edge_meta: list,
    cfg: dict,
) -> Path:
    """Write roads_mesh.npz and roads_meta.json to the processed output dir."""
    out_dir = Path(cfg["preprocess"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "roads_mesh.npz"
    np.savez(npz_path, vertices=vertices.astype(np.float32), indices=indices.astype(np.uint32))
    log.info("Saved road mesh → %s  (%d verts, %d idx)", npz_path, len(vertices), len(indices))

    meta_path = out_dir / "roads_meta.json"
    with open(meta_path, "w") as f:
        json.dump(edge_meta, f)

    return npz_path
