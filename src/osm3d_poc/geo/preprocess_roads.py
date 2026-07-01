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
    offset_m: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 2-D polyline in local XZ to a flat road-strip mesh with miter joins.

    Builds a continuous strip where adjacent quads share vertices at each
    intermediate point.  At corners the shared vertex is placed at the miter
    intersection of the two edge lines, which eliminates gaps.  Miter length is
    capped at 3× half-width to prevent extreme spikes at very sharp corners.

    Vertex layout (even = left, odd = right):
        0,1 — start pair
        2,3 — second pair (shared between segments 0 and 1)
        …
        2*(N-1), 2*(N-1)+1 — end pair

    Triangles per segment i: (L_i, R_{i+1}, R_i) and (L_i, L_{i+1}, R_{i+1})
    — CCW when viewed from above (+Y).

    Args:
        xz:    (N, 2) float array of local (x, z) coordinates.
        width: road width in metres.
        y:     world Y for all vertices (road_y_m from config).

    Returns:
        vertices: float32 (2*N, 3)
        indices:  uint32  (6*(N-1),)
    """
    # Drop consecutive duplicate points that would produce degenerate segments.
    pts: List[np.ndarray] = [xz[0]]
    for pt in xz[1:]:
        if np.hypot(pt[0] - pts[-1][0], pt[1] - pts[-1][1]) >= 1e-6:
            pts.append(pt)
    if len(pts) < 2:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint32)

    n = len(pts)
    half = width * 0.5
    _MAX_MITER = 3.0   # cap as multiple of half_width

    # Per-segment perpendicular (left-side = CCW rotation of direction).
    perps: List[np.ndarray] = []
    for i in range(n - 1):
        d = pts[i + 1] - pts[i]
        d = d / np.hypot(d[0], d[1])
        perps.append(np.array([-d[1], d[0]]))

    # Build one (left, right) pair per polyline vertex.
    verts: List[list] = []
    for i in range(n):
        if i == 0:
            mdir, scale = perps[0], half
        elif i == n - 1:
            mdir, scale = perps[-1], half
        else:
            avg = perps[i - 1] + perps[i]
            avg_len = float(np.hypot(avg[0], avg[1]))
            if avg_len < 1e-6:              # 180° hairpin — use previous perp
                mdir, scale = perps[i - 1], half
            else:
                avg /= avg_len
                cos_h = float(np.dot(avg, perps[i - 1]))
                scale = half / max(cos_h, 1.0 / _MAX_MITER)
                mdir = avg
        x, z = pts[i]
        # offset_m shifts the strip centre left (Japan LHT: positive = left lane)
        cx, cz = x + mdir[0] * offset_m, z + mdir[1] * offset_m
        verts.append([cx + mdir[0] * scale, y, cz + mdir[1] * scale])  # left edge
        verts.append([cx - mdir[0] * scale, y, cz - mdir[1] * scale])  # right edge

    # Triangles: CCW from above — (L0, R1, R0) and (L0, L1, R1) per segment.
    idxs: List[int] = []
    for i in range(n - 1):
        l0, r0 = 2 * i,     2 * i + 1
        l1, r1 = 2 * i + 2, 2 * i + 3
        idxs.extend([l0, r1, r0, l0, l1, r1])

    return (
        np.array(verts, dtype=np.float32),
        np.array(idxs,  dtype=np.uint32),
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
