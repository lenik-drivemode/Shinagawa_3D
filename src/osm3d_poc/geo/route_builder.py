"""Route generation: waypoints → shortest paths → densify → route JSON (PRD §7.4, §10.4).

Algorithm (PRD §11 Phase 5):
  1. Snap configured waypoints to nearest graph nodes (ox.nearest_nodes).
  2. Compute shortest path between each consecutive pair (networkx, weight="length").
  3. Concatenate node sequences, removing duplicate join nodes.
  4. Project node lat/lon to local x/z.
  5. Densify: insert intermediate points so no segment exceeds max_seg_m (PRD §13.3).
  6. Compute cumulative arc-length s.
  7. Save JSON per §10.4.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_DEFAULT_MAX_SEG_M = 5.0   # densification target (PRD §13.3)
_CLOSED_LOOP_THRESH_M = 100.0   # start/end within this → mark closed loop


# ---------------------------------------------------------------------------
# Waypoint snapping (FR-ROUTE-003)
# ---------------------------------------------------------------------------

def snap_waypoints(G, waypoints: List[dict]) -> List:
    """Snap each {lat, lon} waypoint to the nearest graph node ID.

    Uses a cosine-corrected Euclidean distance in lat/lon (equivalent to
    haversine for nearest-neighbour search on a small area like Shinagawa).
    Works with any networkx graph that has 'x' (lon) and 'y' (lat) node attrs.
    """
    node_ids = list(G.nodes)
    lats = np.array([G.nodes[n]["y"] for n in node_ids])
    lons = np.array([G.nodes[n]["x"] for n in node_ids])
    result = []
    for wp in waypoints:
        dlat  = lats - wp["lat"]
        dlon  = (lons - wp["lon"]) * np.cos(np.radians(wp["lat"]))
        d2    = dlat ** 2 + dlon ** 2
        result.append(node_ids[int(np.argmin(d2))])
    return result


# ---------------------------------------------------------------------------
# Path building (FR-ROUTE-003, FR-ROUTE-004)
# ---------------------------------------------------------------------------

def build_node_path(G, node_ids: List) -> Optional[List]:
    """Concatenate shortest paths between consecutive nodes.

    Returns None if any segment has no path (logs a warning).
    Duplicate join nodes between adjacent segments are removed.
    """
    import networkx as nx
    full_path: List = []
    for i in range(len(node_ids) - 1):
        u, v = node_ids[i], node_ids[i + 1]
        if u == v:
            if not full_path:
                full_path.append(u)
            continue
        try:
            seg = nx.shortest_path(G, source=u, target=v, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            log.warning("No path from node %s to node %s: %s", u, v, exc)
            return None
        if full_path:
            full_path.extend(seg[1:])   # skip first node — already the last of previous seg
        else:
            full_path.extend(seg)
    return full_path if len(full_path) >= 2 else None


# ---------------------------------------------------------------------------
# Densification (FR-ROUTE-005, PRD §13.3)
# ---------------------------------------------------------------------------

def _densify_columns(cols: np.ndarray, max_seg_m: float) -> np.ndarray:
    """Densify an (N, D) float64 array so no XZ segment (cols[:,0:2]) exceeds max_seg_m.

    All D columns are linearly interpolated; only columns 0 and 1 (x, z) are
    used for distance measurement.
    """
    if len(cols) < 2:
        return cols.copy()
    out = [cols[0]]
    for i in range(len(cols) - 1):
        p0, p1 = cols[i], cols[i + 1]
        dist = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n_steps = max(1, int(np.ceil(dist / max_seg_m)))
        for k in range(1, n_steps + 1):
            t = k / n_steps
            out.append(p0 * (1.0 - t) + p1 * t)
    return np.array(out, dtype=np.float64)


def densify_xz(xz: np.ndarray, max_seg_m: float = _DEFAULT_MAX_SEG_M) -> np.ndarray:
    """Return densified (M, 2) x/z array where every segment ≤ max_seg_m."""
    result = _densify_columns(np.asarray(xz, dtype=np.float64), max_seg_m)
    return result[:, :2]


# ---------------------------------------------------------------------------
# Cumulative arc-length (FR-ROUTE-005)
# ---------------------------------------------------------------------------

def compute_cumulative_s(xz: np.ndarray) -> np.ndarray:
    """Return float64 (N,) cumulative arc-length array starting at 0."""
    xz = np.asarray(xz, dtype=np.float64)
    diffs = np.diff(xz, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


# ---------------------------------------------------------------------------
# Edge-geometry extraction
# ---------------------------------------------------------------------------

def _collect_path_latlons(G, node_path: List) -> np.ndarray:
    """Walk node_path edge-by-edge and collect the full lat/lon sequence.

    Uses each edge's LineString geometry when present (curved roads), otherwise
    falls back to a straight line between the two endpoint nodes.  Adjacent
    segments share their join point so no duplicates are introduced.
    """
    segments: List[np.ndarray] = []
    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]

        # Pick the shortest-length parallel edge (multigraph).
        best_data: Optional[dict] = None
        best_len = float("inf")
        for data in G[u][v].values():
            l = data.get("length", float("inf"))
            if l < best_len:
                best_len = l
                best_data = data

        geom = best_data.get("geometry") if best_data else None
        if geom is not None:
            coords = np.array(geom.coords)   # shape (K, 2): lon, lat
            seg = coords[:, ::-1]             # → lat, lon
            # osmnx stores geometry from u to v for directed edges; verify
            # by checking which end is closer to node u.
            u_ll = np.array([G.nodes[u]["y"], G.nodes[u]["x"]])
            if np.linalg.norm(seg[0] - u_ll) > np.linalg.norm(seg[-1] - u_ll):
                seg = seg[::-1]
        else:
            seg = np.array([
                [G.nodes[u]["y"], G.nodes[u]["x"]],
                [G.nodes[v]["y"], G.nodes[v]["x"]],
            ])

        # Skip the first point of every segment after the first (already
        # appended as the last point of the previous segment).
        segments.append(seg if not segments else seg[1:])

    if not segments:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(segments, axis=0)


# ---------------------------------------------------------------------------
# Full route pipeline
# ---------------------------------------------------------------------------

def build_route(G, projector, cfg: dict) -> dict:
    """Build the complete route dict (PRD §10.4) from graph + config waypoints.

    Raises RuntimeError if shortest-path computation fails for any segment.
    """
    wps       = cfg["route"]["waypoints"]
    route_y   = cfg["preprocess"]["route_y_m"]
    max_seg   = _DEFAULT_MAX_SEG_M
    d_min     = cfg["route"]["desired_min_length_m"]
    d_max     = cfg["route"]["desired_max_length_m"]
    speed_kmh = cfg["route"]["default_speed_kmh"]

    log.info("Snapping %d waypoints to graph nodes …", len(wps))
    node_ids = snap_waypoints(G, wps)
    log.info("Snapped nodes: %s", node_ids)

    log.info("Computing shortest paths between consecutive waypoints …")
    node_path = build_node_path(G, node_ids)
    if node_path is None:
        raise RuntimeError(
            "Route generation failed: one or more path segments could not be found. "
            "Check that the graph covers the waypoint area and is connected."
        )

    log.info("Route uses %d graph nodes; collecting edge geometries …", len(node_path))
    latlons = _collect_path_latlons(G, node_path)
    lats, lons = latlons[:, 0], latlons[:, 1]
    xs, zs = projector.batch_to_xz(lats, lons)

    # Densify all four columns together for consistent interpolation
    cols  = np.column_stack([xs, zs, lats, lons])   # (N, 4)
    dense = _densify_columns(cols, max_seg)
    xs_d   = dense[:, 0]
    zs_d   = dense[:, 1]
    lats_d = dense[:, 2]
    lons_d = dense[:, 3]

    s_arr        = compute_cumulative_s(np.column_stack([xs_d, zs_d]))
    total_length = float(s_arr[-1])

    if total_length < d_min:
        log.warning(
            "Route length %.1f m < desired minimum %.1f m — consider adding waypoints.",
            total_length, d_min,
        )
    elif total_length > d_max:
        log.warning(
            "Route length %.1f m > desired maximum %.1f m — consider removing waypoints.",
            total_length, d_max,
        )
    else:
        log.info(
            "Route length %.1f m — within desired range [%.1f, %.1f] m.",
            total_length, d_min, d_max,
        )

    start_end_dist = float(np.hypot(xs_d[-1] - xs_d[0], zs_d[-1] - zs_d[0]))
    closed_loop    = start_end_dist < _CLOSED_LOOP_THRESH_M
    log.info(
        "Start/end distance %.1f m → closed_loop=%s", start_end_dist, closed_loop
    )

    points = [
        {
            "x":   float(xs_d[i]),
            "y":   float(route_y),
            "z":   float(zs_d[i]),
            "lat": float(lats_d[i]),
            "lon": float(lons_d[i]),
            "s":   float(s_arr[i]),
        }
        for i in range(len(xs_d))
    ]

    return {
        "schema_version":    SCHEMA_VERSION,
        "name":              "Shinagawa 4-5 km loop",
        "closed_loop":       closed_loop,
        "speed_default_kmh": int(speed_kmh),
        "length_m":          total_length,
        "points":            points,
        "source": {
            "method":    "osmnx_shortest_paths_between_waypoints",
            "waypoints": wps,
        },
    }


# ---------------------------------------------------------------------------
# Serialisation / loading (FR-ROUTE-004, FR-ROUTE-005)
# ---------------------------------------------------------------------------

def save_route(route_dict: dict, cfg: dict) -> Path:
    """Write route JSON to configured output file."""
    out_path = Path(cfg["route"]["output_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(route_dict, f, indent=2)
    log.info(
        "Saved route → %s  (%.1f m, %d points)",
        out_path, route_dict["length_m"], len(route_dict["points"]),
    )
    return out_path


def load_route(cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Load route and return (xyz float32 [N,3], s float32 [N]).

    The runtime only needs x,y,z and cumulative distance — not lat/lon.
    """
    path = Path(cfg["route"]["output_file"])
    with open(path) as f:
        data = json.load(f)
    pts = data["points"]
    xyz = np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=np.float32)
    s   = np.array([p["s"] for p in pts], dtype=np.float32)
    return xyz, s
