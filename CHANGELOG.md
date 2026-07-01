# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

## [0.9.0] — 2026-07-02 — Phase 8: Polish & Diagnostics

### Added
- `scripts/run_all.py` — full pipeline orchestrator (01→04 in order); flags:
  `--rebuild` (force re-preprocess), `--no-viewer` (preprocessing only),
  `--small-bbox`, `--config`; stops with clear error on first failure
- `src/osm3d_poc/render/screenshot.py` — `save_screenshot(bytes, w, h, path)`:
  reshapes raw RGB framebuffer, flips Y (OpenGL bottom-left → image top-left),
  saves as PIL PNG; raises `ValueError` on wrong buffer size, `ImportError` if
  Pillow is absent
- `src/osm3d_poc/render/renderer.py` updated — Phase 8 additions:
  - P key: queues screenshot to `screenshot_<unix_ts>.png`, captured next frame
  - `--screenshot PATH` flag: saves screenshot after first frame
  - `--debug` flag: reduces progress log interval from 5 s → 1 s; adds camera
    mode to each progress log line
  - `_pending_screenshot` attribute; screenshot is captured after all draw calls
    to include the full scene
- `scripts/04_run_viewer.py` updated: `--screenshot PATH`, `--debug` flags
- `requirements.txt`: added `Pillow>=9.0` for PNG screenshot support
- `README.md` overhauled: corrected controls table (removed Q/E and arrow keys
  that were never implemented; added `[/]` speed, `P` screenshot); added
  `run_all.py` usage section; added `--screenshot`/`--debug` flags; fixed
  troubleshooting entries; added performance draw-call count note
- `tests/test_screenshot.py` — 8 tests: file creation, return type, dimensions,
  RGB mode, Y-flip verification (blue top row / red bottom row after flip),
  solid-color preservation, wrong-buffer-size ValueError, string path accepted
- `tests/test_run_all.py` — 13 tests: default 4-step ordering, `--no-viewer`
  skips step 4, `--config` propagated to all, `--small-bbox` propagated to all,
  `--rebuild` only on preprocess+route scripts, script paths under scripts_dir,
  subprocess success→True / failure→False / early-stop on first failure

### Notes
- Screenshot is queued (not captured immediately on keypress) so the full scene
  — including the animated marker — is present in the image
- `--debug` flag is intended for development/diagnostics; `--show-stats` still
  prints mesh counts at startup as a separate one-shot mode
- In-window text overlay remains deferred to post-MVP (see Known Limitations)

## [0.8.0] — 2026-07-02 — Phase 7: Vehicle Simulation

### Added
- `src/osm3d_poc/sim/route_player.py` — `RoutePlayer` class (FR-SIM-000…007):
  - `update(dt)`: advances arc-length position; wraps at route end (FR-SIM-004)
  - `get_pose()`: binary-search interpolation → `(pos float32[3], heading float)`;
    heading = `atan2(dx, dz)` so 0=north, π/2=east (matches `rotate_y` convention)
  - `set_paused`/`toggle_pause`/`reset`; `speed_mps`/`speed_kmh` with clamp [5, 200] (FR-SIM-005)
  - Starts running immediately (FR-SIM-000); no scikit-learn, no osmnx CRS dependency
- `src/osm3d_poc/render/renderer.py` updated — Phase 7 integration:
  - Creates `RoutePlayer` in `_load_assets` from `cfg["route"]["default_speed_kmh"]`
  - `render()` calls `player.update(frame_time)`, `get_pose()`, and
    `camera.set_follow_target(pos, heading)` each frame
  - Marker model matrix now uses live `pos` + `heading` from player (animated)
  - FPS title extended: `Shinagawa 3D | N FPS | mode | K km/h [| PAUSED]`
  - Route progress logged every 5 s (FR-UI-002)
  - Startup controls hint logged at INFO level (FR-UI-004)
  - New key handlers: Space (pause/resume), R (reset), [ (−10 km/h), ] (+10 km/h)
  - `--follow` / `--top-down` CLI flags now apply camera mode at startup
- `tests/test_route_player.py` — 27 tests: initial state, single/multi-frame advance,
  speed proportionality, wrap-around (position near start after wrap), pause (no
  advance), resume, toggle, reset (dist=0, position=route[0]), position interpolation
  (midpoint, boundary, second segment), dtype, no NaN, heading east (π/2) and north
  (0), speed setter/clamp (min 5 km/h, max 200 km/h), increment/decrement clamping

### Notes
- `_KEY_BRACKET_L = 91`, `_KEY_BRACKET_R = 93` (raw ASCII) work on both pyglet
  and glfw backends without requiring backend-specific key constants
- RoutePlayer uses `np.searchsorted(side="right") - 1` for O(log N) segment lookup;
  clipped to `[0, N-2]` so dist exactly at end snaps to last segment

## [0.7.0] — 2026-07-02 — Phase 6: Renderer MVP

### Added
- `src/osm3d_poc/render/camera.py` — Camera class + standalone matrix functions
  (`look_at`, `perspective`, `ortho`, `translate`, `rotate_y`); orbit / top-down /
  follow-close / follow-aerial modes; mouse orbit/pan, scroll zoom, W/S/A/D
  movement; F cycles follow modes, T toggles top-down orthographic view
- `src/osm3d_poc/render/shaders.py` — GLSL sources: `FLAT_VERT`/`FLAT_FRAG`
  (roads, route, marker) + `BUILDING_VERT`/`BUILDING_FRAG` (diffuse, abs-dot for
  two-sided normals, hardcoded sun direction)
- `src/osm3d_poc/render/mesh.py` — `GpuMesh` wrapper (VBO+IBO→VAO, one draw call)
- `src/osm3d_poc/render/marker.py` — `make_marker_mesh`: 8-vertex cuboid in
  marker-local space (±width/2 in X, [0, height] in Y, ±length/2 in Z); 36 indices
- `src/osm3d_poc/render/renderer.py` — `Renderer(mglw.WindowConfig)`: loads roads/
  buildings/route .npz + JSON, uploads all to GPU, draws in order (roads → route →
  buildings → marker), FPS+mode in window title every 0.5s, keyboard/mouse handlers;
  `run_viewer(cfg, cli_args)` entry point; graceful warnings when assets missing
- `scripts/04_run_viewer.py` updated to call `run_viewer` (was "not yet implemented")
- `tests/test_camera.py` — 29 tests: look_at (shape, dtype, eye→origin, target→-Z,
  up alignment), perspective (shape, dtype, w-component, FOV/aspect scaling), ortho
  (centre, corners), translate, rotate_y, Camera mode cycling/clamping/matrix shapes
- `tests/test_marker.py` — 11 tests: vertex/index count, dtype, indices in range,
  no NaN, X/Z symmetry, Y range, vertex shape, real config round-trip

### Notes
- `rotate_y(+π/2)` maps +X → -Z (right-hand rule) — used in marker heading transform
- Marker model matrix: `translate(x, marker_y_m, z) @ rotate_y(heading)`; heading=0
  in Phase 6 (static); Phase 7 RoutePlayer provides actual heading each frame
- Building shader uses `abs(dot(normal, light))` for two-sided diffuse (PRD note)
- Renderer gracefully skips missing .npz/route files with log warnings

## [0.6.0] — 2026-07-02 — Phase 5: Route Generation

### Added
- `src/osm3d_poc/geo/route_builder.py` — complete route pipeline:
  `snap_waypoints` (cosine-corrected Euclidean nearest-node, no scikit-learn dep),
  `build_node_path` (nx.shortest_path with weight="length", dedup join nodes),
  `_densify_columns` (multi-column linear interpolation, distance via XZ cols 0-1),
  `densify_xz` (≤5 m segment length per PRD §13.3),
  `compute_cumulative_s` (float64 arc-length, s[0]=0),
  `build_route` (full pipeline → §10.4 JSON dict),
  `save_route` / `load_route` (JSON → float32 xyz, s arrays)
- `scripts/03_build_route.py` — CLI: `--config`, `--small-bbox`, `--rebuild`;
  writes `data/routes/shinagawa_loop.json`; saves diagnostic JSON on failure
  (NFR-REL-003); updates metadata.json with route counts
- `tests/test_route_builder.py` — 34 tests: densification (no-change, insertion,
  endpoint preservation, max-seg enforcement, multi-dim interpolation),
  cumulative-s (zero start, monotone, dtype, correctness), build_node_path
  (simple, loop, dedup, no-path, same-node), build_route (keys, schema,
  fields, s monotone, length == last s, route_y, densification), save/load
  roundtrip

### Notes
- `snap_waypoints` uses numpy argmin on cosine-corrected lat/lon distance —
  avoids scikit-learn and osmnx's `G.graph["crs"]` requirement;
  equivalent to haversine for nearest-node search on Shinagawa scale
- Route lat/lon columns are linearly interpolated for densified points
  (geographic error < 1 mm over 5 m segments — acceptable for MVP)
- Route JSON follows §10.4 format; `load_route` returns float32 arrays
  for direct GPU use in Phase 7

## [0.5.0] — 2026-07-02 — Phase 4: Building Mesh Generation

### Added
- `src/osm3d_poc/geo/preprocess_buildings.py` — building extrusion pipeline:
  `parse_height` (OSM tag formats), `get_building_height` (priority: height tag
  > building:levels × floor_h > default), `_clean_polygon` (shapely.make_valid),
  `_extrude_building` (walls + flat roof via mapbox-earcut, CCW-oriented rings),
  `build_building_mesh`, `save_building_mesh`
- `scripts/02_preprocess_assets.py` extended for buildings: `--skip-buildings`
  flag now functional; updates metadata.json with building counts
- `tests/test_preprocess_buildings.py` — 46 tests: height parsing, height
  priority chain, extrusion geometry (vertex count, index count, dtype, no NaN,
  indices in range, Y range, roof/wall normals), polygon helpers, full pipeline,
  save/roundtrip

### Notes
- Vertex format: float32 (x, y, z, nx, ny, nz); roof normals (0,1,0), wall
  normals outward-horizontal via CCW-oriented Shapely exterior ring
- Renderer should use abs(dot(normal, light)) for two-sided shading robustness
- Polygon holes ignored (FR-BLDG-008, documented in README Known Limitations)
- mapbox-earcut 2.0 API: takes (N,2) float64 array + ring-size uint32 array

## [0.4.0] — 2026-07-02 — Phase 3: Road Mesh Generation

### Added
- `src/osm3d_poc/geo/preprocess_roads.py` — road strip mesh pipeline:
  `parse_width`, `highway_width`, `get_edge_width` (OSM tag → fallback table),
  `polyline_to_strip` (per-segment quads, MVP corner gaps accepted per FR-ROAD-006),
  `build_road_mesh` (whole graph → single combined mesh), `save_road_mesh` (.npz + .json)
- `scripts/02_preprocess_assets.py` — CLI: `--config`, `--small-bbox`,
  `--skip-buildings`, `--rebuild`; updates metadata.json with road counts
  and projection origin
- `tests/test_preprocess_roads.py` — 47 tests: width parsing, highway fallback
  table, edge-width clamping, strip geometry (quad counts, axis directions,
  Y value, dtype), mesh validity (no NaN, all indices in range), roundtrip save

### Notes
- Vertex format: float32 (x, y, z); uniform road color is set by renderer
  shader — per-vertex color is post-MVP
- Degenerate segments (length < 1e-6 m) are silently skipped
- Width out-of-range (< 1 m or > 50 m) falls back to highway-type default

## [0.3.0] — 2026-07-02 — Phase 2: Projection & Local Coordinates

### Added
- `src/osm3d_poc/geo/projection.py` — centralised WGS84 → UTM 54N → local
  XZ metre conversion (NFR-MAINT-003); `Projector` class bundles transformer
  and origin; `batch_to_xz` and `polygon_to_xz` for NumPy array workflows
- `tests/test_projection.py` — 14 tests: UTM range checks, origin→(0,0),
  axis sign convention (X east / Z north), plausible distances, batch vs
  single-point consistency, polygon exterior projection

### Notes
- pyproj 3.7.1; `always_xy=True` so argument order is always (lon, lat)
  for geographic input and (easting, northing) for projected output
- Two pyproj internal DeprecationWarnings on 1-element array input are
  from pyproj internals, not project code; harmless for now

## [0.2.0] — 2026-07-02 — Phase 1: Data Acquisition

### Added
- `src/osm3d_poc/geo/download_osm.py` — road graph and building footprint
  acquisition via osmnx 2.x (`bbox=(west, south, east, north)` format);
  cache detection; initial `metadata.json` writer
- `scripts/01_download_osm.py` — CLI: `--config`, `--small-bbox`, `--no-cache`
- `tests/test_download_osm.py` — 16 tests covering bbox extraction, cache
  path helpers, and metadata structure (no network required)

### Notes
- osmnx 2.0.7 in use; `bbox` parameter is `(left, bottom, right, top)`
  = `(west, south, east, north)` — different from osmnx 1.x positional order
- Deferred fields in metadata.json (vertices, triangles, route length)
  are written as `null` and filled in by later phases

## [0.1.0] — 2026-07-02 — Phase 0: Environment Setup

### Added
- Project directory scaffold: `src/osm3d_poc/`, `config/`, `data/`, `scripts/`, `tests/`
- `requirements.txt` with all Phase 0–8 dependencies
- `pyproject.toml` for editable package install (`pip install -e .`)
- `config/shinagawa_poc.yaml` — full configuration including all v1.1 additions
  (marker_y_m, initial camera angles, follow-camera distances, speed bounds)
- `src/osm3d_poc/config.py` — YAML config loader
- `src/osm3d_poc/logging_config.py` — structured logging setup
- `src/osm3d_poc/render/demo_cube.py` — rotating coloured cube via ModernGL;
  displays FPS in window title; Esc to exit
- `scripts/04_run_viewer.py` — viewer entry point with all CLI flags;
  `--demo-cube` runs the Phase 0 smoke-test
- `tests/test_config.py` — 11 config validation tests (all passing)

### Notes
- Python 3.10 in use (PRD specifies 3.11+); all code is compatible with 3.10
- Window library: `moderngl-window` with `pyglet` backend
- In-window text overlay deferred to post-MVP (see README Known Limitations)
