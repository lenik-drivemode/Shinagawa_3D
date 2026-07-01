# Shinagawa 3D — Python Desktop OSM Map Renderer

A proof-of-concept desktop application that renders a 3D map of Tokyo's Shinagawa area
using real OpenStreetMap data and animates a vehicle marker along a 4–5 km closed-loop
route.

## Purpose

Validate that a Python + OpenGL stack (OSMnx + ModernGL) can deliver a visually clear
local 3D map with route simulation at 30–60 FPS on a normal laptop.

## Expected Visual Output

- Grey road strips on a dark background.
- Off-white extruded buildings with diffuse shading.
- Coloured route highlight overlaid on the roads.
- A small coloured cuboid marker that drives along the route and rotates with heading.
- FPS, camera mode, and speed shown in the window title bar.

## System Requirements

- Linux (primary), Windows or macOS (secondary — untested)
- Python 3.10 or 3.11+
- OpenGL 3.3+ capable GPU or Mesa software renderer
- ~200 MB disk space for OSM cache and processed assets
- Internet access for first-time OSM data download; subsequent runs use the local cache

## Installation

```bash
git clone <repo-url> shinagawa_3d
cd shinagawa_3d
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

On Debian/Ubuntu, if GeoPandas fails to install, add system dependencies first:

```bash
sudo apt install libgdal-dev libspatialindex-dev
```

## Running the Pipeline

### Option A — all steps at once

```bash
source .venv/bin/activate
python scripts/run_all.py --config config/shinagawa_poc.yaml
```

Add `--rebuild` to force re-preprocessing of already-cached assets.
Add `--no-viewer` to run preprocessing only (useful on headless servers).
Add `--small-bbox` to use a smaller debug area (~0.5 km²) for faster iteration.

### Option B — step by step

Steps 1–3 only need to run once (or when you change the config/area).
Step 4 is the only command needed for subsequent launches.

```bash
# 1. Download OSM road graph and building footprints (cached after first run)
python scripts/01_download_osm.py --config config/shinagawa_poc.yaml

# 2. Preprocess into GPU-ready mesh files (roads_mesh.npz, buildings_mesh.npz)
python scripts/02_preprocess_assets.py --config config/shinagawa_poc.yaml

# 3. Build the route JSON (shinagawa_loop.json)
python scripts/03_build_route.py --config config/shinagawa_poc.yaml

# 4. Launch the 3D viewer (this is the only step needed on subsequent runs)
python scripts/04_run_viewer.py --config config/shinagawa_poc.yaml
```

### Viewer flags

| Flag | Description |
|------|-------------|
| `--no-buildings` | Skip building rendering (significantly faster load and FPS) |
| `--wireframe` | Render geometry in wireframe mode |
| `--top-down` | Start in top-down orthographic camera view |
| `--follow` | Start in close-follow-vehicle camera mode |
| `--speed-kmh 40` | Override vehicle simulation speed (default from config) |
| `--small-bbox` | Use smaller debug bounding box |
| `--show-stats` | Print mesh vertex/triangle counts at startup |
| `--screenshot PATH` | Save a screenshot to PATH after the first frame, then exit |
| `--debug` | Log FPS, camera mode, and route progress every second |
| `--demo-cube` | Render a rotating demo cube (Phase 0 smoke-test, no OSM data needed) |

## Keyboard Controls

| Key | Action |
|-----|--------|
| `W` / `S` | Move camera target forward / backward |
| `A` / `D` | Move camera target left / right |
| `Mouse left drag` | Orbit camera (yaw and pitch) |
| `Mouse right drag` | Pan camera target |
| `Mouse wheel` | Zoom in / out |
| `F` | Cycle camera mode: orbit → follow-close → follow-aerial |
| `T` | Toggle top-down orthographic view |
| `Space` | Pause / resume vehicle |
| `R` | Reset marker to route start |
| `[` / `]` | Decrease / increase speed by 10 km/h (clamped 5–200 km/h) |
| `P` | Save screenshot to `screenshot_<timestamp>.png` |
| `Esc` | Exit |

## Area of Interest

Shinagawa / Takanawa / Konan / Osaki / Gotanda, Tokyo.

Default bounding box: 35.6150 – 35.6420 N, 139.7150 – 139.7600 E (~3 × 4 km).

To use a smaller debug area, add `--small-bbox` to any script.

## OSM Attribution

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
licensed under the [Open Database License](https://opendatacommons.org/licenses/odbl/).

## Known Limitations

- No live GPS, turn-by-turn navigation, or dynamic routing.
- Building polygon holes are ignored in the MVP (flat-roof extrusion only).
- Intersection road joins may have visual gap artifacts at corners.
- No terrain elevation data (all geometry at sea level, Y = 0).
- No street-name label rendering.
- No map tile streaming — one fixed local area only.
- **In-window text overlay deferred to post-MVP.** Route progress, speed, and
  camera mode are reported to the console and window title bar rather than
  rendered on-screen. FPS is shown in the window title. See PRD §FR-UI-005.
- Developed on Python 3.10; PRD targets 3.11+ but 3.10 is fully compatible.

## Troubleshooting

**OpenGL context fails to open**
Check GPU drivers or Mesa version. For a software fallback:
```bash
LIBGL_ALWAYS_SOFTWARE=1 python scripts/04_run_viewer.py --config config/shinagawa_poc.yaml
```

**GeoPandas or GDAL install fails**
Use conda/mamba, or install system geospatial libraries first (`libgdal-dev`
on Debian/Ubuntu).

**OSM download fails (network error)**
Retry later — the Overpass API can be temporarily unavailable. OSM data is
cached after the first successful download; subsequent runs work offline.

**Route file not found / route generation fails**
Run `scripts/03_build_route.py` and check the log output. If no path exists
between waypoints, a `route_diagnostics.json` file is saved in the data
directory with snapped node IDs for debugging. Adjust waypoints in
`config/shinagawa_poc.yaml` or expand the bounding box.

**Low FPS**
- Run with `--no-buildings` for a large speed-up.
- Use `--small-bbox` for a smaller scene.
- Check that the GPU (not CPU/software Mesa) is being used.

**Missing Pillow (screenshot fails)**
```bash
pip install Pillow
```

## Performance Notes

- Target: 60 FPS at 1280 × 720 with the default dataset.
- Minimum acceptable: 30 FPS.
- Preprocessing: under 2 minutes for the default bounding box on a normal laptop.
- Asset loading: under 10 seconds after preprocessing.
- All road and building geometry is uploaded to the GPU once at startup; the render
  loop has no per-frame Python-side geometry work (NFR-PERF-003/004).
- Draw call count: ≤ 4 per frame (roads, route strip, buildings, marker).

## Future Work

See `shinagawa_3d.prd` §18 and §20 for the full list. Short-term candidates:

- In-window status overlay using bitmap font or Dear ImGui
- Proper road intersection mitre joins
- Street-name labels via signed distance field fonts
- Terrain elevation via DEM (e.g., SRTM)
- Tile-based asset loading for larger areas
- C++/Rust preprocessing backend for city-scale datasets
