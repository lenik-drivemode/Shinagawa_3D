# Shinagawa 3D — Python Desktop OSM Map Renderer

A proof-of-concept desktop application that renders a 3D map of Tokyo's Shinagawa area using real OpenStreetMap data and animates a vehicle marker along a 4–5 km closed-loop route.

## Purpose

Validate that a Python + OpenGL stack (OSMnx + ModernGL) can deliver a visually clear local 3D map with route simulation at 30–60 FPS on a normal laptop.

## System Requirements

- Linux (primary), Windows or macOS (secondary)
- Python 3.11+
- OpenGL 3.3+ capable GPU / Mesa driver
- Internet access for first-time OSM data download (subsequent runs use cache)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Debian/Ubuntu, you may need system packages for GeoPandas/GDAL:

```bash
sudo apt install libgdal-dev libspatialindex-dev
```

## Usage

Run the pipeline in order:

```bash
# 1. Download OSM data (cached after first run)
python scripts/01_download_osm.py --config config/shinagawa_poc.yaml

# 2. Preprocess into render-ready assets
python scripts/02_preprocess_assets.py --config config/shinagawa_poc.yaml

# 3. Build the route
python scripts/03_build_route.py --config config/shinagawa_poc.yaml

# 4. Launch the viewer
python scripts/04_run_viewer.py --config config/shinagawa_poc.yaml
```

Or run everything in one go:

```bash
python scripts/run_all.py --config config/shinagawa_poc.yaml --rebuild
```

### Viewer flags

| Flag | Description |
|------|-------------|
| `--no-buildings` | Skip building rendering (faster) |
| `--wireframe` | Render geometry in wireframe mode |
| `--top-down` | Start in top-down camera mode |
| `--follow` | Start in follow-vehicle camera mode |
| `--speed-kmh 40` | Set vehicle speed |
| `--window 1280x720` | Override window size |
| `--show-stats` | Show detailed render stats |
| `--small-bbox` | Use smaller debug bounding box |

## Keyboard Controls

| Key | Action |
|-----|--------|
| W / S | Move camera forward / backward |
| A / D | Move camera left / right |
| Q / E | Rotate camera left / right |
| Arrow keys | Rotate / pitch camera |
| Mouse wheel | Zoom in / out |
| Space | Pause / resume vehicle |
| R | Reset marker to route start |
| F | Toggle follow-vehicle camera |
| T | Toggle top-down camera |
| Esc | Exit |

## Area of Interest

Shinagawa / Takanawa / Konan / Osaki / Gotanda, Tokyo.

Default bounding box: 35.6150 – 35.6420 N, 139.7150 – 139.7600 E (~3 × 4 km).

## OSM Attribution

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), licensed under the [Open Database License](https://opendatacommons.org/licenses/odbl/).

## Known Limitations

- No live GPS or turn-by-turn navigation.
- Building polygon holes are ignored in the MVP (flat-roof extrusion only).
- Intersection road joins may have visual overlap artifacts.
- No terrain elevation.
- No label rendering.
- No map streaming — one fixed local area only.

## Troubleshooting

**OpenGL context fails:** Check GPU drivers, Mesa version, or try `LIBGL_ALWAYS_SOFTWARE=1` for a software fallback.

**GeoPandas install fails:** Use conda/mamba, or install system geospatial dependencies first.

**OSM download fails:** Retry later or use `--use-cache` if data was previously downloaded.

**Route not generated:** Adjust waypoints in `config/shinagawa_poc.yaml` or expand the bounding box.

**Low FPS:** Run with `--no-buildings`, reduce bounding box to debug area, or check GPU batching.

## Performance Notes

- Target: 60 FPS at 1280 × 720 with default dataset.
- Minimum: 30 FPS.
- Preprocessing: under 2 minutes for the default bounding box.
- Asset loading: under 10 seconds.
- All road and building geometry is uploaded to the GPU once at startup — no per-frame Python-side geometry work.

## Future Work

See the PRD (`shinagawa_3d.prd`) sections 18 and 20 for a full list. Short-term candidates:

- Tile-based asset loading for larger areas
- Proper road intersection joins
- Street name labels
- Terrain elevation via DEM
- C++/Rust preprocessing backend for larger datasets
