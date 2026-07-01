# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

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
