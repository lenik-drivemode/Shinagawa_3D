# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

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
