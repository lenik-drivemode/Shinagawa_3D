"""Tests for run_all.py pipeline orchestrator (PRD Phase 8)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# run_all.py lives in scripts/, not in the installed package.
# Import it directly by path manipulation.
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from run_all import build_cmds, _REBUILD_SCRIPTS  # noqa: E402


_FAKE_DIR = Path("/fake/scripts")
_DEFAULT_CONFIG = "config/shinagawa_poc.yaml"


# ---------------------------------------------------------------------------
# build_cmds: number of steps
# ---------------------------------------------------------------------------

def test_default_produces_four_steps():
    cmds = build_cmds(_DEFAULT_CONFIG, False, False, False, _FAKE_DIR)
    assert len(cmds) == 4


def test_no_viewer_produces_three_steps():
    cmds = build_cmds(_DEFAULT_CONFIG, False, False, True, _FAKE_DIR)
    assert len(cmds) == 3


def test_no_viewer_skips_04():
    cmds = build_cmds(_DEFAULT_CONFIG, False, False, True, _FAKE_DIR)
    scripts = [name for name, _ in cmds]
    assert "04_run_viewer.py" not in scripts


def test_all_four_scripts_present_by_default():
    cmds = build_cmds(_DEFAULT_CONFIG, False, False, False, _FAKE_DIR)
    scripts = [name for name, _ in cmds]
    assert scripts == [
        "01_download_osm.py",
        "02_preprocess_assets.py",
        "03_build_route.py",
        "04_run_viewer.py",
    ]


# ---------------------------------------------------------------------------
# build_cmds: --config propagation
# ---------------------------------------------------------------------------

def test_config_propagated_to_all_steps():
    cfg = "custom/path.yaml"
    cmds = build_cmds(cfg, False, False, False, _FAKE_DIR)
    for _, cmd in cmds:
        assert "--config" in cmd
        idx = cmd.index("--config")
        assert cmd[idx + 1] == cfg


# ---------------------------------------------------------------------------
# build_cmds: --small-bbox propagation
# ---------------------------------------------------------------------------

def test_small_bbox_propagated_to_all_steps():
    cmds = build_cmds(_DEFAULT_CONFIG, True, False, False, _FAKE_DIR)
    for _, cmd in cmds:
        assert "--small-bbox" in cmd


def test_no_small_bbox_not_in_cmds():
    cmds = build_cmds(_DEFAULT_CONFIG, False, False, False, _FAKE_DIR)
    for _, cmd in cmds:
        assert "--small-bbox" not in cmd


# ---------------------------------------------------------------------------
# build_cmds: --rebuild propagation
# ---------------------------------------------------------------------------

def test_rebuild_only_on_rebuild_scripts():
    cmds = build_cmds(_DEFAULT_CONFIG, False, True, False, _FAKE_DIR)
    for name, cmd in cmds:
        if name in _REBUILD_SCRIPTS:
            assert "--rebuild" in cmd, f"Expected --rebuild in {name}"
        else:
            assert "--rebuild" not in cmd, f"Unexpected --rebuild in {name}"


def test_rebuild_false_not_in_any_cmd():
    cmds = build_cmds(_DEFAULT_CONFIG, False, False, False, _FAKE_DIR)
    for _, cmd in cmds:
        assert "--rebuild" not in cmd


# ---------------------------------------------------------------------------
# build_cmds: script paths use scripts_dir
# ---------------------------------------------------------------------------

def test_script_paths_under_scripts_dir():
    cmds = build_cmds(_DEFAULT_CONFIG, False, False, False, _FAKE_DIR)
    for name, cmd in cmds:
        script_path = Path(cmd[1])
        assert script_path.parent == _FAKE_DIR


# ---------------------------------------------------------------------------
# run_pipeline: subprocess interaction (mocked)
# ---------------------------------------------------------------------------

def test_run_pipeline_success_returns_true():
    from run_all import run_pipeline
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("run_all.subprocess.run", return_value=mock_result) as mock_run:
        ok = run_pipeline(_DEFAULT_CONFIG, scripts_dir=_FAKE_DIR, no_viewer=True)
    assert ok is True
    assert mock_run.call_count == 3   # steps 1-3 only (no_viewer=True)


def test_run_pipeline_failure_returns_false():
    from run_all import run_pipeline
    mock_ok  = MagicMock(returncode=0)
    mock_err = MagicMock(returncode=1)
    with patch("run_all.subprocess.run", side_effect=[mock_ok, mock_err]):
        ok = run_pipeline(_DEFAULT_CONFIG, scripts_dir=_FAKE_DIR, no_viewer=True)
    assert ok is False


def test_run_pipeline_stops_on_first_failure():
    """Only 2 calls should happen: step 1 succeeds, step 2 fails → step 3 not run."""
    from run_all import run_pipeline
    mock_ok  = MagicMock(returncode=0)
    mock_err = MagicMock(returncode=1)
    with patch("run_all.subprocess.run", side_effect=[mock_ok, mock_err]) as mock_run:
        run_pipeline(_DEFAULT_CONFIG, scripts_dir=_FAKE_DIR, no_viewer=True)
    assert mock_run.call_count == 2
