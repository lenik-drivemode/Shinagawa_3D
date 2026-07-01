"""Tests for screenshot helper (PRD Phase 8)."""
from pathlib import Path

import numpy as np
import pytest

from osm3d_poc.render.screenshot import save_screenshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_buffer(width: int, height: int, fill_rgb=(128, 64, 32)) -> bytes:
    """Create a solid-color RGB framebuffer byte string."""
    arr = np.full((height, width, 3), fill_rgb, dtype=np.uint8)
    return arr.tobytes()


@pytest.fixture
def tmp_png(tmp_path):
    return tmp_path / "out.png"


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

def test_save_screenshot_creates_file(tmp_png):
    buf = _make_buffer(4, 4)
    save_screenshot(buf, 4, 4, tmp_png)
    assert tmp_png.exists()


def test_save_screenshot_returns_path(tmp_png):
    buf = _make_buffer(4, 4)
    result = save_screenshot(buf, 4, 4, tmp_png)
    assert isinstance(result, Path)
    assert result == tmp_png.resolve()


def test_save_screenshot_correct_dimensions(tmp_png):
    from PIL import Image
    buf = _make_buffer(16, 8)
    save_screenshot(buf, 16, 8, tmp_png)
    img = Image.open(tmp_png)
    assert img.size == (16, 8)   # PIL: (width, height)


def test_save_screenshot_rgb_mode(tmp_png):
    from PIL import Image
    buf = _make_buffer(4, 4)
    save_screenshot(buf, 4, 4, tmp_png)
    img = Image.open(tmp_png)
    assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# Y-flip: OpenGL origin is bottom-left, image origin is top-left
# ---------------------------------------------------------------------------

def test_y_flip_top_row_comes_from_bottom(tmp_png):
    """Pixel at OpenGL row 0 (bottom) should end up at image row height-1 (bottom)
    after flip — i.e., a red bottom row in GL becomes a red top row in the image."""
    from PIL import Image

    # 4×2 buffer: bottom row (row 0 in GL) is red, top row (row 1 in GL) is blue
    bottom_red = np.array([[255, 0, 0]] * 4, dtype=np.uint8)   # 4 red pixels
    top_blue   = np.array([[0, 0, 255]] * 4, dtype=np.uint8)   # 4 blue pixels
    # numpy shape (height=2, width=4, 3); row 0 is GL bottom
    arr = np.stack([bottom_red, top_blue])   # shape (2, 4, 3)
    buf = arr.tobytes()

    save_screenshot(buf, 4, 2, tmp_png)
    img = np.array(Image.open(tmp_png))   # shape (height, width, 3)

    # After flip: image row 0 = GL top = blue
    assert np.all(img[0] == [0, 0, 255]), "Image top row should be blue (GL top row)"
    # Image row 1 = GL bottom = red
    assert np.all(img[1] == [255, 0, 0]), "Image bottom row should be red (GL bottom row)"


def test_solid_color_preserved_through_flip(tmp_png):
    """Solid-color buffer: every pixel should have the same value after flip."""
    from PIL import Image
    fill = (200, 150, 100)
    buf  = _make_buffer(8, 8, fill_rgb=fill)
    save_screenshot(buf, 8, 8, tmp_png)
    img = np.array(Image.open(tmp_png))
    assert np.all(img == np.array(fill, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_wrong_buffer_length_raises(tmp_png):
    bad_buf = b"\x00" * 10   # not width*height*3
    with pytest.raises(ValueError, match="Buffer length"):
        save_screenshot(bad_buf, 4, 4, tmp_png)


def test_string_path_accepted(tmp_path):
    buf  = _make_buffer(4, 4)
    path = str(tmp_path / "out.png")
    result = save_screenshot(buf, 4, 4, path)
    assert Path(path).exists()
    assert isinstance(result, Path)
