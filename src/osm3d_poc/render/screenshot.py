"""Screenshot helper: convert raw OpenGL framebuffer bytes to a PNG file (PRD §Phase 8).

OpenGL's origin is at the bottom-left, so the raw read is vertically flipped
relative to image conventions. This module handles the flip transparently.

Usage inside Renderer:
    from .screenshot import save_screenshot
    data = ctx.screen.read(viewport=(0, 0, w, h), components=3)
    path = save_screenshot(data, w, h, "out.png")
"""
from pathlib import Path
from typing import Union

import numpy as np


def save_screenshot(
    framebuffer_bytes: bytes,
    width: int,
    height: int,
    output_path: Union[str, Path],
) -> Path:
    """Save raw RGB framebuffer bytes as a PNG file.

    Args:
        framebuffer_bytes: Raw RGB bytes from ctx.screen.read(components=3).
        width:             Viewport width in pixels.
        height:            Viewport height in pixels.
        output_path:       Destination path (extension determines format).

    Returns:
        Resolved Path of the saved file.

    Raises:
        ImportError: if Pillow is not installed.
        ValueError:  if buffer length doesn't match width × height × 3.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for screenshots: pip install Pillow"
        ) from exc

    expected = width * height * 3
    if len(framebuffer_bytes) != expected:
        raise ValueError(
            f"Buffer length {len(framebuffer_bytes)} != {expected} "
            f"(expected {width}×{height}×3)"
        )

    arr = np.frombuffer(framebuffer_bytes, dtype=np.uint8).reshape(height, width, 3)
    img = Image.fromarray(arr[::-1])   # flip Y: OpenGL origin is bottom-left
    output_path = Path(output_path).resolve()
    img.save(output_path)
    return output_path
