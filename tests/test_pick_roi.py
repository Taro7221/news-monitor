import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import pick_roi


def test_pixel_to_ratio():
    rx, ry = pick_roi.pixel_to_ratio(640, 360, width=1280, height=720)
    assert (rx, ry) == (0.5, 0.5)


def test_normalize_rect_sorts_either_order():
    assert pick_roi.normalize_rect((100, 200), (10, 20)) == (10, 20, 100, 200)
    assert pick_roi.normalize_rect((10, 20), (100, 200)) == (10, 20, 100, 200)


def test_normalize_rect_mixed_order():
    assert pick_roi.normalize_rect((100, 20), (10, 200)) == (10, 20, 100, 200)


def test_format_ratio_bbox_contains_pixel_and_ratio_lines():
    text = pick_roi.format_ratio_bbox(0, 576, 1280, 648, width=1280, height=720)
    assert "x1=0 y1=576 x2=1280 y2=648" in text
    assert "x1: 0.0000" in text
    assert "y1: 0.8000" in text
    assert "y2: 0.9000" in text
