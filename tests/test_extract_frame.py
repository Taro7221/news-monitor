import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import extract_frame


def test_parse_timestamp_hh_mm_ss():
    assert extract_frame.parse_timestamp("00:33:42") == 33 * 60 + 42


def test_parse_timestamp_mm_ss():
    assert extract_frame.parse_timestamp("36:42") == 36 * 60 + 42


def test_parse_timestamp_with_fraction():
    assert extract_frame.parse_timestamp("00:00:01.5") == pytest.approx(1.5)


def test_parse_timestamp_invalid_format_raises():
    with pytest.raises(ValueError):
        extract_frame.parse_timestamp("42")


def test_extract_frame_missing_video_returns_none(tmp_path):
    frame, seconds = extract_frame.extract_frame(tmp_path / "nope.mp4", 10.0)
    assert frame is None
    assert seconds is None
