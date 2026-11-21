import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import detect


def abs_diff(a, b):
    return abs(a - b)


def test_detect_changes_no_change_gives_no_events():
    series = [(t, 0) for t in range(10)]
    events = detect.detect_changes(series, threshold=5, stable_seconds=2, diff=abs_diff)
    assert events == []


def test_detect_changes_single_confirmed_change():
    # changes to 100 at t=3, stays stable afterwards
    series = [(0, 0), (1, 0), (2, 0), (3, 100), (4, 100), (5, 100), (6, 100)]
    events = detect.detect_changes(series, threshold=5, stable_seconds=2, diff=abs_diff)
    assert events == [(3, 100)]


def test_detect_changes_transient_spike_reverts_before_confirm():
    # brief spike at t=3 then reverts to base at t=4 -- should be ignored as noise
    series = [(0, 0), (1, 0), (2, 0), (3, 100), (4, 0), (5, 0), (6, 0)]
    events = detect.detect_changes(series, threshold=5, stable_seconds=2, diff=abs_diff)
    assert events == []


def test_detect_changes_requires_full_stable_window():
    # changes at t=3, but flickers to a third value at t=4 before stabilizing -- not confirmed yet,
    # then settles on the t=4 value for 2 more seconds -> confirmed at t=4
    series = [(0, 0), (1, 0), (2, 0), (3, 100), (4, 200), (5, 200), (6, 200)]
    events = detect.detect_changes(series, threshold=5, stable_seconds=2, diff=abs_diff)
    assert events == [(4, 200)]


def test_detect_changes_never_stabilizes_gives_no_events():
    # keeps flickering to different values every second, never stable for 2s
    series = [(0, 0), (1, 100), (2, 200), (3, 300), (4, 400)]
    events = detect.detect_changes(series, threshold=5, stable_seconds=2, diff=abs_diff)
    assert events == []


def test_detect_changes_multiple_confirmed_events():
    series = [
        (0, 0), (1, 0),
        (2, 100), (3, 100), (4, 100),
        (5, 200), (6, 200), (7, 200),
    ]
    events = detect.detect_changes(series, threshold=5, stable_seconds=2, diff=abs_diff)
    assert events == [(2, 100), (5, 200)]


def test_detect_changes_diff_exactly_at_threshold_is_not_a_change():
    series = [(0, 0), (1, 5), (2, 5), (3, 5)]
    events = detect.detect_changes(series, threshold=5, stable_seconds=2, diff=abs_diff)
    assert events == []


def test_detect_changes_empty_series():
    assert detect.detect_changes([], threshold=5, stable_seconds=2, diff=abs_diff) == []


def test_crop_roi_basic():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    roi = {"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 1.0}
    crop = detect.crop_roi(frame, roi)
    assert crop.shape[0] == 50
    assert crop.shape[1] == 200


def test_crop_roi_upsamples_tiny_region():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    roi = {"x1": 0.0, "y1": 0.0, "x2": 0.02, "y2": 0.02}
    crop = detect.crop_roi(frame, roi)
    assert min(crop.shape[:2]) >= detect.MIN_CROP_SIDE


def test_compute_phash_identical_images_zero_distance():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    h1 = detect.compute_phash(img)
    h2 = detect.compute_phash(img.copy())
    assert (h1 - h2) == 0


def test_compute_phash_different_images_nonzero_distance():
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    h1 = detect.compute_phash(black)
    h2 = detect.compute_phash(white)
    assert (h1 - h2) > 0


def test_load_broadcast_start_parses_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("BROADCAST_START=2026-11-06 12:00:00\n", encoding="utf-8")
    monkeypatch.setattr(detect, "ROOT", tmp_path)
    result = detect.load_broadcast_start()
    assert result == datetime(2026, 11, 6, 12, 0, 0)


def test_load_broadcast_start_falls_back_on_bad_value(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("BROADCAST_START=not-a-date\n", encoding="utf-8")
    monkeypatch.setattr(detect, "ROOT", tmp_path)
    result = detect.load_broadcast_start()
    assert result == datetime(2026, 11, 6, 12, 0, 0)
