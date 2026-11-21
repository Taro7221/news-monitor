import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import calibrate


def test_sample_frame_indices_count_and_bounds():
    indices = calibrate.sample_frame_indices(total_frames=1800, n=10)
    assert len(indices) == 10
    assert indices == sorted(indices)
    assert indices[0] >= 0
    assert indices[-1] <= 1799


def test_sample_frame_indices_short_video():
    indices = calibrate.sample_frame_indices(total_frames=3, n=10)
    assert all(0 <= i < 3 for i in indices)


def test_sample_frame_indices_empty_video():
    assert calibrate.sample_frame_indices(total_frames=0, n=10) == []


def test_ratio_to_pixels_basic():
    roi = {"x1": 0.0, "y1": 0.8, "x2": 1.0, "y2": 0.9}
    x1, y1, x2, y2 = calibrate.ratio_to_pixels(roi, width=1280, height=720)
    assert (x1, y1, x2, y2) == (0, 576, 1280, 648)


def test_ratio_to_pixels_clamps_out_of_range():
    roi = {"x1": -0.5, "y1": -0.5, "x2": 1.5, "y2": 1.5}
    x1, y1, x2, y2 = calibrate.ratio_to_pixels(roi, width=100, height=50)
    assert (x1, y1, x2, y2) == (0, 0, 100, 50)


def test_build_default_config_has_all_channels_and_rois():
    config = calibrate.build_default_config(["ftv", "tvbs"])
    assert set(config.keys()) == {"ftv", "tvbs"}
    for ch_config in config.values():
        assert set(ch_config.keys()) == set(calibrate.ROI_NAMES)


def test_draw_rois_does_not_mutate_input_and_changes_pixels():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    config = {"headline_bar": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.3}}
    result = calibrate.draw_rois(image, config)
    assert np.array_equal(image, np.zeros((100, 200, 3), dtype=np.uint8))
    assert not np.array_equal(result, image)


def test_render_default_yaml_is_parseable():
    import yaml

    text = calibrate.render_default_yaml(["ftv"])
    parsed = yaml.safe_load(text)
    assert "ftv" in parsed
    assert set(parsed["ftv"].keys()) == set(calibrate.ROI_NAMES)
    assert parsed["ftv"]["headline_bar"]["y2"] == 0.90
