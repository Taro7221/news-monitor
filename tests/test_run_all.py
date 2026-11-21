import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_all


def test_main_respects_skip_flags(monkeypatch):
    calls = []

    monkeypatch.setattr(run_all, "run_detect", lambda *a, **k: calls.append("detect"))
    monkeypatch.setattr(run_all, "run_vlm", lambda *a, **k: calls.append("vlm"))
    monkeypatch.setattr(run_all, "run_fill", lambda *a, **k: calls.append("fill") or {})
    monkeypatch.setattr(run_all, "run_evaluate", lambda *a, **k: calls.append("evaluate"))

    monkeypatch.setattr(sys, "argv", ["run_all.py", "--skip-detect", "--skip-evaluate"])
    run_all.main()

    assert calls == ["vlm", "fill"]


def test_main_runs_all_steps_by_default(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(run_all, "run_detect", lambda *a, **k: calls.append("detect"))
    monkeypatch.setattr(run_all, "run_vlm", lambda *a, **k: calls.append("vlm"))
    monkeypatch.setattr(run_all, "run_fill", lambda *a, **k: calls.append("fill") or {})
    monkeypatch.setattr(run_all, "run_evaluate", lambda *a, **k: calls.append("evaluate"))

    monkeypatch.setattr(sys, "argv", ["run_all.py", "--gt", str(tmp_path / "nope.csv")])
    run_all.main()

    assert calls == ["detect", "vlm", "fill", "evaluate"]


def test_run_evaluate_skips_when_gt_missing(tmp_path, capsys):
    result = run_all.run_evaluate(str(tmp_path / "nope.csv"), str(tmp_path / "nope.xlsx"), None)
    assert result is None
    assert "略過評估" in capsys.readouterr().out


def test_main_passes_tag_and_window_through(monkeypatch):
    captured = {}

    def fake_run_detect(channels, tag, start_second, duration_seconds):
        captured["detect"] = (channels, tag, start_second, duration_seconds)

    monkeypatch.setattr(run_all, "run_detect", fake_run_detect)
    monkeypatch.setattr(run_all, "run_vlm", lambda *a, **k: None)
    monkeypatch.setattr(run_all, "run_fill", lambda *a, **k: {})
    monkeypatch.setattr(run_all, "run_evaluate", lambda *a, **k: None)

    monkeypatch.setattr(sys, "argv", [
        "run_all.py", "--tag", "_test15", "--start-second", "1800", "--duration-seconds", "900",
        "--channels", "ftv", "cts",
    ])
    run_all.main()

    assert captured["detect"] == (["ftv", "cts"], "_test15", 1800, 900)


def test_main_passes_use_vlm_judge_through(monkeypatch):
    captured = {}

    monkeypatch.setattr(run_all, "run_detect", lambda *a, **k: None)
    monkeypatch.setattr(run_all, "run_vlm", lambda *a, **k: None)

    def fake_run_fill(channels, tag, output_path, use_vlm_labels, use_vlm_judge=False):
        captured["fill"] = (use_vlm_labels, use_vlm_judge)
        return {}

    monkeypatch.setattr(run_all, "run_fill", fake_run_fill)
    monkeypatch.setattr(run_all, "run_evaluate", lambda *a, **k: None)

    monkeypatch.setattr(sys, "argv", ["run_all.py", "--use-vlm-labels", "--use-vlm-judge"])
    run_all.main()

    assert captured["fill"] == (True, True)
