import sys
from pathlib import Path

import openpyxl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fill


def test_minute_code_to_row_basic():
    assert fill.minute_code_to_row(1200) == 5
    assert fill.minute_code_to_row(1259) == 64
    assert fill.minute_code_to_row(1230) == 35


def test_minute_code_to_row_out_of_range_is_none():
    assert fill.minute_code_to_row(1300) is None
    assert fill.minute_code_to_row(1100) is None
    assert fill.minute_code_to_row(None) is None


def test_default_label_fn_ad_event_is_literal():
    event = {"kind": "ad", "label": "廣告", "confidence": 1.0, "ambiguous_next": None}
    label, low_conf, note = fill.default_label_fn(event)
    assert label == "廣告"
    assert low_conf is False
    assert note == ""


def test_default_label_fn_news_without_vlm_uses_fallback_and_flags_low_conf():
    event = {"kind": "news", "label": "台北市中山區發生車禍", "confidence": 0.9, "ambiguous_next": None}
    label, low_conf, note = fill.default_label_fn(event, use_vlm=False)
    assert len(label) == 5
    assert low_conf is True  # fallback_used 一律標低信心度
    assert "關鍵字擷取" in note


def test_default_label_fn_news_with_vlm_valid_result_not_flagged_for_fallback():
    def fake_vlm_fn(title, attempt, prev):
        return "台北車禍案"

    event = {"kind": "news", "label": "台北市中山區發生車禍", "confidence": 0.9, "ambiguous_next": None}
    label, low_conf, note = fill.default_label_fn(event, use_vlm=True, vlm_text_fn=fake_vlm_fn)
    assert label == "台北車禍案"
    assert low_conf is False
    assert "關鍵字擷取" not in note


def test_default_label_fn_low_confidence_flagged():
    def fake_vlm_fn(title, attempt, prev):
        return "台北車禍案"

    event = {"kind": "news", "label": "台北市中山區發生車禍", "confidence": 0.2, "ambiguous_next": None}
    label, low_conf, note = fill.default_label_fn(event, use_vlm=True, vlm_text_fn=fake_vlm_fn)
    assert low_conf is True
    assert "信心度 0.20" in note


def test_default_label_fn_ambiguous_next_flagged():
    def fake_vlm_fn(title, attempt, prev):
        return "台北車禍案"

    event = {
        "kind": "news", "label": "台北市中山區發生車禍", "confidence": 0.9,
        "ambiguous_next": {"other_title": "x", "other_second": 10, "score": 70},
    }
    label, low_conf, note = fill.default_label_fn(event, use_vlm=True, vlm_text_fn=fake_vlm_fn)
    assert low_conf is True
    assert "換則判斷不確定" in note


def test_build_cell_plan_joins_same_minute_with_slash():
    events = [
        {"kind": "news", "label": "A", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1200, "video_second": 0},
        {"kind": "news", "label": "B", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1200, "video_second": 1},
    ]

    def label_fn(ev):
        return ev["label"], False, ""

    plan = fill.build_cell_plan(events, label_fn)
    assert plan[5]["text"] == "A/B"
    assert plan[5]["low_conf"] is False


def test_build_cell_plan_any_low_conf_marks_whole_cell():
    events = [
        {"kind": "news", "label": "A", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1200, "video_second": 0},
        {"kind": "news", "label": "B", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1200, "video_second": 1},
    ]

    def label_fn(ev):
        return ev["label"], ev["label"] == "B", f"note-{ev['label']}"

    plan = fill.build_cell_plan(events, label_fn)
    assert plan[5]["low_conf"] is True
    assert "note-B" in plan[5]["notes"]


def test_build_cell_plan_skips_out_of_range_minute():
    events = [{"kind": "news", "label": "A", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1300, "video_second": 0}]

    def label_fn(ev):
        return ev["label"], False, ""

    plan = fill.build_cell_plan(events, label_fn)
    assert plan == {}


def test_consolidate_same_minute_events_merges_near_identical_titles():
    events = [
        {"kind": "news", "label": "汽油彈襲警車罪加一等", "confidence": 0.9, "video_second": 300},
        {"kind": "news", "label": "汽油彈襲警車罪一等", "confidence": 0.5, "video_second": 320},
    ]
    result = fill.consolidate_same_minute_events(events)
    assert len(result) == 1
    assert result[0]["label"] == "汽油彈襲警車罪加一等"  # 信心度較高的當代表


def test_consolidate_same_minute_events_keeps_genuinely_different_apart():
    events = [
        {"kind": "news", "label": "持刀警開槍", "confidence": 0.9, "video_second": 300},
        {"kind": "news", "label": "台股二補跌", "confidence": 0.9, "video_second": 320},
    ]
    result = fill.consolidate_same_minute_events(events)
    assert len(result) == 2


def test_consolidate_same_minute_events_ambiguous_uses_vlm_judge():
    def fake_similarity(a, b):
        return 70.0  # 介於門檻之間

    events = [
        {"kind": "news", "label": "標題A", "confidence": 0.9, "video_second": 300},
        {"kind": "news", "label": "標題B", "confidence": 0.9, "video_second": 320},
    ]
    merged = fill.consolidate_same_minute_events(events, vlm_judge_fn=lambda a, b: True, similarity_fn=fake_similarity)
    assert len(merged) == 1
    split = fill.consolidate_same_minute_events(events, vlm_judge_fn=lambda a, b: False, similarity_fn=fake_similarity)
    assert len(split) == 2


def test_consolidate_same_minute_events_preserves_chronological_order():
    events = [
        {"kind": "news", "label": "乙新聞", "confidence": 0.9, "video_second": 320},
        {"kind": "news", "label": "甲新聞", "confidence": 0.9, "video_second": 300},
    ]
    result = fill.consolidate_same_minute_events(events)
    assert [e["label"] for e in result] == ["甲新聞", "乙新聞"]


def test_build_cell_plan_passes_vlm_judge_fn_to_consolidation(monkeypatch):
    events = [
        {"kind": "news", "label": "A", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1200, "video_second": 0},
        {"kind": "news", "label": "B", "confidence": 0.5, "ambiguous_next": None, "minute_code": 1200, "video_second": 10},
    ]
    captured = {}

    def fake_consolidate(news_events, vlm_judge_fn=None):
        captured["vlm_judge_fn"] = vlm_judge_fn
        return news_events[:1]  # 假裝合併成一個，驗證 build_cell_plan 真的會用這個結果

    monkeypatch.setattr(fill, "consolidate_same_minute_events", fake_consolidate)

    def label_fn(ev):
        return ev["label"], False, ""

    sentinel = lambda a, b: True
    plan = fill.build_cell_plan(events, label_fn, vlm_judge_fn=sentinel)

    assert captured["vlm_judge_fn"] is sentinel
    assert plan[5]["text"] == "A"


def _make_minimal_template(path: Path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = fill.SHEET_NAME
    ws["A1"] = "監看日期： "
    ws["A4"] = "時間"
    for i, minute in enumerate(range(1200, 1260)):
        ws[f"A{5+i}"] = minute
    ws["Z1"] = "不該被動到"
    wb.save(path)


def test_fill_workbook_only_touches_target_columns(tmp_path):
    template_path = tmp_path / "template.xlsx"
    _make_minimal_template(template_path)
    output_path = tmp_path / "out.xlsx"

    channels_fill_events = {
        "ftv": [{"kind": "ad", "label": "廣告", "confidence": 1.0, "ambiguous_next": None, "minute_code": 1200, "video_second": 0}],
        "cts": [{"kind": "news", "label": "某則新聞標題", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1201, "video_second": 60}],
    }
    stats = fill.fill_workbook(channels_fill_events, template_path=template_path, output_path=output_path)

    wb = openpyxl.load_workbook(output_path)
    ws = wb[fill.SHEET_NAME]
    assert ws["N5"].value == "廣告"  # ftv -> N
    assert len(ws["R6"].value) == 5  # cts -> R, minute 1201 -> row 6
    assert ws["Z1"].value == "不該被動到"
    assert ws["A4"].value == "時間"
    assert stats["ftv"]["filled_rows"] == 1
    assert stats["cts"]["filled_rows"] == 1


def test_fill_workbook_marks_low_confidence_with_fill_and_comment(tmp_path):
    template_path = tmp_path / "template.xlsx"
    _make_minimal_template(template_path)
    output_path = tmp_path / "out.xlsx"

    channels_fill_events = {
        "ftv": [{"kind": "news", "label": "某則新聞標題", "confidence": 0.1, "ambiguous_next": None, "minute_code": 1200, "video_second": 0}],
    }
    fill.fill_workbook(channels_fill_events, template_path=template_path, output_path=output_path)

    wb = openpyxl.load_workbook(output_path)
    ws = wb[fill.SHEET_NAME]
    cell = ws["N5"]
    assert cell.fill.start_color.rgb in ("00FFFFCC", "FFFFFFCC")
    assert cell.comment is not None


def test_main_passes_vlm_judge_fn_when_flag_set(tmp_path, monkeypatch):
    vlm_path = tmp_path / "ftv_vlm.jsonl"
    vlm_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(fill, "OUT_DIR", tmp_path)

    captured = {}

    def fake_build_fill_events(channel, path, vlm_judge_fn=None):
        captured["vlm_judge_fn"] = vlm_judge_fn
        return []

    monkeypatch.setattr(fill.segment, "build_fill_events", fake_build_fill_events)
    monkeypatch.setattr(fill, "fill_workbook", lambda *a, **k: {})
    monkeypatch.setattr(fill.vlm, "load_vlm_env", lambda: {"url": "http://fake", "user": "", "password": "", "model": "m"})

    import sys
    monkeypatch.setattr(sys, "argv", ["fill.py", "--channels", "ftv", "--use-vlm-judge"])
    fill.main()

    assert captured["vlm_judge_fn"] is not None


def test_main_vlm_judge_fn_none_without_flag(tmp_path, monkeypatch):
    vlm_path = tmp_path / "ftv_vlm.jsonl"
    vlm_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(fill, "OUT_DIR", tmp_path)

    captured = {}

    def fake_build_fill_events(channel, path, vlm_judge_fn=None):
        captured["vlm_judge_fn"] = vlm_judge_fn
        return []

    monkeypatch.setattr(fill.segment, "build_fill_events", fake_build_fill_events)
    monkeypatch.setattr(fill, "fill_workbook", lambda *a, **k: {})

    import sys
    monkeypatch.setattr(sys, "argv", ["fill.py", "--channels", "ftv"])
    fill.main()

    assert captured["vlm_judge_fn"] is None
