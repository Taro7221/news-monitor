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
        {"kind": "news", "label": "A", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1200},
        {"kind": "news", "label": "B", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1200},
    ]

    def label_fn(ev):
        return ev["label"], False, ""

    plan = fill.build_cell_plan(events, label_fn)
    assert plan[5]["text"] == "A/B"
    assert plan[5]["low_conf"] is False


def test_build_cell_plan_any_low_conf_marks_whole_cell():
    events = [
        {"kind": "news", "label": "A", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1200},
        {"kind": "news", "label": "B", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1200},
    ]

    def label_fn(ev):
        return ev["label"], ev["label"] == "B", f"note-{ev['label']}"

    plan = fill.build_cell_plan(events, label_fn)
    assert plan[5]["low_conf"] is True
    assert "note-B" in plan[5]["notes"]


def test_build_cell_plan_skips_out_of_range_minute():
    events = [{"kind": "news", "label": "A", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1300}]

    def label_fn(ev):
        return ev["label"], False, ""

    plan = fill.build_cell_plan(events, label_fn)
    assert plan == {}


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
        "ftv": [{"kind": "ad", "label": "廣告", "confidence": 1.0, "ambiguous_next": None, "minute_code": 1200}],
        "cts": [{"kind": "news", "label": "某則新聞標題", "confidence": 0.9, "ambiguous_next": None, "minute_code": 1201}],
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
        "ftv": [{"kind": "news", "label": "某則新聞標題", "confidence": 0.1, "ambiguous_next": None, "minute_code": 1200}],
    }
    fill.fill_workbook(channels_fill_events, template_path=template_path, output_path=output_path)

    wb = openpyxl.load_workbook(output_path)
    ws = wb[fill.SHEET_NAME]
    cell = ws["N5"]
    assert cell.fill.start_color.rgb in ("00FFFFCC", "FFFFFFCC")
    assert cell.comment is not None
