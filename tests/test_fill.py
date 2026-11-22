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


def test_default_label_fn_ad_item_is_literal():
    item = {"kind": "ad", "label": "廣告", "confidence": 1.0, "frame_count": 10}
    label, low_conf, note = fill.default_label_fn(item)
    assert label == "廣告"
    assert low_conf is False
    assert note == ""


def test_default_label_fn_news_without_vlm_uses_fallback_and_flags_low_conf():
    item = {"kind": "news", "label": "台北市中山區發生車禍", "confidence": 0.9, "frame_count": 3}
    label, low_conf, note = fill.default_label_fn(item, use_vlm=False)
    assert len(label) == 5
    assert low_conf is True  # fallback_used 一律標低信心度
    assert "關鍵字擷取" in note


def test_default_label_fn_news_with_vlm_valid_result_not_flagged_for_fallback():
    def fake_vlm_fn(title, attempt, prev):
        return "台北車禍案"

    item = {"kind": "news", "label": "台北市中山區發生車禍", "confidence": 0.9, "frame_count": 3}
    label, low_conf, note = fill.default_label_fn(item, use_vlm=True, vlm_text_fn=fake_vlm_fn)
    assert label == "台北車禍案"
    assert low_conf is False
    assert "關鍵字擷取" not in note


def test_default_label_fn_low_confidence_flagged():
    def fake_vlm_fn(title, attempt, prev):
        return "台北車禍案"

    item = {"kind": "news", "label": "台北市中山區發生車禍", "confidence": 0.2, "frame_count": 1}
    label, low_conf, note = fill.default_label_fn(item, use_vlm=True, vlm_text_fn=fake_vlm_fn)
    assert low_conf is True
    assert "信心度 0.20" in note


def test_default_label_fn_note_includes_frame_count():
    def fake_vlm_fn(title, attempt, prev):
        return "台北車禍案"

    item = {"kind": "news", "label": "台北市中山區發生車禍", "confidence": 0.9, "frame_count": 7}
    _, _, note = fill.default_label_fn(item, use_vlm=True, vlm_text_fn=fake_vlm_fn)
    assert "7 筆讀值支持" in note


def test_build_cell_plan_single_item_per_minute_no_slash():
    minute_groups = {
        1200: [{"kind": "news", "label": "A", "confidence": 0.9, "frame_count": 10}],
    }

    def label_fn(item):
        return item["label"], False, ""

    plan = fill.build_cell_plan(minute_groups, label_fn)
    assert plan[5]["text"] == "A"


def test_build_cell_plan_two_items_joined_with_slash():
    minute_groups = {
        1200: [
            {"kind": "news", "label": "A", "confidence": 0.9, "frame_count": 10},
            {"kind": "news", "label": "B", "confidence": 0.9, "frame_count": 4},
        ],
    }

    def label_fn(item):
        return item["label"], False, ""

    plan = fill.build_cell_plan(minute_groups, label_fn)
    assert plan[5]["text"] == "A/B"
    assert plan[5]["low_conf"] is False


def test_build_cell_plan_any_low_conf_marks_whole_cell():
    minute_groups = {
        1200: [
            {"kind": "news", "label": "A", "confidence": 0.9, "frame_count": 10},
            {"kind": "news", "label": "B", "confidence": 0.9, "frame_count": 4},
        ],
    }

    def label_fn(item):
        return item["label"], item["label"] == "B", f"note-{item['label']}"

    plan = fill.build_cell_plan(minute_groups, label_fn)
    assert plan[5]["low_conf"] is True
    assert "note-B" in plan[5]["notes"]


def test_build_cell_plan_skips_out_of_range_minute():
    minute_groups = {1300: [{"kind": "news", "label": "A", "confidence": 0.9, "frame_count": 10}]}

    def label_fn(item):
        return item["label"], False, ""

    plan = fill.build_cell_plan(minute_groups, label_fn)
    assert plan == {}


def test_build_cell_plan_skips_minute_with_no_groups():
    minute_groups = {1200: []}

    def label_fn(item):
        return item["label"], False, ""

    plan = fill.build_cell_plan(minute_groups, label_fn)
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

    channels_minute_groups = {
        "ftv": {1200: [{"kind": "ad", "label": "廣告", "confidence": 1.0, "frame_count": 10}]},
        "cts": {1201: [{"kind": "news", "label": "某則新聞標題", "confidence": 0.9, "frame_count": 5}]},
    }
    stats = fill.fill_workbook(channels_minute_groups, template_path=template_path, output_path=output_path)

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

    channels_minute_groups = {
        "ftv": {1200: [{"kind": "news", "label": "某則新聞標題", "confidence": 0.1, "frame_count": 1}]},
    }
    fill.fill_workbook(channels_minute_groups, template_path=template_path, output_path=output_path)

    wb = openpyxl.load_workbook(output_path)
    ws = wb[fill.SHEET_NAME]
    cell = ws["N5"]
    assert cell.fill.start_color.rgb in ("00FFFFCC", "FFFFFFCC")
    assert cell.comment is not None


def test_main_passes_vlm_judge_fn_to_select_dominant_per_minute_when_flag_set(tmp_path, monkeypatch):
    vlm_path = tmp_path / "ftv_vlm.jsonl"
    vlm_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(fill, "OUT_DIR", tmp_path)

    captured = {}

    def fake_select(channel, path, vlm_judge_fn=None):
        captured["vlm_judge_fn"] = vlm_judge_fn
        return {}

    monkeypatch.setattr(fill.segment, "select_dominant_per_minute", fake_select)
    monkeypatch.setattr(fill, "fill_workbook", lambda *a, **k: {})
    monkeypatch.setattr(fill.vlm, "load_vlm_env", lambda: {"url": "http://fake", "user": "", "password": "", "model": "m"})

    monkeypatch.setattr(sys, "argv", ["fill.py", "--channels", "ftv", "--use-vlm-judge"])
    fill.main()

    assert captured["vlm_judge_fn"] is not None


def test_main_vlm_judge_fn_none_without_flag(tmp_path, monkeypatch):
    vlm_path = tmp_path / "ftv_vlm.jsonl"
    vlm_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(fill, "OUT_DIR", tmp_path)

    captured = {}

    def fake_select(channel, path, vlm_judge_fn=None):
        captured["vlm_judge_fn"] = vlm_judge_fn
        return {}

    monkeypatch.setattr(fill.segment, "select_dominant_per_minute", fake_select)
    monkeypatch.setattr(fill, "fill_workbook", lambda *a, **k: {})

    monkeypatch.setattr(sys, "argv", ["fill.py", "--channels", "ftv"])
    fill.main()

    assert captured["vlm_judge_fn"] is None
