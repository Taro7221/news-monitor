import sys
from pathlib import Path

import openpyxl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evaluate


def test_parse_minute_code_plain_int():
    assert evaluate.parse_minute_code("1230") == 1230


def test_parse_minute_code_colon_format():
    assert evaluate.parse_minute_code("12:30") == 1230


def test_parse_bool_variants():
    for v in ["TRUE", "1", "yes", "是", "v"]:
        assert evaluate.parse_bool(v) is True
    for v in ["FALSE", "0", "no", "否", ""]:
        assert evaluate.parse_bool(v) is False


def test_minute_code_to_row():
    assert evaluate.minute_code_to_row(1200) == 5
    assert evaluate.minute_code_to_row(1259) == 64
    assert evaluate.minute_code_to_row(1300) is None


def _write_gt_csv(path: Path, rows: list[dict]):
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["台別", "分鐘", "事件", "是否廣告"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_load_ground_truth_reads_rows(tmp_path):
    path = tmp_path / "gt.csv"
    _write_gt_csv(path, [
        {"台別": "ftv", "分鐘": "1230", "事件": "甲則新聞", "是否廣告": "FALSE"},
        {"台別": "ftv", "分鐘": "1231", "事件": "", "是否廣告": "FALSE"},
        {"台別": "ftv", "分鐘": "1232", "事件": "廣告", "是否廣告": "TRUE"},
    ])
    gt = evaluate.load_ground_truth(path)
    assert gt[("ftv", 1230)]["event"] == "甲則新聞"
    assert gt[("ftv", 1230)]["is_ad"] is False
    assert gt[("ftv", 1231)]["event"] == ""
    assert gt[("ftv", 1232)]["is_ad"] is True


def _make_excel(path: Path, cells: dict[str, str]):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = evaluate.SHEET_NAME
    for coord, value in cells.items():
        ws[coord] = value
    wb.save(path)


def test_load_predicted_reads_correct_columns_and_rows(tmp_path):
    excel_path = tmp_path / "out.xlsx"
    _make_excel(excel_path, {"N5": "甲則新聞", "R6": "廣告"})
    predicted = evaluate.load_predicted(excel_path, ["ftv", "cts"], [1200, 1201])
    assert predicted[("ftv", 1200)] == "甲則新聞"
    assert predicted[("cts", 1201)] == "廣告"
    assert predicted[("ftv", 1201)] == ""


def test_compute_metrics_perfect_match():
    gt = {
        ("ftv", 1200): {"event": "甲", "is_ad": False},
        ("ftv", 1201): {"event": "", "is_ad": False},
        ("ftv", 1202): {"event": "廣告", "is_ad": True},
    }
    predicted = {("ftv", 1200): "甲", ("ftv", 1201): "", ("ftv", 1202): "廣告"}
    metrics = evaluate.compute_metrics(gt, predicted)
    assert metrics["change_accuracy"] == 1.0
    assert metrics["ad_precision"] == 1.0
    assert metrics["ad_recall"] == 1.0
    assert metrics["miss_fill_rate"] == 0.0


def test_compute_metrics_missed_change_counts_as_miss_fill():
    gt = {
        ("ftv", 1200): {"event": "甲", "is_ad": False},
        ("ftv", 1201): {"event": "乙", "is_ad": False},
    }
    predicted = {("ftv", 1200): "甲", ("ftv", 1201): ""}  # 1201 漏填
    metrics = evaluate.compute_metrics(gt, predicted)
    assert metrics["miss_fill_count"] == 1
    assert metrics["miss_fill_rate"] == 0.5
    assert metrics["change_accuracy"] == 0.5


def test_compute_metrics_ad_false_positive():
    gt = {
        ("ftv", 1200): {"event": "甲", "is_ad": False},
        ("ftv", 1201): {"event": "廣告", "is_ad": True},
    }
    predicted = {("ftv", 1200): "廣告", ("ftv", 1201): "廣告"}  # 1200 誤判成廣告
    metrics = evaluate.compute_metrics(gt, predicted)
    assert metrics["ad_tp"] == 1
    assert metrics["ad_fp"] == 1
    assert metrics["ad_fn"] == 0
    assert metrics["ad_precision"] == 0.5
    assert metrics["ad_recall"] == 1.0


def test_compute_metrics_ad_false_negative():
    gt = {("ftv", 1200): {"event": "廣告", "is_ad": True}}
    predicted = {("ftv", 1200): "某則新聞"}  # 漏判廣告
    metrics = evaluate.compute_metrics(gt, predicted)
    assert metrics["ad_fn"] == 1
    assert metrics["ad_precision"] is None  # 沒有任何預測為廣告
    assert metrics["ad_recall"] == 0.0


def test_render_markdown_contains_key_sections():
    gt = {("ftv", 1200): {"event": "甲", "is_ad": False}}
    predicted = {("ftv", 1200): "甲"}
    metrics = evaluate.compute_metrics(gt, predicted)
    report = evaluate.render_markdown(metrics)
    assert "換題分鐘準確率" in report
    assert "廣告分鐘 precision" in report
    assert "漏填率" in report


def test_render_markdown_lists_mismatches():
    gt = {("ftv", 1200): {"event": "甲", "is_ad": False}}
    predicted = {("ftv", 1200): ""}  # 不一致
    metrics = evaluate.compute_metrics(gt, predicted)
    report = evaluate.render_markdown(metrics)
    assert "ftv" in report
    assert "全部一致" not in report
