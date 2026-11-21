"""evaluate.py：讀手動標記的 data/gt.csv，跟 fill.py 輸出的 Excel 比對，計算：
- 換題分鐘準確率：每一分鐘「有沒有換則」的判斷跟人工標記是否一致
- 廣告分鐘的 precision / recall
- 漏填率：人工標記有換則、但我們的輸出留空的比例

輸出 markdown 表格（預設印到 stdout，--output 可存檔）。

data/gt.csv 欄位（每列是一個頻道、一分鐘的人工標記；分鐘可以寫 1230 或 12:30）：
    台別,分鐘,事件,是否廣告
    ftv,1230,甲則新聞標題,FALSE
    ftv,1231,,FALSE             <- 事件欄空白代表「同則延續」，不是換則
    ftv,1232,廣告,TRUE
    ...

使用方式：
    python src/evaluate.py --gt data/gt.csv --excel out/12監看_20261106_auto.xlsx
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import openpyxl

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GT_PATH = ROOT / "data" / "gt.csv"
DEFAULT_EXCEL_PATH = ROOT / "out" / "12監看_20261106_auto.xlsx"
SHEET_NAME = "12監看"

CHANNEL_COLUMNS = {"tvbs": "F", "ebc": "J", "ftv": "N", "cts": "R"}
FIRST_MINUTE_CODE = 1200
FIRST_ROW = 5
LAST_ROW = 64


def minute_code_to_row(minute_code: int) -> Optional[int]:
    row = FIRST_ROW + (minute_code - FIRST_MINUTE_CODE)
    if row < FIRST_ROW or row > LAST_ROW:
        return None
    return row


def parse_minute_code(value: str) -> int:
    value = str(value).strip()
    if ":" in value:
        hh, mm = value.split(":")
        return int(hh) * 100 + int(mm)
    return int(value)


def parse_bool(value: str) -> bool:
    return str(value).strip().upper() in ("TRUE", "1", "YES", "是", "Y", "V")


def load_ground_truth(path: Path) -> dict[tuple[str, int], dict]:
    gt: dict[tuple[str, int], dict] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            channel = (row.get("台別") or "").strip()
            if not channel:
                continue
            minute = parse_minute_code(row["分鐘"])
            event = (row.get("事件") or "").strip()
            is_ad = parse_bool(row.get("是否廣告", ""))
            gt[(channel, minute)] = {"event": event, "is_ad": is_ad}
    return gt


def load_predicted(excel_path: Path, channels: list[str], minutes: list[int]) -> dict[tuple[str, int], str]:
    wb = openpyxl.load_workbook(str(excel_path), data_only=True)
    ws = wb[SHEET_NAME]
    predicted: dict[tuple[str, int], str] = {}
    for channel in channels:
        column = CHANNEL_COLUMNS.get(channel)
        if not column:
            continue
        for minute in minutes:
            row = minute_code_to_row(minute)
            value = ws[f"{column}{row}"].value if row else None
            predicted[(channel, minute)] = str(value).strip() if value is not None else ""
    return predicted


def compute_metrics(gt: dict[tuple[str, int], dict], predicted: dict[tuple[str, int], str]) -> dict:
    keys = sorted(gt.keys())
    total = len(keys)
    change_matches = 0
    ad_tp = ad_fp = ad_fn = 0
    miss_fill = 0
    gt_change_count = 0
    rows = []

    for channel, minute in keys:
        gt_entry = gt[(channel, minute)]
        pred_text = predicted.get((channel, minute), "")
        gt_is_change = bool(gt_entry["event"])
        pred_is_change = bool(pred_text)
        gt_is_ad = gt_entry["is_ad"]
        pred_is_ad = pred_text == "廣告"

        change_match = gt_is_change == pred_is_change
        if change_match:
            change_matches += 1
        if gt_is_ad and pred_is_ad:
            ad_tp += 1
        if pred_is_ad and not gt_is_ad:
            ad_fp += 1
        if gt_is_ad and not pred_is_ad:
            ad_fn += 1
        if gt_is_change:
            gt_change_count += 1
            if not pred_is_change:
                miss_fill += 1

        rows.append({
            "channel": channel, "minute": minute,
            "gt_event": gt_entry["event"], "pred_text": pred_text,
            "gt_is_ad": gt_is_ad, "pred_is_ad": pred_is_ad,
            "change_match": change_match,
        })

    return {
        "total_minutes": total,
        "change_accuracy": (change_matches / total) if total else None,
        "ad_precision": (ad_tp / (ad_tp + ad_fp)) if (ad_tp + ad_fp) else None,
        "ad_recall": (ad_tp / (ad_tp + ad_fn)) if (ad_tp + ad_fn) else None,
        "ad_tp": ad_tp, "ad_fp": ad_fp, "ad_fn": ad_fn,
        "miss_fill_rate": (miss_fill / gt_change_count) if gt_change_count else None,
        "gt_change_count": gt_change_count,
        "miss_fill_count": miss_fill,
        "rows": rows,
    }


def format_pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def render_markdown(metrics: dict) -> str:
    lines = [
        "# 自動填表準確度評估",
        "",
        f"評估分鐘數：{metrics['total_minutes']}",
        "",
        "| 指標 | 數值 |",
        "|---|---|",
        f"| 換題分鐘準確率 | {format_pct(metrics['change_accuracy'])} |",
        f"| 廣告分鐘 precision | {format_pct(metrics['ad_precision'])} |",
        f"| 廣告分鐘 recall | {format_pct(metrics['ad_recall'])} |",
        f"| 漏填率 | {format_pct(metrics['miss_fill_rate'])}"
        f"（{metrics['miss_fill_count']}/{metrics['gt_change_count']}） |",
        "",
        "## 廣告判斷細節",
        "",
        f"- TP（正確判定廣告）：{metrics['ad_tp']}",
        f"- FP（誤判為廣告）：{metrics['ad_fp']}",
        f"- FN（漏判廣告）：{metrics['ad_fn']}",
        "",
        "## 逐分鐘明細（換則判斷不一致的列）",
        "",
        "| 台別 | 分鐘 | GT事件 | 預測文字 |",
        "|---|---|---|---|",
    ]
    mismatches = [r for r in metrics["rows"] if not r["change_match"]]
    if mismatches:
        for r in mismatches:
            lines.append(
                f"| {r['channel']} | {r['minute']} | {r['gt_event'] or '(空)'} | {r['pred_text'] or '(空)'} |"
            )
    else:
        lines.append("| - | - | - | 全部一致 |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="評估自動填表結果：讀 data/gt.csv 跟輸出的 Excel 比對")
    parser.add_argument("--gt", type=str, default=str(DEFAULT_GT_PATH))
    parser.add_argument("--excel", type=str, default=str(DEFAULT_EXCEL_PATH))
    parser.add_argument("--output", type=str, default=None, help="markdown 輸出檔路徑，不給就印到 stdout")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    excel_path = Path(args.excel)
    if not gt_path.exists():
        print(f"[skip] 找不到 {gt_path}，請先手動標記")
        return 1
    if not excel_path.exists():
        print(f"[skip] 找不到 {excel_path}，請先跑 fill.py")
        return 1

    gt = load_ground_truth(gt_path)
    if not gt:
        print(f"[skip] {gt_path} 沒有任何資料列")
        return 1

    channels = sorted({ch for ch, _ in gt.keys()})
    minutes = sorted({m for _, m in gt.keys()})
    predicted = load_predicted(excel_path, channels, minutes)
    metrics = compute_metrics(gt, predicted)

    report = render_markdown(metrics)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"[ok] 已輸出 {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
