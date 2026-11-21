"""fill.py：依 docs/template_spec.md 把 segment.py 的填表事件寫進 Excel 範本，
輸出 out/12監看_20261106_auto.xlsx。

規則（對照 docs/template_spec.md）：
- 複製 data/excel/template.xlsx，只改四台「五字標」欄位，其他儲存格/格式不動。
- 欄位對照：tvbs->F、ebc->J、ftv->N、cts->R；列 = 5 + (minute_code-1200)。
- 同一分鐘有兩則不同內容時用「前則/後則」相連。
- 低信心度（confidence 偏低、換則判斷不確定待覆核、或五字標是用關鍵字擷取
  退路而非 VLM 生成）的儲存格加淡黃底 + 註解，方便人工覆核。

五字標來源：screen_type 是廣告的直接寫「廣告」；新聞則呼叫
segment.generate_five_char_label 轉成 5 個字——預設不打網路，直接用
fallback_keyword_label（避免跟其他還在跑的批次搶 GPU/網路資源）；要真的
呼叫 VLM 生成五字標時加 --use-vlm-labels（建議在全量批次跑完之後再用）。

使用方式：
    python src/fill.py --channels ftv tvbs ebc cts --tag _test15
    python src/fill.py --use-vlm-labels   # 真的呼叫 VLM 產生五字標
"""

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import openpyxl
from openpyxl.comments import Comment
from openpyxl.styles import PatternFill

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment  # noqa: E402
import vlm  # noqa: E402  (load_vlm_env / call_vlm_for_label，僅 --use-vlm-labels 時用到)

OUT_DIR = ROOT / "out"
DEFAULT_TEMPLATE_PATH = ROOT / "data" / "excel" / "template.xlsx"
DEFAULT_OUTPUT_PATH = OUT_DIR / "12監看_20261106_auto.xlsx"
SHEET_NAME = "12監看"

CHANNEL_COLUMNS = {"tvbs": "F", "ebc": "J", "ftv": "N", "cts": "R"}
FIRST_MINUTE_CODE = 1200
FIRST_ROW = 5
LAST_ROW = 64

LOW_CONF_FILL_THRESHOLD = 0.6  # 比 segment.LOW_CONFIDENCE_THRESHOLD 寬鬆一點：多標一點給人工覆核比較安全
LOW_CONF_FILL = PatternFill(start_color="FFFFCC", end_color="FFFFCC", fill_type="solid")  # 淡黃底


def minute_code_to_row(minute_code: Optional[int]) -> Optional[int]:
    if minute_code is None:
        return None
    row = FIRST_ROW + (minute_code - FIRST_MINUTE_CODE)
    if row < FIRST_ROW or row > LAST_ROW:
        return None
    return row


def default_label_fn(
    event: dict,
    use_vlm: bool = False,
    vlm_text_fn: Optional[Callable] = None,
) -> tuple[str, bool, str]:
    """把一筆 fill event 轉成 (實際寫入的文字, 是否低信心度, 註解文字)。"""
    if event["kind"] == "ad":
        return "廣告", False, ""

    title = event["label"]  # 合併後的代表標題（尚未轉五字標）
    fallback_used = True
    if use_vlm and vlm_text_fn is not None:
        result = segment.generate_five_char_label(title, vlm_text_fn)
        label = result["label"]
        fallback_used = result["fallback_used"]
    else:
        label = segment.fallback_keyword_label(title)

    low_conf = (
        event["confidence"] < LOW_CONF_FILL_THRESHOLD
        or event.get("ambiguous_next") is not None
        or fallback_used
    )

    note_parts = [f"原標題：{title}", f"信心度 {event['confidence']:.2f}"]
    if event.get("ambiguous_next"):
        note_parts.append("換則判斷不確定（相似度落在模糊地帶），已保守切成新的一則，建議人工覆核")
    if fallback_used:
        note_parts.append("五字標為關鍵字擷取退路產生（非 VLM 生成），建議人工覆核")
    return label, low_conf, "；".join(note_parts)


def build_cell_plan(fill_events: list[dict], label_fn: Callable[[dict], tuple[str, bool, str]]) -> dict[int, dict]:
    """把一台的 fill_events 轉成 {row: {"text":, "low_conf":, "notes":[...]}}；
    同一列（同一分鐘）出現多筆事件時用「/」相連，任一筆低信心度整格就標低信心度。"""
    by_row: dict[int, list[dict]] = {}
    for ev in fill_events:
        row = minute_code_to_row(ev.get("minute_code"))
        if row is None:
            continue
        label, low_conf, note = label_fn(ev)
        by_row.setdefault(row, []).append({"label": label, "low_conf": low_conf, "note": note})

    plan: dict[int, dict] = {}
    for row, items in by_row.items():
        plan[row] = {
            "text": "/".join(item["label"] for item in items),
            "low_conf": any(item["low_conf"] for item in items),
            "notes": [item["note"] for item in items if item["note"]],
        }
    return plan


def apply_plan_to_sheet(ws, column: str, plan: dict[int, dict]) -> None:
    for row, info in plan.items():
        cell = ws[f"{column}{row}"]
        cell.value = info["text"]
        if info["low_conf"]:
            cell.fill = LOW_CONF_FILL
            if info["notes"]:
                cell.comment = Comment("\n".join(info["notes"]), "auto-fill")


def fill_workbook(
    channels_fill_events: dict[str, list[dict]],
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    use_vlm: bool = False,
    vlm_text_fn: Optional[Callable] = None,
) -> dict:
    wb = openpyxl.load_workbook(str(template_path))
    ws = wb[SHEET_NAME]

    stats: dict[str, dict] = {}
    for channel, fill_events in channels_fill_events.items():
        column = CHANNEL_COLUMNS.get(channel)
        if not column:
            print(f"[skip] 不認得的頻道：{channel}")
            continue

        def label_fn(ev, _use_vlm=use_vlm, _vlm_text_fn=vlm_text_fn):
            return default_label_fn(ev, use_vlm=_use_vlm, vlm_text_fn=_vlm_text_fn)

        plan = build_cell_plan(fill_events, label_fn)
        apply_plan_to_sheet(ws, column, plan)
        stats[channel] = {
            "filled_rows": len(plan),
            "low_conf_rows": sum(1 for p in plan.values() if p["low_conf"]),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(output_path))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="依分段結果填入 Excel 範本（只動四台五字標欄位）")
    parser.add_argument("--channels", nargs="+", default=list(CHANNEL_COLUMNS.keys()))
    parser.add_argument("--tag", type=str, default="", help="讀 out/{channel}{tag}_vlm.jsonl")
    parser.add_argument("--template", type=str, default=str(DEFAULT_TEMPLATE_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--use-vlm-labels", action="store_true", help="真的呼叫 VLM 產生五字標（建議全量批次跑完後再用）")
    args = parser.parse_args()

    channels_fill_events = {}
    for ch in args.channels:
        vlm_path = OUT_DIR / f"{ch}{args.tag}_vlm.jsonl"
        if not vlm_path.exists():
            print(f"[skip] {vlm_path} 不存在")
            continue
        channels_fill_events[ch] = segment.build_fill_events(ch, vlm_path)

    vlm_text_fn = None
    if args.use_vlm_labels:
        env = vlm.load_vlm_env()

        def vlm_text_fn(title, attempt, prev_violations):
            return segment.call_vlm_for_label(title, attempt, prev_violations, env)

    stats = fill_workbook(
        channels_fill_events,
        template_path=Path(args.template), output_path=Path(args.output),
        use_vlm=args.use_vlm_labels, vlm_text_fn=vlm_text_fn,
    )

    print(f"[ok] 輸出 {args.output}")
    for ch, s in stats.items():
        print(f"  {ch}: 填了 {s['filled_rows']} 列，低信心度 {s['low_conf_rows']} 列")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
