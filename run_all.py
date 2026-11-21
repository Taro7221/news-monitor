"""run_all.py：一行指令跑完整個流程（detect -> vlm -> fill -> evaluate）。

使用方式：
    python run_all.py                                                       # 全量：4台、整小時
    python run_all.py --tag _test15 --start-second 1800 --duration-seconds 900  # 15分鐘測試窗口
    python run_all.py --skip-detect --skip-vlm                              # 資料已經有了，只做 fill + evaluate
    python run_all.py --skip-evaluate                                       # 還沒準備 data/gt.csv 時跳過評估
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import calibrate  # noqa: E402
import detect  # noqa: E402
import vlm  # noqa: E402
import segment  # noqa: E402
import fill  # noqa: E402
import evaluate  # noqa: E402


def run_detect(channels: list[str], tag: str, start_second: int, duration_seconds) -> None:
    roi_config = calibrate.load_config()
    broadcast_start = detect.load_broadcast_start()
    detect.OUT_DIR.mkdir(parents=True, exist_ok=True)
    detect.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for channel in channels:
        events_path = detect.OUT_DIR / f"{channel}{tag}_events.jsonl"
        detect.process_channel(
            channel, roi_config, broadcast_start,
            start_second=start_second, max_seconds=duration_seconds,
            events_path=events_path,
        )


def run_vlm(channels: list[str], tag: str, timeout: int, max_retries: int) -> None:
    cache = vlm.ResultCache(vlm.CACHE_PATH)
    timing = vlm.TimingRecorder(vlm.TIMING_PATH)

    def job(channel: str):
        events_path = vlm.OUT_DIR / f"{channel}{tag}_events.jsonl"
        out_path = vlm.OUT_DIR / f"{channel}{tag}_vlm.jsonl"
        return channel, vlm.process_events_file(
            events_path, channel, cache, out_path, timing=timing,
            timeout=timeout, max_retries=max_retries,
        )

    with ThreadPoolExecutor(max_workers=max(1, len(channels))) as pool:
        futures = [pool.submit(job, ch) for ch in channels]
        for future in as_completed(futures):
            future.result()


def run_fill(channels: list[str], tag: str, output_path: str, use_vlm_labels: bool, use_vlm_judge: bool = False) -> dict:
    vlm_judge_fn = None
    if use_vlm_judge:
        judge_env = vlm.load_vlm_env()

        def vlm_judge_fn(title_a, title_b):
            return segment.call_vlm_judge_same_story(title_a, title_b, judge_env)

    channels_fill_events = {}
    for ch in channels:
        vlm_path = fill.OUT_DIR / f"{ch}{tag}_vlm.jsonl"
        if not vlm_path.exists():
            print(f"[skip] {vlm_path} 不存在")
            continue
        channels_fill_events[ch] = segment.build_fill_events(ch, vlm_path, vlm_judge_fn=vlm_judge_fn)

    vlm_text_fn = None
    if use_vlm_labels:
        env = vlm.load_vlm_env()

        def vlm_text_fn(title, attempt, prev_violations):
            return segment.call_vlm_for_label(title, attempt, prev_violations, env)

    return fill.fill_workbook(
        channels_fill_events, output_path=Path(output_path),
        use_vlm=use_vlm_labels, vlm_text_fn=vlm_text_fn, vlm_judge_fn=vlm_judge_fn,
    )


def run_evaluate(gt_path: str, excel_path: str, output: str):
    gt_path = Path(gt_path)
    if not gt_path.exists():
        print(f"[skip] {gt_path} 不存在，略過評估（這是需要你手動標記的檔案）")
        return None

    gt = evaluate.load_ground_truth(gt_path)
    if not gt:
        print(f"[skip] {gt_path} 沒有任何資料列")
        return None

    channels = sorted({ch for ch, _ in gt.keys()})
    minutes = sorted({m for _, m in gt.keys()})
    predicted = evaluate.load_predicted(Path(excel_path), channels, minutes)
    metrics = evaluate.compute_metrics(gt, predicted)
    report = evaluate.render_markdown(metrics)
    if output:
        Path(output).write_text(report, encoding="utf-8")
    print(report)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="一行指令跑完整個流程：detect -> vlm -> fill -> evaluate")
    parser.add_argument("--channels", nargs="+", default=["ftv", "tvbs", "ebc", "cts"])
    parser.add_argument("--tag", type=str, default="", help="輸出檔名標記，例如 --tag _test15（避免覆蓋正式全量資料）")
    parser.add_argument("--start-second", type=int, default=0)
    parser.add_argument("--duration-seconds", type=int, default=None, help="處理幾秒（預設到影片結尾）")
    parser.add_argument("--timeout", type=int, default=vlm.DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=vlm.DEFAULT_MAX_RETRIES)
    parser.add_argument("--output-excel", type=str, default=str(fill.DEFAULT_OUTPUT_PATH))
    parser.add_argument("--gt", type=str, default=str(evaluate.DEFAULT_GT_PATH))
    parser.add_argument("--eval-output", type=str, default=None)
    parser.add_argument("--use-vlm-labels", action="store_true", help="真的呼叫 VLM 產生五字標")
    parser.add_argument("--use-vlm-judge", action="store_true", help="模糊地帶呼叫 VLM 判斷是否同一則")
    parser.add_argument("--skip-detect", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-fill", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()

    if not args.skip_detect:
        print("=== [1/4] detect：偵測畫面變化 ===")
        run_detect(args.channels, args.tag, args.start_second, args.duration_seconds)

    if not args.skip_vlm:
        print("\n=== [2/4] vlm：呼叫 VLM 判讀 ===")
        run_vlm(args.channels, args.tag, args.timeout, args.max_retries)

    if not args.skip_fill:
        print("\n=== [3/4] fill：填入 Excel ===")
        stats = run_fill(args.channels, args.tag, args.output_excel, args.use_vlm_labels, args.use_vlm_judge)
        for ch, s in stats.items():
            print(f"  {ch}: 填了 {s['filled_rows']} 列，低信心度 {s['low_conf_rows']} 列")

    if not args.skip_evaluate:
        print("\n=== [4/4] evaluate：跟人工標記比對 ===")
        run_evaluate(args.gt, args.output_excel, args.eval_output)

    print(f"\n[ok] 全部完成，耗時 {time.perf_counter() - start:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
