"""變化偵測：逐秒抽幀、裁出主標題 bar、用 phash 偵測換則事件。

流程：
1. 每秒取 1 幀，裁出該台的 headline_bar ROI（座標讀 config/channels.yaml）。
2. 用 imagehash.phash 算出這個裁切區域的感知雜湊，跟「目前確認的狀態」比較。
3. 差異超過門檻 -> 進入候選狀態；候選狀態要連續穩定 stable_seconds 秒
   （同一個新雜湊值）才算真的換了一個事件，避免轉場閃爍誤判。
4. 另外每 heartbeat_interval 秒強制存一張全畫面心跳幀（不管有沒有偵測到事件），
   避免廣告等「沒有 headline_bar」的長時間區段完全沒有任何記錄。
5. 事件與心跳都寫進 out/{channel}_events.jsonl（一行一筆 json）。

使用方式：
    python src/detect.py                       # 處理 config/channels.yaml 裡所有頻道
    python src/detect.py --channels ftv tvbs    # 只處理指定頻道
    python src/detect.py --max-seconds 120      # 只處理前 120 秒（除錯用）
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import cv2
import imagehash
import numpy as np
from PIL import Image

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import calibrate  # noqa: E402  (reuse ROI loading / ratio_to_pixels)

VIDEOS_DIR = ROOT / "data" / "videos"
FRAMES_DIR = ROOT / "out" / "frames"
OUT_DIR = ROOT / "out"

# 實測 threshold=10（64 位元 phash 的 hamming 距離）會把同一句沒變的標題，
# 因為裁切邊緣帶到旁邊變動的小細節，重複判成兩次事件；threshold=20 能大幅
# 減少這類重複，仍抓得到真的換則。headline_bar 完全沒有文字（例如 FTV
# 棚內美術模板當下沒有標題框、純粹是會動的攝影棚畫面）時，即使調高門檻也
# 還是會觸發事件——這是預期內的雜訊，留給下一步讀字的 VLM 回報「無標題」
# 來過濾，不在這裡硬擋。
DEFAULT_THRESHOLD = 20
DEFAULT_STABLE_SECONDS = 2
DEFAULT_HEARTBEAT_INTERVAL = 30
DEFAULT_HASH_SIZE = 8
MIN_CROP_SIDE = 32  # phash 需要足夠像素做 DCT，過小的裁切先放大


def load_broadcast_start() -> datetime:
    """讀 .env 的 BROADCAST_START，解析失敗就回退成一個假的基準時間。"""
    from dotenv import load_dotenv
    import os

    load_dotenv(ROOT / ".env")
    raw = os.getenv("BROADCAST_START", "").strip()
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime(2026, 11, 6, 12, 0, 0)


def crop_roi(frame: np.ndarray, roi: dict) -> np.ndarray:
    """依比例座標裁切畫面，裁切結果過小時放大到可以算 phash 的最小尺寸。"""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = calibrate.ratio_to_pixels(roi, width, height)
    if x2 <= x1 or y2 <= y1:
        return frame
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if h < MIN_CROP_SIDE or w < MIN_CROP_SIDE:
        scale = MIN_CROP_SIDE / max(1, min(h, w))
        crop = cv2.resize(crop, (max(MIN_CROP_SIDE, int(w * scale)), max(MIN_CROP_SIDE, int(h * scale))))
    return crop


def compute_phash(image_bgr: np.ndarray, hash_size: int = DEFAULT_HASH_SIZE) -> imagehash.ImageHash:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb), hash_size=hash_size)


def detect_changes(
    series: list[tuple[int, object]],
    threshold: int,
    stable_seconds: int,
    diff: Callable[[object, object], int] = lambda a, b: a - b,
) -> list[tuple[int, object]]:
    """純邏輯：給一串 (second, hash值) 序列，回傳確認換則的 (second, hash值) 清單。

    規則：與目前基準的差異 > threshold 時進入候選狀態；候選狀態要連續
    stable_seconds 秒都跟候選值一致才確認為事件並成為新基準。候選狀態中途
    出現其他數值（既不像候選也不像基準）會改成以新數值重新候選；如果
    候選中途的值其實跟基準很接近，則視為雜訊、取消候選、保持原基準。
    """
    events: list[tuple[int, object]] = []
    if not series:
        return events

    base_t, base_v = series[0]
    pending: Optional[tuple[int, object]] = None

    for t, v in series[1:]:
        if pending is not None:
            p_t, p_v = pending
            if diff(v, p_v) <= threshold:
                if t - p_t >= stable_seconds:
                    events.append((p_t, p_v))
                    base_t, base_v = p_t, p_v
                    pending = None
                # 還沒滿 stable_seconds，繼續等待
                continue
            # 候選狀態被打斷：看這個新值要不要變成新的候選
            if diff(v, base_v) > threshold:
                pending = (t, v)
            else:
                pending = None
        else:
            if diff(v, base_v) > threshold:
                pending = (t, v)

    return events


def iter_seconds(video_path: Path, max_seconds: Optional[int] = None):
    """循序解碼整支影片，每秒 yield 一次 (second, frame)。

    用 grab()/retrieve() 循序解碼而不是重複 seek，避免對壓縮影片做大量
    random seek 造成的不穩定與效能問題。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        cap.release()
        return

    frame_idx = 0
    next_second = 0
    try:
        while True:
            if max_seconds is not None and next_second > max_seconds:
                break
            ok = cap.grab()
            if not ok:
                break
            target_frame_idx = round(next_second * fps)
            if frame_idx >= target_frame_idx:
                ok2, frame = cap.retrieve()
                if not ok2:
                    break
                yield next_second, frame
                next_second += 1
            frame_idx += 1
    finally:
        cap.release()


def process_channel(
    channel: str,
    roi_config: dict,
    broadcast_start: datetime,
    threshold: int = DEFAULT_THRESHOLD,
    stable_seconds: int = DEFAULT_STABLE_SECONDS,
    heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL,
    hash_size: int = DEFAULT_HASH_SIZE,
    max_seconds: Optional[int] = None,
) -> int:
    video_path = VIDEOS_DIR / f"{channel}.mp4"
    if not video_path.exists():
        print(f"[skip] {video_path} 不存在")
        return 0

    headline_roi = roi_config.get(channel, {}).get("headline_bar")
    if not headline_roi:
        print(f"[skip] config/channels.yaml 沒有 {channel}.headline_bar")
        return 0

    channel_frames_dir = FRAMES_DIR / channel
    channel_frames_dir.mkdir(parents=True, exist_ok=True)
    events_path = OUT_DIR / f"{channel}_events.jsonl"

    def make_record(
        second: int, kind: str, phash: imagehash.ImageHash, full_path: Path, crop_path: Path
    ) -> dict:
        clock_dt = broadcast_start + timedelta(seconds=second)
        return {
            "channel": channel,
            "type": kind,
            "video_second": second,
            "clock_time": clock_dt.strftime("%H:%M:%S"),
            "minute_code": int(clock_dt.strftime("%H%M")),
            "phash": str(phash),
            # image_path = 整張畫面（給 VLM 看上下文）；crop_path = headline_bar 裁切（給 VLM 看清楚文字）
            "image_path": str(full_path.relative_to(ROOT).as_posix()),
            "crop_path": str(crop_path.relative_to(ROOT).as_posix()),
        }

    def save_pair(second: int, kind: str, frame: np.ndarray, crop: np.ndarray) -> tuple[Path, Path]:
        full_path = channel_frames_dir / f"{kind}_{second:05d}.jpg"
        crop_path = channel_frames_dir / f"{kind}_{second:05d}_crop.jpg"
        cv2.imwrite(str(full_path), frame)
        cv2.imwrite(str(crop_path), crop)
        return full_path, crop_path

    series: list[tuple[int, imagehash.ImageHash]] = []
    frame_cache: dict[int, np.ndarray] = {}
    last_heartbeat = -heartbeat_interval  # 讓第一秒就觸發一次心跳

    event_count = 0
    with open(events_path, "w", encoding="utf-8") as f:
        for second, frame in iter_seconds(video_path, max_seconds=max_seconds):
            crop = crop_roi(frame, headline_roi)
            phash = compute_phash(crop, hash_size=hash_size)
            series.append((second, phash))
            frame_cache[second] = frame

            if second - last_heartbeat >= heartbeat_interval:
                full_path, crop_path = save_pair(second, "heartbeat", frame, crop)
                f.write(json.dumps(make_record(second, "heartbeat", phash, full_path, crop_path), ensure_ascii=False) + "\n")
                last_heartbeat = second

        events = detect_changes(series, threshold=threshold, stable_seconds=stable_seconds)
        for second, phash in events:
            event_frame = frame_cache.get(second)
            if event_frame is None:
                continue
            crop = crop_roi(event_frame, headline_roi)
            full_path, crop_path = save_pair(second, "event", event_frame, crop)
            f.write(json.dumps(make_record(second, "event", phash, full_path, crop_path), ensure_ascii=False) + "\n")
            event_count += 1

    print(f"[ok] {channel}: {event_count} 個事件, {len(series)} 秒已處理 -> {events_path.relative_to(ROOT)}")
    return event_count


def main() -> int:
    parser = argparse.ArgumentParser(description="變化偵測：逐秒抽幀比對主標題 bar")
    parser.add_argument("--channels", nargs="+", default=calibrate.CHANNELS)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--stable-seconds", type=int, default=DEFAULT_STABLE_SECONDS)
    parser.add_argument("--heartbeat-interval", type=int, default=DEFAULT_HEARTBEAT_INTERVAL)
    parser.add_argument("--hash-size", type=int, default=DEFAULT_HASH_SIZE)
    parser.add_argument("--max-seconds", type=int, default=None, help="只處理前 N 秒（除錯用）")
    args = parser.parse_args()

    roi_config = calibrate.load_config()
    broadcast_start = load_broadcast_start()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for channel in args.channels:
        count = process_channel(
            channel,
            roi_config,
            broadcast_start,
            threshold=args.threshold,
            stable_seconds=args.stable_seconds,
            heartbeat_interval=args.heartbeat_interval,
            hash_size=args.hash_size,
            max_seconds=args.max_seconds,
        )
        summary[channel] = count

    print("\n=== 事件數統計 ===")
    for channel, count in summary.items():
        print(f"{channel}: {count} 個事件")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
