"""依時間戳從影片截出單一畫面。

使用方式：
    python tools/extract_frame.py data/videos/cts.mp4 00:33:42 data/gt/ads/cts_full_01.jpg
    python tools/extract_frame.py data/videos/ftv.mp4 36:42 data/gt/ads/ftv_split_01.jpg

時間戳支援 "HH:MM:SS" 或 "MM:SS" 兩種格式。
"""

import sys
from pathlib import Path

import cv2


def parse_timestamp(ts: str) -> float:
    """把 "HH:MM:SS" 或 "MM:SS" 字串轉成秒數（float）。"""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = "0"
        m, s = parts
    else:
        raise ValueError(f"無法解析時間戳: {ts!r}（請用 HH:MM:SS 或 MM:SS）")
    return int(h) * 3600 + int(m) * 60 + float(s)


def extract_frame(video_path: Path, seconds: float):
    """從影片在指定秒數抓一張畫面，回傳 (frame, actual_seconds) 或 (None, None)。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = int(round(seconds * fps)) if fps else 0
    frame_idx = max(0, min(frame_idx, max(total_frames - 1, 0)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, None

    actual_seconds = frame_idx / fps if fps else seconds
    return frame, actual_seconds


def main() -> int:
    if len(sys.argv) < 4:
        print("用法: python tools/extract_frame.py <影片路徑> <時間戳 HH:MM:SS|MM:SS> <輸出路徑>")
        return 1

    video_path = Path(sys.argv[1])
    ts = sys.argv[2]
    out_path = Path(sys.argv[3])

    if not video_path.exists():
        print(f"找不到影片: {video_path}")
        return 1

    seconds = parse_timestamp(ts)
    frame, actual_seconds = extract_frame(video_path, seconds)
    if frame is None:
        print(f"截圖失敗: {video_path} @ {ts}")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"已存檔: {out_path} (實際時間 {actual_seconds:.2f} 秒)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
