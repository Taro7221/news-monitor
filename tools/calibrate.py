"""ROI 校準工具。

1. 從每台影片抽 N 張不同時間點的截圖，存到 out/calib/。
2. 若 config/channels.yaml 不存在，依預設值建立一份（比例座標，供人工微調）。
3. 讀取 config/channels.yaml，把每台的 ROI 框畫在其中一張截圖上，
   合成一張 2x2 總覽預覽圖 out/calib/roi_preview.jpg，方便檢查/微調。

使用方式：
    python tools/calibrate.py                  # 抽圖 + 建立預設設定 + 畫預覽圖
    python tools/calibrate.py --preview-only    # 只重畫預覽圖（改完 yaml 後重跑這個檢查）
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "data" / "videos"
CALIB_DIR = ROOT / "out" / "calib"
CONFIG_PATH = ROOT / "config" / "channels.yaml"

CHANNELS = ["ftv", "tvbs", "ebc", "cts"]
ROI_NAMES = ["headline_bar", "ticker", "logo"]
ROI_COLORS = {
    "headline_bar": (0, 0, 255),   # 紅：主標題 bar
    "ticker": (0, 200, 0),         # 綠：跑馬燈
    "logo": (255, 0, 0),           # 藍：台標
}
ROI_LABELS = {
    "headline_bar": "headline_bar (主標題)",
    "ticker": "ticker (跑馬燈)",
    "logo": "logo (台標)",
}

DEFAULT_ROI = {
    "headline_bar": {"x1": 0.0, "y1": 0.80, "x2": 1.0, "y2": 0.90},
    "ticker": {"x1": 0.0, "y1": 0.92, "x2": 1.0, "y2": 0.98},
    "logo": {"x1": 0.02, "y1": 0.03, "x2": 0.15, "y2": 0.12},
}


def sample_frame_indices(total_frames: int, n: int, margin: float = 0.02) -> list[int]:
    """在 [margin, 1-margin] 的影片長度範圍內，等距取 n 個 frame index。"""
    if total_frames <= 0:
        return []
    n = max(1, min(n, total_frames))
    lo = margin * total_frames
    hi = (1 - margin) * total_frames
    if n == 1:
        positions = [(lo + hi) / 2]
    else:
        positions = np.linspace(lo, hi, n)
    indices = sorted({int(round(p)) for p in positions})
    return [min(max(i, 0), total_frames - 1) for i in indices]


def ratio_to_pixels(roi: dict, width: int, height: int) -> tuple[int, int, int, int]:
    """把比例座標 (x1,y1,x2,y2 皆 0~1) 轉成像素座標，並夾在影像範圍內。"""
    x1 = int(round(roi["x1"] * width))
    y1 = int(round(roi["y1"] * height))
    x2 = int(round(roi["x2"] * width))
    y2 = int(round(roi["y2"] * height))
    x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
    y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
    return x1, y1, x2, y2


def build_default_config(channels: list[str]) -> dict:
    return {ch: {name: dict(bbox) for name, bbox in DEFAULT_ROI.items()} for ch in channels}


def render_default_yaml(channels: list[str]) -> str:
    lines = [
        "# ROI 校準設定：每台三個區塊的比例座標 (0~1)",
        "# x1,y1 = 左上角比例座標；x2,y2 = 右下角比例座標",
        "# 影片解析度請看 tools/check_env.py 的輸出；用比例座標可以不受解析度影響",
        "#   headline_bar: 主標題 bar（新聞標題/五字標顯示區）",
        "#   ticker       : 跑馬燈",
        "#   logo         : 台標",
        "# 這些是粗略預設值，請對照 out/calib/roi_preview.jpg 自行微調後重跑：",
        "#   python tools/calibrate.py --preview-only",
        "",
    ]
    for ch in channels:
        lines.append(f"{ch}:")
        for name in ROI_NAMES:
            bbox = DEFAULT_ROI[name]
            lines.append(f"  {name}:")
            lines.append(f"    x1: {bbox['x1']}")
            lines.append(f"    y1: {bbox['y1']}")
            lines.append(f"    x2: {bbox['x2']}")
            lines.append(f"    y2: {bbox['y2']}")
        lines.append("")
    return "\n".join(lines)


def extract_screenshots(channel: str, n: int) -> list[Path]:
    video_path = VIDEOS_DIR / f"{channel}.mp4"
    if not video_path.exists():
        print(f"[skip] {video_path} 不存在")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[skip] {video_path} 無法開啟")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total_frames, n)

    saved = []
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[warn] {channel} frame {frame_idx} 讀取失敗")
            continue
        seconds = frame_idx / fps if fps else 0
        mm, ss = divmod(int(seconds), 60)
        out_path = CALIB_DIR / f"{channel}_{i:02d}_{mm:02d}m{ss:02d}s.jpg"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)
    cap.release()
    print(f"[ok] {channel}: 存了 {len(saved)} 張截圖")
    return saved


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_rois(image: np.ndarray, channel_config: dict) -> np.ndarray:
    """在單張影像上畫出該台設定好的 ROI 框，回傳新的影像陣列（不修改原圖）。"""
    img = image.copy()
    height, width = img.shape[:2]
    for name in ROI_NAMES:
        bbox = channel_config.get(name)
        if not bbox:
            continue
        x1, y1, x2, y2 = ratio_to_pixels(bbox, width, height)
        color = ROI_COLORS[name]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, ROI_LABELS[name], (x1 + 2, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return img


def build_preview(channels: list[str], config: dict) -> np.ndarray | None:
    annotated = []
    for ch in channels:
        candidates = sorted(CALIB_DIR.glob(f"{ch}_*.jpg"))
        if not candidates:
            print(f"[skip preview] {ch} 沒有截圖，先執行抽圖")
            continue
        sample_path = candidates[len(candidates) // 2]
        frame = cv2.imread(str(sample_path))
        if frame is None:
            continue
        ch_config = config.get(ch, {})
        annotated_frame = draw_rois(frame, ch_config)
        cv2.putText(
            annotated_frame, ch.upper(), (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
        )
        annotated.append(annotated_frame)

    if not annotated:
        return None

    h, w = annotated[0].shape[:2]
    annotated = [cv2.resize(a, (w, h)) for a in annotated]
    while len(annotated) < 4:
        annotated.append(np.zeros((h, w, 3), dtype=np.uint8))

    top = np.hstack(annotated[0:2])
    bottom = np.hstack(annotated[2:4])
    grid = np.vstack([top, bottom])
    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description="ROI 校準工具")
    parser.add_argument("--channels", nargs="+", default=CHANNELS)
    parser.add_argument("--n", type=int, default=10, help="每台抽幾張截圖")
    parser.add_argument("--preview-only", action="store_true", help="跳過抽圖，只重畫預覽圖")
    parser.add_argument("--force-config", action="store_true", help="覆寫已存在的 config/channels.yaml")
    args = parser.parse_args()

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not args.preview_only:
        for ch in args.channels:
            extract_screenshots(ch, args.n)

    if not CONFIG_PATH.exists() or args.force_config:
        CONFIG_PATH.write_text(render_default_yaml(args.channels), encoding="utf-8")
        print(f"[ok] 已建立預設設定：{CONFIG_PATH.relative_to(ROOT)}")
    else:
        print(f"[skip] {CONFIG_PATH.relative_to(ROOT)} 已存在，不覆寫（請手動編輯或加 --force-config）")

    config = load_config()
    preview = build_preview(args.channels, config)
    if preview is not None:
        preview_path = CALIB_DIR / "roi_preview.jpg"
        cv2.imwrite(str(preview_path), preview)
        print(f"[ok] 預覽圖已產生：{preview_path.relative_to(ROOT)}")
    else:
        print("[warn] 沒有可用截圖，無法產生預覽圖")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
