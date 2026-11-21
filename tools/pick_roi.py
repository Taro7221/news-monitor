"""互動式 ROI 座標工具。

滑鼠移到圖片上即時顯示像素座標 / 比例座標；左鍵點兩下（左上角、右下角）
框出一個矩形，在終端機印出可以直接貼進 config/channels.yaml 的比例座標。

使用方式：
    .\\.venv\\Scripts\\python.exe tools\\pick_roi.py out\\calib\\ftv_00_01m11s.jpg

操作：
    滑鼠移動   -> 畫面左上角顯示目前像素座標與比例座標
    左鍵點兩下 -> 第一下＝左上角，第二下＝右下角，印出 bbox（像素＋比例）
    r         -> 清除目前選取，重新框
    q 或 Esc  -> 離開
"""

import sys
from pathlib import Path

import cv2

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def pixel_to_ratio(x: int, y: int, width: int, height: int) -> tuple[float, float]:
    return x / width, y / height


def normalize_rect(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int, int, int]:
    """把任意順序的兩點整理成 (x1, y1, x2, y2)，x1<=x2, y1<=y2。"""
    (x1, y1), (x2, y2) = p1, p2
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return x1, y1, x2, y2


def format_ratio_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> str:
    rx1, ry1 = pixel_to_ratio(x1, y1, width, height)
    rx2, ry2 = pixel_to_ratio(x2, y2, width, height)
    return (
        f"pixel: x1={x1} y1={y1} x2={x2} y2={y2}\n"
        f"ratio:\n"
        f"    x1: {rx1:.4f}\n"
        f"    y1: {ry1:.4f}\n"
        f"    x2: {rx2:.4f}\n"
        f"    y2: {ry2:.4f}"
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python tools/pick_roi.py <圖片路徑>")
        return 1

    path = Path(sys.argv[1])
    img = cv2.imread(str(path))
    if img is None:
        print(f"無法讀取圖片: {path}")
        return 1

    height, width = img.shape[:2]
    window = "pick_roi  (左鍵點兩下框選 / r 重選 / q,Esc 離開)"
    points: list[tuple[int, int]] = []

    def redraw(mouse_pos: tuple[int, int] | None = None) -> None:
        canvas = img.copy()
        if len(points) == 1:
            cv2.circle(canvas, points[0], 4, (0, 0, 255), -1)
        if len(points) == 2:
            x1, y1, x2, y2 = normalize_rect(points[0], points[1])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if mouse_pos:
            mx, my = mouse_pos
            rx, ry = pixel_to_ratio(mx, my, width, height)
            text = f"px=({mx},{my})  ratio=({rx:.3f},{ry:.3f})"
            cv2.rectangle(canvas, (0, 0), (460, 28), (0, 0, 0), -1)
            cv2.putText(
                canvas, text, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA,
            )
        cv2.imshow(window, canvas)

    def on_mouse(event, x, y, flags, userdata) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            redraw((x, y))
        elif event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 2:
                points.clear()
            points.append((x, y))
            if len(points) == 2:
                x1, y1, x2, y2 = normalize_rect(points[0], points[1])
                print(format_ratio_bbox(x1, y1, x2, y2, width, height))
                print()
            redraw((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    print(f"圖片尺寸: {width}x{height}")
    print("左鍵點兩下框選矩形（左上角→右下角），結果會印在這裡。q 或 Esc 離開。\n")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            points.clear()
            redraw()

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
