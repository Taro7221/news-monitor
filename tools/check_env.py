"""環境健檢腳本。依序檢查 Python 版本、ffmpeg、套件、影片、Excel、VLM 連線，
每一項印出 ✅ 或 ❌ 加原因。全部通過才 exit code 0。"""

import importlib
import subprocess
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OK = "✅"
FAIL = "❌"

failures = 0


def report(ok: bool, label: str, detail: str = "") -> None:
    global failures
    mark = OK if ok else FAIL
    if not ok:
        failures += 1
    line = f"{mark} {label}"
    if detail:
        line += f" — {detail}"
    print(line)


def check_python_version() -> None:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 9)
    report(ok, f"Python 版本 {sys.version.split()[0]}", "需要 3.9 以上" if not ok else "")


def check_ffmpeg() -> None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else ""
        report(result.returncode == 0, "ffmpeg 可呼叫", first_line)
    except FileNotFoundError:
        report(False, "ffmpeg 可呼叫", "找不到 ffmpeg，請確認已安裝並加入 PATH")
    except Exception as e:
        report(False, "ffmpeg 可呼叫", f"例外：{e}")


REQUIRED_PACKAGES = [
    ("cv2", "opencv-python"),
    ("imagehash", "imagehash"),
    ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("openpyxl", "openpyxl"),
    ("requests", "requests"),
    ("dotenv", "python-dotenv"),
    ("yaml", "pyyaml"),
    ("tqdm", "tqdm"),
    ("rapidfuzz", "rapidfuzz"),
    ("pytest", "pytest"),
]
OPTIONAL_PACKAGES = [
    ("librosa", "librosa"),
]


def check_packages() -> None:
    for module_name, package_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
            report(True, f"套件 import：{package_name}")
        except ImportError as e:
            report(False, f"套件 import：{package_name}", str(e))

    for module_name, package_name in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(module_name)
            report(True, f"套件 import：{package_name}（選用）")
        except ImportError:
            print(f"{OK} 套件 import：{package_name}（選用，未安裝，非必要）")


VIDEO_NAMES = ["ftv", "tvbs", "ebc", "cts"]


def check_videos() -> None:
    import cv2

    videos_dir = ROOT / "data" / "videos"
    for name in VIDEO_NAMES:
        path = videos_dir / f"{name}.mp4"
        if not path.exists():
            report(False, f"影片 {path.relative_to(ROOT)}", "檔案不存在")
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            report(False, f"影片 {path.relative_to(ROOT)}", "無法開啟，檔案可能損毀")
            cap.release()
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps else 0
        cap.release()
        detail = f"{width}x{height}, {fps:.2f} fps, 長度 {duration:.1f} 秒"
        report(True, f"影片 {path.relative_to(ROOT)}", detail)


EXCEL_NAMES = ["template.xlsx", "example.xlsx"]


def check_excel() -> None:
    import openpyxl

    excel_dir = ROOT / "data" / "excel"
    for name in EXCEL_NAMES:
        path = excel_dir / name
        if not path.exists():
            report(False, f"Excel {path.relative_to(ROOT)}", "檔案不存在")
            continue
        try:
            wb = openpyxl.load_workbook(str(path), read_only=True)
            sheets = ", ".join(wb.sheetnames)
            wb.close()
            report(True, f"Excel {path.relative_to(ROOT)}", f"工作表：{sheets}")
        except Exception as e:
            report(False, f"Excel {path.relative_to(ROOT)}", f"開啟失敗：{e}")


def check_vlm() -> None:
    import os

    from dotenv import load_dotenv
    import requests
    from requests.auth import HTTPBasicAuth

    load_dotenv(ROOT / ".env")

    vlm_url = os.getenv("VLM_URL", "").strip()
    vlm_user = os.getenv("VLM_USER", "").strip()
    vlm_pass = os.getenv("VLM_PASS", "").strip()
    vlm_model = os.getenv("VLM_MODEL", "").strip()

    if not vlm_url:
        report(False, "VLM 連線", "VLM_URL 未設定於 .env")
        return
    if not vlm_model:
        report(False, "VLM 連線", "VLM_MODEL 未設定於 .env")
        return

    endpoint = vlm_url.rstrip("/") + "/api/chat"
    payload = {
        "model": vlm_model,
        "messages": [{"role": "user", "content": "請回覆「測試成功」兩個字即可。"}],
        "stream": False,
    }

    try:
        start = time.time()
        resp = requests.post(
            endpoint,
            json=payload,
            auth=HTTPBasicAuth(vlm_user, vlm_pass) if vlm_user else None,
            timeout=60,
        )
        elapsed = time.time() - start
        if resp.status_code == 200:
            try:
                data = resp.json()
                content = data.get("message", {}).get("content", "")
            except ValueError:
                content = resp.text
            content = content.strip().replace("\n", " ")
            if len(content) > 200:
                content = content[:200] + "..."
            report(
                True,
                "VLM 連線與回應",
                f"耗時 {elapsed:.2f} 秒，回應：{content}",
            )
        else:
            report(
                False,
                "VLM 連線與回應",
                f"HTTP {resp.status_code}，耗時 {elapsed:.2f} 秒，內容：{resp.text[:200]}",
            )
    except requests.exceptions.RequestException as e:
        report(False, "VLM 連線與回應", f"請求失敗：{e}")


def main() -> int:
    print("=== Python 版本 ===")
    check_python_version()

    print("\n=== ffmpeg ===")
    check_ffmpeg()

    print("\n=== 套件 ===")
    check_packages()

    print("\n=== 影片（data/videos） ===")
    check_videos()

    print("\n=== Excel（data/excel） ===")
    check_excel()

    print("\n=== VLM（.env 設定） ===")
    check_vlm()

    print()
    if failures:
        print(f"{FAIL} 共有 {failures} 項檢查未通過")
        return 1
    print(f"{OK} 全部檢查通過")
    return 0


if __name__ == "__main__":
    sys.exit(main())
