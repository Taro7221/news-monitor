"""VLM 客戶端：呼叫 Colab 上的 Ollama /api/chat，判讀單一秒的畫面。

對每一筆輸入（整張畫面 + headline_bar 裁切圖）送出一次請求，用 JSON schema
強制回傳：
    {screen_type, main_title, main_title_visible, confidence}

screen_type 五選一：
    news     - 正常新聞畫面，看得到主標題 bar
    ad_wrap  - 左側大畫面播廣告、右側小畫面繼續推播新聞（例如「現在最焦點」）
    full_ad  - 整個畫面都是廣告，沒有新聞內容
    promo    - 本台自家宣傳（節目預告、台呼等），不是新聞也不是外部廣告
    other    - 以上皆非（轉場、黑畫面等）

附重試／逾時／結果快取（同一張圖的內容雜湊不會重送）；快取寫在
out/vlm_cache.jsonl，中斷後重跑會自動跳過已經問過的圖，達成斷點續跑。

使用方式：
    python src/vlm.py --events out/ftv_events.jsonl --channel ftv
    python src/vlm.py --channels ftv tvbs ebc cts   # 依序處理 out/{ch}_events.jsonl
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
CACHE_PATH = OUT_DIR / "vlm_cache.jsonl"

DEFAULT_TIMEOUT = 180  # 秒；Colab 冷啟動 + CPU 推論可能偏慢
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 2
FULL_IMAGE_MAX_WIDTH = 1280

SCREEN_TYPES = ["news", "ad_wrap", "full_ad", "promo", "other"]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "screen_type": {"type": "string", "enum": SCREEN_TYPES},
        "main_title": {"type": "string"},
        "main_title_visible": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["screen_type", "main_title", "main_title_visible", "confidence"],
}

PROMPT = """你是台灣電視新聞的監看助手。你會看到兩張圖：
第一張是「整個畫面」（提供上下文），第二張是「主標題 bar 的放大裁切圖」（方便你看清楚文字）。

判斷規則（很重要）：
- 畫面下方的「主標題 bar」（通常是一整條有底色的粗體字，例如深色底、黃底等）才算是「這一則新聞的標題」。
- 最下面持續捲動的跑馬燈、另外的「快訊」色塊、角落的天氣小部件，都不算本則新聞標題，不要拿來填 main_title，也不要讓它們影響你的判斷。
- 如果節目當下主畫面播的就是天氣報告（天氣佔據主標題 bar/主畫面，不是角落小部件），那就正常把天氣相關文字當作 main_title。
- 同一則新聞可能會經過「主播稿頭→記者報導(SOT)→連線」等不同畫面，只要主標題 bar 文字沒變，都算同一則，請照實回報目前畫面上的文字，不用推測是否為同一則。

screen_type 五選一：
- "news"：正常新聞畫面（不管是棚內主播、記者現場、或連線，只要看得到或看不到主標題 bar 都算 news，除非符合下面其他分類）。
- "ad_wrap"：畫面明顯分成左右兩塊，左邊「大畫面」在播商業廣告，右邊「小畫面」繼續顯示新聞內容（有些台會標「現在最焦點」、「焦點新聞」之類的字樣）。
- "full_ad"：整個畫面都是廣告（商品、藥品、金融商品等），完全沒有新聞內容，通常也看不到主標題 bar。
- "promo"：本台自己的宣傳、節目預告、台呼，不是新聞內容，也不是第三方廣告。
- "other"：以上皆非，例如純轉場畫面、黑畫面、開場動畫。

請回傳：
- screen_type：上面五選一。
- main_title：目前主標題 bar 上實際顯示的文字（逐字照抄，看不到就填空字串 ""）。
- main_title_visible：現在能不能清楚讀到完整的主標題 bar（true/false）。screen_type 不是 "news"／"ad_wrap"，或文字被擋住、太模糊、正在轉場時，請填 false。
- confidence：你對這個判斷的信心程度，0 到 1 之間的小數。
"""


def load_vlm_env() -> dict:
    load_dotenv(ROOT / ".env")
    return {
        "url": os.getenv("VLM_URL", "").strip().rstrip("/"),
        "user": os.getenv("VLM_USER", "").strip(),
        "password": os.getenv("VLM_PASS", "").strip(),
        "model": os.getenv("VLM_MODEL", "").strip(),
    }


def resize_for_full(image_bgr: np.ndarray, max_width: int = FULL_IMAGE_MAX_WIDTH) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if width <= max_width:
        return image_bgr
    scale = max_width / width
    return cv2.resize(image_bgr, (max_width, int(round(height * scale))))


def image_to_base64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("無法編碼圖片")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def load_image_pair_base64(full_path: Path, crop_path: Path) -> tuple[str, str, bytes, bytes]:
    """讀取整張畫面 + 裁切圖，整張畫面會縮到 FULL_IMAGE_MAX_WIDTH 寬。回傳
    (full_b64, crop_b64, full_bytes_for_hash, crop_bytes_for_hash)。"""
    full_img = cv2.imread(str(full_path))
    crop_img = cv2.imread(str(crop_path))
    if full_img is None:
        raise FileNotFoundError(f"讀不到整張畫面: {full_path}")
    if crop_img is None:
        raise FileNotFoundError(f"讀不到裁切圖: {crop_path}")

    full_img = resize_for_full(full_img)
    full_b64 = image_to_base64(full_img)
    crop_b64 = image_to_base64(crop_img)

    full_bytes = full_path.read_bytes()
    crop_bytes = crop_path.read_bytes()
    return full_b64, crop_b64, full_bytes, crop_bytes


def compute_cache_key(full_bytes: bytes, crop_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(full_bytes)
    h.update(crop_bytes)
    return h.hexdigest()


def build_payload(model: str, full_b64: str, crop_b64: str) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [full_b64, crop_b64],
            }
        ],
        "format": RESPONSE_SCHEMA,
        "stream": False,
        "options": {"temperature": 0},
    }


class VLMResponseError(ValueError):
    pass


def parse_response(raw_content: str) -> dict:
    """解析 /api/chat 回傳的 message.content（應該是符合 schema 的 JSON 字串）。"""
    try:
        data = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError) as e:
        raise VLMResponseError(f"回應不是合法 JSON: {e!r}") from e

    if not isinstance(data, dict):
        raise VLMResponseError(f"回應不是 JSON object: {data!r}")

    missing = [k for k in RESPONSE_SCHEMA["required"] if k not in data]
    if missing:
        raise VLMResponseError(f"回應缺少欄位: {missing}")

    if data["screen_type"] not in SCREEN_TYPES:
        raise VLMResponseError(f"screen_type 不在允許範圍: {data['screen_type']!r}")
    if not isinstance(data["main_title"], str):
        raise VLMResponseError(f"main_title 不是字串: {data['main_title']!r}")
    if not isinstance(data["main_title_visible"], bool):
        raise VLMResponseError(f"main_title_visible 不是布林值: {data['main_title_visible']!r}")
    if not isinstance(data["confidence"], (int, float)):
        raise VLMResponseError(f"confidence 不是數字: {data['confidence']!r}")

    return {
        "screen_type": data["screen_type"],
        "main_title": data["main_title"],
        "main_title_visible": bool(data["main_title_visible"]),
        "confidence": float(data["confidence"]),
    }


def call_vlm_once(env: dict, full_b64: str, crop_b64: str, timeout: int) -> dict:
    payload = build_payload(env["model"], full_b64, crop_b64)
    auth = HTTPBasicAuth(env["user"], env["password"]) if env["user"] else None
    resp = requests.post(
        f"{env['url']}/api/chat",
        json=payload,
        auth=auth,
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    content = body.get("message", {}).get("content", "")
    return parse_response(content)


def call_vlm_with_retries(
    env: dict,
    full_b64: str,
    crop_b64: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep_fn=time.sleep,
) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return call_vlm_once(env, full_b64, crop_b64, timeout)
        except (requests.exceptions.RequestException, VLMResponseError) as e:
            last_exc = e
            if attempt < max_retries - 1:
                sleep_fn(backoff_seconds * (2 ** attempt))
    raise RuntimeError(f"VLM 呼叫失敗，重試 {max_retries} 次仍失敗: {last_exc!r}") from last_exc


class ResultCache:
    """用內容雜湊當 key 的結果快取，持久化成 jsonl，支援斷點續跑。"""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self.data[record["key"]] = record

    def get(self, key: str) -> Optional[dict]:
        return self.data.get(key)

    def put(self, record: dict) -> None:
        self.data[record["key"]] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def query_image_pair(
    full_path: Path,
    crop_path: Path,
    env: dict,
    cache: ResultCache,
    extra_meta: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> dict:
    """查一組（整張畫面, 裁切圖），先查快取，沒有才真的呼叫 VLM 並寫入快取。"""
    full_b64, crop_b64, full_bytes, crop_bytes = load_image_pair_base64(full_path, crop_path)
    key = compute_cache_key(full_bytes, crop_bytes)

    cached = cache.get(key)
    if cached is not None:
        return cached

    result = call_vlm_with_retries(
        env, full_b64, crop_b64,
        timeout=timeout, max_retries=max_retries, backoff_seconds=backoff_seconds,
    )
    record = {
        "key": key,
        "image_path": str(full_path.relative_to(ROOT).as_posix()) if full_path.is_relative_to(ROOT) else str(full_path),
        "crop_path": str(crop_path.relative_to(ROOT).as_posix()) if crop_path.is_relative_to(ROOT) else str(crop_path),
        "result": result,
    }
    if extra_meta:
        record.update(extra_meta)
    cache.put(record)
    return record


def process_events_file(
    events_path: Path,
    channel: str,
    cache: ResultCache,
    out_path: Path,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    limit: Optional[int] = None,
) -> dict:
    env = load_vlm_env()
    if not env["url"] or not env["model"]:
        print(f"[skip] {channel}: .env 缺少 VLM_URL 或 VLM_MODEL")
        return {}

    if not events_path.exists():
        print(f"[skip] {events_path} 不存在")
        return {}

    counts: dict[str, int] = {t: 0 for t in SCREEN_TYPES}
    counts["_error"] = 0
    processed = 0

    with open(events_path, "r", encoding="utf-8") as fin, open(out_path, "a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if limit is not None and processed >= limit:
                break
            event = json.loads(line)
            full_path = ROOT / event["image_path"]
            crop_path = ROOT / event["crop_path"]

            try:
                record = query_image_pair(
                    full_path, crop_path, env, cache,
                    extra_meta={
                        "channel": channel,
                        "type": event.get("type"),
                        "video_second": event.get("video_second"),
                        "clock_time": event.get("clock_time"),
                        "minute_code": event.get("minute_code"),
                    },
                    timeout=timeout, max_retries=max_retries,
                )
                screen_type = record["result"]["screen_type"]
                counts[screen_type] = counts.get(screen_type, 0) + 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                counts["_error"] += 1
                print(f"[error] {channel} second={event.get('video_second')}: {e}")

            processed += 1

    print(f"[ok] {channel}: 處理 {processed} 筆, 統計 {counts}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="VLM 客戶端：讀取 events jsonl，逐筆問 Ollama 並快取結果")
    parser.add_argument("--channels", nargs="+", default=["ftv", "tvbs", "ebc", "cts"])
    parser.add_argument("--events", type=str, default=None, help="單一 events jsonl 路徑（搭配 --channel 使用）")
    parser.add_argument("--channel", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--limit", type=int, default=None, help="只處理前 N 筆（除錯用）")
    args = parser.parse_args()

    cache = ResultCache(CACHE_PATH)

    if args.events and args.channel:
        jobs = [(args.channel, Path(args.events))]
    else:
        jobs = [(ch, OUT_DIR / f"{ch}_events.jsonl") for ch in args.channels]

    summary = {}
    for channel, events_path in jobs:
        out_path = OUT_DIR / f"{channel}_vlm.jsonl"
        counts = process_events_file(
            events_path, channel, cache, out_path,
            timeout=args.timeout, max_retries=args.max_retries, limit=args.limit,
        )
        summary[channel] = counts

    print("\n=== screen_type 統計 ===")
    for channel, counts in summary.items():
        print(f"{channel}: {counts}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
