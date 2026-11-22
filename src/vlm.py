"""VLM 客戶端：呼叫 Colab 上的 Ollama /api/chat，判讀單一秒的畫面。

對每一筆輸入（整張畫面 + headline_bar 裁切圖）送出一次請求，用 JSON schema
強制回傳：
    {screen_type: news|ad_wrap|full_ad|promo|other, main_title, main_title_visible, confidence}

screen_type 五選一：
    news     - 正常新聞畫面，看得到主標題 bar
    ad_wrap  - 左側大畫面播廣告、右側小畫面繼續推播新聞（例如「現在最焦點」）
    full_ad  - 整個畫面都是廣告，沒有新聞內容
    promo    - 本台自家宣傳（節目預告、台呼等），不是新聞也不是外部廣告
    other    - 以上皆非（轉場、黑畫面等）

效能優化（相對於第一版）：
1. 計時拆解：每次呼叫記錄網路往返時間 + Ollama 回傳的
   total_duration/load_duration/prompt_eval_*/eval_* 到 out/logs/vlm_timing.jsonl。
2. 請求參數：think=false（關掉 gemma4 的思考模式）、num_ctx=4096、
   num_predict=200、temperature=0，減少不必要的生成長度與思考開銷。
3. 整張畫面縮到 768 寬、JPEG 品質 80；bar 裁切圖保留原解析度（讓 VLM 看清楚文字）。
4. 本機先用 phash 做兩層過濾，沒變化就不送網路：
   - 心跳幀：跟上一次「送過 VLM」的心跳整體雜湊差不多就沿用上次結果。
   - 事件：跟上一次「送過 VLM」的事件 bar 雜湊幾乎相同（卡在 detect.py
     門檻邊界的殘留重複）就沿用上次結果。
5. 4 個頻道用 ThreadPoolExecutor 並行處理，共用同一份快取/計時紀錄（皆上鎖）。

附重試／逾時／結果快取（同一張圖的內容雜湊不會重送）；快取寫在
out/vlm_cache.jsonl，中斷後重跑會自動跳過已經問過的圖，達成斷點續跑。

使用方式：
    python src/vlm.py --events out/ftv_events.jsonl --channel ftv
    python src/vlm.py --channels ftv tvbs ebc cts         # 4 台並行處理
    python src/vlm.py --estimate-only                      # 只印本機去重後的筆數估計，不打網路
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tools"))
import detect  # noqa: E402  (重用 compute_phash / crop_roi)
import extract_frame  # noqa: E402  (降級鏈：抓相鄰幀)
import calibrate  # noqa: E402  (降級鏈：讀 ROI 設定)

OUT_DIR = ROOT / "out"
LOGS_DIR = OUT_DIR / "logs"
CACHE_PATH = OUT_DIR / "vlm_cache.jsonl"
TIMING_PATH = LOGS_DIR / "vlm_timing.jsonl"
DEGRADED_LOG_PATH = LOGS_DIR / "vlm_degraded.jsonl"

HEALTH_CHECK_TIMEOUT = 10
HEALTH_POLL_INTERVAL = 30
CONSECUTIVE_FAILURE_HEALTH_CHECK = 3

DEFAULT_TIMEOUT = 180  # 秒；Colab 冷啟動 + CPU/GPU 推論可能偏慢
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 2

FULL_IMAGE_MAX_WIDTH = 768
FULL_IMAGE_JPEG_QUALITY = 80
CROP_JPEG_QUALITY = 95  # 裁切圖保留原解析度+高品質，避免文字模糊

DEFAULT_NUM_CTX = 4096
DEFAULT_NUM_PREDICT = 200
DEFAULT_TEMPERATURE = 0
DEFAULT_THINK = False

HEARTBEAT_REUSE_THRESHOLD = 10  # 心跳整體雜湊：差不多（<=10）就沿用上次結果
EVENT_DEDUP_THRESHOLD = 20      # 事件 bar 雜湊：幾乎相同（<=20）才當重複跳過

# 每次改動 PROMPT 或 RESPONSE_SCHEMA 都要把這個版本號加 1：快取 key 會把它算進去，
# 確保 prompt 改了之後舊的快取結果不會被誤用（cache key 只看圖片內容，不看 prompt）。
PROMPT_VERSION = 3

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

判斷規則（很重要，請依序檢查，不要跳步）：

1. 什麼是「主標題」：通常是一整條有底色的粗體字（例如整條黃底、整條深色底），或是疊在棚內美術圖卡上的大型醒目文字。內容性質是在**描述「這則新聞發生了什麼事」**（例如地點+事件，像「台北市車禍一人送醫」「台股盤中重挫」），是對這則新聞的摘要，不是任何人講的一句完整的話。

2. 下面這些**都不是主標題**，就算畫面上看得到，也絕對不要填進 main_title、也不要讓它們影響你的判斷：
   - 最下面持續捲動的跑馬燈。
   - 另外的「快訊」色塊、角落的天氣小部件。
   - 受訪者/專家的「口白字幕」：通常是一個白底或淺色底的小方塊，上面一行是人名跟職稱（例如「消防專家 林金宏」、「當事民眾(聲源)」），下面接著一句話——**這句話是「某人說的話」（引用句、第一人稱語氣），不是在描述新聞事件本身**，不管這句話看起來多完整、多像一句標題，只要它是某人說話的原音字幕，就不是主標題。

3. **如果畫面上找不到符合第 1 點描述的主標題**（例如：當下只看得到口白字幕、只看得到跑馬燈、標題欄位是空的、或只顯示台名），main_title 請填空字串 ""，main_title_visible 請填 false。**寧可回答「沒有標題」，也絕對不要抓跑馬燈或口白字幕來湊數**——這點非常重要，請嚴格遵守，即使你覺得畫面上「總該有點什麼可以當標題」，只要不符合第 1 點的描述，就是沒有標題。

4. 如果節目當下主畫面播的就是天氣報告（天氣佔據主標題/主畫面，不是角落小部件），那就正常把天氣相關文字當作 main_title，不適用第 3 點的「沒有標題」。

5. 同一則新聞可能會經過「主播稿頭→記者報導(SOT)→連線」等不同畫面，只要主標題文字沒變，都算同一則，請照實回報目前畫面上的文字，不用推測是否為同一則。

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


# ---------- 圖片處理 ----------

def resize_for_full(image_bgr: np.ndarray, max_width: int = FULL_IMAGE_MAX_WIDTH) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if width <= max_width:
        return image_bgr
    scale = max_width / width
    return cv2.resize(image_bgr, (max_width, int(round(height * scale))))


def image_to_base64(image_bgr: np.ndarray, quality: int = 95) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("無法編碼圖片")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def load_image_pair_base64(
    full_path: Path,
    crop_path: Path,
    full_max_width: int = FULL_IMAGE_MAX_WIDTH,
    full_quality: int = FULL_IMAGE_JPEG_QUALITY,
    crop_quality: int = CROP_JPEG_QUALITY,
) -> tuple[str, str, bytes, bytes]:
    """讀取整張畫面 + 裁切圖。整張畫面縮到 full_max_width 寬、quality 壓縮；
    裁切圖保留原解析度，只套用 crop_quality。回傳
    (full_b64, crop_b64, full_bytes_for_hash, crop_bytes_for_hash)。"""
    full_img = cv2.imread(str(full_path))
    crop_img = cv2.imread(str(crop_path))
    if full_img is None:
        raise FileNotFoundError(f"讀不到整張畫面: {full_path}")
    if crop_img is None:
        raise FileNotFoundError(f"讀不到裁切圖: {crop_path}")

    full_img = resize_for_full(full_img, max_width=full_max_width)
    full_b64 = image_to_base64(full_img, quality=full_quality)
    crop_b64 = image_to_base64(crop_img, quality=crop_quality)

    full_bytes = full_path.read_bytes()
    crop_bytes = crop_path.read_bytes()
    return full_b64, crop_b64, full_bytes, crop_bytes


def compute_cache_key(full_bytes: bytes, crop_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(full_bytes)
    h.update(crop_bytes)
    h.update(f"prompt_v{PROMPT_VERSION}".encode("utf-8"))
    return h.hexdigest()


# ---------- 請求 / 回應 ----------

def build_payload(
    model: str,
    full_b64: str,
    crop_b64: str,
    think: bool = DEFAULT_THINK,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
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
        **({"think": think} if think is not None else {}),
        "options": {
            "temperature": temperature,
            **({"num_ctx": num_ctx} if num_ctx is not None else {}),
            **({"num_predict": num_predict} if num_predict is not None else {}),
        },
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


def extract_timing(body: dict) -> dict:
    """從 Ollama /api/chat 回應本體取出耗時欄位，奈秒轉成秒。"""
    def ns_to_s(v):
        return v / 1e9 if isinstance(v, (int, float)) else None

    return {
        "total_duration_s": ns_to_s(body.get("total_duration")),
        "load_duration_s": ns_to_s(body.get("load_duration")),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "prompt_eval_duration_s": ns_to_s(body.get("prompt_eval_duration")),
        "eval_count": body.get("eval_count"),
        "eval_duration_s": ns_to_s(body.get("eval_duration")),
    }


TIMING_NUMERIC_FIELDS = [
    "wall_seconds", "total_duration_s", "load_duration_s",
    "prompt_eval_count", "prompt_eval_duration_s",
    "eval_count", "eval_duration_s",
]


def call_vlm_once(env: dict, full_b64: str, crop_b64: str, timeout: int, **payload_kwargs) -> tuple[dict, dict]:
    payload = build_payload(env["model"], full_b64, crop_b64, **payload_kwargs)
    auth = HTTPBasicAuth(env["user"], env["password"]) if env["user"] else None
    start = time.perf_counter()
    resp = requests.post(
        f"{env['url']}/api/chat",
        json=payload,
        auth=auth,
        timeout=timeout,
    )
    wall_seconds = time.perf_counter() - start
    resp.raise_for_status()
    body = resp.json()
    content = body.get("message", {}).get("content", "")
    result = parse_response(content)

    timing = {"wall_seconds": wall_seconds, "success": True, "error": None}
    timing.update(extract_timing(body))
    return result, timing


# requests.exceptions.ConnectionError / Timeout 代表「根本連不到伺服器」（通道斷線、逾時無回應）；
# 這類錯誤交給健康檢查+等待處理，絕對不能當成「這張圖的結果」快取起來。
# HTTPError（resp.raise_for_status() 丟出）代表伺服器有回應、只是狀態碼不對（如 500）；
# VLMResponseError 代表伺服器有回應、但內容不符合 schema。這兩類都歸類成 "response"，
# 交給降級鏈處理（縮圖／相鄰幀／low_conf），因為問題看起來跟這張圖或這次請求有關，不是斷線。
CONNECTION_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


class VLMCallFailed(RuntimeError):
    """call_vlm_with_retries 重試耗盡後丟出，帶著失敗分類方便上層決定下一步。"""

    def __init__(self, category: str, last_exc: Exception):
        self.category = category  # "connection" | "response"
        self.last_exc = last_exc
        super().__init__(f"VLM 呼叫失敗（分類：{category}），重試耗盡: {last_exc!r}")


def call_vlm_with_retries(
    env: dict,
    full_b64: str,
    crop_b64: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep_fn=time.sleep,
    attempts_out: Optional[list] = None,
    **payload_kwargs,
) -> dict:
    """attempts_out（若提供）會被附加每一次嘗試的計時紀錄，不論最後成功或失敗。
    重試耗盡後丟出 VLMCallFailed，帶著分類（connection/response）給上層決定後續處理。"""
    last_exc: Optional[Exception] = None
    last_category = "response"
    for attempt in range(max_retries):
        start = time.perf_counter()
        try:
            result, timing = call_vlm_once(env, full_b64, crop_b64, timeout, **payload_kwargs)
            timing["attempt"] = attempt + 1
            if attempts_out is not None:
                attempts_out.append(timing)
            return result
        except CONNECTION_EXCEPTIONS as e:
            category = "connection"
            exc: Exception = e
        except (requests.exceptions.RequestException, VLMResponseError) as e:
            category = "response"
            exc = e

        wall_seconds = time.perf_counter() - start
        if attempts_out is not None:
            attempts_out.append({
                "attempt": attempt + 1,
                "wall_seconds": wall_seconds,
                "success": False,
                "error": repr(exc),
                "error_category": category,
            })
        last_exc = exc
        last_category = category
        if attempt < max_retries - 1:
            sleep_fn(backoff_seconds * (2 ** attempt))
    raise VLMCallFailed(last_category, last_exc)


# ---------- 健康檢查 ----------

def check_health(env: dict, timeout: int = HEALTH_CHECK_TIMEOUT) -> bool:
    """呼叫 /api/tags 確認伺服器連得到、回應正常。連不到／逾時／非 200 都回 False。"""
    try:
        auth = HTTPBasicAuth(env["user"], env["password"]) if env["user"] else None
        resp = requests.get(f"{env['url']}/api/tags", auth=auth, timeout=timeout)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def reload_and_check_health(_env: Optional[dict] = None, timeout: int = HEALTH_CHECK_TIMEOUT) -> bool:
    """跟 check_health 一樣，但每次都先重新讀 .env 再檢查——這樣 ngrok 網址換了、
    使用者更新 .env 之後，不用重啟整個批次，下一次健康檢查／等待輪詢就會吃到新網址。
    接受但忽略 _env 參數，只是為了跟 check_health 一樣的呼叫介面（供 wait_until_healthy
    的 check_fn(env) 呼叫慣例相容）。"""
    return check_health(load_vlm_env(), timeout=timeout)


def wait_until_healthy(
    env: dict,
    poll_interval: float = HEALTH_POLL_INTERVAL,
    check_fn: Callable[[dict], bool] = reload_and_check_health,
    sleep_fn=time.sleep,
) -> None:
    """卡住直到健康檢查通過為止，每次等待都印「VLM 離線，等待中」。恢復後直接返回。
    預設的 check_fn 每次都會重新讀 .env（見 reload_and_check_health），所以卡住等待
    期間如果使用者更新了 VLM_URL，下一輪 30 秒輪詢就會用新網址檢查。"""
    while not check_fn(env):
        print("[health] VLM 離線，等待中...")
        sleep_fn(poll_interval)


# ---------- 降級鏈（只用在伺服器有回應、但對特定圖片持續失敗的情況） ----------

def low_conf_placeholder() -> dict:
    return {"screen_type": "other", "main_title": "", "main_title_visible": False, "confidence": 0.0}


def log_degraded(record: dict) -> None:
    DEGRADED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEGRADED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def try_degraded_chain(
    env: dict,
    channel: str,
    video_second: int,
    full_path: Path,
    crop_path: Path,
    roi_config: dict,
    timeout: int = DEFAULT_TIMEOUT,
    attempts_out: Optional[list] = None,
    call_fn: Callable = call_vlm_once,
) -> tuple[dict, str]:
    """依序試：(1) 把整張畫面縮更小再送一次 (2) 改用相鄰 +1/-1 秒的畫面
    (3) 都不行就回傳 low_conf 佔位結果。回傳 (result, degraded_via)。
    只應該在「伺服器有回應、但這次/這張圖一直失敗」時呼叫（category == "response"）。"""

    def record_attempt(tag: str, timing: dict) -> None:
        if attempts_out is not None:
            t = dict(timing)
            t["attempt"] = tag
            attempts_out.append(t)

    # (1) 縮更小的圖再試一次
    try:
        full_img = cv2.imread(str(full_path))
        crop_img = cv2.imread(str(crop_path))
        if full_img is not None and crop_img is not None:
            small_full_b64 = image_to_base64(
                resize_for_full(full_img, max_width=FULL_IMAGE_MAX_WIDTH // 2), quality=60
            )
            crop_b64 = image_to_base64(crop_img, quality=CROP_JPEG_QUALITY)
            result, timing = call_fn(env, small_full_b64, crop_b64, timeout)
            record_attempt("degraded_shrink", timing)
            return result, "shrink"
    except Exception:
        pass

    # (2) 相鄰幀：先 +1 秒，再 -1 秒
    headline_roi = roi_config.get(channel, {}).get("headline_bar")
    if headline_roi:
        video_path = detect.VIDEOS_DIR / f"{channel}.mp4"
        for offset in (1, -1):
            try:
                adjacent_second = max(0, video_second + offset)
                frame, _actual = extract_frame.extract_frame(video_path, adjacent_second)
                if frame is None:
                    continue
                crop = detect.crop_roi(frame, headline_roi)
                full_b64 = image_to_base64(resize_for_full(frame), quality=FULL_IMAGE_JPEG_QUALITY)
                crop_b64 = image_to_base64(crop, quality=CROP_JPEG_QUALITY)
                result, timing = call_fn(env, full_b64, crop_b64, timeout)
                record_attempt(f"degraded_adjacent_{offset:+d}s", timing)
                return result, f"adjacent_{offset:+d}s"
            except Exception:
                continue

    # (3) 放棄：回傳低信心佔位結果，並記錄到 out/logs/vlm_degraded.jsonl 供人工追查
    log_degraded({
        "channel": channel, "video_second": video_second,
        "image_path": str(full_path), "crop_path": str(crop_path),
    })
    return low_conf_placeholder(), "low_conf_fallback"


# ---------- 快取 / 計時紀錄（皆執行緒安全） ----------

class ResultCache:
    """用內容雜湊當 key 的結果快取，持久化成 jsonl，支援斷點續跑。"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
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
        with self._lock:
            return self.data.get(key)

    def put(self, record: dict) -> None:
        with self._lock:
            self.data[record["key"]] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


class TimingRecorder:
    """記錄每次實際網路呼叫的耗時拆解，持久化成 jsonl，可印平均值。"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.records: list[dict] = []

    def record(self, rec: dict) -> None:
        with self._lock:
            self.records.append(rec)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        with self._lock:
            records = list(self.records)
        successes = [r for r in records if r.get("success")]
        out: dict = {"n": len(successes), "n_failed": len(records) - len(successes)}
        for field in TIMING_NUMERIC_FIELDS:
            vals = [r[field] for r in successes if r.get(field) is not None]
            out[field] = sum(vals) / len(vals) if vals else None
        return out


def query_image_pair(
    full_path: Path,
    crop_path: Path,
    env: dict,
    cache: ResultCache,
    timing: Optional[TimingRecorder] = None,
    extra_meta: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    channel: Optional[str] = None,
    video_second: Optional[int] = None,
    roi_config: Optional[dict] = None,
    wait_fn: Callable[[dict], None] = wait_until_healthy,
    **payload_kwargs,
) -> dict:
    """查一組（整張畫面, 裁切圖），先查快取，沒有才真的呼叫 VLM 並寫入快取。

    失敗分類後的處理：
    - "connection"（連不到伺服器）：呼叫 wait_fn 卡住直到健康檢查恢復，
      恢復後重新打一輪完整的重試（不放棄、不當成結果）。
    - "response"（伺服器有回應但失敗，如 5xx／內容不符 schema）：交給
      try_degraded_chain 依序嘗試縮圖／相鄰幀／low_conf 佔位。
    """
    full_b64, crop_b64, full_bytes, crop_bytes = load_image_pair_base64(full_path, crop_path)
    key = compute_cache_key(full_bytes, crop_bytes)

    cached = cache.get(key)
    if cached is not None:
        # 快取命中只代表「這張圖的 VLM 結果」可以重用；呼叫端這次傳進來的
        # extra_meta（channel/video_second/clock_time/minute_code...）才是
        # 這次呼叫真正的上下文，永遠要覆蓋快取裡凍結住的舊 metadata，
        # 不然同一張圖從不同情境（或沒帶 extra_meta）查過一次後，
        # 之後所有呼叫都會拿到第一次查詢當下的舊 metadata。
        record = dict(cached)
        if extra_meta:
            record.update(extra_meta)
        return record

    all_attempts: list[dict] = []
    degraded_via: Optional[str] = None
    result: Optional[dict] = None
    try:
        while True:
            attempts: list[dict] = []
            try:
                result = call_vlm_with_retries(
                    env, full_b64, crop_b64,
                    timeout=timeout, max_retries=max_retries, backoff_seconds=backoff_seconds,
                    attempts_out=attempts, **payload_kwargs,
                )
                all_attempts.extend(attempts)
                break
            except VLMCallFailed as e:
                all_attempts.extend(attempts)
                if e.category == "connection":
                    wait_fn(env)
                    env = load_vlm_env()  # .env 可能被更新了（例如 ngrok 網址換了），重新讀取再繼續
                    continue  # 伺服器恢復後，從頭重新打一輪完整的重試
                result, degraded_via = try_degraded_chain(
                    env, channel, video_second, full_path, crop_path, roi_config or {},
                    timeout=timeout, attempts_out=all_attempts,
                )
                break
    finally:
        if timing is not None:
            for a in all_attempts:
                rec = dict(a)
                if extra_meta:
                    rec.update(extra_meta)
                timing.record(rec)

    record = {
        "key": key,
        "image_path": str(full_path.relative_to(ROOT).as_posix()) if full_path.is_relative_to(ROOT) else str(full_path),
        "crop_path": str(crop_path.relative_to(ROOT).as_posix()) if crop_path.is_relative_to(ROOT) else str(crop_path),
        "result": result,
        "prompt_version": PROMPT_VERSION,
    }
    if degraded_via:
        record["degraded_via"] = degraded_via
    if extra_meta:
        record.update(extra_meta)
    cache.put(record)
    return record


# ---------- 本機去重（不打網路） ----------

def is_similar(hash_a, hash_b, threshold: int, diff: Callable = lambda a, b: a - b) -> bool:
    if hash_a is None or hash_b is None:
        return False
    return diff(hash_a, hash_b) <= threshold


def get_dedup_phash(record: dict) -> "object":
    """心跳用整張畫面的雜湊（整體版面），事件用 bar 裁切圖的雜湊。"""
    path = record["image_path"] if record["type"] == "heartbeat" else record["crop_path"]
    img = cv2.imread(str(ROOT / path) if not Path(path).is_absolute() else path)
    if img is None:
        return None
    return detect.compute_phash(img)


def plan_calls(
    records: list[dict],
    get_phash: Callable[[dict], object] = get_dedup_phash,
    heartbeat_threshold: int = HEARTBEAT_REUSE_THRESHOLD,
    event_threshold: int = EVENT_DEDUP_THRESHOLD,
) -> list[dict]:
    """依序走過 records，決定每一筆是否真的需要呼叫 VLM（needs_call），
    或可以沿用同一條鏈（heartbeat / event）裡上一次「有呼叫」的結果（reused_index）。
    純邏輯：不碰網路，get_phash 可注入以利測試。"""
    decisions: list[dict] = []
    last_heartbeat_hash = None
    last_heartbeat_call_index: Optional[int] = None
    last_event_hash = None
    last_event_call_index: Optional[int] = None

    for i, rec in enumerate(records):
        kind = rec.get("type")
        h = get_phash(rec)

        if kind == "heartbeat":
            reuse = is_similar(h, last_heartbeat_hash, heartbeat_threshold)
            if reuse:
                decisions.append({"index": i, "needs_call": False, "reused_index": last_heartbeat_call_index})
            else:
                decisions.append({"index": i, "needs_call": True, "reused_index": None})
                last_heartbeat_hash = h
                last_heartbeat_call_index = i
        elif kind == "event":
            reuse = is_similar(h, last_event_hash, event_threshold)
            if reuse:
                decisions.append({"index": i, "needs_call": False, "reused_index": last_event_call_index})
            else:
                decisions.append({"index": i, "needs_call": True, "reused_index": None})
                last_event_hash = h
                last_event_call_index = i
        else:
            decisions.append({"index": i, "needs_call": True, "reused_index": None})

    return decisions


def estimate_savings(events_path: Path) -> dict:
    """純本機估計（不打網路）：讀 events jsonl，回傳 {total, needs_call, reused}。"""
    if not events_path.exists():
        return {"total": 0, "needs_call": 0, "reused": 0}
    records = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    decisions = plan_calls(records)
    needs_call = sum(1 for d in decisions if d["needs_call"])
    return {"total": len(records), "needs_call": needs_call, "reused": len(records) - needs_call}


# ---------- 批次處理 ----------

PROGRESS_EVERY = 100


def format_progress(channel: str, done: int, total: int, elapsed_seconds: float) -> str:
    """每處理 N 筆印一次進度，依目前平均速度估剩餘時間。"""
    rate = elapsed_seconds / done if done else 0.0
    remaining = max(total - done, 0)
    eta_seconds = remaining * rate
    eta_min = eta_seconds / 60
    return (
        f"[progress] {channel}: {done}/{total} "
        f"（{rate:.2f}秒/筆，已花 {elapsed_seconds/60:.1f} 分，預估剩餘 {eta_min:.1f} 分）"
    )


def load_done_seconds(out_path: Path, required_prompt_version: Optional[int] = PROMPT_VERSION) -> set:
    """讀取已輸出的 jsonl，取出已經處理過的 video_second，供斷點續跑跳過用。

    required_prompt_version 給定時，只有記錄裡的 prompt_version 跟現在的版本
    一致才算「已完成」；版本不符（包含舊資料根本沒有這個欄位，視為版本不符）
    一律當作還沒處理，確保改了 PROMPT 之後重跑會真的用新 prompt 重新問一次，
    不會被斷點續跑誤判成「已經做過」而跳過。required_prompt_version=None 時
    不檢查版本（舊行為，單純依 video_second 判斷）。"""
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if required_prompt_version is not None and rec.get("prompt_version") != required_prompt_version:
            continue
        done.add(rec.get("video_second"))
    return done


def process_events_file(
    events_path: Path,
    channel: str,
    cache: ResultCache,
    out_path: Path,
    timing: Optional[TimingRecorder] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    limit: Optional[int] = None,
    progress_every: int = PROGRESS_EVERY,
    wait_fn: Callable[[dict], None] = wait_until_healthy,
    health_check_fn: Callable[[dict], bool] = reload_and_check_health,
    consecutive_failure_limit: int = CONSECUTIVE_FAILURE_HEALTH_CHECK,
) -> dict:
    env = load_vlm_env()
    if not env["url"] or not env["model"]:
        print(f"[skip] {channel}: .env 缺少 VLM_URL 或 VLM_MODEL")
        return {}

    if not events_path.exists():
        print(f"[skip] {events_path} 不存在")
        return {}

    # 每次批次開始前先確認連得到伺服器，連不到就卡住等待（不當成失敗）
    if not health_check_fn(env):
        wait_fn(env)
        env = load_vlm_env()  # .env 可能被更新了（例如 ngrok 網址換了），重新讀取再繼續

    roi_config = calibrate.load_config()

    records = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        records = records[:limit]
    decisions = plan_calls(records)

    done_seconds = load_done_seconds(out_path)
    total = len(records)
    session_total = total - len(done_seconds)  # 這次執行還需要處理的筆數（扣掉斷點續跑已完成的）
    if done_seconds:
        print(f"[resume] {channel}: 偵測到 {out_path.name} 已有 {len(done_seconds)} 筆，將跳過已完成的部分")

    counts: dict[str, int] = {t: 0 for t in SCREEN_TYPES}
    counts["_error"] = 0
    counts["_reused"] = 0
    counts["_skipped_resume"] = 0
    counts["_degraded"] = 0
    results_by_index: dict[int, dict] = {}
    processed = 0
    start_time = time.perf_counter()
    consecutive_degraded = 0

    with open(out_path, "a", encoding="utf-8") as fout:
        for decision, event in zip(decisions, records):
            i = decision["index"]

            if event.get("video_second") in done_seconds:
                counts["_skipped_resume"] += 1
                continue

            full_path = ROOT / event["image_path"]
            crop_path = ROOT / event["crop_path"]
            meta = {
                "channel": channel,
                "type": event.get("type"),
                "video_second": event.get("video_second"),
                "clock_time": event.get("clock_time"),
                "minute_code": event.get("minute_code"),
            }

            if not decision["needs_call"]:
                reused = results_by_index.get(decision["reused_index"])
                if reused is not None:
                    record = dict(reused)
                    record.update(meta)
                    record["reused_from_second"] = reused.get("video_second")
                    counts["_reused"] += 1
                    counts[record["result"]["screen_type"]] = counts.get(record["result"]["screen_type"], 0) + 1
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    results_by_index[i] = record
                    processed += 1
                    if processed % progress_every == 0:
                        print(format_progress(channel, processed, session_total, time.perf_counter() - start_time))
                    continue
                # 找不到可沿用的結果（理論上不會發生），退回真的呼叫一次

            try:
                record = query_image_pair(
                    full_path, crop_path, env, cache, timing=timing,
                    extra_meta=meta, timeout=timeout, max_retries=max_retries,
                    channel=channel, video_second=event.get("video_second"),
                    roi_config=roi_config, wait_fn=wait_fn,
                )
                screen_type = record["result"]["screen_type"]
                counts[screen_type] = counts.get(screen_type, 0) + 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                results_by_index[i] = record

                if record.get("degraded_via") == "low_conf_fallback":
                    counts["_degraded"] += 1
                    consecutive_degraded += 1
                else:
                    consecutive_degraded = 0
            except Exception as e:
                counts["_error"] += 1
                consecutive_degraded += 1
                print(f"[error] {channel} second={event.get('video_second')}: {e}")

            if consecutive_degraded >= consecutive_failure_limit:
                print(f"[health] {channel}: 連續 {consecutive_degraded} 筆失敗，主動檢查伺服器狀態")
                if not health_check_fn(env):
                    wait_fn(env)
                    env = load_vlm_env()  # .env 可能被更新了（例如 ngrok 網址換了），重新讀取再繼續
                consecutive_degraded = 0

            processed += 1
            if processed % progress_every == 0:
                print(format_progress(channel, processed, session_total, time.perf_counter() - start_time))

    needs_call = sum(1 for d in decisions if d["needs_call"])
    print(
        f"[ok] {channel}: 處理 {processed} 筆（本機去重後實際呼叫 {needs_call}/{len(records)}）, 統計 {counts}"
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="VLM 客戶端：讀取 events jsonl，逐筆問 Ollama 並快取結果")
    parser.add_argument("--channels", nargs="+", default=["ftv", "tvbs", "ebc", "cts"])
    parser.add_argument("--events", type=str, default=None, help="單一 events jsonl 路徑（搭配 --channel 使用）")
    parser.add_argument("--channel", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--limit", type=int, default=None, help="只處理前 N 筆（除錯用）")
    parser.add_argument("--workers", type=int, default=4, help="並行處理的頻道數")
    parser.add_argument("--estimate-only", action="store_true", help="只印本機去重後的筆數估計，不打網路")
    parser.add_argument("--tag", type=str, default="", help="讀/寫檔名加的標記（跟 detect.py 的 --tag 對應），例如 --tag _test15 -> 讀 out/{channel}_test15_events.jsonl、寫 out/{channel}_test15_vlm.jsonl；channel 本身維持原名，降級鏈抓相鄰幀才找得到正確的影片檔")
    args = parser.parse_args()

    if args.events and args.channel:
        jobs = [(args.channel, Path(args.events))]
    else:
        jobs = [(ch, OUT_DIR / f"{ch}{args.tag}_events.jsonl") for ch in args.channels]

    if args.estimate_only:
        print("=== 本機去重估計（不打網路） ===")
        for channel, events_path in jobs:
            est = estimate_savings(events_path)
            print(f"{channel}: 原本 {est['total']} 筆 -> 實際需呼叫 {est['needs_call']} 筆（省下 {est['reused']} 筆）")
        return 0

    cache = ResultCache(CACHE_PATH)
    timing = TimingRecorder(TIMING_PATH)

    summary = {}

    def run_job(channel, events_path):
        out_path = OUT_DIR / f"{channel}{args.tag}_vlm.jsonl"
        return channel, process_events_file(
            events_path, channel, cache, out_path, timing=timing,
            timeout=args.timeout, max_retries=args.max_retries, limit=args.limit,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, ch, path) for ch, path in jobs]
        for future in as_completed(futures):
            channel, counts = future.result()
            summary[channel] = counts

    print("\n=== screen_type 統計 ===")
    for channel in [ch for ch, _ in jobs]:
        print(f"{channel}: {summary.get(channel)}")

    ts = timing.summary()
    if ts.get("n"):
        print("\n=== VLM 呼叫耗時平均（本次執行） ===")
        print(f"樣本數: {ts['n']}（失敗 {ts['n_failed']}）")
        for field in TIMING_NUMERIC_FIELDS:
            val = ts.get(field)
            print(f"  {field}: {val:.3f}" if val is not None else f"  {field}: N/A")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
