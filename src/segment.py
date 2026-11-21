"""分段與五字標：把 vlm.py 逐秒的判讀結果整理成一串「則」，並產生最終五字標。

流程（對應 out/{channel}_vlm.jsonl 裡每一筆 {video_second, result:{screen_type,
main_title, main_title_visible, confidence}}）：

1. 廣告狀態機（遲滯判斷）：screen_type 連續 AD_ENTER_STREAK 次是 full_ad/ad_wrap
   才真正判定「進廣告」，避免單一幀誤判就整段被切斷；離開廣告必須讀到一筆
   screen_type=="news" 且 main_title_visible、non-empty 的新標題才算恢復，
   不會沿用廣告期間殘留在記憶體裡的舊標題。
2. 信心度低的讀值（confidence 低於門檻）先用「相鄰 3 幀多數決」解析，多數決也
   沒有共識就用信心度最高的當最佳猜測，並記錄到 out/low_conf.jsonl。
3. 合併同一則延續：用文字相似度（rapidfuzz）比對連續的確認標題，高於門檻判定
   同一則、低於門檻判定換則；落在中間的模糊地帶才需要 VLM 語意判斷
   （vlm_judge_fn，介面已寫好，但預設不呼叫——等全量批次跑完、不跟它搶
   GPU 時才接上）。
4. 五字標產生：呼叫純文字 VLM（介面已寫好，call_vlm_for_label 是真的實作，
   但同樣要外部明確接上才會真的打網路）生成剛好 5 個中文字的標題；驗證規則
   （長度、社會新聞要有地點、不可編造標題沒寫的內容）沒過就自動重試最多
   2 次，仍失敗就用 fallback_keyword_label 從原標題擷取關鍵字，保證一定有
   輸出。

使用方式（只做本機分段，不呼叫任何 VLM）：
    python src/segment.py --channel ftv --vlm-results out/ftv_vlm.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

from rapidfuzz import fuzz

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
LOW_CONF_LOG_PATH = OUT_DIR / "low_conf.jsonl"

AD_SCREEN_TYPES = {"full_ad", "ad_wrap"}
AD_ENTER_STREAK = 3

LOW_CONFIDENCE_THRESHOLD = 0.5
MERGE_HIGH_THRESHOLD = 85.0
MERGE_LOW_THRESHOLD = 50.0

LABEL_LENGTH = 5
LABEL_MAX_RETRIES = 2

# 粗略規則：標題含這些詞就當「社會新聞」，五字標必須帶地點
SOCIAL_NEWS_KEYWORDS = [
    "命案", "車禍", "墜", "撞", "竊盜", "搶", "詐騙", "縱火", "襲擊", "逮捕",
    "自殺", "溺", "火警", "縱火", "槍", "持刀", "毒駕", "酒駕", "肇逃", "虐",
    "爆炸", "墜樓", "走私", "偷渡", "性侵", "殺人",
]
LOCATION_SUFFIXES = ["市", "縣", "區", "鄉", "鎮", "村", "路", "街", "站", "橋", "港", "校", "院", "局", "場"]

# 常見縣市簡稱，就算標題/五字標沒帶「市/縣」後綴（例如「台北」而非「台北市」）也要認得
TAIWAN_PLACE_NAMES = [
    "台北", "新北", "桃園", "台中", "台南", "高雄", "基隆", "新竹", "苗栗",
    "彰化", "南投", "雲林", "嘉義", "屏東", "宜蘭", "花蓮", "台東", "澎湖",
    "金門", "連江", "汐止", "三重", "板橋", "中壢",
]


def default_similarity(a: str, b: str) -> float:
    """0~100，100 代表完全相同。"""
    if not a and not b:
        return 100.0
    return fuzz.ratio(a, b)


# ---------- 1. 廣告狀態機（遲滯判斷） ----------

def run_state_machine(frames: list[dict], ad_enter_streak: int = AD_ENTER_STREAK) -> list[dict]:
    """frames 需按 video_second 由小到大排序，每筆至少有
    video_second / screen_type / main_title / main_title_visible / confidence。

    回傳狀態事件序列：
      {"video_second":, "type": "ad_start"}
      {"video_second":, "type": "headline", "title":, "confidence":}
    """
    state = "news"
    ad_streak = 0
    events: list[dict] = []

    for f in frames:
        is_ad_like = f["screen_type"] in AD_SCREEN_TYPES
        has_fresh_title = (
            f["screen_type"] == "news"
            and f.get("main_title_visible")
            and bool(f.get("main_title"))
        )

        if state == "news":
            if is_ad_like:
                ad_streak += 1
                if ad_streak >= ad_enter_streak:
                    state = "ad"
                    events.append({"video_second": f["video_second"], "type": "ad_start"})
            else:
                ad_streak = 0
                if has_fresh_title:
                    events.append({
                        "video_second": f["video_second"], "type": "headline",
                        "title": f["main_title"], "confidence": f.get("confidence", 1.0),
                    })
        else:  # state == "ad"
            if is_ad_like:
                ad_streak = ad_enter_streak  # 留在廣告，不用再累積
            elif has_fresh_title:
                # 回新聞：必須是這一幀真的讀到新標題，不可沿用廣告期間殘留的文字
                state = "news"
                ad_streak = 0
                events.append({
                    "video_second": f["video_second"], "type": "headline",
                    "title": f["main_title"], "confidence": f.get("confidence", 1.0),
                })
            # 其餘（promo/other/讀不到標題）視為雜訊，繼續當作還在廣告，不急著跳出

    return events


# ---------- 2. 低信心度 -> 相鄰 3 幀多數決 ----------

def _group_by_similarity(
    items: list[tuple[str, float]],
    similarity_fn: Callable[[str, str], float] = default_similarity,
    threshold: float = MERGE_HIGH_THRESHOLD,
) -> list[dict]:
    """把 (text, confidence) 依相似度分組。回傳 [{"text":代表字串(組內信心度最高者),
    "count":, "avg_confidence":, "members":[...]}, ...]，未排序。"""
    groups: list[dict] = []
    for text, conf in items:
        placed = False
        for g in groups:
            if similarity_fn(g["text"], text) >= threshold:
                g["members"].append((text, conf))
                if conf > g["_best_conf"]:
                    g["text"] = text
                    g["_best_conf"] = conf
                placed = True
                break
        if not placed:
            groups.append({"text": text, "members": [(text, conf)], "_best_conf": conf})

    for g in groups:
        g["count"] = len(g["members"])
        g["avg_confidence"] = sum(c for _, c in g["members"]) / len(g["members"])
        del g["_best_conf"]
    return groups


def resolve_low_confidence(
    target: dict,
    neighbors: list[dict],
    similarity_fn: Callable[[str, str], float] = default_similarity,
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    merge_threshold: float = MERGE_HIGH_THRESHOLD,
) -> dict:
    """target/neighbors 都是 {"video_second":, "title":, "confidence":}。
    信心度不低就原樣通過；信心度低就用 target+neighbors（通常共 3 幀）多數決，
    多數決也沒有共識（沒有任何一組 >=2 票）就用信心度最高的當最佳猜測，
    並標記 needs_log=True（呼叫端要負責寫進 out/low_conf.jsonl）。"""
    if target.get("confidence", 1.0) >= confidence_threshold:
        return {**target, "resolution": "high_confidence", "needs_log": False}

    candidates = [target] + list(neighbors)
    groups = _group_by_similarity(
        [(c["title"], c.get("confidence", 0.0)) for c in candidates],
        similarity_fn, threshold=merge_threshold,
    )
    groups.sort(key=lambda g: (g["count"], g["avg_confidence"]), reverse=True)

    if groups and groups[0]["count"] >= 2:
        chosen = groups[0]
        return {
            "video_second": target["video_second"],
            "title": chosen["text"],
            "confidence": chosen["avg_confidence"],
            "resolution": "majority_vote",
            "needs_log": False,
        }

    best = max(candidates, key=lambda c: c.get("confidence", 0.0))
    return {
        "video_second": target["video_second"],
        "title": best["title"],
        "confidence": best.get("confidence", 0.0),
        "resolution": "best_guess_fallback",
        "needs_log": True,
    }


def log_low_confidence(record: dict, path: Path = LOW_CONF_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_neighbor_window(frames: list[dict], index: int, window: int = 1) -> list[dict]:
    """取 frames[index] 前後各 window 筆（預設前後各1，共3幀），邊界超出就略過。"""
    lo = max(0, index - window)
    hi = min(len(frames), index + window + 1)
    return [frames[i] for i in range(lo, hi) if i != index]


# ---------- 3. 合併同一則延續 ----------

def merge_continuations(
    headline_events: list[dict],
    similarity_fn: Callable[[str, str], float] = default_similarity,
    vlm_judge_fn: Optional[Callable[[str, str], bool]] = None,
    high_threshold: float = MERGE_HIGH_THRESHOLD,
    low_threshold: float = MERGE_LOW_THRESHOLD,
) -> list[dict]:
    """headline_events 是 run_state_machine（經過低信心度解析後）得到的確認標題序列，
    依序合併成「則」。回傳 [{"start_second":, "end_second":, "titles":[...],
    "representative_title":, "merged_count":, "ambiguous_next": {...} | None}, ...]。

    相似度 >= high_threshold 判定同一則；<= low_threshold 判定換則；介於中間時，
    有給 vlm_judge_fn 就呼叫它做語意判斷，沒有的話保守地當作換則，並在前一段
    記下 ambiguous_next，標記這個邊界之後有空/ VLM 上線時應該重新覆核。"""
    if not headline_events:
        return []

    def new_segment(ev: dict) -> dict:
        return {
            "start_second": ev["video_second"],
            "end_second": ev["video_second"],
            "titles": [ev["title"]],
            "confidences": [ev.get("confidence", 1.0)],
            "ambiguous_next": None,
        }

    segments: list[dict] = []
    current = new_segment(headline_events[0])

    for ev in headline_events[1:]:
        last_title = current["titles"][-1]
        score = similarity_fn(last_title, ev["title"])

        if score >= high_threshold:
            same = True
        elif score <= low_threshold:
            same = False
        elif vlm_judge_fn is not None:
            same = vlm_judge_fn(last_title, ev["title"])
        else:
            same = False
            current["ambiguous_next"] = {
                "other_title": ev["title"], "other_second": ev["video_second"], "score": score,
            }

        if same:
            current["titles"].append(ev["title"])
            current["confidences"].append(ev.get("confidence", 1.0))
            current["end_second"] = ev["video_second"]
        else:
            segments.append(_finalize_segment(current, similarity_fn))
            current = new_segment(ev)

    segments.append(_finalize_segment(current, similarity_fn))
    return segments


def _finalize_segment(segment: dict, similarity_fn: Callable[[str, str], float]) -> dict:
    groups = _group_by_similarity(
        list(zip(segment["titles"], segment["confidences"])), similarity_fn, threshold=MERGE_HIGH_THRESHOLD,
    )
    groups.sort(key=lambda g: (g["count"], g["avg_confidence"]), reverse=True)
    representative = groups[0]["text"] if groups else segment["titles"][0]
    return {
        "start_second": segment["start_second"],
        "end_second": segment["end_second"],
        "titles": segment["titles"],
        "representative_title": representative,
        "merged_count": len(segment["titles"]),
        "avg_confidence": sum(segment["confidences"]) / len(segment["confidences"]),
        "ambiguous_next": segment["ambiguous_next"],
    }


# ---------- 4. 五字標產生 ----------

def looks_like_social_news(title: str) -> bool:
    return any(k in title for k in SOCIAL_NEWS_KEYWORDS)


def extract_location_tokens(text: str) -> list[str]:
    """抓文字裡像地名的片段：(1) 常見縣市簡稱直接比對 (2) 地名後綴字往前取2個字。
    簡單規則版，非精確 NER。"""
    tokens = [name for name in TAIWAN_PLACE_NAMES if name in text]
    for i, ch in enumerate(text):
        if ch in LOCATION_SUFFIXES and i >= 1:
            tokens.append(text[max(0, i - 1):i + 1])
    return tokens


def validate_five_char_label(label: str, original_title: str) -> list[str]:
    """回傳違反的規則清單；空清單代表通過。"""
    violations = []

    if len(label) != LABEL_LENGTH:
        violations.append(f"長度不是{LABEL_LENGTH}個字（實際{len(label)}字：{label!r}）")

    if looks_like_social_news(original_title):
        label_locations = extract_location_tokens(label)
        if not label_locations:
            violations.append("社會新聞五字標缺少地點")
        elif not any(loc in original_title for loc in label_locations):
            violations.append("五字標的地點沒有出現在原標題，疑似編造")

    if label:
        overlap = sum(1 for ch in label if ch in original_title)
        if overlap / len(label) < 0.5:
            violations.append("五字標與原標題字元重疊過低，疑似編造")

    return violations


def fallback_keyword_label(title: str, length: int = LABEL_LENGTH) -> str:
    """決定性退路（不呼叫 VLM）：從原標題去掉標點符號後取前 length 個字，
    保證一定有輸出，不受 VLM 可用性影響。"""
    strip_chars = set('!?！？"\'“”.,，。、:：;；()（）- /\\\n\t ')
    cleaned = "".join(ch for ch in title if ch not in strip_chars)
    if not cleaned:
        return title[:length].ljust(length, "　")
    if len(cleaned) < length:
        return cleaned.ljust(length, "　")
    return cleaned[:length]


def generate_five_char_label(
    title: str,
    vlm_text_fn: Callable[[str, int, Optional[list]], str],
    max_retries: int = LABEL_MAX_RETRIES,
) -> dict:
    """vlm_text_fn(title, attempt, previous_violations) -> 候選五字標字串。
    介面已就緒，但呼叫端要自己決定什麼時候真的接上會打網路的實作（見
    call_vlm_for_label），不要在批次抽幀/偵測還在跑、搶 GPU 資源時呼叫。"""
    attempts = []
    for attempt in range(max_retries + 1):
        previous_violations = attempts[-1]["violations"] if attempts else None
        label = vlm_text_fn(title, attempt, previous_violations)
        violations = validate_five_char_label(label, title)
        attempts.append({"label": label, "violations": violations})
        if not violations:
            return {"label": label, "attempts": attempts, "fallback_used": False}

    fallback = fallback_keyword_label(title)
    return {"label": fallback, "attempts": attempts, "fallback_used": True}


LABEL_PROMPT_TEMPLATE = """你是新聞監看五字標產生器。根據下面的新聞標題，產生一個「五字標」摘要：

新聞標題：{title}

規則：
- 剛好 5 個中文字（不要標點、不要英數字，除非原標題本身就是英數字內容）。
- 只能用原標題裡實際出現過的地點、人物、動作來摘要，不可以自己編造標題沒寫的地點或動作。
- 如果是社會新聞（車禍、刑案、意外等），五字標要帶地點。
{retry_note}
只回傳五字標本身，不要加任何說明文字。"""


def call_vlm_for_label(
    title: str,
    attempt: int,
    previous_violations: Optional[list],
    env: dict,
    model: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """真正會打網路的純文字 VLM 呼叫（給 generate_five_char_label 當 vlm_text_fn 用）。
    這裡只是把介面寫好、放進模組裡方便 import，實際要不要呼叫由上層決定——
    全量批次還在跑、要保留 GPU 資源給它時，不要在這裡接上這個函式。"""
    import requests
    from requests.auth import HTTPBasicAuth

    retry_note = ""
    if previous_violations:
        retry_note = f"（上一次產生的結果不合格，原因：{'; '.join(previous_violations)}，請修正）\n"
    prompt = LABEL_PROMPT_TEMPLATE.format(title=title, retry_note=retry_note)

    payload = {
        "model": model or env.get("model"),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 30},
    }
    auth = HTTPBasicAuth(env["user"], env["password"]) if env.get("user") else None
    resp = requests.post(f"{env['url']}/api/chat", json=payload, auth=auth, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    content = body.get("message", {}).get("content", "")
    return content.strip()


JUDGE_PROMPT_TEMPLATE = """你是新聞監看助手。下面兩個「標題讀值」是從同一台新聞畫面上不同時間點的
OCR/VLM 讀值，文字可能因為辨識誤差有些微不同。請判斷這兩個讀值說的是不是同一件新聞事件
（同一則新聞斷斷續續讀到的不同次結果），而不是兩個不同的新聞事件。

標題A：{title_a}
標題B：{title_b}

如果是同一件新聞事件，回答 same；如果是兩個不同的新聞事件，回答 different。
只回傳 same 或 different，不要加任何其他文字。"""


def parse_judge_response(content: str) -> bool:
    """把 call_vlm_judge_same_story 的原始回應文字解析成布林值。
    無法判斷時保守回傳 False（不合併，維持 merge_continuations 原本沒有
    vlm_judge_fn 時的保守行為）。"""
    normalized = content.strip().lower()
    if normalized.startswith("same"):
        return True
    if normalized.startswith("different"):
        return False
    if "same" in normalized and "different" not in normalized:
        return True
    return False


def call_vlm_judge_same_story(
    title_a: str,
    title_b: str,
    env: dict,
    model: Optional[str] = None,
    timeout: int = 60,
) -> bool:
    """真正會打網路的純文字 VLM 呼叫，判斷兩個標題讀值是不是同一則新聞
    （給 merge_continuations 當 vlm_judge_fn 用）。"""
    import requests
    from requests.auth import HTTPBasicAuth

    prompt = JUDGE_PROMPT_TEMPLATE.format(title_a=title_a, title_b=title_b)
    payload = {
        "model": model or env.get("model"),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 10},
    }
    auth = HTTPBasicAuth(env["user"], env["password"]) if env.get("user") else None
    resp = requests.post(f"{env['url']}/api/chat", json=payload, auth=auth, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    content = body.get("message", {}).get("content", "")
    return parse_judge_response(content)


# ---------- 整合：單一頻道的分段流程（本機、不呼叫任何 VLM） ----------

def load_vlm_results(path: Path) -> list[dict]:
    """讀 out/{channel}_vlm.jsonl，攤平成 run_state_machine 需要的欄位，依秒數排序。"""
    frames = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        result = rec.get("result", {})
        frames.append({
            "video_second": rec["video_second"],
            "minute_code": rec.get("minute_code"),
            "screen_type": result.get("screen_type"),
            "main_title": result.get("main_title", ""),
            "main_title_visible": result.get("main_title_visible", False),
            "confidence": result.get("confidence", 1.0),
        })
    frames.sort(key=lambda f: f["video_second"])
    return frames


def _resolve_headline_events(
    channel: str,
    frames: list[dict],
    headline_events: list[dict],
    confidence_threshold: float,
    low_conf_log_path: Path,
) -> list[dict]:
    frame_index_by_second = {f["video_second"]: i for i, f in enumerate(frames)}
    resolved_events = []
    for ev in headline_events:
        idx = frame_index_by_second[ev["video_second"]]
        neighbor_frames = get_neighbor_window(frames, idx, window=1)
        neighbors = [
            {"video_second": f["video_second"], "title": f["main_title"], "confidence": f["confidence"]}
            for f in neighbor_frames if f["main_title"]
        ]
        resolved = resolve_low_confidence(
            {"video_second": ev["video_second"], "title": ev["title"], "confidence": ev["confidence"]},
            neighbors, confidence_threshold=confidence_threshold,
        )
        if resolved["needs_log"]:
            log_low_confidence({"channel": channel, **resolved}, path=low_conf_log_path)
        resolved_events.append({
            "video_second": resolved["video_second"], "title": resolved["title"],
            "confidence": resolved["confidence"],
        })
    return resolved_events


def build_segments_for_channel(
    channel: str,
    vlm_results_path: Path,
    vlm_judge_fn: Optional[Callable[[str, str], bool]] = None,
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    low_conf_log_path: Path = LOW_CONF_LOG_PATH,
) -> list[dict]:
    """本機分段流程：讀 vlm 結果 -> 狀態機 -> 低信心度多數決 -> 合併同一則。
    不呼叫任何 VLM（vlm_judge_fn 預設 None，五字標生成要另外呼叫
    generate_five_char_label）。"""
    frames = load_vlm_results(vlm_results_path)
    raw_events = run_state_machine(frames)
    headline_events = [e for e in raw_events if e["type"] == "headline"]
    resolved_events = _resolve_headline_events(
        channel, frames, headline_events, confidence_threshold, low_conf_log_path,
    )
    return merge_continuations(resolved_events, vlm_judge_fn=vlm_judge_fn)


def build_fill_events(
    channel: str,
    vlm_results_path: Path,
    vlm_judge_fn: Optional[Callable[[str, str], bool]] = None,
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    low_conf_log_path: Path = LOW_CONF_LOG_PATH,
) -> list[dict]:
    """給 fill.py 用：把「新聞則」跟「進廣告」合併成一條依時間排序的填表事件清單。
    每筆 {"video_second":, "minute_code":, "kind": "news"|"ad", "label":,
    "confidence":, "ambiguous_next":}。
    kind=="ad" 的 label 固定是「廣告」，不需要呼叫任何 VLM；kind=="news" 的
    label 是合併後的代表標題（還不是五字標，fill.py 要另外呼叫
    generate_five_char_label 轉成 5 個字）。"""
    frames = load_vlm_results(vlm_results_path)
    minute_by_second = {f["video_second"]: f["minute_code"] for f in frames}

    raw_events = run_state_machine(frames)
    headline_events = [e for e in raw_events if e["type"] == "headline"]
    resolved_events = _resolve_headline_events(
        channel, frames, headline_events, confidence_threshold, low_conf_log_path,
    )
    segments = merge_continuations(resolved_events, vlm_judge_fn=vlm_judge_fn)

    fill_events = []
    for seg in segments:
        fill_events.append({
            "video_second": seg["start_second"],
            "minute_code": minute_by_second.get(seg["start_second"]),
            "kind": "news",
            "label": seg["representative_title"],
            "confidence": seg["avg_confidence"],
            "ambiguous_next": seg["ambiguous_next"],
        })
    for ev in raw_events:
        if ev["type"] == "ad_start":
            fill_events.append({
                "video_second": ev["video_second"],
                "minute_code": minute_by_second.get(ev["video_second"]),
                "kind": "ad",
                "label": "廣告",
                "confidence": 1.0,
                "ambiguous_next": None,
            })

    fill_events.sort(key=lambda e: e["video_second"])
    return fill_events


def main() -> int:
    parser = argparse.ArgumentParser(description="分段：把 vlm.py 的逐秒判讀結果整理成一串「則」（本機運算，不呼叫 VLM）")
    parser.add_argument("--channel", required=True)
    parser.add_argument("--vlm-results", type=str, default=None, help="預設 out/{channel}_vlm.jsonl")
    args = parser.parse_args()

    vlm_results_path = Path(args.vlm_results) if args.vlm_results else OUT_DIR / f"{args.channel}_vlm.jsonl"
    if not vlm_results_path.exists():
        print(f"[skip] {vlm_results_path} 不存在")
        return 1

    segments = build_segments_for_channel(args.channel, vlm_results_path)
    print(f"[ok] {args.channel}: {len(segments)} 個則（合併後）")
    for seg in segments:
        ambiguous = " [待覆核]" if seg["ambiguous_next"] else ""
        print(f"  {seg['start_second']:>5}-{seg['end_second']:<5} "
              f"x{seg['merged_count']:<2} {seg['representative_title']}{ambiguous}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
