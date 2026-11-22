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
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from rapidfuzz import fuzz

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlm  # noqa: E402  (重用連線類例外分類 + 健康檢查等待，讓文字 VLM 呼叫也扛得住斷線)

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

# 常見姓氏，用來抓標題裡像人名的片段（簡化版，非精確 NER）。地點是硬性規則
# （社會新聞一定要帶），人名只是「有出現就盡量填上」的加分項，不當硬性擋關卡。
PERSON_SURNAMES = [
    "蘇", "李", "沈", "柯", "賴", "藍", "張", "韓", "王", "陳", "林", "黃", "吳",
    "劉", "楊", "周", "許", "鄭", "謝", "郭", "洪", "曾", "廖", "徐", "余", "潘",
    "杜", "馬", "方", "彭", "曹", "董", "蕭", "蔡", "康", "邱", "施", "羅", "高",
    "梁", "宋", "唐", "孫", "胡", "朱", "阮", "任", "熊", "游", "卓", "范", "顏",
    "簡", "江", "侯", "連", "錢", "尹", "魏", "盧", "白", "石", "秦", "姚", "韋",
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
            try:
                same = vlm_judge_fn(last_title, ev["title"])
            except Exception:
                # vlm_judge_fn 自己的重試／健康檢查等待已經用盡才會走到這裡；
                # 保守當作換則並標記待覆核，不要讓單一次判斷失敗拖垮整批
                same = False
                current["ambiguous_next"] = {
                    "other_title": ev["title"], "other_second": ev["video_second"], "score": score,
                }
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


def _is_chinese_char(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def extract_person_name_tokens(text: str) -> list[str]:
    """抓文字裡像人名的片段：姓氏開頭往後取 2~3 個中文字當候選。簡單規則版，
    非精確 NER，只用來做「有出現人名就盡量填進五字標」的 best-effort 判斷。"""
    tokens: list[str] = []
    for i, ch in enumerate(text):
        if ch not in PERSON_SURNAMES:
            continue
        for name_len in (3, 2):
            if i + name_len > len(text):
                continue
            candidate = text[i:i + name_len]
            if all(_is_chinese_char(c) for c in candidate):
                tokens.append(candidate)
                break
    return tokens


def missing_person_name(label: str, original_title: str) -> Optional[str]:
    """回傳原標題裡偵測到、但五字標完全沒帶到的人名候選（取第一個）；
    原標題沒有可辨識的人名、或五字標已經帶到（哪怕只有姓）就回傳 None。
    這不是 validate_five_char_label 的硬性規則——人名是「盡量」而非「一定要」，
    只用來記錄到 out/logs 供事後檢查。"""
    for name in extract_person_name_tokens(original_title):
        if any(ch in label for ch in name):
            return None
        return name
    return None


LOW_OVERLAP_LOG_PATH = OUT_DIR / "logs" / "possible_fabrication.jsonl"
LOW_OVERLAP_LOG_THRESHOLD = 0.2  # 僅供記錄用的軟性門檻，不是驗證失敗的硬性規則


def low_overlap_warning(label: str, original_title: str) -> bool:
    """五字標不需要照抄原標題的完整詞語，用一個有代表性的字/詞代表同一個概念
    就可以（例如用「漲」代表「通膨」），所以不拿字元重疊率當驗證失敗的硬性
    規則。但重疊率真的非常低時，值得記錄下來讓人事後抽查是不是真的編造內容，
    而不是語意代換——這裡只回傳 True/False，呼叫端自行決定要不要寫 log。"""
    if not label:
        return False
    overlap = sum(1 for ch in label if ch in original_title)
    return (overlap / len(label)) < LOW_OVERLAP_LOG_THRESHOLD


def validate_five_char_label(label: str, original_title: str) -> list[str]:
    """回傳違反的規則清單；空清單代表通過。

    注意：五字標不要求照抄原標題的完整詞語，只要語意對應（例如用「漲」代表
    「通膨」、用一個字代表一整個詞）就算合格，所以這裡不會因為字元跟原標題
    重疊率低就判定失敗——那種情況改成 low_overlap_warning 做非阻擋性的記錄。"""
    violations = []

    if len(label) != LABEL_LENGTH:
        violations.append(f"長度不是{LABEL_LENGTH}個字（實際{len(label)}字：{label!r}）")

    if looks_like_social_news(original_title):
        label_locations = extract_location_tokens(label)
        if not label_locations:
            violations.append("社會新聞五字標缺少地點")
        elif not any(loc in original_title for loc in label_locations):
            violations.append("五字標的地點沒有出現在原標題，疑似編造")

    return violations


CLAUSE_SEPARATORS = set("!?！？,，。；;")


def fallback_keyword_label(title: str, length: int = LABEL_LENGTH) -> str:
    """決定性退路（不呼叫 VLM）：原標題常常是「起因/背景」+「後續發展/衝突/
    結果」的複合句，單純取最前面 length 個字容易只摘到背景（例如「想買單口味
    涼圓，店家不賣爆口角肉身擋車」會變成「想買單口味」，漏掉真正的衝突）。
    這裡優先找第一個分句標點（！？，。；等），取它後面的 length 個字；分句
    後剩下的字不夠長（例如分句點根本就在標題結尾附近）就放棄、改回從頭取。
    去掉標點符號，保證一定有輸出，不受 VLM 可用性影響。"""
    strip_chars = set('!?！？"\'“”.,，。、:：;；()（）- /\\\n\t ')

    def clean(s: str) -> str:
        return "".join(ch for ch in s if ch not in strip_chars)

    for i, ch in enumerate(title):
        if ch in CLAUSE_SEPARATORS:
            after = clean(title[i + 1:])
            if len(after) >= length:
                return after[:length]
            break  # 只看第一個分句點；分句後字數不夠就不用再往後找

    cleaned = clean(title)
    if not cleaned:
        return title[:length].ljust(length, "　")
    if len(cleaned) < length:
        return cleaned.ljust(length, "　")
    return cleaned[:length]


PERSON_NAME_LOG_PATH = OUT_DIR / "logs" / "person_name_missed.jsonl"


def _log_person_name_missed(title: str, label: str, missed_name: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(
            {"title": title, "label": label, "missed_person_name": missed_name},
            ensure_ascii=False,
        ) + "\n")


def _log_low_overlap(title: str, label: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"title": title, "label": label}, ensure_ascii=False) + "\n")


def _fit_label_to_length(label: str, original_title: str, length: int = LABEL_LENGTH) -> Optional[str]:
    """label 比規定的 length 多出 1~2 個字時，嘗試只刪掉最少的字湊成剛好
    length 個字，而不是整個放棄、退回去粗暴地從原標題擷取關鍵字——VLM 給的
    答案通常語意是對的，只是字數沒抓準。優先回傳刪字後「完全合格」（含地點
    規則）的版本；找不到完全合格的版本，就回傳長度至少對的版本，讓呼叫端
    自行判斷還有沒有其他違規。

    字數差距超過 2 個字、或 label 太短（無法在不編造新內容的前提下補字）
    時回傳 None，交給呼叫端走原本的重試/退回路徑。

    刪字順序優先從尾端開始：中文標題/摘要通常把最關鍵的專有名詞（人名、
    地名、機構名、艦名等）放在前面，後面接描述性的字——從前面刪很容易把
    專有名詞拆散、讓人認不出是哪個人事物（例如「福建艦強能力」刪成
    「建艦強能力」就認不出是「福建艦」），從尾端刪通常比較安全。"""
    excess = len(label) - length
    if excess <= 0 or excess > 2:
        return None

    # combinations 是遞增排列（索引小的組合先出現），reversed 之後會先嘗試
    # 刪掉靠尾端的位置組合，符合「優先從尾端刪」的偏好。
    candidates_order = list(reversed(list(itertools.combinations(range(len(label)), excess))))

    best_with_remaining_violations: Optional[str] = None
    for drop_positions in candidates_order:
        candidate = "".join(ch for i, ch in enumerate(label) if i not in drop_positions)
        if not validate_five_char_label(candidate, original_title):
            return candidate
        if best_with_remaining_violations is None:
            best_with_remaining_violations = candidate
    return best_with_remaining_violations


def generate_five_char_label(
    title: str,
    vlm_text_fn: Callable[[str, int, Optional[list]], str],
    max_retries: int = LABEL_MAX_RETRIES,
    person_name_log_path: Path = PERSON_NAME_LOG_PATH,
    low_overlap_log_path: Path = LOW_OVERLAP_LOG_PATH,
) -> dict:
    """vlm_text_fn(title, attempt, previous_violations) -> 候選五字標字串。
    介面已就緒，但呼叫端要自己決定什麼時候真的接上會打網路的實作（見
    call_vlm_for_label），不要在批次抽幀/偵測還在跑、搶 GPU 資源時呼叫。

    地點是硬性規則（validate_five_char_label 會擋）。人名、以及五字標跟原
    標題的字元重疊率，都只是「盡量/供參考」，不會因為漏掉人名或重疊率低
    就重試或判定失敗（五字標本來就可以用代表字代換，不用照抄原文），但都
    會各自記錄到 out/logs/person_name_missed.jsonl、
    out/logs/possible_fabrication.jsonl 供事後抽查。"""
    attempts = []
    for attempt in range(max_retries + 1):
        previous_violations = None
        if attempts:
            last = attempts[-1]
            previous_violations = list(last["violations"])
            if last["label"]:
                # 把上一次實際的答案本身也附進去——不然 vlm_text_fn 只拿得到
                # 「長度不對」這種抽象描述，不知道自己上次具體說了什麼，重試
                # 時往往會原封不動地重複同一個錯誤答案（溫度設 0，輸入不變就
                # 會得到一樣的輸出）。附上原文後，retry prompt 才能明確要求
                # 「從這個答案增減字數」而不是整個重新生成。
                previous_violations.append(f"上一次的答案是「{last['label']}」（{len(last['label'])}個字）")
        try:
            label = vlm_text_fn(title, attempt, previous_violations)
        except Exception as e:
            # vlm_text_fn 自己的重試／健康檢查等待已經用盡才會走到這裡；不要讓
            # 單一則新聞的呼叫失敗拖垮整批，當作這次嘗試不合格，繼續重試/退回關鍵字擷取
            attempts.append({"label": None, "violations": [f"VLM 呼叫例外：{e!r}"]})
            continue
        violations = validate_five_char_label(label, title)
        attempts.append({"label": label, "violations": violations})
        if not violations:
            missed_name = missing_person_name(label, title)
            if missed_name:
                _log_person_name_missed(title, label, missed_name, person_name_log_path)
            if low_overlap_warning(label, title):
                _log_low_overlap(title, label, low_overlap_log_path)
            return {
                "label": label, "attempts": attempts, "fallback_used": False,
                "missed_person_name": missed_name,
            }

        # 這個模型對「請修正字數」這種重試指示幾乎沒反應（溫度0、同一組輸入
        # 會原封不動地重複同一個答案），但它給的答案通常只是多/少個一兩個字、
        # 語意本身是好的——與其整個丟掉、退回去粗暴地從原標題擷取關鍵字，
        # 不如直接幫它刪字湊成剛好 5 個字，盡量保留 VLM 已經做好的摘要。
        if label and len(label) != LABEL_LENGTH:
            trimmed = _fit_label_to_length(label, title)
            if trimmed is not None and trimmed != label:
                trimmed_violations = validate_five_char_label(trimmed, title)
                attempts.append({"label": trimmed, "violations": trimmed_violations, "source": "trimmed"})
                if not trimmed_violations:
                    missed_name = missing_person_name(trimmed, title)
                    if missed_name:
                        _log_person_name_missed(title, trimmed, missed_name, person_name_log_path)
                    if low_overlap_warning(trimmed, title):
                        _log_low_overlap(title, trimmed, low_overlap_log_path)
                    return {
                        "label": trimmed, "attempts": attempts, "fallback_used": False,
                        "missed_person_name": missed_name,
                    }

    fallback = fallback_keyword_label(title)
    missed_name = missing_person_name(fallback, title)
    if missed_name:
        _log_person_name_missed(title, fallback, missed_name, person_name_log_path)
    if low_overlap_warning(fallback, title):
        _log_low_overlap(title, fallback, low_overlap_log_path)
    return {
        "label": fallback, "attempts": attempts, "fallback_used": True,
        "missed_person_name": missed_name,
    }


LABEL_PROMPT_TEMPLATE = """你是新聞監看五字標產生器。根據下面的新聞標題，產生一個「五字標」摘要：

新聞標題：{title}

規則：
- 剛好 5 個中文字（不要標點、不要英數字，除非原標題本身就是英數字內容）。
- 目的是讓看的人一眼就能分辨出這是在播哪一則新聞，不是要你抄原標題的完整
  詞語——可以用語意對應的代表字/詞來濃縮一整個詞或概念，不必是原標題裡逐字
  出現過的字（例如原標題在講通膨/物價上漲，五字標可以直接用「漲」這個字，
  就算原標題沒有這個字也沒關係）。但不可以憑空多加原標題根本沒提到的新事實
  （例如不存在的地名、人名、數字、或結果）。
- 原標題常常是「起因/背景」+「後續發展/衝突/結果」的複合句（例如「想買單口味
  涼圓，店家不賣爆口角肉身擋車」：前半句只是背景，後半句「爆口角」「肉身擋車」
  才是真正讓這則新聞成為新聞的重點）。摘要時優先濃縮**後半段真正發生的事件/
  衝突/結果**，不要只摘要前面的起因或背景，除非原標題根本沒有後續發展可摘要。
- 如果是社會新聞（車禍、刑案、意外等），五字標「一定要」帶地點，這是最優先的規則。
- 如果原標題有出現人名，在不犧牲地點（若必要）的前提下，也「盡量」把人名放進五字標；
  五字空間有限、兩者衝突時，地點優先。
{retry_note}
只回傳五字標本身，不要加任何說明文字。"""


TEXT_CALL_MAX_RETRIES = 3
TEXT_CALL_BACKOFF_SECONDS = 2.0


def _resilient_text_call(
    make_request: Callable[[dict], object],
    env: dict,
    max_retries: int = TEXT_CALL_MAX_RETRIES,
    backoff_seconds: float = TEXT_CALL_BACKOFF_SECONDS,
    wait_fn: Callable[[dict], None] = vlm.wait_until_healthy,
    sleep_fn=time.sleep,
) -> object:
    """包一層重試／健康檢查等待，給純文字 VLM 呼叫用，避免暫時性斷線
    （ngrok 重啟之類）直接讓呼叫端整個崩潰：
    - 連線類例外（連不到伺服器）：卡住等健康檢查恢復、重新讀 .env，然後重打。
    - 其他例外（伺服器回應但內容有問題）：簡單重試幾次，仍失敗才真的往外丟。
    make_request(current_env) 要自己組 payload、送出請求、回傳解析後的結果。"""
    current_env = env
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return make_request(current_env)
        except vlm.CONNECTION_EXCEPTIONS as e:
            last_exc = e
            wait_fn(current_env)
            current_env = vlm.load_vlm_env()  # .env 可能被更新了，重新讀取再繼續
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                sleep_fn(backoff_seconds * (2 ** attempt))
    raise RuntimeError(f"文字 VLM 呼叫失敗，重試 {max_retries} 次仍失敗: {last_exc!r}") from last_exc


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
    retry_note = ""
    if previous_violations:
        retry_note = (
            f"（上一次產生的結果不合格，原因：{'; '.join(previous_violations)}。"
            "如果問題只是字數不是剛好5個字，不要整個重新想一個新的五字標——"
            "直接在上一次的答案裡增字或減字，湊成剛好5個字就好，盡量保留原本的用字；"
            "如果問題是社會新聞缺地點，才需要換掉內容、放進原標題裡的地點。）\n"
        )
    prompt = LABEL_PROMPT_TEMPLATE.format(title=title, retry_note=retry_note)

    def make_request(current_env: dict) -> str:
        import requests
        from requests.auth import HTTPBasicAuth

        payload = {
            "model": model or current_env.get("model"),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 30},
        }
        auth = HTTPBasicAuth(current_env["user"], current_env["password"]) if current_env.get("user") else None
        resp = requests.post(f"{current_env['url']}/api/chat", json=payload, auth=auth, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        content = body.get("message", {}).get("content", "")
        return content.strip()

    return _resilient_text_call(make_request, env)


JUDGE_PROMPT_TEMPLATE = """你是新聞監看助手。下面兩個「標題讀值」是從同一台新聞畫面上不同時間點的
OCR/VLM 讀值。請判斷這兩個讀值說的是不是同一件新聞事件，而不是兩個不同的新聞事件。

請注意，下面兩種情況都算「同一件新聞事件」(same)：
1. 文字因為辨識誤差有些微不同，但明顯是同一句話。
2. 同一個新聞議題/同一則深度報導正在繼續發展，只是換了個子標題、換了個舉例、
   換了個數字、或換了個切入角度（例如同一則物價上漲的報導，前段標題寫整體
   通膨數據，後段標題換成某個賣場的因應措施；或同一則意外事故，前段標題寫
   事故本身，後段標題換成後續調查/傷亡更新）——即使兩次讀到的文字幾乎沒有
   重疊的字，只要是同一個議題/事件在繼續報導，也算同一件。

下面這種情況要回答 different，不要因為「大主題/類別相同」就判成同一件：
兩個讀值講的是同一個類別底下不同的具體案例/事件（例如都跟「火災/消防安全」
有關，但一個是居家滅火器爆炸、另一個是另一個地點的工廠大火——這是兩個各自
獨立的具體事件，只是剛好屬於同一個新聞類別，不是同一則新聞在繼續發展）。
判斷關鍵是：標題B講的具體人事時地物，是不是標題A那個「同一個事件」接下來的
發展/後續/補充說明；如果標題B其實是另一個獨立的具體案例（即使主題類別一樣），
就是 different。

當兩個讀值明顯是兩個完全不相關的主題（例如一個在報天氣、一個在報車禍）時，
也同樣回答 different。

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
    prompt = JUDGE_PROMPT_TEMPLATE.format(title_a=title_a, title_b=title_b)

    def make_request(current_env: dict) -> bool:
        import requests
        from requests.auth import HTTPBasicAuth

        payload = {
            "model": model or current_env.get("model"),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 10},
        }
        auth = HTTPBasicAuth(current_env["user"], current_env["password"]) if current_env.get("user") else None
        resp = requests.post(f"{current_env['url']}/api/chat", json=payload, auth=auth, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        content = body.get("message", {}).get("content", "")
        return parse_judge_response(content)

    return _resilient_text_call(make_request, env)


# ---------- 整合：單一頻道的分段流程（本機、不呼叫任何 VLM） ----------

def load_vlm_results(path: Path) -> list[dict]:
    """讀 out/{channel}_vlm.jsonl，攤平成 run_state_machine 需要的欄位，依秒數排序。

    同一個 video_second 可能因為斷點續跑 + prompt 版本升級，在檔案裡留下好幾筆
    （舊 prompt 寫的一筆、重跑後新 prompt 又寫一筆）——resume 是用 append 寫檔，
    不會覆蓋舊的那行。這裡依 prompt_version 取最新的那筆（沒有這個欄位的舊資料
    版本視為最低），版本相同就取檔案裡較後面出現的那筆，確保不會把同一秒的
    新舊兩筆都當成獨立樣本重複計算。"""
    best_by_second: dict[int, tuple[int, dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        second = rec.get("video_second")
        if second is None:
            # 理論上不該發生（例如快取命中卻沒帶 extra_meta 寫壞的舊資料），
            # 但寧可跳過這一行、印警告，也不要讓整批分段因為一行壞資料而崩潰
            print(f"[warn] load_vlm_results: 跳過缺少 video_second 的壞資料行：{rec!r}")
            continue
        version = rec.get("prompt_version", -1)
        existing = best_by_second.get(second)
        if existing is None or version >= existing[0]:
            best_by_second[second] = (version, rec)

    frames = []
    for second, (_version, rec) in best_by_second.items():
        result = rec.get("result", {})
        frames.append({
            "video_second": second,
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


DEFAULT_MAX_ITEMS_PER_MINUTE = 1
DEFAULT_MIN_FRAMES_FOR_SECONDARY = 2


def select_dominant_per_minute(
    channel: str,
    vlm_results_path: Path,
    vlm_judge_fn: Optional[Callable[[str, str], bool]] = None,
    similarity_fn: Callable[[str, str], float] = default_similarity,
    high_threshold: float = MERGE_HIGH_THRESHOLD,
    low_threshold: float = MERGE_LOW_THRESHOLD,
    max_items_per_minute: int = DEFAULT_MAX_ITEMS_PER_MINUTE,
    min_frames_for_secondary: int = DEFAULT_MIN_FRAMES_FOR_SECONDARY,
) -> dict[int, list[dict]]:
    """給 fill.py 用：範本實際上幾乎每格只有一個五字標，所以每一分鐘只統計這一分鐘內
    （不分事件邊界，整分鐘所有 heartbeat/event 讀值）各候選的「比例」
    （= 支持的讀值筆數），比例最高者留在本分鐘：
    - 廣告（screen_type 是 full_ad/ad_wrap）一律算同一組，輸出「廣告」，不會因為
      偶爾文字不同就被拆成好幾個廣告候選。
    - 新聞讀值依文字相似度（模糊地帶可用 vlm_judge_fn）分組，組內讀值數最多的
      字串當代表。

    同分鐘內若有第二則「不同的新聞」（非廣告、讀值數達 min_frames_for_secondary），
    不會用「/」塞進本分鐘，而是取它第一個（video_second 最早）抓到的 bar 原文，
    帶到下一分鐘參與分組——如果下一分鐘本來就有同一則新聞在成長，會自然合併進該組
    （可能因此躍居本分鐘比例最高）；如果下一分鐘沒有相近的候選、追不上下一分鐘
    自己讀值數最高的候選，就直接捨棄，不會硬塞成下一分鐘的第二格（範本幾乎每格
    都只有一個五字標，這個候選既然沒能真的在下一分鐘成長/站穩，代表它多半只是
    過場雜訊）。只有恰好緊接著的下一個分鐘（minute+1）才會收到這個延後候選；
    中間若整分鐘沒有任何讀值（資料缺口），則不強行跨越去找更後面的分鐘。

    回傳 {minute_code: [{"kind": "news"|"ad", "label":, "frame_count":,
    "confidence":}, ...]}（已依讀值數由高到低排序；通常只有 1 筆，只有在真的
    並列第一時才會到 max_items_per_minute 筆）。"""
    frames = load_vlm_results(vlm_results_path)

    by_minute: dict[int, list[dict]] = {}
    for f in frames:
        minute = f.get("minute_code")
        if minute is None:
            continue
        by_minute.setdefault(minute, []).append(f)

    def _group(minute_frames: list[dict], carry_in: Optional[dict]) -> list[dict]:
        ad_frames = [f for f in minute_frames if f["screen_type"] in AD_SCREEN_TYPES]
        news_frames = [
            f for f in minute_frames
            if f["screen_type"] == "news" and f.get("main_title_visible") and f.get("main_title")
        ]

        groups: list[dict] = []
        if ad_frames:
            groups.append({"kind": "ad", "label": "廣告", "members": list(ad_frames), "forced": False})

        def _place(f: dict) -> bool:
            for g in groups:
                if g["kind"] != "news":
                    continue
                rep_title = g["members"][0]["main_title"]
                score = similarity_fn(rep_title, f["main_title"])
                if score >= high_threshold:
                    same = True
                elif score <= low_threshold:
                    same = False
                elif vlm_judge_fn is not None:
                    try:
                        same = vlm_judge_fn(rep_title, f["main_title"])
                    except Exception:
                        same = False
                else:
                    same = False
                if same:
                    g["members"].append(f)
                    return True
            return False

        for f in news_frames:
            if not _place(f):
                groups.append({"kind": "news", "label": f["main_title"], "members": [f], "forced": False})

        if carry_in is not None and not _place(carry_in):
            groups.append({
                "kind": "news", "label": carry_in["main_title"], "members": [carry_in], "forced": True,
            })

        return groups

    minutes_sorted = sorted(by_minute)
    result: dict[int, list[dict]] = {}
    pending_carry: Optional[dict] = None
    prev_minute: Optional[int] = None
    for minute in minutes_sorted:
        carry_in = pending_carry if prev_minute is not None and minute == prev_minute + 1 else None
        pending_carry = None

        groups = _group(by_minute[minute], carry_in)

        def _earliest_second(g: dict) -> int:
            return min(m.get("video_second", 0) for m in g["members"])

        # 讀值數由高到低；打平時本分鐘「原生」的候選優先於上一分鐘延後帶入、
        # 沒能合併的候選（延後帶入的候選只有在這一分鐘真的沒有更強的原生內容時
        # 才會顯示），同樣原生/同樣延後再用「先出現的那個」當決定性的 tiebreak
        # （範本幾乎每格只有一個五字標，打平時也要選出唯一一個，不能真的留兩個）。
        groups.sort(key=lambda g: (-len(g["members"]), 1 if g.get("forced") else 0, _earliest_second(g)))

        def _as_item(g: dict) -> dict:
            return {
                "kind": g["kind"],
                "label": g["label"],
                "frame_count": len(g["members"]),
                "confidence": max(m.get("confidence", 1.0) for m in g["members"]),
            }

        top_count = len(groups[0]["members"]) if groups else 0
        top_groups = [g for g in groups if len(g["members"]) == top_count]
        chosen = [_as_item(g) for g in top_groups[:max_items_per_minute]]
        result[minute] = chosen

        # 排名在並列第一之後、非強制帶入、非廣告的新聞候選，讀值數若達門檻，
        # 其「第一次出現」的原文延後到下一分鐘；不夠格（或是廣告/剛被強制帶入
        # 卻沒追上排名）的候選就直接捨棄，不會逐分鐘一直往後遞延。
        for g in groups:
            if any(g is tg for tg in top_groups) or g.get("forced") or g["kind"] != "news":
                continue
            if len(g["members"]) < min_frames_for_secondary:
                continue
            pending_carry = min(g["members"], key=lambda m: m.get("video_second", 0))
            break

        prev_minute = minute

    return result


def mark_continuations_blank(
    minute_groups: dict[int, list[dict]],
    vlm_judge_fn: Optional[Callable[[str, str], bool]] = None,
    similarity_fn: Callable[[str, str], float] = default_similarity,
    high_threshold: float = MERGE_HIGH_THRESHOLD,
) -> dict[int, list[dict]]:
    """給 fill.py 用：select_dominant_per_minute 對每一分鐘是獨立選代表候選的，
    同一則新聞橫跨好幾分鐘時，bar 文字常常字字句句都在換（同一事件換個角度/
    子標題），各分鐘各自轉出來的五字標看起來會像一直冒出新新聞，但其實只是
    前一分鐘的延續——範本的慣例是「同則延續」留白、換則才填（見
    docs/template_spec.md）。這裡把「跟最近一次真正顯示內容是同一則」的分鐘
    改成空清單，build_cell_plan 看到空清單會直接跳過該列（維持留白），呼叫端
    不用再處理，也省了這些分鐘的五字標 VLM 呼叫。

    同一則的判斷：廣告 vs 廣告一律算同一則；廣告 vs 新聞（或反過來）一律算
    不同；新聞 vs 新聞先看文字相似度，>= high_threshold 直接判同一則，否則
    只要有給 vlm_judge_fn 就一定問它做語意判斷（這裡跟 select_dominant_per_minute
    內部分組不同——那邊一分鐘要比對很多幀，只在模糊地帶才問以控制呼叫量；
    這裡一個分鐘只問一次，划算去抓「文字變很多但其實是同一個深度報導」的
    情況）；沒有 vlm_judge_fn 時保守判定不同則，避免在沒有語意判斷能力時
    誤判留白蓋掉真正的新內容。

    中間遇到完全沒有候選的分鐘（空清單，例如沒讀到任何有效讀值）原樣跳過、
    不會更新「最近一次真正顯示的內容」，讓資料缺口之後的分鐘還是能正確跟
    缺口前最後顯示的內容比較。"""
    result: dict[int, list[dict]] = {}
    last_shown_kind: Optional[str] = None
    last_shown_label: Optional[str] = None

    for minute in sorted(minute_groups):
        groups = minute_groups[minute]
        if not groups:
            result[minute] = []
            continue

        primary = groups[0]
        current_kind = primary["kind"]
        current_label = primary["label"]

        if last_shown_kind is None or current_kind != last_shown_kind:
            is_continuation = False
        elif current_kind == "ad":
            is_continuation = True
        else:
            score = similarity_fn(last_shown_label, current_label)
            if score >= high_threshold:
                is_continuation = True
            elif vlm_judge_fn is not None:
                try:
                    is_continuation = vlm_judge_fn(last_shown_label, current_label)
                except Exception:
                    is_continuation = False
            else:
                is_continuation = False

        if is_continuation:
            result[minute] = []
        else:
            result[minute] = groups
            last_shown_kind = current_kind
            last_shown_label = current_label

    return result


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
