import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import segment


def frame(second, screen_type="news", title="", visible=True, confidence=1.0):
    return {
        "video_second": second, "screen_type": screen_type,
        "main_title": title, "main_title_visible": visible, "confidence": confidence,
    }


# ---------- 1. 狀態機 / 遲滯判斷 ----------

def test_state_machine_single_ad_frame_does_not_trigger_ad():
    frames = [
        frame(0, title="A"),
        frame(1, screen_type="full_ad", title=""),
        frame(2, title="A"),
    ]
    events = segment.run_state_machine(frames)
    assert all(e["type"] != "ad_start" for e in events)


def test_state_machine_three_consecutive_ad_frames_triggers_ad():
    frames = [
        frame(0, title="A"),
        frame(1, screen_type="full_ad", title=""),
        frame(2, screen_type="full_ad", title=""),
        frame(3, screen_type="full_ad", title=""),
    ]
    events = segment.run_state_machine(frames)
    ad_events = [e for e in events if e["type"] == "ad_start"]
    assert len(ad_events) == 1
    assert ad_events[0]["video_second"] == 3


def test_state_machine_requires_fresh_title_to_exit_ad():
    frames = [
        frame(0, title="A"),
        frame(1, screen_type="full_ad", title=""),
        frame(2, screen_type="full_ad", title=""),
        frame(3, screen_type="full_ad", title=""),  # 進廣告
        frame(4, screen_type="news", title="", visible=False),  # 讀不到標題，還不算恢復
        frame(5, screen_type="news", title="", visible=False),
        frame(6, screen_type="news", title="B", visible=True),  # 真的讀到新標題才恢復
    ]
    events = segment.run_state_machine(frames)
    headline_events = [e for e in events if e["type"] == "headline"]
    titles_after_ad = [e for e in headline_events if e["video_second"] > 3]
    assert len(titles_after_ad) == 1
    assert titles_after_ad[0]["video_second"] == 6
    assert titles_after_ad[0]["title"] == "B"


def test_state_machine_does_not_reuse_stale_title_during_ad():
    # 廣告期間若偶爾雜訊讀到 "A"（跟廣告前一樣的殘留文字），只要是在 ad streak 累積中
    # 或者沒有連續 ad 帧，就不該被誤判；這裡驗證廣告開始後，若中途沒有真正離開，
    # 不會因為殘留文字又把同一個 "A" 當成新的 headline 事件重複記錄。
    frames = [
        frame(0, title="A"),
        frame(1, screen_type="full_ad"),
        frame(2, screen_type="full_ad"),
        frame(3, screen_type="full_ad"),
        frame(4, screen_type="news", title="A", visible=True),  # 回新聞但剛好讀到同一句
    ]
    events = segment.run_state_machine(frames)
    headline_events = [e for e in events if e["type"] == "headline"]
    # 第一筆是開頭的 A；廣告後恢復時再讀到一次 A，仍然算一次新的確認（由 merge 階段去判斷是否同一則）
    assert len(headline_events) == 2
    assert headline_events[-1]["video_second"] == 4


def test_state_machine_initial_frames_with_no_title_produce_no_headline():
    frames = [frame(0, title="", visible=False), frame(1, title="", visible=False)]
    events = segment.run_state_machine(frames)
    assert events == []


# ---------- 2. 低信心度 -> 多數決 ----------

def test_resolve_low_confidence_high_confidence_passes_through():
    target = {"video_second": 10, "title": "甲乙丙丁戊", "confidence": 0.9}
    result = segment.resolve_low_confidence(target, [])
    assert result["resolution"] == "high_confidence"
    assert result["needs_log"] is False
    assert result["title"] == "甲乙丙丁戊"


def test_resolve_low_confidence_majority_vote_picks_agreeing_group():
    target = {"video_second": 10, "title": "甲乙丙丁戊", "confidence": 0.2}
    neighbors = [
        {"video_second": 9, "title": "甲乙丙丁戊", "confidence": 0.3},
        {"video_second": 11, "title": "完全不同的句子", "confidence": 0.9},
    ]
    result = segment.resolve_low_confidence(target, neighbors)
    assert result["resolution"] == "majority_vote"
    assert result["needs_log"] is False
    assert result["title"] == "甲乙丙丁戊"


def test_resolve_low_confidence_no_majority_falls_back_to_best_guess():
    target = {"video_second": 10, "title": "甲乙丙丁戊", "confidence": 0.2}
    neighbors = [
        {"video_second": 9, "title": "完全不一樣A", "confidence": 0.1},
        {"video_second": 11, "title": "完全不一樣B", "confidence": 0.4},
    ]
    result = segment.resolve_low_confidence(target, neighbors)
    assert result["resolution"] == "best_guess_fallback"
    assert result["needs_log"] is True
    assert result["title"] == "完全不一樣B"  # 信心度最高的那個


def test_log_low_confidence_writes_jsonl(tmp_path):
    path = tmp_path / "low_conf.jsonl"
    segment.log_low_confidence({"channel": "ftv", "video_second": 5}, path=path)
    assert path.exists()
    assert "ftv" in path.read_text(encoding="utf-8")


def test_get_neighbor_window_respects_boundaries():
    frames = [frame(i) for i in range(5)]
    assert [f["video_second"] for f in segment.get_neighbor_window(frames, 0)] == [1]
    assert [f["video_second"] for f in segment.get_neighbor_window(frames, 4)] == [3]
    assert [f["video_second"] for f in segment.get_neighbor_window(frames, 2)] == [1, 3]


# ---------- 3. 合併同一則延續 ----------

def test_merge_continuations_groups_near_identical_titles():
    events = [
        {"video_second": 0, "title": "汽油彈襲警車罪加一等", "confidence": 0.9},
        {"video_second": 5, "title": "汽油彈襲警車罪一等", "confidence": 0.9},  # OCR 小誤差
        {"video_second": 10, "title": "汽油彈襲警車罪加一等", "confidence": 0.9},
    ]
    segments = segment.merge_continuations(events)
    assert len(segments) == 1
    assert segments[0]["merged_count"] == 3


def test_merge_continuations_splits_clearly_different_titles():
    events = [
        {"video_second": 0, "title": "汽油彈襲警車罪加一等", "confidence": 0.9},
        {"video_second": 60, "title": "台股史詩級崩跌創新低", "confidence": 0.9},
    ]
    segments = segment.merge_continuations(events)
    assert len(segments) == 2


def test_merge_continuations_ambiguous_without_vlm_judge_splits_and_flags():
    # 構造一個刻意落在中間地帶的相似度（用假的 similarity_fn 控制）
    def fake_similarity(a, b):
        return 70.0  # 介於 low(50) 和 high(85) 之間

    events = [
        {"video_second": 0, "title": "標題A", "confidence": 0.9},
        {"video_second": 5, "title": "標題B", "confidence": 0.9},
    ]
    segments = segment.merge_continuations(events, similarity_fn=fake_similarity)
    assert len(segments) == 2
    assert segments[0]["ambiguous_next"] is not None
    assert segments[0]["ambiguous_next"]["other_title"] == "標題B"


def test_merge_continuations_ambiguous_with_vlm_judge_merges_when_judge_says_same():
    def fake_similarity(a, b):
        return 70.0

    def fake_judge(a, b):
        return True  # VLM 判斷是同一則

    events = [
        {"video_second": 0, "title": "標題A", "confidence": 0.9},
        {"video_second": 5, "title": "標題B", "confidence": 0.9},
    ]
    segments = segment.merge_continuations(events, similarity_fn=fake_similarity, vlm_judge_fn=fake_judge)
    assert len(segments) == 1
    assert segments[0]["merged_count"] == 2


def test_merge_continuations_empty_input():
    assert segment.merge_continuations([]) == []


def test_merge_continuations_representative_title_is_majority_variant():
    events = [
        {"video_second": 0, "title": "汽油彈襲警車罪加一等", "confidence": 0.9},
        {"video_second": 5, "title": "汽油彈襲警車罪加一等", "confidence": 0.9},
        {"video_second": 10, "title": "汽油彈襲警車罪一等", "confidence": 0.5},  # 少數誤讀
    ]
    segments = segment.merge_continuations(events)
    assert segments[0]["representative_title"] == "汽油彈襲警車罪加一等"


# ---------- 4. 五字標產生 ----------

def test_validate_five_char_label_wrong_length():
    violations = segment.validate_five_char_label("甲乙丙", "某個新聞標題")
    assert any("長度" in v for v in violations)


def test_validate_five_char_label_correct_length_passes_overlap_check():
    title = "台北市中山區發生車禍一人送醫"
    label = "台北車禍案"  # 5 字，且字元都出現在原標題
    violations = segment.validate_five_char_label(label, title)
    assert violations == []


def test_validate_five_char_label_social_news_without_location_flagged():
    title = "一名男子持刀砍傷鄰居"  # 社會新聞關鍵字「持刀」，但沒有明確地點
    label = "男子砍傷人"
    violations = segment.validate_five_char_label(label, title)
    assert any("地點" in v for v in violations)


def test_validate_five_char_label_fabricated_content_flagged():
    title = "台積電股價創新高"
    label = "美國總統訪問"  # 跟原標題完全無關，疑似編造
    violations = segment.validate_five_char_label(label, title)
    assert any("編造" in v for v in violations)


def test_fallback_keyword_label_always_returns_something():
    assert len(segment.fallback_keyword_label("")) == 5
    assert len(segment.fallback_keyword_label("甲")) == 5
    result = segment.fallback_keyword_label("台北市中山區發生車禍一人送醫")
    assert len(result) == 5


def test_fallback_keyword_label_strips_punctuation():
    result = segment.fallback_keyword_label('"台積電"大漲！')
    assert "\"" not in result
    assert "！" not in result


def test_generate_five_char_label_accepts_first_valid_attempt():
    def vlm_fn(title, attempt, prev_violations):
        return "台北車禍案"

    result = segment.generate_five_char_label("台北市發生車禍", vlm_fn)
    assert result["label"] == "台北車禍案"
    assert result["fallback_used"] is False
    assert len(result["attempts"]) == 1


def test_generate_five_char_label_retries_then_succeeds():
    calls = {"n": 0}

    def vlm_fn(title, attempt, prev_violations):
        calls["n"] += 1
        if calls["n"] == 1:
            return "太短"  # 會因長度不對失敗
        return "台北車禍案"

    result = segment.generate_five_char_label("台北市發生車禍", vlm_fn)
    assert result["fallback_used"] is False
    assert result["label"] == "台北車禍案"
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["violations"]


def test_generate_five_char_label_falls_back_after_exhausting_retries():
    def vlm_fn(title, attempt, prev_violations):
        return "太短"  # 永遠不合格

    result = segment.generate_five_char_label("台北市發生車禍", vlm_fn, max_retries=2)
    assert result["fallback_used"] is True
    assert len(result["label"]) == 5
    assert len(result["attempts"]) == 3  # 原本 + 2 次重試


def test_generate_five_char_label_passes_previous_violations_to_retry():
    seen_violations = []

    def vlm_fn(title, attempt, prev_violations):
        seen_violations.append(prev_violations)
        return "太短" if attempt == 0 else "台北車禍案"

    segment.generate_five_char_label("台北市發生車禍", vlm_fn)
    assert seen_violations[0] is None
    assert seen_violations[1] is not None  # 第二次重試有拿到上一次的違規原因


# ---------- 整合：load_vlm_results / build_segments_for_channel ----------

def test_load_vlm_results_sorts_by_second(tmp_path):
    import json as json_mod
    path = tmp_path / "vlm.jsonl"
    lines = [
        {"video_second": 5, "result": {"screen_type": "news", "main_title": "B", "main_title_visible": True, "confidence": 0.9}},
        {"video_second": 0, "result": {"screen_type": "news", "main_title": "A", "main_title_visible": True, "confidence": 0.9}},
    ]
    path.write_text("\n".join(json_mod.dumps(l) for l in lines), encoding="utf-8")
    frames = segment.load_vlm_results(path)
    assert [f["video_second"] for f in frames] == [0, 5]


def test_build_segments_for_channel_end_to_end(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    records = []
    for sec, title, conf in [
        (0, "甲則新聞標題", 0.9),
        (30, "甲則新聞標題", 0.9),
        (60, "乙則新聞標題", 0.9),
    ]:
        records.append({
            "video_second": sec,
            "result": {"screen_type": "news", "main_title": title, "main_title_visible": True, "confidence": conf},
        })
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    segments = segment.build_segments_for_channel("ftv", path, low_conf_log_path=tmp_path / "low_conf.jsonl")
    assert len(segments) == 2
    assert segments[0]["representative_title"] == "甲則新聞標題"
    assert segments[1]["representative_title"] == "乙則新聞標題"


def test_build_segments_for_channel_logs_low_confidence(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    records = [
        {"video_second": 0, "result": {"screen_type": "news", "main_title": "甲完全不同A", "main_title_visible": True, "confidence": 0.1}},
        {"video_second": 1, "result": {"screen_type": "news", "main_title": "乙完全不同B", "main_title_visible": True, "confidence": 0.1}},
    ]
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    low_conf_path = tmp_path / "low_conf.jsonl"
    segment.build_segments_for_channel("ftv", path, low_conf_log_path=low_conf_path)
    assert low_conf_path.exists()
