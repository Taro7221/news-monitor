import json
import sys
from pathlib import Path

import pytest
import requests

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


def test_merge_continuations_vlm_judge_exception_degrades_to_ambiguous_flag():
    def fake_similarity(a, b):
        return 70.0  # 介於門檻之間，會呼叫 vlm_judge_fn

    def failing_judge(a, b):
        raise RuntimeError("VLM 呼叫失敗，重試 3 次仍失敗")

    events = [
        {"video_second": 0, "title": "標題A", "confidence": 0.9},
        {"video_second": 5, "title": "標題B", "confidence": 0.9},
    ]
    # 不應該讓例外往外丟，應該保守當作換則並標記 ambiguous_next
    segments = segment.merge_continuations(events, similarity_fn=fake_similarity, vlm_judge_fn=failing_judge)
    assert len(segments) == 2
    assert segments[0]["ambiguous_next"] is not None


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


def test_extract_person_name_tokens_finds_surname_candidates():
    tokens = segment.extract_person_name_tokens("沈伯洋蘇巧慧攜手上節目")
    assert any(t.startswith("沈") for t in tokens)
    assert any(t.startswith("蘇") for t in tokens)


def test_extract_person_name_tokens_empty_when_no_surname():
    assert segment.extract_person_name_tokens("台積電股價創新高") == []


def test_missing_person_name_flags_when_title_has_name_but_label_does_not():
    missed = segment.missing_person_name("訪談移花接", "沈伯洋訪談被移花接木")
    assert missed == "沈伯洋"


def test_missing_person_name_none_when_label_includes_surname():
    missed = segment.missing_person_name("沈伯洋訪談", "沈伯洋訪談被移花接木")
    assert missed is None


def test_missing_person_name_none_when_title_has_no_name():
    assert segment.missing_person_name("台北車禍案", "台北市發生車禍") is None


def test_validate_five_char_label_does_not_block_on_missing_person_name():
    # 人名只是「盡量」，不是硬性規則，缺人名不應該被 validate 擋下來
    violations = segment.validate_five_char_label("股價創新高", "沈伯洋股價創新高")
    assert violations == []


def test_generate_five_char_label_logs_missed_person_name(tmp_path):
    def vlm_fn(title, attempt, prev_violations):
        return "股價創新高"  # 5 個字、通過驗證，但沒有帶到人名「沈伯洋」

    log_path = tmp_path / "person_name_missed.jsonl"
    result = segment.generate_five_char_label(
        "沈伯洋股價創新高", vlm_fn, person_name_log_path=log_path,
    )
    assert result["fallback_used"] is False
    assert result["missed_person_name"] == "沈伯洋"
    assert log_path.exists()
    logged = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert logged["missed_person_name"] == "沈伯洋"


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


def test_generate_five_char_label_vlm_fn_exception_does_not_crash_falls_back():
    def always_raises(title, attempt, prev_violations):
        raise RuntimeError("VLM 呼叫失敗，重試 3 次仍失敗")

    result = segment.generate_five_char_label("台北市發生車禍", always_raises, max_retries=2)
    assert result["fallback_used"] is True
    assert len(result["label"]) == 5
    assert len(result["attempts"]) == 3
    assert all("VLM 呼叫例外" in a["violations"][0] for a in result["attempts"])


def test_generate_five_char_label_recovers_after_one_exception():
    calls = {"n": 0}

    def flaky(title, attempt, prev_violations):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return "台北車禍案"

    result = segment.generate_five_char_label("台北市發生車禍", flaky)
    assert result["fallback_used"] is False
    assert result["label"] == "台北車禍案"


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


def test_parse_judge_response_same():
    assert segment.parse_judge_response("same") is True
    assert segment.parse_judge_response("Same.") is True
    assert segment.parse_judge_response("  SAME  ") is True


def test_parse_judge_response_different():
    assert segment.parse_judge_response("different") is False
    assert segment.parse_judge_response("Different.") is False


def test_parse_judge_response_ambiguous_defaults_false():
    assert segment.parse_judge_response("I'm not sure") is False
    assert segment.parse_judge_response("") is False


# ---------- 純文字 VLM 呼叫的連線韌性 ----------

def test_resilient_text_call_succeeds_first_try():
    calls = []

    def make_request(env):
        calls.append(env["url"])
        return "ok"

    result = segment._resilient_text_call(make_request, {"url": "http://fake"}, sleep_fn=lambda s: None)
    assert result == "ok"
    assert calls == ["http://fake"]


def test_resilient_text_call_connection_error_waits_and_reloads_env(monkeypatch):
    attempts = []

    def make_request(env):
        attempts.append(env["url"])
        if env["url"] == "http://old":
            raise requests.exceptions.ConnectionError("down")
        return "ok"

    wait_calls = []
    monkeypatch.setattr(segment.vlm, "load_vlm_env", lambda: {"url": "http://new"})

    result = segment._resilient_text_call(
        make_request, {"url": "http://old"},
        wait_fn=wait_calls.append, sleep_fn=lambda s: None,
    )
    assert result == "ok"
    assert attempts == ["http://old", "http://new"]
    assert len(wait_calls) == 1


def test_resilient_text_call_non_connection_error_retries_then_raises():
    def make_request(env):
        raise ValueError("bad response")

    with pytest.raises(RuntimeError):
        segment._resilient_text_call(
            make_request, {"url": "http://fake"}, max_retries=2, sleep_fn=lambda s: None,
        )


def test_call_vlm_for_label_uses_resilient_call(monkeypatch):
    captured = {}

    def fake_resilient(make_request, env, **kwargs):
        captured["env"] = env
        return make_request(env)

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "台北車禍案"}}

    def fake_post(url, json, auth, timeout):
        captured["url"] = url
        return FakeResp()

    monkeypatch.setattr(segment, "_resilient_text_call", fake_resilient)
    monkeypatch.setattr(requests, "post", fake_post)

    env = {"url": "http://fake", "user": "", "password": "", "model": "m"}
    result = segment.call_vlm_for_label("台北市發生車禍", 0, None, env)
    assert result == "台北車禍案"
    assert captured["url"] == "http://fake/api/chat"


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


def test_load_vlm_results_dedupes_by_second_keeping_higher_prompt_version(tmp_path):
    import json as json_mod
    path = tmp_path / "vlm.jsonl"
    lines = [
        {"video_second": 5, "prompt_version": 1, "result": {"screen_type": "news", "main_title": "舊版本被字幕污染", "main_title_visible": True, "confidence": 0.9}},
        {"video_second": 5, "prompt_version": 2, "result": {"screen_type": "news", "main_title": "新版本修正後", "main_title_visible": True, "confidence": 0.9}},
    ]
    path.write_text("\n".join(json_mod.dumps(l) for l in lines), encoding="utf-8")
    frames = segment.load_vlm_results(path)
    assert len(frames) == 1
    assert frames[0]["main_title"] == "新版本修正後"


def test_load_vlm_results_missing_prompt_version_treated_as_lowest(tmp_path):
    import json as json_mod
    path = tmp_path / "vlm.jsonl"
    lines = [
        {"video_second": 5, "result": {"screen_type": "news", "main_title": "沒有版本號的舊資料", "main_title_visible": True, "confidence": 0.9}},
        {"video_second": 5, "prompt_version": 2, "result": {"screen_type": "news", "main_title": "有版本號的新資料", "main_title_visible": True, "confidence": 0.9}},
    ]
    path.write_text("\n".join(json_mod.dumps(l) for l in lines), encoding="utf-8")
    frames = segment.load_vlm_results(path)
    assert len(frames) == 1
    assert frames[0]["main_title"] == "有版本號的新資料"


def test_load_vlm_results_same_version_keeps_later_occurrence(tmp_path):
    import json as json_mod
    path = tmp_path / "vlm.jsonl"
    lines = [
        {"video_second": 5, "prompt_version": 2, "result": {"screen_type": "news", "main_title": "第一次寫入", "main_title_visible": True, "confidence": 0.9}},
        {"video_second": 5, "prompt_version": 2, "result": {"screen_type": "news", "main_title": "第二次寫入", "main_title_visible": True, "confidence": 0.9}},
    ]
    path.write_text("\n".join(json_mod.dumps(l) for l in lines), encoding="utf-8")
    frames = segment.load_vlm_results(path)
    assert len(frames) == 1
    assert frames[0]["main_title"] == "第二次寫入"


def test_load_vlm_results_skips_malformed_lines_missing_video_second(tmp_path, capsys):
    import json as json_mod
    path = tmp_path / "vlm.jsonl"
    lines = [
        {"video_second": 0, "result": {"screen_type": "news", "main_title": "正常那筆", "main_title_visible": True, "confidence": 0.9}},
        {"result": {"screen_type": "news", "main_title": "壞掉缺video_second", "main_title_visible": True, "confidence": 0.9}},
    ]
    path.write_text("\n".join(json_mod.dumps(l) for l in lines), encoding="utf-8")
    frames = segment.load_vlm_results(path)
    assert len(frames) == 1
    assert frames[0]["main_title"] == "正常那筆"
    assert "跳過" in capsys.readouterr().out


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


def test_build_fill_events_combines_ad_and_news_in_time_order(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    records = []

    def rec(sec, screen_type, title="", visible=True, conf=0.9, minute=1200):
        return {
            "video_second": sec, "minute_code": minute,
            "result": {"screen_type": screen_type, "main_title": title, "main_title_visible": visible, "confidence": conf},
        }

    records.append(rec(0, "news", "甲則新聞", minute=1200))
    for sec in (30, 31, 32):
        records.append(rec(sec, "full_ad", "", minute=1200))
    records.append(rec(60, "news", "乙則新聞", minute=1201))
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    events = segment.build_fill_events("ftv", path, low_conf_log_path=tmp_path / "low_conf.jsonl")
    kinds = [(e["kind"], e["label"]) for e in events]
    assert kinds == [("news", "甲則新聞"), ("ad", "廣告"), ("news", "乙則新聞")]
    assert [e["video_second"] for e in events] == [0, 32, 60]
    assert events[1]["minute_code"] == 1200
    assert events[2]["minute_code"] == 1201


def test_build_fill_events_no_ad_when_streak_too_short(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    records = [
        {"video_second": 0, "minute_code": 1200, "result": {"screen_type": "news", "main_title": "甲", "main_title_visible": True, "confidence": 0.9}},
        {"video_second": 30, "minute_code": 1200, "result": {"screen_type": "full_ad", "main_title": "", "main_title_visible": False, "confidence": 0.9}},
        {"video_second": 60, "minute_code": 1201, "result": {"screen_type": "news", "main_title": "甲", "main_title_visible": True, "confidence": 0.9}},
    ]
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")
    events = segment.build_fill_events("ftv", path, low_conf_log_path=tmp_path / "low_conf.jsonl")
    assert all(e["kind"] == "news" for e in events)


def _vlm_rec(sec, minute, screen_type="news", title="", visible=True, conf=0.9):
    return {
        "video_second": sec, "minute_code": minute,
        "result": {"screen_type": screen_type, "main_title": title, "main_title_visible": visible, "confidence": conf},
    }


def test_select_dominant_per_minute_single_item_no_second(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    records = [_vlm_rec(i, 1200, title="甲則新聞") for i in range(5)]
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    result = segment.select_dominant_per_minute("ftv", path)
    assert len(result[1200]) == 1
    assert result[1200][0]["label"] == "甲則新聞"
    assert result[1200][0]["frame_count"] == 5


def test_select_dominant_per_minute_minority_not_shown_same_minute(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    # 同分鐘內的次要新聞不再用「/」塞進本分鐘，只留比例最高者；
    # 這裡沒有下一分鐘的資料，次要新聞的延後候選無處可去，直接消失。
    records = (
        [_vlm_rec(i, 1200, title="多數新聞") for i in range(6)]
        + [_vlm_rec(10 + i, 1200, title="次要新聞") for i in range(3)]
        + [_vlm_rec(20, 1200, title="極少雜訊", conf=0.1)]
    )
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    result = segment.select_dominant_per_minute("ftv", path, min_frames_for_secondary=2)
    assert [g["label"] for g in result[1200]] == ["多數新聞"]


def test_select_dominant_per_minute_defers_minority_to_next_minute(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    records = (
        [_vlm_rec(i, 1200, title="多數新聞") for i in range(5)]
        + [_vlm_rec(10 + i, 1200, title="次要新聞") for i in range(3)]  # 第一筆在 sec=10
        + [_vlm_rec(70 + i, 1201, title="下一分鐘新聞") for i in range(4)]
    )
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    result = segment.select_dominant_per_minute("ftv", path, min_frames_for_secondary=2)
    assert [g["label"] for g in result[1200]] == ["多數新聞"]
    # 下一分鐘自己的新聞讀值數（4）比延後帶入、沒能合併的候選（1）多，
    # 後者被直接捨棄，不會硬塞成下一分鐘的第二格
    assert [g["label"] for g in result[1201]] == ["下一分鐘新聞"]


def test_select_dominant_per_minute_native_wins_tie_over_carried_item(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    # 下一分鐘自己只有 1 筆讀值，跟延後帶入的候選（也是 1 筆）打平——範本每格只有
    # 一個五字標，打平時一律讓這一分鐘「原生」的候選贏，不會兩個都顯示。
    records = (
        [_vlm_rec(i, 1200, title="多數新聞") for i in range(5)]
        + [_vlm_rec(10 + i, 1200, title="次要新聞") for i in range(2)]
        + [_vlm_rec(70, 1201, title="下一分鐘新聞")]
    )
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    result = segment.select_dominant_per_minute("ftv", path, min_frames_for_secondary=2)
    assert [g["label"] for g in result[1200]] == ["多數新聞"]
    assert [g["label"] for g in result[1201]] == ["下一分鐘新聞"]


def test_select_dominant_per_minute_carried_item_becomes_sole_when_next_minute_otherwise_empty(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    # 下一分鐘完全沒有可用的原生讀值（唯一一筆標題不可見），延後帶入的候選
    # 就直接成為下一分鐘的唯一內容——這才是「填入下一分鐘」字面上的意思。
    records = (
        [_vlm_rec(i, 1200, title="多數新聞") for i in range(5)]
        + [_vlm_rec(10 + i, 1200, title="次要新聞") for i in range(2)]
        + [_vlm_rec(70, 1201, title="", visible=False)]
    )
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    result = segment.select_dominant_per_minute("ftv", path, min_frames_for_secondary=2)
    assert [g["label"] for g in result[1200]] == ["多數新聞"]
    assert [g["label"] for g in result[1201]] == ["次要新聞"]


def test_select_dominant_per_minute_carried_item_merges_into_next_minutes_dominant(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    records = (
        [_vlm_rec(i, 1200, title="多數新聞") for i in range(5)]
        + [_vlm_rec(10 + i, 1200, title="故事X") for i in range(2)]
        + [_vlm_rec(70 + i, 1201, title="故事X") for i in range(3)]
    )
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    result = segment.select_dominant_per_minute("ftv", path, min_frames_for_secondary=2)
    assert [g["label"] for g in result[1200]] == ["多數新聞"]
    # 延後帶入的候選跟下一分鐘本來就在長大的同一則新聞合併，不會重複顯示成兩格
    assert len(result[1201]) == 1
    assert result[1201][0]["label"] == "故事X"
    assert result[1201][0]["frame_count"] == 4  # 自己的3筆 + 延後帶入的1筆


def test_select_dominant_per_minute_ad_minority_is_dropped_not_deferred(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    records = (
        [_vlm_rec(i, 1200, title="多數新聞") for i in range(5)]
        + [_vlm_rec(10 + i, 1200, screen_type="full_ad", title="", visible=False) for i in range(3)]
        + [_vlm_rec(70 + i, 1201, title="下一分鐘新聞") for i in range(2)]
    )
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    result = segment.select_dominant_per_minute("ftv", path, min_frames_for_secondary=2)
    assert [g["label"] for g in result[1200]] == ["多數新聞"]
    assert [g["label"] for g in result[1201]] == ["下一分鐘新聞"]  # 廣告的次要候選不會被帶過去


def test_select_dominant_per_minute_respects_max_items_per_minute(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    # 同一分鐘內兩則新聞剛好打平並列第一（都是3筆），max_items_per_minute 控制
    # 這種真的分不出高下時最多顯示幾個並列候選。
    records = (
        [_vlm_rec(i, 1200, title="A") for i in range(3)]
        + [_vlm_rec(10 + i, 1200, title="B") for i in range(3)]
    )
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    result = segment.select_dominant_per_minute("ftv", path, max_items_per_minute=1)
    assert [g["label"] for g in result[1200]] == ["A"]

    result2 = segment.select_dominant_per_minute("ftv", path, max_items_per_minute=2)
    assert [g["label"] for g in result2[1200]] == ["A", "B"]


def test_select_dominant_per_minute_ads_collapse_into_one_group_regardless_of_text_noise(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    # 廣告讀值 screen_type 不同（full_ad / ad_wrap）、文字也雜訊不一，應該都算同一組「廣告」
    records = [
        _vlm_rec(0, 1200, screen_type="full_ad", title="", visible=False),
        _vlm_rec(5, 1200, screen_type="ad_wrap", title="雜訊文字A", visible=True),
        _vlm_rec(10, 1200, screen_type="full_ad", title="雜訊文字B", visible=True),
        _vlm_rec(15, 1200, title="唯一新聞", visible=True),
    ]
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    result = segment.select_dominant_per_minute("ftv", path, min_frames_for_secondary=2)
    assert len(result[1200]) == 1
    assert result[1200][0]["kind"] == "ad"
    assert result[1200][0]["label"] == "廣告"
    assert result[1200][0]["frame_count"] == 3  # 3 筆廣告讀值都算同一組


def test_select_dominant_per_minute_uses_vlm_judge_for_ambiguous(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    records = [_vlm_rec(0, 1200, title="標題A")] + [_vlm_rec(5, 1200, title="標題B")]
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    def fake_similarity(a, b):
        return 70.0  # 模糊地帶

    merged = segment.select_dominant_per_minute(
        "ftv", path, vlm_judge_fn=lambda a, b: True, similarity_fn=fake_similarity,
        min_frames_for_secondary=1,
    )
    assert len(merged[1200]) == 1

    split = segment.select_dominant_per_minute(
        "ftv", path, vlm_judge_fn=lambda a, b: False, similarity_fn=fake_similarity,
        min_frames_for_secondary=1,
    )
    # 判定為不同則之後，兩則讀值數打平（都只有1筆）；預設 max_items_per_minute=1，
    # 打平時用先出現的那個當 tiebreak，只會顯示一個
    assert len(split[1200]) == 1
    assert split[1200][0]["label"] == "標題A"


def test_select_dominant_per_minute_vlm_judge_exception_defaults_to_different(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    records = [_vlm_rec(0, 1200, title="標題A")] + [_vlm_rec(5, 1200, title="標題B")]
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    def fake_similarity(a, b):
        return 70.0

    def failing_judge(a, b):
        raise RuntimeError("boom")

    result = segment.select_dominant_per_minute(
        "ftv", path, vlm_judge_fn=failing_judge, similarity_fn=fake_similarity,
        min_frames_for_secondary=1,
    )
    # 不應該崩潰；保守當作不同則後，兩則讀值數打平，預設只顯示先出現的那個
    assert len(result[1200]) == 1
    assert result[1200][0]["label"] == "標題A"


def test_select_dominant_per_minute_ignores_frames_without_visible_title(tmp_path):
    import json as json_mod
    path = tmp_path / "ftv_vlm.jsonl"
    records = [
        _vlm_rec(0, 1200, title="", visible=False),
        _vlm_rec(5, 1200, screen_type="other", title="", visible=False),
        _vlm_rec(10, 1200, title="唯一新聞", visible=True),
    ]
    path.write_text("\n".join(json_mod.dumps(r) for r in records), encoding="utf-8")

    result = segment.select_dominant_per_minute("ftv", path)
    assert len(result[1200]) == 1
    assert result[1200][0]["label"] == "唯一新聞"
    assert result[1200][0]["frame_count"] == 1


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
