import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import vlm


# ---------- pure helpers ----------

def test_resize_for_full_shrinks_wide_image():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = vlm.resize_for_full(img, max_width=768)
    assert out.shape[1] == 768
    assert out.shape[0] == 432


def test_resize_for_full_leaves_narrow_image_untouched():
    img = np.zeros((720, 700, 3), dtype=np.uint8)
    out = vlm.resize_for_full(img, max_width=768)
    assert out.shape == img.shape


def test_image_to_base64_roundtrip():
    img = np.random.randint(0, 255, (40, 60, 3), dtype=np.uint8)
    b64 = vlm.image_to_base64(img, quality=80)
    decoded = base64.b64decode(b64)
    arr = np.frombuffer(decoded, dtype=np.uint8)
    restored = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert restored.shape == img.shape


def test_image_to_base64_lower_quality_is_smaller():
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    b64_high = vlm.image_to_base64(img, quality=95)
    b64_low = vlm.image_to_base64(img, quality=50)
    assert len(b64_low) < len(b64_high)


def test_compute_cache_key_deterministic_and_sensitive():
    k1 = vlm.compute_cache_key(b"abc", b"def")
    k2 = vlm.compute_cache_key(b"abc", b"def")
    k3 = vlm.compute_cache_key(b"abc", b"xyz")
    assert k1 == k2
    assert k1 != k3


def test_build_payload_contains_model_images_schema_and_perf_params():
    payload = vlm.build_payload("gemma4:12b", "FULL_B64", "CROP_B64")
    assert payload["model"] == "gemma4:12b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"] == vlm.RESPONSE_SCHEMA
    assert payload["messages"][0]["images"] == ["FULL_B64", "CROP_B64"]
    assert payload["options"]["num_ctx"] == vlm.DEFAULT_NUM_CTX
    assert payload["options"]["num_predict"] == vlm.DEFAULT_NUM_PREDICT
    assert payload["options"]["temperature"] == 0


def test_build_payload_none_omits_fields_entirely():
    payload = vlm.build_payload("m", "F", "C", think=None, num_ctx=None, num_predict=None)
    assert "think" not in payload
    assert "num_ctx" not in payload["options"]
    assert "num_predict" not in payload["options"]
    assert payload["options"] == {"temperature": 0}


def test_build_payload_overrides():
    payload = vlm.build_payload("m", "F", "C", think=True, num_ctx=1000, num_predict=50, temperature=0.5)
    assert payload["think"] is True
    assert payload["options"] == {"temperature": 0.5, "num_ctx": 1000, "num_predict": 50}


# ---------- parse_response ----------

def test_parse_response_valid():
    raw = json.dumps({
        "screen_type": "news",
        "main_title": "測試標題",
        "main_title_visible": True,
        "confidence": 0.9,
    })
    result = vlm.parse_response(raw)
    assert result["screen_type"] == "news"
    assert result["main_title"] == "測試標題"


def test_parse_response_invalid_json_raises():
    with pytest.raises(vlm.VLMResponseError):
        vlm.parse_response("not json at all")


def test_parse_response_missing_field_raises():
    raw = json.dumps({"screen_type": "news", "main_title": "x"})
    with pytest.raises(vlm.VLMResponseError):
        vlm.parse_response(raw)


def test_parse_response_bad_enum_raises():
    raw = json.dumps({
        "screen_type": "not_a_real_type", "main_title": "x",
        "main_title_visible": True, "confidence": 0.5,
    })
    with pytest.raises(vlm.VLMResponseError):
        vlm.parse_response(raw)


# ---------- extract_timing ----------

def test_extract_timing_converts_ns_to_seconds():
    body = {
        "total_duration": 2_000_000_000,
        "load_duration": 500_000_000,
        "prompt_eval_count": 19,
        "prompt_eval_duration": 300_000_000,
        "eval_count": 10,
        "eval_duration": 100_000_000,
    }
    t = vlm.extract_timing(body)
    assert t["total_duration_s"] == pytest.approx(2.0)
    assert t["load_duration_s"] == pytest.approx(0.5)
    assert t["prompt_eval_count"] == 19
    assert t["prompt_eval_duration_s"] == pytest.approx(0.3)
    assert t["eval_count"] == 10
    assert t["eval_duration_s"] == pytest.approx(0.1)


def test_extract_timing_missing_fields_are_none():
    t = vlm.extract_timing({})
    assert t["total_duration_s"] is None
    assert t["prompt_eval_count"] is None


# ---------- retries + timing attempts ----------

class FakeResponse:
    def __init__(self, status_code=200, json_body=None):
        self.status_code = status_code
        self._json_body = json_body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json_body


def make_valid_body(title="標題", total_duration=1_000_000_000):
    content = json.dumps({
        "screen_type": "news", "main_title": title,
        "main_title_visible": True, "confidence": 0.8,
    })
    return {"message": {"content": content}, "total_duration": total_duration, "eval_count": 5}


def test_call_vlm_with_retries_succeeds_first_try_and_records_attempt(monkeypatch):
    def fake_post(url, json, auth, timeout):
        return FakeResponse(200, make_valid_body())

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    env = {"url": "http://fake", "user": "u", "password": "p", "model": "m"}
    attempts = []
    result = vlm.call_vlm_with_retries(env, "FULL", "CROP", max_retries=3, sleep_fn=lambda s: None, attempts_out=attempts)
    assert result["screen_type"] == "news"
    assert len(attempts) == 1
    assert attempts[0]["success"] is True
    assert attempts[0]["total_duration_s"] == pytest.approx(1.0)
    assert "wall_seconds" in attempts[0]


def test_call_vlm_with_retries_recovers_after_transient_failure(monkeypatch):
    attempts_n = {"n": 0}

    def fake_post(url, json, auth, timeout):
        attempts_n["n"] += 1
        if attempts_n["n"] < 2:
            raise requests.exceptions.ConnectionError("boom")
        return FakeResponse(200, make_valid_body())

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    env = {"url": "http://fake", "user": "u", "password": "p", "model": "m"}
    attempts = []
    result = vlm.call_vlm_with_retries(env, "FULL", "CROP", max_retries=3, sleep_fn=lambda s: None, attempts_out=attempts)
    assert result["screen_type"] == "news"
    assert len(attempts) == 2
    assert attempts[0]["success"] is False
    assert attempts[1]["success"] is True


def test_call_vlm_with_retries_exhausts_and_raises_with_attempts_recorded(monkeypatch):
    def fake_post(url, json, auth, timeout):
        raise requests.exceptions.Timeout("too slow")

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    env = {"url": "http://fake", "user": "u", "password": "p", "model": "m"}
    attempts = []
    with pytest.raises(RuntimeError):
        vlm.call_vlm_with_retries(env, "FULL", "CROP", max_retries=2, sleep_fn=lambda s: None, attempts_out=attempts)
    assert len(attempts) == 2
    assert all(a["success"] is False for a in attempts)


def test_call_vlm_with_retries_passes_payload_kwargs(monkeypatch):
    captured = {}

    def fake_post(url, json, auth, timeout):
        captured["payload"] = json
        return FakeResponse(200, make_valid_body())

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    env = {"url": "http://fake", "user": "u", "password": "p", "model": "m"}
    vlm.call_vlm_with_retries(env, "FULL", "CROP", sleep_fn=lambda s: None, num_predict=42)
    assert captured["payload"]["options"]["num_predict"] == 42


# ---------- ResultCache / TimingRecorder ----------

def test_result_cache_put_then_get(tmp_path):
    cache = vlm.ResultCache(tmp_path / "cache.jsonl")
    assert cache.get("abc") is None
    cache.put({"key": "abc", "result": {"screen_type": "news"}})
    assert cache.get("abc")["result"]["screen_type"] == "news"


def test_result_cache_reloads_from_disk(tmp_path):
    path = tmp_path / "cache.jsonl"
    vlm.ResultCache(path).put({"key": "k1", "result": {"screen_type": "ad_wrap"}})
    cache2 = vlm.ResultCache(path)
    assert cache2.get("k1")["result"]["screen_type"] == "ad_wrap"


def test_timing_recorder_summary_averages_successes_only(tmp_path):
    rec = vlm.TimingRecorder(tmp_path / "t.jsonl")
    rec.record({"success": True, "wall_seconds": 2.0, "total_duration_s": 1.0})
    rec.record({"success": True, "wall_seconds": 4.0, "total_duration_s": 3.0})
    rec.record({"success": False, "wall_seconds": 99.0})
    summary = rec.summary()
    assert summary["n"] == 2
    assert summary["n_failed"] == 1
    assert summary["wall_seconds"] == pytest.approx(3.0)
    assert summary["total_duration_s"] == pytest.approx(2.0)


# ---------- is_similar / plan_calls ----------

def test_is_similar_basic():
    assert vlm.is_similar(10, 15, threshold=5, diff=lambda a, b: abs(a - b)) is True
    assert vlm.is_similar(10, 16, threshold=5, diff=lambda a, b: abs(a - b)) is False


def test_is_similar_none_is_never_similar():
    assert vlm.is_similar(None, 5, threshold=100) is False
    assert vlm.is_similar(5, None, threshold=100) is False


def _fake_hash_lookup(table):
    def get_phash(record):
        return table[record["_id"]]
    return get_phash


def test_plan_calls_heartbeat_reuses_when_similar():
    records = [
        {"_id": 0, "type": "heartbeat"},
        {"_id": 1, "type": "heartbeat"},  # similar to 0 -> reuse
        {"_id": 2, "type": "heartbeat"},  # very different -> new call
    ]
    table = {0: 0, 1: 2, 2: 100}
    decisions = vlm.plan_calls(
        records, get_phash=_fake_hash_lookup(table),
        heartbeat_threshold=5, event_threshold=5,
    )
    assert decisions[0] == {"index": 0, "needs_call": True, "reused_index": None}
    assert decisions[1] == {"index": 1, "needs_call": False, "reused_index": 0}
    assert decisions[2] == {"index": 2, "needs_call": True, "reused_index": None}


def test_plan_calls_event_dedup_independent_of_heartbeat_chain():
    records = [
        {"_id": 0, "type": "event"},
        {"_id": 1, "type": "heartbeat"},
        {"_id": 2, "type": "event"},  # close to event 0 -> reuse
    ]
    table = {0: 50, 1: 999, 2: 52}
    decisions = vlm.plan_calls(
        records, get_phash=_fake_hash_lookup(table),
        heartbeat_threshold=5, event_threshold=5,
    )
    assert decisions[0]["needs_call"] is True
    assert decisions[1]["needs_call"] is True  # first heartbeat always calls
    assert decisions[2] == {"index": 2, "needs_call": False, "reused_index": 0}


def test_plan_calls_chain_of_reuses_points_back_to_last_real_call():
    records = [{"_id": i, "type": "heartbeat"} for i in range(4)]
    table = {0: 0, 1: 1, 2: 2, 3: 3}  # all within threshold of each other and of 0
    decisions = vlm.plan_calls(
        records, get_phash=_fake_hash_lookup(table),
        heartbeat_threshold=10, event_threshold=10,
    )
    assert decisions[0]["needs_call"] is True
    for d in decisions[1:]:
        assert d["needs_call"] is False
        assert d["reused_index"] == 0


# ---------- query_image_pair (end to end with fake network) ----------

def _write_test_image(path: Path, color=(0, 0, 0)):
    img = np.full((50, 80, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_query_image_pair_uses_cache_on_second_call(tmp_path, monkeypatch):
    full_path = tmp_path / "full.jpg"
    crop_path = tmp_path / "crop.jpg"
    _write_test_image(full_path)
    _write_test_image(crop_path)

    calls = {"n": 0}

    def fake_post(url, json, auth, timeout):
        calls["n"] += 1
        return FakeResponse(200, make_valid_body(title=f"call-{calls['n']}"))

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    monkeypatch.setattr(vlm, "ROOT", tmp_path)

    cache = vlm.ResultCache(tmp_path / "cache.jsonl")
    env = {"url": "http://fake", "user": "u", "password": "p", "model": "m"}

    first = vlm.query_image_pair(full_path, crop_path, env, cache)
    second = vlm.query_image_pair(full_path, crop_path, env, cache)

    assert calls["n"] == 1
    assert first["result"]["main_title"] == second["result"]["main_title"] == "call-1"


def test_query_image_pair_logs_timing(tmp_path, monkeypatch):
    full_path = tmp_path / "full.jpg"
    crop_path = tmp_path / "crop.jpg"
    _write_test_image(full_path)
    _write_test_image(crop_path)

    def fake_post(url, json, auth, timeout):
        return FakeResponse(200, make_valid_body())

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    monkeypatch.setattr(vlm, "ROOT", tmp_path)

    cache = vlm.ResultCache(tmp_path / "cache.jsonl")
    timing = vlm.TimingRecorder(tmp_path / "timing.jsonl")
    env = {"url": "http://fake", "user": "u", "password": "p", "model": "m"}

    vlm.query_image_pair(full_path, crop_path, env, cache, timing=timing, extra_meta={"channel": "ftv"})
    assert len(timing.records) == 1
    assert timing.records[0]["channel"] == "ftv"
    assert timing.records[0]["success"] is True
