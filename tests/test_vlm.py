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
    img = np.zeros((720, 2560, 3), dtype=np.uint8)
    out = vlm.resize_for_full(img, max_width=1280)
    assert out.shape[1] == 1280
    assert out.shape[0] == 360


def test_resize_for_full_leaves_narrow_image_untouched():
    img = np.zeros((720, 1000, 3), dtype=np.uint8)
    out = vlm.resize_for_full(img, max_width=1280)
    assert out.shape == img.shape


def test_image_to_base64_roundtrip():
    img = np.random.randint(0, 255, (40, 60, 3), dtype=np.uint8)
    b64 = vlm.image_to_base64(img)
    decoded = base64.b64decode(b64)
    arr = np.frombuffer(decoded, dtype=np.uint8)
    restored = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert restored.shape == img.shape


def test_compute_cache_key_deterministic_and_sensitive():
    k1 = vlm.compute_cache_key(b"abc", b"def")
    k2 = vlm.compute_cache_key(b"abc", b"def")
    k3 = vlm.compute_cache_key(b"abc", b"xyz")
    assert k1 == k2
    assert k1 != k3


def test_build_payload_contains_model_images_and_schema():
    payload = vlm.build_payload("gemma3:12b", "FULL_B64", "CROP_B64")
    assert payload["model"] == "gemma3:12b"
    assert payload["stream"] is False
    assert payload["format"] == vlm.RESPONSE_SCHEMA
    assert payload["messages"][0]["images"] == ["FULL_B64", "CROP_B64"]


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
    assert result["main_title_visible"] is True
    assert result["confidence"] == 0.9


def test_parse_response_invalid_json_raises():
    with pytest.raises(vlm.VLMResponseError):
        vlm.parse_response("not json at all")


def test_parse_response_missing_field_raises():
    raw = json.dumps({"screen_type": "news", "main_title": "x"})
    with pytest.raises(vlm.VLMResponseError):
        vlm.parse_response(raw)


def test_parse_response_bad_enum_raises():
    raw = json.dumps({
        "screen_type": "not_a_real_type",
        "main_title": "x",
        "main_title_visible": True,
        "confidence": 0.5,
    })
    with pytest.raises(vlm.VLMResponseError):
        vlm.parse_response(raw)


def test_parse_response_wrong_type_raises():
    raw = json.dumps({
        "screen_type": "news",
        "main_title": "x",
        "main_title_visible": "yes",  # should be bool
        "confidence": 0.5,
    })
    with pytest.raises(vlm.VLMResponseError):
        vlm.parse_response(raw)


# ---------- retries ----------

class FakeResponse:
    def __init__(self, status_code=200, json_body=None):
        self.status_code = status_code
        self._json_body = json_body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json_body


def make_valid_body(title="標題"):
    content = json.dumps({
        "screen_type": "news",
        "main_title": title,
        "main_title_visible": True,
        "confidence": 0.8,
    })
    return {"message": {"content": content}}


def test_call_vlm_with_retries_succeeds_first_try(monkeypatch):
    calls = []

    def fake_post(url, json, auth, timeout):
        calls.append(url)
        return FakeResponse(200, make_valid_body())

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    env = {"url": "http://fake", "user": "u", "password": "p", "model": "m"}
    result = vlm.call_vlm_with_retries(env, "FULL", "CROP", max_retries=3, sleep_fn=lambda s: None)
    assert result["screen_type"] == "news"
    assert len(calls) == 1


def test_call_vlm_with_retries_recovers_after_transient_failure(monkeypatch):
    attempts = {"n": 0}

    def fake_post(url, json, auth, timeout):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise requests.exceptions.ConnectionError("boom")
        return FakeResponse(200, make_valid_body())

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    env = {"url": "http://fake", "user": "u", "password": "p", "model": "m"}
    result = vlm.call_vlm_with_retries(env, "FULL", "CROP", max_retries=3, sleep_fn=lambda s: None)
    assert result["screen_type"] == "news"
    assert attempts["n"] == 2


def test_call_vlm_with_retries_exhausts_and_raises(monkeypatch):
    def fake_post(url, json, auth, timeout):
        raise requests.exceptions.Timeout("too slow")

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    env = {"url": "http://fake", "user": "u", "password": "p", "model": "m"}
    with pytest.raises(RuntimeError):
        vlm.call_vlm_with_retries(env, "FULL", "CROP", max_retries=2, sleep_fn=lambda s: None)


def test_call_vlm_with_retries_retries_on_bad_schema_response(monkeypatch):
    attempts = {"n": 0}

    def fake_post(url, json, auth, timeout):
        attempts["n"] += 1
        if attempts["n"] < 2:
            return FakeResponse(200, {"message": {"content": "garbage not json"}})
        return FakeResponse(200, make_valid_body())

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    env = {"url": "http://fake", "user": "u", "password": "p", "model": "m"}
    result = vlm.call_vlm_with_retries(env, "FULL", "CROP", max_retries=3, sleep_fn=lambda s: None)
    assert result["screen_type"] == "news"
    assert attempts["n"] == 2


# ---------- ResultCache ----------

def test_result_cache_put_then_get(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    cache = vlm.ResultCache(cache_path)
    assert cache.get("abc") is None
    cache.put({"key": "abc", "result": {"screen_type": "news"}})
    assert cache.get("abc")["result"]["screen_type"] == "news"


def test_result_cache_reloads_from_disk(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    cache1 = vlm.ResultCache(cache_path)
    cache1.put({"key": "k1", "result": {"screen_type": "ad_wrap"}})

    cache2 = vlm.ResultCache(cache_path)
    assert cache2.get("k1")["result"]["screen_type"] == "ad_wrap"


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

    assert calls["n"] == 1  # second call reused the cache, no new network call
    assert first["result"]["main_title"] == second["result"]["main_title"] == "call-1"
