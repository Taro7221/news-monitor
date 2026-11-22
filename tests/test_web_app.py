import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))

import app as web_app  # noqa: E402
import jobs  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "web_jobs")
    monkeypatch.setattr(jobs, "_jobs", {})
    # 路由測試不應該真的啟動背景 thread 去打 VLM/讀影片，只測上傳驗證跟
    # job 建立/查詢/下載這幾條路徑本身接得對不對。
    monkeypatch.setattr(jobs, "start_processing", lambda job: None)
    web_app.app.config["TESTING"] = True
    return web_app.app.test_client()


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "新聞監看".encode("utf-8") in resp.data


def test_upload_without_any_file_is_rejected(client):
    resp = client.post("/api/upload", data={"date": "2026-11-06"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_upload_with_bad_date_is_rejected(client):
    data = {"date": "not-a-date", "ftv": (io.BytesIO(b"fake video bytes"), "ftv.mp4")}
    resp = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_upload_with_bad_duration_is_rejected(client):
    data = {
        "date": "2026-11-06",
        "duration_minutes": "not-a-number",
        "ftv": (io.BytesIO(b"fake video bytes"), "ftv.mp4"),
    }
    resp = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_upload_single_channel_creates_job_and_queues_only_that_channel(client):
    data = {
        "date": "2026-11-06",
        "ftv": (io.BytesIO(b"fake video bytes"), "ftv.mp4"),
    }
    resp = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]

    job = jobs.get_job(job_id)
    assert job is not None
    assert job.channels["ftv"]["status"] == "queued"
    assert job.channels["tvbs"]["status"] == "not_uploaded"


def test_status_endpoint_for_unknown_job_is_404(client):
    resp = client.get("/api/jobs/doesnotexist/status")
    assert resp.status_code == 404


def test_download_before_anything_done_returns_409(client):
    data = {"date": "2026-11-06", "ftv": (io.BytesIO(b"fake video bytes"), "ftv.mp4")}
    resp = client.post("/api/upload", data=data, content_type="multipart/form-data")
    job_id = resp.get_json()["job_id"]

    resp = client.get(f"/api/jobs/{job_id}/download")
    assert resp.status_code == 409


def test_download_after_a_channel_completes_serves_file(client):
    data = {"date": "2026-11-06", "ftv": (io.BytesIO(b"fake video bytes"), "ftv.mp4")}
    resp = client.post("/api/upload", data=data, content_type="multipart/form-data")
    job_id = resp.get_json()["job_id"]
    job = jobs.get_job(job_id)

    # 模擬 ftv 已經處理完成、Excel 已經寫出（跳過真正的背景處理）
    job.channel_plans["ftv"] = {5: {"text": "測試", "low_conf": False, "notes": []}}
    jobs._write_excel(job)

    resp = client.get(f"/api/jobs/{job_id}/download")
    assert resp.status_code == 200
    assert resp.headers["Content-Disposition"].startswith("attachment")
