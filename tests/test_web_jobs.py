import io
import sys
from datetime import datetime
from pathlib import Path

import openpyxl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jobs  # noqa: E402
import fill  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_jobs_dir(tmp_path, monkeypatch):
    # 每個測試都用自己的 job 根目錄，不要互相污染、也不要動到真正的 out/web_jobs
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "web_jobs")
    monkeypatch.setattr(jobs, "_jobs", {})
    yield


def test_create_job_and_get_job_roundtrip():
    job = jobs.create_job(datetime(2026, 11, 6, 12, 0, 0), None)
    assert jobs.get_job(job.id) is job
    assert jobs.get_job("doesnotexist") is None


def test_create_job_sets_all_channels_not_uploaded():
    job = jobs.create_job(datetime(2026, 11, 6, 12, 0, 0), 900)
    status = job.to_status_dict()
    assert set(status["channels"].keys()) == set(jobs.CHANNELS)
    for ch in jobs.CHANNELS:
        assert status["channels"][ch]["status"] == "not_uploaded"
    assert status["excel_ready"] is False
    assert status["all_done"] is True  # 沒有任何頻道在跑，視為「目前沒有進行中的工作」


def test_save_upload_marks_channel_queued_and_writes_file():
    job = jobs.create_job(datetime(2026, 11, 6, 12, 0, 0), None)
    fake_file = FakeFileStorage(b"not a real video, just bytes")
    jobs.save_upload(job, "ftv", fake_file)

    assert job.channels["ftv"]["status"] == "queued"
    assert job.video_path("ftv").exists()
    assert job.video_path("ftv").read_bytes() == b"not a real video, just bytes"
    # 還沒上傳的頻道不受影響
    assert job.channels["tvbs"]["status"] == "not_uploaded"


def test_write_excel_applies_cached_plans_to_correct_columns(tmp_path, monkeypatch):
    job = jobs.create_job(datetime(2026, 11, 6, 12, 0, 0), None)
    # 繞過真正會打 VLM 的 _build_channel_plan，直接塞一個假的 plan 進快取
    job.channel_plans["ftv"] = {
        fill.minute_code_to_row(1230): {"text": "測試五字標", "low_conf": False, "notes": []},
    }
    jobs._write_excel(job)

    assert job.excel_path.exists()
    wb = openpyxl.load_workbook(str(job.excel_path))
    ws = wb[fill.SHEET_NAME]
    row = fill.minute_code_to_row(1230)
    assert ws[f"{fill.CHANNEL_COLUMNS['ftv']}{row}"].value == "測試五字標"
    # 沒有填的頻道欄位應該維持範本原樣（空白），不應該被誤寫
    assert ws[f"{fill.CHANNEL_COLUMNS['tvbs']}{row}"].value in (None, "")


def test_write_excel_is_cumulative_across_multiple_calls():
    job = jobs.create_job(datetime(2026, 11, 6, 12, 0, 0), None)
    row = fill.minute_code_to_row(1230)

    job.channel_plans["ftv"] = {row: {"text": "第一台完成", "low_conf": False, "notes": []}}
    jobs._write_excel(job)

    job.channel_plans["tvbs"] = {row: {"text": "第二台完成", "low_conf": False, "notes": []}}
    jobs._write_excel(job)

    wb = openpyxl.load_workbook(str(job.excel_path))
    ws = wb[fill.SHEET_NAME]
    assert ws[f"{fill.CHANNEL_COLUMNS['ftv']}{row}"].value == "第一台完成"
    assert ws[f"{fill.CHANNEL_COLUMNS['tvbs']}{row}"].value == "第二台完成"


class FakeFileStorage:
    """模擬 werkzeug 的 FileStorage，測試不用真的啟動 Flask。"""

    def __init__(self, data: bytes):
        self._data = data

    def save(self, path: str) -> None:
        Path(path).write_bytes(self._data)
