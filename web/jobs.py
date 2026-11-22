"""web/jobs.py：上傳影片後的 job 管理與背景處理。

目前的實作刻意是「一次處理一台」：一個 job 裡上傳了哪幾台，就依序（不是
同時）對每一台跑 detect -> vlm -> segment -> fill，每一台處理完就立刻把
目前已完成的頻道結果寫進這個 job 專屬的 Excel，所以使用者不用等 4 台都跑完，
隨時可以按「立即輸出」下載目前的進度。

要怎麼改成「4 台同時監看、填進同一份 Excel」，見 docs/web_demo_roadmap.md。
"""

import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import calibrate  # noqa: E402
import detect  # noqa: E402
import vlm  # noqa: E402
import segment  # noqa: E402
import fill  # noqa: E402

import openpyxl  # noqa: E402

JOBS_DIR = ROOT / "out" / "web_jobs"
CHANNELS = ["ftv", "tvbs", "ebc", "cts"]
CHANNEL_LABELS = {"ftv": "民視", "tvbs": "TVBS-N", "ebc": "東森", "cts": "華視"}

_jobs_lock = threading.Lock()
_jobs: dict[str, "Job"] = {}

# 共用同一份 VLM 快取/計時紀錄，不同 job 的影片畫面內容不同、cache key
# （圖片位元組的 hash）本來就不會互相命中，可以安全共用。
_cache = vlm.ResultCache(vlm.CACHE_PATH)
_timing = vlm.TimingRecorder(vlm.TIMING_PATH)


class Job:
    def __init__(self, job_id: str, broadcast_start: datetime, duration_seconds: Optional[int]):
        self.id = job_id
        self.dir = JOBS_DIR / job_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "videos").mkdir(parents=True, exist_ok=True)
        self.broadcast_start = broadcast_start
        self.duration_seconds = duration_seconds
        self.created_at = datetime.now()
        self.channels: dict[str, dict] = {
            ch: {"status": "not_uploaded", "stage": None, "error": None}
            for ch in CHANNELS
        }
        self.excel_path = self.dir / "output.xlsx"
        self.excel_lock = threading.Lock()
        # 每台處理完成後快取它的填表內容（row -> {text, low_conf, notes}），
        # 避免每次有新頻道完成就把所有頻道的五字標重新呼叫一次 VLM 生成。
        self.channel_plans: dict[str, dict] = {}
        self.worker_thread: Optional[threading.Thread] = None

    def video_path(self, channel: str) -> Path:
        return self.dir / "videos" / f"{channel}.mp4"

    def to_status_dict(self) -> dict:
        return {
            "job_id": self.id,
            "created_at": self.created_at.isoformat(),
            "broadcast_start": self.broadcast_start.isoformat(),
            "duration_seconds": self.duration_seconds,
            "channels": {
                ch: {"status": info["status"], "stage": info["stage"], "error": info["error"]}
                for ch, info in self.channels.items()
            },
            "excel_ready": self.excel_path.exists(),
            "all_done": all(
                self.channels[ch]["status"] in ("done", "error", "not_uploaded")
                for ch in CHANNELS
            ),
        }


def create_job(broadcast_start: datetime, duration_seconds: Optional[int]) -> Job:
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, broadcast_start, duration_seconds)
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)


def save_upload(job: Job, channel: str, file_storage) -> None:
    dest = job.video_path(channel)
    file_storage.save(str(dest))
    job.channels[channel]["status"] = "queued"


def start_processing(job: Job) -> None:
    if job.worker_thread and job.worker_thread.is_alive():
        return  # 已經在跑了，不要重複啟動
    t = threading.Thread(target=_run_job, args=(job,), daemon=True)
    job.worker_thread = t
    t.start()


def _run_job(job: Job) -> None:
    roi_config = calibrate.load_config()
    channels_to_run = [ch for ch in CHANNELS if job.channels[ch]["status"] == "queued"]

    for ch in channels_to_run:
        info = job.channels[ch]
        try:
            info["status"] = "processing"

            info["stage"] = "偵測換則事件（detect）"
            events_path = job.dir / f"{ch}_events.jsonl"
            frames_dir = job.dir / "frames" / ch
            detect.process_channel(
                ch, roi_config, job.broadcast_start,
                max_seconds=job.duration_seconds,
                events_path=events_path,
                video_path=job.video_path(ch),
                frames_dir=frames_dir,
            )

            info["stage"] = "VLM 判讀標題（可能需要幾分鐘）"
            vlm_path = job.dir / f"{ch}_vlm.jsonl"
            vlm.process_events_file(events_path, ch, _cache, vlm_path, timing=_timing)

            info["stage"] = "產生五字標並填表"
            job.channel_plans[ch] = _build_channel_plan(ch, vlm_path)
            _write_excel(job)

            info["status"] = "done"
            info["stage"] = None
        except Exception as e:
            info["status"] = "error"
            info["error"] = str(e)
            info["stage"] = None
            traceback.print_exc()


def _build_channel_plan(channel: str, vlm_path: Path) -> dict:
    env = vlm.load_vlm_env()

    def vlm_judge_fn(title_a: str, title_b: str) -> bool:
        return segment.call_vlm_judge_same_story(title_a, title_b, env)

    def vlm_text_fn(title: str, attempt: int, prev_violations):
        return segment.call_vlm_for_label(title, attempt, prev_violations, env)

    minute_groups = segment.select_dominant_per_minute(channel, vlm_path, vlm_judge_fn=vlm_judge_fn)
    minute_groups = segment.mark_continuations_blank(minute_groups, vlm_judge_fn=vlm_judge_fn)

    def label_fn(item):
        return fill.default_label_fn(item, use_vlm=True, vlm_text_fn=vlm_text_fn)

    return fill.build_cell_plan(minute_groups, label_fn)


def _write_excel(job: Job) -> None:
    """從範本重新套用目前所有已快取的頻道 plan。只動 CLAUDE.md 允許的四台
    五字標欄位，範本其他內容/格式不變。"""
    with job.excel_lock:
        wb = openpyxl.load_workbook(str(fill.DEFAULT_TEMPLATE_PATH))
        ws = wb[fill.SHEET_NAME]
        for ch, plan in job.channel_plans.items():
            column = fill.CHANNEL_COLUMNS[ch]
            fill.apply_plan_to_sheet(ws, column, plan)
        job.excel_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(str(job.excel_path))
