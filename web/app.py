"""web/app.py：給三立測試用的上傳 demo。

上傳 1~4 台的側錄影片，背景依序（目前先一次一台，見
docs/web_demo_roadmap.md 之後怎麼改成 4 台同時）跑 detect -> vlm ->
segment -> fill，隨時可以下載目前已完成頻道的 Excel。

使用方式：
    python web/app.py
然後瀏覽器打開 http://127.0.0.1:5000
"""

import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobs  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB，整小時側錄通常幾百MB~1GB


@app.route("/")
def index():
    return render_template("index.html", channels=jobs.CHANNELS, labels=jobs.CHANNEL_LABELS)


@app.route("/api/upload", methods=["POST"])
def upload():
    date_str = request.form.get("date", "").strip()
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime.now()
    except ValueError:
        return jsonify({"error": "監看日期格式錯誤，請用 YYYY-MM-DD"}), 400
    broadcast_start = date.replace(hour=12, minute=0, second=0, microsecond=0)

    duration_str = request.form.get("duration_minutes", "").strip()
    duration_seconds = None
    if duration_str:
        try:
            duration_seconds = int(float(duration_str) * 60)
        except ValueError:
            return jsonify({"error": "測試用處理分鐘數需為數字"}), 400

    uploaded_channels = [ch for ch in jobs.CHANNELS if request.files.get(ch) and request.files[ch].filename]
    if not uploaded_channels:
        return jsonify({"error": "至少要上傳一台的影片"}), 400

    job = jobs.create_job(broadcast_start, duration_seconds)
    for ch in uploaded_channels:
        jobs.save_upload(job, ch, request.files[ch])
    jobs.start_processing(job)

    return jsonify({"job_id": job.id})


@app.route("/api/jobs/<job_id>/status")
def status(job_id):
    job = jobs.get_job(job_id)
    if job is None:
        abort(404)
    return jsonify(job.to_status_dict())


@app.route("/api/jobs/<job_id>/download")
def download(job_id):
    job = jobs.get_job(job_id)
    if job is None:
        abort(404)
    if not job.excel_path.exists():
        return jsonify({"error": "還沒有任何頻道處理完成，無法輸出"}), 409
    return send_file(
        str(job.excel_path),
        as_attachment=True,
        download_name=f"12監看_{job.broadcast_start.strftime('%Y%m%d')}_{job_id}.xlsx",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
