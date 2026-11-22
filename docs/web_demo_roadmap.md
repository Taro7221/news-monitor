# 從「上傳 demo」到「4 台同時監看、填進同一份 Excel」

這份文件說明 `web/` 目前這個 demo 的架構、為什麼現在是「一次處理一台」，
以及要演進成「4 台真的同時跑、結果都寫進同一份 Excel」需要改哪些地方。

## 現況架構（`web/jobs.py`）

一個上傳 job：

1. 使用者上傳 1~4 台的影片，存到 `out/web_jobs/{job_id}/videos/{channel}.mp4`。
2. `_run_job()` 用**一個背景 thread**、**依序**處理每一台：
   detect → vlm → `select_dominant_per_minute` + `mark_continuations_blank`
   → 產生五字標（`_build_channel_plan`）。
3. 每一台處理完，把它的 `{row: {text, low_conf, notes}}` 存進
   `job.channel_plans[channel]`（算過一次就不會重算，不會浪費 VLM 呼叫）。
4. 每次有新頻道完成，就呼叫 `_write_excel()`：從範本重新套用**目前所有**
   已快取的頻道 plan、存檔——便宜（純本機 openpyxl 操作，不呼叫 VLM），
   所以可以放心每次都整份重寫，不用自己維護「只改動哪幾格」的差異邏輯。
5. 使用者隨時可以下載 `job.excel_path`，不用等全部跑完。

這個設計已經具備「4 台結果寫進同一份 Excel」的核心——`channel_plans` 跟
`_write_excel` 本來就是以「所有已完成頻道的聯集」在運作。真正缺的只是
**「4 台同時在跑」**這一步，目前是故意依序做的。

## 要改成 4 台同時處理，具體要動哪裡

### 1. 併發模型：執行緒 vs 行程

`_run_job()` 目前是 `for ch in channels_to_run: ...` 依序跑。要同時跑，
把這個迴圈改成類似 `run_all.py` 的 `run_vlm()` 已經在用的
`ThreadPoolExecutor`：

```python
from concurrent.futures import ThreadPoolExecutor

def _run_job(job):
    roi_config = calibrate.load_config()
    channels_to_run = [ch for ch in CHANNELS if job.channels[ch]["status"] == "queued"]
    with ThreadPoolExecutor(max_workers=len(channels_to_run)) as pool:
        futures = [pool.submit(_process_one_channel, job, ch, roi_config) for ch in channels_to_run]
        for f in futures:
            f.result()  # 把例外往外丟，不要默默吞掉
```

用「執行緒」而不是「行程」是因為：這條 pipeline 大部分時間花在等網路
（VLM 呼叫），Python 的 GIL 在等 I/O 時會釋放，執行緒已經可以有效並行；
`detect.py` 的 phash 運算是 CPU-bound，4 支影片同時解碼+算 phash 會在
CPU 上競爭，但通常還是比完全依序跑快——除非機器核心數很少，才需要考慮
`ProcessPoolExecutor`（會更複雜：`job`/`channel_plans` 这些共享狀態要
改成跨行程安全的機制，例如 `multiprocessing.Manager` 或寫共享檔案）。

### 2. 同時寫同一份 Excel 的競爭問題

`_write_excel()` 已經用 `job.excel_lock`（`threading.Lock`）包起來，
所以**執行緒版本不用再改**——多個頻道同時完成時，寫入會自動排隊、不會
互相覆蓋或寫出壞檔。

如果之後改成跨「行程」而不是「執行緒」，`threading.Lock` 就不夠了，需要
換成跨行程安全的鎖（例如 `multiprocessing.Lock`，或第三方 `filelock`
套件鎖住 `output.xlsx` 這個檔案），或者乾脆讓「寫 Excel」這件事只交給
一個固定的 writer（例如一個專門的執行緒/行程，其他 worker 處理完一個
頻道後用 queue 通知它，而不是自己直接寫檔）。

### 3. 真正的瓶頸：VLM 伺服器本身能不能同時處理 4 份請求

這是最容易被忽略、但最重要的一點：**就算我們這邊 4 個頻道同時發出請求，
如果 Colab 上跑的 Ollama 只用單一個模型實例、單一張 GPU，它很可能還是
把這些請求排隊、一個一個處理**，跟我們依序呼叫在總時間上不會差太多
（甚至因為連線數變多、互相搶 context 切換，可能更慢）。

要讓「同時」真的對 VLM 階段有加速效果，需要 Colab 那邊也能同時處理多筆
請求，例如：
- 調高 Ollama 的並行設定（`OLLAMA_NUM_PARALLEL` 環境變數），讓同一個
  模型實例可以同時處理多筆請求（前提是 GPU 記憶體夠放得下）。
- 或在 Colab 開多個 Ollama 服務/模型副本，4 個頻道各打不同的 port，
  `.env` 要能讓每個頻道指定不同的 `VLM_URL`（目前整個專案是全域一個
  `VLM_URL`，要支援這個需要在 `vlm.load_vlm_env()` 加上 per-channel 覆寫）。

在還沒驗證 Colab 端真的能並行處理之前，光改前端的併發模型，對 VLM 這個
階段的實際處理時間不一定有幫助——`detect.py` 的偵測階段（純本機運算）
會確實變快，但那本來就不是這條 pipeline 最花時間的部分。

### 4. 更進一步：真正「即時」監看（不是上傳已錄好的檔案）

目前不管是 CLI 還是這個 demo，都是處理「已經錄好的一整支影片」。如果
目標是「電視台正在播、系統邊看邊填表」，架構要再往前一步：

- 影片來源要從「上傳的 mp4 檔」換成「持續成長的串流/錄影來源」（例如
  用 ffmpeg 持續把 IPTV/擷取卡訊號切成滾動的片段，或直接對一個不斷
  成長的檔案做 tailing 讀取）。`detect.iter_seconds()` 目前假設影片有
  固定終點（`cv2.VideoCapture` 讀到檔尾就結束），要改成能夠「讀到目前
  已錄到的位置就先等一下、新內容進來再繼續」的模式。
- Excel 更新頻率要從「一個頻道全部處理完才重寫一次」改成「每確定一分鐘
  的內容就寫那一列」，這樣監看人員可以邊播邊看到表格被即時填上，而不是
  等到整個小時結束才看到結果。`_write_excel()` 現在的「整份重寫」做法
  在這個情境下依然適用，只是觸發頻率要提高（例如每分鐘觸發一次，而不是
  每個頻道完成才觸發）。
- 這個情境下「資料缺口」會是常態而非例外（即時監看隨時可能因為訊號問題
  中斷），`segment.mark_continuations_blank()` 已經有處理資料缺口不重設
  anchor 的邏輯，這部分不需要大改。

這一步工作量明顯比「4 台同時處理已上傳的檔案」大一截，建議先把「4 台
同時處理上傳檔案」（上面第 1、2、3 點）做穩、也跟 Colab 那邊確認並行
處理可行之後，再評估是否要做到真正即時監看。
