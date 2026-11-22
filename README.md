# 新聞監看自動填表（三立 S01）

監看 FTV（民視）、TVBS-N、東森（EBC）、華視（CTS）四台的午間新聞錄影，自動判讀
畫面上的主標題 bar、產生五字標，並填入「12監看」Excel 範本——全流程不需要任何
人工步驟；VLM 判斷不確定時給最佳結果，並把需要人工覆核的儲存格標黃、加註解。

## 這個專案在做什麼

輸入是四支側錄影片（同一小時、四台）；輸出是一份填好「12監看」分頁四台
五字標欄位的 Excel。中間經過：

```
detect.py   逐秒抽幀，裁出 headline_bar 區域，用 phash 偵測「畫面換了」的事件
    ↓
vlm.py      把事件幀連同整張畫面送給 VLM（Ollama，經 ngrok），判讀 screen_type
            （新聞/廣告/宣傳...）跟主標題文字
    ↓
segment.py  整理成「這一分鐘內比例最高的新聞是哪一則」，同一則新聞橫跨多分鐘
            時依範本慣例留白、不重複填；呼叫 VLM 把代表標題濃縮成五字標
    ↓
fill.py     複製 Excel 範本，只寫入四台五字標欄位，低信心度儲存格標黃+加註解
    ↓
evaluate.py 有人工標記的 ground truth 時，算換題準確率/廣告 precision-recall
```

`run_all.py` 把以上四步串成一行指令；`web/` 是給非技術使用者用的上傳式 demo
（見下方「網頁版 demo」）。

## 專案結構

```
src/            五個核心模組（detect/vlm/segment/fill/evaluate），各自可獨立用 CLI 跑
run_all.py      一行指令跑完整個流程
web/            Flask 上傳式 demo（app.py 路由、jobs.py 背景處理、templates/）
tools/          環境健檢、ROI 校準/預覽、單張幀擷取等輔助小工具
config/         channels.yaml：四台 headline_bar/ticker/logo 的 ROI 座標
data/           影片（data/videos/）、Excel 範本（data/excel/）—— 不進版控
out/            所有中間結果（jsonl）、輸出 Excel、log —— 不進版控
colab/          Colab 端架設 Ollama + ngrok 的筆記本程式碼
docs/           範本格式說明、效能調校紀錄、網頁 demo 演進規劃
tests/          215 個 pytest，對應每個模組
```

## 安裝

需要 Python 3.9+ 和 ffmpeg（`opencv-python` 讀影片需要）。

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # 填入 VLM_URL / VLM_USER / VLM_PASS / VLM_MODEL
python tools/check_env.py     # 健檢：Python 版本、ffmpeg、套件、影片、Excel、VLM 連線
```

VLM 是一個跑在 Colab 上、透過 ngrok 對外的 Ollama 服務（視覺語言模型，用來讀
畫面上的文字）——啟動方式見 `colab/colab_code.txt`。把 Colab 印出的 ngrok 網址
填進 `.env` 的 `VLM_URL`，帳密要跟 Colab 設定的一致。

影片放進 `data/videos/{ftv,tvbs,ebc,cts}.mp4`，範本放在
`data/excel/template.xlsx`。

## 跑完整流程（CLI）

```bash
# 全量：4 台、整小時
python run_all.py --use-vlm-labels --use-vlm-judge

# 先用 15 分鐘測試（避免跑整小時才發現哪裡要調）
python run_all.py --tag _test15 --start-second 1800 --duration-seconds 900 \
    --use-vlm-labels --use-vlm-judge

# 影片/VLM 判讀已經有了，只想重新填表
python run_all.py --tag _test15 --skip-detect --skip-vlm --use-vlm-labels --use-vlm-judge
```

`--use-vlm-labels` 是真的呼叫 VLM 把標題濃縮成五字標（不加這個旗標只會用
關鍵字擷取的退路，品質較差，但不用等 VLM）；`--use-vlm-judge` 讓模糊地帶的
「這是不是同一則新聞」交給 VLM 語意判斷，而不是保守地當作換則。

每個模組也能單獨跑（例如只想重新校準 ROI、只想重跑 segment 不重打 VLM），
細節看各模組檔頭的 docstring。

## 網頁版 demo

給不想碰 CLI 的人用：上傳 1~4 台的側錄影片，背景依序處理，隨時可以下載
目前已完成頻道的 Excel。

```bash
python web/app.py
```

瀏覽器打開 http://127.0.0.1:5000。目前是「一次處理一台」，之後要改成 4 台
同時監看、填進同一份 Excel 的具體作法，寫在
[`docs/web_demo_roadmap.md`](docs/web_demo_roadmap.md)。

## 測試

```bash
python -m pytest tests/ -q
```

## 相關文件

- [`docs/template_spec.md`](docs/template_spec.md) — Excel 範本欄位對照、留白/廣告/
  同分鐘多則的填表慣例
- [`docs/perf.md`](docs/perf.md) — VLM 呼叫效能調校紀錄
- [`docs/web_demo_roadmap.md`](docs/web_demo_roadmap.md) — 從「上傳 demo」演進到
  「4 台同時即時監看」要改哪裡
- [`CLAUDE.md`](CLAUDE.md) — 專案規則（只能動哪些欄位、不可有人工步驟等）
