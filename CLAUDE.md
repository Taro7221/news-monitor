# 新聞監看自動填表（三立 S01）

- 影片：data/videos/{ftv,tvbs,ebc,cts}.mp4，2026/11/06 12:00–13:00；set_ref.mp4 僅供參考
- Excel：data/excel/template.xlsx（填這份）、example.xlsx（10/26 範例，只參考格式）
- VLM：Colab 上的 Ollama，經 ngrok，設定讀 .env；本機無 GPU，影像處理只用 CPU
- 只寫入「12監看」分頁四台五字標欄位，其他儲存格與格式不可動
- 全流程不可有人工步驟；不確定就給最佳結果並寫 out/logs
- 中間結果一律存 jsonl；每完成一個模組就寫 pytest 並 commit
