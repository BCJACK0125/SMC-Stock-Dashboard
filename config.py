# -*- coding: utf-8 -*-
"""
config.py
==========
在這裡設定你要追蹤的股票清單、資料期間，以及警報門檻。
"""

# 股票代號：yfinance 格式。台股請加 .TW（上市）或 .TWO（上櫃）。
WATCHLIST = [
    {"symbol": "2330.TW", "name": "台積電"},
    {"symbol": "2317.TW", "name": "鴻海"},
    {"symbol": "0050.TW", "name": "元大台灣50"},
    {"symbol": "AAPL",    "name": "Apple"},
    {"symbol": "NVDA",    "name": "NVIDIA"},
]

# yfinance 下載參數
DATA_PERIOD = "2y"       # 抓多久的歷史資料
DATA_INTERVAL = "1d"     # 時間框架：日線（配合每日執行一次的排程最穩定）

# SMC 演算法參數
SWING_LOOKBACK = 3       # fractal 左右比較根數，越大代表 swing 越「大格局」，雜訊越少
EQ_TOLERANCE_PCT = 0.0015

# 評分機制參數
RECENT_BARS_FOR_SCORE = 5    # 只看最近幾根K棒內發生的訊號才計分
ALERT_THRESHOLD = 60         # 綜合分數 >= 此值才寄信通知

# 輸出的網頁檔名（GitHub Pages 會直接讀取根目錄的 index.html）
OUTPUT_HTML = "index.html"
