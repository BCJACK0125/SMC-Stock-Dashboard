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
# yfinance 沒有原生的 4h 區間，所以做法是抓「小時線」(60m) 回來，
# 再用 pandas resample 把它合併成 4 小時一根。
# 注意：yfinance 對 60m 這種 intraday 資料只保留最近約 730 天（~2年），
# 所以 DATA_PERIOD 不能設超過這個範圍，否則 yfinance 會自動截斷。
DATA_PERIOD = "730d"     # intraday(60m) 資料 yfinance 最多只給到這個長度
RAW_INTERVAL = "60m"     # 實際跟 yfinance 要的原始區間
RESAMPLE_RULE = "4h"     # 把 RAW_INTERVAL 合併成這個週期；不想合併就設成 None

# 合併時要注意股市不是 24 小時交易，resample 的區塊邊界不會完全對齊
# 「開盤第一根算一組」的直覺分法，會有一點誤差，但對 SMC 結構分析影響不大。

# SMC 演算法參數
SWING_LOOKBACK = 3       # fractal 左右比較根數，越大代表 swing 越「大格局」，雜訊越少
EQ_TOLERANCE_PCT = 0.0015

# 技術指標共振參數
ADX_TREND_THRESHOLD = 20    # ADX 高於這個值才視為「趨勢有效」，DI方向才計分

# 輕量 ML 模型參數（GradientBoostingClassifier，見 ml_model.py）
ML_ENABLED = True
ML_HORIZON_BARS = 5         # 預測未來幾根K棒的方向（4H下，5根約等於1個交易日）
ML_MIN_TRAIN_ROWS = 80      # 可訓練樣本數低於此值就不給 ML 分數（避免用太少資料硬訓練）

# 評分機制參數
RECENT_BARS_FOR_SCORE = 6    # 只看最近幾根K棒內發生的訊號才計分（4H下，6根約等於1個交易日）
ALERT_THRESHOLD = 60         # 綜合分數 >= 此值才寄信通知（沒有足夠回測資料時的預設/備援門檻）

# --------------------------------------------------------------------------
# Walk-forward 回測參數（backtest.py）：每次排程執行時，都會用「不看未來」
# 的方式回測過去資料，幫每一檔標的分別找出「入場次數 vs 勝率」平衡後的
# 建議門檻，取代所有標的共用同一個 ALERT_THRESHOLD。
# --------------------------------------------------------------------------
BACKTEST_HORIZON_BARS = ML_HORIZON_BARS   # 進場後看幾根K棒的結果來判斷輸贏
BACKTEST_WARMUP_BARS = 90                 # 前面跳過幾根K棒（等指標/均線資料備齊）
BACKTEST_ML_RETRAIN_EVERY = 10            # 回測時 ML 模型每隔幾根K棒重新訓練一次
BACKTEST_CANDIDATE_THRESHOLDS = list(range(35, 90, 5))  # 掃描的候選門檻
BACKTEST_WIN_RETURN_THRESHOLD = 0.0       # 報酬率超過這個值才算「贏」（0 = 只要方向對就算贏）
BACKTEST_MIN_TRADES = 8                   # 樣本數至少要幾筆，回測結果才採信
BACKTEST_TARGET_WIN_RATE = 0.55           # 期望達到的最低歷史勝率

# 輸出的網頁檔名（GitHub Pages 會直接讀取根目錄的 index.html）
OUTPUT_HTML = "index.html"
BACKTEST_RESULTS_JSON = "backtest_results.json"  # 回測明細另存一份，方便追蹤歷史演變
