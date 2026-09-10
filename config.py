# -*- coding: utf-8 -*-
"""
config.py
==========
在這裡設定你要追蹤的股票清單、資料期間，以及警報門檻。
"""

# 股票代號：yfinance 格式。台股請加 .TW（上市）或 .TWO（上櫃）。
WATCHLIST = [
    { "symbol": "2330.TW", "name": "台積電" },
    { "symbol": "2317.TW", "name": "鴻海" },
    { "symbol": "0050.TW", "name": "元大台灣50" },
    { "symbol": "AAPL", "name": "Apple" },
    { "symbol": "NVDA", "name": "NVIDIA" },
    { "symbol": "BE", "name": "Bloom Energy" },
    { "symbol": "MU", "name": "美光科技" },
    { "symbol": "SNDK", "name": "SanDisk" },
    { "symbol": "LITE", "name": "Lumentum" },
    { "symbol": "SKHY", "name": "SK海力士" },
    { "symbol": "SPCX", "name": "SpaceX" },
    { "symbol": "2327.TW", "name": "國巨" },
    { "symbol": "MSFT", "name": "微軟" },
    { "symbol": "PLTR", "name": "Palantir" },
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
ALERT_THRESHOLD = 60         # 綜合分數 >= 此值才寄信通知

# 輸出的網頁檔名（GitHub Pages 會直接讀取根目錄的 index.html）
OUTPUT_HTML = "index.html"
