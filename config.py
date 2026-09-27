# -*- coding: utf-8 -*-
"""
config.py
==========
在這裡設定你要追蹤的股票清單、資料期間，以及警報門檻。
"""

# 股票代號：yfinance 格式。台股請加 .TW（上市）或 .TWO（上櫃）。
WATCHLIST = [
    # 台股
    {"symbol": "2330.TW", "name": "台積電"},
    {"symbol": "2317.TW", "name": "鴻海"},
    {"symbol": "2327.TW", "name": "國巨"},
    {"symbol": "0050.TW", "name": "元大台灣50", "etf": True},
    # 美股 — 半導體 / 記憶體
    {"symbol": "NVDA", "name": "NVIDIA"},
    {"symbol": "AVGO", "name": "博通"},
    {"symbol": "AMD",  "name": "超微"},
    {"symbol": "INTC", "name": "英特爾"},
    {"symbol": "MU",   "name": "美光科技"},
    {"symbol": "MRVL", "name": "美滿電子科技"},
    # 美股 — 光通訊
    {"symbol": "LITE", "name": "Lumentum"},
    {"symbol": "COHR", "name": "Coherent"},
    {"symbol": "AAOI", "name": "應用光電"},
    # 美股 — 軟體 / 平台 / 硬體
    {"symbol": "AAPL", "name": "Apple"},
    {"symbol": "MSFT", "name": "微軟"},
    {"symbol": "GOOG", "name": "Google"},
    {"symbol": "AMZN", "name": "亞馬遜"},
    {"symbol": "NOW",  "name": "ServiceNow"},
    {"symbol": "PLTR", "name": "Palantir"},
    {"symbol": "DELL", "name": "戴爾"},
    {"symbol": "NOK",  "name": "諾基亞"},
    {"symbol": "BE",   "name": "Bloom Energy"},

    # ⚠️ 以下三檔上市時間太短，**無法回測**（BACKTEST_WARMUP_BARS = 250，
    #    光暖機就吃掉 250 根）。圖表與即時評分仍可產生，但回測會回報
    #    insufficient_data、門檻會退回 ALERT_THRESHOLD。等累積約 2 年
    #    日線之後再納入回測才有意義。
    #      SNDK  2025-02 自 WD 分拆，目前約 406 根（1.6 年）
    #      SPCX  2026-06 上市，目前約  73 根（0.3 年）
    #      SKHY  2026-07 美股掛牌，目前約 55 根（0.2 年）
    {"symbol": "SNDK", "name": "SanDisk"},
    {"symbol": "SPCX", "name": "SpaceX"},
    {"symbol": "SKHY", "name": "SK海力士"},
]

# ⚠️ 集中度提醒：上面 25 檔裡有 15 檔屬於 AI／半導體／光通訊同一條產業鏈
# （NVDA/AVGO/AMD/INTC/MU/MRVL/LITE/COHR/AAOI/2330/2327/DELL/PLTR/SNDK/SKHY）。
# 這對回測解讀有兩個影響：
#   1. 組合分散效果遠低於標的數量給人的印象——它們會一起漲、一起跌。
#   2. 等權買進持有這個對照組在過去 10 年會非常強，是很高的比較門檻。

# --------------------------------------------------------------------------
# 驗證池（validate.py 用，不影響每日排程）
#
# WATCHLIST 是「每天要看的」，要少而精；VALIDATION_UNIVERSE 是「用來判斷
# 這套訊號到底有沒有優勢的」，要多而分散。兩者分開的理由：5 檔標的、單一
# 市場環境完全無法區分「策略有效」與「這幾檔剛好」——一檔亮燈很可能只是
# 運氣。跨 40 檔、跨產業、跨市場看一致性，才有判斷力。
#
# 刻意混入不同產業（半導體/金融/傳產/航運/電信/消費）與大盤 ETF，避免
# 整個驗證池被單一產業的行情帶著走。
# --------------------------------------------------------------------------
VALIDATION_UNIVERSE = [
    # 台股 — 半導體 / 電子
    {"symbol": "2330.TW", "name": "台積電"},
    {"symbol": "2454.TW", "name": "聯發科"},
    {"symbol": "2308.TW", "name": "台達電"},
    {"symbol": "2317.TW", "name": "鴻海"},
    {"symbol": "2382.TW", "name": "廣達"},
    {"symbol": "2379.TW", "name": "瑞昱"},
    {"symbol": "3037.TW", "name": "欣興"},
    {"symbol": "3008.TW", "name": "大立光"},
    # 台股 — 金融 / 傳產 / 航運 / 電信
    {"symbol": "2881.TW", "name": "富邦金"},
    {"symbol": "2882.TW", "name": "國泰金"},
    {"symbol": "2891.TW", "name": "中信金"},
    {"symbol": "1301.TW", "name": "台塑"},
    {"symbol": "1303.TW", "name": "南亞"},
    {"symbol": "1216.TW", "name": "統一"},
    {"symbol": "2603.TW", "name": "長榮"},
    {"symbol": "2609.TW", "name": "陽明"},
    {"symbol": "2412.TW", "name": "中華電"},
    # 台股 — ETF
    {"symbol": "0050.TW", "name": "元大台灣50", "etf": True},
    {"symbol": "0056.TW", "name": "元大高股息", "etf": True},
    # 美股 — 科技
    {"symbol": "AAPL",  "name": "Apple"},
    {"symbol": "MSFT",  "name": "Microsoft"},
    {"symbol": "NVDA",  "name": "NVIDIA"},
    {"symbol": "GOOGL", "name": "Alphabet"},
    {"symbol": "AMZN",  "name": "Amazon"},
    {"symbol": "META",  "name": "Meta"},
    {"symbol": "AVGO",  "name": "Broadcom"},
    {"symbol": "AMD",   "name": "AMD"},
    {"symbol": "INTC",  "name": "Intel"},
    {"symbol": "CRM",   "name": "Salesforce"},
    {"symbol": "NFLX",  "name": "Netflix"},
    {"symbol": "TSLA",  "name": "Tesla"},
    # 美股 — 非科技
    {"symbol": "JPM", "name": "JPMorgan"},
    {"symbol": "V",   "name": "Visa"},
    {"symbol": "JNJ", "name": "Johnson & Johnson"},
    {"symbol": "WMT", "name": "Walmart"},
    {"symbol": "XOM", "name": "Exxon Mobil"},
    {"symbol": "KO",  "name": "Coca-Cola"},
    {"symbol": "PG",  "name": "Procter & Gamble"},
    {"symbol": "DIS", "name": "Disney"},
    # 美股 — 大盤 ETF
    {"symbol": "SPY", "name": "S&P 500 ETF", "etf": True},
    {"symbol": "QQQ", "name": "Nasdaq 100 ETF", "etf": True},
]

# --------------------------------------------------------------------------
# 交易成本（買進到賣出的「來回」總成本，佔成交金額比例）
#
# 回測如果不扣成本，等於在衡量一個不存在的策略。實測顯示這不是小數：
# horizon=5 根的平均單筆報酬多在 ±0.5% 這個量級，台股來回成本 0.47%
# 就足以把絕大部分帳面優勢吃掉。
#
# 成本拆成三個獨立來源，換券商時才知道該調哪一個：
#   手續費（券商收，可能為 0）、稅費（政府/法規收，換券商不會變）、
#   價差與滑價（市場結構造成，永遠不會是 0）。
#
# 台股：手續費 0.1425% 買賣各一次（多數券商有折扣，預設抓常見的 6 折），
#       加上賣出時的證交稅——一般股票 0.3%，ETF 只有 0.1%。
#
# 美股：Firstrade / Robinhood 這類零佣券商，股票與 ETF 佣金是 0，
#       只剩法規費：SEC fee 每百萬美元 $20.60（僅賣出，約 0.00206%）
#       與 FINRA TAF 每股 $0.000166（上限 $8.30，約 0.0001%），
#       合計約 0.003%——相對於價差幾乎可以忽略。
#
#       ⚠️ 但「零手續費」不等於「零成本」：真正的成本是**買賣價差與
#       滑價**。流動性極佳的大型股（AAPL/NVDA/SPY）價差約 1 分錢，
#       以 $200 計來回約 0.01%；再考慮零佣券商普遍採用 PFOF（訂單流
#       付款），成交價可能略差於最佳報價，這部分隱含在價差裡、不會
#       出現在對帳單上。所以預設保留 0.02% 而不是設成 0。
# --------------------------------------------------------------------------
BROKER_FEE_DISCOUNT = 0.6        # 台股券商手續費折扣；1.0 = 不打折
US_COMMISSION_FREE = True        # 美股券商是否零佣（Firstrade/Robinhood 等）
US_COMMISSION_PCT = 0.0          # 若非零佣，填單邊手續費比例
US_SPREAD_SLIPPAGE = 0.0002      # 美股價差+滑價估計（來回）；流動性差的標的應調高
US_REGULATORY = 0.00003          # SEC fee + FINRA TAF（僅賣出，約 0.003%）


def transaction_cost(symbol: str, is_etf: bool = False) -> float:
    """回傳這檔標的的來回交易成本（佔成交金額比例）。"""
    if symbol.endswith((".TW", ".TWO")):
        fee = 0.001425 * BROKER_FEE_DISCOUNT * 2         # 手續費買賣各一次
        tax = 0.001 if is_etf else 0.003                 # 證交稅：ETF 0.1% / 一般 0.3%
        return fee + tax
    commission = 0.0 if US_COMMISSION_FREE else US_COMMISSION_PCT * 2
    return commission + US_REGULATORY + US_SPREAD_SLIPPAGE

# --------------------------------------------------------------------------
# yfinance 下載參數
#
# 為什麼從「60m 合併成 4H」改成直接用日線：
#   1. 時區錯位：yfinance 的 intraday 資料帶交易所時區，對台股 09:00-13:30
#      的 5 根小時線做 4H resample，會切成 08:00(含3根) 與 12:00(含2根) 兩根
#      長度不等、且不對齊開盤的「4H」K棒。台股與美股時區還不一樣，同一個
#      WATCHLIST 裡兩者的「一根K棒」意義並不相同。
#   2. 樣本長度：intraday 資料 yfinance 只保留約 730 天，實測只能拿到
#      1444~1704 根、且全部落在 2023 年以後的單一多頭行情裡。日線沒有這個
#      限制，10 年可拿到 2433~2514 根，還涵蓋 2018 修正、2020 疫情崩盤、
#      2022 熊市——回測要能區分「策略有效」與「剛好遇到好行情」，就需要
#      這種跨市場環境的樣本。
#   3. SMC 的結構分析（swing / BOS / OB）本來就更適合日線以上的週期。
# --------------------------------------------------------------------------
DATA_PERIOD = "10y"      # 日線沒有 intraday 的 730 天限制
RAW_INTERVAL = "1d"      # 實際跟 yfinance 要的原始區間
RESAMPLE_RULE = None     # 日線不需要再合併；設成 "1wk" 之類可改成週線

# SMC 演算法參數
SWING_LOOKBACK = 3       # fractal 左右比較根數，越大代表 swing 越「大格局」，雜訊越少
EQ_TOLERANCE_PCT = 0.0015

# FVG 最小缺口寬度，以「該根K棒的 ATR 倍數」計。0 = 不過濾。
# 不過濾時，實測 32~54% 的K棒都會產生 FVG，評分裡的 FVG 項近似恆真、
# 不帶資訊量。0.5 ATR 大約濾掉一半（雜訊級的小缺口），保留真正的失衡區。
# 用 ATR 相對值而非固定百分比：實測台股的 gap/ATR 系統性高於美股，
# 固定百分比會讓不同標的的過濾強度不一致。
FVG_MIN_GAP_ATR = 0.5

# 技術指標共振參數
ADX_TREND_THRESHOLD = 20    # ADX 高於這個值才視為「趨勢有效」，DI方向才計分

# 輕量 ML 模型參數（HistGradientBoostingClassifier，見 ml_model.py）
ML_HORIZON_BARS = 5         # 預測未來幾根K棒的方向（日線下 = 1 個交易週）
ML_MIN_TRAIN_ROWS = 80      # 可訓練樣本數低於此值就不給 ML 分數（避免用太少資料硬訓練）

# 樣本外 AUC 低於此值就不給 ML 分數。0.5 = 跟擲硬幣一樣。
#
# ⚠️ 實測結果：這組特徵在 15 檔標的、日線 10 年下的樣本外 AUC 平均只有
# 0.518，只有 1 檔超過 0.55（而 15 次試驗出現 1 個好看的本來就是雜訊）。
# 也就是說，ML 這 20 分的評分權重目前**幾乎必然會被這道閘門擋掉**——
# 這是正確的行為：對隨機數字加權比不加權更糟。
#
# 這道閘門同時解決了執行時間問題：擋掉之後就不必每隔幾根K棒重訓一次。
ML_MIN_AUC = 0.55

# 直接關掉 ML 可省下絕大部分回測時間（實測 ML 佔 41 檔驗證池的 56 分鐘中
# 的絕大部分）。在找到真正有預測力的特徵之前，建議維持 False。
ML_ENABLED = False

# 評分機制參數
RECENT_BARS_FOR_SCORE = 3    # 只看最近幾根K棒內發生的訊號才計分（日線下 = 3 個交易日）

# 綜合分數 >= 此值才寄信通知。這是「回測樣本不足」時的備援門檻，不是主要判準
# （正常情況下每檔標的會用 backtest.py 算出來的自適應門檻）。
# 校準依據：對 5 檔 watchlist 標的、共 14,566 個 walk-forward 分數樣本實測，
# 分數分佈為 p50=21 / p90=43 / p95=52 / p99=60 / max=73。
# 取 50（約 p95、6% 的K棒）當備援：夠保守，但對新標的仍然可能觸發；
# 舊值 60 已經接近 p99，實際上幾乎不可能對一檔沒有回測資料的標的發出警報。
ALERT_THRESHOLD = 50

# --------------------------------------------------------------------------
# Walk-forward 回測參數（backtest.py）：每次排程執行時，都會用「不看未來」
# 的方式回測過去資料，幫每一檔標的分別找出「入場次數 vs 勝率」平衡後的
# 建議門檻，取代所有標的共用同一個 ALERT_THRESHOLD。
# --------------------------------------------------------------------------
BARS_PER_YEAR = 252                       # 一年約幾根K棒（日線；週線改 52）
                                          # 用來把總報酬換算成年化報酬
BACKTEST_HORIZON_BARS = ML_HORIZON_BARS   # 進場後看幾根K棒的結果來判斷輸贏
BACKTEST_WARMUP_BARS = 250                # 前面跳過幾根K棒（等指標/均線資料備齊）
                                          # 日線下必須 > 200，EMA200 才有意義
BACKTEST_ML_RETRAIN_EVERY = 10            # 回測時 ML 模型每隔幾根K棒重新訓練一次
# 掃描的候選門檻。上限從 85 下修到 65：實測 5 檔標的的分數 p99 只有 60、
# 最大值 73，門檻 70 以上在 14,566 個樣本裡連一筆訊號都掃不到，純粹是
# 空轉。下限維持 35（約 17% 的K棒），再往下放會讓警報變成每天洗版。
BACKTEST_CANDIDATE_THRESHOLDS = list(range(35, 70, 5))
BACKTEST_WIN_RETURN_THRESHOLD = 0.0       # 報酬率超過這個值才算「贏」（0 = 只要方向對就算贏）
BACKTEST_MIN_TRADES = 30                  # 樣本數至少要幾筆，回測結果才採信
                                          # （原本是 8：8 筆贏 5 筆的 95% 信賴區間
                                          #   大約 [24%, 91%]，等於沒有資訊）

# 挑門檻的判準是「扣成本期望值的信賴下界 > 0」，不再是勝率門檻——
# 勝率沒有考慮賠率，實測會推薦出勝率 59% 但每筆虧 0.75% 的設定。
# 因此 BACKTEST_TARGET_WIN_RATE 已移除，改由下面兩個參數控制嚴格程度。
BACKTEST_ALPHA = 0.05                     # 信賴下界的顯著水準（0.05 = 95%）

# 掃描 N 個門檻挑歷史最好的那個，本身就會高估被選中門檻的績效
# （Bailey & López de Prado, Deflated Sharpe Ratio）。開啟後會用 Šidák
# 修正把顯著水準除以試驗次數，讓下界跟著變嚴。
BACKTEST_MULTIPLE_TESTING_CORRECTION = True

# 只在「回測證明得了正期望值」(confidence == "ok") 的方向寄警報信。
# 預設 False：照樣寄，但信裡會明確標示 ⚠️ 無可證實優勢，由你自己判斷。
# 改成 True 會安靜很多——實測 5 檔標的裡，多數方向都是 no_edge。
ALERT_ONLY_WHEN_EDGE_PROVEN = False

# --------------------------------------------------------------------------
# 部位大小建議（position_sizing.py）
#
# 這是整個系統裡唯一**不依賴訊號有預測力**的產出。它量的是「這檔標的
# 對這套停損停利結構的適配度」——實測 22 檔在 2:1 結構下先碰停利的機率
# 從 25% 到 57% 不等，那個差距與訊號無關，是各檔漂移與波動結構的差異。
#
# 刻意用隨機進場的統計而非訊號進場：訊號的超額只有 +3.1 個百分點且不
# 顯著（CI [−1.7, +8.0]），用訊號統計等於把不顯著的超額當真；而隨機
# 進場的樣本可以放大到約 600 筆，估計穩定得多。
#
# 三層保守：Wilson 下界（非點估計）、1/4 Kelly、硬上限。實測把點估計
# 68% 的標的從「全 Kelly 52%」壓到 3.8%。
# --------------------------------------------------------------------------
SIZING_ENABLED = True
SIZING_RR = 2.0                  # 停利設在幾倍風險（與 Kelly 的賠率一致）
SIZING_SAMPLES = 600             # 隨機進場的模擬次數
SIZING_KELLY_DIVISOR = 4.0       # 只用 1/4 Kelly
SIZING_MAX_FRACTION = 0.20       # 單一部位上限
SIZING_MIN_SAMPLES = 100         # 少於這麼多筆就不給建議

# --------------------------------------------------------------------------
# 出場邏輯（trade_model.py）
#
# 原本的回測是「進場後固定持有 N 根K棒看方向」，等於丟掉 SMC 最有價值的
# 部分。SMC 的核心是用結構位取得不對稱的風險報酬比：停損放訊號失效處
# （Order Block 外緣），目標放對向流動性池（EQH/EQL），而不是固定 R:R。
# 業界慣例只說「放在失效區之外」，沒有標準緩衝距離，所以這裡用 ATR 倍數
# 表示，才能跨標的與跨波動度環境通用。
# --------------------------------------------------------------------------
TRADE_STOP_BUFFER_ATR = 0.25    # 停損放在 OB 外緣再往外推幾倍 ATR（避免被影線掃到）
TRADE_ATR_STOP_MULT = 1.5       # 找不到合適 OB 時，改用幾倍 ATR 當停損距離
TRADE_FALLBACK_TARGET_R = 2.0   # 找不到對向流動性池時，目標設為幾倍風險
TRADE_MIN_RR = 0.5              # 流動性池太近（RR 低於此值）就改用 R 倍數目標
TRADE_MAX_RR = 10.0             # 流動性池太遠（RR 高於此值）同上，避免不切實際
# 時間停損：超過這麼多根還沒出場就平倉。兩種出場模式的合理值差很多：
#   fixed    — 目標明確，20 根（日線約 1 個月）沒走到就代表動能不足
#   trailing — 整個重點就是讓獲利奔跑，設太短會把趨勢單提早砍掉，
#              等於退化成固定持有；250 根（約 1 年）只是最後的保險
TRADE_MAX_HOLDING_BARS = 20
TRADE_TRAILING_MAX_HOLDING_BARS = 250

# 出場模式：
#   "fixed"    — 固定目標（對向流動性或 R 倍數）+ 時間停損
#   "trailing" — 不設目標，只用移動停損，讓獲利奔跑
#
# 為什麼提供 trailing：固定目標的組合回測年化只有等權買進持有的三分之一
# （+9~12.9% vs +29.2%），即使在場時間已拉到 99%。原因是固定目標在趨勢
# 行情裡會砍掉贏家——下檔有保護，但上檔也被截斷了。
# 實測（41檔、日線10年、20個並存部位、已做逐根市價評估）：
#   固定目標 2R    年化 +11.0%  回撤 15%  Calmar 0.76
#   移動停損 2ATR  年化 +12.0%  回撤  9%  Calmar 1.39  ← 預設
#   移動停損 3ATR  年化 +13.3%  回撤 16%  Calmar 0.85
#   等權買進持有   年化 +29.2%  回撤 39%  Calmar 0.75
# 移動停損 2ATR 在報酬與回撤上都優於固定目標，所以設為預設。
# --------------------------------------------------------------------------
# 進場方式
#
# 舊版用「訊號當根的收盤價」進場，但那個分數就是用同一個收盤價算的——
# 你看到訊號時市場已經收盤，下不了那個價。改成隔天進場才是做得到的。
#
# 實測（661 筆配對）：隔天市價 +2.400% vs 隔天限價−0.5ATR +2.509%，
# 配對差異 95% CI [−0.131%, +0.373%] **不顯著**。兩者報酬無差別，
# 選限價的理由是執行面：收盤後掛好單、隔天不必盯盤。
#
# ⚠️ ENTRY_FALLBACK 不要改成 "skip"：實測同樣限價距離，未成交改市價
# 總報酬 +1553%、未成交放棄只有 +1137%（差 27%），因為放棄掉的那批
# 包含了最強的走勢。
# --------------------------------------------------------------------------
ENTRY_LIMIT_OFFSET_ATR = 0.5    # 限價掛在訊號收盤下方幾倍 ATR；0 = 掛在收盤價
ENTRY_LIMIT_VALID_BARS = 1      # 限價有效幾根K棒（實測 1 根優於 3 根）
ENTRY_FALLBACK = "market"       # 未成交時："market" 改市價 / "skip" 放棄

TRADE_EXIT_MODE = "trailing"
TRADE_TRAIL_ATR_MULT = 2.0      # 移動停損距離（幾倍 ATR）；越大越能抱住趨勢
                                # 但實測 3ATR/5ATR 的 Calmar 反而較差

# --------------------------------------------------------------------------
# 警報去重：SMC 訊號成立後通常會連續好幾根K棒維持成立，不去重的話同一個
# 訊號會天天寄一封一樣的信，很快就會被當成雜訊忽略——那比不寄還糟。
# --------------------------------------------------------------------------
ALERT_COOLDOWN_DAYS = 5          # 同一標的同方向，幾天內不重複通知
ALERT_RENOTIFY_SCORE_JUMP = 15   # 但分數變化超過這麼多分，視為新訊號重新通知
ALERT_STATE_JSON = "alert_state.json"

# --------------------------------------------------------------------------
# 回測快取（backtest_cache.py）
#
# 完整 walk-forward 是每日流程最重的一步（25 檔 × 約 2,250 根，實測
# 20~30 分鐘），但結果每天幾乎不變——多一根 K 棒不會改變用 2,250 根算
# 出來的門檻。改成每 N 天重算一次，平日執行可降到 1 分鐘以內。
#
# 快取鍵含「設定指紋」：任何影響回測的參數變動都會自動失效，不需要
# 記得手動清快取。
# --------------------------------------------------------------------------
BACKTEST_CACHE_JSON = "backtest_cache.json"
BACKTEST_CACHE_DAYS = 7          # 幾天重算一次；設 0 等於停用快取

# 輸出的網頁檔名（GitHub Pages 會直接讀取根目錄的 index.html）
OUTPUT_HTML = "index.html"
BACKTEST_RESULTS_JSON = "backtest_results.json"  # 回測明細另存一份，方便追蹤歷史演變
