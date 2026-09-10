# SMC 股票分析儀表板

用 Python + yfinance 計算 Smart Money Concepts（市場結構 BOS/CHoCH、Order Block、
Fair Value Gap、流動性池、溢價/折價區），畫成互動式 K 線圖並產出靜態網頁，
透過 GitHub Actions 每日自動更新並在訊號分數過高時寄信通知。

## 目錄結構

```
smc-stock-dashboard/
├── smc/
│   ├── __init__.py
│   └── analyzer.py        # 階段一：SMC 核心演算法
├── plot_report.py          # 階段二：Plotly 繪圖 + index.html 產生
├── notifier.py              # 階段三：評分機制 + Gmail 寄信
├── main.py                  # 主流程（本機測試 / GitHub Actions 都跑這支）
├── config.py                # 股票清單、參數、警報門檻設定
├── requirements.txt
└── .github/workflows/update_stocks.yml   # 排程設定
```

## 本機測試

```bash
pip install -r requirements.txt
python main.py --dry-run     # 只產生 index.html，不寄信
open index.html              # 或直接用瀏覽器打開檢查
```

先確認：
1. `config.py` 裡的 `WATCHLIST` 是你想追蹤的股票（台股記得加 `.TW` / `.TWO`）。
2. 圖表、分數、位階看起來合理，再進行下一步部署。

## 部署到 GitHub Pages + Actions

### 1. 建立 Gmail 應用程式密碼（App Password）
不要用 Gmail 登入密碼，Google 已停用「較不安全應用程式」存取：
1. 前往 Google 帳戶 → 安全性 → 開啟「兩步驟驗證」
2. 安全性 → 應用程式密碼 → 建立一組給「郵件」用的 16 碼密碼
3. 複製下來，等一下會貼到 GitHub Secrets

### 2. 設定 GitHub Secrets
到你的 repo → Settings → Secrets and variables → Actions → New repository secret，新增：

| Secret 名稱 | 內容 |
|---|---|
| `GMAIL_SENDER` | 你的 Gmail 帳號，例如 `you@gmail.com` |
| `GMAIL_APP_PASSWORD` | 上一步產生的 16 碼應用程式密碼 |
| `ALERT_RECIPIENT` | 收信信箱（可以跟 GMAIL_SENDER 相同） |

### 3. 開啟 GitHub Pages
Settings → Pages → Build and deployment → Source 選 `Deploy from a branch`，
Branch 選你 push 這個 repo 的分支（例如 `main`）＋ 目錄選 `/ (root)`。
之後每次 Actions 把新的 `index.html` push 回這個分支，Pages 就會自動重新部署。

### 4. 確認排程設定
`.github/workflows/update_stocks.yml` 預設在 **每天 UTC 10:00（台北時間 18:00）** 執行。
也可以到 repo 的 Actions 分頁，手動點 `Run workflow` 立刻測試一次（不用等排程時間）。

## 你可能想接著調整的地方

- **調整訊號權重／門檻**：`notifier.py` 的 `score()`、`config.py` 的 `ALERT_THRESHOLD`。
  目前評分滿分 100，拆成三塊：SMC 結構 60 分、技術指標共振 20 分、輕量 ML
  模型機率 20 分，權重都在 `notifier.py` 檔頭的註解跟 `config.py` 裡可調。
- **調整 swing 靈敏度**：`config.py` 的 `SWING_LOOKBACK`，數字越大代表只抓「大格局」的
  swing，雜訊（假訊號）越少，但反應會比較慢。
- **多時間框架（HTF 定方向、LTF 找進場）**：目前為求 GitHub Actions 排程穩定性，
  全部使用同一個時間框架（目前設定為 4H，由小時線 resample 而來）。如果要做真正
  的多時間框架分析，可以另外對同一標的抓日線資料再跑一次 `SMCAnalyzer`，
  再把兩者的分數加權合併。
- **Order Block 定義更嚴謹版本**：目前用「造成突破的最後一根反向K棒」的最寬區間
  （含影線）當作 OB，你也可以改成只用實體(body)範圍，或加上「該K棒之後必須
  有一段強勢位移（displacement）」的條件，減少雜訊 OB。

## 技術指標與 ML 模型（新增）

### 技術指標共振（`indicators.py`）
在 SMC 結構之外，額外算了業界最常拿來做「多指標確認」的經典技術指標：
RSI(14)、MACD(12,26,9)、EMA(20/50/200)、ADX+DI(14)、ATR(14)、OBV。
評分邏輯只在 RSI/MACD/均線排列/ADX+DI 的方向「一致」時才加分，且 ADX 未過
`ADX_TREND_THRESHOLD`(預設20，代表盤整、沒有值得跟的趨勢)時，DI方向那項不計分。
圖表下方也會多一張 RSI/MACD 副圖，方便你肉眼覆核分數是怎麼來的，而不是只信
一個黑盒子數字。

**誠實的免責提醒**：多篇學術研究（含針對台股/恆生/日經的實證論文）指出，單獨
使用 RSI 或 MACD 長期不必然顯著贏過單純的 buy-and-hold。這裡把它們當成
「跟 SMC 結構訊號互相驗證的濾網」，而不是宣稱這些指標本身有穩定的超額報酬。

### 輕量 ML 模型（`ml_model.py`）
用 `scikit-learn` 的 `GradientBoostingClassifier`，每次 GitHub Actions 執行時
用當下抓到的歷史資料**現場重新訓練**，預測「未來 `ML_HORIZON_BARS` 根K棒後，
收盤價是否上漲」的機率，機率偏離 50% 越多，加分越多（上限 20分）。

選這個模型是因為：GitHub Actions 免費 runner 只有 CPU、沒有 GPU，執行時間也
有限制；多篇量化實證研究（Indonesia/Poland/Korea 股市）都顯示梯度提升樹
（Gradient Boosting / XGBoost 系列）在「股價方向分類」任務上，是 CPU 訓練
最快、實務中相對穩定的模型類型，複雜度遠低於 LSTM 等深度學習模型，
也不需要另外存模型權重（每次都重新訓練，架構最單純、不會有版本漂移問題）。

**局限（務必知道）**：訓練樣本通常只有幾百到一千多筆、每天重新訓練，模型容易
overfit，今天的機率跟明天可能差異不小，本身不構成穩定的預測能力保證。
資料量不足（`ML_MIN_TRAIN_ROWS`，預設80筆）或標籤嚴重失衡時，程式會自動
不給這項分數，而不是硬產生一個不可靠的數字。

相關參數都在 `config.py`：`ML_ENABLED`、`ML_HORIZON_BARS`、`ML_MIN_TRAIN_ROWS`、
`ADX_TREND_THRESHOLD`。

## 免責聲明
本專案僅為技術指標的程式化實作與教學示範，所有分數、訊號、圖表僅供研究參考，
不構成任何投資建議，請自行審慎判斷並承擔交易風險。
