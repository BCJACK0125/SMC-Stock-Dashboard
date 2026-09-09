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
- **調整 swing 靈敏度**：`config.py` 的 `SWING_LOOKBACK`，數字越大代表只抓「大格局」的
  swing，雜訊（假訊號）越少，但反應會比較慢。
- **多時間框架（HTF 定方向、LTF 找進場）**：目前為求 GitHub Actions 排程穩定性，
  全部使用日線（1d）。如果要做真正的多時間框架分析，可以另外對同一標的抓
  4H / 1H 資料各跑一次 `SMCAnalyzer`，再把兩者的分數加權合併。
- **Order Block 定義更嚴謹版本**：目前用「造成突破的最後一根反向K棒」的最寬區間
  （含影線）當作 OB，你也可以改成只用實體(body)範圍，或加上「該K棒之後必須
  有一段強勢位移（displacement）」的條件，減少雜訊 OB。

## 免責聲明
本專案僅為技術指標的程式化實作與教學示範，所有分數、訊號、圖表僅供研究參考，
不構成任何投資建議，請自行審慎判斷並承擔交易風險。
