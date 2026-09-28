# -*- coding: utf-8 -*-
"""
backtest_cache.py
==================
把回測結果快取起來，讓每日排程不必重跑完整的 walk-forward。

為什麼值得做：
    完整回測是每日流程裡最重的一步——25 檔 × 約 2,250 根 K 棒，每根都要
    建 AsOfView、算評分、模擬一筆交易。本機實測 22 檔約 10 分鐘，
    GitHub Actions 的 runner 更慢，估計 20~30 分鐘。

    但**它的結果每天幾乎不會變**：多一根 K 棒不會改變用 2,250 根算出來
    的建議門檻。每天重算等於花 20 分鐘得到跟昨天一樣的數字。

    改成「每 N 天才重算一次」之後，平日只需要算當下分數與進場計畫，
    執行時間可以降到 1 分鐘以內。

⚠️ 快取失效的判斷不能只看日期：
    如果改了 config（門檻範圍、出場模式、成本假設…），舊的回測結果就
    不再對應現在的設定，必須強制重算。所以快取鍵包含**設定指紋**——
    任何影響回測的參數變動都會讓快取自動失效。

    這比「記得手動清快取」可靠得多：人會忘記，雜湊不會。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

# 影響回測結果的設定項。改動其中任何一個都應該讓快取失效。
_CONFIG_KEYS = (
    "SWING_LOOKBACK", "EQ_TOLERANCE_PCT", "FVG_MIN_GAP_ATR",
    "RECENT_BARS_FOR_SCORE", "ADX_TREND_THRESHOLD",
    "ALERT_THRESHOLD", "BACKTEST_CANDIDATE_THRESHOLDS",
    "BACKTEST_HORIZON_BARS", "BACKTEST_WARMUP_BARS",
    "BACKTEST_MIN_TRADES", "BACKTEST_ALPHA",
    "BACKTEST_MULTIPLE_TESTING_CORRECTION",
    "ML_ENABLED", "ML_HORIZON_BARS", "ML_MIN_AUC",
    "TRADE_EXIT_MODE", "TRADE_TRAIL_ATR_MULT",
    "TRADE_STOP_BUFFER_ATR", "TRADE_ATR_STOP_MULT",
    "TRADE_MAX_HOLDING_BARS", "TRADE_TRAILING_MAX_HOLDING_BARS",
    "ENTRY_LIMIT_OFFSET_ATR", "ENTRY_LIMIT_VALID_BARS", "ENTRY_FALLBACK",
    "BROKER_FEE_DISCOUNT", "US_SPREAD_SLIPPAGE",
)


# 回測結果的欄位格式版本。設定沒變、但結果的**結構**改了時（例如交易明細
# 開始帶成交價），舊快取雖然「新鮮」卻缺欄位，必須一起失效。
# 改動 run_symbol_backtest 回傳內容的形狀時要 +1。
_RESULT_SCHEMA = 2


def config_fingerprint(config) -> str:
    """把影響回測的設定雜湊成一個短字串。"""
    payload = {k: getattr(config, k, None) for k in _CONFIG_KEYS}
    payload["__schema__"] = _RESULT_SCHEMA
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}          # 壞掉的快取視為沒有快取，不該讓排程整個失敗


def save(path: str, cache: dict) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception:
        return False


def is_fresh(entry: dict, fingerprint: str, max_age_days: int,
             now: Optional[datetime] = None) -> bool:
    """快取是否還能用：設定沒變、且未超過保鮮期。"""
    if not entry or entry.get("fingerprint") != fingerprint:
        return False
    try:
        ts = datetime.fromisoformat(entry["computed_at"])
    except Exception:
        return False
    now = now or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() / 86400 < max_age_days


def get_or_compute(
    cache: dict, symbol: str, config, compute: Callable[[], dict],
    *, max_age_days: int = 7, now: Optional[datetime] = None,
) -> tuple:
    """
    回傳 (回測結果, 是否來自快取)。

    compute 是個 callable，只有在真的需要重算時才會被呼叫——這樣呼叫端
    不必自己判斷就能省下完整的 walk-forward。
    """
    fp = config_fingerprint(config)
    entry = cache.get(symbol)
    if is_fresh(entry, fp, max_age_days, now=now):
        return entry["result"], True

    result = compute()
    cache[symbol] = {
        "fingerprint": fp,
        "computed_at": (now or datetime.now(timezone.utc)).isoformat(),
        "result": result,
    }
    return result, False


def restore_timestamps(result: dict) -> dict:
    """
    JSON 會把交易的 Timestamp 存成字串，還原成 pandas Timestamp，
    否則圖表標記那邊的 `ts in df.index` 比對會全部落空。
    """
    for t in result.get("trades") or []:
        for k in ("signal_ts", "entry_ts", "exit_ts"):
            if isinstance(t.get(k), str):
                try:
                    t[k] = pd.Timestamp(t[k])
                except Exception:
                    pass
    return result
