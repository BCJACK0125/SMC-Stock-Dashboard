# -*- coding: utf-8 -*-
"""
positions.py
=============
把「移動停損 2×ATR」這種**規則描述**換成**今天的實際價格**。

為什麼需要這個模組：
    排程每天跑一次，而移動停損是一個遞推狀態——它只取決於
    (進場價, 進場日, 之後的最高價, 目前 ATR)，全部都是已知的。
    既然每天都算得出來，就沒有理由只告訴使用者規則、讓他自己算。

    使用者要做的只是在 positions.json 記下實際成交的進場，
    之後每天的停損價由這裡算給他。

positions.json 格式（手動維護，欄位少到可以用手機改）：
    [
      {"symbol": "NVDA", "entry_date": "2026-09-15", "entry_price": 178.5},
      {"symbol": "MU",   "entry_date": "2026-08-02", "entry_price": 96.2,
       "stop": 88.0}
    ]
    stop 省略時用「進場日收盤 − trail_mult × 當時 ATR」當初始停損。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd


def load(path: str) -> List[Dict]:
    """讀取持倉檔。不存在或格式壞掉都回空清單——不能讓它中斷整場排程。"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return [p for p in data if isinstance(p, dict) and p.get("symbol")]


def track(position: Dict, df: pd.DataFrame, atr: pd.Series,
          trail_mult: float) -> Optional[Dict]:
    """
    算出這筆持倉「今天」的移動停損價。

    移動停損用**當前** ATR 而不是進場時的 ATR：波動放大時停損跟著放寬，
    才不會在正常震盪中被掃出去。因為取 max()，停損仍然只升不降。

    回傳 None 表示這筆資料無法對應到行情（代號打錯、進場日還沒有K棒）。
    """
    entry_price = position.get("entry_price")
    if not entry_price or entry_price <= 0:
        return None
    try:
        entry_date = pd.Timestamp(position["entry_date"])
    except (KeyError, ValueError, TypeError):
        return None

    idx = df.index
    if getattr(idx, "tz", None) is not None and entry_date.tz is None:
        entry_date = entry_date.tz_localize(idx.tz)
    after = df[idx >= entry_date]
    if after.empty:
        return None

    atr_now = float(atr.iloc[-1]) if len(atr) and pd.notna(atr.iloc[-1]) else 0.0
    peak = float(after["High"].max())
    last = float(df["Close"].iloc[-1])

    initial_stop = position.get("stop")
    if initial_stop is None:
        initial_stop = entry_price - trail_mult * atr_now
    initial_stop = float(initial_stop)

    stop = max(initial_stop, peak - trail_mult * atr_now) if atr_now > 0 else initial_stop
    risk = entry_price - initial_stop

    return {
        "symbol": position["symbol"],
        "entry_date": str(entry_date.date()),
        "entry_price": float(entry_price),
        "initial_stop": initial_stop,
        "stop": stop,
        "peak": peak,
        "last": last,
        "bars_held": len(after),
        # 停損已經爬過進場價 = 這筆最差也是打平出場
        "locked_in": stop >= entry_price,
        "unrealized_pct": last / entry_price - 1,
        "unrealized_r": (last - entry_price) / risk if risk > 0 else None,
        "stop_distance_pct": (last - stop) / last if last > 0 else None,
        # 今天就該出場了：收盤已經跌破停損（盤中應該已經觸發）
        "breached": last <= stop,
    }


def describe(t: Dict) -> str:
    """一行純文字摘要，郵件用。"""
    lock = "（已鎖定獲利）" if t["locked_in"] else ""
    warn = "  ⚠️ 已跌破，應出場" if t["breached"] else ""
    return (f"{t['symbol']}：停損 {t['stop']:.2f}{lock}"
            f"｜現價 {t['last']:.2f}（{t['unrealized_pct']:+.1%}）"
            f"｜距停損 {t['stop_distance_pct']:.1%}{warn}")
