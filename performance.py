# -*- coding: utf-8 -*-
"""
performance.py
===============
把「一連串交易」換算成「這樣做到底會不會賺錢」。

為什麼單看期望值不夠：
    每筆 +2% 的期望值聽起來很好，但它沒有回答三個實際問題——
      1. 一年會有幾筆？只有 3 筆的話，年化報酬其實很低。
      2. 複利之後呢？連續虧損會讓後面的本金變小，算術平均會高估。
      3. 贏得過「什麼都不做、直接買進持有」嗎？如果贏不過，那這套系統
         的價值就只剩下「波動比較小」，而不是「賺比較多」。

    實測就出現過這種落差：某檔標的的訊號每筆期望值是正的，但因為進場
    次數少、又常常空手，累計報酬遠輸給買進持有。

    另外，只在有訊號時進場代表**大部分時間空手**。空手期間的機會成本
    在「每筆期望值」裡完全看不到，只有在權益曲線上才會現形。

⚠️ 這裡的模擬假設：
    - 每次進場投入全部資金、同一時間只持有一個部位（不做資金配置）。
    - 報酬已扣來回交易成本，但沒有計入滑價、跳空、稅務級距。
    - 買進持有的對照沒有扣任何成本（對它有利），所以這是保守比較。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def equity_curve(trade_returns: Sequence[float]) -> np.ndarray:
    """把單筆報酬序列複利成權益曲線（起點 1.0）。"""
    r = np.asarray(list(trade_returns), dtype=float)
    if r.size == 0:
        return np.array([1.0])
    return np.concatenate([[1.0], np.cumprod(1.0 + r)])


def max_drawdown(curve: np.ndarray) -> float:
    """權益曲線的最大回撤（正值，0.25 = 最深跌掉 25%）。"""
    if curve.size < 2:
        return 0.0
    peak = np.maximum.accumulate(curve)
    return float((1.0 - curve / peak).max())


def cagr(total_return: float, years: float) -> Optional[float]:
    """由總報酬與年數換算年化報酬率。"""
    if years <= 0 or total_return <= -1.0:
        return None
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def summarize_performance(
    trade_returns: Sequence[float],
    bars_held: Sequence[int],
    total_bars: int,
    bars_per_year: float,
    buy_hold_return: Optional[float] = None,
    buy_hold_max_drawdown: Optional[float] = None,
) -> dict:
    """
    trade_returns : 每筆交易「已扣成本」的報酬率
    bars_held     : 每筆交易實際持有幾根K棒（用來算在場時間佔比）
    total_bars    : 回測期間總共幾根K棒
    bars_per_year : 一年約有幾根K棒（日線約 252）
    buy_hold_return : 同期間買進持有的總報酬，用來對照
    """
    r = np.asarray(list(trade_returns), dtype=float)
    held = np.asarray(list(bars_held), dtype=float)
    years = total_bars / bars_per_year if bars_per_year else 0.0

    curve = equity_curve(r)
    total = float(curve[-1] - 1.0)

    out = {
        "n_trades": int(r.size),
        "years": round(years, 2),
        "trades_per_year": float(r.size / years) if years > 0 else None,
        "total_return": total,
        "cagr": cagr(total, years),
        "max_drawdown": max_drawdown(curve),
        # 在場時間佔比：只在有訊號時進場，代表大部分時間是空手的。
        # 這個數字越低，「贏過買進持有」就越難，但承擔的市場風險也越小。
        "time_in_market": float(held.sum() / total_bars) if total_bars else None,
        "buy_hold_return": buy_hold_return,
        "buy_hold_cagr": cagr(buy_hold_return, years) if buy_hold_return is not None else None,
        "buy_hold_max_drawdown": buy_hold_max_drawdown,
    }

    if buy_hold_return is not None:
        out["beats_buy_hold"] = total > buy_hold_return
        out["excess_vs_buy_hold"] = total - buy_hold_return

        # 風險調整後的對照用 Calmar（年化報酬 ÷ 最大回撤）。
        # 不用「總報酬 ÷ 回撤」是因為總報酬會隨期間長度複利放大，回撤不會，
        # 期間一長就會把長期持有講得過度漂亮；年化才是可比的。
        # 這個對照很重要：系統常常「報酬輸買進持有，但回撤小很多」。
        if out["max_drawdown"] > 0 and out["cagr"] is not None:
            out["calmar"] = out["cagr"] / out["max_drawdown"]
        if buy_hold_max_drawdown and out["buy_hold_cagr"] is not None:
            out["buy_hold_calmar"] = out["buy_hold_cagr"] / buy_hold_max_drawdown
        if out.get("calmar") is not None and out.get("buy_hold_calmar") is not None:
            out["beats_buy_hold_risk_adjusted"] = out["calmar"] > out["buy_hold_calmar"]

    return out


def buy_and_hold(df: pd.DataFrame, start_idx: int = 0) -> dict:
    """同期間「什麼都不做、直接持有」的報酬與最大回撤。"""
    close = df["Close"].iloc[start_idx:]
    if len(close) < 2:
        return {"return": 0.0, "max_drawdown": 0.0}
    curve = (close / close.iloc[0]).to_numpy()
    return {"return": float(curve[-1] - 1.0), "max_drawdown": max_drawdown(curve)}
