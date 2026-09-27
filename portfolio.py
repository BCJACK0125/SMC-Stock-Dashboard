# -*- coding: utf-8 -*-
"""
portfolio.py
=============
組合層回測：同時操作多檔標的，把「閒置資金」變成報酬。

為什麼需要這一層：
    單檔回測顯示，這套策略的**風險調整報酬**（Calmar）在 5 檔裡贏過
    買進持有 3 檔，但**絕對報酬全數落後**。原因不是訊號差，而是
    ——**資金有 50~72% 的時間是空手的**。

    單檔策略只在有訊號時進場，其餘時間現金閒置；買進持有則是 100%
    在場。在一段長期上漲的行情裡，這個「在場時間差」幾乎不可能靠
    每筆多賺一點補回來。

    解法不是讓單檔更常進場（那會稀釋訊號品質），而是**橫向擴張**：
    同時追蹤數十檔標的，任何時刻總有幾檔有訊號，組合層的在場時間
    就能拉高到接近 100%，而每一筆進場仍然維持原本的訊號品質。

⚠️ 這個模擬的假設：
    - 資金平均分配到最多 `max_positions` 個並存部位，每個部位吃一個
      「槽位」的資金；沒有槓桿、沒有加碼。
    - 同一時間同一標的只持有一個部位。
    - 訊號多於空槽時，依綜合分數由高到低挑選。
    - 進出場價用當根收盤價，報酬已扣來回交易成本。
    - 未平倉部位按收盤價逐根市價評估（需傳入 prices），最終報酬仍以
      單檔停損/目標模擬的結果為準，所以出場當根會有小幅不連續。
    - 沒有模擬滑價、跳空、除權息、個股流動性限制。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import performance


@dataclass
class OpenPosition:
    symbol: str
    entry_bar: int          # 以組合共用的時間軸為準
    exit_bar: int           # 預先算好的出場時點（來自單檔模擬）
    ret: float              # 該筆交易「已扣成本」的報酬率
    slot_capital: float     # 進場時投入的金額
    entry_price: float = 0.0   # 進場價，供逐根市價評估用
    side_sign: float = 1.0     # 多單 +1 / 空單 -1


@dataclass
class PortfolioResult:
    equity: pd.Series
    trades: List[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def simulate_portfolio(
    per_symbol: Dict[str, pd.DataFrame],
    thresholds: Dict[str, int],
    costs: Dict[str, float],
    *,
    prices: Optional[Dict[str, pd.Series]] = None,
    side: str = "bull",
    max_positions: int = 10,
    initial_capital: float = 1.0,
    bars_per_year: float = 252,
) -> PortfolioResult:
    """
    per_symbol : {symbol: walk_forward_scores 的輸出}，需含 ts / *_score /
                 {long|short}_ret / {long|short}_bars
    thresholds : {symbol: 該檔採用的進場門檻}
    costs      : {symbol: 來回交易成本}
    prices     : {symbol: 收盤價序列}。提供時會對未平倉部位做逐根市價
                 評估，最大回撤才是誠實的；不提供則只在平倉時更新權益，
                 曲線會偏平滑、**回撤被低估**。
    """
    prefix = "long" if side == "bull" else "short"
    score_col = f"{'bull' if side == 'bull' else 'bear'}_score"

    # 把各標的的時間軸合併成一條共用的日曆（不同市場交易日不同）
    all_ts = sorted({ts for df in per_symbol.values() for ts in df["ts"]})
    if not all_ts:
        return PortfolioResult(equity=pd.Series(dtype=float))
    bar_of = {ts: i for i, ts in enumerate(all_ts)}

    # 預先把每檔的「某根K棒是否有訊號、進場的話結果如何」查表建好
    signals: Dict[int, List[tuple]] = {}      # bar -> [(score, symbol, ret, hold_bars)]
    for sym, df in per_symbol.items():
        th = thresholds.get(sym)
        if th is None:
            continue
        cost = costs.get(sym, 0.0)
        sub = df[df[score_col] >= th]
        for _, row in sub.iterrows():
            b = bar_of[row["ts"]]
            signals.setdefault(b, []).append(
                (float(row[score_col]), sym,
                 float(row[f"{prefix}_ret"]) - cost,
                 max(int(row[f"{prefix}_bars"]), 1),
                 row["ts"])
            )

    capital = initial_capital
    open_pos: List[OpenPosition] = []
    equity_points = []
    trades: List[dict] = []
    in_market_bars = 0

    for b, ts in enumerate(all_ts):
        # ---- 先結算今天到期的部位 ----
        still_open = []
        for p in open_pos:
            if p.exit_bar <= b:
                capital += p.slot_capital * (1.0 + p.ret)
                trades.append({"symbol": p.symbol, "entry_bar": p.entry_bar,
                               "exit_bar": p.exit_bar, "ret": p.ret})
            else:
                still_open.append(p)
        open_pos = still_open

        # ---- 再用空出來的槽位進場 ----
        free_slots = max_positions - len(open_pos)
        if free_slots > 0 and b in signals:
            held = {p.symbol for p in open_pos}
            # 訊號多於空槽時，分數高的優先
            candidates = sorted(signals[b], key=lambda x: -x[0])
            for score, sym, ret, hold, ts_entry in candidates:
                if free_slots == 0:
                    break
                if sym in held:
                    continue            # 同一標的同時只持有一個部位
                slot_cap = capital / max(free_slots + len(open_pos), 1)
                if slot_cap <= 0:
                    break
                entry_px = 0.0
                if prices is not None and sym in prices:
                    ser = prices[sym]
                    if ts_entry in ser.index:
                        entry_px = float(ser.loc[ts_entry])
                capital -= slot_cap
                open_pos.append(OpenPosition(
                    symbol=sym, entry_bar=b, exit_bar=b + hold, ret=ret,
                    slot_capital=slot_cap, entry_price=entry_px,
                    side_sign=1.0 if side == "bull" else -1.0))
                held.add(sym)
                free_slots -= 1

        if open_pos:
            in_market_bars += 1

        # 權益 = 現金 + 未平倉部位的市值。
        # 有 prices 時按收盤價逐根評估未實現損益——沒有這一步的話，部位
        # 中途跌 15% 再拉回來完全不會反映在曲線上，最大回撤會被嚴重低估，
        # 連帶讓 Calmar 被高估。這是「風險調整後比較好」這個結論成不成立
        # 的關鍵，不能省。
        mtm = 0.0
        for p in open_pos:
            if prices is not None and p.entry_price > 0:
                ser = prices.get(p.symbol)
                if ser is not None and ts in ser.index:
                    px = float(ser.loc[ts])
                    mtm += p.slot_capital * (1.0 + p.side_sign * (px / p.entry_price - 1.0))
                    continue
            mtm += p.slot_capital      # 沒有價格資料就退回帳面成本
        equity_points.append(capital + mtm)

    # 收尾：把還沒平倉的部位按其最終報酬結算
    for p in open_pos:
        capital += p.slot_capital * (1.0 + p.ret)
        trades.append({"symbol": p.symbol, "entry_bar": p.entry_bar,
                       "exit_bar": p.exit_bar, "ret": p.ret})
    if equity_points:
        equity_points[-1] = capital

    equity = pd.Series(equity_points, index=pd.DatetimeIndex(all_ts))
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    years = len(all_ts) / bars_per_year if bars_per_year else 0.0

    stats = {
        "n_trades": len(trades),
        "years": round(years, 2),
        "trades_per_year": len(trades) / years if years > 0 else None,
        "total_return": total_return,
        "cagr": performance.cagr(total_return, years),
        "max_drawdown": performance.max_drawdown(equity.to_numpy()),
        "time_in_market": in_market_bars / len(all_ts) if all_ts else 0.0,
        "max_positions": max_positions,
        "n_symbols": len(per_symbol),
    }
    if stats["max_drawdown"] > 0 and stats["cagr"] is not None:
        stats["calmar"] = stats["cagr"] / stats["max_drawdown"]

    return PortfolioResult(equity=equity, trades=trades, stats=stats)


def equal_weight_buy_hold(
    prices: Dict[str, pd.Series], bars_per_year: float = 252
) -> dict:
    """
    對照組：把資金平均買進全部標的後什麼都不做。
    這才是組合策略該比的對象——拿組合去比單一檔股票並不公平。
    """
    norm = []
    for s in prices.values():
        s = s.dropna()
        if len(s) > 1:
            norm.append(s / s.iloc[0])
    if not norm:
        return {}
    curve = pd.concat(norm, axis=1).ffill().dropna(how="all").mean(axis=1)
    total = float(curve.iloc[-1] - 1.0)
    years = len(curve) / bars_per_year if bars_per_year else 0.0
    out = {
        "total_return": total,
        "cagr": performance.cagr(total, years),
        "max_drawdown": performance.max_drawdown(curve.to_numpy()),
        "years": round(years, 2),
    }
    if out["max_drawdown"] > 0 and out["cagr"] is not None:
        out["calmar"] = out["cagr"] / out["max_drawdown"]
    return out
