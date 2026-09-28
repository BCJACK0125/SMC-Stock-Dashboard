# -*- coding: utf-8 -*-
"""
tests/test_trade_markers_prices.py
===================================
圖表標記畫的價格，必須是交易**實際的成交價**。

為什麼需要這個測試：
    標記原本用 `df["Close"].loc[entry_ts]` 與 `df["Close"].loc[exit_ts]`
    重新推導價格，但實際交易是「隔天用限價成交」、「盤中打到停損價出場」，
    兩者都不是收盤價。結果圖上兩點算出來的報酬跟標籤顯示的報酬對不上：
    NVDA 實測 30 筆裡有 29 筆不一致，最大差到 9 個百分點
    （顯示進場 195.10 → 出場 176.78 = −9.39%，標籤卻寫 −3.83%）。

    這類錯誤不會讓任何東西壞掉，只會讓人看著圖做出錯誤的判斷。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest as BT


def wf_frame():
    """一筆多單：訊號在 index 0，隔天成交，持有 3 根後打到停損。"""
    ts = pd.date_range("2026-01-05", periods=8, freq="B")
    n = len(ts)
    base = {
        "ts": ts,
        "bull_score": [60] + [0] * (n - 1),
        "bear_score": [0] * n,
        "future_return": [0.0] * n,
        "entry_px": [95.0] + [0.0] * (n - 1),      # 限價成交，非訊號棒收盤 100
        "entry_off": [0] * n,                       # 隔天成交
        "long_ret": [-0.05] + [0.0] * (n - 1),
        "long_bars": [4] + [1] * (n - 1),           # 1 根等待 + 3 根持有
        "long_r": [-1.0] + [0.0] * (n - 1),
        "long_outcome": ["stop"] + ["none"] * (n - 1),
        "long_rr": [2.0] * n,
        "long_exit_px": [90.25] + [0.0] * (n - 1),  # 停損價，非出場棒收盤
    }
    return pd.DataFrame(base)


def test_selected_trades_carry_the_actual_fill_prices():
    t = BT.selected_trades(wf_frame(), threshold=50, side="bull", cost=0.0)[0]
    assert t["entry_px"] == 95.0
    assert t["exit_px"] == 90.25


def test_entry_timestamp_is_the_fill_bar_not_the_signal_bar():
    """訊號當根收盤後才看得到，成交一定在下一根。"""
    t = BT.selected_trades(wf_frame(), threshold=50, side="bull", cost=0.0)[0]
    assert t["signal_ts"] == pd.Timestamp("2026-01-05")
    assert t["entry_ts"] == pd.Timestamp("2026-01-06")


def test_the_two_plotted_points_reproduce_the_labelled_return():
    """圖上兩點算出來的報酬，要跟標籤顯示的報酬一致——這就是原本壞掉的地方。"""
    t = BT.selected_trades(wf_frame(), threshold=50, side="bull", cost=0.0)[0]
    from_points = t["exit_px"] / t["entry_px"] - 1
    assert from_points == pytest.approx(t["ret"], abs=1e-9)


def test_older_cached_trades_without_prices_still_work():
    """設定沒變時快取會沿用舊結果，缺欄位不能讓圖表整個掛掉。"""
    wf = wf_frame().drop(columns=["entry_px", "long_exit_px", "entry_off"])
    t = BT.selected_trades(wf, threshold=50, side="bull", cost=0.0)[0]
    assert t["entry_px"] is None and t["exit_px"] is None
    assert t["entry_ts"] == pd.Timestamp("2026-01-05")   # 退回訊號棒


def test_marker_falls_back_to_close_when_prices_are_missing():
    import plot_report
    ts = pd.date_range("2026-01-05", periods=8, freq="B")
    df = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                       "Close": 100.0, "Volume": 1}, index=ts)

    class Fig:
        def __init__(self): self.traces = []
        def add_shape(self, **k): pass
        def add_trace(self, tr, **k): self.traces.append(tr)

    fig = Fig()
    shown = plot_report._add_trade_markers(fig, df, [{
        "entry_ts": ts[0], "exit_ts": ts[3], "ret": -0.05,
        "bars": 3, "outcome": "stop", "side": "bull"}])
    assert shown == 1


def test_cache_schema_bump_invalidates_old_entries():
    """結果欄位變了，舊快取即使還在保鮮期也必須失效。"""
    import backtest_cache as BC
    from types import SimpleNamespace
    cfg = SimpleNamespace(SWING_LOOKBACK=3)
    fp = BC.config_fingerprint(cfg)
    original = BC._RESULT_SCHEMA
    try:
        BC._RESULT_SCHEMA = original + 1
        assert BC.config_fingerprint(cfg) != fp
    finally:
        BC._RESULT_SCHEMA = original
