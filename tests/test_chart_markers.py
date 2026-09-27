# -*- coding: utf-8 -*-
"""
tests/test_chart_markers.py
============================
把回測交易畫回 K 線圖上。統計數字看不出「這套邏輯在哪些位置進出場」，
畫在圖上才能用肉眼檢查訊號是否合理。
"""
from __future__ import annotations

import pandas as pd
import pytest
from plotly.subplots import make_subplots

from backtest import selected_trades
from plot_report import _add_trade_markers


def frame(n=20, start="2024-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    base = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    return pd.DataFrame({"Open": base, "High": base + 2,
                         "Low": base - 2, "Close": base}, index=idx)


def trade(df, i=2, j=8, ret=0.05, outcome="stop"):
    return {"entry_ts": df.index[i], "exit_ts": df.index[j], "ret": ret,
            "bars": j - i, "outcome": outcome, "side": "bull"}


def fig():
    return make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])


# --------------------------------------------------------------------------
# 標記繪製
# --------------------------------------------------------------------------
def test_each_trade_gets_a_connecting_line():
    df = frame()
    f = fig()
    n = _add_trade_markers(f, df, [trade(df, 2, 6), trade(df, 8, 12)])
    assert n == 2
    assert len(f.layout.shapes) == 2


def test_entry_and_exit_marker_traces_are_added():
    df = frame()
    f = fig()
    _add_trade_markers(f, df, [trade(df)])
    names = [t.name for t in f.data]
    assert "回測進場" in names and "回測出場" in names


def test_winners_and_losers_use_different_colours():
    df = frame()
    f = fig()
    _add_trade_markers(f, df, [trade(df, 2, 6, ret=0.05),
                               trade(df, 8, 12, ret=-0.03)])
    colours = {s.line.color for s in f.layout.shapes}
    assert len(colours) == 2, "獲利與虧損要能一眼分辨"


def test_hover_text_states_profit_or_loss_explicitly():
    """顏色語意（台股紅漲綠跌）容易被誤讀，hover 必須明講。"""
    df = frame()
    f = fig()
    _add_trade_markers(f, df, [trade(df, ret=-0.04)])
    hover = " ".join(" ".join(t.hovertext) for t in f.data if t.hovertext)
    assert "虧損" in hover and "-4" in hover.replace("−", "-")


def test_no_trades_is_a_noop():
    f = fig()
    assert _add_trade_markers(f, frame(), []) == 0
    assert len(f.layout.shapes) == 0
    assert len(f.data) == 0


def test_trades_outside_the_visible_window_are_skipped():
    """圖表只畫最近 N 根，更早的交易不該畫在邊界上造成誤解。"""
    full = frame(40)
    visible = full.tail(10)
    old = trade(full, 1, 5)
    f = fig()
    assert _add_trade_markers(f, visible, [old]) == 0


def test_exit_beyond_window_is_clamped_to_the_last_bar():
    """出場在可視範圍之後時，連線收在最後一根，而不是整筆消失。"""
    df = frame(20)
    t = trade(df, 15, 19)
    t["exit_ts"] = pd.Timestamp("2025-01-01")      # 遠在窗外
    f = fig()
    assert _add_trade_markers(f, df, [t]) == 1
    assert len(f.layout.shapes) == 1


# --------------------------------------------------------------------------
# 交易來源：selected_trades
# --------------------------------------------------------------------------
def _wf(n=40):
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="D"),
        "bull_score": [60] * n,
        "bear_score": [0] * n,
        "future_return": [0.01] * n,
        "long_ret": [0.03] * n,
        "long_bars": [5] * n,
        "long_r": [1.0] * n,
        "long_outcome": ["target"] * n,
    })


def test_selected_trades_respects_non_overlap():
    tr = selected_trades(_wf(40), threshold=60, cost=0.0)
    assert len(tr) == 8                      # 40 根 / 持有 5 根
    for a, b in zip(tr, tr[1:]):
        assert b["entry_ts"] > a["entry_ts"]


def test_selected_trades_deducts_cost():
    tr = selected_trades(_wf(10), threshold=60, cost=0.01)
    assert tr[0]["ret"] == pytest.approx(0.02)


def test_selected_trades_ignores_low_scores():
    assert selected_trades(_wf(20), threshold=99) == []


def test_selected_trades_handles_missing_columns():
    """沒有模擬交易欄位時（舊格式）應安全回傳空清單，而不是炸掉。"""
    wf = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5),
                       "bull_score": [60] * 5, "bear_score": [0] * 5,
                       "future_return": [0.01] * 5})
    assert selected_trades(wf, threshold=60) == []


def test_selected_trades_on_empty_input():
    assert selected_trades(pd.DataFrame(), threshold=50) == []
