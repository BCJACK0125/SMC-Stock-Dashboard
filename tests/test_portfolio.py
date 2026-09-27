# -*- coding: utf-8 -*-
"""
tests/test_portfolio.py
========================
組合層回測：槽位管理、資金複利、在場時間。
"""
from __future__ import annotations

import pandas as pd
import pytest

from portfolio import equal_weight_buy_hold, simulate_portfolio


def wf(ts, scores, rets, bars, prefix="long"):
    """組一份最小可用的 walk_forward_scores 輸出。"""
    return pd.DataFrame({
        "ts": ts,
        "bull_score": scores,
        "bear_score": [0] * len(ts),
        f"{prefix}_ret": rets,
        f"{prefix}_bars": bars,
    })


DAYS = pd.date_range("2024-01-01", periods=20, freq="D")


# --------------------------------------------------------------------------
# 槽位管理
# --------------------------------------------------------------------------
def test_max_positions_is_respected():
    per = {f"S{i}": wf(DAYS, [99] * 20, [0.0] * 20, [5] * 20) for i in range(5)}
    r = simulate_portfolio(per, {f"S{i}": 50 for i in range(5)},
                           {f"S{i}": 0.0 for i in range(5)}, max_positions=2)
    # 每 5 天一輪、最多 2 個部位 -> 20 天最多 8 筆
    assert r.stats["n_trades"] <= 8


def test_same_symbol_is_not_doubled_up():
    per = {"A": wf(DAYS, [99] * 20, [0.0] * 20, [10] * 20)}
    r = simulate_portfolio(per, {"A": 50}, {"A": 0.0}, max_positions=5)
    entries = sorted(t["entry_bar"] for t in r.trades)
    for a, b in zip(entries, entries[1:]):
        assert b - a >= 10, "同一標的的部位不該重疊"


def test_higher_score_wins_the_slot():
    """空槽不足時，分數高的訊號優先。"""
    per = {
        "LOW": wf(DAYS, [40] * 20, [-0.5] * 20, [20] * 20),
        "HIGH": wf(DAYS, [90] * 20, [+0.5] * 20, [20] * 20),
    }
    r = simulate_portfolio(per, {"LOW": 35, "HIGH": 35},
                           {"LOW": 0.0, "HIGH": 0.0}, max_positions=1)
    assert r.trades and all(t["symbol"] == "HIGH" for t in r.trades)


def test_signals_below_threshold_are_ignored():
    per = {"A": wf(DAYS, [10] * 20, [0.5] * 20, [5] * 20)}
    r = simulate_portfolio(per, {"A": 50}, {"A": 0.0}, max_positions=3)
    assert r.stats["n_trades"] == 0


def test_symbol_without_threshold_is_skipped():
    per = {"A": wf(DAYS, [99] * 20, [0.1] * 20, [5] * 20)}
    r = simulate_portfolio(per, {}, {}, max_positions=3)
    assert r.stats["n_trades"] == 0


# --------------------------------------------------------------------------
# 資金與報酬
# --------------------------------------------------------------------------
def test_single_full_capital_trade_compounds_exactly():
    ts = pd.date_range("2024-01-01", periods=6, freq="D")
    per = {"A": wf(ts, [99, 0, 0, 0, 0, 0], [0.10] * 6, [5] * 6)}
    r = simulate_portfolio(per, {"A": 50}, {"A": 0.0}, max_positions=1)
    assert r.stats["n_trades"] == 1
    assert r.stats["total_return"] == pytest.approx(0.10)


def test_cost_is_deducted_from_trade_return():
    ts = pd.date_range("2024-01-01", periods=6, freq="D")
    per = {"A": wf(ts, [99, 0, 0, 0, 0, 0], [0.10] * 6, [5] * 6)}
    r = simulate_portfolio(per, {"A": 50}, {"A": 0.02}, max_positions=1)
    assert r.stats["total_return"] == pytest.approx(0.08)


def test_losses_reduce_capital_for_later_trades():
    """複利是雙向的：先虧損會讓後面的部位變小。"""
    ts = pd.date_range("2024-01-01", periods=12, freq="D")
    scores = [99 if i in (0, 5) else 0 for i in range(12)]
    per = {"A": wf(ts, scores, [-0.5] * 12, [5] * 12)}
    r = simulate_portfolio(per, {"A": 50}, {"A": 0.0}, max_positions=1)
    assert r.stats["n_trades"] == 2
    assert r.stats["total_return"] == pytest.approx(0.25 - 1.0)   # 0.5 * 0.5


# --------------------------------------------------------------------------
# 在場時間：這是做組合的主要理由
# --------------------------------------------------------------------------
def test_more_symbols_raises_time_in_market():
    """
    單檔策略大部分時間空手，這是它輸給買進持有的主因。
    橫向擴張標的數應該要把組合的在場時間拉高。
    """
    sparse = [99 if i % 10 == 0 else 0 for i in range(20)]
    one = {"A": wf(DAYS, sparse, [0.01] * 20, [3] * 20)}
    many = {f"S{k}": wf(DAYS, [99 if i % 10 == k % 10 else 0 for i in range(20)],
                        [0.01] * 20, [3] * 20) for k in range(10)}
    r1 = simulate_portfolio(one, {"A": 50}, {"A": 0.0}, max_positions=5)
    r2 = simulate_portfolio(many, {f"S{k}": 50 for k in range(10)},
                            {f"S{k}": 0.0 for k in range(10)}, max_positions=5)
    assert r2.stats["time_in_market"] > r1.stats["time_in_market"]


def test_empty_input_is_handled():
    r = simulate_portfolio({}, {}, {}, max_positions=5)
    assert r.equity.empty


def test_symbols_with_different_calendars_are_merged():
    """台股與美股的交易日不同，時間軸要能合併而不是對不上。"""
    a = wf(pd.date_range("2024-01-01", periods=10, freq="D"),
           [99] * 10, [0.01] * 10, [2] * 10)
    b = wf(pd.date_range("2024-01-02", periods=10, freq="2D"),
           [99] * 10, [0.01] * 10, [2] * 10)
    r = simulate_portfolio({"A": a, "B": b}, {"A": 50, "B": 50},
                           {"A": 0.0, "B": 0.0}, max_positions=4)
    assert r.stats["n_trades"] > 0
    assert len(r.equity) >= 10


# --------------------------------------------------------------------------
# 對照組
# --------------------------------------------------------------------------
def test_equal_weight_buy_hold_averages_the_symbols():
    """組合策略該比的是「平均買進全部標的」，拿去比單一檔並不公平。"""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = {"A": pd.Series([100, 200, 200], index=idx),   # +100%
              "B": pd.Series([100, 100, 100], index=idx)}   # 0%
    bh = equal_weight_buy_hold(prices)
    assert bh["total_return"] == pytest.approx(0.5)


def test_equal_weight_handles_empty():
    assert equal_weight_buy_hold({}) == {}


# --------------------------------------------------------------------------
# 未平倉部位的市價評估
# --------------------------------------------------------------------------
def test_mark_to_market_exposes_intra_trade_drawdown():
    """
    部位中途大跌再拉回、最後小賺出場。沒有市價評估的話，權益曲線完全
    看不到那段跌幅，最大回撤會被嚴重低估、Calmar 被高估——而「風險調整
    後比較好」正是這套系統唯一站得住的結論，不能建立在這種偏差上。
    """
    ts = pd.date_range("2024-01-01", periods=6, freq="D")
    per = {"A": wf(ts, [99, 0, 0, 0, 0, 0], [0.02] * 6, [5] * 6)}
    # 進場 100 -> 中途砍到 60 -> 收在 102
    px = {"A": pd.Series([100, 80, 60, 70, 90, 102], index=ts)}

    without = simulate_portfolio(per, {"A": 50}, {"A": 0.0}, max_positions=1)
    with_mtm = simulate_portfolio(per, {"A": 50}, {"A": 0.0},
                                  max_positions=1, prices=px)

    assert without.stats["max_drawdown"] == pytest.approx(0.0)
    assert with_mtm.stats["max_drawdown"] > 0.35, "中途跌到 60 應該要現形"
    # 最終報酬仍由模擬交易的結果決定，不受評估方式影響
    assert with_mtm.stats["total_return"] == pytest.approx(
        without.stats["total_return"])


def test_mark_to_market_falls_back_without_prices():
    ts = pd.date_range("2024-01-01", periods=6, freq="D")
    per = {"A": wf(ts, [99, 0, 0, 0, 0, 0], [0.02] * 6, [5] * 6)}
    r = simulate_portfolio(per, {"A": 50}, {"A": 0.0}, max_positions=1,
                           prices={"A": pd.Series(dtype=float)})
    assert r.stats["total_return"] == pytest.approx(0.02)


def test_mark_to_market_tracks_gains_too():
    ts = pd.date_range("2024-01-01", periods=6, freq="D")
    per = {"A": wf(ts, [99, 0, 0, 0, 0, 0], [0.50] * 6, [5] * 6)}
    px = {"A": pd.Series([100, 120, 140, 150, 150, 150], index=ts)}
    r = simulate_portfolio(per, {"A": 50}, {"A": 0.0}, max_positions=1, prices=px)
    # 中途權益應該高於起點（未實現獲利有被計入）
    assert r.equity.iloc[2] > 1.0
