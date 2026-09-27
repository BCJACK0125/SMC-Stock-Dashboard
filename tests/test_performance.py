# -*- coding: utf-8 -*-
"""
tests/test_performance.py
==========================
複利績效與買進持有對照。重點在「算術平均會高估、複利才是實際拿到的」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from performance import (
    buy_and_hold,
    cagr,
    equity_curve,
    max_drawdown,
    summarize_performance,
)


# --------------------------------------------------------------------------
# 權益曲線
# --------------------------------------------------------------------------
def test_equity_curve_compounds():
    c = equity_curve([0.10, 0.10])
    assert c[-1] == pytest.approx(1.21)


def test_equity_curve_starts_at_one():
    assert equity_curve([])[0] == 1.0
    assert len(equity_curve([])) == 1


def test_compounding_is_worse_than_arithmetic_mean_when_volatile():
    """
    +50% 之後 −50% 的算術平均是 0，但實際剩下 0.75。
    只看「每筆平均報酬」會系統性高估真正拿到的錢——這正是需要
    權益曲線的理由。
    """
    rets = [0.5, -0.5]
    assert np.mean(rets) == pytest.approx(0.0)
    assert equity_curve(rets)[-1] == pytest.approx(0.75)


def test_total_loss_is_floored_sensibly():
    assert equity_curve([-1.0])[-1] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 最大回撤
# --------------------------------------------------------------------------
def test_max_drawdown_measures_peak_to_trough():
    curve = np.array([1.0, 1.5, 0.9, 1.2])
    assert max_drawdown(curve) == pytest.approx(0.4)   # 1.5 -> 0.9


def test_monotonic_rise_has_no_drawdown():
    assert max_drawdown(np.array([1.0, 1.1, 1.2])) == 0.0


def test_drawdown_of_flat_curve_is_zero():
    assert max_drawdown(np.array([1.0])) == 0.0


# --------------------------------------------------------------------------
# 年化
# --------------------------------------------------------------------------
def test_cagr_matches_compound_definition():
    assert cagr(1.0, 2.0) == pytest.approx(2 ** 0.5 - 1)   # 兩年翻倍


def test_cagr_handles_total_wipeout():
    assert cagr(-1.0, 3.0) is None


def test_cagr_needs_positive_years():
    assert cagr(0.5, 0.0) is None


# --------------------------------------------------------------------------
# 綜合摘要
# --------------------------------------------------------------------------
def test_summary_reports_time_in_market():
    """只在有訊號時進場，代表大部分時間空手；這個機會成本只有在這裡看得到。"""
    s = summarize_performance([0.02] * 5, [10] * 5,
                              total_bars=1000, bars_per_year=252)
    assert s["time_in_market"] == pytest.approx(0.05)
    assert s["n_trades"] == 5


def test_summary_flags_losing_to_buy_and_hold():
    """
    每筆都賺、期望值漂亮，但只做了 3 筆，累計仍遠輸買進持有。
    這是「期望值為正 ≠ 值得做」的具體案例。
    """
    s = summarize_performance([0.02] * 3, [10] * 3,
                              total_bars=2520, bars_per_year=252,
                              buy_hold_return=1.50)
    assert s["total_return"] == pytest.approx(1.02 ** 3 - 1)
    assert s["beats_buy_hold"] is False
    assert s["excess_vs_buy_hold"] < 0


def test_summary_can_beat_buy_and_hold():
    s = summarize_performance([0.05] * 30, [10] * 30,
                              total_bars=2520, bars_per_year=252,
                              buy_hold_return=0.50)
    assert s["beats_buy_hold"] is True


def test_summary_handles_no_trades():
    s = summarize_performance([], [], total_bars=1000, bars_per_year=252)
    assert s["n_trades"] == 0
    assert s["total_return"] == pytest.approx(0.0)
    assert s["max_drawdown"] == 0.0


def test_trades_per_year_uses_bar_count():
    s = summarize_performance([0.01] * 10, [5] * 10,
                              total_bars=2520, bars_per_year=252)
    assert s["years"] == pytest.approx(10.0)
    assert s["trades_per_year"] == pytest.approx(1.0)


def test_calmar_is_reported_when_there_is_a_drawdown():
    """報酬輸給買進持有、但回撤小很多的情況，只看報酬會漏掉。"""
    s = summarize_performance([0.1, -0.05, 0.1], [10] * 3,
                              total_bars=756, bars_per_year=252,
                              buy_hold_return=0.30, buy_hold_max_drawdown=0.4)
    assert s["max_drawdown"] > 0
    assert s["calmar"] == pytest.approx(s["cagr"] / s["max_drawdown"])


# --------------------------------------------------------------------------
# 買進持有基準
# --------------------------------------------------------------------------
def _df(closes):
    return pd.DataFrame({"Close": closes},
                        index=pd.date_range("2024-01-01", periods=len(closes), freq="D"))


def test_buy_and_hold_return_and_drawdown():
    bh = buy_and_hold(_df([100, 150, 90, 120]))
    assert bh["return"] == pytest.approx(0.2)
    assert bh["max_drawdown"] == pytest.approx(0.4)


def test_buy_and_hold_respects_start_index():
    """回測跳過暖身期，買進持有的對照也必須從同一個起點算，否則不公平。"""
    bh = buy_and_hold(_df([10, 100, 110]), start_idx=1)
    assert bh["return"] == pytest.approx(0.1)


def test_buy_and_hold_handles_short_series():
    assert buy_and_hold(_df([100]))["return"] == 0.0


def test_calmar_compares_annualised_return_per_drawdown():
    """
    風險調整對照用 Calmar（年化 ÷ 回撤），不用「總報酬 ÷ 回撤」：
    總報酬會隨期間複利放大、回撤不會，期間一長就會把買進持有講得過度漂亮。
    """
    # 刻意夾一筆虧損，否則沒有回撤、Calmar 無從定義
    s = summarize_performance([0.10] * 5 + [-0.15] + [0.10] * 5, [20] * 11,
                              total_bars=2520, bars_per_year=252,
                              buy_hold_return=1.0, buy_hold_max_drawdown=0.50)
    assert s["calmar"] == pytest.approx(s["cagr"] / s["max_drawdown"])
    assert s["buy_hold_calmar"] == pytest.approx(s["buy_hold_cagr"] / 0.50)
    assert isinstance(s["beats_buy_hold_risk_adjusted"], bool)


def test_strategy_can_lose_on_return_but_win_risk_adjusted():
    """報酬輸買進持有、但回撤只有一半 —— 只看報酬會錯過這種情況。"""
    s = summarize_performance([0.03] * 10 + [-0.10] + [0.03] * 10, [10] * 21,
                              total_bars=2520, bars_per_year=252,
                              buy_hold_return=1.2, buy_hold_max_drawdown=0.60)
    assert s["beats_buy_hold"] is False
    assert s["max_drawdown"] < 0.60
    assert s["beats_buy_hold_risk_adjusted"] is True


def test_risk_adjusted_flag_absent_without_buy_hold_drawdown():
    s = summarize_performance([0.02] * 5, [10] * 5,
                              total_bars=1000, bars_per_year=252,
                              buy_hold_return=0.3)
    assert "beats_buy_hold_risk_adjusted" not in s
