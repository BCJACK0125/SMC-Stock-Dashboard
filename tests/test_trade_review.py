# -*- coding: utf-8 -*-
"""
tests/test_trade_review.py
===========================
交易覆盤：MAE/MFE 量測與診斷邏輯。
"""
from __future__ import annotations

import pandas as pd
import pytest

from trade_review import TradeRecord, diagnose, measure_excursions, summarize_review


def path(rows):
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"],
                        index=pd.date_range("2024-01-02", periods=len(rows), freq="D"))


# --------------------------------------------------------------------------
# MAE / MFE 量測
# --------------------------------------------------------------------------
def test_long_mae_is_worst_drawdown_from_entry():
    p = path([(100, 102, 95, 101), (101, 108, 99, 107)])
    e = measure_excursions(p, entry=100, side="bullish", stop=90)
    assert e["mae"] == pytest.approx(0.05)    # 最低 95
    assert e["mfe"] == pytest.approx(0.08)    # 最高 108
    assert e["bars_to_mae"] == 1
    assert e["bars_to_mfe"] == 2


def test_short_excursions_are_mirrored():
    p = path([(100, 106, 98, 99)])
    e = measure_excursions(p, entry=100, side="bearish", stop=110)
    assert e["mae"] == pytest.approx(0.06)    # 空單最不利是漲到 106
    assert e["mfe"] == pytest.approx(0.02)    # 最有利是跌到 98


def test_excursions_are_never_negative():
    """一路順風的單子，MAE 應為 0 而不是負數。"""
    p = path([(100, 110, 101, 109)])
    e = measure_excursions(p, entry=100, side="bullish", stop=90)
    assert e["mae"] == 0.0
    assert e["mfe"] == pytest.approx(0.10)


def test_entry_heat_only_looks_at_early_bars():
    """進場時機只看前幾根；後面才發生的浮虧不算進場的錯。"""
    p = path([(100, 101, 99.5, 100), (100, 101, 99.5, 100),
              (100, 101, 99.5, 100), (100, 101, 80, 85)])
    e = measure_excursions(p, entry=100, side="bullish", stop=70, early_bars=3)
    assert e["entry_heat"] == pytest.approx(0.005)
    assert e["mae"] == pytest.approx(0.20)


def test_empty_path_is_safe():
    e = measure_excursions(path([]), entry=100, side="bullish", stop=90)
    assert e == {"mae": 0.0, "mfe": 0.0, "bars_to_mae": 0,
                 "bars_to_mfe": 0, "entry_heat": 0.0}


# --------------------------------------------------------------------------
# 彙總與診斷
# --------------------------------------------------------------------------
def rec(ret, mae=0.02, mfe=0.05, mae_r=0.4, entry_heat=0.005, outcome="target"):
    return TradeRecord(
        symbol="X", entry_ts=pd.Timestamp("2024-01-01"),
        exit_ts=pd.Timestamp("2024-01-10"), entry=100.0, exit=100 * (1 + ret),
        stop=95.0, side="bullish", outcome=outcome, bars_held=9, ret=ret,
        r_multiple=ret / 0.05, mae=mae, mfe=mfe, mae_r=mae_r, mfe_r=mfe / 0.05,
        bars_to_mae=2, bars_to_mfe=7, entry_heat=entry_heat)


def test_summary_counts_and_rates():
    s = summarize_review([rec(0.05), rec(0.03), rec(-0.02)])
    assert s["n_trades"] == 3
    assert s["win_rate"] == pytest.approx(2 / 3)


def test_shallow_winner_mae_suggests_tightening_the_stop():
    """贏家很少深套 → 停損可以收緊，直接改善 R:R。"""
    s = summarize_review([rec(0.05, mae_r=0.2) for _ in range(10)])
    msgs = " ".join(diagnose(s))
    assert "停損可以收緊" in msgs


def test_deep_winner_mae_warns_against_tightening():
    s = summarize_review([rec(0.05, mae_r=0.85) for _ in range(10)])
    msgs = " ".join(diagnose(s))
    assert "停損偏緊" in msgs


def test_losers_that_were_profitable_flag_slow_exits():
    """
    虧損單過程中曾經浮盈，代表出場太慢——這是保本停損能直接解決的問題。
    """
    losers = [rec(-0.03, mfe=0.06) for _ in range(8)]
    winners = [rec(0.04, mfe=0.06) for _ in range(2)]
    msgs = " ".join(diagnose(summarize_review(losers + winners)))
    assert "出場太慢" in msgs


def test_low_capture_flags_exit_inefficiency():
    s = summarize_review([rec(0.01, mfe=0.10) for _ in range(10)])
    msgs = " ".join(diagnose(s))
    assert "出場效率偏低" in msgs


def test_entry_timing_needs_a_baseline_to_be_judged():
    """
    沒有隨機進場的對照組就不該下判斷。高波動股票任何進場點都會先吃幾個
    百分點——實測訊號 66% vs 隨機 67%，其實沒有差別。少了對照就會把正常
    波動誤判成「訊號在追高」，給出沒有必要的修改建議。
    """
    s = summarize_review([rec(0.02, entry_heat=0.04) for _ in range(10)])
    msgs = " ".join(diagnose(s))
    assert "無法判斷" in msgs
    assert "進場時機偏差" not in msgs


def test_entry_timing_flagged_only_when_worse_than_baseline():
    s = summarize_review([rec(0.02, entry_heat=0.04) for _ in range(10)])
    worse = diagnose(s, baseline={"immediately_underwater": 0.40})
    assert "進場時機偏差" in " ".join(worse)


def test_entry_timing_matching_baseline_is_reported_as_neutral():
    """與隨機無異時要明說「這不是缺點」，避免使用者去修一個不存在的問題。"""
    s = summarize_review([rec(0.02, entry_heat=0.04) for _ in range(10)])
    same = diagnose(s, baseline={"immediately_underwater": 1.0})
    joined = " ".join(same)
    assert "與隨機無異" in joined
    assert "進場時機偏差" not in joined


def test_trailing_mode_does_not_warn_about_stop_exits():
    """移動停損模式下 100% 由停損出場是正常設計，不是停損被打爆。"""
    s = summarize_review([rec(0.03, outcome="stop") for _ in range(10)])
    trailing = " ".join(diagnose(s, exit_mode="trailing"))
    fixed = " ".join(diagnose(s, exit_mode="fixed"))
    assert "正常行為" in trailing and "比例偏高" not in trailing
    assert "比例偏高" in fixed


def test_capture_rate_is_robust_to_skew():
    """
    交易報酬高度右偏。用逐筆比值的中位數會被一堆小額交易主導，算出接近
    0 的假象；改用總實現/總浮盈才反映真實的出場效率。
    """
    trades = [rec(-0.003, mfe=0.058) for _ in range(9)] + [rec(0.50, mfe=0.60)]
    s = summarize_review(trades)
    assert s["mfe_capture"] > 0.3, "總量口徑應反映那筆大贏"
    assert s["mfe_capture_median_ratio"] < 0, "逐筆中位數會給出誤導性的負值"


def test_timeout_heavy_mix_is_reported():
    s = summarize_review([rec(0.01, outcome="timeout") for _ in range(9)]
                         + [rec(0.05, outcome="target")])
    msgs = " ".join(diagnose(s))
    assert "時間停損出場" in msgs


def test_empty_input_is_handled():
    assert summarize_review([]) == {}
    assert diagnose({}) == ["樣本不足，無法診斷"]
