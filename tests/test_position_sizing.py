# -*- coding: utf-8 -*-
"""
tests/test_position_sizing.py
==============================
部位大小建議。這個模組刻意做得保守——三層機制（信賴下界、1/4 Kelly、
硬上限）疊起來會比「理論最適」小很多，測試要把這個保守性釘住。
"""
from __future__ import annotations

import pytest

from position_sizing import (
    barrier_stats,
    describe,
    kelly_fraction,
    suggest_position,
    wilson_lower_bound,
)


# --------------------------------------------------------------------------
# Kelly 公式
# --------------------------------------------------------------------------
def test_kelly_at_breakeven_is_zero():
    """2:1 賠率的損益兩平點是 1/3；剛好在那裡不該下注。"""
    assert kelly_fraction(1 / 3, 2.0) == pytest.approx(0.0, abs=1e-9)


def test_kelly_matches_textbook_values():
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.25)
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.40)


def test_kelly_never_negative():
    """劣勢時回 0，而不是建議放空自己。"""
    assert kelly_fraction(0.2, 2.0) == 0.0


def test_better_odds_need_lower_win_rate():
    assert kelly_fraction(0.4, 3.0) > kelly_fraction(0.4, 2.0)


def test_kelly_handles_degenerate_odds():
    assert kelly_fraction(0.9, 0.0) == 0.0


# --------------------------------------------------------------------------
# 三層保守機制
# --------------------------------------------------------------------------
def test_uses_lower_bound_not_point_estimate():
    """
    這是最重要的一層。實測 DELL 點估計 68%（全 Kelly 52%），但 n 只有
    22、下界僅 47%——用點估計會建議 10 倍大的部位。
    """
    r = suggest_position(wins=15, n=22, rr=2.0)
    assert r["p_point"] > r["p_lower"]
    full_on_point = kelly_fraction(r["p_point"], 2.0)
    assert r["fraction"] < full_on_point / 2


def test_quarter_kelly_is_applied():
    r = suggest_position(wins=60, n=100, rr=2.0, kelly_divisor=4.0)
    assert r["kelly_used"] == pytest.approx(r["kelly_full"] / 4)


def test_hard_cap_is_respected():
    r = suggest_position(wins=95, n=100, rr=2.0, max_fraction=0.10)
    assert r["fraction"] == pytest.approx(0.10)
    assert "上限" in r["reason"]


def test_below_breakeven_means_no_bet():
    """機率下界低於兩平點 -> 這個出場結構對該標的不利，不是訊號的問題。"""
    r = suggest_position(wins=6, n=25, rr=2.0)
    assert r["fraction"] == 0.0
    assert "兩平點" in r["reason"]


def test_small_sample_refuses_to_estimate():
    r = suggest_position(wins=8, n=10, rr=2.0, min_samples=20)
    assert r["fraction"] == 0.0
    assert "樣本不足" in r["reason"]


def test_large_sample_narrows_the_gap():
    """樣本越大，下界越靠近點估計，建議部位才會放大。"""
    small = suggest_position(wins=30, n=50, rr=2.0)
    large = suggest_position(wins=300, n=500, rr=2.0)
    assert large["p_lower"] > small["p_lower"]
    assert large["fraction"] > small["fraction"]


def test_more_samples_at_breakeven_still_refuses():
    """樣本大但機率就是在兩平點附近時，仍然不該下注。"""
    r = suggest_position(wins=334, n=1000, rr=2.0)
    assert r["fraction"] == 0.0


# --------------------------------------------------------------------------
# Wilson 下界
# --------------------------------------------------------------------------
def test_wilson_is_below_point_estimate():
    assert wilson_lower_bound(15, 22) < 15 / 22


def test_wilson_handles_empty():
    assert wilson_lower_bound(0, 0) is None


# --------------------------------------------------------------------------
# 三重障礙統計
# --------------------------------------------------------------------------
def test_barrier_stats_excludes_timeouts_from_win_rate():
    """timeout 既不是贏也不是輸，不該混進勝率的分母。"""
    s = barrier_stats(["target", "stop", "timeout", "target"])
    assert s["n_total"] == 4
    assert s["n_resolved"] == 3
    assert s["wins"] == 2
    assert s["timeout_rate"] == pytest.approx(0.25)


def test_barrier_stats_on_empty():
    s = barrier_stats([])
    assert s["n_total"] == 0 and s["timeout_rate"] == 0.0


# --------------------------------------------------------------------------
# 文字輸出
# --------------------------------------------------------------------------
def test_describe_states_the_divisor_used():
    r = suggest_position(wins=300, n=500, rr=2.0, kelly_divisor=4.0)
    assert "1/4 Kelly" in describe("TEST", r)


def test_describe_explains_a_refusal():
    r = suggest_position(wins=5, n=30, rr=2.0)
    text = describe("TEST", r)
    assert "不建議下注" in text and "兩平點" in text
