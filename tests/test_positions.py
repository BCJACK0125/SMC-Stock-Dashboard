# -*- coding: utf-8 -*-
"""
tests/test_positions.py
========================
持倉追蹤：把「移動停損 2×ATR」換算成今天的實際價格。

這裡最要緊的性質是「停損只升不降」——一旦會下降，使用者按表操作就會
在已經鎖定獲利之後又把停損放回去，等於白做。
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

import positions as P


def make_df(highs, closes=None, start="2026-01-01"):
    n = len(highs)
    closes = closes or highs
    return pd.DataFrame(
        {"Open": closes, "High": highs,
         "Low": [c * 0.98 for c in closes], "Close": closes,
         "Volume": [1_000_000] * n},
        index=pd.date_range(start, periods=n, freq="B"))


def atr_of(df, v=2.0):
    return pd.Series([v] * len(df), index=df.index)


# --------------------------------------------------------------------------
# 讀檔
# --------------------------------------------------------------------------
def test_missing_file_is_empty_not_an_error(tmp_path):
    assert P.load(str(tmp_path / "nope.json")) == []


def test_corrupt_file_does_not_break_the_run(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ 壞掉", encoding="utf-8")
    assert P.load(str(p)) == []


def test_entries_without_a_symbol_are_dropped(tmp_path):
    p = tmp_path / "pos.json"
    p.write_text(json.dumps([{"symbol": "NVDA", "entry_price": 1},
                             {"entry_price": 2}, "垃圾"]), encoding="utf-8")
    assert [x["symbol"] for x in P.load(str(p))] == ["NVDA"]


# --------------------------------------------------------------------------
# 停損計算
# --------------------------------------------------------------------------
def test_stop_starts_at_the_given_structural_stop():
    df = make_df([100, 100, 100])
    t = P.track({"symbol": "X", "entry_date": "2026-01-01",
                 "entry_price": 100, "stop": 92}, df, atr_of(df), 2.0)
    # 峰值 100 − 2×ATR(2) = 96，比結構停損 92 高，所以會被推上去
    assert t["initial_stop"] == 92
    assert t["stop"] == pytest.approx(96)


def test_stop_never_moves_down_when_price_falls_back():
    """漲上去再跌回來，停損要停在高點推上去的位置。"""
    df = make_df([100, 130, 105])
    t = P.track({"symbol": "X", "entry_date": "2026-01-01",
                 "entry_price": 100, "stop": 92}, df, atr_of(df), 2.0)
    assert t["peak"] == 130
    assert t["stop"] == pytest.approx(126)      # 130 − 4，不是 105 − 4


def test_wider_multiplier_gives_a_looser_stop():
    df = make_df([100, 130])
    a = P.track({"symbol": "X", "entry_date": "2026-01-01",
                 "entry_price": 100, "stop": 92}, df, atr_of(df), 2.0)["stop"]
    b = P.track({"symbol": "X", "entry_date": "2026-01-01",
                 "entry_price": 100, "stop": 92}, df, atr_of(df), 4.0)["stop"]
    assert b < a


def test_stop_defaults_to_atr_distance_when_not_supplied():
    df = make_df([100])
    t = P.track({"symbol": "X", "entry_date": "2026-01-01", "entry_price": 100},
                df, atr_of(df), 3.0)
    assert t["initial_stop"] == pytest.approx(94)     # 100 − 3×2


def test_only_bars_from_the_entry_date_count_toward_the_peak():
    """進場前的高點不能拿來墊高停損。"""
    df = make_df([200, 100, 110])
    t = P.track({"symbol": "X", "entry_date": "2026-01-02",
                 "entry_price": 100, "stop": 90}, df, atr_of(df), 2.0)
    assert t["peak"] == 110                            # 不是 200
    assert t["bars_held"] == 2


# --------------------------------------------------------------------------
# 衍生欄位
# --------------------------------------------------------------------------
def test_locked_in_once_the_stop_clears_the_entry():
    df = make_df([100, 120])
    t = P.track({"symbol": "X", "entry_date": "2026-01-01",
                 "entry_price": 100, "stop": 90}, df, atr_of(df), 2.0)
    assert t["stop"] == pytest.approx(116) and t["locked_in"] is True


def test_not_locked_in_while_the_stop_is_still_below_entry():
    df = make_df([100, 101])
    t = P.track({"symbol": "X", "entry_date": "2026-01-01",
                 "entry_price": 100, "stop": 90}, df, atr_of(df), 2.0)
    assert t["locked_in"] is False


def test_breach_is_flagged_when_price_closes_under_the_stop():
    df = make_df([100, 100, 100], closes=[100, 100, 93])
    t = P.track({"symbol": "X", "entry_date": "2026-01-01",
                 "entry_price": 100, "stop": 96}, df, atr_of(df), 2.0)
    assert t["breached"] is True


def test_unrealized_r_uses_the_initial_risk():
    df = make_df([100, 110], closes=[100, 110])
    t = P.track({"symbol": "X", "entry_date": "2026-01-01",
                 "entry_price": 100, "stop": 95}, df, atr_of(df), 2.0)
    assert t["unrealized_r"] == pytest.approx(2.0)     # +10 / 風險 5


# --------------------------------------------------------------------------
# 壞資料
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"symbol": "X", "entry_date": "2026-01-01"},              # 沒有進場價
    {"symbol": "X", "entry_date": "2026-01-01", "entry_price": 0},
    {"symbol": "X", "entry_price": 100},                      # 沒有進場日
    {"symbol": "X", "entry_date": "不是日期", "entry_price": 100},
    {"symbol": "X", "entry_date": "2099-01-01", "entry_price": 100},  # 未來
])
def test_unusable_records_return_none(bad):
    df = make_df([100, 101])
    assert P.track(bad, df, atr_of(df), 2.0) is None


def test_zero_atr_falls_back_to_the_initial_stop():
    df = make_df([100, 130])
    t = P.track({"symbol": "X", "entry_date": "2026-01-01",
                 "entry_price": 100, "stop": 92}, df,
                pd.Series([0.0, 0.0], index=df.index), 2.0)
    assert t["stop"] == 92


def test_describe_mentions_the_actual_number():
    df = make_df([100, 120])
    t = P.track({"symbol": "X", "entry_date": "2026-01-01",
                 "entry_price": 100, "stop": 90}, df, atr_of(df), 2.0)
    s = P.describe(t)
    assert "116.00" in s and "已鎖定獲利" in s
