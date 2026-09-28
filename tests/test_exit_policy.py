# -*- coding: utf-8 -*-
"""
tests/test_exit_policy.py
==========================
回撤預算 → 移動停損寬度。

這張表最容易被誤用的地方是把它當成「回撤保證」。樣本外實測預算 20% 時
仍有 18/22 檔超標，所以 describe() 一定要把超標率帶出來，讓儀表板揭露。
"""
from __future__ import annotations

import pytest

import exit_policy as EP


def test_bigger_budget_never_gives_a_tighter_stop():
    prev = 0.0
    for b, *_ in EP.CALIBRATION:
        m = EP.trail_mult_for(b)
        assert m >= prev, f"預算放寬到 {b} 反而收緊停損"
        prev = m


def test_budget_below_the_table_falls_back_to_the_most_conservative_row():
    assert EP.trail_mult_for(0.01) == EP.CALIBRATION[0][1]
    assert EP.trail_mult_for(0.05) == EP.CALIBRATION[0][1]


def test_budget_above_the_table_uses_the_last_row():
    assert EP.trail_mult_for(0.95) == EP.CALIBRATION[-1][1]


def test_a_budget_between_rows_takes_the_lower_row():
    """預算 0.33 不能拿 0.35 那列——那會超出使用者說的上限。"""
    assert EP.trail_mult_for(0.33) == EP.trail_mult_for(0.30)


@pytest.mark.parametrize("budget", [0.20, 0.30, 0.40, 0.50])
def test_expected_drawdown_stays_within_the_budget(budget):
    d = EP.describe(budget)
    assert abs(d["expected_drawdown"]) <= budget + 1e-9


def test_describe_exposes_the_out_of_sample_breach_rate():
    """沒有這個欄位，儀表板就會把預算講成保證。"""
    d = EP.describe(0.20)
    assert d["oos_breach"] == "18/22"
    assert abs(d["oos_drawdown"]) > 0.20, "樣本外實際回撤確實超過預算，要如實呈現"


def test_the_tightest_setting_is_flagged_as_the_floor():
    assert EP.describe(0.20)["floor_reached"] is True
    assert EP.describe(0.40)["floor_reached"] is False


def test_tightening_below_the_floor_buys_nothing():
    """收到最緊也換不到更低的樣本外回撤，只換到更低的報酬。"""
    tight, loose = EP.describe(0.20), EP.describe(0.50)
    assert tight["expected_cagr"] < loose["expected_cagr"]
    assert abs(tight["oos_drawdown"]) >= abs(loose["oos_drawdown"]) - 0.05


def test_config_derives_the_multiplier_from_the_budget():
    import config
    assert config.TRADE_TRAIL_ATR_MULT == EP.trail_mult_for(config.DRAWDOWN_BUDGET)
