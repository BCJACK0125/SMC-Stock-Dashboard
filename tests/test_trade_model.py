# -*- coding: utf-8 -*-
"""
tests/test_trade_model.py
==========================
出場邏輯的測試：停損/目標的來源選擇，以及模擬時的保守假設。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import pytest

from trade_model import TradePlan, plan_trade, simulate_trade


# --------------------------------------------------------------------------
# 假的 view：只要提供 active_order_blocks() 與 liquidity_pools 就夠了
# --------------------------------------------------------------------------
@dataclass
class FakeOB:
    top: float
    bottom: float
    side: str


@dataclass
class FakePool:
    price: float
    kind: str
    swept: bool = False


class FakeView:
    def __init__(self, obs: Optional[List[FakeOB]] = None,
                 pools: Optional[List[FakePool]] = None):
        self._obs = obs or []
        self.liquidity_pools = pools or []

    def active_order_blocks(self, side=None):
        return [o for o in self._obs if side is None or o.side == side]


def bars(rows) -> pd.DataFrame:
    """rows: [(open, high, low, close), ...]"""
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"],
                        index=pd.date_range("2024-01-01", periods=len(rows), freq="4h"))


# --------------------------------------------------------------------------
# 停損來源
# --------------------------------------------------------------------------
def test_long_stop_sits_below_the_order_block_with_buffer():
    view = FakeView(obs=[FakeOB(top=98, bottom=95, side="bullish")])
    p = plan_trade(view, "bullish", entry=100, atr=2.0, stop_buffer_atr=0.25)
    assert p.stop_source == "order_block"
    assert p.stop == pytest.approx(95 - 0.5)   # OB 下緣 - 0.25*ATR


def test_short_stop_sits_above_the_order_block():
    view = FakeView(obs=[FakeOB(top=105, bottom=102, side="bearish")])
    p = plan_trade(view, "bearish", entry=100, atr=2.0, stop_buffer_atr=0.25)
    assert p.stop_source == "order_block"
    assert p.stop == pytest.approx(105 + 0.5)


def test_falls_back_to_atr_stop_when_no_order_block():
    p = plan_trade(FakeView(), "bullish", entry=100, atr=2.0, atr_stop_mult=1.5)
    assert p.stop_source == "atr"
    assert p.stop == pytest.approx(97.0)


def test_ignores_order_blocks_on_the_wrong_side_of_entry():
    """進場價下方才是多單的失效區；上方的 OB 當停損會變成負風險。"""
    view = FakeView(obs=[FakeOB(top=110, bottom=108, side="bullish")])
    p = plan_trade(view, "bullish", entry=100, atr=2.0)
    assert p.stop_source == "atr"
    assert p.stop < p.entry


def test_picks_the_nearest_order_block_when_several_exist():
    view = FakeView(obs=[FakeOB(top=98, bottom=95, side="bullish"),
                         FakeOB(top=92, bottom=90, side="bullish")])
    p = plan_trade(view, "bullish", entry=100, atr=2.0, stop_buffer_atr=0.0)
    assert p.stop == pytest.approx(95), "應該用最近的那個 OB，風險才最小"


# --------------------------------------------------------------------------
# 目標來源
# --------------------------------------------------------------------------
def test_long_target_is_the_nearest_unswept_eqh_above():
    view = FakeView(obs=[FakeOB(top=98, bottom=95, side="bullish")],
                    pools=[FakePool(price=106, kind="EQH"),
                           FakePool(price=112, kind="EQH")])
    p = plan_trade(view, "bullish", entry=100, atr=2.0, stop_buffer_atr=0.0)
    assert p.target_source == "liquidity"
    assert p.target == pytest.approx(106)


def test_already_swept_pools_are_not_targets():
    """被掃過的流動性池，停損單已經被吃掉了，不再是價格的磁鐵。"""
    view = FakeView(obs=[FakeOB(top=98, bottom=95, side="bullish")],
                    pools=[FakePool(price=106, kind="EQH", swept=True)])
    p = plan_trade(view, "bullish", entry=100, atr=2.0, stop_buffer_atr=0.0)
    assert p.target_source == "r_multiple"


def test_target_falls_back_to_r_multiple_without_pools():
    view = FakeView(obs=[FakeOB(top=98, bottom=95, side="bullish")])
    p = plan_trade(view, "bullish", entry=100, atr=2.0,
                   stop_buffer_atr=0.0, fallback_target_r=2.0)
    assert p.target_source == "r_multiple"
    assert p.target == pytest.approx(100 + 2 * 5)   # 風險 5 -> 目標 +10


def test_too_close_liquidity_is_rejected():
    view = FakeView(obs=[FakeOB(top=98, bottom=95, side="bullish")],
                    pools=[FakePool(price=100.5, kind="EQH")])
    p = plan_trade(view, "bullish", entry=100, atr=2.0,
                   stop_buffer_atr=0.0, min_rr=0.5)
    assert p.target_source == "r_multiple", "RR 只有 0.1 的目標不該採用"


def test_short_target_uses_eql_below():
    view = FakeView(obs=[FakeOB(top=105, bottom=102, side="bearish")],
                    pools=[FakePool(price=94, kind="EQL")])
    p = plan_trade(view, "bearish", entry=100, atr=2.0, stop_buffer_atr=0.0)
    assert p.target_source == "liquidity" and p.target == pytest.approx(94)


# --------------------------------------------------------------------------
# 模擬出場
# --------------------------------------------------------------------------
def _plan(side="bullish", entry=100, stop=95, target=110):
    return TradePlan(side=side, entry=entry, stop=stop, target=target,
                     stop_source="order_block", target_source="liquidity")


def test_target_hit_produces_positive_return():
    res = simulate_trade(bars([(100, 103, 99, 102), (102, 111, 101, 110)]), _plan())
    assert res.outcome == "target"
    assert res.bars_held == 2
    assert res.return_pct == pytest.approx(0.10)
    assert res.r_multiple == pytest.approx(2.0)


def test_stop_hit_produces_negative_return():
    res = simulate_trade(bars([(100, 101, 94, 96)]), _plan())
    assert res.outcome == "stop"
    assert res.return_pct == pytest.approx(-0.05)
    assert res.r_multiple == pytest.approx(-1.0)


def test_same_bar_touching_both_is_counted_as_a_stop():
    """
    用 OHLC 無法還原日內先後順序。假設先到目標會系統性高估績效，
    所以一律當作先觸及停損——這是回測的標準保守做法。
    """
    res = simulate_trade(bars([(100, 115, 90, 100)]), _plan())
    assert res.outcome == "stop"


def test_timeout_exits_at_last_close():
    rows = [(100, 101, 99, 100)] * 5
    res = simulate_trade(bars(rows), _plan(), max_holding_bars=3)
    assert res.outcome == "timeout"
    assert res.bars_held == 3
    assert res.exit_price == pytest.approx(100)


def test_short_trade_profits_when_price_falls():
    plan = _plan(side="bearish", entry=100, stop=105, target=90)
    res = simulate_trade(bars([(100, 101, 89, 92)]), plan)
    assert res.outcome == "target"
    assert res.return_pct == pytest.approx(0.10)
    assert res.r_multiple == pytest.approx(2.0)


def test_no_future_bars_is_a_flat_timeout():
    res = simulate_trade(bars([]), _plan())
    assert res.outcome == "timeout" and res.return_pct == 0.0


def test_max_holding_bars_caps_the_scan():
    """時間停損之後發生的事，不該影響這筆交易的結果。"""
    rows = [(100, 101, 99, 100)] * 4 + [(100, 120, 99, 118)]
    res = simulate_trade(bars(rows), _plan(), max_holding_bars=4)
    assert res.outcome == "timeout", "第 5 根才觸及的目標不該被算進來"


# --------------------------------------------------------------------------
# 風險報酬比
# --------------------------------------------------------------------------
def test_rr_is_reward_over_risk():
    p = _plan(entry=100, stop=95, target=115)
    assert p.risk == pytest.approx(5) and p.reward == pytest.approx(15)
    assert p.rr == pytest.approx(3.0)


def test_plan_rejects_degenerate_inputs():
    assert plan_trade(FakeView(), "bullish", entry=100, atr=0.0) is None
    assert plan_trade(FakeView(), "bullish", entry=0.0, atr=1.0) is None


# --------------------------------------------------------------------------
# 移動停損出場
# --------------------------------------------------------------------------
def test_trailing_lets_winners_run_past_the_fixed_target():
    """
    固定目標模式在 +10% 就出場；移動停損應該一路跟著趨勢跑。
    這正是為了修正「固定目標在趨勢行情砍掉贏家」而加的模式。
    """
    from trade_model import simulate_trailing_trade
    rows = [(100 + i, 102 + i, 99 + i, 101 + i) for i in range(30)]
    plan = _plan(entry=100, stop=95, target=110)
    fixed = simulate_trade(bars(rows), plan)
    trail = simulate_trailing_trade(bars(rows), plan, atr=2.0, trail_atr_mult=3.0)
    assert fixed.outcome == "target" and fixed.return_pct == pytest.approx(0.10)
    assert trail.return_pct > fixed.return_pct


def test_trailing_still_honours_the_initial_structural_stop():
    from trade_model import simulate_trailing_trade
    r = simulate_trailing_trade(bars([(100, 101, 94, 96)]), _plan(), atr=2.0)
    assert r.outcome == "stop"
    assert r.return_pct == pytest.approx(-0.05)


def test_trailing_stop_never_moves_backwards():
    """價格衝高後回落，停損應該守在推高後的位置，而不是退回原點。"""
    from trade_model import simulate_trailing_trade
    rows = [(100, 120, 99, 118),    # 衝高 -> 停損上移到 120-6=114
            (118, 119, 100, 101)]   # 回落跌破 114 -> 該在 114 出場
    r = simulate_trailing_trade(bars(rows), _plan(), atr=2.0, trail_atr_mult=3.0)
    assert r.outcome == "stop"
    assert r.exit_price == pytest.approx(114.0)
    assert r.return_pct > 0, "推高後的停損應該鎖住獲利"


def test_trailing_short_mirrors_long():
    from trade_model import simulate_trailing_trade
    plan = _plan(side="bearish", entry=100, stop=105, target=90)
    rows = [(100, 101, 80, 82), (82, 95, 81, 94)]
    r = simulate_trailing_trade(bars(rows), plan, atr=2.0, trail_atr_mult=3.0)
    assert r.outcome == "stop"
    assert r.return_pct > 0


def test_trailing_with_no_future_bars():
    from trade_model import simulate_trailing_trade
    r = simulate_trailing_trade(bars([]), _plan(), atr=2.0)
    assert r.outcome == "timeout" and r.return_pct == 0.0


def test_trailing_needs_a_long_time_stop_to_work():
    """
    移動停損配上短時間停損等於自廢武功——趨勢單會在還沒跑完就被砍掉，
    退化成固定持有。config 因此為兩種模式分開設定時間停損長度。
    """
    from trade_model import simulate_trailing_trade
    rows = [(100 + i, 102 + i, 99 + i, 101 + i) for i in range(60)]
    short_stop = simulate_trailing_trade(bars(rows), _plan(), atr=2.0,
                                         trail_atr_mult=3.0, max_holding_bars=20)
    long_stop = simulate_trailing_trade(bars(rows), _plan(), atr=2.0,
                                        trail_atr_mult=3.0, max_holding_bars=250)
    assert long_stop.return_pct > short_stop.return_pct


def test_config_gives_trailing_a_longer_time_stop():
    import config
    assert (config.TRADE_TRAILING_MAX_HOLDING_BARS
            > config.TRADE_MAX_HOLDING_BARS)


# --------------------------------------------------------------------------
# 停損縮放（覆盤發現：獲利單的 MAE 中位數只有初始風險的 16%）
# --------------------------------------------------------------------------
def test_stop_risk_scale_moves_stop_closer_to_entry():
    view = FakeView(obs=[FakeOB(top=98, bottom=90, side="bullish")])
    full = plan_trade(view, "bullish", entry=100, atr=2.0, stop_buffer_atr=0.0)
    tight = plan_trade(view, "bullish", entry=100, atr=2.0, stop_buffer_atr=0.0,
                       stop_risk_scale=0.4)
    assert full.stop == pytest.approx(90.0)
    assert tight.stop == pytest.approx(96.0)      # 100 - 0.4*(100-90)
    assert tight.risk == pytest.approx(full.risk * 0.4)


def test_stop_scale_of_one_is_a_noop():
    view = FakeView(obs=[FakeOB(top=98, bottom=95, side="bullish")])
    a = plan_trade(view, "bullish", entry=100, atr=2.0, stop_risk_scale=1.0)
    b = plan_trade(view, "bullish", entry=100, atr=2.0)
    assert a.stop == pytest.approx(b.stop)


def test_tighter_stop_improves_rr_for_the_same_target():
    view = FakeView(obs=[FakeOB(top=98, bottom=90, side="bullish")],
                    pools=[FakePool(price=120, kind="EQH")])
    full = plan_trade(view, "bullish", entry=100, atr=2.0, stop_buffer_atr=0.0)
    tight = plan_trade(view, "bullish", entry=100, atr=2.0, stop_buffer_atr=0.0,
                       stop_risk_scale=0.4)
    assert tight.rr > full.rr


# --------------------------------------------------------------------------
# 保本停損（覆盤發現：72% 的虧損單過程中曾經浮盈 >1%）
# --------------------------------------------------------------------------
def test_breakeven_stop_converts_a_giveback_into_flat():
    from trade_model import simulate_trailing_trade
    plan = _plan(entry=100, stop=90, target=130)          # 風險 10
    rows = [(100, 120, 99, 118), (118, 119, 95, 96)]      # 先到 +2R 再回跌
    with_be = simulate_trailing_trade(bars(rows), plan, atr=5.0,
                                      trail_atr_mult=10.0, breakeven_at_r=2.0)
    without = simulate_trailing_trade(bars(rows), plan, atr=5.0, trail_atr_mult=10.0)
    assert with_be.exit_price == pytest.approx(100.0)
    assert with_be.return_pct == pytest.approx(0.0)
    assert without.return_pct < 0


def test_breakeven_does_not_arm_below_threshold():
    from trade_model import simulate_trailing_trade
    plan = _plan(entry=100, stop=90, target=130)
    rows = [(100, 105, 99, 104), (104, 105, 88, 89)]      # 只到 +0.5R
    r = simulate_trailing_trade(bars(rows), plan, atr=5.0,
                                trail_atr_mult=10.0, breakeven_at_r=2.0)
    assert r.exit_price == pytest.approx(90.0), "沒達門檻不該保本"


def test_breakeven_never_loosens_an_already_tighter_trail():
    """移動停損已經推到成本價之上時，保本不該把它往回拉。"""
    from trade_model import simulate_trailing_trade
    plan = _plan(entry=100, stop=90, target=200)
    rows = [(100, 130, 99, 128), (128, 129, 118, 119)]
    r = simulate_trailing_trade(bars(rows), plan, atr=5.0,
                                trail_atr_mult=1.0, breakeven_at_r=1.0)
    assert r.exit_price > 100.0


def test_breakeven_preserves_big_winners():
    """
    關鍵風險：前 10% 的交易貢獻全部獲利，保本門檻設太低會把未來的大贏家
    在正常回檔時洗掉。一路上漲的單子不該受保本影響。
    """
    from trade_model import simulate_trailing_trade
    plan = _plan(entry=100, stop=90, target=500)
    rows = [(100 + 3 * i, 104 + 3 * i, 99 + 3 * i, 103 + 3 * i) for i in range(30)]
    with_be = simulate_trailing_trade(bars(rows), plan, atr=5.0,
                                      trail_atr_mult=3.0, breakeven_at_r=2.0)
    without = simulate_trailing_trade(bars(rows), plan, atr=5.0, trail_atr_mult=3.0)
    assert with_be.return_pct == pytest.approx(without.return_pct)


def test_breakeven_is_opt_in():
    from trade_model import simulate_trailing_trade
    rows = [(100, 120, 99, 118), (118, 119, 95, 96)]
    plan = _plan(entry=100, stop=90, target=130)
    a = simulate_trailing_trade(bars(rows), plan, atr=5.0, trail_atr_mult=10.0)
    b = simulate_trailing_trade(bars(rows), plan, atr=5.0, trail_atr_mult=10.0,
                                breakeven_at_r=None)
    assert a.return_pct == pytest.approx(b.return_pct)


# --------------------------------------------------------------------------
# 進場計畫（隔天限價 + fallback）
# --------------------------------------------------------------------------
def test_limit_is_placed_below_the_signal_close():
    from trade_model import plan_entry
    p = plan_entry(100.0, 4.0, offset_atr=0.5)
    assert p.limit_price == pytest.approx(98.0)
    assert p.reference_close == pytest.approx(100.0)


def test_zero_offset_places_limit_at_the_close():
    from trade_model import plan_entry
    assert plan_entry(100.0, 4.0, offset_atr=0.0).limit_price == pytest.approx(100.0)


def test_plan_entry_rejects_degenerate_inputs():
    from trade_model import plan_entry
    assert plan_entry(0.0, 4.0) is None
    assert plan_entry(100.0, 0.0) is None


def test_fill_at_limit_when_low_touches():
    from trade_model import plan_entry, resolve_entry
    p = plan_entry(100.0, 4.0, offset_atr=0.5)          # 限價 98
    assert resolve_entry(bars([(99, 100, 97, 99)]), p) == (98.0, 0)


def test_gap_down_fills_at_the_better_open_price():
    """跳空開低時成交價是開盤價，不是限價——你拿到的是更好的價格。"""
    from trade_model import plan_entry, resolve_entry
    p = plan_entry(100.0, 4.0, offset_atr=0.5)
    assert resolve_entry(bars([(96, 99, 95, 98)]), p) == (96.0, 0)


def test_unfilled_limit_falls_back_to_market():
    """
    這是整個進場設計裡最不能省的一環：實測同樣的限價距離，
    未成交改市價總報酬 +1553%、未成交放棄只有 +1137%（差 27%），
    因為放棄掉的那批包含了最強的走勢。
    """
    from trade_model import plan_entry, resolve_entry
    p = plan_entry(100.0, 4.0, offset_atr=0.5, valid_bars=1, fallback="market")
    price, off = resolve_entry(bars([(101, 103, 99, 102), (103, 104, 102, 103)]), p)
    assert off == 1 and price == pytest.approx(103.0)


def test_skip_fallback_returns_none():
    from trade_model import plan_entry, resolve_entry
    p = plan_entry(100.0, 4.0, offset_atr=0.5, valid_bars=1, fallback="skip")
    assert resolve_entry(bars([(101, 103, 99, 102), (103, 104, 102, 103)]), p) is None


def test_longer_validity_window_keeps_waiting():
    from trade_model import plan_entry, resolve_entry
    p = plan_entry(100.0, 4.0, offset_atr=0.5, valid_bars=3)
    price, off = resolve_entry(
        bars([(101, 102, 99, 101), (101, 102, 100, 101), (100, 101, 97, 98)]), p)
    assert off == 2 and price == pytest.approx(98.0)


def test_describe_mentions_the_fallback():
    from trade_model import plan_entry
    assert "市價" in plan_entry(100.0, 4.0).describe()
    assert "放棄" in plan_entry(100.0, 4.0, fallback="skip").describe()


def test_config_gives_trailing_a_longer_time_stop():
    """
    移動停損配上短時間停損等於自廢武功——趨勢單會在還沒跑完就被砍掉。
    config 因此為兩種出場模式分開設定時間停損長度。
    """
    import config
    assert (config.TRADE_TRAILING_MAX_HOLDING_BARS
            > config.TRADE_MAX_HOLDING_BARS)
