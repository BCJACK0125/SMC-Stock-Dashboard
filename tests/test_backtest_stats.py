# -*- coding: utf-8 -*-
"""
tests/test_backtest_stats.py
=============================
回測統計層的測試：Wilson 下界的正確性，以及門檻挑選是否真的用了下界。
"""
from __future__ import annotations

import pandas as pd
import pytest

import numpy as np

from backtest import (
    bootstrap_mean_lower_bound,
    evaluate_thresholds,
    recommend_threshold,
    select_non_overlapping,
    summarize_trades,
    wilson_lower_bound,
)


# --------------------------------------------------------------------------
# Wilson score interval 下界
# --------------------------------------------------------------------------
def test_wilson_matches_known_values():
    """對照 Wilson score interval 的標準值（95%，z=1.96）。"""
    assert wilson_lower_bound(5, 8) == pytest.approx(0.3058, abs=1e-3)
    assert wilson_lower_bound(19, 30) == pytest.approx(0.4551, abs=1e-3)
    assert wilson_lower_bound(125, 200) == pytest.approx(0.5556, abs=1e-3)


def test_wilson_is_below_point_estimate():
    for wins, n in [(5, 8), (19, 30), (62, 100), (125, 200)]:
        assert wilson_lower_bound(wins, n) < wins / n


def test_wilson_tightens_as_sample_grows():
    """點估計固定在 62.5%，樣本數越大，下界必須單調上升。"""
    lbs = [wilson_lower_bound(round(n * 0.625), n) for n in (8, 16, 30, 50, 100, 200)]
    assert lbs == sorted(lbs)
    assert lbs[0] < 0.35, "小樣本的下界應該要被壓得很低"
    assert lbs[-1] > 0.55, "大樣本的下界應該要收斂到接近點估計"


def test_wilson_stays_in_unit_interval():
    for wins, n in [(0, 1), (1, 1), (0, 50), (50, 50), (1, 3)]:
        lb = wilson_lower_bound(wins, n)
        assert 0.0 <= lb <= 1.0


def test_wilson_handles_empty_sample():
    assert wilson_lower_bound(0, 0) is None


# --------------------------------------------------------------------------
# 不重疊進場
# --------------------------------------------------------------------------
def test_non_overlapping_skips_bars_inside_an_open_trade():
    """連續 10 根訊號其實是在賭同一段行情，不該計成 10 筆獨立樣本。"""
    mask = pd.Series([True] * 10)
    hold = pd.Series([5] * 10)
    assert select_non_overlapping(mask, hold) == [0, 5]


def test_non_overlapping_keeps_separated_signals():
    mask = pd.Series([True, False, False, False, False, False, True])
    hold = pd.Series([3] * 7)
    assert select_non_overlapping(mask, hold) == [0, 6]


def test_non_overlapping_shrinks_sample_size():
    mask = pd.Series([True] * 100)
    assert len(select_non_overlapping(mask, pd.Series([5] * 100))) == 20


# --------------------------------------------------------------------------
# bootstrap 平均值下界
# --------------------------------------------------------------------------
def test_bootstrap_lower_bound_is_below_mean():
    v = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.015, -0.005])
    lb = bootstrap_mean_lower_bound(v)
    assert lb < v.mean()


def test_bootstrap_lower_bound_tightens_with_sample_size():
    rng = np.random.default_rng(0)
    gaps = []
    for n in (20, 200, 2000):
        v = rng.normal(0.01, 0.02, n)
        gaps.append(v.mean() - bootstrap_mean_lower_bound(v))
    assert gaps == sorted(gaps, reverse=True), "樣本越大，下界應該越靠近平均值"


def test_bootstrap_is_deterministic():
    v = np.array([0.01, -0.02, 0.03, 0.005])
    assert bootstrap_mean_lower_bound(v) == bootstrap_mean_lower_bound(v)


# --------------------------------------------------------------------------
# 績效摘要：勝率與期望值的分離
# --------------------------------------------------------------------------
def test_high_win_rate_can_still_have_negative_expectancy():
    """
    這是換掉目標函數的核心理由：7 勝 3 敗（70% 勝率）但賠率只有 0.2，
    期望值是負的。舊的勝率判準會推薦它，新的期望值判準不會。
    """
    returns = np.array([0.01] * 7 + [-0.05] * 3)
    st = summarize_trades(returns, cost=0.0, alpha=0.05)
    assert st["win_rate"] == pytest.approx(0.7)
    assert st["payoff"] == pytest.approx(0.2)
    assert st["expectancy"] < 0


def test_low_win_rate_can_have_positive_expectancy():
    returns = np.array([0.06] * 4 + [-0.01] * 6)
    st = summarize_trades(returns, cost=0.0, alpha=0.05)
    assert st["win_rate"] == pytest.approx(0.4)
    assert st["expectancy"] > 0


def test_cost_is_deducted_from_every_trade():
    returns = np.array([0.01] * 10)
    free = summarize_trades(returns, cost=0.0, alpha=0.05)
    paid = summarize_trades(returns, cost=0.004, alpha=0.05)
    assert paid["expectancy"] == pytest.approx(free["expectancy"] - 0.004)


def test_cost_can_flip_a_marginal_edge_negative():
    returns = np.array([0.002] * 10)
    assert summarize_trades(returns, cost=0.0047, alpha=0.05)["expectancy"] < 0


def test_summarize_handles_empty():
    st = summarize_trades(np.array([]), cost=0.0, alpha=0.05)
    assert st["n"] == 0 and st["expectancy"] is None


# --------------------------------------------------------------------------
# 門檻挑選
# --------------------------------------------------------------------------
def _stat(threshold, n, expectancy, expectancy_lb):
    return {"threshold": threshold, "n": n, "wins": int(n * 0.6),
            "win_rate": 0.6, "win_rate_lb": 0.5,
            "expectancy": expectancy, "expectancy_lb": expectancy_lb,
            "payoff": 1.0, "profit_factor": 1.2, "excess_expectancy": 0.0}


def test_picks_loosest_threshold_with_positive_lower_bound():
    stats = [_stat(35, 200, 0.004, -0.001),
             _stat(40, 120, 0.006, 0.0012),
             _stat(45, 60, 0.009, 0.0030)]
    rec = recommend_threshold(stats, min_trades=30, default_threshold=50)
    assert rec["threshold"] == 40 and rec["confidence"] == "ok"


def test_reports_no_edge_when_nothing_proves_positive():
    """
    沒有任何門檻能證明正期望值時，要明確說「無優勢」，
    而不是挑一個點估計好看的硬推。
    """
    stats = [_stat(35, 200, 0.004, -0.002), _stat(40, 100, 0.008, -0.0005)]
    rec = recommend_threshold(stats, min_trades=30, default_threshold=50)
    assert rec["confidence"] == "no_edge"
    assert rec["threshold"] == 40, "應回報下界最高的那個"


def test_small_samples_are_excluded_regardless_of_expectancy():
    stats = [_stat(35, 200, 0.001, -0.0004), _stat(60, 8, 0.05, 0.02)]
    rec = recommend_threshold(stats, min_trades=30, default_threshold=50)
    assert rec["threshold"] == 35, "不該被 n=8 的漂亮期望值吸引"
    assert rec["confidence"] == "no_edge"


def test_insufficient_data_falls_back_to_default():
    stats = [_stat(35, 8, 0.01, 0.005), _stat(40, 3, 0.02, 0.01)]
    rec = recommend_threshold(stats, min_trades=30, default_threshold=55)
    assert rec["confidence"] == "insufficient_data"
    assert rec["threshold"] == 55 and rec["expectancy"] is None


# --------------------------------------------------------------------------
# evaluate_thresholds 端到端
# --------------------------------------------------------------------------
def _wf(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "bull_score": rng.integers(0, 70, n),
        "bear_score": rng.integers(0, 70, n),
        "future_return": rng.normal(0.001, 0.02, n),
    })


def test_evaluate_reports_base_rate_for_comparison():
    """沒有基準率對照，多頭行情裡任何做多訊號的勝率看起來都很漂亮。"""
    out = evaluate_thresholds(_wf(), [35, 50], cost=0.001, holding_bars=5)
    assert out["base"]["long"]["n"] > 0
    assert out["base"]["long"]["expectancy"] is not None
    for s in out["bull"]:
        if s["expectancy"] is not None:
            assert s["excess_expectancy"] == pytest.approx(
                s["expectancy"] - out["base"]["long"]["expectancy"])


def test_multiple_testing_correction_widens_the_bound():
    """掃越多門檻，下界應該越保守。"""
    wf = _wf()
    cands = [35, 40, 45, 50, 55, 60, 65]
    off = evaluate_thresholds(wf, cands, cost=0.0, multiple_testing_correction=False)
    on = evaluate_thresholds(wf, cands, cost=0.0, multiple_testing_correction=True)
    assert on["base"]["alpha_used"] < 0.05
    pairs = [(a["expectancy_lb"], b["expectancy_lb"])
             for a, b in zip(off["bull"], on["bull"])
             if a["expectancy_lb"] is not None and b["expectancy_lb"] is not None]
    assert pairs and all(corrected <= plain for plain, corrected in pairs)


def test_bear_side_profits_from_falling_prices():
    wf = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=20, freq="4h"),
        "bull_score": [0] * 20,
        "bear_score": [50] * 20,
        "future_return": [-0.02] * 20,
    })
    bear = evaluate_thresholds(wf, [50], cost=0.0, holding_bars=5)["bear"][0]
    assert bear["expectancy"] == pytest.approx(0.02)
    assert bear["win_rate"] == 1.0


# --------------------------------------------------------------------------
# 不重疊進場：搭配「實際持有長度」（第二優先加入停損/停利後的情況）
# --------------------------------------------------------------------------
def test_non_overlap_respects_variable_holding_length():
    """
    加入停損/目標後，每筆交易的持有長度不再固定。早早被停損的交易應該
    很快釋放出下一次進場機會，不該還被當成佔用 horizon 根。
    """
    mask = pd.Series([True] * 8)
    hold = pd.Series([1, 9, 9, 9, 2, 9, 9, 9])   # 第 0 筆 1 根就出場
    assert select_non_overlapping(mask, hold) == [0, 1]


def test_non_overlap_with_long_holds_yields_few_trades():
    mask = pd.Series([True] * 40)
    assert select_non_overlapping(mask, pd.Series([20] * 40)) == [0, 20]


def test_evaluate_uses_simulated_trades_when_present():
    """
    wf_df 帶有 long_ret/long_bars 時，統計必須走模擬交易那條路徑，
    而不是繼續用固定持有的 future_return。
    """
    n = 40
    wf = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="D"),
        "bull_score": [60] * n,
        "bear_score": [0] * n,
        "future_return": [0.99] * n,      # 故意設成離譜值，若被採用會很明顯
        "long_ret": [0.02] * n,
        "long_bars": [4] * n,
        "long_r": [1.0] * n,
        "long_outcome": ["target"] * n,
        "short_ret": [-0.01] * n,
        "short_bars": [4] * n,
        "short_r": [-1.0] * n,
        "short_outcome": ["stop"] * n,
    })
    bull = evaluate_thresholds(wf, [60], cost=0.0)["bull"][0]
    assert bull["expectancy"] == pytest.approx(0.02), "應採用模擬交易報酬"
    assert bull["n"] == 10, "持有 4 根 -> 40 根裡只容得下 10 筆不重疊交易"
    assert bull["avg_bars_held"] == pytest.approx(4.0)


# --------------------------------------------------------------------------
# 評分元件 / 權重分離
# --------------------------------------------------------------------------
def test_default_weights_cover_every_component():
    from notifier import COMPONENT_NAMES, DEFAULT_WEIGHTS
    assert set(DEFAULT_WEIGHTS) == set(COMPONENT_NAMES)


def test_default_weights_reproduce_the_original_scale():
    """
    重構前的權重配置要原封不動保留下來，之後擬合的結果才有對照基準。

    註：原始 docstring 寫「SMC 滿分 60」，但 CHoCH 與 BOS 互斥時的實際
    上限是 15+15+10+15+7 = 62。這裡以實際值為準。
    """
    from notifier import DEFAULT_WEIGHTS as W
    smc = sum(W[k] for k in ("zone", "order_block", "fvg", "choch", "sweep"))
    ind_ = sum(W[k] for k in ("rsi", "macd", "ema_stack", "adx_di"))
    assert smc == 62 and ind_ == 20 and W["ml"] == 20
    assert W["bos"] == 8


def test_choch_outweighs_bos_by_default():
    from notifier import DEFAULT_WEIGHTS as W
    assert W["choch"] > W["bos"], "結構轉變的權重本來就設計得比延續高"


def test_components_are_bounded():
    """每個元件都該在 0~1，權重才是唯一決定尺度的東西。"""
    import numpy as np
    import pandas as pd
    from smc.analyzer import SMCAnalyzer
    import indicators as ind_mod
    from notifier import score_components

    rng = np.random.default_rng(3)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    op = close * (1 + rng.normal(0, 0.003, n))
    hi = np.maximum(op, close) * (1 + abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(op, close) * (1 - abs(rng.normal(0, 0.004, n)))
    df = pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close,
                       "Volume": rng.integers(1e6, 5e6, n)},
                      index=pd.date_range("2024-01-01", periods=n, freq="D"))
    a = SMCAnalyzer(df, swing_lookback=3).run_all()
    c = score_components(a, recent_bars=3, ind_df=ind_mod.compute_indicator_set(df),
                         ml_result={"prob_up": 0.93, "horizon": 5, "trained_rows": 300})
    for side in ("bull", "bear"):
        for k, v in c[side].items():
            assert 0.0 <= v <= 1.0, f"{side}.{k} = {v} 超出 0~1"
    assert c["bull"]["ml"] == pytest.approx(0.86)   # |0.93-0.5|*2


def test_custom_weights_change_the_score():
    """擬合出來的權重要能實際生效，否則整個擬合是白做的。"""
    import numpy as np
    import pandas as pd
    from smc.analyzer import SMCAnalyzer
    from notifier import DEFAULT_WEIGHTS, score

    rng = np.random.default_rng(4)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    op = close * (1 + rng.normal(0, 0.003, n))
    hi = np.maximum(op, close) * (1 + abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(op, close) * (1 - abs(rng.normal(0, 0.004, n)))
    df = pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close,
                       "Volume": rng.integers(1e6, 5e6, n)},
                      index=pd.date_range("2024-01-01", periods=n, freq="D"))
    a = SMCAnalyzer(df, swing_lookback=3).run_all()
    base = score(a, recent_bars=3)
    doubled = score(a, recent_bars=3,
                    weights={k: v * 2 for k, v in DEFAULT_WEIGHTS.items()})
    assert doubled["bull_score"] >= base["bull_score"]
    assert doubled["bear_score"] >= base["bear_score"]
