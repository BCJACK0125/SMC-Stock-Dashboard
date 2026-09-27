# -*- coding: utf-8 -*-
"""
tests/test_analyzer_regression.py
==================================
針對曾經出現過的五個靜默失效 bug 的回歸測試。

這些 bug 的共通點是「不會拋例外，只會讓評分項永遠拿不到分」，
所以用合成 K 線斷言統計特徵，而不是斷言單一數值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smc.analyzer import SMCAnalyzer


def make_ohlcv(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    """產生一段有趨勢也有回檔的隨機遊走，模擬 4H K 線。"""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(1_000_000, 5_000_000, n)},
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


@pytest.fixture(scope="module")
def analyzer() -> SMCAnalyzer:
    return SMCAnalyzer(make_ohlcv(), swing_lookback=3).run_all()


@pytest.fixture(scope="module")
def bar_pos(analyzer: SMCAnalyzer) -> dict:
    return {ts: i for i, ts in enumerate(analyzer.df.index)}


# --------------------------------------------------------------------------
# Bug 1：OB 一形成就被判定為已緩解，active_order_blocks() 永遠是空的
# --------------------------------------------------------------------------
def test_active_order_blocks_not_all_mitigated(analyzer: SMCAnalyzer):
    assert analyzer.order_blocks, "測試資料應該要產生一些 Order Block"
    assert analyzer.active_order_blocks(), (
        "所有 OB 都被標記為 mitigated —— notifier.score() 的 OB 項（多空各 15 分）"
        "會永遠拿不到分"
    )


def test_order_block_mitigation_is_not_immediate(analyzer: SMCAnalyzer, bar_pos: dict):
    """OB 不該在形成後 1 根 K 棒內就被判定緩解（那是單邊條件造成的假陽性）。"""
    lags = [
        bar_pos[ob.mitigated_index] - bar_pos[ob.start_index]
        for ob in analyzer.order_blocks
        if ob.mitigated
    ]
    assert lags, "測試資料應該要有一些 OB 最終被緩解"
    immediate = sum(1 for lag in lags if lag <= 1)
    assert immediate == 0, f"{immediate}/{len(lags)} 個 OB 在 1 根K棒內就被判定緩解"


def test_mitigation_requires_range_overlap(analyzer: SMCAnalyzer):
    """被標記緩解的那根 K 棒，必須真的與 OB 區間重疊。"""
    for ob in analyzer.order_blocks:
        if not ob.mitigated:
            continue
        bar = analyzer.df.loc[ob.mitigated_index]
        assert bar["Low"] <= ob.top and bar["High"] >= ob.bottom, (
            f"{ob.side} OB @ {ob.start_index} 的緩解棒並未進入 "
            f"[{ob.bottom:.2f}, {ob.top:.2f}] 區間"
        )


# --------------------------------------------------------------------------
# Bug 2：突破後結構位被設回同一個 swing，導致連續重複觸發同一個 BOS
# --------------------------------------------------------------------------
def test_no_duplicate_structure_events(analyzer: SMCAnalyzer):
    events = analyzer.structure_events
    assert events, "測試資料應該要產生一些結構事件"
    dups = [
        (b.index, b.broken_level)
        for a, b in zip(events, events[1:])
        if a.side == b.side and a.broken_level == b.broken_level
    ]
    assert not dups, f"有 {len(dups)} 個結構事件重複突破同一個價位，前三個：{dups[:3]}"


def test_each_structure_level_broken_at_most_once(analyzer: SMCAnalyzer):
    """同一個 swing（以 broken_index 識別）在同方向上只能被突破一次。"""
    seen = set()
    for ev in analyzer.structure_events:
        key = (ev.broken_index, ev.side)
        assert key not in seen, f"swing @ {ev.broken_index} 被重複突破（{ev.side}）"
        seen.add(key)


def test_structure_event_actually_breaks_its_level(analyzer: SMCAnalyzer):
    for ev in analyzer.structure_events:
        assert ev.broken_index < ev.index, "被突破的 swing 必須早於突破確認時間"
        if ev.side == "bullish":
            assert ev.price > ev.broken_level
        else:
            assert ev.price < ev.broken_level


# --------------------------------------------------------------------------
# Bug 3：OB 搜尋窗口從資料最開頭算起，抓到推動段以外的 K 棒
# --------------------------------------------------------------------------
def test_order_block_sits_inside_its_impulse_leg(analyzer: SMCAnalyzer):
    """OB 必須落在「被突破的 swing」到「突破確認」之間，且不能是突破棒本身。"""
    for ob in analyzer.order_blocks:
        ev = ob.caused_structure
        assert ev is not None
        assert ev.broken_index <= ob.start_index < ev.index, (
            f"OB @ {ob.start_index} 落在推動腿 "
            f"[{ev.broken_index}, {ev.index}) 之外"
        )


def test_order_block_is_an_opposing_candle(analyzer: SMCAnalyzer):
    for ob in analyzer.order_blocks:
        bar = analyzer.df.loc[ob.start_index]
        if ob.side == "bullish":
            assert bar["Close"] < bar["Open"], "看漲 OB 應該是一根收黑的反向棒"
        else:
            assert bar["Close"] > bar["Open"], "看跌 OB 應該是一根收紅的反向棒"


# --------------------------------------------------------------------------
# Bug 5：LiquidityPool.swept / swept_index 宣告了卻從未被賦值
# --------------------------------------------------------------------------
def test_liquidity_sweeps_are_marked(analyzer: SMCAnalyzer):
    assert analyzer.liquidity_pools, "測試資料應該要產生一些流動性池"
    assert any(p.swept for p in analyzer.liquidity_pools), (
        "沒有任何流動性池被標記為 swept —— notifier.score() 的掃蕩項（7 分）"
        "會永遠拿不到分"
    )


def test_sweep_index_is_consistent(analyzer: SMCAnalyzer):
    for pool in analyzer.liquidity_pools:
        if not pool.swept:
            continue
        assert pool.swept_index is not None
        assert pool.swept_index > pool.confirmed_index, "掃蕩必須發生在池子被確認之後"
        bar = analyzer.df.loc[pool.swept_index]
        if pool.kind == "EQH":
            assert bar["High"] > pool.price
        else:
            assert bar["Low"] < pool.price


# --------------------------------------------------------------------------
# FVG 最小缺口過濾（ATR 相對值）
# --------------------------------------------------------------------------
def test_fvg_atr_filter_removes_noise_gaps():
    """
    不過濾時 FVG 幾乎每兩根K棒就出現一個，評分裡的 FVG 項近似恆真。

    斷言的是「保留比例」而不是「佔K棒的比例」：缺口的絕對密度取決於價格
    序列本身（實測隨機遊走比真實股價更容易產生缺口），但過濾器能控制的是
    保留比例——實測合成資料與 5 檔真實標的在 0.5 ATR 下都落在 40~60%。
    """
    df = make_ohlcv()
    unfiltered = SMCAnalyzer(df, swing_lookback=3).run_all().fvgs
    filtered = SMCAnalyzer(df, swing_lookback=3, fvg_min_gap_atr=0.5).run_all().fvgs
    assert unfiltered, "測試資料應該要產生一些 FVG"
    kept = len(filtered) / len(unfiltered)
    assert 0.30 <= kept <= 0.70, f"0.5 ATR 下保留了 {kept:.0%}，過濾強度不在預期範圍"


def test_fvg_filter_is_monotonic():
    """門檻越嚴格，留下的缺口必須越少。"""
    df = make_ohlcv()
    counts = [
        len(SMCAnalyzer(df, swing_lookback=3, fvg_min_gap_atr=t).run_all().fvgs)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert counts == sorted(counts, reverse=True), counts


def test_fvg_atr_filter_keeps_only_wide_enough_gaps():
    from indicators import atr as _atr

    df = make_ohlcv()
    a = SMCAnalyzer(df, swing_lookback=3, fvg_min_gap_atr=0.5).run_all()
    atr_series = _atr(df)
    for g in a.fvgs:
        ref = float(atr_series.loc[g.end_index])
        if ref <= 0:
            continue  # ATR 尚未暖機，依設計不過濾
        assert (g.top - g.bottom) / ref >= 0.5


def test_fvg_filter_is_opt_in():
    """預設不帶參數時行為不變，避免既有呼叫端被靜默改變。"""
    df = make_ohlcv()
    assert len(SMCAnalyzer(df, swing_lookback=3).run_all().fvgs) == \
           len(SMCAnalyzer(df, swing_lookback=3, fvg_min_gap_atr=0.0).run_all().fvgs)
