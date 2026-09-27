# -*- coding: utf-8 -*-
"""
tests/test_concentration.py
============================
集中度監控：同時出現多個訊號時，它們實際上是幾個獨立風險。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from concentration import (
    assess_signals,
    correlation_matrix,
    describe,
    effective_independent_count,
)


def corr_of(mapping) -> pd.DataFrame:
    """由指定的相關矩陣直接建 DataFrame。"""
    syms = list(mapping)
    return pd.DataFrame(mapping, index=syms, columns=syms)


# --------------------------------------------------------------------------
# 有效獨立標的數
# --------------------------------------------------------------------------
def test_uncorrelated_symbols_count_fully():
    c = corr_of({"A": [1.0, 0.0], "B": [0.0, 1.0]})
    assert effective_independent_count(c) == pytest.approx(2.0)


def test_identical_symbols_count_as_one():
    """完全相同的兩檔，實質上只有一個部位的風險。"""
    c = corr_of({"A": [1.0, 1.0], "B": [1.0, 1.0]})
    assert effective_independent_count(c) == pytest.approx(1.0)


def test_partial_correlation_lands_between():
    c = corr_of({"A": [1.0, 0.5], "B": [0.5, 1.0]})
    n = effective_independent_count(c)
    assert 1.0 < n < 2.0
    assert n == pytest.approx(1 / 0.75)


def test_subset_selection():
    c = corr_of({"A": [1.0, 1.0, 0.0], "B": [1.0, 1.0, 0.0], "C": [0.0, 0.0, 1.0]})
    assert effective_independent_count(c, ["A", "B"]) == pytest.approx(1.0)
    assert effective_independent_count(c, ["A", "C"]) == pytest.approx(2.0)


def test_single_or_empty_input():
    c = corr_of({"A": [1.0, 0.0], "B": [0.0, 1.0]})
    assert effective_independent_count(c, ["A"]) == 1.0
    assert effective_independent_count(pd.DataFrame()) is None
    assert effective_independent_count(c, ["ZZZ"]) is None


# --------------------------------------------------------------------------
# 相關矩陣
# --------------------------------------------------------------------------
def test_correlation_matrix_from_prices():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(0)
    base = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 100)), index=idx)
    prices = {"A": base, "B": base * 2, "C": pd.Series(
        100 * np.cumprod(1 + rng.normal(0, 0.01, 100)), index=idx)}
    c = correlation_matrix(prices)
    assert c.loc["A", "B"] == pytest.approx(1.0, abs=1e-6), "等比例的兩檔應完全相關"
    assert abs(c.loc["A", "C"]) < 0.5


def test_correlation_matrix_skips_short_series():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = {"A": pd.Series(range(1, 11), index=idx, dtype=float)}
    assert correlation_matrix(prices).empty


# --------------------------------------------------------------------------
# 訊號評估
# --------------------------------------------------------------------------
def test_five_identical_signals_are_one_risk():
    """
    同一天 5 個訊號、但全部高度相關——看起來分散，其實是同一個賭注押 5 倍。
    這正是這個模組要抓的情況。
    """
    syms = list("ABCDE")
    c = pd.DataFrame(np.ones((5, 5)), index=syms, columns=syms)
    a = assess_signals(syms, c)
    assert a["n_signals"] == 5
    assert a["effective_n"] == pytest.approx(1.0)
    assert a["concentration_ratio"] == pytest.approx(5.0)


def test_diversified_signals_report_low_ratio():
    syms = list("ABCD")
    c = pd.DataFrame(np.eye(4), index=syms, columns=syms)
    a = assess_signals(syms, c)
    assert a["concentration_ratio"] == pytest.approx(1.0)
    assert a["high_corr_pairs"] == []


def test_high_correlation_pairs_are_listed_worst_first():
    syms = ["A", "B", "C"]
    m = np.array([[1.0, 0.95, 0.10],
                  [0.95, 1.0, 0.70],
                  [0.10, 0.70, 1.0]])
    a = assess_signals(syms, pd.DataFrame(m, index=syms, columns=syms))
    assert [p[:2] for p in a["high_corr_pairs"]] == [("A", "B"), ("B", "C")]


def test_single_signal_needs_no_warning():
    c = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"])
    assert describe(assess_signals(["A"], c)) == []


def test_unknown_symbols_are_ignored_gracefully():
    c = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"])
    a = assess_signals(["A", "ZZZ"], c)
    assert a["n_signals"] == 2 and a["n_measured"] == 1


# --------------------------------------------------------------------------
# 文字輸出
# --------------------------------------------------------------------------
def test_describe_warns_on_high_concentration():
    syms = list("ABCDE")
    c = pd.DataFrame(np.ones((5, 5)), index=syms, columns=syms)
    text = " ".join(describe(assess_signals(syms, c)))
    assert "集中度偏高" in text
    assert "獨立風險" in text


def test_describe_confirms_good_diversification():
    syms = list("ABCD")
    c = pd.DataFrame(np.eye(4), index=syms, columns=syms)
    text = " ".join(describe(assess_signals(syms, c)))
    assert "相關性不高" in text
