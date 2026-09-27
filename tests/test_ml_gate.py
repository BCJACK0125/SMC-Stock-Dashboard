# -*- coding: utf-8 -*-
"""
tests/test_ml_gate.py
======================
ML 模型的樣本外閘門。

背景：實測這組特徵在 15 檔標的、日線 10 年下的樣本外 AUC 平均只有 0.518
（只有 1 檔超過 0.55，而 15 次試驗出現 1 個好看的本來就是雜訊）。
沒有這道閘門的話，ML 那 20 分的評分權重就是在對隨機數字加權。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ml_model


def make_df(n=600, seed=0, drift=0.0004):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    op = close * (1 + rng.normal(0, 0.003, n))
    hi = np.maximum(op, close) * (1 + abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(op, close) * (1 - abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {"Open": op, "High": hi, "Low": lo, "Close": close,
         "Volume": rng.integers(1_000_000, 5_000_000, n)},
        index=pd.date_range("2020-01-01", periods=n, freq="D"))


# --------------------------------------------------------------------------
# 樣本外評估本身
# --------------------------------------------------------------------------
def test_random_features_score_near_half():
    """純雜訊特徵的 AUC 應該在 0.5 附近——這是閘門的校準基準。"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 8))
    y = (rng.normal(size=800) > 0).astype(int)
    auc = ml_model.evaluate_out_of_sample(X, y)
    assert 0.40 < auc < 0.60


def test_leaky_feature_scores_high():
    """特徵直接洩漏答案時 AUC 應該接近 1——確認評估函式本身是有效的。"""
    rng = np.random.default_rng(1)
    y = (rng.normal(size=800) > 0).astype(int)
    X = np.column_stack([y + rng.normal(0, 0.1, 800), rng.normal(size=800)])
    assert ml_model.evaluate_out_of_sample(X, y) > 0.9


def test_single_class_returns_none():
    X = np.random.default_rng(2).normal(size=(300, 5))
    assert ml_model.evaluate_out_of_sample(X, np.ones(300, dtype=int)) is None


# --------------------------------------------------------------------------
# 閘門行為
# --------------------------------------------------------------------------
def test_gate_rejects_a_model_without_predictive_power():
    """
    真實股價資料上，這組特徵的樣本外 AUC 約 0.5。
    預設 min_auc=0.55 應該要把它擋下來，回傳 None。
    """
    df = make_df()
    ind_df = __import__("indicators").compute_indicator_set(df)
    out = ml_model.predict_next_move_probability(df, ind_df, min_auc=0.55)
    assert out is None, "沒有預測力的模型不該貢獻分數"


def test_gate_can_be_relaxed():
    """把門檻降到 0 就一定放行，回傳值要帶著實際 AUC 供人判斷。"""
    df = make_df()
    ind_df = __import__("indicators").compute_indicator_set(df)
    out = ml_model.predict_next_move_probability(df, ind_df, min_auc=0.0)
    assert out is not None
    assert 0.0 <= out["prob_up"] <= 1.0
    assert out["auc"] is not None


def test_validation_can_be_skipped_for_speed():
    """回測內部大量重訓時可以關掉驗證；此時不回報 AUC。"""
    df = make_df()
    ind_df = __import__("indicators").compute_indicator_set(df)
    out = ml_model.predict_next_move_probability(df, ind_df, validate=False)
    assert out is not None and out["auc"] is None


def test_insufficient_rows_still_returns_none():
    df = make_df(n=60)
    ind_df = __import__("indicators").compute_indicator_set(df)
    assert ml_model.predict_next_move_probability(
        df, ind_df, min_train_rows=500, min_auc=0.0) is None


def test_result_shape_is_stable():
    df = make_df()
    ind_df = __import__("indicators").compute_indicator_set(df)
    out = ml_model.predict_next_move_probability(df, ind_df, min_auc=0.0)
    assert set(out) == {"prob_up", "trained_rows", "horizon", "auc"}
