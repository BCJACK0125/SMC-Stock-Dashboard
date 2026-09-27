# -*- coding: utf-8 -*-
"""
tests/test_fit_weights.py
==========================
權重擬合的樣本外紀律。
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest import wilson_lower_bound


# --------------------------------------------------------------------------
# 權重擬合的樣本外紀律
# --------------------------------------------------------------------------
def test_threshold_is_chosen_on_train_not_test():
    """
    門檻必須在訓練段挑、測試段只負責套用。在測試段挑門檻是選擇偏誤，
    會同時灌水所有比較組，讓三組看起來都「顯著優於基準」。
    """
    import numpy as np
    import fit_weights as F

    rng = np.random.default_rng(7)
    n = 800
    cols = ["c_zone", "c_fvg"]
    def seg(bias):
        return pd.DataFrame({
            "c_zone": rng.integers(0, 2, n).astype(float),
            "c_fvg": rng.integers(0, 2, n).astype(float),
            "ret": rng.normal(bias, 0.02, n),
            "bars": np.full(n, 5),
        })
    train, test = seg(0.001), seg(0.001)
    w = {"zone": 50.0, "fvg": 50.0}
    out = F.evaluate_weights(train, test, w, cols, cost=0.0,
                             thresholds=[40, 60, 90], min_trades=20)
    assert out is not None
    assert "train_expectancy" in out, "應記錄訓練段的表現供對照"
    assert out["threshold"] in (40, 60, 90)


def test_evaluate_weights_returns_none_without_enough_train_signals():
    import numpy as np
    import fit_weights as F

    n = 100
    df = pd.DataFrame({
        "c_zone": np.zeros(n),          # 永遠不觸發
        "ret": np.zeros(n),
        "bars": np.full(n, 5),
    })
    out = F.evaluate_weights(df, df, {"zone": 50.0}, ["c_zone"],
                             cost=0.0, thresholds=[40], min_trades=20)
    assert out is None


def test_fitted_weights_are_normalised_to_score_scale():
    import numpy as np
    from fit_weights import weights_to_score_scale

    w = weights_to_score_scale(np.array([2.0, 1.0, -5.0]), ["a", "b", "c"])
    assert sum(w.values()) == pytest.approx(100.0)
    assert w["c"] == 0.0, "負係數應歸零，不該製造反訊號"
    assert w["a"] == pytest.approx(2 * w["b"])


def test_all_negative_coefficients_yield_zero_weights():
    import numpy as np
    from fit_weights import weights_to_score_scale

    w = weights_to_score_scale(np.array([-1.0, -2.0]), ["a", "b"])
    assert all(v == 0.0 for v in w.values())
